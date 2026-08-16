#!/usr/bin/env bash
# =============================================================================
# workplace_assistant GRPO training + vllm-router, running on Slurm.
#
# Usage (run on a login node):
#   MODEL_PROFILE=nano9b   bash agent_run/scripts/phase4/grpo_workplace_assistant.sh
#   MODEL_PROFILE=super120b bash agent_run/scripts/phase4/grpo_workplace_assistant.sh   # default
#   MAX_STEPS=5 TIME_LIMIT=02:00:00 MODEL_PROFILE=nano9b bash ...grpo_workplace_assistant.sh
#
# Two ways to run:
#   One-shot (STAGE=submit, default)   bring up cluster -> run once -> whole job exits
#   Persistent (STAGE=cluster + attach) cluster stays up; each iteration just attaches
#                                       and runs training
#     STAGE=cluster bash ...             # bring up a persistent cluster, note the jobid
#     JOB=<jobid> STAGE=attach bash ...  # run once, repeatable
#     scancel <jobid>                    # must be released manually, otherwise it holds
#                                        # the nodes forever
#   Persistent mode saves the ~90s container startup + Ray cluster formation each time;
#   the worktree is a bind mount, so code changes need no cluster restart — they take
#   effect on the next attach.
#
# Call chain:
#   this script (STAGE=submit/cluster/attach, login node)
#     └─ sbatch ray.sub          ships with NeMo-RL; starts container + Ray cluster, unmodified
#          └─ this script (STAGE=run, inside the head-node container)   <- ray.sub's $COMMAND
#               ├─ vllm-router
#               └─ run_grpo_nemo_gym.py --config <YAML>
#
# Same split as the official super_launch.sh:
#   YAML         machine-independent parts — batch shape, parallelism, node count, Gym env
#   this script  machine-dependent parts — model path, data paths, vLLM engine args, cache dirs
# The script computes nothing (no DP, no batch math); it only injects.
# =============================================================================
set -uo pipefail

# --- Fixed paths and ports ---------------------------------------------------
NMU=/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu
RL_ROOT="$NMU/pr_worktrees/gym-router-url"          # NeMo-RL worktree (changes all uncommitted)
SELF="$NMU/agent_run/scripts/phase4/grpo_workplace_assistant.sh"  # must be hardcoded: $0 is unavailable inside the container
GPUS_PER_NODE=4                 # ray.sub asserts this against the partition GRES, gpu:nvidia_gb200:4; not tunable
PORT=30000                      # router's public service port
METRICS_PORT=29000              # router's Prometheus port
SESSION_HEADER=X-Session-ID     # session affinity header; must match the router's --request-id-headers

# --- Pick a profile ----------------------------------------------------------
# The `++` prefix is for keys absent from the parent config — Hydra runs in struct mode,
# and without `++` it errors out with "Key not in struct" (job 12872). The official
# super_launch.sh:62-68 uses the same form.
GYM_DATA=3rdparty/Gym-workspace/Gym/data/workplace_assistant
case "${MODEL_PROFILE:=super120b}" in
  super120b)
    CONFIG=examples/nemo_gym/nemotron-3-super/small_scale/stage1_rlvr_convergence_27node_h100.yaml
    MODEL=/mnt/lustre01/users/inf-aoshen/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    TRAIN_DATA="$GYM_DATA/train.jsonl"
    VAL_DATA="$GYM_DATA/validation.jsonl"
    TIME_LIMIT="${TIME_LIMIT:-04:00:00}"   # first run must convert 231GB of weights + compile the 120B MoE
    ENGINE_ARGS=(
      ++policy.generation.vllm_kwargs.max_num_batched_tokens=8480
      # Don't override max_new_tokens: in the parent config it is
      # ${policy.max_total_sequence_length}. The real constraint on generation is the
      # total length budget; adding another cap would just truncate tool calls for nothing.
      # Disable FlashInfer's kernel autotune (in 12910, "Tuning flashinfer::trtllm_bf16_moe"
      # appeared 334 times). FlashInfer doesn't persist those results, so caching can't help —
      # the only fix is turning it off. vLLM's -O0 preset has it off anyway
      # (config/vllm.py:254). Cost: MoE kernels use the default heuristic config, so
      # generation throughput may suffer — revert this along with enforce_eager before
      # real training.
      ++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false
      # moe_backend=auto picks FlashInfer TRTLLM on GB200, which repacks the BF16 expert
      # weights into a backend-specific blocked layout (the dumped w13_weight is 4D
      # (512,16,768,64), the last dim 64 being the tile). Generic load_weights can't write
      # into that, so refit is guaranteed to hit a shape mismatch. NeMo-RL's own error
      # message at refit_loader.py:73-78 says to set triton.
      ++policy.generation.vllm_kwargs.moe_backend=triton
      # Gym starts a server for every definition in the config that has an entrypoint,
      # regardless of config_paths. The parent config stage1_rlvr inlines three judge
      # models (two 235B, one 4B) plus a reasoning_off variant; none are available locally
      # and no GPUs are allocated to Gym for them, so they must be deleted with `~`,
      # otherwise it hangs at "0 / 7 servers ready" (job 12897). After deletion only
      # 3 servers remain.
      ~env.nemo_gym.safety_judge_model
      ~env.nemo_gym.nl2bash_judge_model
      ~env.nemo_gym.genrm_model
      ~env.nemo_gym.policy_model_reasoning_off
    )
    ;;
  codex)
    # Trajectory replay: no training, just measuring the routing policy. Data is generated
    # by agent_run/scripts/phase4/build_codex_replay_dataset.py, one session per line;
    # context accumulates naturally through codex_replay_agent's multi-turn loop. Turns are
    # not truncated — sessions run their full recorded length (6 to 99 turns), sampled
    # stratified by turn count so every length band is present. Random sampling would draw
    # roughly zero of the 76+ turn sessions, and the long tail is what separates the policies.
    # CODEX_MODEL=9b is the iteration tier: the same replay workload on
    # Nemotron-Nano-9B-v2 (also the nemotron_h hybrid Mamba architecture, so prefix
    # cache behavior is isomorphic), runs on 3 nodes, for quickly validating the routing
    # policy itself. The cost is a 128K context cap (120B has 256K), so the dataset must
    # be clipped separately with the 9B tokenizer.
    if [ "${CODEX_MODEL:-9b}" = 9b ]; then
      CONFIG=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml
      MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2
      DEFAULT_DATASET=p30_9b
      SEQ_LEN=131072
      NODES="${NODES:-3}"
      GEN_NODES="${GEN_NODES:-1}"
      # Generation TP1: 9B bf16 takes only 18G, plenty for a single 186G GPU; 1
      # generation node = 4 engines.
      # Training TP4: the remaining 2 nodes have 8 GPUs; TP4 exactly fills one node,
      # DP2. 9B has no MoE, EP stays 1.
      # code_gen only leaves a num_processes definition in the parent config, but even
      # after config_paths is swapped Gym may still treat it as a server to start (that
      # is exactly how the four judges on the 120B side got stuck at servers ready).
      MODEL_ARGS=(
        policy.generation.vllm_cfg.tensor_parallel_size=1
        ++policy.megatron_cfg.tensor_model_parallel_size=4
        # This parent config colocates generation and training by default; splitting by
        # node requires turning it off (the 120B config inherits false).
        # gpus_per_node must explicitly equal cluster.gpus_per_node; null is caught by
        # the assertion at grpo.py:799.
        policy.generation.colocated.enabled=false
        ++policy.generation.colocated.resources.gpus_per_node="$GPUS_PER_NODE"
        ~env.nemo_gym.code_gen
      )
    else
      CONFIG=examples/nemo_gym/nemotron-3-super/small_scale/stage1_rlvr_convergence_27node_h100.yaml
      MODEL=/mnt/lustre01/users/inf-aoshen/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
      DEFAULT_DATASET=p30
      SEQ_LEN=262144
      NODES="${NODES:-6}"
      GEN_NODES="${GEN_NODES:-4}"
      # TP2: 231G bf16 weights = 116G/GPU, leaving ~50G/GPU KV at gmu 0.9 (Mamba-dominant,
      # KV per token is tiny). 4 gen nodes x 2 engines/node = 8 engines.
      # Training TP8 spans the two training nodes (GB200 NVLink domain), DP1; Megatron
      # asserts ETP1 x EP == TP x DP, so EP must equal the training GPU count.
      MODEL_ARGS=(
        policy.generation.vllm_cfg.tensor_parallel_size=2
        # At TP2 weights are 115G/GPU; FlashInfer TRT-LLM MoE's weight repacking needs
        # another ~60G conversion buffer, so loading OOMs immediately (13175-13177).
        # Only at TP4 do the 58G weights fit. Triton MoE uses the weights in place.
        ++policy.generation.vllm_kwargs.kernel_config.moe_backend=triton
        ++policy.megatron_cfg.tensor_model_parallel_size=$(( (NODES - GEN_NODES) * GPUS_PER_NODE ))
        ++policy.megatron_cfg.expert_model_parallel_size=$(( (NODES - GEN_NODES) * GPUS_PER_NODE ))
        ~env.nemo_gym.safety_judge_model
        ~env.nemo_gym.nl2bash_judge_model
        ~env.nemo_gym.genrm_model
        ~env.nemo_gym.policy_model_reasoning_off
      )
    fi
    # DATASET=min is the minimal debugging tier: 12 sessions x 2 turns, last-turn
    # context p50 28K. Starts fast, finishes reliably, and makes it easy to print every
    # turn's actual request and response.
    CODEX_DATA="$NMU/agent_run/data/codex_replay_${DATASET:-$DEFAULT_DATASET}"
    TRAIN_DATA="$CODEX_DATA/train.jsonl"
    VAL_DATA="$CODEX_DATA/validation.jsonl"
    TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
    ENGINE_ARGS=(
      ++cluster.num_nodes="$NODES"
      ++policy.generation.colocated.resources.num_nodes="$GEN_NODES"
      "${MODEL_ARGS[@]}"
      # mamba_cache_mode=align requires block_size(2080) <= max_num_batched_tokens;
      # the parent config's default 2048 doesn't fit and engine startup fails an assertion.
      ++policy.generation.vllm_kwargs.max_num_batched_tokens=8480
      ++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false
      # Sessions carry 100K+ token contexts, so KV capacity is what caps concurrency. The
      # generation nodes are not colocated with training (colocated.enabled=false), so
      # nothing else competes for their memory and the YAML's 0.5 just wastes half of it.
      policy.generation.vllm_cfg.gpu_memory_utilization="${GMU:-0.9}"
      # Do not default enforce_eager on: skipping CUDA graph capture also skips the
      # startup dummy run, so the Triton kernels (_causal_conv1d_fwd/
      # _selective_scan_update/_zero_kv_blocks/batch_memcpy etc.) compile on first
      # inference instead, blocking the worker's HTTP event loop for 10+ seconds; smg's
      # connect timeout is hardcoded at 10s, so only the first request per engine
      # succeeds while the rest time out and trip the circuit breaker (13146 died
      # exactly like this). Trading 24s of startup time for a failed round is a bad deal.
      policy.generation.vllm_cfg.enforce_eager="${ENFORCE_EAGER:-false}"
      # Full-length replay: the dataset builder already clipped every session to fit this
      # budget with the matching tokenizer, so no request can exceed it (one 400 aborts the
      # whole run at nemo_gym.py:520).
      policy.max_total_sequence_length="$SEQ_LEN"
      # The 8 rollouts of a prompt carry zero information here: the agent discards the model's
      # output and splices the recording back in, so all 8 issue a byte-identical request
      # sequence. They only inflate the prefix cache hit rate (1 miss, 7 free hits) — the very
      # metric being measured — while costing 8x the wall clock.
      grpo.num_generations_per_prompt=1
      # 246 = production median output per call (upstream HF data card). Paired with
      # ignore_eos below it becomes fixed-length decode: per-request work matches the
      # production magnitude and is byte-deterministic across arms and runs.
      policy.generation.max_new_tokens=246
      # Force decode to run to max_new_tokens. Injected via the Gym model server's
      # config-level extra_body (vllm_model/app.py merges it into every chat body);
      # vLLM's chat protocol accepts ignore_eos natively, the router passes it through.
      "++env.nemo_gym.policy_model.responses_api_models.vllm_model.extra_body.ignore_eos=true"
      # Swap out the servers Gym starts: keep only the policy model + replay agent. Tools
      # are not executed, so no resources_server is needed. The four judge models inlined
      # by the parent config are likewise deleted (otherwise it hangs at servers ready).
      "++env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,responses_api_agents/codex_replay_agent/configs/codex_replay_agent.yaml]"
      # The session affinity header follows the router: smg's consistent_hashing only
      # recognizes X-SMG-Routing-Key.
      "++env.nemo_gym.policy_model.responses_api_models.vllm_model.session_affinity_header=$SESSION_HEADER"
      # Sending the whole batch at once gives each engine 32 simultaneous 100K-token
      # requests; its HTTP serving thread stays occupied for a long time, smg's
      # hardcoded 10s connect timeout starts failing, and 10 failures trip the breaker
      # for the whole engine (13145 died like this: only 27 of 128 succeeded). Bound
      # in-flight rollouts by engine count.
      "++env.nemo_gym.max_concurrent_rollouts=${ROLLOUT_CONCURRENCY:-$(( GEN_NODES * GPUS_PER_NODE * 2 ))}"
    )
    ;;
  dapo)
    # Just a different dataset: not one line of YAML changes, we reuse the super120b one
    # (parallelism, 8 nodes x 4 GPUs, and 16384 are all plenty for DAPO — prompts are only
    # about 120 tokens).
    # The Gym-side env resources_servers/math_with_judge is already fully wired: when
    # should_use_judge is false it uses local math_verify, no judge model needed. The only
    # missing piece is the data files — upstream pulls them from an internal GitLab
    # artifact; ours are generated from HF by agent_run/scripts/phase4/build_dapo_gym_dataset.py,
    # with filenames and line counts aligned to data/*_metrics.json (1.79M / 960 lines).
    CONFIG=examples/nemo_gym/nemotron-3-super/small_scale/stage1_rlvr_convergence_27node_h100.yaml
    MODEL=/mnt/lustre01/users/inf-aoshen/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    MWJ=3rdparty/Gym-workspace/Gym/resources_servers/math_with_judge
    # DATASET=gsm8k switches to GSM8K for accuracy checks: same math_with_judge env and
    # simple agent, only the data changes. We go through GRPO rather than run_eval.py
    # because here vLLM starts with dummy weights (2 seconds) and the real weights are
    # poured in via Megatron refit; run_eval.py would make vLLM read 231GB from Lustre
    # itself, and it would test the original HF weights, not the post-refit copy.
    if [ "${DATASET:-dapo}" = gsm8k ]; then
      TRAIN_DATA="$NMU/agent_run/data/gsm8k_gym/train.jsonl"
      VAL_DATA="$NMU/agent_run/data/gsm8k_gym/validation.jsonl"
    else
      TRAIN_DATA="$MWJ/data/dapo17k_bytedtsinghua_train.jsonl"
      VAL_DATA="$MWJ/data/aime24_bytedtsinghua_validation.jsonl"
    fi
    TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
    # 16 nodes: nothing changes on the training side, inference side goes 4->12 nodes.
    # The training node count at grpo.py:807 is derived as
    # cluster.num_nodes - colocated.resources.num_nodes, and 16-12 is still 4 nodes,
    # so TP2/EP16/ETP1/DP8 carry over unchanged; vLLM TP4 stays, engine count 4->12.
    NODES="${NODES:-8}"
    GEN_NODES="${GEN_NODES:-6}"
    ENGINE_ARGS=(
      ++cluster.num_nodes="$NODES"
      ++policy.generation.colocated.resources.num_nodes="$GEN_NODES"
      # See the codex profile: EP must equal the training GPU count, which shrinks with the
      # training node count.
      ++policy.megatron_cfg.expert_model_parallel_size=$(( (NODES - GEN_NODES) * GPUS_PER_NODE ))
      # Not optional: mamba_cache_mode=align requires max_num_batched_tokens >= Mamba's
      # block_size(2080); the default 2048 makes the engine fail an assertion and never start.
      ++policy.generation.vllm_kwargs.max_num_batched_tokens=8480
      ++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false
      ++policy.generation.vllm_kwargs.moe_backend=triton
      "++env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/math_with_judge/configs/dapo17k.yaml]"
      # The parent config stage1_rlvr.yaml:512 hardcodes should_use_judge=true and points
      # the judge at nl2bash_judge_model, taking priority over the false in Gym-side
      # dapo17k.yaml. It must be forced back to false here to use local math_verify;
      # judge_model_server.name is also filled in to avoid a dangling reference after
      # nl2bash_judge_model is deleted.
      ++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.should_use_judge=false
      ++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
      ~env.nemo_gym.safety_judge_model
      ~env.nemo_gym.nl2bash_judge_model
      ~env.nemo_gym.genrm_model
      ~env.nemo_gym.policy_model_reasoning_off
    )
    ;;
  nano9b)
    CONFIG=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml
    MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2
    TRAIN_DATA="$GYM_DATA/train.jsonl"
    VAL_DATA="$NMU/agent_run/data/wa_validation_48.jsonl"   # official 545 rows trimmed to 48
    TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
    ENGINE_ARGS=(
      policy.generation.vllm_cfg.enforce_eager=true   # smoke test: skip CUDA graph capture
      policy.generation.max_new_tokens="${MAX_NEW_TOKENS:-128}"
    )
    ;;
  *) echo "unknown MODEL_PROFILE=$MODEL_PROFILE (super120b|codex|dapo|nano9b)" >&2; exit 1 ;;
esac

# Node count defaults to the YAML's cluster.num_nodes, since sbatch needs it too — avoids
# writing it in two places. A profile that changes scale via ++cluster.num_nodes must also
# set NODES, otherwise the machine count sbatch requests won't match what Hydra sees and
# the placement group will wait forever.
NODES="${NODES:-$(awk '/^cluster:/{c=1} c&&/^  num_nodes:/{print $2; exit}' "$RL_ROOT/$CONFIG")}"

STAGE="${STAGE:-submit}"
# Include the policy name: submitting several arms in parallel can hit the same-second
# timestamp, and the jobs would then write into one directory and clobber each other.
NEW_RUN_DIR() { echo "$NMU/agent_run/results/wa_${MODEL_PROFILE}_${NODES}n_${ROUTER_POLICY:-none}_$(date -u +%Y%m%dT%H%M%SZ)"; }
# The env-var string passed to STAGE=run, shared by the submit and attach paths
RUN_CMD() {
  echo "MODEL_PROFILE=$MODEL_PROFILE STAGE=run RUN_DIR=$1 MAX_STEPS=${MAX_STEPS:-1} \
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-} ROUTER_POLICY=${ROUTER_POLICY:-consistent_hash} \
ROLLOUT_ONLY=${ROLLOUT_ONLY:-0} PROMPTS=${PROMPTS:-} TENSORBOARD=${TENSORBOARD:-false} bash $SELF"
}

# =============================================================================
# Aside from stage one: attach (login node, dialing into an already-running persistent cluster)
# Once the cluster is ready, ray.sub writes <jobid>-attach.sh into SLURM_SUBMIT_DIR (i.e.
# RL_ROOT); inside it is a single srun --overlap --container-name=ray-head that reuses the
# same container instance.
# =============================================================================
if [ "$STAGE" = attach ]; then
  set -e
  ATTACH="$RL_ROOT/${JOB:?attach requires JOB=<persistent cluster jobid>}-attach.sh"
  [ -f "$ATTACH" ] || { echo "cannot find $ATTACH (cluster not ready yet?)" >&2; exit 1; }
  # When a previous attach fails the process doesn't necessarily exit (the driver hangs
  # after a refit error), and the leftover Ray actors keep holding the placement group, so
  # the next attach deadlocks at "Timed out waiting for placement groups to be ready".
  # Clean them up first.
  # `|| true`: with no leftovers pgrep returns 1, which under set -e would silently exit
  # the whole script.
  STALE=$(pgrep -f "srun .*--jobid $JOB .*STAGE=run" | tr '\n' ' ' || true)
  [ -n "$STALE" ] && { echo "cleaning up leftover attach from last run: $STALE"; kill $STALE; sleep 10; }
  RUN_DIR=$(NEW_RUN_DIR)
  mkdir -p "$RUN_DIR"
  echo "attach job=$JOB profile=$MODEL_PROFILE"
  echo "run_dir=$RUN_DIR"
  COMMAND="$(RUN_CMD "$RUN_DIR")" bash "$ATTACH"
  exit
fi

# =============================================================================
# Stage one: submit (login node)
# With STAGE=cluster, COMMAND is left empty: ray.sub brings up the cluster, then sleeps
# forever and writes out the attach script.
# =============================================================================
if [ "$STAGE" = submit ] || [ "$STAGE" = cluster ]; then
  set -e
  if [ "$STAGE" = cluster ]; then
    RUN_DIR="$NMU/agent_run/results/cluster_${MODEL_PROFILE}_${NODES}n_$(date -u +%Y%m%dT%H%M%SZ)"
    COMMAND=""
    TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
  else
    RUN_DIR=$(NEW_RUN_DIR)
    COMMAND="$(RUN_CMD "$RUN_DIR")"
  fi
  mkdir -p "$RUN_DIR"
  echo "stage=$STAGE profile=$MODEL_PROFILE nodes=$NODES model=$MODEL"
  echo "run_dir=$RUN_DIR"

  # About the mounts:
  #   The first two are identity mounts (same path inside and outside the container), so
  #   RUN_DIR / RL_ROOT need no translation.
  #   The third is the critical one: in the Ray actor's venv, nemo_gym is installed as an
  #   editable pointing at the image's /opt/nemo-rl/3rdparty/Gym-workspace/Gym, and
  #   PYTHONPATH can only redirect nemo_rl. Without shadowing that path with the worktree,
  #   every Gym-side change is silently ignored (the job 12859 trap).
  cd "$RL_ROOT"
  # NODELIST pins the job to specific machines: running the same config on nodes that
  # produced NaNs and on nodes that never did is the only way to separate the "machine"
  # variable from the "software" one. It must be defined before the env-var prefix chain
  # below; inserting it in the middle breaks the line continuation and CONTAINER etc.
  # never reach sbatch.
  NODELIST_ARG=()
  [ -n "${NODELIST:-}" ] && NODELIST_ARG=(--nodelist="$NODELIST")
  COMMAND="$COMMAND" \
  CONTAINER=/mnt/lustre01/users/inf-aoshen/enroot/containers/nemo-rl-vllm-latest.sqsh \
  MOUNTS="$NMU:$NMU,/mnt/lustre01:/mnt/lustre01,$RL_ROOT/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym" \
  GPUS_PER_NODE="$GPUS_PER_NODE" BASE_LOG_DIR="$RUN_DIR" \
  sbatch --nodes="$NODES" --account=inferact --partition=batch --time="$TIME_LIMIT" \
         --job-name="${JOB_NAME:-wa_$MODEL_PROFILE}" --gres="gpu:$GPUS_PER_NODE" --exclusive \
         "${NODELIST_ARG[@]}" \
         --output="$RUN_DIR/slurm-%j.log" ray.sub
  exit
fi

# =============================================================================
# Stage two: execute (head node, inside the container, invoked by ray.sub)
# =============================================================================
RUN_DIR="${RUN_DIR:?}"
cd "$RL_ROOT"

# --- Preflight check: confirm Gym comes from the worktree, not the image -----
# session_affinity_header is something we added in the worktree; the image's copy lacks it,
# so we use it as a fingerprint.
# If the mount fails the process still runs fine, only every Gym change is inert, and the
# logs show nothing at all — it has to be caught up front.
grep -q session_affinity_header \
  /opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py ||
  { echo "FATAL: Gym not bind-mounted from worktree" >&2; exit 1; }

# --- Environment variables ---------------------------------------------------
# The cache dirs under PERSISTENT_CACHE come from the official super_launch.sh and are
# reused across jobs:
#   megatron_ckpt_cache  HF->Megatron weight conversion output (~8 min for one 120B conversion)
#   hf_config_locks      file locks used when Megatron-Bridge reads the HF config
#   hf_modules           dynamic modules generated by trust_remote_code
#   vllm_compile_cache   vLLM's torch.compile artifacts
#   gym_venvs            venvs for each Gym server (Super's YAML references it via oc.env)
export HF_HOME=/mnt/lustre01/users/inf-aoshen/huggingface
export PERSISTENT_CACHE=/mnt/lustre01/users/inf-aoshen/nemo_persistent_cache
export PYTHONPATH="$RL_ROOT:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export NRL_MEGATRON_CHECKPOINT_DIR="$PERSISTENT_CACHE/megatron_ckpt_cache"
export MEGATRON_CONFIG_LOCK_DIR="$PERSISTENT_CACHE/hf_config_locks"
export HF_MODULES_CACHE="$PERSISTENT_CACHE/hf_modules"
export VLLM_CACHE_ROOT="$PERSISTENT_CACHE/vllm_compile_cache"
# VLLM_CACHE_ROOT only covers vLLM's own torch.compile artifacts, not Triton's. Nemotron-H's
# Mamba2 SSD kernel goes through Triton autotune (mamba_mixer2.py:596) and by default writes
# to the container's ~/.triton/cache, which disappears when the job ends, forcing a recompile
# every time — in 12910 that alone cost 50 seconds.
# Per-node directories; no cross-node shared writes. Root cause of the NaN incident: the
# alignment specialization of the Mamba _causal_conv1d kernel (two variants with/without
# tt.divisibility=16, equal-length cubins) first compiles during inference; 24 workers
# across nodes wrote the same Lustre directory concurrently, the cache group metadata
# (__grp__*.json + multiple cubin files) landed non-atomically, and engines losing the
# race got a mismatched specialization — the aligned-assumption kernel applied to
# unaligned cache_indices reads wrong block indices → all-NaN logits. The three NaN
# windows match the kernel's three on-disk compile mtimes (13:50 / 01:14 / 02:58)
# one-to-one; after 02:58, zero compiles and zero recurrences over 40+ rounds.
# Do not splice $(hostname) here: this line is evaluated on the head node and forwarded
# verbatim via Ray runtime_env, so all nodes would share one subdirectory. Per-node
# directories are handled in vllm_worker.py's worker init (each worker process reads its
# own hostname), giving cross-node isolation while keeping the warm-start benefit.
export TRITON_CACHE_DIR="${TRITON_CACHE_OVERRIDE:-$PERSISTENT_CACHE/triton_cache}"
# FlashInfer's JIT artifacts land in $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<ver>/<arch>;
# without a persistent dir they go to the container's ~/.cache and vanish when the job ends.
# We don't set FLASHINFER_CUBIN_DIR: that's the download dir for prebuilt cubins and is
# version-checked (flashinfer/jit/env.py:63-94); we have no prebuilt package, so setting it
# would just leave it empty.
# As for the 300+ "Tuning flashinfer::trtllm_bf16_moe" autotunes, FlashInfer doesn't persist
# those results — caching can't help, only disabling autotune can.
export FLASHINFER_WORKSPACE_BASE="$PERSISTENT_CACHE/flashinfer_workspace"
# RAY_DEDUP_LOGS=0: otherwise Ray collapses identical logs from multiple workers into
# "[repeated N x]", and when engine startup hangs you can't tell which rank it is.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 RAY_DEDUP_LOGS=0
# NRL_IGNORE_VERSION_MISMATCH: the worktree code version doesn't exactly match the version
#   baked into the image.
# UV_LOCK_TIMEOUT: multiple Gym servers concurrently building the same editable nemo-gym
#   contend for the same lock under /root/.cache/uv, and the default 300s isn't enough
#   (job 12876 died right there).
export NRL_IGNORE_VERSION_MISMATCH=1 UV_LOCK_TIMEOUT=1200
# Only dump rollout/NaN requests to disk when NRL_DUMP=1 (debug only; keep it off in
# normal tests to avoid writing every rollout to disk)
[ "${NRL_DUMP:-0}" = 1 ] && export NRL_DUMP_TRAJECTORIES="$RUN_DIR/trajectories"
mkdir -p "$PERSISTENT_CACHE"/{megatron_ckpt_cache,hf_config_locks,hf_modules,vllm_compile_cache,triton_cache,gym_venvs}

# --- Warm up the HF dynamic module cache -------------------------------------
# Nemotron needs trust_remote_code, which generates module files on the fly. Many nodes
# writing the same cache dir for the first time simultaneously is a known failure point, so
# we run it once on the head node first to populate it.
python -c "from transformers import AutoConfig, AutoTokenizer
AutoConfig.from_pretrained('$MODEL', trust_remote_code=True)
AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True, use_fast=True)" || exit 1

# --- GPU-clean barrier -------------------------------------------------------
# Slurm reschedules onto a node whose previous job just died within seconds, while a
# crashed process takes tens of seconds to minutes to release GPU memory (five jobs
# today — 13163/13178/13179/13187/13188 — all died this way: the engine or Megatron
# started on GPUs still holding 158G). Before running, query nvidia-smi on every node
# via Ray and only proceed once all are clean.
python - <<'GPUBARRIER' || exit 1
import ray, subprocess, sys, time
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ray.init(address="auto", log_to_driver=False)

@ray.remote(num_cpus=0)
def gpu_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]

deadline = time.time() + 600
while True:
    nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU")]
    refs = {n["NodeManagerHostname"]: gpu_used_mib.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
    ).remote() for n in nodes}
    dirty = {}
    for h, r in refs.items():
        used = ray.get(r)
        if any(x > 2048 for x in used):
            dirty[h] = used
    if not dirty:
        print(f"GPU barrier: all {len(nodes)} nodes clean")
        break
    if time.time() > deadline:
        print(f"GPU barrier: still not released after 600s, giving up: {dirty}")
        sys.exit(1)
    print(f"GPU barrier: {len(dirty)} nodes with unreleased GPU memory {dirty}, rechecking in 15s")
    time.sleep(15)
GPUBARRIER

# Only override when PROMPTS is given explicitly; otherwise always follow the YAML.
PROMPT_ARG=()
[ -n "${PROMPTS:-}" ] && PROMPT_ARG=(grpo.num_prompts_per_step="$PROMPTS")

# --- Start the router --------------------------------------------------------
# With ROUTER_POLICY=none no router is started at all: when router_url is empty,
# nemo_gym.py:233 hands Gym the base_urls of all vLLM replicas and lets Gym pick — that's
# the no-router baseline.
# This installs a locally built wheel, not PyPI's 0.1.15: on the chat completions path
# the latter's extract_text_for_routing only digs out session_params.session_id and
# returns an empty string otherwise (spec.rs:544); cache_aware's radix tree then matches
# on empty strings and effectively degrades to min-load — i.e. that arm was not doing
# cache-aware routing at all. The change is a single function returning the real session
# body, aligned with the sibling completion / responses implementations in the same
# file. The build artifact is an abi3 wheel, installable by any python 3.8+ in the
# container.
# --request-id-headers is not optional: 0.1.15's default list lacks x-session-id;
# without explicitly adding it the session header is ignored, consistent_hash degrades
# to hashing the request body, and session affinity is lost.
ROUTER_POLICY="${ROUTER_POLICY:-consistent_hash}"
ROUTER_ARG=(env.nemo_gym.router_url=null)
if [ "$ROUTER_POLICY" != none ]; then
  python -m pip install --quiet \
    "$NMU/agent_run/artifacts/vllm_router-0.1.15-cp38-abi3-linux_aarch64.whl"
  # ROUTER_DEBUG=1 enables per-request routing decision logs. INFO level only shows
  # startup and add_worker — nothing about which request went to which worker or how
  # long the prefix match was; without that the routing distribution is guesswork.
  ROUTER_LOG=(); [ "${ROUTER_DEBUG:-0}" = 1 ] && ROUTER_LOG=(--log-level debug)
  vllm-router --host 0.0.0.0 --port $PORT --policy "$ROUTER_POLICY" \
    --prometheus-port $METRICS_PORT --request-id-headers x-session-id x-request-id \
    "${ROUTER_LOG[@]}" > "$RUN_DIR/router.log" 2>&1 &
  ROUTER=$!
  ROUTER_ARG=(env.nemo_gym.router_url="http://$(hostname -I | awk '{print $1}'):$PORT")
fi

# Per-worker /metrics time series. A single exit-time snapshot cannot show how queue
# depth, KV usage, and preemption counts changed around a NaN — exactly what has to be
# aligned with the nonfinite dump timestamps.
# Kept outside the router branch: with ROUTER_POLICY=none, watch_router pulls engine
# addresses itself from the Reserved line in nemo_rl.log (the 13191 arm had no timeline
# because the sampler never started).
( sleep 20; bash "$NMU/agent_run/scripts/phase4/watch_router.sh" "$RUN_DIR" 30 \
    > "$RUN_DIR/metrics_timeline.txt" 2>&1 ) &
SAMPLER=$!

# Leave evidence before exiting: worker registry + router-side counters + /metrics of every
# vLLM replica (prefix cache hit rate only exists on the backend, the router can't see it).
# This is the only evidence for the routing path — a training exit code of 0 does not mean
# routing actually took effect.
snapshot() {
  local tag="${1:-exit}"
  [ "$ROUTER_POLICY" = none ] || {
    curl -s "http://127.0.0.1:$PORT/workers" > "$RUN_DIR/list_workers.json"
    curl -s "http://127.0.0.1:$METRICS_PORT/metrics" > "$RUN_DIR/router_metrics.$tag.txt"
  }
  # Engine URLs come from the Reserved line in nemo_rl.log, not the router's /workers —
  # with ROUTER_POLICY=none there is no router, and this arm's engine metrics used to
  # come out empty.
  : > "$RUN_DIR/worker_metrics.$tag.txt"
  for u in $(grep -m1 -oE "Reserved [0-9]+ vLLM server URLs: \[[^]]*\]" "$RUN_DIR/nemo_rl.log" 2>/dev/null |
             grep -oE "http://[0-9.]+:[0-9]+"); do
    echo "### $u" >> "$RUN_DIR/worker_metrics.$tag.txt"
    curl -s --max-time 5 "$u/metrics" >> "$RUN_DIR/worker_metrics.$tag.txt"
  done
}
trap 'snapshot; kill ${ROUTER:-} ${SAMPLER:-} 2>/dev/null' EXIT

# Wait for the router to come up; if it dies first, exit right away instead of making
# training wait for nothing.
if [ "$ROUTER_POLICY" != none ]; then
  until curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    kill -0 $ROUTER 2>/dev/null || { echo "router died on startup" >&2; exit 1; }
    sleep 2
  done
fi

# --- Run training ------------------------------------------------------------
# Only the few things known at runtime or deliberately disabled for this run are overridden
# here: step count, router address (the head IP is only known now), the two logging
# backends, and checkpointing. Everything else follows the YAML.
python -u examples/nemo_gym/run_grpo_nemo_gym.py --config "$CONFIG" \
  policy.model_name="$MODEL" \
  data.train.data_path="$TRAIN_DATA" \
  data.validation.data_path="$VAL_DATA" \
  "${ENGINE_ARGS[@]}" "${PROMPT_ARG[@]}" ${EXTRA_ARGS:-} \
  ++env.nemo_gym.nemo_gym_log_dir="$RUN_DIR/gym_logs" \
  grpo.max_num_steps="${MAX_STEPS:-1}" \
  "${ROUTER_ARG[@]}" \
  logger.wandb_enabled=false logger.tensorboard_enabled="${TENSORBOARD:-false}" \
  checkpointing.enabled=false 2>&1 | tee "$RUN_DIR/nemo_rl.log" &
TRAIN=$!

# ROLLOUT_ONLY=1: we only want the rollout data, so the training part (measured at 114s last
# round, and over ten minutes once the batch grows to 1024) need not run. NeMo-RL has no
# rollout-only switch, so we watch the log for "Computing logprobs" — the first stage after
# generation ends — and once it appears, collect evidence and wrap up.
if [ "${ROLLOUT_ONLY:-0}" = 1 ]; then
  while kill -0 $TRAIN 2>/dev/null; do
    grep -q "Computing logprobs" "$RUN_DIR/nemo_rl.log" 2>/dev/null && {
      echo "rollout done, collecting evidence and stopping"; snapshot rollout; kill -INT $TRAIN 2>/dev/null; sleep 20; break
    }
    sleep 10
  done
fi
wait $TRAIN
RC=$?

exit $RC
