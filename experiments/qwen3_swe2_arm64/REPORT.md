# NeMo-RL flagship SWE2 recipe, end to end on one ARM64 GB200 node

One GB200 node (4 GPUs, aarch64) running the full chain:
`NeMo-RL (Megatron) → NeMo-Gym → vLLM → OpenHands → ARM64 SWE sandbox → SWE-bench eval → reward → replay buffer → one training step`. Every number below traces to a run directory under `agent_run/results/`.

## Changes (the real fix is in Gym, not NeMo-RL)

**Blocking bug.** `openhands.sh` hardcodes `jq-linux-amd64`. On aarch64 the download succeeds and setup looks green, but the binary is copied into the SWE container where it dies with `Exec format error`. The entry script's instance lookup then returns nothing and hits `exit 1` — and because the runtime *sources* that script, it kills the caller's shell. OpenHands only sees `command timed out after 600.0 seconds`, three retries, 1800s wasted per rollout. **The symptom is a timeout; the failure happens at second zero.** Fix: dispatch on `uname -m`; re-download when the existing binary cannot execute; drop `jq --version || true`, which is what swallowed the error in the first place.

**Optimization.** `NEMO_GYM_SWE_NO_COPY=1` replaces `cp -al /testbed` with `ln -s`. The container's writable layer is RAM-backed, so the recursive copy costs both time and memory. It is safe because of `--writable-tmpfs`: the SIF is shared read-only and each container gets its own in-memory overlay — measured directly, two concurrent rollouts of the same instance produced different patches. Branches: `aoshen02/Gym` @ `fix/swe-arm64-jq` (`openhands.sh` +17 −4, `swe_agents/app.py` +62) · `aoshen02/RL` @ `review/qwen3-swe2-singlenode` (harness, docs, evidence).

## Setup and data

`Qwen3-30B-A3B-Thinking-2507` (MoE) · recipe `grpo_qwen3_30ba3b_thinking_swe2.yaml` · training Megatron TP2/PP1/CP1/EP1 with full optimizer CPU offload · generation vLLM TP2, non-colocated, async engine, `max_model_len=32768` · the agent is **OpenHands** (Gym's default; `CodeActAgent`, 200-turn cap) and the evaluator is upstream `swebench.harness.run_local_evaluation` — the two are fully separate.

Data: **10 django instances** from `SWE-bench_Verified`, stratified by gold-patch size (1–29 lines) rather than cherry-picked easy ones. One ARM64 SIF per instance, converted straight from `docker://swebench/sweb.eval.arm64.<id>` with no customisation. GRPO: 10 prompts × 2 samples = 20 rollouts, `max_num_steps=1`, sandbox concurrency 10.

## Results

Three runs of the same configuration, all `exit_code=0`, reward mean **0.5000 / 0.4500 / 0.3000** (resolved 10/20, 9/20, 6/20) — sampling spread at n=20. Taking the latest (13206): reward `min 0.0 / max 1.0 / mean 0.5000 / std 0.5130`, resolved **10/20**, 14m53s wall clock.

Pass rate falls **monotonically** with gold-patch size: 1–2 lines 6/8 · 5 lines 2/4 · 11–29 lines 2/8. **Advantage is non-zero** (`±0.7071, std 0.3162`), so this step carries real gradient — a group needs both successes and failures to produce signal; an all-pass or all-fail group has advantage 0 and updates nothing no matter how healthy the chain is.

**The reward is not hacked**: ① the `/root/dataset/data.jsonl` mounted into the agent container and into the eval container are two different files — the agent's is 925B with `patch`, `test_patch`, `FAIL_TO_PASS` and `PASS_TO_PASS` all absent; ② the target test is not in the image, and all 20 rollouts show `touched_tests=0`; ③ `eval.sh` does `git checkout <base_commit> tests/...` before applying `test_patch`, and only the *model's* patch reaches the source tree; ④ a patch with broken indentation was correctly scored `resolved=False`. The monotonic difficulty curve is itself something a leak could not produce.

## Reproduce

```bash
git clone https://github.com/aoshen02/RL  -b review/qwen3-swe2-singlenode
git clone https://github.com/aoshen02/Gym -b fix/swe-arm64-jq <RL>/3rdparty/Gym-workspace/Gym
cd <RL>/experiments/qwen3_swe2_arm64
bash   scripts/build_swe_sif_arm64.sh   django__django-17029          # build the SIF, ~2 min
python scripts/make_swe2_dataset_row.py django__django-17029 data/django17029_swe2.jsonl
bash   scripts/prewarm_nemo_gym_qwen3_swe2.sh                          # prewarm venvs, must be serial

SWE_DATASET=/workspace/agent_run/data/qwen3_swe2_smoke/django_mix10_swe2.jsonl \
NUM_PROMPTS=10 NUM_GENERATIONS=2 SWE_CONCURRENCY=10 \
  bash scripts/launch_qwen3_swe2_tp2_e2e.sh
bash handoff_qwen3_swe2_cold_start.sh --verify [run_dir]   # 0=all required pass, 1=some failed, 2=usage error
```

Prerequisites: Slurm + pyxis/enroot, one node with 4 GPUs (aarch64), image `nemo-rl-vllm-latest.sqsh`. Docker Hub's arm64 images cover only **part** of the benchmark (~141 django instances); missing ones return 401. Optional: `SWE_INSTANCE=` for a single instance · `GYM_DIR=` to point at a different Gym tree without touching the main checkout · `SWE_VERIFY_GOLDEN=1` to calibrate the eval chain with the gold patch (**calibration, not a deliverable**). Concurrency is 10 rather than 20: a sandbox measures 1.1–1.6 GB, but `apptainer_memory_limit_mb=32768` has **no cgroup enforcement** under enroot and is only watchdog-polled, so the worst case is `concurrency × 32 GB`; the bulk of memory sits on the training side, leaving ~185 GB of the node's 893 GB once it is resident.

## Limits

**The only claim this supports is that the chain above completed with `exit_code=0`.** It is not a model-capability evaluation (10 instances × 2 samples); it is not evidence that training works (`max_num_steps=1`, no convergence, no multi-step); evidence items 3/4/5/8 **cover 1 of the 20 rollouts** (the collector takes `head -1`) and only the aggregate items cover all of them; item 6 is permanently NO and exempted (OpenHands' own `output.jsonl` is always 0 bytes on this path, so the trajectory comes from item 7, decoded from `train_data_step*.jsonl`); and **this was only validated on aarch64 — an x86 regression is required before upstreaming**. One transient failure is on record: job 13204 lost two vLLM TP workers to the system (`SYSTEM_ERROR` → `Executor failed.`); the same code on the same node passed on retry, so it is logged as infrastructure flake. Note also that the `batch` partition is `OverSubscribe=EXCLUSIVE` — every job takes a whole node, so N small jobs cost N times the wall clock.

See `ROOT_CAUSE_swe_entry_600s_timeout.md` (the jq chain, including two self-corrections) · `ROOT_CAUSE_vllm_enginedead_keyerror.md` (`mamba_cache_mode` inheritance trap, including one retracted attribution) · `REWARD1_EVIDENCE.md` (with 9 falsified hypotheses) · `AUDIT.md` (independent audit, including the verifier's own false positives and negatives).
