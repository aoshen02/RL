#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
RL_ROOT="${RL_ROOT:-/workspace/nemo-rl}"
MODEL=/mnt/lustre01/users/inf-aoshen/models/Qwen3-30B-A3B-Thinking-2507
# SWE_INSTANCE 是唯一开关:数据集行和 SIF 路径都由它推导,避免两者改得不一致 ——
# 那会让 agent 在 A 的仓库里改 B 的题,而且评测仍然"成功"运行、只是恒为 resolved=false。
# 保持 container_formatter 为**精确路径**而不是 {instance_id} 模板:上游文件名把 `__`
# 写成 `_1776_`,模板匹配要靠 app.py 的 replace 回退链,精确路径是已验证过的那条。
SWE_INSTANCE="${SWE_INSTANCE:-django__django-15037}"
SWE_SHORT="${SWE_INSTANCE##*django-}"        # 17029
SIF_DIR=/mnt/lustre01/users/inf-aoshen/swe-sif-arm64
# SWE_DATASET:多题模式。给一个多行 jsonl,SIF 走 {instance_id} 模板 —— app.py:3135
# 的 replacements=["_1776_","_s_"] 会把 `__` 换成 `_1776_` 后精确命中我们的文件名,
# 所以模板不需要自己拼 `_1776_`。单题模式仍走已验证过的精确路径。
SWE_DATASET="${SWE_DATASET:-}"
if [[ -n "$SWE_DATASET" ]]; then
  DATA="$SWE_DATASET"
  SIF="$SIF_DIR/swebench_sweb.eval.arm64.{instance_id}.sif"
else
  DATA="/workspace/agent_run/data/qwen3_swe2_smoke/django${SWE_SHORT}_swe2.jsonl"
  SIF="$SIF_DIR/swebench_sweb.eval.arm64.${SWE_INSTANCE/__/_1776_}.sif"
fi
CONFIG=examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml

mkdir -p "$RUN_DIR"
cd "$RL_ROOT"

# 跨作业持久化缓存(路径与 grpo_workplace_assistant.sh 逐字一致,复用 wa_* 已暖好的
# 那批)+ stamp():把关键路径的时间戳写进 RUN_DIR/timeline.tsv,交付时按区间分别汇报。
# attach 路径会先 source 同一个文件,这里再 source 一次是幂等的。
source /workspace/agent_run/scripts/phase4/qwen3_swe2_caches.sh
stamp container_ready
export PYTHONPATH="$RL_ROOT:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_FORCE_REBUILD_VENVS=false
export NRL_JOB_START_EPOCH="$(date +%s.%N)"
export RAY_DEDUP_LOGS=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# The container image already exports NEMO_GYM_VENV_DIR=/opt/gym_venvs and that
# value wins over sbatch --export, so a `${NEMO_GYM_VENV_DIR:-...}` default is
# silently ignored. Take the path from a name the image does not define, then
# overwrite NEMO_GYM_VENV_DIR unconditionally.
export NEMO_GYM_VENV_DIR="${QWEN3_SWE2_GYM_VENV_DIR:-/mnt/lustre01/users/inf-aoshen/nemo_gym_venvs/qwen3_swe2}"
# The ARM64 smoke SIF is a read-only squashfs image. Avoid its slow recursive
# /testbed copy; the injected SWE entry script uses a same-container symlink.
export NEMO_GYM_SWE_NO_COPY=1
export PATH="/root/.local/bin:${PATH}"

# The prewarm job creates both venvs serially. Fail early instead of allowing
# three Gym servers to race through the same editable install. `test -x` is not
# enough: it follows a symlink into the prewarm container's overlay and passes
# against an interpreter that no longer exists here. Execute it instead.
echo "preflight: verifying Gym venvs at $NEMO_GYM_VENV_DIR"
for v in responses_api_models/vllm_model responses_api_agents/swe_agents; do
  py="$NEMO_GYM_VENV_DIR/$v/.venv/bin/python"
  if ! "$py" -c 'import ray, openai, sys; print(sys.base_prefix)'; then
    echo "FATAL: Gym venv interpreter unusable: $py" >&2
    echo "       -> rerun prewarm_nemo_gym_qwen3_swe2.sh" >&2
    exit 1
  fi
done
stamp gym_venv_verified

{
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "node=$(hostname)"
  echo "rl_commit=${RL_COMMIT:-unknown}"
  echo "gym_commit=${GYM_COMMIT:-unknown}"
  echo "config=$CONFIG"
  echo "model=$MODEL"
  echo "data=$DATA"
  echo "sif=$SIF"
  echo "cluster=1x4"
  echo "train=TP2_PP1_CP1_EP1"
  echo "generation=vLLM_TP2_non_colocated_async"
  # 证据收集要据此定位 sandbox 产物。GYM_DIR 可以指向别的 Gym 树,写死路径
  # 会让收集器在空目录里找证据,把健康的运行判成失败。
  echo "gym_results=${GYM_RESULTS_HOST:-/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents}"
} | tee "$RUN_DIR/run.env"

test -f "$CONFIG"
test -d "$MODEL"
test -s "$DATA"
command -v apptainer
command -v uv
uv --version | tee "$RUN_DIR/uv-version.txt"
if [[ -z "$SWE_DATASET" ]]; then
  test -s "$SIF"
  test "$(wc -l < "$DATA")" -eq 1
  apptainer inspect "$SIF" > "$RUN_DIR/sif-inspect.json"
fi

python - <<'PY' "$DATA" "$SIF" "$SWE_INSTANCE" "${SWE_DATASET:-}" | tee "$RUN_DIR/preflight.txt"
import json, pathlib, sys

data, sif, want, multi = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], bool(sys.argv[4])
rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
if not multi:
    assert len(rows) == 1, f"single-instance mode but {len(rows)} rows"

for row in rows:
    meta = row["responses_create_params"]["metadata"]
    inst = json.loads(meta["instance_dict"])
    iid = meta["instance_id"]
    # 三处 instance 身份必须一致:数据集顶层、instance_dict、SIF 文件名。任何一处错位
    # 都不会报错,只会让 agent 在错误的仓库里改题、评测恒 resolved=false。
    assert inst["instance_id"] == iid, f"instance_dict: {inst['instance_id']} != {iid}"
    if not multi:
        assert iid == want, f"dataset top: {iid} != {want}"
    # 多题模式下 SIF 是 {instance_id} 模板;逐个把它解析成真实路径并要求存在,
    # 否则会在 rollout 半途才炸,而且只炸那一条、整跑看起来仍然"成功"。
    p = pathlib.Path(sif.replace("{instance_id}", iid.replace("__", "_1776_")))
    assert p.is_file(), f"missing sif for {iid}: {p}"
    # instance_dict 缺 test_patch 会让评测阶段拿不到 FAIL_TO_PASS 的测试代码。
    for k in ("patch", "test_patch", "base_commit", "environment_setup_commit"):
        assert inst.get(k), f"{iid}: instance_dict missing {k}"
    print("instance=%-22s f2p=%-2d p2p=%-4d gold_lines=%-3d sif_mb=%d" % (
        iid, len(json.loads(meta["FAIL_TO_PASS"])), len(json.loads(meta["PASS_TO_PASS"])),
        sum(1 for x in inst["patch"].splitlines() if x[:1] in "+-" and x[1:2] not in "+-"),
        p.stat().st_size // (1 << 20)))

print("n_instances=" + str(len(rows)))
print("dataset_name=" + rows[0]["responses_create_params"]["metadata"]["dataset_name"])
PY

# NUM_GENERATIONS:组内采样条数。recipe 默认 8;smoke 用 1 只为最短冷启动。
# 单样本组的 advantage 恒为 0(std=nan),既拿不到有效梯度,也把命中 resolved=true
# 的机会压到一次。SWE agent 的失败大量是随机的机械失误(str_replace 缩进/行号),
# 多采几条正是 GRPO 该有的样子。
NUM_PROMPTS="${NUM_PROMPTS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-1}"
# SWE_CONCURRENCY:同时开几个 sandbox。SIF 只读、NO_COPY 在各自 writable-tmpfs 里
# symlink /testbed,所以并发是安全的;串行 8 条会把墙钟拉到 8 倍。
SWE_CONCURRENCY="${SWE_CONCURRENCY:-1}"
# SWE_VERIFY_GOLDEN=1:上游内建金 patch 自检(app.py:2844)。跳过 agent,把
# instance_dict['patch'] 当模型 patch 送进评测容器。**不是交付物,是校准** —— 证明
# 这道题在本环境确实判得出 resolved=true;金 patch 都判不出,说明问题在评测侧。
SWE_VERIFY_GOLDEN="${SWE_VERIFY_GOLDEN:-false}"
[[ "$SWE_VERIFY_GOLDEN" == "1" ]] && SWE_VERIFY_GOLDEN=true
GOLDEN_OVERRIDES=()
if [[ "$SWE_VERIFY_GOLDEN" == "true" ]]; then
  echo "MODE: golden-patch calibration (agent skipped; NOT an end-to-end deliverable)"
  GOLDEN_OVERRIDES=(
    "++env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.verify_golden_patch=true"
    "++env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.verify_golden_patch=true"
  )
fi
echo "MODE: prompts=$NUM_PROMPTS data=$DATA num_generations=$NUM_GENERATIONS concurrency=$SWE_CONCURRENCY golden=$SWE_VERIFY_GOLDEN"

# Keep the async data iterator alive while the single rollout is in flight;
# max_num_steps=1 remains the actual training stop condition.
stamp nemo_rl_launch
python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "$CONFIG" \
  "${GOLDEN_OVERRIDES[@]}" \
  policy.model_name="$MODEL" \
  data.train.data_path="$DATA" \
  data.validation.data_path="$DATA" \
  data.shuffle=false \
  data.num_workers=0 \
  grpo.num_prompts_per_step="$NUM_PROMPTS" \
  grpo.num_generations_per_prompt="$NUM_GENERATIONS" \
  grpo.max_num_steps=1 \
  grpo.max_num_epochs=100000 \
  grpo.val_at_start=false \
  grpo.val_at_end=false \
  grpo.val_period=1000 \
  grpo.use_leave_one_out_baseline=false \
  policy.train_global_batch_size="$((NUM_PROMPTS * NUM_GENERATIONS))" \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=1 \
  policy.generation_batch_size=1 \
  policy.max_total_sequence_length=32768 \
  policy.make_sequence_length_divisible_by=2 \
  policy.offload_optimizer_for_logprob=true \
  policy.megatron_cfg.tensor_model_parallel_size=2 \
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  policy.megatron_cfg.expert_model_parallel_size=1 \
  policy.megatron_cfg.pipeline_model_parallel_size=1 \
  policy.megatron_cfg.context_parallel_size=1 \
  policy.megatron_cfg.sequence_parallel=true \
  policy.megatron_cfg.optimizer.optimizer_cpu_offload=true \
  policy.megatron_cfg.optimizer.optimizer_offload_fraction=1.0 \
  policy.generation.max_new_tokens=32768 \
  policy.generation.vllm_cfg.async_engine=true \
  ++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false \
  ++policy.generation.vllm_kwargs.mamba_cache_mode=none \
  ++policy.generation.vllm_kwargs.mamba_ssm_cache_dtype=auto \
  policy.generation.vllm_cfg.tensor_parallel_size=2 \
  policy.generation.vllm_cfg.max_model_len=32768 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.8 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.gpus_per_node=2 \
  policy.generation.colocated.resources.num_nodes=1 \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=200 \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency="$SWE_CONCURRENCY" \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.dataset_path="$DATA" \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter="'$SIF'" \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.concurrency="$SWE_CONCURRENCY" \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.dataset_path="$DATA" \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.container_formatter="'$SIF'" \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=4 \
  logger.log_dir="$RUN_DIR/nemo-rl-logs" \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=false \
  checkpointing.enabled=false \
  2>&1 | tee "$RUN_DIR/nemo_rl.log"
RC="${PIPESTATUS[0]}"
stamp nemo_rl_exit

{
  echo "exit_code=$RC"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "$RUN_DIR/run.env"
exit "$RC"
