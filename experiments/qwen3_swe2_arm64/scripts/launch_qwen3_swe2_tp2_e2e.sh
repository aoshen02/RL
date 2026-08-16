#!/usr/bin/env bash
set -euo pipefail

NMU=/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu
RL_ROOT="$NMU/pr_worktrees/gym-router-url"
AGENT_RUN="$NMU/agent_run"
IMAGE=/mnt/lustre01/users/inf-aoshen/enroot/containers/nemo-rl-vllm-latest.sqsh
GYM_VENV_DIR=/mnt/lustre01/users/inf-aoshen/nemo_gym_venvs/qwen3_swe2

SWE_INSTANCE="${SWE_INSTANCE:-django__django-15037}"
SIF_DIR=/mnt/lustre01/users/inf-aoshen/swe-sif-arm64
SIF="$SIF_DIR/swebench_sweb.eval.arm64.${SWE_INSTANCE/__/_1776_}.sif"
SWE_SHORT="${SWE_INSTANCE##*django-}"
# 在宿主上先挡一次:SIF 造好前提交只会浪费一次 GPU 排队 —— 单节点集群上那可能是几十分钟。
if [[ -n "${SWE_DATASET:-}" ]]; then
  # 多题模式:SWE_DATASET 是容器内路径,换算回宿主路径后逐行校验每个 instance 的 SIF。
  DATA_HOST="${SWE_DATASET/\/workspace\/agent_run/$AGENT_RUN}"
  [[ -s "$DATA_HOST" ]] || { echo "FATAL: missing dataset: $DATA_HOST" >&2; exit 1; }
  missing=()
  while read -r iid; do
    f="$SIF_DIR/swebench_sweb.eval.arm64.${iid/__/_1776_}.sif"
    [[ -s "$f" ]] || missing+=("$iid")
  done < <(python3 -c "
import json,sys
for l in open('$DATA_HOST'):
    if l.strip(): print(json.loads(l)['responses_create_params']['metadata']['instance_id'])")
  if ((${#missing[@]})); then
    echo "FATAL: ${#missing[@]} SIF(s) not built yet:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "  -> bash agent_run/scripts/phase4/build_swe_sif_arm64.sh <instance_id>" >&2
    exit 1
  fi
  echo "multi-instance mode: $(wc -l < "$DATA_HOST") instances, all SIFs present"
else
  DATA_HOST="$AGENT_RUN/data/qwen3_swe2_smoke/django${SWE_SHORT}_swe2.jsonl"
  [[ -s "$SIF" ]]      || { echo "FATAL: missing SIF for $SWE_INSTANCE: $SIF" >&2
                            echo "  -> bash agent_run/scripts/phase4/build_swe_sif_arm64.sh $SWE_INSTANCE" >&2; exit 1; }
  [[ -s "$DATA_HOST" ]] || { echo "FATAL: missing dataset row: $DATA_HOST" >&2
                            echo "  -> make_swe2_dataset_row.py $SWE_INSTANCE $DATA_HOST" >&2; exit 1; }
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="qwen3_swe2_tp2_e2e_${TS}"
HOST_RUN_DIR="$AGENT_RUN/results/$RUN_NAME"
mkdir -p "$HOST_RUN_DIR"
RL_COMMIT="$(git -C "$RL_ROOT" rev-parse HEAD)"
GYM_COMMIT="$(git -C "$RL_ROOT/3rdparty/Gym-workspace/Gym" rev-parse HEAD)"

sbatch \
  --job-name="$RUN_NAME" \
  --account=inferact \
  --partition=batch \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=144 \
  --gres=gpu:4 \
  --exclusive \
  --time=06:00:00 \
  --output="$HOST_RUN_DIR/slurm-%j.log" \
  --container-image="$IMAGE" \
  --container-mounts="$RL_ROOT:/workspace/nemo-rl,$RL_ROOT/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym,$AGENT_RUN:/workspace/agent_run,/mnt/lustre01:/mnt/lustre01" \
  --container-workdir=/workspace/nemo-rl \
  --container-remap-root \
  --export=ALL,RUN_DIR="/workspace/agent_run/results/$RUN_NAME",RL_ROOT=/workspace/nemo-rl,RL_COMMIT="$RL_COMMIT",GYM_COMMIT="$GYM_COMMIT",QWEN3_SWE2_GYM_VENV_DIR="$GYM_VENV_DIR",SWE_INSTANCE="$SWE_INSTANCE",NUM_PROMPTS="${NUM_PROMPTS:-1}",NUM_GENERATIONS="${NUM_GENERATIONS:-1}",SWE_DATASET="${SWE_DATASET:-}",SWE_CONCURRENCY="${SWE_CONCURRENCY:-1}",SWE_VERIFY_GOLDEN="${SWE_VERIFY_GOLDEN:-false}" \
  --wrap="bash /workspace/agent_run/scripts/phase4/run_qwen3_swe2_tp2_e2e.sh"

echo "run_dir=$HOST_RUN_DIR"
