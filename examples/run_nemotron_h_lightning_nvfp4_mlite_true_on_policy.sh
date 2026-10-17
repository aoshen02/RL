#!/usr/bin/env bash
# Nemotron 3.5 Lightning NVFP4 math GRPO, true on-policy: batch-invariant vLLM
# rollout plus a Megatron Lite policy on the same vLLM kernels, so every step's
# old logprobs equal the rollout logprobs bitwise. 1 node x 4 GB200; DAPO-Math-17K
# training, AIME-2024 greedy validation before and after.
# Config: examples/configs/recipes/llm/grpo-nemotron-h-lightning-nvfp4-1n4g-mlite-true-on-policy.yaml
# Image: docker/Dockerfile.nemotron_h_mlite_true_on_policy (nothing is mounted
# besides the weights, the data and the output directory).
#
# Required env:
#   MODEL_PATH        NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
#   BF16_MASTER_PATH  NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
#   TRAIN_FILES       verl DAPO-Math-17K parquet (dapo-math-17k.parquet)
#   VAL_FILES         verl AIME-2024 parquet (aime-2024.parquet)
# Optional env: OUTPUT_DIR (default ./nemotron-h-mlite-<date>), extra arguments
# are passed to run_grpo.py as overrides (e.g. grpo.max_num_steps=10).
#
# The summary at the end checks the old-vs-rollout logprobs of every step and
# prints PASS or FAIL. Greedy validation is deterministic: AIME-2024 before
# training is 18/30. Sampled rollouts (T=1) carry no per-request seed, so
# rewards and lengths vary slightly between runs.
set -euo pipefail

: "${MODEL_PATH:?}" "${BF16_MASTER_PATH:?}" "${TRAIN_FILES:?}" "${VAL_FILES:?}"
NRL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$NRL_ROOT/examples/configs/recipes/llm/grpo-nemotron-h-lightning-nvfp4-1n4g-mlite-true-on-policy.yaml
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/nemotron-h-mlite-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUTPUT_DIR/data"

# Batch-invariant kernels and fixed NCCL reduction order, shared by the rollout
# and the trainer.
export VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 PYTHONHASHSEED=42 \
  NCCL_LAUNCH_MODE=GROUP NCCL_COLLNET_ENABLE=0 NCCL_NVLS_ENABLE=0 \
  NCCL_P2P_NET_DISABLE=1 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 \
  NCCL_PROTO=Simple NCCL_ALGO=allreduce:tree NCCL_NTHREADS=1 NCCL_SOCKET_NTHREADS=1
# Every actor runs in the image's environment.
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# verl parquet -> {input, output} jsonl, without duplicate (prompt, answer) rows.
python - "$TRAIN_FILES" "$OUTPUT_DIR/data/train.jsonl" "$VAL_FILES" "$OUTPUT_DIR/data/val.jsonl" <<'PY'
import json
import sys

import pandas as pd

args = sys.argv[1:]
for src, dst in zip(args[::2], args[1::2]):
    df = pd.read_parquet(src, columns=["prompt", "reward_model"])
    seen = set()
    with open(dst, "w") as f:
        for prompt, reward_model in zip(df["prompt"], df["reward_model"]):
            assert len(prompt) == 1 and prompt[0]["role"] == "user"
            key = (prompt[0]["content"], reward_model["ground_truth"])
            if key not in seen:
                seen.add(key)
                f.write(json.dumps({"input": key[0], "output": key[1]}) + "\n")
    print(f"{src}: {len(df)} rows, {len(seen)} unique")
PY

python "$NRL_ROOT/examples/run_grpo.py" --config "$CONFIG" \
  policy.model_name="$MODEL_PATH" \
  policy.external_cfg.impl_cfg.bf16_master_path="$BF16_MASTER_PATH" \
  data.train.data_path="$OUTPUT_DIR/data/train.jsonl" \
  data.validation.data_path="$OUTPUT_DIR/data/val.jsonl" \
  logger.log_dir="$OUTPUT_DIR/logs" \
  "$@" 2>&1 | tee "$OUTPUT_DIR/run.log"

python - "$OUTPUT_DIR" <<'PY' | tee "$OUTPUT_DIR/summary.txt"
import glob
import json
import re
import sys

out = sys.argv[1]
log = open(f"{out}/run.log", errors="replace").read()
accuracy = re.findall(r"• Accuracy: ([0-9.]+)", log)
rewards = re.findall(r"• Avg Reward: (-?[0-9.]+)", log)
lengths = re.findall(r"• Mean Generation Length: ([0-9.]+)", log)
print(f"AIME-2024 greedy accuracy (before, after): {accuracy}")
steps = sorted(
    glob.glob(f"{out}/logs/exp_*/train_data_step*.jsonl"),
    key=lambda p: int(re.search(r"step(\d+)", p).group(1)),
)
ok = bool(accuracy) and bool(steps)
for i, path in enumerate(steps):
    tokens = equal = 0
    for line in open(path):
        row = json.loads(line)
        for prev, gen, mask in zip(
            row["prev_logprobs"][0], row["generation_logprobs"][0], row["token_loss_mask"][0]
        ):
            tokens += bool(mask)
            equal += bool(mask) and prev == gen
    ok &= equal == tokens
    print(
        f"step {i + 1}: reward {rewards[i] if i < len(rewards) else '?'}, "
        f"mean response {lengths[i] if i < len(lengths) else '?'}, "
        f"old vs rollout logprobs bitwise equal {equal}/{tokens}"
    )
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
