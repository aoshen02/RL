#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu/pr_worktrees/multi-server-group
NEMU=/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu
IMAGE=/mnt/lustre01/users/inf-aoshen/enroot/containers/nemo-rl-v0.7.0.sqsh
RESULTS=/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu/agent_run/results/nemo_rl_multi_server_group_e2e
RUN_DIR="$RESULTS/$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$RUN_DIR"

read -r -d '' COMMAND <<EOF || true
set -euo pipefail
cd $ROOT
export PYTHONPATH=$ROOT:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
export HF_HOME=/mnt/lustre01/users/inf-aoshen/.cache/hf
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Reuse the Ray worker environments baked into the .sqsh; do not build on host.
export PATH=/root/.local/bin:/opt/nemo_rl_venv/bin:$PATH
export NEMO_RL_VENV_DIR=/opt/ray_venvs

/opt/nemo_rl_venv/bin/python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_multi_server_group_e2e.yaml \
  cluster.gpus_per_node=4 \
  cluster.num_nodes=3 \
  policy.generation.colocated.resources.gpus_per_node=4 \
  policy.generation.colocated.resources.num_nodes=2 \
  'policy.generation.server_groups=[{name:low_latency,gpus:4,overrides:{vllm_cfg:{tensor_parallel_size:1},vllm_kwargs:{max_num_seqs:16}}},{name:high_throughput,gpus:4,overrides:{vllm_cfg:{tensor_parallel_size:4},vllm_kwargs:{max_num_seqs:512}}}]' \
  logger.log_dir=$RUN_DIR/nemo-rl \
  2>&1 | tee $RUN_DIR/driver.log &
driver_pid=\$!

for _ in \$(seq 1 900); do
  if grep -q 'Reserved 5 vLLM server URLs' $RUN_DIR/driver.log; then
    break
  fi
  kill -0 \$driver_pid 2>/dev/null || break
  sleep 2
done

/opt/nemo_rl_venv/bin/python - $RUN_DIR/driver.log $RUN_DIR/models-health.txt <<'PY'
import ast
import re
import subprocess
import sys

text = open(sys.argv[1]).read()
match = re.search(r"Reserved 5 vLLM server URLs: (\[.*?\])", text)
if not match:
    raise SystemExit("reserved URL list not found")
urls = ast.literal_eval(match.group(1))
with open(sys.argv[2], "w") as out:
    for url in urls:
        endpoint = f"{url.removesuffix('/v1')}/tokenize"
        for _ in range(180):
            result = subprocess.run(
                [
                    "curl",
                    "-fsS",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    '{"model":"Qwen/Qwen3-0.6B","prompt":"health"}',
                    endpoint,
                ],
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                break
            import time

            time.sleep(2)
        out.write(f"{url} status={result.returncode}\n{result.stdout}\n")
        if result.returncode:
            raise SystemExit(result.stderr)
PY

wait \$driver_pid

grep -F "Ray inference cluster for server group 'low_latency' initialized with 4 GPUs" $RUN_DIR/driver.log
grep -F "Ray inference cluster for server group 'high_throughput' initialized with 4 GPUs" $RUN_DIR/driver.log
grep -F 'Reserved 5 vLLM server URLs' $RUN_DIR/driver.log
test "\$(grep -c 'status=0' $RUN_DIR/models-health.txt)" -eq 5
grep -F 'Step 2/2' $RUN_DIR/driver.log
grep -F 'Max number of steps has been reached' $RUN_DIR/driver.log
printf 'PASS: heterogeneous TP1/TP4 groups, 5 HTTP endpoints, and the post-step-1 refit completed before step 2.\n' | tee $RUN_DIR/validation.txt
EOF

cd "$ROOT"
COMMAND="$COMMAND" \
CONTAINER="$IMAGE" \
MOUNTS="$NEMU:$NEMU,/mnt/lustre01:/mnt/lustre01" \
GPUS_PER_NODE=4 \
BASE_LOG_DIR="$RUN_DIR" \
sbatch \
  --account=inferact \
  --partition=batch \
  --nodes=3 \
  --gres=gpu:4 \
  --time=02:00:00 \
  --job-name=nemo-multi-server-group \
  --output="$RUN_DIR/slurm-%j.log" \
  ray.sub | tee "$RUN_DIR/submission.txt"

echo "$RUN_DIR"
