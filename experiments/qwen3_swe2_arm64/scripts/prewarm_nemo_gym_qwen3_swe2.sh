#!/usr/bin/env bash
set -euo pipefail

GYM_ROOT=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
VENV_ROOT=/mnt/lustre01/users/inf-aoshen/nemo_gym_venvs/qwen3_swe2
export PATH="/root/.local/bin:${PATH}"
# Single source of truth for every persistent cache path. UV_PYTHON_INSTALL_DIR
# matters most here: uv downloads a managed CPython for --python 3.13.14, and
# its default location (/root/.local/share/uv/python) lives in the container's
# ephemeral overlay, so the venv interpreter symlinks dangle in every later job.
source /workspace/agent_run/scripts/phase4/qwen3_swe2_caches.sh
mkdir -p "$VENV_ROOT" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

# setup_command.py derives these paths as:
#   $uv_venv_dir/responses_api_models/vllm_model/.venv
#   $uv_venv_dir/responses_api_agents/swe_agents/.venv
# Keep the two installs serial: train and validation share the SWE venv.

setup() {
  local name="$1" dir="$2" install="$3"
  local venv="$VENV_ROOT/$name/.venv"
  echo "[$name] setup $(date -u +%FT%TZ)"
  # A venv left over from a run without UV_PYTHON_INSTALL_DIR points at the
  # container overlay and cannot be repaired in place. Rebuild it.
  if [[ -e "$venv" ]] && ! "$venv/bin/python" -c 'import sys' 2>/dev/null; then
    echo "[$name] stale venv (interpreter does not run); rebuilding"
    rm -rf "$venv"
  fi
  uv venv --seed --allow-existing --python 3.13.14 "$venv"
  (cd "$dir" && bash -lc "source '$venv/bin/activate' && $install")
}

setup responses_api_models/vllm_model "$GYM_ROOT/responses_api_models/vllm_model" \
  "uv pip install -e . ray[default]==2.56.1 openai==2.6.1"
setup responses_api_agents/swe_agents "$GYM_ROOT/responses_api_agents/swe_agents" \
  "uv pip install -r requirements.txt ray[default]==2.56.1 openai==2.6.1"

# test -x only proves a symlink resolves. Execute each interpreter and import
# the package the server needs, which is what actually has to hold at run time.
# base_prefix is /root/.local/... — that interpreter ships inside the image, so
# it is present in every job from this image; only the venvs are on Lustre.
for v in responses_api_models/vllm_model responses_api_agents/swe_agents; do
  py="$VENV_ROOT/$v/.venv/bin/python"
  echo "[$v] base_prefix=$("$py" -c 'import sys; print(sys.base_prefix)')"
  "$py" -c 'import ray, openai' || { echo "FATAL: $v venv broken" >&2; exit 1; }
done
echo "prewarm_finished=$(date -u +%FT%TZ)"
