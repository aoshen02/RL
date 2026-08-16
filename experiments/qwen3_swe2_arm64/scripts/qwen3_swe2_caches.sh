#!/usr/bin/env bash
# 跨作业复用的持久化缓存，来源与 grpo_workplace_assistant.sh 一致（官方
# super_launch.sh 的那套）。sourceable：只设环境变量，不做别的事。
#
# 这些目录不设的话会落到容器内的 ~/.cache，作业一结束就没，每次冷启动都要重算。
# 注意 UV_CACHE_DIR 由 Gym 的 global_config.py 无条件覆写成配置里的值，所以这里
# 设的 UV_CACHE_DIR 只对 Gym 启动之前的阶段有效；venv 已预热时它本来也不参与。
export PERSISTENT_CACHE=/mnt/lustre01/users/inf-aoshen/nemo_persistent_cache
export HF_HOME=/mnt/lustre01/users/inf-aoshen/huggingface
export NRL_MEGATRON_CHECKPOINT_DIR="$PERSISTENT_CACHE/megatron_ckpt_cache"
export MEGATRON_CONFIG_LOCK_DIR="$PERSISTENT_CACHE/hf_config_locks"
export HF_MODULES_CACHE="$PERSISTENT_CACHE/hf_modules"
# vLLM 的 torch.compile 产物。Qwen3-30B-A3B 每次冷启动重编译是几十秒到几分钟。
export VLLM_CACHE_ROOT="$PERSISTENT_CACHE/vllm_compile_cache"
# FlashInfer 的 MoE kernel autotune 由它自己触发，和 enforce_eager 是两套机制。
export FLASHINFER_CUBIN_DIR="$PERSISTENT_CACHE/flashinfer_cubins"
export FLASHINFER_WORKSPACE_BASE="$PERSISTENT_CACHE/flashinfer_workspace"
# 目录名与 grpo_workplace_assistant.sh 逐字一致（triton_cache，不是 triton），
# 否则复用不到 wa_* 那批作业已经暖好的缓存 —— 那才是配这些路径的全部意义。
export TRITON_CACHE_DIR="$PERSISTENT_CACHE/triton_cache"
# 参考脚本里没有这一条，是额外补的：不设的话 inductor 产物落进容器内 ~/.cache。
export TORCHINDUCTOR_CACHE_DIR="$PERSISTENT_CACHE/inductor_cache"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 RAY_DEDUP_LOGS=0
export NRL_IGNORE_VERSION_MISMATCH=1
# 多个 Gym server 并发 build 同一个 editable nemo-gym 时会抢同一把 uv 锁，
# 默认 300s 不够（作业 12876 死在这）。venv 预热后走不到这里，留作兜底。
export UV_LOCK_TIMEOUT=1200
export UV_CACHE_DIR=/mnt/lustre01/users/inf-aoshen/nemo_gym_uv_cache
export UV_PYTHON_INSTALL_DIR=/mnt/lustre01/users/inf-aoshen/uv_python

mkdir -p "$PERSISTENT_CACHE"/{megatron_ckpt_cache,hf_config_locks,hf_modules,vllm_compile_cache,triton_cache,inductor_cache,flashinfer_cubins,flashinfer_workspace,gym_venvs} 2>/dev/null || true

# 冷启动可观测性：把每个阶段的时间戳追加到 RUN_DIR/timeline.tsv，交付时按区间分别汇报，
# 而不是给一个含糊的"启动耗时"。stamp 只写文件，不改任何 fd。
export QWEN3_SWE2_TIMELINE="${RUN_DIR:-/tmp}/timeline.tsv"
stamp() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" "$(date +%s.%N)" "$1" \
    2>/dev/null >>"$QWEN3_SWE2_TIMELINE"
  echo "[stamp] $(date -u +%FT%TZ) $1"
}
export -f stamp
