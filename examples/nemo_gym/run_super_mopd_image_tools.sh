#!/usr/bin/env bash
# Launcher for the Super Omni multi-turn MOPD image-tools recipe
# (10 nodes × 8 GPUs). Mirrors run_polygon_naming_smoke.sh but adds the Super
# refit environment the run depends on. Run this from INSIDE the NeMo-RL
# container, WD = /opt/nemo-rl.
#
# Usage:
#   examples/nemo_gym/run_super_mopd_image_tools.sh
#   examples/nemo_gym/run_super_mopd_image_tools.sh wandb
#   examples/nemo_gym/run_super_mopd_image_tools.sh grpo.max_num_steps=1
#
# Required:
#   MODEL_NAME  HF export of the Super Omni tool-calling checkpoint
#   DATA_PATH   NeMo-Gym JSONL whose rows target image_tools_simple_agent
#
# The teacher defaults to MODEL_NAME (self-distillation). Set TEACHER_MODEL_NAME
# to distill from a stronger checkpoint.

set -euo pipefail

# -- Optional "wandb" token --------------------------------------------------
EXTRA_HYDRA_ARGS=()
if [[ "${1:-}" == "wandb" ]]; then
    EXTRA_HYDRA_ARGS+=("logger.wandb_enabled=true")
    shift
fi

RECIPE="examples/nemo_gym/mopd_nemotron_omni_super_image_tools.yaml"
# The multimodal entry point builds the image processor and asserts
# policy.is_vlm; run_grpo_nemo_gym.py sets up a tokenizer only.
ENTRY="examples/nemo_gym/run_multimodal_grpo_nemo_gym.py"

if [[ ! -f "${RECIPE}" ]]; then
    echo "error: recipe not found at ${RECIPE}" >&2
    exit 1
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
    echo "error: MODEL_NAME unset — point it at the Super Omni HF export" >&2
    exit 1
fi
if [[ -z "${DATA_PATH:-}" ]]; then
    echo "error: DATA_PATH unset — point it at the image-tools MOPD JSONL" >&2
    exit 1
fi
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-${MODEL_NAME}}"

# -- Parallelism -------------------------------------------------------------
# expert_model_parallel_size must equal the training GPU count for the 512
# experts to shard: EP=16, 32, 64 for 2, 4, 8 training nodes.
POLICY_EP="${POLICY_EP:-64}"
POLICY_CP="${POLICY_CP:-2}"
TEACHER_TP="${TEACHER_TP:-8}"
TEACHER_EP="${TEACHER_EP:-8}"

# -- Super refit transport ---------------------------------------------------
# These must agree across the policy and vLLM processes. A mismatched buffer
# ratio produces incompatible packed-refit chunks: the generation side reads
# chunk boundaries that the policy side never wrote.
export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.006}"
export NRL_REFIT_BUFFER_BYTES="${NRL_REFIT_BUFFER_BYTES:-1073741824}"
export NRL_REFIT_NUM_BUFFERS="${NRL_REFIT_NUM_BUFFERS:-1}"

# The Super HF export ships only 16 of 512 MTP experts, and the recipe trains
# with mtp_num_layers=0, so the Megatron model holds no MTP parameters at all.
# The bridge still derives mtp.* mappings from the HF config, which yields
# export tasks that are asymmetric across expert-parallel ranks. Skipping them
# keeps vLLM's original MTP weights and refits only the complete backbone.
# Requires Megatron-Bridge support for this flag.
export NRL_REFIT_SKIP_MTP="${NRL_REFIT_SKIP_MTP:-1}"

# Image-heavy trajectories fragment the allocator across rollout and refit.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "==> recipe:  ${RECIPE}"
echo "==> entry:   ${ENTRY}"
echo "==> policy:  ${MODEL_NAME}"
echo "==> teacher: ${TEACHER_MODEL_NAME}"
echo "==> data:    ${DATA_PATH}"
echo "==> policy EP=${POLICY_EP} CP=${POLICY_CP} | teacher TP=${TEACHER_TP} EP=${TEACHER_EP}"
echo "==> refit:   skip_mtp=${NRL_REFIT_SKIP_MTP} buffer_ratio=${NRL_REFIT_BUFFER_MEMORY_RATIO}"

exec uv run --locked --no-sync \
    "${ENTRY}" \
    --config "${RECIPE}" \
    policy.model_name="${MODEL_NAME}" \
    policy.megatron_cfg.expert_model_parallel_size="${POLICY_EP}" \
    policy.megatron_cfg.context_parallel_size="${POLICY_CP}" \
    data.train.data_path="${DATA_PATH}" \
    data.validation.data_path="${DATA_PATH}" \
    on_policy_distillation.teacher_model_by_agent_name.super_teacher="${TEACHER_MODEL_NAME}" \
    on_policy_distillation.teacher_model_by_agent_name.image_tools_simple_agent="${TEACHER_MODEL_NAME}" \
    on_policy_distillation.non_colocated_teachers.default_teacher_cfg.tensor_model_parallel_size="${TEACHER_TP}" \
    on_policy_distillation.non_colocated_teachers.default_teacher_cfg.expert_model_parallel_size="${TEACHER_EP}" \
    ${EXTRA_HYDRA_ARGS[@]+"${EXTRA_HYDRA_ARGS[@]}"} \
    "$@"
