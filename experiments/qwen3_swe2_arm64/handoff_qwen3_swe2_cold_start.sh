#!/usr/bin/env bash
# Handoff for the next agent. This file is intentionally executable so it can
# print the complete context without depending on the previous conversation.
set -euo pipefail

# --verify: 让本文件成为校验入口,而不只是上下文文档。
# 打印上下文(无参)和 --check 的三个 test -f 都不能证明端到端跑通 —— 前者 rc=0
# 只说明文本打得出来,后者只说明三个文件存在。--verify 执行 REQUIRED VERIFICATION
# 里那 12 项证据的实际校验,并用退出码表达结论:
#   0 = 全部必需项通过   1 = 有必需项未通过   2 = 用法/环境错误
# 可选参数:--verify [run_dir] [since];省略时自动取最新的 e2e run 目录。
if [[ "${1:-}" == "--verify" ]]; then
  root="/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu"
  exec bash "$root/agent_run/scripts/phase4/verify_qwen3_swe2.sh" "${@:2}"
fi

cat <<'HANDOFF'
OBJECTIVE
=========
Run the NeMo-RL flagship Qwen3 Thinking SWE2 recipe end to end on one GB200
node: NeMo-RL -> NeMo-Gym -> vLLM -> OpenHands -> one ARM64 SWE sandbox ->
SWE reward/eval -> replay buffer -> one training step. Use one sample:
django__django-15037. Keep reproducible evidence and minimize cold start.

SESSION / TRACEABILITY
======================
Codex thread: 019f99d4-c5c4-7fc2-87fc-ce0bc95b7983
Conversation JSONL:
/home/inf-aoshen/.codex/sessions/2026/07/25/rollout-2026-07-25T15-11-26-019f99d4-c5c4-7fc2-87fc-ce0bc95b7983.jsonl
Codex home index:
/var/tmp/inf-aoshen-codex-home/session_index.jsonl
Read the session JSONL if historical reasoning or exact previous commands are
needed. Do not assume that a previous claimed result is proof; verify logs.

AUTHORITATIVE WORKSPACE
=======================
NeMo-RL worktree:
/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu/pr_worktrees/gym-router-url
Experiment scripts/results:
/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu/agent_run
Recipe:
examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml
Launch:
agent_run/scripts/phase4/launch_qwen3_swe2_tp2_e2e.sh
Run:
agent_run/scripts/phase4/run_qwen3_swe2_tp2_e2e.sh
Prewarm:
agent_run/scripts/phase4/prewarm_nemo_gym_qwen3_swe2.sh

RUNTIME INPUTS
==============
Enroot image:
/mnt/lustre01/users/inf-aoshen/enroot/containers/nemo-rl-vllm-latest.sqsh
Model:
/mnt/lustre01/users/inf-aoshen/models/Qwen3-30B-A3B-Thinking-2507
One-line dataset:
agent_run/data/qwen3_swe2_smoke/django15037_swe2.jsonl
ARM64 SIF (verify exact filename):
/mnt/lustre01/users/inf-aoshen/swe-sif-arm64/swebench_sweb.eval.arm64.django_1776_django-15037.sif
Persistent Gym venv root:
/mnt/lustre01/users/inf-aoshen/nemo_gym_venvs/qwen3_swe2
Persistent uv cache:
/mnt/lustre01/users/inf-aoshen/nemo_gym_uv_cache

PARALLELISM / STOP CONDITION
============================
One node, four GPUs. Training TP2/PP1/CP1/EP1; rollout vLLM TP2,
non-colocated. Batch/prompt/generation count is one; max_num_steps=1.
The current run script uses dummy weights for vLLM startup and the training
side performs the normal NeMo-RL initialization; do not silently replace this
with an external vLLM-only test.

COLD-START RULES
=================
1. Prewarm Gym environments once, serially, before the e2e job. setup_command.py
   computes exactly:
   $NEMO_GYM_VENV_DIR/responses_api_models/vllm_model/.venv
   $NEMO_GYM_VENV_DIR/responses_api_agents/swe_agents/.venv
   Train and validation SWE servers share the second venv.
2. The e2e process must set skip_venv_if_present=true, NEMO_GYM_VENV_DIR to the
   persistent root, and UV_CACHE_DIR to the persistent Lustre cache. Verify
   both venv Python binaries before starting NeMo-RL; fail early if missing.
3. Do not let multiple Gym servers race through uv pip install. Subsequent logs
   must contain no uv pip install/build phase; they should activate the existing
   venv and start app.py.
4. Keep the ARM64 SIF read-only and use NEMO_GYM_SWE_NO_COPY=1 for this smoke
   run to avoid recursively copying /testbed. Preserve the default path for a
   non-smoke benchmark; confirm the no-copy path still preserves required
   activate.d variables before claiming full benchmark equivalence.
5. Reuse persistent HF/vLLM/Triton/Inductor caches where supported. Do not
   lower model context or change recipe semantics merely to hide startup cost.

CURRENT EVIDENCE / JOBS
=======================
12881: reached NeMo-Gym/OpenHands runtime but timed out in instance_swe_entry.sh;
no model request was sent.
12889: vLLM v0.25.1 TP2 and Megatron TP2 initialized successfully, but three
Gym servers concurrently ran uv pip install; stopped before rollout.
12891: failed immediately because the first submission mounted the worktree
without agent_run, so /opt/nemo-rl/agent_run did not exist. It is not evidence
about dependencies.
12892: corrected prewarm submission; inspect current state with squeue/sacct.
Prewarm log directory:
agent_run/results/prewarm_qwen3_swe2_20260813T073009Z
Earlier e2e logs:
agent_run/results/qwen3_swe2_tp2_e2e_20260813T071221Z

REQUIRED VERIFICATION
=====================
Before rerun:
  squeue -u inf-aoshen
  test -x /mnt/lustre01/users/inf-aoshen/nemo_gym_venvs/qwen3_swe2/responses_api_models/vllm_model/.venv/bin/python
  test -x /mnt/lustre01/users/inf-aoshen/nemo_gym_venvs/qwen3_swe2/responses_api_agents/swe_agents/.venv/bin/python

After rerun, collect authoritative evidence for all of:
  initialize_runtime_time; Gym server readiness; first /v1/chat/completions;
  vLLM generation; OpenHands tool call; ARM64 SIF command execution;
  output.jsonl; llm completions; SWE evaluation report; reward;
  replay buffer size > 0; Step 1/1; exit_code=0.
Also record timestamps for image/container start, vLLM load_model_complete,
Gym readiness, first request, sandbox start, and final evaluation. Report each
interval separately instead of one vague startup number.

AUDIT REQUIREMENT
=================
Use Codex repeatedly for independent, evidence-based audits (read-only first,
then audit each correction)
for: venv path derivation, package/import compatibility, environment propagation
inside Apptainer, no-copy sandbox semantics, cache/lock races, and whether the
final run is genuinely the flagship NeMo-RL/Gym path. Preserve review output in
the run result directory. A green preflight is not proof of end-to-end success.

SAFE NEXT COMMANDS
==================
  squeue -j 12892 -o '%.18i %.12T %.10M %.20R'
  sacct -j 12892 --format=JobID,State,Elapsed,ExitCode,MaxRSS
  bash agent_run/scripts/phase4/launch_qwen3_swe2_tp2_e2e.sh

Do not touch unrelated job 12890. Do not delete prior logs. Do not mark the
objective complete until the complete chain and every evidence item above are
verified from current files/logs.
HANDOFF

if [[ "${1:-}" == "--check" ]]; then
  root="/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu"
  test -f "$root/pr_worktrees/gym-router-url/examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml"
  test -f "$root/agent_run/scripts/phase4/launch_qwen3_swe2_tp2_e2e.sh"
  test -f "$root/agent_run/scripts/phase4/run_qwen3_swe2_tp2_e2e.sh"
  echo "handoff_path_checks=ok"
fi
