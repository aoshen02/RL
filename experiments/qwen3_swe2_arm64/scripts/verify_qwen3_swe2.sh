#!/usr/bin/env bash
# 单命令校验:把 handoff 要求的全部检查串起来,并用退出码表达结论。
#
# 为什么需要它:handoff_qwen3_swe2_cold_start.sh 是**上下文文档**,不是校验器 ——
# 139 行里只有一个 `cat` 和三个 `test -f`(见 handoff_exec.txt),跑它 rc=0 只说明
# 三个文件存在,不说明端到端跑通。真正的判据是 collect_evidence 的 12 项。
#
# 用法:
#   verify_qwen3_swe2.sh                      # 自动取最新的 e2e run 目录
#   verify_qwen3_swe2.sh <run_dir> [since]
# 退出码:
#   0 = 全部必需项通过    1 = 有必需项未通过    2 = 用法/环境错误
set -uo pipefail

NMU=/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu
SCRIPTS="$NMU/agent_run/scripts/phase4"
HANDOFF="$NMU/agent_run/handoff_qwen3_swe2_cold_start.sh"
PY=/home/inf-aoshen/vllm/.venv/bin/python   # 宿主上唯一带 transformers 的解释器

RD="${1:-$(ls -dt "$NMU"/agent_run/results/qwen3_swe2_tp2_e2e_* 2>/dev/null | head -1)}"
[[ -d "$RD" ]] || { echo "no run dir: ${RD:-<none>}" >&2; exit 2; }
SINCE="${2:-$(grep -m1 '^started=' "$RD/run.env" 2>/dev/null | cut -d= -f2 | tr 'TZ' '  ')}"

echo "### verify: $(basename "$RD")   since='${SINCE:-<all>}'"
echo

echo "--- 1/4 handoff 路径校验 ---"
bash "$HANDOFF" --check | tail -1 || { echo "handoff --check FAILED" >&2; exit 1; }

echo
echo "--- 2/4 轨迹导出(缺失时才做;需要 tokenizer)---"
if ls "$RD"/trajectory_step*.json >/dev/null 2>&1; then
  echo "already present: $(ls "$RD"/trajectory_step*.json | wc -l) file(s)"
elif [[ -x "$PY" ]]; then
  "$PY" "$SCRIPTS/dump_trajectory_qwen3_swe2.py" "$RD" || echo "(轨迹导出失败,继续)"
else
  echo "(跳过:$PY 不存在)"
fi

echo
echo "--- 3/4 证据链 ---"
EV="$RD/EVIDENCE.txt"
bash "$SCRIPTS/collect_evidence_qwen3_swe2.sh" "$RD" "$SINCE" > "$EV" 2>&1
grep -E "^[0-9]+ |^  cold-start" "$EV" | cut -c1-120

echo
echo "--- 4/4 判定 ---"
# item 6(OpenHands output.jsonl)在这条路径上恒为 0 字节,轨迹由 item 7 提供,
# 因此它不是必需项;其余 12 项全部必需。
fails=$(grep -E "^[0-9]+ .*NO|^  cold-start.*NO" "$EV" | grep -v "^6 output.jsonl" || true)
n_yes=$(grep -cE "YES" "$EV")
if [[ -z "$fails" ]]; then
  echo "PASS  必需项全部通过(YES 计数=$n_yes)"
  echo "      证据:$EV"
  echo "      轨迹:$(ls "$RD"/trajectory_step*.txt 2>/dev/null | head -1)"
  exit 0
fi
echo "FAIL  以下必需项未通过:"
printf '%s\n' "$fails" | sed 's/^/      /'
exit 1
