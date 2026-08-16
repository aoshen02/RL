#!/usr/bin/env bash
# Extract the evidence chain required by the handoff from a run dir.
# Usage: collect_evidence_qwen3_swe2.sh <run_dir> [since]
#
# The first version of this script grepped the slurm log for `Added request`
# and `/v1/chat/completions`. Both are impossible to hit: NeMo-RL installs a
# vLLM logging filter for the former and a uvicorn filter for the latter
# ("Adding a ... logging filter so that the logs aren't spammed"), so those
# checks reported NO on a healthy run. Every signal below is taken from a
# source that was confirmed to actually carry it.
set -uo pipefail

RD="${1:?usage: $0 <run_dir> [since]}"
SINCE="${2:-}"
LOG="$(ls "$RD"/slurm-*.log 2>/dev/null | head -1)"
[[ -s "$LOG" ]] || { echo "no non-empty slurm log in $RD" >&2; exit 1; }

SWE_RESULTS=/mnt/lustre01/users/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu/pr_worktrees/gym-router-url/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
OH_EVAL="$SWE_RESULTS/swe_openhands_setup/OpenHands/evaluation/oh"

# Restrict to artifacts from this run. A stale sandbox dir from an earlier job
# previously made a failed run look like it had produced evidence.
newer() { if [[ -n "$SINCE" ]]; then find "$1" -newermt "$SINCE" "${@:2}"; else find "$1" "${@:2}"; fi; }

# entry_trace.log (xtrace) is gone: its `exec` redirection permanently rewired
# the sourcing shell's fds, broke OpenHands' PS1 completion detection and hung
# every action for 600s. The replacement redirects single commands only and is
# named entry.log. Accept either so older run dirs still parse.
SANDBOX="$(newer "$SWE_RESULTS/results" \( -name entry.log -o -name entry_trace.log \) 2>/dev/null | head -1)"
SANDBOX="$(dirname "${SANDBOX:-/nonexistent}")"

ok()   { printf '%-26s YES  %s\n' "$1" "${2:-}"; }
no()   { printf '%-26s NO   %s\n' "$1" "${2:-}"; }
show() { sed 's/^/      /'; }

echo "=== run   : $RD"
echo "=== log   : $LOG ($(wc -l < "$LOG") lines)"
echo "=== sandbox: ${SANDBOX:-none}"
echo
echo "--- evidence chain ---"

# 1. Gym venvs were prewarmed, and this process installed nothing.
grep -q "preflight: verifying Gym venvs" "$LOG" \
  && ok "1 gym venv preflight" || no "1 gym venv preflight"
if grep -qE "uv pip install|Building [^ ]+ @ file" "$LOG"; then
  no "  cold-start: no install" "(unexpected install in e2e process)"
else
  ok "  cold-start: no install"
fi

# 2. Gym servers came up. uvicorn's 200 OK lines are filtered, so count startups.
n=$(grep -c "Application startup complete" "$LOG")
[[ "$n" -gt 0 ]] && ok "2 gym servers ready" "($n servers)" || no "2 gym servers ready"

# 3. initialize_runtime — authoritative source is Gym's own metrics file.
M="$SANDBOX/nemo_gym_metrics.json"
if [[ -s "$M" ]]; then
  ok "3 initialize_runtime" "$(tr -d '\n' < "$M")"
else
  no "3 initialize_runtime" "(no nemo_gym_metrics.json)"
fi

# 4. ARM64 SIF actually executed, and the entry script completed.
T="$SANDBOX/entry.log"; [[ -s "$T" ]] || T="$SANDBOX/entry_trace.log"
if [[ -s "$T" ]]; then
  bad=$(grep -c "No item found" "$T")
  sym=$(grep -c "Using /testbed directly" "$T" 2>/dev/null)
  # the symlink echo goes to the agent log; entry.log carries workspace= instead
  [[ "$sym" -eq 0 ]] && sym=$(grep -c "completed workspace=." "$T" 2>/dev/null)
  fin=$(grep -c "\[entry\].*completed" "$T")
  # The old xtrace had a "jq check" probe; entry.log does not. Report the host
  # binary's arch instead — an empty `jq=` field read as "jq broken" when in
  # fact the aarch64 fallback was fine.
  jqbin="$SWE_RESULTS/swe_openhands_setup/miniforge3/bin/jq"
  jqv="$(file -b "$jqbin" 2>/dev/null | cut -d, -f2 | tr -d ' ')"
  jqv="${jqv:-absent}"
  if [[ "$bad" -eq 0 && "$sym" -gt 0 && "$fin" -gt 0 ]]; then
    ok "4 arm64 sif + swe entry" "(jq_binary=$jqv, symlink workspace, completed)"
  else
    no "4 arm64 sif + swe entry" "(no_item=$bad symlink=$sym completed=$fin jq=$jqv)"
  fi
else
  no "4 arm64 sif + swe entry" "(no entry_trace.log)"
fi

# 5. OpenHands ran and emitted an output file.
A="$(ls "$SANDBOX"/apptainer_logs/*_agent.log 2>/dev/null | head -1)"
if [[ -s "$A" ]]; then
  ok "5 openhands started" "$(grep -c 'connect to runtime' "$A") runtime connects"
  grep -E "connect to runtime|OUTPUT FILE" "$A" | tail -2 | show
else
  no "5 openhands started"
fi

# 6. OpenHands' own output.jsonl. Measured empty (0 bytes) on every run of this
# path, so it is reported for completeness only and is NOT the trajectory
# source — see item 7.
OJ="$(newer "$OH_EVAL" -name output.jsonl 2>/dev/null | head -1)"
if [[ -s "$OJ" ]]; then
  ok "6 output.jsonl" "$OJ ($(wc -c < "$OJ") bytes)"
else
  no "6 output.jsonl" "(0 bytes on this path; trajectory comes from item 7)"
fi

# 7. The full interaction trajectory, and the model patch it produced. The
# authoritative record is train_data_step*.jsonl — what training actually sees.
# Its `content` field is empty, so dump_trajectory_qwen3_swe2.py decodes
# token_ids and splits on token_loss_mask to recover each turn.
TD="$(ls "$RD"/nemo-rl-logs/*/train_data_step*.jsonl 2>/dev/null | head -1)"
TJ="$(ls "$RD"/trajectory_step*.json 2>/dev/null | head -1)"
if [[ -s "$TJ" ]]; then
  ok "7 full trajectory" "$TJ"
  python3 - "$TJ" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
for r in rows:
    segs = r["segments"]
    gen = [s for s in segs if s["generated"]]
    env = [s for s in segs if not s["generated"]]
    print(f"      sample {r['sample']}: tokens={r['num_tokens']} "
          f"generated={r['num_generated_tokens']} reward={r['reward']} "
          f"turns={len(gen)} env_segments={len(env)}")
    for i, s in enumerate(gen):
        head = " ".join(s["text"].split())[:70]
        print(f"        model turn {i + 1} [{s['start']}:{s['end']}] {head}...")
PY
elif [[ -s "$TD" ]]; then
  no "7 full trajectory" "(train_data present but not decoded — run dump_trajectory_qwen3_swe2.py)"
else
  no "7 full trajectory" "(no train_data_step*.jsonl; needs should_log_nemo_gym_responses=false)"
fi
if [[ -s "$M" ]]; then
  python3 - "$M" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
patch = m.get("model_patch") or ""
print(f"      model_patch: {len(patch)} bytes, patch_exists={m.get('patch_exists')}, "
      f"resolved={m.get('resolved')}")
lat = (m.get("per_turn_metrics") or {}).get("response_latencies") or []
acts = (m.get("per_turn_metrics") or {}).get("action_execution_latencies") or []
print(f"      llm_calls={len(lat)} tool_observations={len(acts)}")
for a in acts:
    print(f"        {a.get('observation_type')}: {a.get('message', '')[:70]}")
PY
fi

# 8. SWE-bench evaluation report.
REP="$(newer "$SWE_RESULTS/results" -name report.json 2>/dev/null | head -1)"
if [[ -s "$REP" ]]; then
  ok "8 swe eval report" "$REP"
  # `head -c` does not end on a newline, which glued the next check's "9 reward"
  # line onto this one — so a NO on item 9 would not be line-anchored and the
  # verifier's failure grep would miss it entirely.
  { head -c 400 "$REP"; echo; } | show
else
  no "8 swe eval report"
fi

# 9. Reward. A bare `reward` grep also matches config echoes such as
# `reward_weight` in the dumped MasterConfig, so require a metric-shaped key
# and drop the config dump lines.
# The authoritative line is the collector's own "Rewards stats:" summary. An
# earlier pattern required `reward=<num>` and so reported NO on a healthy run
# whose reward was legitimately 0.0 (agent patched but FAIL_TO_PASS failed).
rw=$(grep -oE "Rewards stats:.*|Advantages stats:.*" "$LOG" | tail -2)
if [[ -n "$rw" ]]; then
  ok "9 reward"; printf '%s\n' "$rw" | show
else
  no "9 reward"
fi
# `Starting async GRPO training with buffer_size=2` reports the buffer's
# configured capacity, not its occupancy. Counting it as evidence would pass
# this check on a run whose buffer never held a single trajectory, so read
# occupancy only from the collector's own periodic report.
buf=$(grep -E "Wait iteration [0-9]+: buffer_size=" "$LOG" \
      | grep -oE "buffer_size=[0-9]+" | sort -t= -k2 -n | tail -1)
if [[ -n "$buf" && "${buf#buffer_size=}" -gt 0 ]]; then
  ok "10 replay buffer > 0" "(max $buf)"
else
  no "10 replay buffer > 0" "(max ${buf:-none})"
fi
grep -qE "Step 1/1|step 1/1" "$LOG" && ok "11 training step 1/1" || no "11 training step 1/1"

# 12. Exit code. Must be 0 — an earlier version only checked that the key was
# present, so a run that exited 1 still passed this item. Caught by a negative
# test that rewrote run.env to exit_code=1 and still got YES.
ec=$(grep -m1 '^exit_code=' "$RD/run.env" 2>/dev/null | cut -d= -f2)
if [[ -z "$ec" ]]; then
  no "12 exit_code" "(run not finished)"
elif [[ "$ec" == "0" ]]; then
  ok "12 exit_code" "exit_code=0"
else
  no "12 exit_code" "(exit_code=$ec)"
fi

echo
echo "--- timeline (each interval reported separately) ---"
grep -hE "^started=|^finished=|^exit_code=" "$RD/run.env" 2>/dev/null | show
# stamp() writes RUN_DIR/timeline.tsv as "iso<TAB>epoch<TAB>label". Report the
# gap between consecutive stamps, since the handoff asks for each interval
# separately rather than one aggregate startup number.
if [[ -s "$RD/timeline.tsv" ]]; then
  awk -F'\t' '{ if (p) printf "      %-22s +%8.1fs  (%s)\n", $3, $2-p, $1;
                else printf "      %-22s %10s  (%s)\n", $3, "start", $1;
                p=$2 }' "$RD/timeline.tsv"
fi
for f in generation_apptainer_spinup_timestamp final_eval_apptainer_spinup_timestamp; do
  [[ -s "$SANDBOX/$f" ]] && printf '      %-42s %s\n' "$f" "$(cat "$SANDBOX/$f")"
done
[[ -s "$M" ]] && printf '      %-42s %s\n' "nemo_gym_metrics" "$(tr -d '\n' < "$M")"
grep -nE "preflight:|Application startup complete|load_model_complete|Model loading took" "$LOG" | head -8 | show
[[ -s "$T" ]] && grep -E "^\[entry\]" "$T" | head -2 | show

echo
echo "--- errors ---"
grep -nE "Traceback|FATAL|CheckpointException|timed out after|CUDA out of memory|AssertionError" "$LOG" | head -10 || echo "      (none)"
