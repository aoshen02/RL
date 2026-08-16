#!/usr/bin/env bash
# Live view of how the router routes multi-turn sessions.
#
#   bash watch_router.sh <run_dir> [interval seconds]
#
# One per-worker line per sampling point:
#   hit%   prefix cache hit tokens / query tokens added during this interval
#          — the tell for whether session-aware routing works: turn N of a session
#            can only hit the KV of turns 1..N-1 if it lands on the same worker
#   req    cumulative completed requests (shows load skew)
#   run/wt current running / waiting, shows queueing
set -u

RUN_DIR="${1:?usage: watch_router.sh <run_dir> [interval seconds]}"
INTERVAL="${2:-15}"
# The caller may start the sampler before the engines are up, so wait here instead of
# exiting immediately
for _ in $(seq 1 120); do
  ROUTER=$(grep -oE "router_url=http://[0-9.]+:[0-9]+" "$RUN_DIR/nemo_rl.log" 2>/dev/null |
    head -1 | cut -d= -f2-)
  [ -n "${ROUTER:-}" ] && break
  # Engines are up but router_url never appears ⇒ ROUTER_POLICY=none, stop waiting
  grep -q "Reserved [0-9]* vLLM server URLs" "$RUN_DIR/nemo_rl.log" 2>/dev/null && break
  sleep 10
done
# ROUTER_POLICY=none has no router, but engine metrics still need sampling:
# take engine URLs straight from the Reserved line in nemo_rl.log — the shared fallback
# for all three policies
engines_from_log() {
  grep -m1 -oE "Reserved [0-9]+ vLLM server URLs: \[[^]]*\]" "$RUN_DIR/nemo_rl.log" 2>/dev/null |
    grep -oE "http://[0-9.]+:[0-9]+"
}

if [ -n "${ROUTER:-}" ]; then
  for _ in $(seq 1 120); do
    WORKERS=$(curl -s --max-time 5 "$ROUTER/workers" 2>/dev/null |
      grep -oE "http://[0-9.]+:[0-9]+")
    [ -n "${WORKERS:-}" ] && break
    sleep 10
  done
else
  ROUTER="(none)"
  for _ in $(seq 1 120); do
    WORKERS=$(engines_from_log)
    [ -n "${WORKERS:-}" ] && break
    sleep 10
  done
fi
[ -n "${WORKERS:-}" ] || { echo "cannot get worker list (router=$ROUTER)"; exit 0; }

echo "router $ROUTER"
echo "worker $(echo $WORKERS | tr '\n' ' ')"

declare -A PQ PH
metric() { grep -m1 "^vllm:$1{" <<<"$2" | awk '{print $NF}'; }

while :; do
  # Print the epoch too, so ts values in nonfinite dumps can be aligned to sampling points
  line="$(date -u +%H:%M:%S) t=$(date +%s)"
  tq=0; th=0
  for w in $WORKERS; do
    m=$(curl -s --max-time 5 "$w/metrics" 2>/dev/null)
    [ -n "$m" ] || { line="$line | $(basename $w) --"; continue; }
    q=$(metric prefix_cache_queries_total "$m"); h=$(metric prefix_cache_hits_total "$m")
    req=$(grep "^vllm:request_success_total{" <<<"$m" | awk '{s+=$NF} END{print s+0}')
    run=$(metric num_requests_running "$m"); wt=$(metric num_requests_waiting "$m")
    # Preemption count and KV usage: the only direct evidence for the "KV pressure"
    # hypothesis; the earlier global-capacity arithmetic was an average that hid each
    # engine's real level
    pre=$(grep -m1 "^vllm:num_preemptions_total{" <<<"$m" | awk '{print $NF}')
    # Metric name changed across versions: old gpu_cache_usage_perc, new
    # kv_cache_usage_perc; try both
    kv=$(grep -m1 -E "^vllm:(gpu|kv)_cache_usage_perc\{" <<<"$m" | awk '{print $NF}')
    dq=$(awk -v a="${q:-0}" -v b="${PQ[$w]:-0}" 'BEGIN{print a-b}')
    dh=$(awk -v a="${h:-0}" -v b="${PH[$w]:-0}" 'BEGIN{print a-b}')
    PQ[$w]=${q:-0}; PH[$w]=${h:-0}
    tq=$(awk -v a=$tq -v b=$dq 'BEGIN{print a+b}'); th=$(awk -v a=$th -v b=$dh 'BEGIN{print a+b}')
    pct=$(awk -v h=$dh -v q=$dq 'BEGIN{if(q>0) printf "%5.1f%%",100*h/q; else printf "    -"}')
    kvp=$(awk -v v="${kv:-0}" 'BEGIN{printf "%2.0f%%",100*v}')
    line="$line | ${w##*:} $pct req=${req%.*} ${run%.*}/${wt%.*} kv=$kvp pre=${pre%.*}"
  done
  tot=$(awk -v h=$th -v q=$tq 'BEGIN{if(q>0) printf "%.1f%%",100*h/q; else printf "-"}')
  echo "$line || total $tot"
  sleep "$INTERVAL"
done
