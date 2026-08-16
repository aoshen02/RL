#!/usr/bin/env bash
# Routing acceptance check: given a run_dir, report whether this arm's routing actually
# took effect, plus all verifiable evidence.
#
#   bash routing_postcheck.sh <run_dir> [expected session count]
#
# Checks (by applicable policy):
#   all      : wall clock / HTTP 500 / preemption count / per-engine request count and prefix hit rate
#   cache_aware + debug     : decision branch ratios, misroute (large input to min-load) count, radix residency
#   consistent_hash + debug : distinct hash keys vs sessions, per-key occurrence distribution, key source headers
set -u

RUN_DIR="${1:?usage: routing_postcheck.sh <run_dir> [expected session count]}"
EXPECT_SESSIONS="${2:-}"
cd "$RUN_DIR" || exit 1

POLICY=$(basename "$RUN_DIR" | grep -oE "cache_aware|consistent_hash|none")
pass=0; fail=0
ok()   { echo "  ✓ $*"; pass=$((pass+1)); }
bad()  { echo "  ✗ $*"; fail=$((fail+1)); }
info() { echo "    $*"; }

echo "== $(basename "$RUN_DIR") (policy=$POLICY) =="

# ---- Common ----
WALL=$(grep -oE "rollouts: 100%[^[]*\[[0-9:]+" nemo_rl.log 2>/dev/null | grep -oE "[0-9]+:[0-9]+$" | tail -1)
[ -n "$WALL" ] && ok "rollout complete wall=$WALL" || bad "rollout incomplete"

N500=$(grep -c " 500 Internal" nemo_rl.log 2>/dev/null || true)
[ "${N500:-0}" = 0 ] && ok "HTTP 500 = 0" || bad "HTTP 500 = $N500"

# Prefer the rollout snapshot: the exit snapshot is often taken after the engines quit,
# leaving only an empty shell
M=worker_metrics.rollout.txt
[ -s "$M" ] || M=worker_metrics.exit.txt
Q=$(awk '/^vllm:prefix_cache_queries_total/{s+=$NF} END{print s+0}' "$M" 2>/dev/null)
if [ "${Q%.*}" != 0 ] && [ -n "${Q:-}" ]; then
  PRE=$(awk '/^vllm:num_preemptions_total/{s+=$NF} END{print s+0}' "$M")
  [ "${PRE%.*}" = 0 ] && ok "preemptions = 0" || bad "preemptions = $PRE (violates no-preemption requirement)"
  awk '/^### /{u=$2} /^vllm:prefix_cache_queries_total/{q[u]+=$NF; tq+=$NF}
       /^vllm:prefix_cache_hits_total/{h[u]+=$NF; th+=$NF}
       /^vllm:request_success_total/{r[u]+=$NF}
       END{for(u in q) printf "    engine %s req=%d hit=%.1f%%\n", u, r[u], (q[u]>0?100*h[u]/q[u]:0);
           if(tq>0) printf "  ✓ overall prefix hit rate %.1f%%\n", 100*th/tq}' "$M"
  pass=$((pass+1))
else
  bad "engine metrics snapshot empty ($M) — engines may have exited before sampling"
fi

# ---- Routing decisions (when debug logs exist) ----
if [ "$POLICY" = cache_aware ] && [ -f router.log ]; then
  S=$(sed -e 's/\x1b\[[0-9;]*m//g' router.log)
  N=$(grep -c "Cache match" <<<"$S" || true)
  if [ "${N:-0}" -gt 0 ]; then
    grep -oE "matched_chars=[0-9]+, input_chars=[0-9]+, match_rate=[0-9.]+" <<<"$S" |
      sed 's/[^0-9.]\+/ /g' |
      awk '{n++; if($3>0.3) hit++; else {miss++; if($2>60000) mis++}}
           END{printf "    decisions %d: cache-hit branch %d (%.1f%%), min-load %d\n", n, hit, 100*hit/n, miss;
               if(mis>0) printf "  ✗ large inputs (>60K chars) misrouted to min-load: %d\n", mis;
               else print "  ✓ no large-input misroutes, all min-load decisions are session first turns"}'
    LAST_TREE=$(grep -A5 "After eviction" <<<"$S" | grep -oE "Size: [0-9]+" | tail -4 | awk '{s+=$2} END{print s+0}')
    [ "${LAST_TREE:-0}" -gt 0 ] && ok "radix tree has residency (last snapshot total $LAST_TREE chars)" ||
      info "radix tree last snapshot is 0 (may have been cleared at end of run; trust the decision logs)"
  else
    info "no decision logs (ROUTER_DEBUG=0); only hit rate and distribution available"
  fi
fi

if [ "$POLICY" = consistent_hash ] && [ -f router.log ]; then
  S=$(sed -e 's/\x1b\[[0-9;]*m//g' router.log)
  KEYS=$(grep -oE "found session key in header '[^']+': [^ ]+" <<<"$S" | awk '{print $NF}')
  if [ -n "$KEYS" ]; then
    NKEY=$(sort -u <<<"$KEYS" | wc -l)
    NREQ=$(wc -l <<<"$KEYS")
    DIST=$(sort <<<"$KEYS" | uniq -c | awk '{print $1}' | sort -n)
    MIN=$(head -1 <<<"$DIST"); MED=$(sed -n "$(( $(wc -l <<<"$DIST") / 2 ))p" <<<"$DIST"); MAX=$(tail -1 <<<"$DIST")
    info "hash key: $NKEY distinct keys / $NREQ requests, per-key occurrences min=$MIN p50=$MED max=$MAX"
    HDRS=$(grep -oE "in header '[^']+'" <<<"$S" | sort -u | tr -d "'" | sed 's/in header //' | tr '\n' ' ')
    info "key source headers: $HDRS"
    if [ -n "$EXPECT_SESSIONS" ]; then
      if [ "$NKEY" -le $((EXPECT_SESSIONS + EXPECT_SESSIONS / 4)) ]; then
        ok "key count $NKEY ≈ session count $EXPECT_SESSIONS, session affinity holds"
      else
        bad "key count $NKEY far exceeds session count $EXPECT_SESSIONS ⇒ session id drifts across turns, affinity broken"
      fi
    fi
    [ "${MED:-0}" -ge 4 ] && ok "keys survive across turns (p50=$MED occurrences)" || bad "each key appears only $MED times ⇒ affinity broken"
  else
    info "no hash key logs (ROUTER_DEBUG=0)"
  fi
fi

echo "Result: $pass passed, $fail failed"
[ "$fail" = 0 ]
