#!/usr/bin/env bash
# Curate one raw run directory into the compact form kept under results/.
#
#   bash curate_run.sh <src_run_dir> <dst_run_dir>
#
# Raw runs are 1-4 MB of driver log plus a ~900 KB Prometheus dump per arm.
# What the analysis actually reads is: the resolved Hydra overrides (which pin
# node count, generation nodes, routing policy and max_concurrent_rollouts),
# the tqdm completion line, the per-engine metrics timeline, and five vLLM
# counters. Everything else stays in agent_run/results.
#
# prompt_tokens_total is retained because prefix_cache_queries_total /
# prompt_tokens_total is how a queue-inflated hit rate is detected: the counter
# records one query per admission *attempt*, so a saturated engine counts the
# same request several times and the reported hit rate collapses without the
# cache actually degrading.
set -euo pipefail

SRC="${1:?usage: curate_run.sh <src_run_dir> <dst_run_dir>}"
DST="${2:?usage: curate_run.sh <src_run_dir> <dst_run_dir>}"
mkdir -p "$DST"

KEEP='^vllm:(prefix_cache_(queries|hits)_total|request_success_total|num_preemptions_total|prompt_tokens_total)'

if [ -f "$SRC/nemo_rl.log" ]; then
  # Held in a variable rather than piped: `sed | grep -q` short-circuits on the
  # first match, sed dies on SIGPIPE, and pipefail then reports the whole
  # pipeline as failed -- which would duplicate the overrides line.
  HEAD=$(sed -n '1,30p' "$SRC/nemo_rl.log")
  {
    printf '%s\n' "$HEAD"
    # The overrides line pins every experimental variable; keep it even when it
    # falls outside the head window.
    case "$HEAD" in
      *"Overrides: ["*) ;;
      *) grep -m1 '^Overrides: \[' "$SRC/nemo_rl.log" || true ;;
    esac
    echo '[... trimmed: full log in agent_run/results ...]'
    grep -aE 'rollouts: 100%' "$SRC/nemo_rl.log" | tail -2 || true
  } > "$DST/nemo_rl.log"
fi

for f in metrics_timeline.txt list_workers.json; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/$f"
done

for f in worker_metrics.rollout.txt worker_metrics.exit.txt; do
  [ -f "$SRC/$f" ] && grep -aE "^### |$KEEP" "$SRC/$f" > "$DST/$f" || true
done

echo "curated $(basename "$SRC") -> $DST ($(du -sh "$DST" | cut -f1))"
