# Routing validation experiments

Validation of vLLM-Router routing policies (none / consistent_hash / cache_aware)
under a NeMo-RL + NeMo-Gym replay workload. Full narrative:
[reports/routing-experiment-report-20260816.md](reports/routing-experiment-report-20260816.md).

## Key results at a glance

120B / 28 engines / 610 sessions, three-arm comparison:

| Arm | Rollout wall | Prefix-cache hit rate | Run directory |
|---|---|---|---|
| cache_aware | 18:57 | 93.5% | `results/wa_codex_16n_cache_aware_20260815T162244Z` |
| consistent_hash | 20:26 | 93.5% | `results/wa_codex_16n_consistent_hash_20260815T175723Z` |
| none | 22:03 | 85.6% | `results/wa_codex_16n_none_20260815T175720Z` |

Where each number comes from (inside any run directory):

- **Wall time**: the tqdm completion line in `nemo_rl.log` —
  `grep -oE "rollouts: 100%[^[]*\[[0-9:]+"`, last time field.
- **Hit rate / per-engine load / preemptions**: `worker_metrics.rollout.txt`
  (per-engine Prometheus snapshot taken as rollout finishes, filtered to the
  four metric families used as evidence; hit rate =
  `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total`).
- **Routing decision logs**: kept as `router.log.gz` samples on the 9B debug
  arms only (cache_aware matched/input chars, consistent_hash hash keys);
  the 120B statistics are in the report.
- **Timeline**: `metrics_timeline.txt` (busy engines, KV usage samples).
- **One-command recheck**: `scripts/routing_postcheck.sh <run_dir> [sessions]`.

## Layout

- `reports/` — final experiment report (conclusions, mechanism attribution,
  bug chain).
- `scripts/` — launcher (`grpo_workplace_assistant.sh`), acceptance check
  (`routing_postcheck.sh`), timeline sampler (`watch_router.sh`), dataset
  builder (`build_codex_replay_dataset.py`).
- `results/` — nine golden runs: the 120B three arms, the 9B post-fix parity
  arms (`wa_codex_3n_*_1517*Z`), the 9B decision-log evidence arms
  (cache_aware `142909Z` with 1,322 decisions; consistent_hash `151650Z` with
  16 keys = 16 sessions), and the num_workers causal run
  (`wa_codex_3n_none_20260816T015414Z`: hit rate 95.8% → 94.2% when
  num_workers goes 1 → 4).

Datasets (610 sessions, 210M + 138M source pool) are not committed; any subset
can be rebuilt from the source pool with `scripts/build_codex_replay_dataset.py`.
