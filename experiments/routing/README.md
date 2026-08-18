# Routing validation experiments

Validation of vLLM-Router routing policies (none / consistent_hash / cache_aware)
under a NeMo-RL + NeMo-Gym replay workload. Full narrative:
[reports/routing-experiment-report-20260816.md](reports/routing-experiment-report-20260816.md).

## Key results at a glance

One workload throughout: 610 recorded coding-agent sessions, replayed serially
per session, ~80k prompt tokens per request. Only three things vary — routing
policy, engine count, and the in-flight rollout cap
(`env.nemo_gym.max_concurrent_rollouts`, where `0` means uncapped).

| Engines | Cap | Policy | Wall | queries/prompt | Reported hit rate |
|---:|---:|---|---:|---:|---:|
| 28 | 112 | cache_aware | 18:57 | 1.02 | 93.5% |
| 28 | 112 | consistent_hash | 20:26 | 1.01 | 93.5% |
| 28 | 112 | none | 22:03 | 1.09 | 85.6% |
| 28 | uncapped | cache_aware | 13:47 | 1.09 | 91.7% |
| 28 | uncapped | consistent_hash | 12:39 | 1.10 | 91.6% |
| 28 | uncapped | none | **13:13** | 1.55 | 76.1% |
| 12 | uncapped | cache_aware | 13:57 | 1.11 | 91.6% |
| 12 | uncapped | consistent_hash | 15:39 | 1.12 | 91.8% |
| 12 | uncapped | none | **50:33** | 5.61 | 12.3% |
| 12 | uncapped | none (repeat) | 46:11 | 5.50 | 13.4% |
| 12 | 48 | none | 41:19 | 1.05 | 88.0% |

Three things follow, in order of how much they change the reading:

**1. `none` degrades gracefully until the system saturates, then collapses.**
At 28 engines with no cap it costs 76.1% versus 91.7% on cache locality but
finishes in the same time as the routed arms (13:13 vs 13:47). At 12 engines,
the same policy with the same cap takes **50:33** — 3.8x the routed arms on the
same node count. The collapse is not a property of the policy; it is what
happens when the policy's cache penalty pushes the scheduler past a threshold.

**2. The threshold is the per-step token budget, not KV capacity.**
`max_num_batched_tokens` is 8480 while the mean request carries ~80k prompt
tokens, so whether a request admits in one scheduler pass depends on how much of
its prefix is already cached. High hit rate leaves a few thousand uncached
tokens and the request admits immediately; a degraded hit rate leaves tens of
thousands and it waits. Waiting requests then evict each other's blocks, which
degrades the hit rate further. KV utilisation never exceeds 36% in any collapsed
run, and `vllm:num_preemptions_total` is 0 everywhere — memory is not the
constraint. Capping in-flight rollouts at 48 keeps 12 engines out of the
feedback loop entirely (41:19, no queueing).

**3. Concurrency is non-monotonic.** At 28 engines a cap of 112 is *slower* than
no cap at all (18:57 / 20:26 / 22:03 versus 13:47 / 12:39 / 13:13) — the cap
starves the engines. The same knob rescues 12 engines. There is no single good
value; it tracks engine count and cache hit rate.

### Reading the hit-rate column

**`queries/prompt` must be read before the hit rate.** vLLM records a
prefix-cache query per admission *attempt*, before it knows whether the request
can be admitted, so a request that waits N scheduler passes is counted N times.
The ratio `prefix_cache_queries_total / prompt_tokens_total` is ~1.0 when each
request is counted once and rises with queue depth.

At 5.5-5.6 the denominator is inflated ~5.5x and the reported 12.3% is not a
cache-miss rate. Hits are multiply counted as well, so the true value cannot be
recovered from these runs — it lies somewhere between 12.3% and 69.2%
(`hits/prompt_tokens`, itself an upper bound). Rows at 1.0-1.6 are close enough
to clean to quote directly.

Upstream fixed this in
[vllm#48860](https://github.com/vllm-project/vllm/pull/48860) ("Prefix-cache
metrics double-counted when a KV connector defers requests"), which moved the
accounting to admission time. The image used for these runs predates the fix.

**What survives regardless**: wall clock, `queries/prompt` itself (it measures
queueing directly), request counts, prompt/generation token totals, and
preemption counts. The three collapsed rows agree with each other to within 2%
on every one of those.

### Reproducibility of the collapsed case

`50:33` and `46:11` are two independent jobs with byte-identical configuration —
the second was submitted specifically to test whether the first was a fluke.
Their workload counters match exactly (19,594 requests, 1,570,738,525 prompt
tokens, 4,817,220 generation tokens, 0 preemptions) and their queueing matches
closely (waiting depth mean 9.38 vs 9.95, peak 47 vs 46). The 8.6% wall-clock
gap is within the 12.8-16.7% node-to-node range measured separately for this
workload, and the two jobs ran on different nodes.

## Where each number comes from

Inside any run directory:

- **Wall time**: the tqdm completion line in `nemo_rl.log` —
  `grep -oE "rollouts: 100%[^[]*\[[0-9:]+"`, last time field.
- **Experimental variables**: the `Overrides: [...]` line in `nemo_rl.log` pins
  `cluster.num_nodes`, `colocated.resources.num_nodes` (engines = 2x),
  `max_concurrent_rollouts`, `router_url` and `num_prompts_per_step`.
- **Hit rate, queries/prompt, per-engine load, preemptions**:
  `worker_metrics.rollout.txt` (per-engine Prometheus snapshot taken as rollout
  finishes, filtered to the five metric families used as evidence).
- **Queue depth and KV usage**: `metrics_timeline.txt`, sampled every ~30s; each
  engine field is `req=<done> <running>/<waiting> kv=<pct>`.
- **Routing decision logs**: kept as `router.log.gz` samples on the 9B debug
  arms only; the 120B statistics are in the report.
- **One-command recheck**: `scripts/routing_postcheck.sh <run_dir> [sessions]`.

## Layout

- `reports/` — final experiment report (conclusions, mechanism attribution,
  bug chain).
- `scripts/` — launcher (`grpo_workplace_assistant.sh`), acceptance check
  (`routing_postcheck.sh`), timeline sampler (`watch_router.sh`), dataset
  builder (`build_codex_replay_dataset.py`), run curation
  (`curate_run.sh`, raw run directory -> the compact form kept here).
- `results/` — 17 golden runs:
  - 120B / 28 engines, capped and uncapped, three arms each;
  - 120B / 12 engines, uncapped, three arms, plus the collapsed-case repeat and
    the capped-at-48 rescue;
  - 9B post-fix parity arms (`wa_codex_3n_*_1517*Z`), the 9B decision-log
    evidence arms (cache_aware `142909Z` with 1,322 decisions; consistent_hash
    `151650Z` with 16 keys = 16 sessions), and the num_workers causal run
    (`wa_codex_3n_none_20260816T015414Z`: hit rate 95.8% -> 94.2% when
    num_workers goes 1 -> 4).

Datasets (610 sessions, 210M + 138M source pool) are not committed; any subset
can be rebuilt from the source pool with `scripts/build_codex_replay_dataset.py`.

## Open cells

The grid is complete except for **12 engines x cap 48 x the two routed
policies**. Without those the cap's effect at 12 engines is measured only for
`none`.
