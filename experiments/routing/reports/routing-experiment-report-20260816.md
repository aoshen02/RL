# vLLM-Router Routing Policy Experiment Report

2026-08-15/16 · NeMo-RL + NeMo-Gym · nova-hazel GB200 cluster

## TL;DR

At 28 engines / 610 sessions / ~19.6K requests, the three routing policies separate clearly:

| Arm | Rollout wall | Prefix-cache hit rate | Per-engine load (req) |
|---|---|---|---|
| **cache_aware** | **18:57** | **93.5%** | 505–942 (1.9×) |
| consistent_hash | 20:26 | **93.5%** | 446–1100 (2.5×) |
| none (no router) | 22:03 | 85.6% | 562–828 |

Three main conclusions:

1. **cache_aware is the best default in the current setup**: same hit rate as
   consistent_hash, 7% faster wall clock, narrowest load spread.
2. **The router's core value is being the only process-independent session-affinity
   layer.** Without a router, Gym's own affinity fragments per process once
   `num_workers > 1`: hit rate drops 8 points and wall clock grows 14%. This
   mechanism is confirmed both in source and by a causal experiment.
3. **All policy differences concentrate in one decision: which engine a session's
   first turn lands on.** All three keep affinity on later turns (once the bugs are
   fixed), so the gap shows up as load balance (→ wall clock), not hit rate.

## 1. Experiment design

### Workload: deterministic replay

We built `codex_replay_agent`: it replays recorded codex trajectories turn by
turn — each turn sends the full conversation so far as one real request, then
**discards the model output** and appends the recorded assistant/user messages.
All arms therefore see a byte-identical request sequence with zero sampling
noise, which is the precondition for comparing routing policies head-to-head.

### Dataset: `codex_replay_p30_all`

- 610 sessions (full pool), stratified by turn count
  6-15:106 / 16-30:222 / 31-50:198 / 51-75:64 / 76+:19;
- turns p50 29 / p90 56 / max 99; final-turn context p50 81K / max 207K tokens
  (250K budget truncation, only 3 rows clipped);
- ~19.6K requests per arm, wall clock on the order of 20 minutes.

### Shapes

- **9B functional validation**: 3 nodes, 4×TP1 engines, 128 sessions
  (tail-heavy subset), Nemotron-Nano-9B.
- **120B scale experiment**: 16 nodes = 2 training (TP8/EP8) + 14 generation ×
  2 TP2 engines per node = **28 engines**,
  NVIDIA-Nemotron-3-Super-120B-A12B-BF16.
- Every run finished with zero preemptions and zero HTTP 500s (enforced by the
  acceptance script).

### Request path

```
rollout loop (optional semaphore cap)
  → codex_replay_agent (self-call /v1/responses, cookies forwarded)
    → policy_model vllm_model (num_workers uvicorn processes)
      → [vllm-router (absent in the none arm)]
        → 28 × vLLM engines (TP2)
```

## 2. Policy mechanics (verified in source)

**none (baseline)**: no router. Gym's `vllm_model._resolve_client` assigns each
session id a client by **arrival order** (Nth new session → engine `N % 28`) and
caches it in an in-process dict. First turn ≈ round-robin; later turns pinned by
the dict.

**consistent_hash**: a hash ring. Each engine contributes 160 virtual nodes on a
u64 ring (BTreeMap); each request walks clockwise from `hash(session id)`.
**Zero load awareness anywhere**, stateless, ring rebuilt only when the engine
set changes. One design wart we found: when the target engine is unhealthy it
falls back to `healthy_indices[0]` — all misrouted traffic piles onto the first
healthy engine.

**cache_aware**: three-stage decision. (1) Imbalance valve: when max-min > 64
AND ratio > 1.5, force min-load (fired 0 times in 39,164 decisions here);
(2) radix-tree prefix match rate > 0.3 → the matching engine;
(3) ≤ 0.3 → min-load. Every request is `tree.insert`ed (sticky re-homing);
tree capacity 67M chars with background LRU eviction every 120s. The tree is
in-memory — a router restart forgets it.

**First-turn placement, side by side** (the entire source of the differences):

| | First-turn placement | Load-aware | Process-independent |
|---|---|---|---|
| none | arrival-order round-robin % 28 (per-process counter) | no | ✗ |
| consistent_hash | hash-ring position | no | ✓ |
| cache_aware | min-load (when the tree has no match) | yes | ✓ |

## 3. Results and attribution

### 9B: three-way tie (expected and correct)

With all bugs fixed, none / consistent_hash / cache_aware all land at ≈95.8%
hit rate, ≈22 minutes. Reason: the 9B recipe runs policy_model as a **single
process**, so Gym's dict-based affinity is lossless; the KV pool is under zero
pressure, so any placement fits. A router showing no gain here is correct
behavior, not a failed experiment.

### 120B: the none arm loses 8 points — num_workers process fragmentation

The 120B recipe sets `num_workers: 4` for policy_model (4 uvicorn processes).
Gym's session→engine pinning is **per-process memory**: four processes keep four
independent ledgers, and the same session gets pinned to different engines in
different processes. Which process serves a turn depends on which pooled TCP
connection carries it (new connections are distributed by a kernel accept race;
keep-alive provides only accidental stickiness). At concurrency 112, ~10% of
turns drift to another process → another engine → the whole prefix misses,
spread evenly across all 28 engines (82–88% each) — exactly the 8-point gap.

**Causal confirmation (job 13201)**: 9B none arm, the only variable changed
being num_workers 1→4: hit rate 95.8% → 94.2% (−1.6pp). The smaller drop than
120B's 8pp is consistent with the mechanism — 9B ran at concurrency 8 vs 120B's
112, so connection churn is an order of magnitude rarer. Conclusion:
**num_workers > 1 is the necessary condition; high concurrency is the
amplifier.** The router sits behind policy_model and is inherently immune to
process fragmentation.

### 120B: ch and ca tie on hit rate, differ 7% on wall — load spread

Both router policies hit 93.5% (affinity intact, KV fits everywhere); what
differs is load distribution: consistent_hash has no load feedback, so long-tail
sessions pile up on hash positions (446–1100, 2.5×); cache_aware places first
turns by min-load and carries an imbalance valve, so its band is much narrower
(505–942, 1.9×). Rollout is a **drain-the-stragglers** problem — the most loaded
engine's tail sets the wall clock, so load spread converts directly into time.

The none arm's worst wall (22:03) follows the same logic: static pinning plus
per-process round-robin does no load balancing at all.

### Two solved mysteries worth recording

- **"Every hash key appears exactly twice"**: the router logs each request's
  decision twice (39,164 log lines = 2 × 19,582 requests). During the cookie bug
  era, "every key twice" = a fresh session id per turn × double logging — the
  most misleading false lead of the campaign.
- **cache_aware's 20 large-input min-load fallbacks** (0.05%): all occurred
  after the radix tree started evicting (capacity 67M chars vs 200M+ total
  text) — by-design fallback, not misrouting.

## 4. The bug chain fixed along the way

In order of discovery; each one had directly polluted the comparison:

1. **Router cache_aware ineffective for chat completions** (router PR #213,
   fixed before this campaign): `extract_text_for_routing` returned the session
   id instead of the conversation text, so the radix tree never matched and the
   policy silently degraded to min-load. Post-fix decision logs show 96.8% of
   requests taking the cache-hit branch.
2. **Gym two-hop cookie loss** (the true root cause of broken session
   affinity): the agent's `run()` first POSTs to its own `/v1/responses`, which
   forwards to policy_model. The inner hop did not pass
   `cookies=request.cookies`, so policy_model created a fresh session per call.
   All upstream self-call agents follow the cookie-forwarding convention; our
   new agent missed it. Located with a full-chain probe on a login node (both
   real apps + a stub backend + wire logging); an earlier "outer-loop cookie
   merge" fix had failed because the break was the inner hop.
3. **TP2 + flashinfer TRT-LLM MoE OOMs on load** (jobs 13175-13177): 115G
   weights per GPU plus ~60G flashinfer weight-relayout buffer does not fit in
   184G. Fix: pin `moe_backend=triton` for 120B (uses weights in place).
4. **Slurm dirty-GPU trap** (5 jobs killed on arrival): a crashed job's GPU
   memory releases slower than Slurm reschedules, so the next job dies with
   `Free memory 23/184G`. Fix: a Ray-based per-node GPU-clean barrier in the
   launcher (>2048 MiB counts as dirty, 15s recheck, 600s deadline).
5. **num_workers process fragmentation** (§3): a deployment-shape trap rather
   than a bug, but it polluted the comparison the same way — it is the entire
   explanation of the none arm's 85.6%.

## 5. Observability assets

- `routing_postcheck.sh <run_dir> [expected-sessions]`: one-command acceptance
  for any arm — wall / 500s / preemptions / per-engine hit rates; cache_aware
  decision-branch and misroute counts; consistent_hash key-count = session-count
  and key-lifetime checks. Self-validated against known-bad old data.
- Engine `/metrics` (Prometheus) mount: the source of every hit-rate number.
- `ROUTER_DEBUG=1` per-request decision logs (cache_aware matched/input chars,
  consistent_hash hash keys).
- `watch_router.sh` timeline sampling (busy engines, KV usage), with a
  log-anchor fallback for the none arm.
- Launcher: GPU-clean barrier, NaN response tripwire (since removed after the
  root cause was fixed).

## 6. Not yet closed

- **120B none + num_workers=1 symmetric verification** (expected to return to
  ~93.5%): needs 16 nodes.
- **W∈{1,2,4} @ c=112 sweep** (9B, 3 arms, ~50 min): first-principles matrix
  predicting drift ≈ (W-1)/W × connection-churn rate, with W=1 as the immune
  control. Designed, awaiting a slot.
- **Eviction-pressure tier**: engine-side KV was under zero pressure throughout
  (10.7M tokens per engine vs ~2M per engine of session text). "Who stays
  stable under KV eviction" needs a separate experiment with a compressed KV
  pool.
- Upstreaming: cookie fix + a footnote on the hash-key body path;
  consistent_hash's unhealthy-fallback pile-up is worth an issue.

## Appendix: material index

Result data (`results/`, provenance for every number above):

| Run directory | Contents |
|---|---|
| `wa_codex_16n_cache_aware_...162244Z` | 120B ca arm (job 13180) |
| `wa_codex_16n_consistent_hash_...175723Z` | 120B ch arm (13192) |
| `wa_codex_16n_none_...175720Z` | 120B none arm (13191) |
| `wa_codex_3n_{none,cache_aware,consistent_hash}_...1517*Z` | 9B post-fix three arms |
| `wa_codex_3n_cache_aware_...142909Z` | 9B ca debug arm (1,322 decision-log lines) |
| `wa_codex_3n_consistent_hash_...151650Z` | 9B ch debug arm (16 keys = 16 sessions) |
| `wa_codex_3n_none_20260816T015414Z` | num_workers causal run (13201) |

Datasets: `codex_replay_p30_all` (120B, 610 sessions),
`codex_replay_p30` / `codex_replay_p30_9b` (9B), rebuildable from the source
pool with `scripts/build_codex_replay_dataset.py`.

Scripts (`scripts/`): `grpo_workplace_assistant.sh` (launcher),
`routing_postcheck.sh` (acceptance), `watch_router.sh` (timeline),
`build_codex_replay_dataset.py` (dataset builder).
