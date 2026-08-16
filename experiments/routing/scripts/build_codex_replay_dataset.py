#!/usr/bin/env python3
"""Convert codex_swebenchpro trajectories into a Gym replay dataset (one line = one session).

The source data is a ShareGPT-format recording, strictly alternating human/gpt:

    msg0 human  system prompt + permission statement + skills + task (~12K tokens)
    msg1 gpt    ← placeholder (Lorem ipsum); only its length is real
    msg2 human  real output of Command: bash -lc "..."
    ...

Replay executes no commands: msg2/msg4/... already are the tools' real return values
from the original run. Context accumulates naturally through codex_replay_agent's
multi-turn loop, so only the raw messages are stored here — no O(n^2) cumulative
materialization (that would be 1.3 GB for 256 sessions).

Usage:
    python build_codex_replay_dataset.py --out-dir agent_run/data/codex_replay \
        --num-sessions 256 --max-turns 40
"""

import argparse
import json
import random
from pathlib import Path

SRC = "/home/inf-aoshen/crusoe-inference/datasets/hf_codex_swebenchpro/codex_swebenchpro.json"
AGENT = {"type": "responses_api_agents", "name": "codex_replay_agent"}
# Strata boundaries by turn count. Pure random sampling of a few dozen out of 610 would
# most likely draw zero of the 78+ turn extra-long sessions, and the long tail is exactly
# what the routing policies are tested on. Stratification no longer reproduces the
# production turn distribution; in exchange every length band is present and policy
# differences are not averaged away.
STRATA = [(6, 15), (16, 30), (31, 50), (51, 75), (76, 10**9)]


def build_row(idx: int, conversations: list[dict], max_turns: int) -> dict | None:
    if not conversations or conversations[0]["from"] != "human":
        return None
    humans = [m["value"] for m in conversations if m["from"] == "human"]
    gpts = [m["value"] for m in conversations if m["from"] == "gpt"]
    turns = [
        {"assistant": g, "user": h} for g, h in zip(gpts, humans[1:])
    ][:max_turns]
    if not turns:
        return None
    return {
        "agent_ref": AGENT,
        "responses_create_params": {
            "input": [{"role": "user", "content": humans[0]}]
        },
        "replay_turns": turns,
        "trace_session_id": f"codex-{idx}",
    }


def stratify(
    rows: list[dict], total: int, rng: random.Random, quotas: list[int] | None = None
) -> list[dict]:
    buckets = [[r for r in rows if lo <= len(r["replay_turns"]) <= hi] for lo, hi in STRATA]
    qs = quotas or [total // len(STRATA)] * len(STRATA)
    out = []
    for b, q in zip(buckets, qs):
        rng.shuffle(b)
        out += b[:q]
    # When a stratum runs short, top up from the leftovers: better a slightly skewed
    # distribution than fewer rows than requested
    leftover = [r for b, q in zip(buckets, qs) for r in b[q:]]
    rng.shuffle(leftover)
    return out + leftover[: total - len(out)]


def clip_to_budget(rows: list[dict], model: str, budget: int) -> int:
    """Accumulate real tokens turn by turn, truncate at the turn that exceeds the
    budget, and return the number of clipped sessions.

    The chars/4 estimate undercounts codex-style code + shell output text by over 20%:
    a session measured at 288K was estimated at 236K. And once vLLM returns 400, the
    agent gets a response with no assistant content and
    nemo_rl/environments/nemo_gym.py:520 raises immediately — one over-limit request
    kills the whole run.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    n = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    clipped = 0
    for r in rows:
        ctx = n(r["responses_create_params"]["input"][0]["content"])
        for k, t in enumerate(r["replay_turns"]):
            if ctx > budget:
                r["replay_turns"] = r["replay_turns"][:k]
                clipped += 1
                break
            ctx += n(t["assistant"]) + n(t["user"])
    return clipped


def report(path: Path, rows: list[dict]) -> None:
    n = len(rows)
    turns = sorted(len(r["replay_turns"]) for r in rows)
    # Context carried by the last-turn request = first message + all replay turns;
    # chars / 4 approximates token count
    ctx = sorted(
        len(r["responses_create_params"]["input"][0]["content"])
        + sum(len(t["assistant"]) + len(t["user"]) for t in r["replay_turns"])
        for r in rows
    )
    hist = "  ".join(
        f"{lo}-{hi if hi < 10**9 else ''}:{sum(1 for t in turns if lo <= t <= hi)}"
        for lo, hi in STRATA
    )
    print(
        f"{path}  sessions={n}  size={path.stat().st_size / 2**20:.0f}MiB\n"
        f"  turn strata    {hist}\n"
        f"  turns          p50 {turns[n // 2]:5d}  p90 {turns[int(n * 0.9)]:5d}  max {turns[-1]:5d}\n"
        f"  last-turn ctx  p50 {ctx[n // 2] // 4:7d}  p90 {ctx[int(n * 0.9)] // 4:7d}  "
        f"max {ctx[-1] // 4:7d}  (estimated tokens)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-sessions", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratify", action="store_true", help="stratified sampling by turn count")
    ap.add_argument("--quotas", help="per-stratum quotas (comma-separated, matching STRATA); overrides uniform quotas and --num-sessions when given")
    ap.add_argument("--model", help="when a model path is given, clip over-budget sessions with the real tokenizer")
    ap.add_argument("--token-budget", type=int, default=250000)
    args = ap.parse_args()

    traces = json.load(open(args.src))
    rows = [
        row
        for i, t in enumerate(traces)
        if (row := build_row(i, t["conversations"], args.max_turns)) is not None
    ]
    rng = random.Random(args.seed)
    quotas = [int(x) for x in args.quotas.split(",")] if args.quotas else None
    if quotas:
        args.num_sessions = sum(quotas)
    if args.stratify:
        rows = stratify(rows, args.num_sessions, rng, quotas)
    else:
        rng.shuffle(rows)
        rows = rows[: args.num_sessions]

    if args.model:
        clipped = clip_to_budget(rows, args.model, args.token_budget)
        print(f"clipped {clipped} / {len(rows)} sessions to the {args.token_budget:,} token budget")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # After stratification rows are ordered by stratum; strided slicing lets validation
    # sample every stratum
    for split, subset in (("train", rows), ("validation", rows[:: max(1, len(rows) // 8)][:8])):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")
        report(path, subset)


if __name__ == "__main__":
    main()
