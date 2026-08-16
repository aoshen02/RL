#!/usr/bin/env python
"""把一次 e2e 运行的**完整交互轨迹**导出成可读文本 + 结构化 JSON。

为什么读 train_data_step*.jsonl 而不是 OpenHands 的 output.jsonl:
后者在这条路径上恒为 0 字节(已实测多次运行),而前者是 NeMo-RL 在
``env.should_log_nemo_gym_responses=false`` 时写出的、**训练真正看到的**那份数据。
代价是它只有 token_ids,``content`` 字段是空串列表,所以必须用 tokenizer 解回来。

``token_loss_mask`` 给出每个 token 是否参与 loss:1 = 模型生成的 token,
0 = prompt / 工具返回 / 环境观测。按它切段就能还原"谁说了什么"这一层,
这比只看聚合指标(token 数、latency)强得多 —— 后者看不出模型到底干了什么。

用法:
    dump_trajectory_qwen3_swe2.py <run_dir> [--model PATH]
输出:
    <run_dir>/trajectory_step<N>.txt    人读
    <run_dir>/trajectory_step<N>.json   机读(含每段的 token 区间与 mask)
"""

import argparse
import json
import pathlib
import sys

DEFAULT_MODEL = "/mnt/lustre01/users/inf-aoshen/models/Qwen3-30B-A3B-Thinking-2507"


def segments(token_ids: list[int], loss_mask: list[int]):
    """按 loss mask 的 0/1 边界切段,返回 (is_generated, start, end, ids)。"""
    if not token_ids:
        return
    # mask 可能比 token_ids 短(padding 对齐差异),缺失部分按 0 处理。
    mask = list(loss_mask) + [0] * max(0, len(token_ids) - len(loss_mask))
    start, cur = 0, mask[0]
    for i in range(1, len(token_ids)):
        if mask[i] != cur:
            yield bool(cur), start, i, token_ids[start:i]
            start, cur = i, mask[i]
    yield bool(cur), start, len(token_ids), token_ids[start:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    files = sorted(run_dir.glob("nemo-rl-logs/*/train_data_step*.jsonl"))
    if not files:
        print(f"no train_data_step*.jsonl under {run_dir}/nemo-rl-logs", file=sys.stderr)
        print("（该文件仅在 env.should_log_nemo_gym_responses=false 时写出）", file=sys.stderr)
        return 1

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    for f in files:
        step = f.stem.replace("train_data_step", "")
        rows = [json.loads(line) for line in f.open() if line.strip()]
        out_txt = run_dir / f"trajectory_step{step}.txt"
        out_json = run_dir / f"trajectory_step{step}.json"
        dumped = []

        with out_txt.open("w") as w:
            w.write(f"# trajectory step {step}  ({f})\n")
            w.write(f"# samples={len(rows)}  model={args.model}\n\n")
            for si, row in enumerate(rows):
                ids_all = row.get("token_ids") or []
                masks_all = row.get("token_loss_mask") or []
                rewards = row.get("rewards") or []
                agent = row.get("agent_ref") or []
                # token_ids 是 list[list[int]](每个 sample 一条序列)
                if ids_all and isinstance(ids_all[0], int):
                    ids_all, masks_all = [ids_all], [masks_all]
                for ti, ids in enumerate(ids_all):
                    mask = masks_all[ti] if ti < len(masks_all) else []
                    ids = [t for t in ids if t is not None]
                    reward = rewards[ti] if ti < len(rewards) else (rewards[0] if rewards else None)
                    w.write("=" * 78 + "\n")
                    w.write(f"sample {si}.{ti}  tokens={len(ids)}  reward={reward}  agent={agent}\n")
                    w.write("=" * 78 + "\n")
                    segs = []
                    for is_gen, s, e, chunk in segments(ids, mask):
                        text = tok.decode(chunk, skip_special_tokens=False)
                        if not text.strip():
                            continue
                        who = "MODEL (loss=1)" if is_gen else "PROMPT/ENV (loss=0)"
                        w.write(f"\n----- {who}  tokens[{s}:{e}] -----\n{text}\n")
                        segs.append(
                            {"generated": is_gen, "start": s, "end": e, "text": text}
                        )
                    dumped.append(
                        {
                            "sample": f"{si}.{ti}",
                            "num_tokens": len(ids),
                            "reward": reward,
                            "agent_ref": agent,
                            "num_generated_tokens": sum(1 for m in mask if m),
                            "segments": segs,
                        }
                    )

        out_json.write_text(json.dumps(dumped, ensure_ascii=False, indent=2))
        gen = sum(d["num_generated_tokens"] for d in dumped)
        tot = sum(d["num_tokens"] for d in dumped)
        print(f"step {step}: samples={len(dumped)} tokens={tot} generated={gen}")
        print(f"  {out_txt}")
        print(f"  {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
