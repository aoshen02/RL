#!/usr/bin/env python
"""从 SWE-bench Verified parquet 生成一行 NeMo-Gym swe_agents 数据集。

schema 是逆向 django15037_swe2.jsonl 得来的(见下),不是猜的:
    {"responses_create_params": {"input": [], "metadata": {...}}, "agent_ref": {...}}
metadata 里 instance_dict 是**整行 instance 的 JSON 字符串**(含 patch/test_patch/
environment_setup_commit),SWE 评测容器靠它拿 FAIL_TO_PASS 与 test_patch;顶层那几个
字段是 OpenHands 侧读的。两处都要有,少一处会在不同阶段各自炸。

用法:
    make_swe2_dataset_row.py <instance_id> <out.jsonl> [--parquet PATH]
"""

import argparse
import json
import pathlib
import sys

PARQUET = (
    "/mnt/lustre01/users/inf-aoshen/hf_cache/hub/"
    "datasets--princeton-nlp--SWE-bench_Verified/snapshots/"
    "c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data/test-00000-of-00001.parquet"
)
# instance_dict 必须完整覆盖这些键,否则 swebench harness 侧会 KeyError。
INSTANCE_KEYS = [
    "repo", "instance_id", "base_commit", "patch", "test_patch",
    "problem_statement", "hints_text", "created_at", "version",
    "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit",
]
TOP_KEYS = [
    "instance_id", "base_commit", "repo", "version", "problem_statement",
    "hints_text", "created_at", "FAIL_TO_PASS", "PASS_TO_PASS",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_id")
    ap.add_argument("out")
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--agent", default="swe_agents_train")
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.parquet)
    hit = df[df.instance_id == args.instance_id]
    if hit.empty:
        print(f"instance not in dataset: {args.instance_id}", file=sys.stderr)
        return 1
    row = hit.iloc[0]

    missing = [k for k in INSTANCE_KEYS if k not in row.index]
    if missing:
        print(f"parquet lacks keys: {missing}", file=sys.stderr)
        return 1

    inst = {k: str(row[k]) for k in INSTANCE_KEYS}
    meta = {k: str(row[k]) for k in TOP_KEYS}
    meta["dataset_name"] = "princeton-nlp/SWE-bench_Verified"
    meta["split"] = "test"
    meta["instance_dict"] = json.dumps(inst)

    out = {
        "responses_create_params": {"input": [], "metadata": meta},
        "agent_ref": {"type": "responses_api_agents", "name": args.agent},
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"wrote {p} ({p.stat().st_size} bytes)")
    print(f"  instance_id      = {inst['instance_id']}")
    print(f"  repo/version     = {inst['repo']} @ {inst['version']}")
    print(f"  base_commit      = {inst['base_commit']}")
    print(f"  FAIL_TO_PASS     = {inst['FAIL_TO_PASS']}")
    print(f"  PASS_TO_PASS n   = {len(json.loads(inst['PASS_TO_PASS']))}")
    print(f"  gold patch lines = {len(inst['patch'].splitlines())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
