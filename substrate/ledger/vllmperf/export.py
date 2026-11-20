#!/usr/bin/env python3
"""Export the vLLM CI perf rows and the perf-eval recipes that ledger/vllmperf reads.

    export.py --api perf.json --cutoff YYYY-MM-DD --perf-eval DIR --out DIR

perf.json is the body of https://ci.vllm.ai/api/perf saved once ({"rows": [...]}); DIR of
--perf-eval is a checkout of vllm-project/perf-eval at the commit sources.yaml pins. Writes
rows.jsonl (the rows dated up to the cutoff day, one canonical JSON object per line, sorted)
and recipes.jsonl (one line per benchmark and concurrency of each workload recipe, sorted). Neither file is
committed: the ledger commits only identifiers and hashes of them.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import yaml


def dump(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--perf-eval", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.api) as f:
        rows = json.load(f)["rows"]
    keep = sorted(dump(r) for r in rows if str(r["date"])[:10] <= a.cutoff)

    recipes = []
    for path in sorted(glob.glob(os.path.join(a.perf_eval, "workloads", "*.yaml"))):
        with open(path) as f:
            w = yaml.safe_load(f)
        vllm = w.get("vllm") or {}
        for c in (w.get("vllm_bench") or {}).get("configs") or []:
            conc = c["max_concurrency"]
            for n in conc if isinstance(conc, list) else [conc]:
                recipes.append(dump({
                    "workload": w["name"],
                    "gpu": w["gpu"],
                    "model": vllm["model"],
                    "serve_args": " ".join(str(vllm.get("serve_args", "")).split()),
                    "benchmark": c["name"],
                    "input_len": c["input_len"],
                    "output_len": c["output_len"],
                    "max_concurrency": n,
                }))
    recipes.sort()

    os.makedirs(a.out, exist_ok=True)
    for name, lines in (("rows.jsonl", keep), ("recipes.jsonl", recipes)):
        data = ("\n".join(lines) + "\n").encode()
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)
        print(name, len(lines), hashlib.sha256(data).hexdigest(), file=sys.stderr)


if __name__ == "__main__":
    main()
