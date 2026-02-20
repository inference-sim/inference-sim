#!/usr/bin/env python3
"""Analysis script for Prefix-Affinity hypothesis experiment.

Parses BLIS output files and produces formatted comparison tables.
Called by run.sh with experiment type and output file paths.

Usage:
    python3 analyze.py multi-turn exp1_pa.txt exp1_qd.txt exp1_rr.txt ...
    python3 analyze.py shared-prompt exp3_pa5.txt exp3_qd.txt exp3_rr.txt ...
"""

import json
import math
import re
import sys
from pathlib import Path


LABELS = {
    "exp1_pa": "prefix-affinity:3,qd:2",
    "exp1_qd": "queue-depth:1",
    "exp1_rr": "round-robin",
    "exp1_llmd": "llm-d default",
    "exp2_pa": "prefix-affinity:3,qd:2",
    "exp2_qd": "queue-depth:1",
    "exp2_rr": "round-robin",
    "exp3_pa5": "pa:5,qd:1 (concentrated)",
    "exp3_pa1": "pa:1,qd:1 (balanced)",
    "exp3_qd": "queue-depth:1",
    "exp3_rr": "round-robin",
}


def parse_output(filepath):
    """Parse multi-block BLIS output into cluster metrics + distribution."""
    content = Path(filepath).read_text()

    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    dist = {}
    dist_match = re.search(
        r"Target Distribution:\n((?:\s+instance_\d+: \d+\n?)+)", content
    )
    if dist_match:
        for line in dist_match.group(1).strip().split("\n"):
            parts = line.strip().split(": ")
            dist[parts[0]] = int(parts[1])

    counts = [dist[k] for k in sorted(dist.keys())] if dist else [0]
    mean_d = sum(counts) / len(counts)
    stddev = math.sqrt(sum((x - mean_d) ** 2 for x in counts) / len(counts))

    hit_rate = 0.0
    hr_match = re.search(r"Cache Hit Rate:\s*([\d.]+)", content)
    if hr_match:
        hit_rate = float(hr_match.group(1))

    return {
        "ttft_mean": cluster["ttft_mean_ms"] if cluster else 0,
        "ttft_p99": cluster["ttft_p99_ms"] if cluster else 0,
        "throughput": cluster["responses_per_sec"] if cluster else 0,
        "dist": dist,
        "stddev": stddev,
        "hit_rate": hit_rate,
    }


def dist_str(dist):
    return str([dist[k] for k in sorted(dist.keys())])


def print_table(results, order):
    """Print a comparison table from results dict."""
    hdr = (
        f"  {'Configuration':<28} |"
        f" {'TTFT Mean':>10} {'TTFT P99':>10} {'Tput':>8}"
        f" | {'CacheHit':>8} | Distribution"
    )
    print(hdr)
    print(
        f"  {'-' * 28}-+-{'-' * 30}-+-{'-' * 8}-+-{'-' * 30}"
    )
    for key in order:
        r = results.get(key)
        if not r:
            continue
        label = LABELS.get(key, key)
        print(
            f"  {label:<28} |"
            f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
            f" {r['throughput']:>8.1f}"
            f" | {r['hit_rate']:>7.1%} | {dist_str(r['dist'])}"
        )


def analyze_multi_turn(files):
    """Experiments 1 & 2: Multi-turn chat."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    # Detect prefix: exp1_ or exp2_
    prefixes = sorted({n.rsplit("_", 1)[0] for n in results})
    for prefix in prefixes:
        keys = sorted(k for k in results if k.startswith(prefix))
        print_table(results, keys)

        # Effect summary
        pa_key = f"{prefix}_pa"
        qd_key = f"{prefix}_qd"
        if pa_key in results and qd_key in results:
            pa = results[pa_key]
            qd = results[qd_key]
            ratio = qd["ttft_mean"] / pa["ttft_mean"] if pa["ttft_mean"] > 0 else 0
            print(
                f"\n  prefix-affinity is {ratio:.1f}x better TTFT mean than queue-depth"
                f" ({pa['ttft_mean']:.1f} vs {qd['ttft_mean']:.1f} ms)"
            )
            print(
                f"  Cache hit: pa={pa['hit_rate']:.1%} vs qd={qd['hit_rate']:.1%}"
                f" ({pa['hit_rate']/qd['hit_rate']:.1f}x more reuse)"
                if qd["hit_rate"] > 0
                else ""
            )


def analyze_shared_prompt(files):
    """Experiment 3: Shared system prompt."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    order = ["exp3_pa5", "exp3_pa1", "exp3_qd", "exp3_rr"]
    print_table(results, order)


ANALYZERS = {
    "multi-turn": analyze_multi_turn,
    "shared-prompt": analyze_shared_prompt,
}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <experiment-type> <files...>")
        sys.exit(1)
    experiment_type = sys.argv[1]
    files = sys.argv[2:]
    analyzer = ANALYZERS.get(experiment_type)
    if not analyzer:
        print(f"Unknown type: {experiment_type}. Valid: {list(ANALYZERS.keys())}")
        sys.exit(1)
    analyzer(files)
