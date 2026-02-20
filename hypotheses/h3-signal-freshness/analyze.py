#!/usr/bin/env python3
"""Analysis script for H3: Signal Freshness hypothesis experiment.

Parses BLIS multi-block output files and produces formatted comparison tables.
Called by run.sh with experiment type and output file paths.

Usage:
    python3 analyze.py core exp1_qd_42.txt exp1_kv_42.txt ...
    python3 analyze.py rate-scaling exp2_qd_r100.txt exp2_kv_r100.txt ...
    python3 analyze.py refresh-interval exp3_i0.txt exp3_i500.txt ...
    python3 analyze.py combined exp4_qd.txt exp4_kv.txt ...
"""

import json
import math
import re
import sys
from pathlib import Path


def parse_output(filepath):
    """Parse multi-block BLIS output into cluster metrics + distribution."""
    content = Path(filepath).read_text()

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    # Extract target distribution from trace summary
    dist = {}
    dist_match = re.search(
        r"Target Distribution:\n((?:\s+instance_\d+: \d+\n?)+)", content
    )
    if dist_match:
        for line in dist_match.group(1).strip().split("\n"):
            parts = line.strip().split(": ")
            dist[parts[0]] = int(parts[1])

    # Compute distribution standard deviation
    counts = [dist[k] for k in sorted(dist.keys())] if dist else [0]
    mean_d = sum(counts) / len(counts)
    stddev = math.sqrt(sum((x - mean_d) ** 2 for x in counts) / len(counts))

    # Extract HOL blocking count
    hol = 0
    hol_match = re.search(r"HOL Blocking Events: (\d+)", content)
    if hol_match:
        hol = int(hol_match.group(1))

    return {
        "ttft_mean": cluster["ttft_mean_ms"] if cluster else 0,
        "ttft_p99": cluster["ttft_p99_ms"] if cluster else 0,
        "e2e_p99": cluster["e2e_p99_ms"] if cluster else 0,
        "throughput": cluster["responses_per_sec"] if cluster else 0,
        "dist": dist,
        "stddev": stddev,
        "hol": hol,
    }


def dist_str(dist):
    """Format distribution as compact list."""
    return str([dist[k] for k in sorted(dist.keys())])


def analyze_core(files):
    """Experiment 1: Core hypothesis across seeds."""
    results = {}
    for f in files:
        name = Path(f).stem  # exp1_qd_42, exp1_kv_123, etc.
        results[name] = parse_output(f)

    seeds = sorted({name.split("_")[2] for name in results})
    for seed in seeds:
        qd = results.get(f"exp1_qd_{seed}")
        kv = results.get(f"exp1_kv_{seed}")
        if not qd or not kv:
            continue

        ttft_ratio = kv["ttft_mean"] / qd["ttft_mean"]
        p99_ratio = kv["ttft_p99"] / qd["ttft_p99"]
        std_ratio = kv["stddev"] / qd["stddev"] if qd["stddev"] > 0 else float("inf")

        print(f"  Seed {seed}:")
        print(
            f"    queue-depth:     TTFT mean={qd['ttft_mean']:7.1f}ms"
            f"  p99={qd['ttft_p99']:7.1f}ms"
            f"  stddev={qd['stddev']:5.1f}"
            f"  HOL={qd['hol']}"
            f"  dist={dist_str(qd['dist'])}"
        )
        print(
            f"    kv-utilization:  TTFT mean={kv['ttft_mean']:7.1f}ms"
            f"  p99={kv['ttft_p99']:7.1f}ms"
            f"  stddev={kv['stddev']:5.1f}"
            f"  HOL={kv['hol']}"
            f"  dist={dist_str(kv['dist'])}"
        )
        print(
            f"    Effect: KV is {ttft_ratio:.1f}x worse TTFT mean,"
            f" {p99_ratio:.1f}x worse P99,"
            f" {std_ratio:.0f}x worse distribution"
        )
        print()


def analyze_rate_scaling(files):
    """Experiment 2: Rate scaling."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    rates = sorted(
        {name.split("_r")[1] for name in results if "_r" in name}, key=int
    )

    hdr = (
        f"  {'Rate':>5} | {'--- queue-depth ---':^35}"
        f" | {'--- kv-utilization ---':^35} | KV/QD"
    )
    print(hdr)
    print(
        f"  {'':>5} | {'TTFT Mean':>10} {'P99':>8} {'StdDev':>7}"
        f" | {'TTFT Mean':>10} {'P99':>8} {'StdDev':>7} | {'Ratio':>5}"
    )
    print(f"  {'-'*5}-+-{'-'*35}-+-{'-'*35}-+-{'-'*5}")

    for rate in rates:
        qd = results.get(f"exp2_qd_r{rate}")
        kv = results.get(f"exp2_kv_r{rate}")
        if not qd or not kv:
            continue
        ratio = kv["ttft_mean"] / qd["ttft_mean"] if qd["ttft_mean"] > 0 else 0
        print(
            f"  {rate:>5} |"
            f" {qd['ttft_mean']:>10.1f} {qd['ttft_p99']:>8.1f} {qd['stddev']:>7.1f}"
            f" | {kv['ttft_mean']:>10.1f} {kv['ttft_p99']:>8.1f} {kv['stddev']:>7.1f}"
            f" | {ratio:>5.2f}x"
        )


def analyze_refresh_interval(files):
    """Experiment 3: Snapshot refresh interval compounding."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    intervals = sorted(
        {int(name.split("_i")[1]) for name in results if "_i" in name}
    )

    print(
        f"  {'Interval':>12} | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'StdDev':>8} | Distribution"
    )
    print(f"  {'-'*12}-+-{'-'*21}-+-{'-'*8}-+-{'-'*30}")

    for interval in intervals:
        r = results.get(f"exp3_i{interval}")
        if not r:
            continue
        label = "immediate" if interval == 0 else f"{interval}us ({interval/1000:.1f}ms)"
        print(
            f"  {label:>12} |"
            f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
            f" | {r['stddev']:>8.1f} | {dist_str(r['dist'])}"
        )


def analyze_combined(files):
    """Experiment 4: Combined scorer configurations."""
    # Map filenames to labels
    labels = {
        "exp4_qd": "queue-depth:1 only",
        "exp4_kv": "kv-utilization:1 only",
        "exp4_equal": "kv:2,qd:2 (equal weight)",
        "exp4_kv_dom": "kv:5,qd:1 (KV dominant)",
        "exp4_qd_dom": "qd:5,kv:1 (QD dominant)",
        "exp4_llmd": "llm-d default (pa:3,qd:2,kv:2)",
    }
    order = ["exp4_qd", "exp4_kv", "exp4_equal", "exp4_kv_dom", "exp4_qd_dom", "exp4_llmd"]

    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    print(
        f"  {'Configuration':<38} | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'StdDev':>8} | Distribution"
    )
    print(f"  {'-'*38}-+-{'-'*21}-+-{'-'*8}-+-{'-'*30}")

    for key in order:
        r = results.get(key)
        if not r:
            continue
        label = labels.get(key, key)
        print(
            f"  {label:<38} |"
            f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
            f" | {r['stddev']:>8.1f} | {dist_str(r['dist'])}"
        )


ANALYZERS = {
    "core": analyze_core,
    "rate-scaling": analyze_rate_scaling,
    "refresh-interval": analyze_refresh_interval,
    "combined": analyze_combined,
}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <experiment-type> <files...>")
        print(f"Types: {', '.join(ANALYZERS.keys())}")
        sys.exit(1)

    experiment_type = sys.argv[1]
    files = sys.argv[2:]

    analyzer = ANALYZERS.get(experiment_type)
    if not analyzer:
        print(f"Unknown experiment type: {experiment_type}")
        print(f"Valid types: {', '.join(ANALYZERS.keys())}")
        sys.exit(1)

    analyzer(files)
