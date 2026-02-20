#!/usr/bin/env python3
"""Analysis script for H14: Pathological Templates hypothesis experiment.

Parses BLIS output files and produces formatted comparison tables
showing anomaly counters, TTFT, and distribution uniformity.

Usage:
    python3 analyze.py core normal_42.txt patho_42.txt ...
    python3 analyze.py decomposed normal.txt routing_only.txt sched_only.txt patho.txt
"""

import json
import math
import re
import sys
from pathlib import Path


def parse_output(filepath):
    """Parse BLIS output into metrics dict."""
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

    # Distribution standard deviation (pad to 4 instances — instances with
    # 0 requests don't appear in trace summary but must count for stddev)
    num_instances = 4
    counts = [0] * num_instances
    for i, k in enumerate(sorted(dist.keys())):
        if i < num_instances:
            counts[i] = dist[k]
    mean_d = sum(counts) / len(counts)
    stddev = math.sqrt(sum((x - mean_d) ** 2 for x in counts) / len(counts))

    # Anomaly counters (only present when > 0)
    hol = 0
    hol_match = re.search(r"HOL Blocking Events: (\d+)", content)
    if hol_match:
        hol = int(hol_match.group(1))

    inversions = 0
    inv_match = re.search(r"Priority Inversions: (\d+)", content)
    if inv_match:
        inversions = int(inv_match.group(1))

    rejected = 0
    rej_match = re.search(r"Rejected Requests: (\d+)", content)
    if rej_match:
        rejected = int(rej_match.group(1))

    return {
        "ttft_mean": cluster["ttft_mean_ms"] if cluster else 0,
        "ttft_p99": cluster["ttft_p99_ms"] if cluster else 0,
        "e2e_mean": cluster["e2e_mean_ms"] if cluster else 0,
        "e2e_p99": cluster["e2e_p99_ms"] if cluster else 0,
        "throughput": cluster["responses_per_sec"] if cluster else 0,
        "dist": dist,
        "stddev": stddev,
        "hol": hol,
        "inversions": inversions,
        "rejected": rejected,
    }


def dist_str(dist):
    """Format distribution as compact list."""
    return str([dist[k] for k in sorted(dist.keys())])


def analyze_core(files):
    """Experiment 1: Normal vs Pathological across seeds."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    seeds = sorted({name.split("_")[-1] for name in results})

    print("  Normal vs Pathological (per seed):")
    print()
    hdr = (
        f"  {'Seed':>4} {'Config':<12} | {'TTFT Mean':>10} {'TTFT P99':>10}"
        f" | {'HOL':>4} {'Inv':>4} | {'StdDev':>8} | Distribution"
    )
    print(hdr)
    print(
        f"  {'-' * 4} {'-' * 12}-+-{'-' * 21}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 30}"
    )

    for seed in seeds:
        for config in ["normal", "patho"]:
            key = f"{config}_{seed}"
            r = results.get(key)
            if not r:
                continue
            label = "normal" if config == "normal" else "pathological"
            print(
                f"  {seed:>4} {label:<12} |"
                f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
                f" | {r['hol']:>4} {r['inversions']:>4}"
                f" | {r['stddev']:>8.1f} | {dist_str(r['dist'])}"
            )
        # Print ratio
        n = results.get(f"normal_{seed}")
        p = results.get(f"patho_{seed}")
        if n and p and n["ttft_p99"] > 0:
            ratio = p["ttft_p99"] / n["ttft_p99"]
            print(
                f"       {'':12} |"
                f" {'Effect:':>10} {ratio:>9.1f}x worse P99"
            )
        print()


def analyze_decomposed(files):
    """Experiment 2: Routing-only vs Scheduling-only (seed 42)."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    configs = [
        ("normal", "Normal (all correct)"),
        ("routing_only", "Pathological routing only"),
        ("sched_only", "Pathological scheduling only"),
        ("patho", "All pathological"),
    ]

    print(
        f"  {'Configuration':<30} | {'TTFT P99':>10}"
        f" | {'HOL':>4} {'Inv':>4} | {'StdDev':>8} | Distribution"
    )
    print(
        f"  {'-' * 30}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 30}"
    )

    for key, label in configs:
        r = results.get(key)
        if not r:
            continue
        print(
            f"  {label:<30} |"
            f" {r['ttft_p99']:>10.1f}"
            f" | {r['hol']:>4} {r['inversions']:>4}"
            f" | {r['stddev']:>8.1f} | {dist_str(r['dist'])}"
        )


def analyze_scheduling(files):
    """Experiment 3: Scheduling effect at 1 instance (ED-2 rate awareness)."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    configs = [
        ("sched_1inst_normal", "Normal (slo-based + priority-fcfs)"),
        ("sched_1inst_double", "Double inversion (inverted + reverse)"),
        ("sched_1inst_inv_prio", "Inverted priority only (inverted + pfcfs)"),
        ("sched_1inst_rev_sched", "Reverse scheduler only (slo + reverse)"),
    ]

    print(
        f"  {'Configuration':<35} | {'TTFT Mean':>10} {'TTFT P99':>10}"
        f" | {'E2E Mean':>10} {'E2E P99':>10} | {'Inv':>5}"
    )
    print(
        f"  {'-' * 35}-+-{'-' * 21}-+-{'-' * 21}-+-{'-' * 5}"
    )

    for key, label in configs:
        r = results.get(key)
        if not r:
            continue
        print(
            f"  {label:<35} |"
            f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
            f" | {r['e2e_mean']:>10.1f} {r['e2e_p99']:>10.1f}"
            f" | {r['inversions']:>5}"
        )

    n = results.get("sched_1inst_normal")
    for key, label in configs[1:]:
        p = results.get(key)
        if n and p and n["e2e_p99"] > 0:
            e2e_ratio = p["e2e_p99"] / n["e2e_p99"]
            print(f"  vs normal: {label} is {e2e_ratio:.2f}x on E2E P99")


ANALYZERS = {
    "core": analyze_core,
    "decomposed": analyze_decomposed,
    "scheduling": analyze_scheduling,
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
