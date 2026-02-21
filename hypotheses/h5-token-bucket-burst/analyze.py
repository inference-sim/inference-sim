#!/usr/bin/env python3
"""Analysis script for H5: Token-Bucket Admission Control Under Burst.

Parses BLIS multi-block output and produces comparison tables for:
  - Core: always-admit vs token-bucket across 3 seeds
  - Rate-scaling: burst effect at different aggregate rates
  - Tuning: token-bucket capacity/refill sensitivity

Usage:
    python3 analyze.py core exp1_always_42.txt exp1_bucket_42.txt ...
    python3 analyze.py rate-scaling exp2_always_r200.txt exp2_bucket_r200.txt ...
    python3 analyze.py tuning exp3_c100_r100.txt exp3_c500_r400.txt ...
"""

import json
import math
import re
import sys
from pathlib import Path


def parse_output(filepath):
    """Parse multi-block BLIS output into cluster metrics."""
    content = Path(filepath).read_text()

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    # Extract rejected count
    rejected = 0
    rejected_match = re.search(r"Rejected Requests: (\d+)", content)
    if rejected_match:
        rejected = int(rejected_match.group(1))

    # Extract target distribution
    dist = {}
    dist_match = re.search(
        r"Target Distribution:\n((?:\s+instance_\d+: \d+\n?)+)", content
    )
    if dist_match:
        for line in dist_match.group(1).strip().split("\n"):
            parts = line.strip().split(": ")
            dist[parts[0]] = int(parts[1])

    # Distribution stddev
    counts = [dist[k] for k in sorted(dist.keys())] if dist else [0]
    mean_d = sum(counts) / len(counts) if counts else 0
    stddev = (
        math.sqrt(sum((x - mean_d) ** 2 for x in counts) / len(counts))
        if counts
        else 0
    )

    # Preemption count
    preemptions = 0
    preempt_match = re.search(r"Preemptions?: (\d+)", content)
    if preempt_match:
        preemptions = int(preempt_match.group(1))

    return {
        "ttft_mean": cluster["ttft_mean_ms"] if cluster else 0,
        "ttft_p99": cluster["ttft_p99_ms"] if cluster else 0,
        "e2e_mean": cluster["e2e_mean_ms"] if cluster else 0,
        "e2e_p99": cluster["e2e_p99_ms"] if cluster else 0,
        "throughput": cluster["responses_per_sec"] if cluster else 0,
        "completed": cluster["completed_requests"] if cluster else 0,
        "injected": cluster["injected_requests"] if cluster else 0,
        "rejected": rejected,
        "preemptions": preemptions,
        "stddev": stddev,
        "dist": dist,
    }


def analyze_core(files):
    """Experiment 1: Core hypothesis across seeds."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    seeds = sorted(
        {name.split("_")[2] for name in results if "exp1_" in name}
    )

    print(
        f"  {'Seed':<6} {'Policy':<16}"
        f" | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'E2E Mean':>10} {'P99':>10}"
        f" | {'Thru':>6} {'Rej':>5} {'Comp':>5}"
    )
    print(
        f"  {'-'*6} {'-'*16}"
        f"-+-{'-'*10} {'-'*10}"
        f"-+-{'-'*10} {'-'*10}"
        f"-+-{'-'*6} {'-'*5} {'-'*5}"
    )

    ratios = []
    for seed in seeds:
        aa = results.get(f"exp1_always_{seed}")
        tb = results.get(f"exp1_bucket_{seed}")
        if not aa or not tb:
            continue

        for label, r in [("always-admit", aa), ("token-bucket", tb)]:
            print(
                f"  {seed:<6} {label:<16}"
                f" | {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
                f" | {r['e2e_mean']:>10.1f} {r['e2e_p99']:>10.1f}"
                f" | {r['throughput']:>6.1f} {r['rejected']:>5} {r['completed']:>5}"
            )

        if tb["ttft_p99"] > 0:
            ratio = aa["ttft_p99"] / tb["ttft_p99"]
            ratios.append(ratio)
            print(
                f"  {'':>6} {'Effect':>16}"
                f" | token-bucket TTFT p99 is"
                f" {ratio:.2f}x {'better' if ratio > 1 else 'worse'}"
                f" (rejected {tb['rejected']} requests)"
            )
        print()

    # Summary
    if ratios:
        all_better = all(r > 1.0 for r in ratios)
        significant = all(r > 1.2 for r in ratios)
        min_r = min(ratios)
        max_r = max(ratios)

        print(f"  Summary: TTFT p99 ratio (always-admit/token-bucket):")
        print(f"    min={min_r:.2f}x  max={max_r:.2f}x  all_seeds_better={all_better}")
        if significant:
            print(
                f"    CONFIRMED: Token-bucket consistently reduces tail latency"
                f" (>{(min_r-1)*100:.0f}% improvement across all seeds)"
            )
        elif all_better:
            print(
                f"    INCONCLUSIVE: Token-bucket is better but effect <20%"
                f" in some seeds"
            )
        else:
            print(
                f"    REFUTED: Token-bucket is not consistently better"
            )


def analyze_rate_scaling(files):
    """Experiment 2: Rate scaling."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    rates = sorted(
        {
            int(name.split("_r")[1])
            for name in results
            if "_r" in name
        }
    )

    print(
        f"  {'Rate':>5}"
        f" | {'--- always-admit ---':^30}"
        f" | {'--- token-bucket ---':^30}"
        f" | {'Ratio':>6}"
    )
    print(
        f"  {'':>5}"
        f" | {'TTFT Mean':>10} {'P99':>10} {'Rej':>5}"
        f" | {'TTFT Mean':>10} {'P99':>10} {'Rej':>5}"
        f" | {'P99':>6}"
    )
    print(f"  {'-'*5}-+-{'-'*30}-+-{'-'*30}-+-{'-'*6}")

    for rate in rates:
        aa = results.get(f"exp2_always_r{rate}")
        tb = results.get(f"exp2_bucket_r{rate}")
        if not aa or not tb:
            continue

        ratio = (
            aa["ttft_p99"] / tb["ttft_p99"]
            if tb["ttft_p99"] > 0
            else float("inf")
        )
        print(
            f"  {rate:>5}"
            f" | {aa['ttft_mean']:>10.1f} {aa['ttft_p99']:>10.1f}"
            f" {aa['rejected']:>5}"
            f" | {tb['ttft_mean']:>10.1f} {tb['ttft_p99']:>10.1f}"
            f" {tb['rejected']:>5}"
            f" | {ratio:>6.2f}x"
        )


def analyze_tuning(files):
    """Experiment 3: Token-bucket parameter sensitivity."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    print(
        f"  {'Configuration':<28}"
        f" | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'E2E P99':>10}"
        f" | {'Rej':>5} {'Comp':>5} {'Thru':>6}"
    )
    print(
        f"  {'-'*28}-+-{'-'*10} {'-'*10}-+-{'-'*10}-+-{'-'*5} {'-'*5} {'-'*6}"
    )

    # Always-admit baseline first
    aa = results.get("exp3_always")
    if aa:
        print(
            f"  {'always-admit (baseline)':<28}"
            f" | {aa['ttft_mean']:>10.1f} {aa['ttft_p99']:>10.1f}"
            f" | {aa['e2e_p99']:>10.1f}"
            f" | {aa['rejected']:>5} {aa['completed']:>5}"
            f" {aa['throughput']:>6.1f}"
        )

    # Token-bucket configurations (sorted by capacity)
    tb_configs = sorted(
        [
            (name, r)
            for name, r in results.items()
            if name.startswith("exp3_c")
        ],
        key=lambda x: int(x[0].split("_c")[1].split("_")[0]),
    )

    for name, r in tb_configs:
        # Parse capacity and refill from name: exp3_c500_r400
        parts = name.replace("exp3_c", "").split("_r")
        cap = parts[0]
        refill = parts[1] if len(parts) > 1 else "?"
        label = f"bucket (cap={cap}, refill={refill})"

        print(
            f"  {label:<28}"
            f" | {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
            f" | {r['e2e_p99']:>10.1f}"
            f" | {r['rejected']:>5} {r['completed']:>5}"
            f" {r['throughput']:>6.1f}"
        )


def analyze_calibrated(files):
    """Experiment 4: Calibrated bucket (cap >> mean_input) across seeds."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    seeds = sorted(
        {name.split("_")[2] for name in results if "exp4_" in name}
    )

    print(
        f"  {'Seed':<6} {'Policy':<30}"
        f" | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'E2E P99':>10}"
        f" | {'Rej':>5} {'Comp':>5} {'Thru':>6}"
    )
    print(
        f"  {'-'*6} {'-'*30}"
        f"-+-{'-'*10} {'-'*10}"
        f"-+-{'-'*10}"
        f"-+-{'-'*5} {'-'*5} {'-'*6}"
    )

    ratios = []
    for seed in seeds:
        aa = results.get(f"exp4_always_{seed}")
        cb = results.get(f"exp4_calibrated_{seed}")
        if not aa or not cb:
            continue

        for label, r in [
            ("always-admit", aa),
            ("calibrated (cap=100K,ref=600K)", cb),
        ]:
            print(
                f"  {seed:<6} {label:<30}"
                f" | {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
                f" | {r['e2e_p99']:>10.1f}"
                f" | {r['rejected']:>5} {r['completed']:>5}"
                f" {r['throughput']:>6.1f}"
            )

        if cb["ttft_p99"] > 0:
            ratio = aa["ttft_p99"] / cb["ttft_p99"]
            pct_rejected = (
                cb["rejected"] / (cb["completed"] + cb["rejected"]) * 100
                if (cb["completed"] + cb["rejected"]) > 0
                else 0
            )
            ratios.append(ratio)
            print(
                f"  {'':>6} {'Effect':>30}"
                f" | P99 ratio: {ratio:.2f}x,"
                f" rejection: {pct_rejected:.1f}%"
                f" ({cb['rejected']}/{cb['completed']+cb['rejected']})"
            )
        print()

    # Summary
    if ratios:
        min_r = min(ratios)
        max_r = max(ratios)
        all_better = all(r > 1.0 for r in ratios)
        print(f"  Summary: Calibrated bucket (cap=100K >> mean_input=512):")
        print(f"    P99 ratio: min={min_r:.2f}x  max={max_r:.2f}x")
        if all_better and min_r > 1.2:
            print(
                f"    CONFIRMED: Calibrated token-bucket reduces tail latency"
                f" with practical rejection rate"
            )
        elif all_better:
            print(
                f"    INCONCLUSIVE: Better but <20% in some seeds"
            )
        else:
            print(
                f"    MIXED: Not consistently better across seeds"
            )


ANALYZERS = {
    "core": analyze_core,
    "rate-scaling": analyze_rate_scaling,
    "tuning": analyze_tuning,
    "calibrated": analyze_calibrated,
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
