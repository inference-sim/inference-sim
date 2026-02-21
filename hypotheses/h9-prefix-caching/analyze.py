#!/usr/bin/env python3
"""Analysis script for H9: Prefix Caching Effectiveness hypothesis experiment.

Parses BLIS multi-block output files and produces formatted comparison tables.
Called by run.sh with experiment type and output file paths.

Usage:
    python3 analyze.py monotonicity exp1_p0_42.txt exp1_p64_42.txt ...
    python3 analyze.py cluster exp2_p64_42.txt exp2_p256_42.txt ...
    python3 analyze.py capacity exp3_b100.txt exp3_b500.txt ...
"""

import json
import re
import sys
from pathlib import Path


def _warn_if_section_present(content, section_header, metric_name, filepath):
    """Warn on stderr if a section header exists but a metric regex didn't match."""
    if section_header in content:
        print(f"WARNING: '{metric_name}' not found in '{filepath}' "
              f"despite '{section_header}' section being present. "
              f"Check regex against cmd/root.go format strings.",
              file=sys.stderr)


def parse_output(filepath):
    """Parse multi-block BLIS output into cluster metrics + cache stats."""
    content = Path(filepath).read_text()

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    # Extract cache hit rate from KV Cache Metrics section
    cache_hit_rate = 0.0
    hit_match = re.search(r"Cache Hit Rate:\s+([\d.]+)", content)
    if hit_match:
        cache_hit_rate = float(hit_match.group(1))
    else:
        _warn_if_section_present(content, "=== KV Cache Metrics ===",
                                 "Cache Hit Rate", filepath)

    # Extract preemption rate
    preempt_rate = 0.0
    preempt_match = re.search(r"Preemption Rate:\s+([\d.]+)", content)
    if preempt_match:
        preempt_rate = float(preempt_match.group(1))
    else:
        _warn_if_section_present(content, "=== KV Cache Metrics ===",
                                 "Preemption Rate", filepath)

    # Extract target distribution from trace summary
    dist = {}
    dist_match = re.search(
        r"Target Distribution:\n((?:\s+instance_\d+: \d+\n?)+)", content
    )
    if dist_match:
        for line in dist_match.group(1).strip().split("\n"):
            parts = line.strip().split(": ")
            dist[parts[0]] = int(parts[1])

    return {
        "ttft_mean": cluster["ttft_mean_ms"] if cluster else 0,
        "ttft_p99": cluster["ttft_p99_ms"] if cluster else 0,
        "e2e_mean": cluster["e2e_mean_ms"] if cluster else 0,
        "throughput": cluster["responses_per_sec"] if cluster else 0,
        "total_input": cluster["total_input_tokens"] if cluster else 0,
        "completed": cluster["completed_requests"] if cluster else 0,
        "cache_hit_rate": cache_hit_rate,
        "preempt_rate": preempt_rate,
        "dist": dist,
    }


def analyze_monotonicity(files):
    """Experiment 1: Core monotonicity — TTFT vs prefix_length (single instance)."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    seeds = sorted({name.split("_")[2] for name in results})
    prefixes = sorted(
        {int(name.split("_")[1][1:]) for name in results}, key=int
    )

    # Table header
    print(
        f"  {'PfxLen':>6} | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'Cache Hit':>10} | {'Preempt':>8} | {'Input/Req':>10} | Seed"
    )
    print(f"  {'-'*6}-+-{'-'*21}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

    for seed in seeds:
        for pfx in prefixes:
            r = results.get(f"exp1_p{pfx}_{seed}")
            if not r:
                continue
            avg_input = r["total_input"] / r["completed"] if r["completed"] > 0 else 0
            print(
                f"  {pfx:>6} |"
                f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
                f" | {r['cache_hit_rate']:>10.4f}"
                f" | {r['preempt_rate']:>8.4f}"
                f" | {avg_input:>10.0f}"
                f" | {seed}"
            )
        print()

    # Summary: average across seeds with monotonicity check
    print("  Summary (averaged across seeds):")
    print(
        f"  {'PfxLen':>6} | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'Cache Hit':>10} | {'TTFT Δ vs 0':>12}"
    )
    print(f"  {'-'*6}-+-{'-'*21}-+-{'-'*10}-+-{'-'*12}")

    baseline_ttft = None
    prev_ttft = None
    monotonic = True

    for pfx in prefixes:
        ttft_vals = []
        p99_vals = []
        hit_vals = []
        for seed in seeds:
            r = results.get(f"exp1_p{pfx}_{seed}")
            if r:
                ttft_vals.append(r["ttft_mean"])
                p99_vals.append(r["ttft_p99"])
                hit_vals.append(r["cache_hit_rate"])

        if not ttft_vals:
            continue

        avg_ttft = sum(ttft_vals) / len(ttft_vals)
        avg_p99 = sum(p99_vals) / len(p99_vals)
        avg_hit = sum(hit_vals) / len(hit_vals)

        if baseline_ttft is None:
            baseline_ttft = avg_ttft
            delta = "baseline"
        else:
            pct = ((avg_ttft - baseline_ttft) / baseline_ttft) * 100
            delta = f"{pct:+.1f}%"

        if prev_ttft is not None and avg_ttft > prev_ttft * 1.05:
            monotonic = False

        prev_ttft = avg_ttft

        print(
            f"  {pfx:>6} |"
            f" {avg_ttft:>10.1f} {avg_p99:>10.1f}"
            f" | {avg_hit:>10.4f}"
            f" | {delta:>12}"
        )

    print()
    verdict = "CONFIRMED" if monotonic else "VIOLATED"
    print(f"  Monotonicity: {verdict}")
    if monotonic:
        if baseline_ttft and prev_ttft:
            reduction = ((baseline_ttft - prev_ttft) / baseline_ttft) * 100
            print(
                f"  Total TTFT reduction (p0 → p{prefixes[-1]}): {reduction:.1f}%"
            )


def analyze_cluster(files):
    """Experiment 2: Cluster-scale with prefix-affinity routing."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    seeds = sorted({name.split("_")[2] for name in results})
    prefixes = sorted(
        {int(name.split("_")[1][1:]) for name in results}, key=int
    )

    # Per-seed table
    print(
        f"  {'PfxLen':>6} | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'Cache Hit':>10} | {'Throughput':>10} | Seed"
    )
    print(f"  {'-'*6}-+-{'-'*21}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")

    for seed in seeds:
        for pfx in prefixes:
            r = results.get(f"exp2_p{pfx}_{seed}")
            if not r:
                continue
            print(
                f"  {pfx:>6} |"
                f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
                f" | {r['cache_hit_rate']:>10.4f}"
                f" | {r['throughput']:>10.1f}"
                f" | {seed}"
            )
        print()

    # Summary
    print("  Summary (averaged across seeds):")
    prev_ttft = None
    monotonic = True
    for pfx in prefixes:
        ttft_vals = [
            results[f"exp2_p{pfx}_{s}"]["ttft_mean"]
            for s in seeds
            if f"exp2_p{pfx}_{s}" in results
        ]
        hit_vals = [
            results[f"exp2_p{pfx}_{s}"]["cache_hit_rate"]
            for s in seeds
            if f"exp2_p{pfx}_{s}" in results
        ]
        if not ttft_vals:
            continue
        avg_ttft = sum(ttft_vals) / len(ttft_vals)
        avg_hit = sum(hit_vals) / len(hit_vals)
        if prev_ttft is not None and avg_ttft > prev_ttft * 1.05:
            monotonic = False
        prev_ttft = avg_ttft
        print(f"    p{pfx}: TTFT={avg_ttft:.1f}ms, CacheHit={avg_hit:.4f}")

    verdict = "CONFIRMED" if monotonic else "VIOLATED"
    print(f"  Cluster monotonicity: {verdict}")


def analyze_capacity(files):
    """Experiment 3: Cache capacity stress test."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    blocks = sorted(
        {int(name.split("_b")[1].split("_")[0]) for name in results if "_b" in name}
    )
    seeds = sorted(
        {name.split("_")[-1] for name in results}
    )

    print(
        f"  {'Blocks':>8} | {'TTFT Mean':>10} {'P99':>10}"
        f" | {'Cache Hit':>10} | {'Preempt':>8} | Seed"
    )
    print(f"  {'-'*8}-+-{'-'*21}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")

    for seed in seeds:
        for b in blocks:
            r = results.get(f"exp3_b{b}_{seed}")
            if not r:
                continue
            print(
                f"  {b:>8} |"
                f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
                f" | {r['cache_hit_rate']:>10.4f}"
                f" | {r['preempt_rate']:>8.4f}"
                f" | {seed}"
            )
        print()

    # Summary
    print("  Summary (averaged across seeds):")
    for b in blocks:
        ttft_vals = [
            results[f"exp3_b{b}_{s}"]["ttft_mean"]
            for s in seeds
            if f"exp3_b{b}_{s}" in results
        ]
        hit_vals = [
            results[f"exp3_b{b}_{s}"]["cache_hit_rate"]
            for s in seeds
            if f"exp3_b{b}_{s}" in results
        ]
        preempt_vals = [
            results[f"exp3_b{b}_{s}"]["preempt_rate"]
            for s in seeds
            if f"exp3_b{b}_{s}" in results
        ]
        if not ttft_vals:
            continue
        avg_ttft = sum(ttft_vals) / len(ttft_vals)
        avg_hit = sum(hit_vals) / len(hit_vals)
        avg_preempt = sum(preempt_vals) / len(preempt_vals)
        print(
            f"    {b:>6} blocks: TTFT={avg_ttft:.1f}ms,"
            f" CacheHit={avg_hit:.4f}, Preempt={avg_preempt:.4f}"
        )


ANALYZERS = {
    "monotonicity": analyze_monotonicity,
    "cluster": analyze_cluster,
    "capacity": analyze_capacity,
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
