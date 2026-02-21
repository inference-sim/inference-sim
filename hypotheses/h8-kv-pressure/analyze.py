#!/usr/bin/env python3
"""Analysis script for H8: KV Cache Pressure hypothesis experiment.

Parses BLIS multi-block output files and produces comparison tables.
Called by run.sh with experiment type and output file paths.

Usage:
    python3 analyze.py monotonicity exp1_b5000_s42.txt exp1_b3000_s42.txt ...
    python3 analyze.py conservation exp1_b5000_s42.txt ...
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
    """Parse multi-block BLIS output into cluster metrics + KV cache metrics."""
    content = Path(filepath).read_text()
    if not content.strip():
        return None

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    if not cluster:
        return None

    # Extract preemption rate from KV Cache Metrics text section
    preemption_rate = 0.0
    pr_match = re.search(r"Preemption Rate: ([\d.]+)", content)
    if pr_match:
        preemption_rate = float(pr_match.group(1))
    else:
        _warn_if_section_present(content, "=== KV Cache Metrics ===",
                                 "Preemption Rate", filepath)

    # Extract cache hit rate
    cache_hit_rate = 0.0
    chr_match = re.search(r"Cache Hit Rate: ([\d.]+)", content)
    if chr_match:
        cache_hit_rate = float(chr_match.group(1))
    else:
        _warn_if_section_present(content, "=== KV Cache Metrics ===",
                                 "Cache Hit Rate", filepath)

    # Sum preemption_count from per-instance JSON blocks (new field)
    json_preemption_total = 0
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") != "cluster":
            json_preemption_total += block.get("preemption_count", 0)

    # Fallback: compute from rate if JSON field not available
    preemption_count = json_preemption_total
    if preemption_count == 0 and preemption_rate > 0:
        preemption_count = int(round(preemption_rate * cluster["completed_requests"]))

    return {
        "ttft_mean": cluster["ttft_mean_ms"],
        "ttft_p99": cluster["ttft_p99_ms"],
        "e2e_mean": cluster["e2e_mean_ms"],
        "e2e_p99": cluster["e2e_p99_ms"],
        "throughput": cluster["responses_per_sec"],
        "completed": cluster["completed_requests"],
        "still_queued": cluster["still_queued"],
        "still_running": cluster["still_running"],
        "injected": cluster["injected_requests"],
        "preemption_rate": preemption_rate,
        "preemption_count": preemption_count,
        "cache_hit_rate": cache_hit_rate,
    }


def parse_filename(filepath):
    """Extract blocks and seed from filename like exp1_b5000_s42.txt."""
    name = Path(filepath).stem
    blocks_match = re.search(r"_b(\d+)", name)
    seed_match = re.search(r"_s(\d+)", name)
    blocks = int(blocks_match.group(1)) if blocks_match else 0
    seed = int(seed_match.group(1)) if seed_match else 0
    return blocks, seed


def analyze_monotonicity(files):
    """Experiment 1: Verify monotonic increase in preemptions as blocks decrease."""
    results = {}
    for f in files:
        blocks, seed = parse_filename(f)
        r = parse_output(f)
        if r:
            results[(blocks, seed)] = r

    seeds = sorted({s for _, s in results})
    block_counts = sorted({b for b, _ in results}, reverse=True)

    # Per-seed detailed table
    for seed in seeds:
        print(f"  Seed {seed}:")
        print(
            f"    {'Blocks':>7} | {'Preempt Rate':>12} {'Preempt #':>9}"
            f" | {'TTFT p99':>10} {'E2E p99':>10}"
            f" | {'Throughput':>10} {'Cache Hit':>9}"
        )
        print(f"    {'-' * 7}-+-{'-' * 22}-+-{'-' * 21}-+-{'-' * 20}")

        for blocks in block_counts:
            r = results.get((blocks, seed))
            if not r:
                print(f"    {blocks:>7} | {'TIMEOUT':>12}")
                continue
            print(
                f"    {blocks:>7} |"
                f" {r['preemption_rate']:>12.4f} {r['preemption_count']:>9d}"
                f" | {r['ttft_p99']:>10.1f} {r['e2e_p99']:>10.1f}"
                f" | {r['throughput']:>10.1f} {r['cache_hit_rate']:>9.4f}"
            )
        print()

    # Monotonicity check (blocks descending = pressure increasing)
    print("  Monotonicity Check:")
    all_monotonic_preemption = True
    all_monotonic_ttft = True
    for seed in seeds:
        prev_preempt = -1.0
        prev_ttft = -1.0
        mono_p = True
        mono_t = True
        for blocks in block_counts:  # descending blocks
            r = results.get((blocks, seed))
            if not r:
                mono_p = False
                mono_t = False
                break
            # As blocks decrease (later in loop), preemption should increase
            if r["preemption_rate"] < prev_preempt - 0.001:
                mono_p = False
            if r["ttft_p99"] < prev_ttft - 0.1:
                mono_t = False
            prev_preempt = r["preemption_rate"]
            prev_ttft = r["ttft_p99"]

        status_p = "PASS" if mono_p else "FAIL"
        status_t = "PASS" if mono_t else "FAIL"
        print(f"    Seed {seed}: preemption [{status_p}]  TTFT p99 [{status_t}]")
        if not mono_p:
            all_monotonic_preemption = False
        if not mono_t:
            all_monotonic_ttft = False

    print()
    verdict_p = "CONFIRMED" if all_monotonic_preemption else "REFUTED"
    verdict_t = "CONFIRMED" if all_monotonic_ttft else "REFUTED"
    print(f"  Preemption monotonicity: {verdict_p}")
    print(f"  TTFT p99 monotonicity:   {verdict_t}")

    # Summary table (averaged across seeds)
    print()
    print("  Summary (averaged across seeds):")
    print(
        f"    {'Blocks':>7} | {'Preempt Rate':>12} {'TTFT p99':>10}"
        f" {'E2E p99':>10} | {'vs Baseline':>11}"
    )
    print(f"    {'-' * 7}-+-{'-' * 33}-+-{'-' * 11}")

    baseline_ttft = None
    for blocks in block_counts:
        vals = [results[(blocks, s)] for s in seeds if (blocks, s) in results]
        if not vals:
            continue
        avg_pr = sum(v["preemption_rate"] for v in vals) / len(vals)
        avg_ttft = sum(v["ttft_p99"] for v in vals) / len(vals)
        avg_e2e = sum(v["e2e_p99"] for v in vals) / len(vals)
        if baseline_ttft is None:
            baseline_ttft = avg_ttft
        ratio = avg_ttft / baseline_ttft if baseline_ttft > 0 else 0
        label = "baseline" if abs(ratio - 1.0) < 0.01 else f"{ratio:.2f}x"
        print(
            f"    {blocks:>7} |"
            f" {avg_pr:>12.4f} {avg_ttft:>10.1f} {avg_e2e:>10.1f}"
            f" | {label:>11}"
        )


def analyze_conservation(files):
    """Experiment 2: Verify INV-1 (request conservation) at each config."""
    results = {}
    for f in files:
        blocks, seed = parse_filename(f)
        r = parse_output(f)
        if r:
            results[(blocks, seed)] = r

    all_pass = True
    for (blocks, seed), r in sorted(results.items()):
        actual = r["completed"] + r["still_queued"] + r["still_running"]
        expected = r["injected"]
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  blocks={blocks:>5} seed={seed}: "
            f"injected={expected} completed={r['completed']} "
            f"queued={r['still_queued']} running={r['still_running']} "
            f"[{status}]"
        )

    print()
    verdict = "ALL PASS" if all_pass else "VIOLATIONS FOUND"
    print(f"  Conservation (INV-1): {verdict}")


ANALYZERS = {
    "monotonicity": analyze_monotonicity,
    "conservation": analyze_conservation,
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
