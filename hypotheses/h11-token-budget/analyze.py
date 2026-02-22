#!/usr/bin/env python3
"""Analysis script for H11: Token Budget hypothesis experiment.

Parses BLIS multi-block output files and produces comparison tables.
Called by run.sh with experiment type and output file paths.

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
- Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
- Cluster block has "instance_id": "cluster"
- JSON field names verified against sim/metrics_utils.go MetricsOutput struct:
    responses_per_sec, itl_mean_ms, itl_p99_ms, ttft_mean_ms, ttft_p99_ms,
    e2e_mean_ms, e2e_p99_ms, completed_requests, still_queued, still_running,
    injected_requests

Usage:
    python3 analyze.py monotonicity exp1_t512_s42.txt exp1_t1024_s42.txt ...
    python3 analyze.py conservation exp1_t512_s42.txt ...
"""

import json
import re
import sys
from pathlib import Path


def parse_output(filepath):
    """Parse BLIS output -> cluster metrics dict."""
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

    return {
        "throughput": cluster["responses_per_sec"],
        "itl_mean": cluster["itl_mean_ms"],
        "itl_p99": cluster["itl_p99_ms"],
        "ttft_mean": cluster["ttft_mean_ms"],
        "ttft_p99": cluster["ttft_p99_ms"],
        "e2e_mean": cluster["e2e_mean_ms"],
        "e2e_p99": cluster["e2e_p99_ms"],
        "completed": cluster["completed_requests"],
        "still_queued": cluster["still_queued"],
        "still_running": cluster["still_running"],
        "injected": cluster["injected_requests"],
        "tokens_per_sec": cluster["tokens_per_sec"],
        "preemption_count": cluster.get("preemption_count", 0),
    }


def parse_filename(filepath):
    """Extract token budget and seed from filename like exp1_t512_s42.txt."""
    name = Path(filepath).stem
    budget_match = re.search(r"_t(\d+)", name)
    seed_match = re.search(r"_s(\d+)", name)
    budget = int(budget_match.group(1)) if budget_match else 0
    seed = int(seed_match.group(1)) if seed_match else 0
    return budget, seed


def check_monotonicity(values, direction="increasing", tolerance=0.05):
    """Check if values are monotonically increasing/decreasing within tolerance.

    Args:
        values: list of (x, y) tuples sorted by x
        direction: "increasing" or "decreasing"
        tolerance: relative tolerance for violations (0.05 = 5%)

    Returns:
        (is_monotonic, violations) where violations is a list of
        (x_prev, y_prev, x_curr, y_curr) tuples for each violation
    """
    violations = []
    for i in range(1, len(values)):
        x_prev, y_prev = values[i - 1]
        x_curr, y_curr = values[i]

        if direction == "increasing":
            # Allow y_curr < y_prev only if the drop is within tolerance
            if y_prev > 0 and y_curr < y_prev * (1 - tolerance):
                violations.append((x_prev, y_prev, x_curr, y_curr))
            elif y_prev == 0 and y_curr < 0:
                violations.append((x_prev, y_prev, x_curr, y_curr))
        else:  # decreasing
            if y_prev > 0 and y_curr > y_prev * (1 + tolerance):
                violations.append((x_prev, y_prev, x_curr, y_curr))

    return len(violations) == 0, violations


def analyze_monotonicity(files):
    """Experiment 1: Verify monotonic trends as token budget increases."""
    results = {}
    for f in files:
        budget, seed = parse_filename(f)
        r = parse_output(f)
        if r:
            results[(budget, seed)] = r

    if not results:
        print("  ERROR: No valid results found")
        return

    seeds = sorted({s for _, s in results})
    budgets = sorted({b for b, _ in results})

    # Per-seed detailed table
    for seed in seeds:
        print(f"  Seed {seed}:")
        print(
            f"    {'Budget':>7} | {'Throughput':>10} {'ITL mean':>10} {'ITL p99':>10}"
            f" | {'TTFT p99':>10} {'E2E mean':>10} {'E2E p99':>10}"
        )
        print(f"    {'-' * 7}-+-{'-' * 31}-+-{'-' * 31}")

        for budget in budgets:
            r = results.get((budget, seed))
            if not r:
                print(f"    {budget:>7} | {'TIMEOUT':>10}")
                continue
            print(
                f"    {budget:>7} |"
                f" {r['throughput']:>10.2f} {r['itl_mean']:>10.3f} {r['itl_p99']:>10.3f}"
                f" | {r['ttft_p99']:>10.2f} {r['e2e_mean']:>10.2f} {r['e2e_p99']:>10.2f}"
            )
        print()

    # Compute seed-averaged values
    avg = {}
    for budget in budgets:
        vals = [results[(budget, s)] for s in seeds if (budget, s) in results]
        if not vals:
            continue
        avg[budget] = {
            "throughput": sum(v["throughput"] for v in vals) / len(vals),
            "itl_mean": sum(v["itl_mean"] for v in vals) / len(vals),
            "itl_p99": sum(v["itl_p99"] for v in vals) / len(vals),
            "ttft_mean": sum(v["ttft_mean"] for v in vals) / len(vals),
            "ttft_p99": sum(v["ttft_p99"] for v in vals) / len(vals),
            "e2e_mean": sum(v["e2e_mean"] for v in vals) / len(vals),
            "e2e_p99": sum(v["e2e_p99"] for v in vals) / len(vals),
            "tokens_per_sec": sum(v["tokens_per_sec"] for v in vals) / len(vals),
        }

    # Summary table (averaged across seeds)
    print("  Summary (averaged across seeds):")
    print(
        f"    {'Budget':>7} | {'Throughput':>10} {'Tok/sec':>10}"
        f" | {'ITL mean':>10} {'ITL p99':>10}"
        f" | {'TTFT p99':>10} {'E2E mean':>10}"
    )
    print(f"    {'-' * 7}-+-{'-' * 21}-+-{'-' * 21}-+-{'-' * 21}")

    for budget in budgets:
        if budget not in avg:
            continue
        a = avg[budget]
        print(
            f"    {budget:>7} |"
            f" {a['throughput']:>10.2f} {a['tokens_per_sec']:>10.2f}"
            f" | {a['itl_mean']:>10.3f} {a['itl_p99']:>10.3f}"
            f" | {a['ttft_p99']:>10.2f} {a['e2e_mean']:>10.2f}"
        )

    # Monotonicity checks
    print()
    print("  Monotonicity Checks (5% tolerance):")

    # Throughput should increase with budget
    throughput_vals = [(b, avg[b]["throughput"]) for b in budgets if b in avg]
    mono_tp, viol_tp = check_monotonicity(throughput_vals, "increasing")
    status_tp = "PASS" if mono_tp else "FAIL"
    print(f"    Throughput (expect increasing):  [{status_tp}]")
    if viol_tp:
        for xp, yp, xc, yc in viol_tp:
            print(f"      Violation: budget {xp}->{xc}: {yp:.2f} -> {yc:.2f}")

    # ITL mean should increase with budget
    itl_vals = [(b, avg[b]["itl_mean"]) for b in budgets if b in avg]
    mono_itl, viol_itl = check_monotonicity(itl_vals, "increasing")
    status_itl = "PASS" if mono_itl else "FAIL"
    print(f"    ITL mean (expect increasing):    [{status_itl}]")
    if viol_itl:
        for xp, yp, xc, yc in viol_itl:
            print(f"      Violation: budget {xp}->{xc}: {yp:.3f} -> {yc:.3f}")

    # E2E mean -- direction not predicted, just report
    e2e_vals = [(b, avg[b]["e2e_mean"]) for b in budgets if b in avg]
    mono_e2e_inc, _ = check_monotonicity(e2e_vals, "increasing")
    mono_e2e_dec, _ = check_monotonicity(e2e_vals, "decreasing")
    if mono_e2e_inc:
        print(f"    E2E mean:                        [INCREASING]")
    elif mono_e2e_dec:
        print(f"    E2E mean:                        [DECREASING]")
    else:
        print(f"    E2E mean:                        [NON-MONOTONIC]")

    # TTFT p99 -- direction not predicted, just report
    ttft_vals = [(b, avg[b]["ttft_p99"]) for b in budgets if b in avg]
    mono_ttft_inc, _ = check_monotonicity(ttft_vals, "increasing")
    mono_ttft_dec, _ = check_monotonicity(ttft_vals, "decreasing")
    if mono_ttft_inc:
        print(f"    TTFT p99:                        [INCREASING]")
    elif mono_ttft_dec:
        print(f"    TTFT p99:                        [DECREASING]")
    else:
        print(f"    TTFT p99:                        [NON-MONOTONIC]")

    # Per-seed monotonicity for primary metrics
    print()
    print("  Per-Seed Monotonicity:")
    for seed in seeds:
        tp_vals = [(b, results[(b, seed)]["throughput"]) for b in budgets if (b, seed) in results]
        itl_vals_seed = [(b, results[(b, seed)]["itl_mean"]) for b in budgets if (b, seed) in results]
        mono_tp_s, _ = check_monotonicity(tp_vals, "increasing")
        mono_itl_s, _ = check_monotonicity(itl_vals_seed, "increasing")
        status_tp_s = "PASS" if mono_tp_s else "FAIL"
        status_itl_s = "PASS" if mono_itl_s else "FAIL"
        print(f"    Seed {seed}: throughput [{status_tp_s}]  ITL mean [{status_itl_s}]")

    # Overall verdict
    print()
    if mono_tp and mono_itl:
        print("  VERDICT: CONFIRMED -- throughput increases and ITL worsens with larger token budget")
    elif mono_tp:
        print("  VERDICT: PARTIALLY CONFIRMED -- throughput increases but ITL trend not monotonic")
    elif mono_itl:
        print("  VERDICT: PARTIALLY CONFIRMED -- ITL worsens but throughput trend not monotonic")
    else:
        print("  VERDICT: REFUTED -- neither throughput nor ITL show predicted monotonic trends")


def analyze_conservation(files):
    """Experiment 2: Verify INV-1 (request conservation) at each config."""
    results = {}
    for f in files:
        budget, seed = parse_filename(f)
        r = parse_output(f)
        if r:
            results[(budget, seed)] = r

    all_pass = True
    for (budget, seed), r in sorted(results.items()):
        actual = r["completed"] + r["still_queued"] + r["still_running"]
        expected = r["injected"]
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  budget={budget:>5} seed={seed}: "
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
