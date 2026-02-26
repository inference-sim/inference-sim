#!/usr/bin/env python3
"""H28: Decode Attention maxKVLen Overestimation — Analysis.

Analyzes the relationship between batch composition and decode step time
in the roofline model. Compares heterogeneous batches (one long-KV anchor
+ N short-KV requests) against homogeneous short-KV batches to quantify
the overestimation caused by using maxKVLen for attention FLOPs.

Usage:
    python3 analyze.py <output_dir>

Expected files in output_dir:
    step_times.csv             — Main step time data
    attn_flops_comparison.csv  — Roofline vs ideal attention FLOPs
"""

import csv
import sys
from pathlib import Path


def load_csv(filepath):
    """Load CSV file and return list of dicts."""
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def mean(values):
    """Compute mean of a list of numbers."""
    if not values:
        return 0
    return sum(values) / len(values)


def linear_regression_slope(xs, ys):
    """Compute slope of simple linear regression."""
    n = len(xs)
    if n < 2:
        return 0
    x_mean = mean(xs)
    y_mean = mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0
    return num / den


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    step_times_path = output_dir / "step_times.csv"
    attn_flops_path = output_dir / "attn_flops_comparison.csv"

    if not step_times_path.exists():
        print(f"ERROR: {step_times_path} not found", file=sys.stderr)
        sys.exit(1)

    # --- Load step times ---
    rows = load_csv(step_times_path)
    hetero_rows = [r for r in rows if r["batch_type"] == "heterogeneous"]
    homo_rows = [r for r in rows if r["batch_type"] == "homogeneous"]

    # --- Load attention FLOPs comparison ---
    attn_rows = []
    if attn_flops_path.exists():
        attn_rows = load_csv(attn_flops_path)

    # ================================================================
    # Table 1: Step Time Scaling — Heterogeneous Batches
    # ================================================================
    print("=" * 100)
    print("  H28: Decode Attention maxKVLen Overestimation — Analysis")
    print("=" * 100)
    print()

    print("--- Table 1: Heterogeneous Batch Step Times (Anchor KV=4096, Added KV=64) ---")
    print()
    hdr = f"{'Batch Size':>10} {'Total (us)':>12} {'Marginal (us)':>14} {'Attn FLOPs':>16}"
    print(hdr)
    print("-" * len(hdr))

    for r in hetero_rows:
        print(f"{r['batch_size']:>10} {r['total_time_us']:>12} {r['marginal_time_us']:>14} {r['attn_flops']:>16}")

    # ================================================================
    # Table 2: Step Time Scaling — Homogeneous Batches
    # ================================================================
    print()
    print("--- Table 2: Homogeneous Batch Step Times (All KV=64) ---")
    print()
    hdr2 = f"{'Batch Size':>10} {'Total (us)':>12} {'Marginal (us)':>14} {'Attn FLOPs':>16}"
    print(hdr2)
    print("-" * len(hdr2))

    for r in homo_rows:
        print(f"{r['batch_size']:>10} {r['total_time_us']:>12} {r['marginal_time_us']:>14} {r['attn_flops']:>16}")

    # ================================================================
    # Table 3: Heterogeneous vs Homogeneous Comparison (same batch sizes)
    # ================================================================
    print()
    print("--- Table 3: Heterogeneous vs Homogeneous Time Comparison ---")
    print()

    # Build lookup by batch_size
    homo_by_size = {r["batch_size"]: int(r["total_time_us"]) for r in homo_rows}

    hdr3 = f"{'Batch Size':>10} {'Hetero (us)':>12} {'Homo (us)':>12} {'Ratio':>8} {'Excess (us)':>12}"
    print(hdr3)
    print("-" * len(hdr3))

    hetero_excess = []
    for r in hetero_rows:
        bs = r["batch_size"]
        h_time = int(r["total_time_us"])
        if bs in homo_by_size:
            o_time = homo_by_size[bs]
            ratio = h_time / o_time if o_time > 0 else float("inf")
            excess = h_time - o_time
            hetero_excess.append((int(bs), excess, ratio))
            print(f"{bs:>10} {h_time:>12} {o_time:>12} {ratio:>8.3f} {excess:>12}")

    # ================================================================
    # Table 4: Attention FLOPs Overestimation
    # ================================================================
    if attn_rows:
        print()
        print("--- Table 4: Attention FLOPs Overestimation (Roofline vs Ideal) ---")
        print()
        hdr4 = f"{'Batch Size':>10} {'Roofline FLOPs':>16} {'Ideal FLOPs':>16} {'Overestimation':>16}"
        print(hdr4)
        print("-" * len(hdr4))

        for r in attn_rows:
            print(f"{r['batch_size']:>10} {r['roofline_attn_flops']:>16} {r['ideal_attn_flops']:>16} {r['overestimation_factor']:>16}")

    # ================================================================
    # Marginal Cost Analysis
    # ================================================================
    print()
    print("=" * 80)
    print("  Marginal Cost Analysis")
    print("=" * 80)
    print()

    # Heterogeneous marginal costs (skip anchor-only row)
    hetero_marginals = []
    for r in hetero_rows:
        if int(r["batch_size"]) >= 2:
            hetero_marginals.append((int(r["batch_size"]), int(r["marginal_time_us"])))

    if hetero_marginals:
        first_marginal = hetero_marginals[0][1]
        last_marginal = hetero_marginals[-1][1]
        avg_marginal = mean([m for _, m in hetero_marginals])

        print(f"  Heterogeneous batches (anchor KV=4096 + N short KV=64):")
        print(f"    First marginal cost (bs=2):  {first_marginal} us")
        print(f"    Last marginal cost (bs=16):  {last_marginal} us")
        print(f"    Average marginal cost:       {avg_marginal:.1f} us")

        # Compute slope of marginal cost trend
        xs = [float(bs) for bs, _ in hetero_marginals]
        ys = [float(m) for _, m in hetero_marginals]
        slope = linear_regression_slope(xs, ys)
        print(f"    Marginal cost slope:         {slope:.2f} us/request")

        if slope > 1.0:
            print(f"    --> INCREASING marginal cost (superlinear scaling)")
        elif slope < -1.0:
            print(f"    --> DECREASING marginal cost (sublinear scaling)")
        else:
            print(f"    --> APPROXIMATELY CONSTANT marginal cost (linear scaling)")

    # Homogeneous marginal costs
    homo_marginals = []
    for r in homo_rows:
        if int(r["batch_size"]) >= 2:
            homo_marginals.append((int(r["batch_size"]), int(r["marginal_time_us"])))

    if homo_marginals:
        first_homo = homo_marginals[0][1]
        last_homo = homo_marginals[-1][1]
        avg_homo = mean([m for _, m in homo_marginals])

        print()
        print(f"  Homogeneous batches (all KV=64):")
        print(f"    First marginal cost (bs=2):  {first_homo} us")
        print(f"    Last marginal cost (bs=16):  {last_homo} us")
        print(f"    Average marginal cost:       {avg_homo:.1f} us")

        xs_h = [float(bs) for bs, _ in homo_marginals]
        ys_h = [float(m) for _, m in homo_marginals]
        slope_h = linear_regression_slope(xs_h, ys_h)
        print(f"    Marginal cost slope:         {slope_h:.2f} us/request")

    # ================================================================
    # Overestimation Summary
    # ================================================================
    if attn_rows:
        print()
        print("=" * 80)
        print("  Attention FLOPs Overestimation Summary")
        print("=" * 80)
        print()

        factors = [float(r["overestimation_factor"]) for r in attn_rows]
        print(f"  Min overestimation factor:  {min(factors):.4f}x (bs={attn_rows[0]['batch_size']})")
        print(f"  Max overestimation factor:  {max(factors):.4f}x (bs={attn_rows[-1]['batch_size']})")
        print(f"  Mean overestimation factor: {mean(factors):.4f}x")
        print()
        print(f"  Interpretation:")
        print(f"    At batch_size=2 (1 anchor + 1 short): roofline computes {factors[0]:.2f}x the ideal FLOPs")
        print(f"    At batch_size=16 (1 anchor + 15 short): roofline computes {factors[-1]:.2f}x the ideal FLOPs")
        print(f"    The overestimation grows with batch size because more short requests")
        print(f"    are each attributed maxKVLen={4096} instead of their actual KV={64}")

    # ================================================================
    # Hypothesis Verdict
    # ================================================================
    print()
    print("=" * 80)
    print("  Hypothesis Verdict")
    print("=" * 80)
    print()

    # Check 1: Do attention FLOPs use maxKVLen for all requests?
    if attn_rows and all(float(r["overestimation_factor"]) > 1.0 for r in attn_rows):
        print("  CHECK 1: Attention FLOPs use maxKVLen for all requests")
        print("    CONFIRMED - Overestimation factor > 1.0 for all batch sizes")
        check1 = True
    else:
        print("  CHECK 1: Attention FLOPs use maxKVLen for all requests")
        print("    REFUTED - Overestimation factor is not consistently > 1.0")
        check1 = False

    # Check 2: Overestimation ratio matches maxKVLen/meanKVLen?
    # (This is verified in the Go test with tolerance checks)
    if attn_rows:
        # Verify first entry: bs=2, anchor=4096, short=64
        # meanKV = (4096+64)/2 = 2080, expected = 4096/2080 = 1.9692
        expected_first = 4096.0 / ((4096.0 + 64.0) / 2.0)
        actual_first = float(attn_rows[0]["overestimation_factor"])
        ratio_match = abs(actual_first - expected_first) / expected_first < 0.01
        print(f"\n  CHECK 2: Overestimation ratio = maxKVLen/meanKVLen")
        print(f"    Expected (bs=2): {expected_first:.4f}, Actual: {actual_first:.4f}")
        if ratio_match:
            print(f"    CONFIRMED - Ratio matches within 1%")
            check2 = True
        else:
            print(f"    REFUTED - Ratio does not match")
            check2 = False
    else:
        check2 = False

    # Check 3: Superlinear total step time scaling?
    if hetero_marginals:
        # Look at the trend: are later marginals larger than earlier ones?
        increasing_count = sum(1 for i in range(1, len(hetero_marginals))
                               if hetero_marginals[i][1] > hetero_marginals[i - 1][1])
        decreasing_count = sum(1 for i in range(1, len(hetero_marginals))
                               if hetero_marginals[i][1] < hetero_marginals[i - 1][1])

        print(f"\n  CHECK 3: Total step time scales superlinearly")
        print(f"    Marginal cost transitions: {increasing_count} increasing, {decreasing_count} decreasing")

        # The step time may not be purely superlinear because the attention
        # FLOPs overestimation competes with GEMM and memory bandwidth effects.
        # We check if the overall trend has meaningful non-linearity.
        if hetero_excess:
            first_excess = hetero_excess[0][2]  # ratio at smallest batch
            last_excess = hetero_excess[-1][2]   # ratio at largest batch

            if last_excess > first_excess * 1.05:
                print(f"    CONFIRMED - Hetero/homo ratio grows: {first_excess:.3f} -> {last_excess:.3f}")
                check3 = True
            elif last_excess < first_excess * 0.95:
                print(f"    REFUTED - Hetero/homo ratio shrinks: {first_excess:.3f} -> {last_excess:.3f}")
                print(f"    Attention overestimation is masked by other cost components")
                check3 = False
            else:
                print(f"    INCONCLUSIVE - Hetero/homo ratio stable: {first_excess:.3f} -> {last_excess:.3f}")
                print(f"    Attention overestimation may be masked by GEMM or memory dominance")
                check3 = False
        else:
            check3 = False
    else:
        check3 = False

    # Overall verdict
    print()
    if check1 and check2:
        print("  OVERALL: HYPOTHESIS CONFIRMED (at the FLOPs level)")
        print("    The roofline model's decode attention FLOPs DO overestimate by")
        print("    maxKVLen/meanKVLen for heterogeneous batches.")
        if check3:
            print("    This overestimation IS visible in total step time (superlinear scaling).")
        else:
            print("    However, the overestimation may be partially or fully masked in total")
            print("    step time by GEMM costs, memory bandwidth, or MFU lookup effects.")
    else:
        print("  OVERALL: HYPOTHESIS REFUTED")
        print("    The roofline model does not exhibit the predicted maxKVLen overestimation.")

    print()


if __name__ == "__main__":
    main()
