#!/usr/bin/env python3
"""H33: Decode Attention MFU Shape Mismatch -- Analysis.

Parses Go test output and computes summary statistics to determine whether
using maxKVLen for the MFU lookup systematically underestimates decode
attention time in heterogeneous batches.

Usage:
    python3 analyze.py <output_dir>

Expected files in output_dir:
    test_output.txt  -- Raw Go test output (stdout + verbose logs)
"""

import re
import sys
from pathlib import Path


def parse_results_table(lines):
    """Parse the H33_RESULTS section into a list of dicts."""
    rows = []
    in_section = False
    for line in lines:
        if "H33_RESULTS_START" in line:
            in_section = True
            continue
        if "H33_RESULTS_END" in line:
            break
        if not in_section:
            continue
        if line.startswith("---") or line.startswith("scenario"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 9:
            try:
                rows.append({
                    "scenario": parts[0],
                    "bs": int(parts[1]),
                    "maxKV": int(parts[2]),
                    "meanKV": float(parts[3]),
                    "range": parts[4].replace("x", ""),
                    "currentTimeS": float(parts[5]),
                    "perReqTimeS": float(parts[6]),
                    "ratio": float(parts[7]),
                    "mfuMaxKV": float(parts[8]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def parse_range_sweep(lines):
    """Parse the H33_RANGE_SWEEP section."""
    rows = []
    in_section = False
    for line in lines:
        if "H33_RANGE_SWEEP_START" in line:
            in_section = True
            continue
        if "H33_RANGE_SWEEP_END" in line:
            break
        if not in_section:
            continue
        if line.startswith("---") or line.startswith("shortKV"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 7:
            try:
                rows.append({
                    "shortKV": int(parts[0]),
                    "maxKV": int(parts[1]),
                    "meanKV": float(parts[2]),
                    "range": parts[3].replace("x", ""),
                    "currentTimeS": float(parts[4]),
                    "perReqTimeS": float(parts[5]),
                    "ratio": float(parts[6]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def parse_step_time(lines):
    """Parse the H33_STEP_TIME section."""
    rows = []
    in_section = False
    for line in lines:
        if "H33_STEP_TIME_START" in line:
            in_section = True
            continue
        if "H33_STEP_TIME_END" in line:
            break
        if not in_section:
            continue
        if line.startswith("---") or line.startswith("scenario"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 7:
            try:
                rows.append({
                    "scenario": parts[0],
                    "bs": int(parts[1]),
                    "maxKV": int(parts[2]),
                    "baselineStepUS": int(parts[3]),
                    "adjustedStepUS": int(parts[4]),
                    "stepRatio": float(parts[5]),
                    "attnFraction": parts[6].replace("%", ""),
                })
            except (ValueError, IndexError):
                continue
    return rows


def mean(values):
    """Compute mean of a list of numbers."""
    if not values:
        return 0
    return sum(values) / len(values)


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    test_output = output_dir / "test_output.txt"

    if not test_output.exists():
        print(f"ERROR: {test_output} not found", file=sys.stderr)
        sys.exit(1)

    lines = test_output.read_text().splitlines()

    # ================================================================
    # Parse sections
    # ================================================================
    results = parse_results_table(lines)
    range_sweep = parse_range_sweep(lines)
    step_times = parse_step_time(lines)

    print("=" * 100)
    print("  H33: Decode Attention MFU Shape Mismatch -- Analysis")
    print("=" * 100)
    print()

    # ================================================================
    # Table 1: Core Results (current vs per-request)
    # ================================================================
    if results:
        print("--- Table 1: Attention Time Ratio (Current / Per-Request) ---")
        print()
        hdr = f"{'Scenario':<30} {'BS':>4} {'MaxKV':>6} {'Range':>6} {'Ratio':>10} {'Status':>12}"
        print(hdr)
        print("-" * len(hdr))

        homo_ratios = []
        hetero_ratios = []
        high_hetero_ratios = []

        for r in results:
            ratio = r["ratio"]
            rng = float(r["range"])
            status = "WITHIN_5%" if abs(ratio - 1.0) < 0.05 else "OUTSIDE_5%"

            if rng <= 1.0:
                homo_ratios.append(ratio)
            elif rng >= 10.0:
                high_hetero_ratios.append(ratio)
                hetero_ratios.append(ratio)
            else:
                hetero_ratios.append(ratio)

            print(f"{r['scenario']:<30} {r['bs']:>4} {r['maxKV']:>6} {rng:>5.0f}x {ratio:>10.6f} {status:>12}")

        print()

        # Summary statistics
        all_ratios = [r["ratio"] for r in results]
        print(f"  Overall: min ratio = {min(all_ratios):.6f}, max ratio = {max(all_ratios):.6f}, mean = {mean(all_ratios):.6f}")
        if homo_ratios:
            print(f"  Homogeneous baselines: mean ratio = {mean(homo_ratios):.6f} (expect ~1.0)")
        if hetero_ratios:
            print(f"  Heterogeneous (all): mean ratio = {mean(hetero_ratios):.6f}")
        if high_hetero_ratios:
            print(f"  High heterogeneity (10x+ range): mean ratio = {mean(high_hetero_ratios):.6f}")

    # ================================================================
    # Table 2: Range Sweep
    # ================================================================
    if range_sweep:
        print()
        print("--- Table 2: KV Range vs Ratio (Anchor=8192, 1 long + 3 short) ---")
        print()
        hdr2 = f"{'ShortKV':>8} {'Range':>6} {'Ratio':>10}"
        print(hdr2)
        print("-" * len(hdr2))

        for r in range_sweep:
            rng = float(r["range"])
            print(f"{r['shortKV']:>8} {rng:>5.0f}x {r['ratio']:>10.6f}")

    # ================================================================
    # Table 3: Full Step Time Impact
    # ================================================================
    if step_times:
        print()
        print("--- Table 3: Full Step Time Impact (Baseline vs Adjusted) ---")
        print()
        hdr3 = f"{'Scenario':<30} {'Baseline(us)':>12} {'Adjusted(us)':>12} {'StepRatio':>10} {'AttnFrac':>10}"
        print(hdr3)
        print("-" * len(hdr3))

        for r in step_times:
            print(f"{r['scenario']:<30} {r['baselineStepUS']:>12} {r['adjustedStepUS']:>12} "
                  f"{r['stepRatio']:>10.6f} {r['attnFraction']:>9}%")

        step_ratios = [r["stepRatio"] for r in step_times]
        print()
        print(f"  Step time ratio: min = {min(step_ratios):.6f}, max = {max(step_ratios):.6f}, mean = {mean(step_ratios):.6f}")

    # ================================================================
    # Hypothesis Verdict
    # ================================================================
    print()
    print("=" * 80)
    print("  Hypothesis Verdict")
    print("=" * 80)
    print()

    if not results:
        print("  INCONCLUSIVE -- no results data parsed")
        return

    # Check 1: Do homogeneous baselines have ratio ~1.0?
    if homo_ratios:
        homo_ok = all(abs(r - 1.0) < 0.10 for r in homo_ratios)
        print(f"  CHECK 1: Homogeneous baseline ratio ~1.0")
        if homo_ok:
            print(f"    PASS -- all homogeneous ratios within 10% of 1.0 (mean={mean(homo_ratios):.4f})")
        else:
            print(f"    FAIL -- some homogeneous ratios deviate from 1.0 (mean={mean(homo_ratios):.4f})")
    else:
        homo_ok = True
        print(f"  CHECK 1: No homogeneous baselines to check")

    # Check 2: Do heterogeneous batches (10x+ range) have ratio < 1.0?
    # (ratio < 1.0 means current method underestimates relative to per-request)
    if high_hetero_ratios:
        under_count = sum(1 for r in high_hetero_ratios if r < 0.95)
        over_count = sum(1 for r in high_hetero_ratios if r > 1.05)
        within_count = sum(1 for r in high_hetero_ratios if abs(r - 1.0) < 0.05)

        print(f"\n  CHECK 2: Direction of bias for high-heterogeneity batches (10x+ range)")
        print(f"    Underestimates (ratio < 0.95): {under_count}/{len(high_hetero_ratios)}")
        print(f"    Within 5% (0.95-1.05):         {within_count}/{len(high_hetero_ratios)}")
        print(f"    Overestimates (ratio > 1.05):  {over_count}/{len(high_hetero_ratios)}")

        if under_count > len(high_hetero_ratios) / 2:
            bias_direction = "UNDERESTIMATE"
        elif over_count > len(high_hetero_ratios) / 2:
            bias_direction = "OVERESTIMATE"
        else:
            bias_direction = "NO_CONSISTENT_BIAS"
        print(f"    Direction: {bias_direction}")
    else:
        bias_direction = "UNKNOWN"

    # Check 3: Refutation threshold -- are ALL 10x+ range scenarios within 5%?
    print(f"\n  CHECK 3: Refutation threshold (all 10x+ scenarios within 5%)")
    if high_hetero_ratios:
        all_within = all(abs(r - 1.0) < 0.05 for r in high_hetero_ratios)
        if all_within:
            print(f"    REFUTED -- all {len(high_hetero_ratios)} high-heterogeneity scenarios within 5%")
            verdict = "REFUTED"
        else:
            outside = [r for r in high_hetero_ratios if abs(r - 1.0) >= 0.05]
            print(f"    NOT REFUTED -- {len(outside)}/{len(high_hetero_ratios)} scenarios exceed 5% threshold")
            verdict = "CONFIRMED"
    else:
        verdict = "INCONCLUSIVE"
        print(f"    INCONCLUSIVE -- no high-heterogeneity scenarios")

    # Overall
    print()
    if verdict == "CONFIRMED":
        print(f"  OVERALL: HYPOTHESIS CONFIRMED")
        print(f"    Using maxKVLen for the MFU lookup produces a systematic bias in")
        print(f"    decode attention time for heterogeneous batches.")
        print(f"    Direction: {bias_direction}")
        if high_hetero_ratios:
            print(f"    Mean ratio for 10x+ batches: {mean(high_hetero_ratios):.4f}")
    elif verdict == "REFUTED":
        print(f"  OVERALL: HYPOTHESIS REFUTED")
        print(f"    The single maxKVLen MFU lookup is within 5% of the per-request")
        print(f"    MFU-weighted time for all tested heterogeneous batches.")
    else:
        print(f"  OVERALL: INCONCLUSIVE")
        print(f"    Insufficient data to determine verdict.")

    print()


if __name__ == "__main__":
    main()
