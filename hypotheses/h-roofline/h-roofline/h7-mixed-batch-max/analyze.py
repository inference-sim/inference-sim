#!/usr/bin/env python3
"""H27 Mixed-Batch Max Combination — Analysis.

Reads the CSV output from the Go test and evaluates whether the current
weighted-average combination systematically underpredicts mixed-step latency
compared to max(prefillTime, decodeTime).

Usage:
    python3 analyze.py <output_dir>

Expected files in output_dir:
    h27_results.csv  — Per-case comparison (weighted-avg vs max)
    h27_summary.txt  — Aggregate statistics from Go test
"""

import csv
import sys
from pathlib import Path


def mean(values):
    if not values:
        return 0
    return sum(values) / len(values)


def fmt_pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def fmt_delta_pp(base, treatment):
    """Format delta in percentage points (base - treatment)."""
    if base is None or treatment is None:
        return "N/A"
    delta = base - treatment
    direction = "+" if delta > 0 else ""
    return f"{direction}{delta * 100:.1f}pp"


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    csv_path = output_dir / "h27_results.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load results
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("ERROR: no results found in CSV", file=sys.stderr)
        sys.exit(1)

    # Parse numeric fields
    for row in rows:
        row["tp"] = int(row["tp"])
        row["num_prefill_requests"] = int(row["num_prefill_requests"])
        row["total_prefill_tokens"] = int(row["total_prefill_tokens"])
        row["num_decode_requests"] = int(row["num_decode_requests"])
        row["prefill_only_us"] = int(row["prefill_only_us"])
        row["decode_only_us"] = int(row["decode_only_us"])
        row["max_combination_us"] = int(row["max_combination_us"])
        row["weighted_avg_us"] = int(row["weighted_avg_us"])
        row["ratio_wa_to_max"] = float(row["ratio_wa_to_max"])
        row["delta_us"] = int(row["delta_us"])

    # ================================================================
    # Table 1: Per-Case Results
    # ================================================================
    print("=" * 130)
    print("  H27: Mixed-Batch Combination — Weighted-Average vs Max(Prefill, Decode)")
    print("=" * 130)
    print()

    hdr = (f"{'Case':<52} {'TP':>2}  {'PF tok':>6} {'DC req':>6}  "
           f"{'WA (us)':>10} {'Max (us)':>10} {'Ratio':>7} {'Delta':>8}  {'Regime':<18}")
    print(hdr)
    print("-" * len(hdr))

    for row in rows:
        print(f"{row['case_name']:<52} {row['tp']:>2}  "
              f"{row['total_prefill_tokens']:>6} {row['num_decode_requests']:>6}  "
              f"{row['weighted_avg_us']:>10} {row['max_combination_us']:>10} "
              f"{row['ratio_wa_to_max']:>7.4f} {row['delta_us']:>+8}  "
              f"{row['regime']:<18}")

    # ================================================================
    # Table 2: Aggregate by Regime
    # ================================================================
    print()
    print("=" * 100)
    print("  Aggregate by Token Ratio Regime")
    print("=" * 100)
    print()

    regime_groups = {}
    for row in rows:
        regime = row["regime"]
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(row)

    hdr2 = (f"{'Regime':<20} {'N':>4}  {'Avg Ratio':>10}  {'Min Ratio':>10}  "
            f"{'Max Ratio':>10}  {'Underpredict':>12}  {'Avg Delta(us)':>14}")
    print(hdr2)
    print("-" * len(hdr2))

    for regime in ["prefill-dominated", "balanced", "decode-dominated"]:
        group = regime_groups.get(regime, [])
        if not group:
            continue

        ratios = [r["ratio_wa_to_max"] for r in group]
        deltas = [r["delta_us"] for r in group]
        underpredicts = sum(1 for r in group if r["weighted_avg_us"] < r["max_combination_us"])

        print(f"{regime:<20} {len(group):>4}  "
              f"{mean(ratios):>10.4f}  {min(ratios):>10.4f}  {max(ratios):>10.4f}  "
              f"{underpredicts:>8}/{len(group):<3}  {mean(deltas):>14.1f}")

    # Overall
    all_ratios = [r["ratio_wa_to_max"] for r in rows]
    all_deltas = [r["delta_us"] for r in rows]
    all_underpredicts = sum(1 for r in rows if r["weighted_avg_us"] < r["max_combination_us"])

    print("-" * len(hdr2))
    print(f"{'OVERALL':<20} {len(rows):>4}  "
          f"{mean(all_ratios):>10.4f}  {min(all_ratios):>10.4f}  {max(all_ratios):>10.4f}  "
          f"{all_underpredicts:>8}/{len(rows):<3}  {mean(all_deltas):>14.1f}")

    # ================================================================
    # Table 3: Aggregate by TP
    # ================================================================
    print()
    print("=" * 80)
    print("  Aggregate by Tensor Parallelism")
    print("=" * 80)
    print()

    tp_groups = {}
    for row in rows:
        tp = row["tp"]
        if tp not in tp_groups:
            tp_groups[tp] = []
        tp_groups[tp].append(row)

    hdr3 = (f"{'TP':>4}  {'N':>4}  {'Avg Ratio':>10}  "
            f"{'Underpredict':>12}  {'Avg Delta(us)':>14}")
    print(hdr3)
    print("-" * len(hdr3))

    for tp in sorted(tp_groups.keys()):
        group = tp_groups[tp]
        ratios = [r["ratio_wa_to_max"] for r in group]
        deltas = [r["delta_us"] for r in group]
        underpredicts = sum(1 for r in group if r["weighted_avg_us"] < r["max_combination_us"])

        print(f"{tp:>4}  {len(group):>4}  "
              f"{mean(ratios):>10.4f}  "
              f"{underpredicts:>8}/{len(group):<3}  {mean(deltas):>14.1f}")

    # ================================================================
    # Table 4: Underprediction Magnitude Analysis
    # ================================================================
    print()
    print("=" * 80)
    print("  Underprediction Magnitude Analysis")
    print("=" * 80)
    print()

    underpredict_rows = [r for r in rows if r["weighted_avg_us"] < r["max_combination_us"]]
    overpredict_rows = [r for r in rows if r["weighted_avg_us"] >= r["max_combination_us"]]

    if underpredict_rows:
        under_ratios = [r["ratio_wa_to_max"] for r in underpredict_rows]
        under_pct_errors = [(1.0 - r["ratio_wa_to_max"]) for r in underpredict_rows]
        print(f"  Underprediction cases:  {len(underpredict_rows)}")
        print(f"  Avg underprediction:    {mean(under_pct_errors)*100:.2f}%")
        print(f"  Max underprediction:    {max(under_pct_errors)*100:.2f}%")
        print(f"  Min underprediction:    {min(under_pct_errors)*100:.2f}%")
    else:
        print("  No underprediction cases found.")

    if overpredict_rows:
        over_pct_errors = [(r["ratio_wa_to_max"] - 1.0) for r in overpredict_rows]
        print(f"\n  Overprediction cases:   {len(overpredict_rows)}")
        print(f"  Avg overprediction:     {mean(over_pct_errors)*100:.2f}%")
        print(f"  Max overprediction:     {max(over_pct_errors)*100:.2f}%")

    # ================================================================
    # Accept Criteria Evaluation
    # ================================================================
    print()
    print("=" * 80)
    print("  Accept Criteria Evaluation")
    print("=" * 80)
    print()

    # Criterion 1: Systematic underprediction (>50% of cases have WA < Max)
    underpredict_pct = len(underpredict_rows) / len(rows) * 100 if rows else 0
    c1_pass = underpredict_pct > 50
    print(f"  1. Systematic underprediction (>50% of cases):")
    print(f"     {len(underpredict_rows)}/{len(rows)} cases ({underpredict_pct:.1f}%)")
    print(f"     {'PASS' if c1_pass else 'FAIL'}: "
          f"{'majority' if c1_pass else 'minority'} of cases show underprediction")

    # Criterion 2: Underprediction concentrated in prefill-dominated regime
    prefill_dom = regime_groups.get("prefill-dominated", [])
    decode_dom = regime_groups.get("decode-dominated", [])
    if prefill_dom and decode_dom:
        pd_under = sum(1 for r in prefill_dom if r["weighted_avg_us"] < r["max_combination_us"])
        dd_under = sum(1 for r in decode_dom if r["weighted_avg_us"] < r["max_combination_us"])
        pd_pct = pd_under / len(prefill_dom) * 100
        dd_pct = dd_under / len(decode_dom) * 100
        c2_pass = pd_pct > dd_pct
        print(f"\n  2. Underprediction concentrated at prefill-dominated regime:")
        print(f"     Prefill-dominated: {pd_under}/{len(prefill_dom)} ({pd_pct:.1f}%)")
        print(f"     Decode-dominated:  {dd_under}/{len(decode_dom)} ({dd_pct:.1f}%)")
        print(f"     {'PASS' if c2_pass else 'FAIL'}: "
              f"{'prefill-dominated' if c2_pass else 'decode-dominated'} "
              f"shows more underprediction")
    else:
        print("\n  2. Regime comparison: UNABLE TO EVALUATE (missing regime data)")

    # Criterion 3: Average WA/Max ratio < 1.0 (underprediction on average)
    avg_ratio = mean(all_ratios)
    c3_pass = avg_ratio < 1.0
    print(f"\n  3. Average WA/Max ratio < 1.0 (systematic underprediction):")
    print(f"     Average ratio: {avg_ratio:.4f}")
    print(f"     {'PASS' if c3_pass else 'FAIL'}: "
          f"{'weighted-avg underpredicts' if c3_pass else 'weighted-avg does NOT systematically underpredict'}")

    # Criterion 4 (refutation check): synchronous-rate effect
    # At synchronous rate, there are no mixed batches, so the combination
    # method is irrelevant. We check that the effect is concentrated in
    # mixed-batch scenarios (all our test cases ARE mixed batches, so this
    # is about whether the effect size is meaningful enough to affect E2E).
    if underpredict_rows:
        avg_under_pct = mean([(1.0 - r["ratio_wa_to_max"]) for r in underpredict_rows]) * 100
        c4_note = avg_under_pct > 5.0
        print(f"\n  4. Effect size assessment:")
        print(f"     Avg underprediction magnitude: {avg_under_pct:.2f}%")
        if c4_note:
            print(f"     NOTE: >5% avg underprediction — likely impacts E2E MAPE")
        else:
            print(f"     NOTE: <5% avg underprediction — may be too small to affect E2E MAPE")
    else:
        print(f"\n  4. Effect size: N/A (no underprediction cases)")

    # Overall verdict
    print()
    print("=" * 80)
    print("  VERDICT")
    print("=" * 80)
    print()

    if c1_pass and c3_pass:
        print("  CONFIRMED: Weighted-average systematically underpredicts compared to max().")
        print("  The hypothesis that max(prefillTime, decodeTime) would produce higher")
        print("  latency estimates is supported. Next step: run E2E simulation to measure")
        print("  actual MAPE improvement.")
    else:
        print("  REFUTED: Weighted-average does NOT systematically underpredict compared")
        print("  to max(). The hypothesis is not supported at the step-level.")
        if avg_ratio > 1.0:
            print(f"  The weighted-average actually OVERPREDICTS (ratio={avg_ratio:.4f}).")
            print("  Switching to max() would WORSEN predictions.")

    print()


if __name__ == "__main__":
    main()
