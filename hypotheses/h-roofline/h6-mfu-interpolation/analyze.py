#!/usr/bin/env python3
"""H6: MFU Grid-Boundary Discontinuity Analysis

Analyzes CSV sweep data produced by mfu_sweep.go to quantify:
1. Number of >=5% discontinuities between adjacent parameter values
2. Max discontinuity magnitude
3. Flat regions (consecutive identical MFU values)
4. Unique MFU values vs total sweep points
"""

import csv
import os
import sys
from pathlib import Path


def load_csv(filepath, mfu_col="mfu", param_col=None):
    """Load a sweep CSV file and return (params, mfus) lists."""
    params = []
    mfus = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mfu = float(row[mfu_col])
            if param_col:
                params.append(int(row[param_col]))
            else:
                # Use first numeric column
                for k in reader.fieldnames:
                    if k != mfu_col:
                        try:
                            params.append(int(row[k]))
                            break
                        except (ValueError, TypeError):
                            continue
            mfus.append(mfu)
    return params, mfus


def analyze_discontinuities(params, mfus, threshold=0.05):
    """Count discontinuities >= threshold between adjacent points.

    A discontinuity is defined as:
        |mfu[i] - mfu[i-1]| / max(mfu[i], mfu[i-1]) >= threshold

    Returns dict with:
        total_points: number of sweep points
        unique_mfu: number of unique MFU values
        discontinuities_5pct: count of >=5% jumps
        discontinuities_10pct: count of >=10% jumps
        discontinuities_20pct: count of >=20% jumps
        max_discontinuity_pct: largest relative jump
        max_disc_location: param value where max jump occurs
        flat_regions: number of maximal runs of identical MFU
        longest_flat_run: longest run of identical MFU
        mfu_range: (min, max) of MFU values
    """
    n = len(mfus)
    if n < 2:
        return {"total_points": n, "error": "too few points"}

    disc_5 = 0
    disc_10 = 0
    disc_20 = 0
    max_disc = 0.0
    max_disc_loc = None

    for i in range(1, n):
        prev = mfus[i - 1]
        curr = mfus[i]
        denom = max(abs(prev), abs(curr))
        if denom < 1e-10:
            continue
        rel_change = abs(curr - prev) / denom
        if rel_change >= 0.05:
            disc_5 += 1
        if rel_change >= 0.10:
            disc_10 += 1
        if rel_change >= 0.20:
            disc_20 += 1
        if rel_change > max_disc:
            max_disc = rel_change
            max_disc_loc = params[i]

    # Count flat regions (consecutive identical values)
    flat_regions = 0
    current_run = 1
    longest_flat = 1
    for i in range(1, n):
        if mfus[i] == mfus[i - 1]:
            current_run += 1
        else:
            if current_run > 1:
                flat_regions += 1
            longest_flat = max(longest_flat, current_run)
            current_run = 1
    if current_run > 1:
        flat_regions += 1
    longest_flat = max(longest_flat, current_run)

    unique_mfu = len(set(mfus))
    non_zero = [m for m in mfus if m > 0]

    return {
        "total_points": n,
        "unique_mfu": unique_mfu,
        "unique_ratio": unique_mfu / n,
        "discontinuities_5pct": disc_5,
        "discontinuities_10pct": disc_10,
        "discontinuities_20pct": disc_20,
        "max_discontinuity_pct": max_disc * 100,
        "max_disc_location": max_disc_loc,
        "flat_regions": flat_regions,
        "longest_flat_run": longest_flat,
        "mfu_min": min(non_zero) if non_zero else 0,
        "mfu_max": max(mfus),
    }


def print_report(name, stats):
    """Print a formatted report for one sweep."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        return

    print(f"  Total sweep points:     {stats['total_points']}")
    print(f"  Unique MFU values:      {stats['unique_mfu']} ({stats['unique_ratio']:.1%})")
    print(f"  Discontinuities >=5%:   {stats['discontinuities_5pct']}")
    print(f"  Discontinuities >=10%:  {stats['discontinuities_10pct']}")
    print(f"  Discontinuities >=20%:  {stats['discontinuities_20pct']}")
    print(f"  Max discontinuity:      {stats['max_discontinuity_pct']:.1f}% at param={stats['max_disc_location']}")
    print(f"  Flat regions:           {stats['flat_regions']}")
    print(f"  Longest flat run:       {stats['longest_flat_run']} consecutive identical values")
    print(f"  MFU range:              [{stats['mfu_min']:.4f}, {stats['mfu_max']:.4f}]")


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    # Define sweep files and their parameter columns
    sweeps = [
        ("GEMM MFU (K=4096, N=6144, M=1..512)", "gemm_sweep.csv", "m"),
        ("GEMM MFU (K=4096, N=11008, M=1..512)", "gemm_sweep_mlp.csv", "m"),
        ("Decode Attn MFU (KV=1024, BS=1..256)", "decode_attn_kv1024_sweep.csv", "batch_size"),
        ("Decode Attn MFU (KV=4096, BS=1..256)", "decode_attn_kv4096_sweep.csv", "batch_size"),
        ("Decode Attn MFU (KV=8192, BS=1..256)", "decode_attn_kv8192_sweep.csv", "batch_size"),
        ("Decode Attn MFU (BS=1, KV=128..16384)", "decode_attn_bs1_kvsweep.csv", "kv_len"),
        ("Decode Attn MFU (BS=32, KV=128..16384)", "decode_attn_bs32_kvsweep.csv", "kv_len"),
        ("Decode Attn MFU (BS=128, KV=128..16384)", "decode_attn_bs128_kvsweep.csv", "kv_len"),
        ("Prefill Attn MFU (seq=512..32768)", "prefill_attn_sweep.csv", "seq_len"),
    ]

    all_stats = {}
    total_disc_5 = 0
    total_disc_10 = 0
    total_disc_20 = 0
    total_points = 0
    total_unique = 0

    print("=" * 60)
    print("  H6: MFU GRID-BOUNDARY DISCONTINUITY ANALYSIS")
    print("=" * 60)

    for name, filename, param_col in sweeps:
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            print(f"\nWARNING: {filepath} not found, skipping", file=sys.stderr)
            continue

        params, mfus = load_csv(filepath, "mfu", param_col)
        stats = analyze_discontinuities(params, mfus)
        all_stats[name] = stats
        print_report(name, stats)

        if "error" not in stats:
            total_disc_5 += stats["discontinuities_5pct"]
            total_disc_10 += stats["discontinuities_10pct"]
            total_disc_20 += stats["discontinuities_20pct"]
            total_points += stats["total_points"]
            total_unique += stats["unique_mfu"]

    # Summary
    print(f"\n{'=' * 60}")
    print("  AGGREGATE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total sweep points:          {total_points}")
    print(f"  Total unique MFU values:     {total_unique} ({total_unique/max(total_points,1):.1%})")
    print(f"  Total >=5% discontinuities:  {total_disc_5}")
    print(f"  Total >=10% discontinuities: {total_disc_10}")
    print(f"  Total >=20% discontinuities: {total_disc_20}")
    print(f"  Disc rate (>=5%):            {total_disc_5/max(total_points-len(all_stats),1):.2%} of adjacent pairs")

    # H6 accept criterion: >=80% of discontinuities could be eliminated by interpolation
    # For Part A, we just measure the discontinuity count
    print(f"\n  H6 Part A Accept Criterion: Count artificial discontinuities")
    print(f"  Result: {total_disc_5} discontinuities (>=5%) across {total_points} sweep points")

    # Identify the most problematic lookup type
    worst_name = max(all_stats, key=lambda k: all_stats[k].get("discontinuities_5pct", 0))
    worst = all_stats[worst_name]
    print(f"\n  Worst case: {worst_name}")
    print(f"    {worst['discontinuities_5pct']} discontinuities in {worst['total_points']} points")
    print(f"    Only {worst['unique_mfu']} unique MFU values ({worst['unique_ratio']:.1%})")
    print(f"    Max jump: {worst['max_discontinuity_pct']:.1f}%")

    # Print flat-region analysis
    print(f"\n  FLAT REGION ANALYSIS (step-function behavior):")
    for name, stats in all_stats.items():
        if "error" not in stats:
            print(f"    {name}: {stats['flat_regions']} flat regions, longest={stats['longest_flat_run']}")


if __name__ == "__main__":
    main()
