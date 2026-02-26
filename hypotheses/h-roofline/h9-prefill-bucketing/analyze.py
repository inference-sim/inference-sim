#!/usr/bin/env python3
"""H29: Prefill Bucketing Overestimates Short Sequences -- Analysis.

Parses CSV output from the Go experiment tests and determines whether
the power-of-2 bucketing causes latency overestimates exceeding 2x
for short prefill sequences (seqLen <= 100).

Key questions:
  1. Does bucketing cause seqLen=50 to have similar step time as seqLen=512?
  2. Is the overestimation ratio > 2x for seqLen <= 100?
  3. Are short-prefill steps compute-bound or memory-bound?
  4. Does the bucketed attention FLOPs dominate the step time at short seqLen?

Usage: python3 analyze.py <output_dir>
"""

import csv
import sys
from pathlib import Path


def load_csv(filepath):
    """Load CSV file into list of dicts."""
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def print_table(title, headers, rows, col_widths=None):
    """Print a formatted text table."""
    if col_widths is None:
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]

    print(f"\n{'=' * sum(col_widths)}")
    print(title)
    print('=' * sum(col_widths))
    header_line = "".join(h.rjust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        line = "".join(str(v).rjust(w) for v, w in zip(row, col_widths))
        print(line)


def analyze_sweep(output_dir):
    """Analyze the full seqLen sweep."""
    filepath = Path(output_dir) / "bucketing_sweep.csv"
    if not filepath.exists():
        print(f"WARNING: {filepath} not found, skipping sweep analysis")
        return

    rows = load_csv(str(filepath))

    print("\n" + "=" * 80)
    print("TABLE 1: Prefill Bucketing Sweep (seqLen -> step time)")
    print("=" * 80)
    print(f"{'seqLen':>8} {'bucket':>8} {'step(us)':>10} {'compute(us)':>12} "
          f"{'memory(us)':>12} {'regime':>15}")
    print("-" * 80)

    for row in rows:
        print(f"{row['seq_len']:>8} {row['bucket']:>8} {row['step_time_us']:>10} "
              f"{row['compute_only_us']:>12} {row['memory_only_us']:>12} "
              f"{row['regime']:>15}")

    # Identify bucket transitions
    print("\n--- Bucket Transition Analysis ---")
    for i in range(1, len(rows)):
        prev_bucket = int(rows[i-1]['bucket'])
        curr_bucket = int(rows[i]['bucket'])
        if curr_bucket != prev_bucket:
            prev_time = int(rows[i-1]['step_time_us'])
            curr_time = int(rows[i]['step_time_us'])
            prev_seq = int(rows[i-1]['seq_len'])
            curr_seq = int(rows[i]['seq_len'])
            jump_pct = (curr_time - prev_time) / prev_time * 100 if prev_time > 0 else 0
            print(f"  seqLen {prev_seq}->{curr_seq}: bucket {prev_bucket}->{curr_bucket}, "
                  f"step time {prev_time}->{curr_time} us (jump: {jump_pct:+.1f}%)")


def analyze_bucket_comparison(output_dir):
    """Analyze same-bucket comparisons."""
    filepath = Path(output_dir) / "bucket_comparison.csv"
    if not filepath.exists():
        print(f"WARNING: {filepath} not found, skipping comparison analysis")
        return

    rows = load_csv(str(filepath))

    print("\n" + "=" * 80)
    print("TABLE 2: Bucket Boundary Comparison")
    print("  Shows: actual seqLen vs bucket-top seqLen step times")
    print("  Ratio < 1.0 means actual is faster (GEMM/memory use real tokens)")
    print("  Ratio near 1.0 would mean bucketing dominates (attention uses bucket)")
    print("=" * 80)
    print(f"{'actual':>8} {'bucket':>8} {'time(act)':>10} {'time(bkt)':>10} "
          f"{'ratio':>8} {'overest%':>10}")
    print("-" * 65)

    for row in rows:
        print(f"{row['actual_seq_len']:>8} {row['bucket_top']:>8} "
              f"{row['step_actual_us']:>10} {row['step_bucket_top_us']:>10} "
              f"{row['ratio']:>8} {row['overestimate_pct']:>10}")

    # Key finding: ratio < 1.0 means actual seqLen is faster than bucket top
    print("\n--- Key Finding ---")
    for row in rows:
        actual = int(row['actual_seq_len'])
        ratio = float(row['ratio'])
        if actual <= 100:
            savings_pct = (1 - ratio) * 100
            print(f"  seqLen={actual}: {savings_pct:.1f}% faster than bucket-top "
                  f"(ratio={ratio:.3f})")
            print(f"    -> GEMM (uses actual tokens) and memory (uses actual) "
                  f"partially offset attention bucketing")


def analyze_decomposition(output_dir):
    """Analyze within-bucket decomposition."""
    filepath = Path(output_dir) / "within_bucket_decomposition.csv"
    if not filepath.exists():
        print(f"WARNING: {filepath} not found, skipping decomposition analysis")
        return

    rows = load_csv(str(filepath))

    print("\n" + "=" * 80)
    print("TABLE 3: Within-Bucket Decomposition (all seqLen -> bucket 512)")
    print("  flatness_vs_512: 1.0 = same as seqLen=512, lower = faster")
    print("  If attention bucketing dominates, flatness should be ~1.0 for all")
    print("=" * 80)
    print(f"{'seqLen':>8} {'step(us)':>10} {'compute':>10} {'memory':>10} "
          f"{'regime':>10} {'flatness':>10}")
    print("-" * 65)

    for row in rows:
        print(f"{row['seq_len']:>8} {row['step_time_us']:>10} "
              f"{row['compute_only_us']:>10} {row['memory_only_us']:>10} "
              f"{row['gemm_compute_regime']:>10} {row['flatness_vs_512']:>10}")

    # Check if very short seqLens are memory-bound (refutation condition)
    print("\n--- Regime Analysis ---")
    memory_bound_short = []
    compute_bound_short = []
    for row in rows:
        seqLen = int(row['seq_len'])
        if seqLen <= 100:
            if row['gemm_compute_regime'] == 'memory':
                memory_bound_short.append(seqLen)
            else:
                compute_bound_short.append(seqLen)

    if memory_bound_short:
        print(f"  Memory-bound for seqLen <= 100: {memory_bound_short}")
        print(f"  -> For these, attention overestimate is absorbed by max(compute, memory)")
    if compute_bound_short:
        print(f"  Compute-bound for seqLen <= 100: {compute_bound_short}")
        print(f"  -> For these, attention bucketing directly affects step time")


def analyze_overestimation(output_dir):
    """Analyze overestimation ratios -- the core hypothesis test."""
    filepath = Path(output_dir) / "overestimation_ratio.csv"
    if not filepath.exists():
        print(f"WARNING: {filepath} not found, skipping overestimation analysis")
        return

    rows = load_csv(str(filepath))

    print("\n" + "=" * 80)
    print("TABLE 4: Overestimation Ratio (vs linear token-count scaling)")
    print("  Linear expected = step_time(512) * seqLen/512")
    print("  Overestimation ratio = actual / linear_expected")
    print("  HYPOTHESIS: ratio > 2x for seqLen <= 100")
    print("=" * 80)
    print(f"{'seqLen':>8} {'actual(us)':>12} {'linear(us)':>12} "
          f"{'ratio':>8} {'> 2x?':>8}")
    print("-" * 55)

    exceeds_2x_at_100 = False
    exceeds_2x_count = 0
    total_below_200 = 0

    for row in rows:
        seqLen = int(row['seq_len'])
        ratio = float(row['overestimation_ratio'])
        exceeds = row['exceeds_2x']

        marker = " ***" if exceeds == "true" else ""
        print(f"{seqLen:>8} {row['step_time_us']:>12} {row['linear_expected_us']:>12} "
              f"{ratio:>8.2f} {exceeds:>8}{marker}")

        if exceeds == "true":
            exceeds_2x_count += 1
        if seqLen <= 200:
            total_below_200 += 1
        if seqLen == 100:
            exceeds_2x_at_100 = (exceeds == "true")

    return exceeds_2x_at_100, exceeds_2x_count


def analyze_compute_vs_memory(output_dir):
    """Analyze compute vs memory bound regime."""
    filepath = Path(output_dir) / "compute_vs_memory.csv"
    if not filepath.exists():
        print(f"WARNING: {filepath} not found, skipping regime analysis")
        return False

    rows = load_csv(str(filepath))

    print("\n" + "=" * 80)
    print("TABLE 5: Compute vs Memory Bound Regime")
    print("  compute_to_memory_ratio > 1: compute-bound (attention FLOPs dominate)")
    print("  compute_to_memory_ratio < 1: memory-bound (bandwidth dominates)")
    print("=" * 80)
    print(f"{'seqLen':>8} {'compute(us)':>12} {'memory(us)':>12} "
          f"{'C/M ratio':>10} {'regime':>15}")
    print("-" * 65)

    short_memory_bound = True  # tracks if ALL seqLen <= 100 are memory-bound
    for row in rows:
        seqLen = int(row['seq_len'])
        print(f"{seqLen:>8} {row['compute_time_us']:>12} {row['memory_time_us']:>12} "
              f"{row['compute_to_memory_ratio']:>10} {row['regime']:>15}")

        if seqLen <= 100 and row['regime'] != 'memory-bound':
            short_memory_bound = False

    return short_memory_bound


def render_ascii_chart(output_dir):
    """Render a simple ASCII chart of step time vs seqLen."""
    filepath = Path(output_dir) / "bucketing_sweep.csv"
    if not filepath.exists():
        return

    rows = load_csv(str(filepath))

    print("\n" + "=" * 80)
    print("CHART: Step Time vs Sequence Length (ASCII)")
    print("  Each '#' = ~500 us | '*' marks bucket boundaries")
    print("=" * 80)

    max_time = max(int(r['step_time_us']) for r in rows)
    scale = max_time / 60  # Aim for ~60 char wide

    for row in rows:
        seqLen = int(row['seq_len'])
        bucket = int(row['bucket'])
        step_time = int(row['step_time_us'])

        bar_len = int(step_time / scale)
        is_boundary = (seqLen == bucket)
        marker = '*' if is_boundary else '#'

        label = f"  {seqLen:>5} [{bucket:>5}]"
        bar = marker * bar_len
        time_str = f" {step_time:>6} us"
        print(f"{label} |{bar}{time_str}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = sys.argv[1]

    print("=" * 80)
    print("H29: Prefill Bucketing Overestimates Short Sequences")
    print("=" * 80)
    print()
    print("HYPOTHESIS: Power-of-2 bucketing (min=512) causes attention FLOPs")
    print("overestimate for short prefills, producing >2x latency overestimate")
    print("for seqLen <= 100.")
    print()
    print("REFUTED IF: Bucketed latency within 2x for seqLen=100, OR short")
    print("prefills are memory-bound under both calculations.")

    # Run all analyses
    analyze_sweep(output_dir)
    analyze_bucket_comparison(output_dir)
    analyze_decomposition(output_dir)

    overestimation_result = analyze_overestimation(output_dir)
    exceeds_2x_at_100 = overestimation_result[0] if overestimation_result else None
    exceeds_2x_count = overestimation_result[1] if overestimation_result else 0

    short_all_memory_bound = analyze_compute_vs_memory(output_dir)
    render_ascii_chart(output_dir)

    # === VERDICT ===
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Check refutation conditions
    refuted = False
    confirmed = False

    print("\nRefutation condition 1: Bucketed latency within 2x for seqLen=100")
    if exceeds_2x_at_100 is not None:
        if not exceeds_2x_at_100:
            print("  -> REFUTED: seqLen=100 overestimation is within 2x")
            refuted = True
        else:
            print("  -> NOT REFUTED: seqLen=100 overestimation exceeds 2x")
    else:
        print("  -> INCONCLUSIVE: overestimation data not available")

    print("\nRefutation condition 2: Short-prefill steps are memory-bound")
    if short_all_memory_bound is not None:
        if short_all_memory_bound:
            print("  -> REFUTED: all seqLen <= 100 are memory-bound")
            print("     Attention FLOPs overestimate is absorbed by max(compute, memory)")
            refuted = True
        else:
            print("  -> NOT REFUTED: some seqLen <= 100 are compute-bound")
            print("     Attention bucketing directly affects step time")
    else:
        print("  -> INCONCLUSIVE: regime data not available")

    print()
    if refuted:
        print("OVERALL: REFUTED")
        print("  At least one refutation condition is met.")
    elif exceeds_2x_at_100:
        print("OVERALL: CONFIRMED")
        print(f"  Overestimation exceeds 2x for seqLen <= 100.")
        print(f"  Short prefills are compute-bound, so bucketing directly impacts latency.")
        print(f"  Total seqLen values with > 2x overestimation: {exceeds_2x_count}")
        confirmed = True
    else:
        print("OVERALL: INCONCLUSIVE")
        print("  Data insufficient to confirm or refute.")

    print()
    print("NOTE: Overestimation ratio uses linear token-count scaling as the")
    print("'unbucketed' reference. This is an approximation since the actual")
    print("unbucketed calculation would also change MFU lookups and attention")
    print("FLOPs non-linearly. The analysis captures the dominant effect.")

    return 0 if not confirmed else 1


if __name__ == "__main__":
    sys.exit(main())
