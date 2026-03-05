#!/usr/bin/env python3
"""Analysis script for h-perf-wallclock.

Reads wall-clock timing files (4 inputs) and produces comparison tables.

Usage: analyze.py <baseline_times> <pa_times> <lb_opt_times> <lb_base_times>
"""
import sys
import statistics
from pathlib import Path


def load_times(filepath):
    """Load wall-clock times from a file (one ms value per line)."""
    path = Path(filepath)
    if not path.exists():
        print(f"WARNING: missing {filepath}", file=sys.stderr)
        return []
    return [int(line.strip()) for line in path.read_text().strip().split("\n") if line.strip()]


def summarize(times, label):
    """Print summary statistics for a list of times."""
    if not times:
        print(f"  {label}: NO DATA")
        return {}
    med = statistics.median(times)
    mean = statistics.mean(times)
    mn = min(times)
    mx = max(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    print(f"  {label}:")
    print(f"    Median: {med:.0f}ms  Mean: {mean:.0f}ms  Min: {mn}ms  Max: {mx}ms  Stdev: {stdev:.0f}ms")
    print(f"    Runs: {times}")
    return {"median": med, "mean": mean, "min": mn, "max": mx, "stdev": stdev}


def main():
    if len(sys.argv) < 5:
        print("Usage: analyze.py <baseline_times> <pa_times> <lb_opt_times> <lb_base_times>",
              file=sys.stderr)
        sys.exit(1)

    baseline_times = load_times(sys.argv[1])
    pa_times = load_times(sys.argv[2])
    lb_opt_times = load_times(sys.argv[3])
    lb_base_times = load_times(sys.argv[4])

    print("=" * 60)
    print("  h-perf-wallclock: Wall-Clock Performance Results")
    print("=" * 60)
    print()

    print("Baseline (pre-optimization, PA enabled):")
    base_stats = summarize(baseline_times, "baseline")
    print()

    print("Optimized (PA enabled):")
    pa_stats = summarize(pa_times, "optimized")
    print()

    print("Negative control — LB-only optimized:")
    lb_opt_stats = summarize(lb_opt_times, "lb-only optimized")
    print()

    print("Negative control — LB-only baseline:")
    lb_base_stats = summarize(lb_base_times, "lb-only baseline")
    print()

    # Wall-clock reduction (PA path)
    if base_stats and pa_stats:
        base_med = base_stats["median"]
        pa_med = pa_stats["median"]
        reduction_pct = (1 - pa_med / base_med) * 100
        print("-" * 60)
        print(f"  Wall-clock reduction (PA path): {reduction_pct:.1f}%")
        print(f"    Baseline median: {base_med:.0f}ms -> Optimized median: {pa_med:.0f}ms")
        print()

        if reduction_pct >= 50:
            print(f"  H-perf-3 (compound >50%): PASS ({reduction_pct:.1f}%)")
        else:
            print(f"  H-perf-3 (compound >50%): FAIL ({reduction_pct:.1f}%)")
        print()

    # Negative control analysis
    if lb_opt_stats and lb_base_stats:
        lb_opt_med = lb_opt_stats["median"]
        lb_base_med = lb_base_stats["median"]
        lb_diff_pct = abs(1 - lb_opt_med / lb_base_med) * 100
        print(f"  Negative control (LB-only):")
        print(f"    Optimized median: {lb_opt_med:.0f}ms  Baseline median: {lb_base_med:.0f}ms")
        print(f"    Difference: {lb_diff_pct:.1f}%")
        if lb_diff_pct < 5:
            print(f"    Result: PASS (<5% difference confirms PA-specific bottleneck)")
        else:
            print(f"    Result: WARN (>{lb_diff_pct:.1f}% difference — investigate)")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
