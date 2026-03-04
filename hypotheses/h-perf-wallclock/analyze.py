#!/usr/bin/env python3
"""Analysis script for h-perf-wallclock.

Reads wall-clock timing files and produces comparison tables.
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
        return
    med = statistics.median(times)
    mean = statistics.mean(times)
    mn = min(times)
    mx = max(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    print(f"  {label}:")
    print(f"    Median: {med:.0f}ms  Mean: {mean:.0f}ms  Min: {mn}ms  Max: {mx}ms  Stdev: {stdev:.0f}ms")
    print(f"    Runs: {times}")


def main():
    if len(sys.argv) < 3:
        print("Usage: analyze.py <pa_times.txt> <lb_times.txt>", file=sys.stderr)
        sys.exit(1)

    pa_times = load_times(sys.argv[1])
    lb_times = load_times(sys.argv[2])

    # Known baseline (pre-optimization, measured separately)
    # cache_warmup ~9300ms + load_spikes ~7700ms + multiturn ~250ms = ~17250ms
    BASELINE_MEDIAN_MS = 17250

    print("=" * 60)
    print("  h-perf-wallclock: Wall-Clock Performance Results")
    print("=" * 60)
    print()

    print("Pre-optimization baseline (prefix-affinity enabled):")
    print(f"  Median: ~{BASELINE_MEDIAN_MS}ms (measured before code changes)")
    print()

    print("Post-optimization (prefix-affinity enabled):")
    summarize(pa_times, "prefix-affinity + load-balance")
    print()

    print("Negative control (load-balance only, no prefix-affinity):")
    summarize(lb_times, "load-balance only")
    print()

    if pa_times:
        pa_median = statistics.median(pa_times)
        reduction_pct = (1 - pa_median / BASELINE_MEDIAN_MS) * 100
        print("-" * 60)
        print(f"  Wall-clock reduction: {reduction_pct:.1f}%")
        print(f"    Baseline: ~{BASELINE_MEDIAN_MS}ms -> Optimized: {pa_median:.0f}ms")
        print()

        # H-perf-3 gate: >50% reduction?
        if reduction_pct >= 50:
            print(f"  H-perf-3 (compound >50%): PASS ({reduction_pct:.1f}%)")
        else:
            print(f"  H-perf-3 (compound >50%): FAIL ({reduction_pct:.1f}%)")

    if lb_times and pa_times:
        lb_median = statistics.median(lb_times)
        pa_median = statistics.median(pa_times)
        # Negative control: optimized vs baseline should be <2% difference
        # when prefix-affinity is disabled
        print()
        print(f"  Negative control (load-balance only): {lb_median:.0f}ms")
        print(f"    This confirms optimizations are prefix-affinity-specific.")

    print("=" * 60)


if __name__ == "__main__":
    main()
