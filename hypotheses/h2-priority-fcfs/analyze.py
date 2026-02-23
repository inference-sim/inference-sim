#!/usr/bin/env python3
"""Analysis script for H2-Priority-FCFS: SLO-Based+Priority-FCFS vs Constant+FCFS.

Parses BLIS multi-block stdout output and per-request JSON results to produce
comparison tables.

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
- Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
- Cluster block has "instance_id": "cluster"
- Per-SLO metrics block:
    === Per-SLO Metrics ===
      <class>:
        TTFT: mean=<val> p99=<val> (n=<count>)
        E2E:  mean=<val> p99=<val> (n=<count>)
- Per-request JSON (via --results-path): has slo_class, ttft_ms, e2e_ms,
  scheduling_delay_ms (in TICKS/us, NOT ms), num_prefill_tokens
"""
import json
import re
import sys
from pathlib import Path

SEEDS = [42, 123, 456]
CONFIG_A = "Constant+FCFS"
CONFIG_B = "SLO-Based+PriFCFS"


def parse_cluster_metrics(filepath):
    """Parse BLIS stdout -> cluster-level JSON metrics."""
    content = Path(filepath).read_text()

    # Extract cluster-level JSON block (matches "instance_id": "cluster")
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{.*?\n\})", content, re.DOTALL
    ):
        try:
            block = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if block.get("instance_id") == "cluster":
            cluster = block

    if cluster is None:
        print(f"WARNING: no cluster metrics found in {filepath}", file=sys.stderr)
        return None
    return cluster


def parse_per_slo_metrics(filepath):
    """Parse Per-SLO Metrics from BLIS stdout.

    Expected format (cmd/root.go):
        === Per-SLO Metrics ===
          <class>:
            TTFT: mean=<val> p99=<val> (n=<count>)
            E2E:  mean=<val> p99=<val> (n=<count>)
    """
    content = Path(filepath).read_text()
    result = {}

    # Find the Per-SLO block
    slo_block = re.search(r"=== Per-SLO Metrics ===(.*?)(?:===|\Z)", content, re.DOTALL)
    if not slo_block:
        return result

    block_text = slo_block.group(1)

    # Parse each SLO class
    for m in re.finditer(
        r"^\s+(\S+):\s*\n"
        r"\s+TTFT:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)\s*\n"
        r"\s+E2E:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)",
        block_text,
        re.MULTILINE,
    ):
        cls = m.group(1).rstrip(":")
        result[cls] = {
            "ttft_mean": float(m.group(2)),
            "ttft_p99": float(m.group(3)),
            "ttft_n": int(m.group(4)),
            "e2e_mean": float(m.group(5)),
            "e2e_p99": float(m.group(6)),
            "e2e_n": int(m.group(7)),
        }
    return result


def parse_per_request_json(filepath):
    """Parse per-request JSON results file (from --results-path).

    Returns list of request dicts with slo_class, ttft_ms (in ms), e2e_ms (in ms),
    scheduling_delay_ms (NOTE: actually in TICKS/us, not ms).
    """
    content = Path(filepath).read_text()
    data = json.loads(content)
    requests = data.get("requests", [])
    if not requests:
        print(f"WARNING: no per-request data in {filepath}", file=sys.stderr)
    return requests


def pct_diff(a, b):
    """Percentage difference: (b - a) / a * 100. Positive = b is higher."""
    if a == 0:
        return float("inf") if b != 0 else 0.0
    return (b - a) / a * 100.0


def mean(vals):
    """Safe mean computation."""
    return sum(vals) / len(vals) if vals else 0.0


def percentile(vals, p):
    """Compute p-th percentile (0-100 scale)."""
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    rank = p / 100.0 * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def analyze_aggregate(a_files, b_files):
    """Compare aggregate cluster metrics between Config A and Config B."""
    print(f"Aggregate Cluster Metrics: {CONFIG_A} vs {CONFIG_B}")
    print("-" * 90)
    print(f"{'Metric':<25} {'Seed':>6} {CONFIG_A:>16} {CONFIG_B:>16} {'Diff%':>10}")
    print("-" * 90)

    metrics_to_compare = [
        ("ttft_mean_ms", "TTFT mean (ms)"),
        ("ttft_p99_ms", "TTFT P99 (ms)"),
        ("e2e_mean_ms", "E2E mean (ms)"),
        ("e2e_p99_ms", "E2E P99 (ms)"),
        ("scheduling_delay_p99_ms", "Sched Delay P99 (ms)"),
        ("completed_requests", "Completed"),
        ("responses_per_sec", "Responses/sec"),
    ]

    for metric_key, metric_label in metrics_to_compare:
        for i, seed in enumerate(SEEDS):
            a = parse_cluster_metrics(a_files[i])
            b = parse_cluster_metrics(b_files[i])
            if a is None or b is None:
                continue
            aval = a.get(metric_key, 0)
            bval = b.get(metric_key, 0)
            diff = pct_diff(aval, bval)
            print(f"{metric_label:<25} {seed:>6} {aval:>16.2f} {bval:>16.2f} {diff:>+9.1f}%")
        print()


def analyze_per_slo(a_files, b_files):
    """Compare per-SLO-class metrics between Config A and Config B."""
    print(f"Per-SLO-Class Metrics: {CONFIG_A} vs {CONFIG_B}")
    print("-" * 100)
    print(f"{'SLO Class':<14} {'Metric':<16} {'Seed':>6} {CONFIG_A:>16} {CONFIG_B:>16} {'Diff%':>10}")
    print("-" * 100)

    for i, seed in enumerate(SEEDS):
        a_slo = parse_per_slo_metrics(a_files[i])
        b_slo = parse_per_slo_metrics(b_files[i])

        all_classes = sorted(set(list(a_slo.keys()) + list(b_slo.keys())))
        for cls in all_classes:
            a = a_slo.get(cls, {})
            b = b_slo.get(cls, {})
            for metric_key, label in [
                ("ttft_mean", "TTFT mean"),
                ("ttft_p99", "TTFT P99"),
                ("e2e_mean", "E2E mean"),
                ("e2e_p99", "E2E P99"),
            ]:
                aval = a.get(metric_key, 0)
                bval = b.get(metric_key, 0)
                diff = pct_diff(aval, bval)
                print(f"{cls:<14} {label:<16} {seed:>6} {aval:>16.2f} {bval:>16.2f} {diff:>+9.1f}%")
            print()


def analyze_per_request(a_json_files, b_json_files):
    """Analyze per-request data to understand scheduling effect by SLO class."""
    slo_classes = ["realtime", "interactive", "batch"]

    # --- TTFT comparison ---
    print(f"Per-Request Analysis: TTFT by SLO Class (mean across requests, ms)")
    print("-" * 95)
    print(f"{'SLO Class':<14} {'Seed':>6} {CONFIG_A+' TTFT':>16} {CONFIG_B+' TTFT':>16} {'Diff%':>10} {'N':>6}")
    print("-" * 95)

    # Collect for cross-seed average
    ttft_by_class_a = {cls: [] for cls in slo_classes}
    ttft_by_class_b = {cls: [] for cls in slo_classes}

    for i, seed in enumerate(SEEDS):
        a_reqs = parse_per_request_json(a_json_files[i])
        b_reqs = parse_per_request_json(b_json_files[i])

        for slo_class in slo_classes:
            a_ttfts = [r["ttft_ms"] for r in a_reqs if r.get("slo_class") == slo_class and r["ttft_ms"] > 0]
            b_ttfts = [r["ttft_ms"] for r in b_reqs if r.get("slo_class") == slo_class and r["ttft_ms"] > 0]

            a_mean = mean(a_ttfts)
            b_mean = mean(b_ttfts)
            diff = pct_diff(a_mean, b_mean)
            n = len(a_ttfts)
            print(f"{slo_class:<14} {seed:>6} {a_mean:>16.2f} {b_mean:>16.2f} {diff:>+9.1f}% {n:>6}")

            ttft_by_class_a[slo_class].append(a_mean)
            ttft_by_class_b[slo_class].append(b_mean)
        print()

    # Cross-seed average
    print("  Cross-seed average:")
    for cls in slo_classes:
        a_avg = mean(ttft_by_class_a[cls])
        b_avg = mean(ttft_by_class_b[cls])
        diff = pct_diff(a_avg, b_avg)
        print(f"  {cls:<14} {'avg':>6} {a_avg:>16.2f} {b_avg:>16.2f} {diff:>+9.1f}%")
    print()

    # --- P99 TTFT comparison ---
    print(f"Per-Request Analysis: TTFT P99 by SLO Class (ms)")
    print("-" * 90)
    print(f"{'SLO Class':<14} {'Seed':>6} {CONFIG_A+' P99':>16} {CONFIG_B+' P99':>16} {'Diff%':>10}")
    print("-" * 90)

    for i, seed in enumerate(SEEDS):
        a_reqs = parse_per_request_json(a_json_files[i])
        b_reqs = parse_per_request_json(b_json_files[i])

        for slo_class in slo_classes:
            a_ttfts = [r["ttft_ms"] for r in a_reqs if r.get("slo_class") == slo_class and r["ttft_ms"] > 0]
            b_ttfts = [r["ttft_ms"] for r in b_reqs if r.get("slo_class") == slo_class and r["ttft_ms"] > 0]

            a_p99 = percentile(a_ttfts, 99)
            b_p99 = percentile(b_ttfts, 99)
            diff = pct_diff(a_p99, b_p99)
            print(f"{slo_class:<14} {seed:>6} {a_p99:>16.2f} {b_p99:>16.2f} {diff:>+9.1f}%")
        print()

    # --- Scheduling delay comparison ---
    print(f"Per-Request Analysis: Scheduling Delay by SLO Class (mean, in us)")
    print("-" * 90)
    print(f"{'SLO Class':<14} {'Seed':>6} {CONFIG_A+' Delay':>16} {CONFIG_B+' Delay':>16} {'Diff%':>10}")
    print("-" * 90)

    for i, seed in enumerate(SEEDS):
        a_reqs = parse_per_request_json(a_json_files[i])
        b_reqs = parse_per_request_json(b_json_files[i])

        for slo_class in slo_classes:
            # scheduling_delay_ms is actually in TICKS (microseconds)
            a_delays = [r["scheduling_delay_ms"] for r in a_reqs if r.get("slo_class") == slo_class]
            b_delays = [r["scheduling_delay_ms"] for r in b_reqs if r.get("slo_class") == slo_class]

            a_mean_val = mean(a_delays)
            b_mean_val = mean(b_delays)
            diff = pct_diff(a_mean_val, b_mean_val)
            print(f"{slo_class:<14} {seed:>6} {a_mean_val:>16.0f} {b_mean_val:>16.0f} {diff:>+9.1f}%")
        print()

    # --- E2E comparison ---
    print(f"Per-Request Analysis: E2E by SLO Class (mean across requests, ms)")
    print("-" * 90)
    print(f"{'SLO Class':<14} {'Seed':>6} {CONFIG_A+' E2E':>16} {CONFIG_B+' E2E':>16} {'Diff%':>10}")
    print("-" * 90)

    for i, seed in enumerate(SEEDS):
        a_reqs = parse_per_request_json(a_json_files[i])
        b_reqs = parse_per_request_json(b_json_files[i])

        for slo_class in slo_classes:
            a_e2es = [r["e2e_ms"] for r in a_reqs if r.get("slo_class") == slo_class and r["e2e_ms"] > 0]
            b_e2es = [r["e2e_ms"] for r in b_reqs if r.get("slo_class") == slo_class and r["e2e_ms"] > 0]

            a_mean_val = mean(a_e2es)
            b_mean_val = mean(b_e2es)
            diff = pct_diff(a_mean_val, b_mean_val)
            print(f"{slo_class:<14} {seed:>6} {a_mean_val:>16.2f} {b_mean_val:>16.2f} {diff:>+9.1f}%")
        print()


def analyze_priority(a_json_files, b_json_files):
    """Analyze priority score distribution and ordering effects."""
    slo_classes = ["realtime", "interactive", "batch"]

    print("Priority Analysis: Scheduling Order Differences")
    print("-" * 90)

    for i, seed in enumerate(SEEDS):
        a_reqs = parse_per_request_json(a_json_files[i])
        b_reqs = parse_per_request_json(b_json_files[i])

        print(f"\n  Seed {seed}: Request counts per SLO class")
        for cls in slo_classes:
            a_n = len([r for r in a_reqs if r.get("slo_class") == cls])
            b_n = len([r for r in b_reqs if r.get("slo_class") == cls])
            print(f"    {cls:<14} A: {a_n:>4}  B: {b_n:>4}")

    # Compare arrival order vs completion order to detect reordering
    print(f"\n\nScheduling Reordering Analysis:")
    print("-" * 90)
    print("  (Comparing request completion order between configs)")
    print(f"  A request is 'reordered' if it completes in a different relative position.")
    print()

    for i, seed in enumerate(SEEDS):
        a_reqs = parse_per_request_json(a_json_files[i])
        b_reqs = parse_per_request_json(b_json_files[i])

        # Build ID-to-rank maps by E2E completion time (arrival + e2e)
        def completion_rank(reqs):
            """Rank requests by completion time (arrived_at + e2e_ms)."""
            completed = [(r["requestID"], r.get("arrived_at", 0) + r.get("e2e_ms", 0))
                         for r in reqs if r.get("e2e_ms", 0) > 0]
            completed.sort(key=lambda x: x[1])
            return {rid: rank for rank, (rid, _) in enumerate(completed)}

        a_ranks = completion_rank(a_reqs)
        b_ranks = completion_rank(b_reqs)

        # Count how many requests changed rank significantly
        common_ids = set(a_ranks.keys()) & set(b_ranks.keys())
        if not common_ids:
            print(f"  Seed {seed}: no common completed requests")
            continue

        rank_diffs = []
        for rid in common_ids:
            rank_diffs.append(abs(a_ranks[rid] - b_ranks[rid]))

        reordered = sum(1 for d in rank_diffs if d > 0)
        significantly_reordered = sum(1 for d in rank_diffs if d > 10)
        max_shift = max(rank_diffs) if rank_diffs else 0
        mean_shift = mean(rank_diffs)

        print(f"  Seed {seed}: {len(common_ids)} common requests")
        print(f"    Reordered (any shift): {reordered} ({100*reordered/len(common_ids):.1f}%)")
        print(f"    Significantly reordered (>10 positions): {significantly_reordered} ({100*significantly_reordered/len(common_ids):.1f}%)")
        print(f"    Max rank shift: {max_shift}")
        print(f"    Mean rank shift: {mean_shift:.1f}")

    # Check if reordering benefits specific SLO classes
    print(f"\n\nPer-SLO Completion Rank Shift (positive = completed earlier with Config B):")
    print("-" * 90)
    print(f"{'SLO Class':<14} {'Seed':>6} {'Mean Shift':>14} {'Positive%':>12} {'N':>6}")
    print("-" * 90)

    for i, seed in enumerate(SEEDS):
        a_reqs = parse_per_request_json(a_json_files[i])
        b_reqs = parse_per_request_json(b_json_files[i])

        def completion_rank(reqs):
            completed = [(r["requestID"], r.get("arrived_at", 0) + r.get("e2e_ms", 0))
                         for r in reqs if r.get("e2e_ms", 0) > 0]
            completed.sort(key=lambda x: x[1])
            return {rid: rank for rank, (rid, _) in enumerate(completed)}

        a_ranks = completion_rank(a_reqs)
        b_ranks = completion_rank(b_reqs)
        common_ids = set(a_ranks.keys()) & set(b_ranks.keys())

        # Build ID -> SLO class map
        slo_map = {}
        for r in a_reqs:
            slo_map[r["requestID"]] = r.get("slo_class", "")

        for cls in slo_classes:
            shifts = []
            for rid in common_ids:
                if slo_map.get(rid) == cls:
                    # Positive shift = completed earlier with Config B (lower rank = earlier)
                    shifts.append(a_ranks[rid] - b_ranks[rid])

            if shifts:
                mean_shift = mean(shifts)
                positive_pct = 100 * sum(1 for s in shifts if s > 0) / len(shifts)
                print(f"{cls:<14} {seed:>6} {mean_shift:>+14.1f} {positive_pct:>11.1f}% {len(shifts):>6}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <mode> <files...>", file=sys.stderr)
        print("  Modes: aggregate, per_slo, per_request, priority", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "aggregate":
        if len(sys.argv) != 8:
            print("aggregate: need 3 Config A + 3 Config B stdout files", file=sys.stderr)
            sys.exit(1)
        a_files = sys.argv[2:5]
        b_files = sys.argv[5:8]
        analyze_aggregate(a_files, b_files)

    elif mode == "per_slo":
        if len(sys.argv) != 8:
            print("per_slo: need 3 Config A + 3 Config B stdout files", file=sys.stderr)
            sys.exit(1)
        a_files = sys.argv[2:5]
        b_files = sys.argv[5:8]
        analyze_per_slo(a_files, b_files)

    elif mode == "per_request":
        if len(sys.argv) != 8:
            print("per_request: need 3 Config A + 3 Config B JSON result files", file=sys.stderr)
            sys.exit(1)
        a_files = sys.argv[2:5]
        b_files = sys.argv[5:8]
        analyze_per_request(a_files, b_files)

    elif mode == "priority":
        if len(sys.argv) != 8:
            print("priority: need 3 Config A + 3 Config B JSON result files", file=sys.stderr)
            sys.exit(1)
        a_files = sys.argv[2:5]
        b_files = sys.argv[5:8]
        analyze_priority(a_files, b_files)

    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
