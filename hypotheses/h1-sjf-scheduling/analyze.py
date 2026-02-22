#!/usr/bin/env python3
"""Analysis script for H1-SJF: SJF vs FCFS scheduling for mixed-length workloads.

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

    Expected format (cmd/root.go:566-568):
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

    Returns list of request dicts with slo_class, ttft_ms, e2e_ms,
    scheduling_delay_ms (NOTE: scheduling_delay_ms is in TICKS/us, not ms),
    num_prefill_tokens.
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


def analyze_aggregate(fcfs_files, sjf_files):
    """Compare aggregate cluster metrics between FCFS and SJF."""
    seeds = [42, 123, 456]

    print("Aggregate Cluster Metrics: FCFS vs SJF")
    print("-" * 80)
    print(f"{'Metric':<25} {'Seed':>6} {'FCFS':>12} {'SJF':>12} {'Diff%':>10}")
    print("-" * 80)

    metrics_to_compare = [
        ("ttft_mean_ms", "TTFT mean (ms)"),
        ("ttft_p99_ms", "TTFT P99 (ms)"),
        ("e2e_mean_ms", "E2E mean (ms)"),
        ("e2e_p99_ms", "E2E P99 (ms)"),
        ("scheduling_delay_p99_ms", "Sched Delay P99 (ms)"),
        ("completed_requests", "Completed"),
    ]

    for metric_key, metric_label in metrics_to_compare:
        for i, seed in enumerate(seeds):
            fcfs = parse_cluster_metrics(fcfs_files[i])
            sjf = parse_cluster_metrics(sjf_files[i])
            if fcfs is None or sjf is None:
                continue
            fval = fcfs.get(metric_key, 0)
            sval = sjf.get(metric_key, 0)
            diff = pct_diff(fval, sval)
            print(f"{metric_label:<25} {seed:>6} {fval:>12.2f} {sval:>12.2f} {diff:>+9.1f}%")
        print()


def analyze_per_slo(fcfs_files, sjf_files):
    """Compare per-SLO-class metrics between FCFS and SJF."""
    seeds = [42, 123, 456]

    print("Per-SLO-Class Metrics: FCFS vs SJF")
    print("-" * 90)
    print(f"{'SLO Class':<14} {'Metric':<16} {'Seed':>6} {'FCFS':>12} {'SJF':>12} {'Diff%':>10}")
    print("-" * 90)

    for i, seed in enumerate(seeds):
        fcfs_slo = parse_per_slo_metrics(fcfs_files[i])
        sjf_slo = parse_per_slo_metrics(sjf_files[i])

        all_classes = sorted(set(list(fcfs_slo.keys()) + list(sjf_slo.keys())))
        for cls in all_classes:
            f = fcfs_slo.get(cls, {})
            s = sjf_slo.get(cls, {})
            for metric_key, label in [
                ("ttft_mean", "TTFT mean"),
                ("ttft_p99", "TTFT P99"),
                ("e2e_mean", "E2E mean"),
                ("e2e_p99", "E2E P99"),
            ]:
                fval = f.get(metric_key, 0)
                sval = s.get(metric_key, 0)
                diff = pct_diff(fval, sval)
                print(f"{cls:<14} {label:<16} {seed:>6} {fval:>12.2f} {sval:>12.2f} {diff:>+9.1f}%")
            print()


def analyze_per_request(fcfs_json_files, sjf_json_files):
    """Analyze per-request data to understand scheduling effect by SLO class."""
    seeds = [42, 123, 456]

    print("Per-Request Analysis: TTFT by SLO Class (mean across requests)")
    print("-" * 80)
    print(f"{'SLO Class':<14} {'Seed':>6} {'FCFS TTFT':>12} {'SJF TTFT':>12} {'Diff%':>10} {'N':>6}")
    print("-" * 80)

    for i, seed in enumerate(seeds):
        fcfs_reqs = parse_per_request_json(fcfs_json_files[i])
        sjf_reqs = parse_per_request_json(sjf_json_files[i])

        for slo_class in ["interactive", "batch"]:
            fcfs_ttfts = [r["ttft_ms"] for r in fcfs_reqs if r.get("slo_class") == slo_class and r["ttft_ms"] > 0]
            sjf_ttfts = [r["ttft_ms"] for r in sjf_reqs if r.get("slo_class") == slo_class and r["ttft_ms"] > 0]

            fcfs_mean = sum(fcfs_ttfts) / len(fcfs_ttfts) if fcfs_ttfts else 0
            sjf_mean = sum(sjf_ttfts) / len(sjf_ttfts) if sjf_ttfts else 0
            diff = pct_diff(fcfs_mean, sjf_mean)
            n = len(fcfs_ttfts)
            print(f"{slo_class:<14} {seed:>6} {fcfs_mean:>12.2f} {sjf_mean:>12.2f} {diff:>+9.1f}% {n:>6}")
        print()

    # Also show scheduling delay comparison
    print("Per-Request Analysis: Scheduling Delay by SLO Class (mean, in us)")
    print("-" * 80)
    print(f"{'SLO Class':<14} {'Seed':>6} {'FCFS Delay':>12} {'SJF Delay':>12} {'Diff%':>10}")
    print("-" * 80)

    for i, seed in enumerate(seeds):
        fcfs_reqs = parse_per_request_json(fcfs_json_files[i])
        sjf_reqs = parse_per_request_json(sjf_json_files[i])

        for slo_class in ["interactive", "batch"]:
            # scheduling_delay_ms is actually in TICKS (microseconds)
            fcfs_delays = [r["scheduling_delay_ms"] for r in fcfs_reqs if r.get("slo_class") == slo_class]
            sjf_delays = [r["scheduling_delay_ms"] for r in sjf_reqs if r.get("slo_class") == slo_class]

            fcfs_mean = sum(fcfs_delays) / len(fcfs_delays) if fcfs_delays else 0
            sjf_mean = sum(sjf_delays) / len(sjf_delays) if sjf_delays else 0
            diff = pct_diff(fcfs_mean, sjf_mean)
            print(f"{slo_class:<14} {seed:>6} {fcfs_mean:>12.0f} {sjf_mean:>12.0f} {diff:>+9.1f}%")
        print()

    # E2E comparison by SLO class
    print("Per-Request Analysis: E2E by SLO Class (mean across requests, ms)")
    print("-" * 80)
    print(f"{'SLO Class':<14} {'Seed':>6} {'FCFS E2E':>12} {'SJF E2E':>12} {'Diff%':>10}")
    print("-" * 80)

    for i, seed in enumerate(seeds):
        fcfs_reqs = parse_per_request_json(fcfs_json_files[i])
        sjf_reqs = parse_per_request_json(sjf_json_files[i])

        for slo_class in ["interactive", "batch"]:
            fcfs_e2es = [r["e2e_ms"] for r in fcfs_reqs if r.get("slo_class") == slo_class and r["e2e_ms"] > 0]
            sjf_e2es = [r["e2e_ms"] for r in sjf_reqs if r.get("slo_class") == slo_class and r["e2e_ms"] > 0]

            fcfs_mean = sum(fcfs_e2es) / len(fcfs_e2es) if fcfs_e2es else 0
            sjf_mean = sum(sjf_e2es) / len(sjf_e2es) if sjf_e2es else 0
            diff = pct_diff(fcfs_mean, sjf_mean)
            print(f"{slo_class:<14} {seed:>6} {fcfs_mean:>12.2f} {sjf_mean:>12.2f} {diff:>+9.1f}%")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <mode> <files...>", file=sys.stderr)
        print("  Modes: aggregate, per_slo, per_request", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "aggregate":
        # 3 FCFS stdout files, 3 SJF stdout files
        if len(sys.argv) != 8:
            print("aggregate: need 3 FCFS + 3 SJF stdout files", file=sys.stderr)
            sys.exit(1)
        fcfs_files = sys.argv[2:5]
        sjf_files = sys.argv[5:8]
        analyze_aggregate(fcfs_files, sjf_files)

    elif mode == "per_slo":
        # 3 FCFS stdout files, 3 SJF stdout files
        if len(sys.argv) != 8:
            print("per_slo: need 3 FCFS + 3 SJF stdout files", file=sys.stderr)
            sys.exit(1)
        fcfs_files = sys.argv[2:5]
        sjf_files = sys.argv[5:8]
        analyze_per_slo(fcfs_files, sjf_files)

    elif mode == "per_request":
        # 3 FCFS JSON files, 3 SJF JSON files
        if len(sys.argv) != 8:
            print("per_request: need 3 FCFS + 3 SJF JSON result files", file=sys.stderr)
            sys.exit(1)
        fcfs_files = sys.argv[2:5]
        sjf_files = sys.argv[5:8]
        analyze_per_request(fcfs_files, sjf_files)

    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
