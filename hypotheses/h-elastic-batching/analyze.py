#!/usr/bin/env python3
"""Analyze elastic batching experiment results.

Extracts per-SLO TTFT, batch occupancy, throughput, and preemption counts
from BLIS output. Produces the dual-objective comparison table.

Usage: python3 analyze.py <results_dir>
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout


def parse_per_slo_metrics(filepath):
    """Extract per-SLO-class TTFT from BLIS text output.

    Returns dict: {slo_class: {"ttft_mean": float, "ttft_p99": float, "e2e_mean": float, "e2e_p99": float}}
    """
    content = Path(filepath).read_text()
    slo_metrics = {}

    # Parse "=== Per-SLO Metrics ===" section
    # Format:
    #   critical:
    #     TTFT: mean=123.45 p99=678.90 (n=300)
    #     E2E:  mean=234.56 p99=789.01 (n=300)
    slo_pattern = re.compile(
        r"^\s+(\w+):\s*\n"
        r"\s+TTFT:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)\s*\n"
        r"\s+E2E:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)",
        re.MULTILINE,
    )
    for m in slo_pattern.finditer(content):
        cls = m.group(1)
        slo_metrics[cls] = {
            "ttft_mean": float(m.group(2)),
            "ttft_p99": float(m.group(3)),
            "ttft_n": int(m.group(4)),
            "e2e_mean": float(m.group(5)),
            "e2e_p99": float(m.group(6)),
            "e2e_n": int(m.group(7)),
        }
    return slo_metrics


def parse_batch_occupancy(filepath):
    """Extract batch occupancy from BLIS text output.

    Returns dict with avg_batch_occupancy and total_steps.
    """
    content = Path(filepath).read_text()
    result = {"avg_batch_occupancy": 0.0, "total_steps": 0}

    m = re.search(r"Avg Batch Occupancy:\s+([0-9.]+)", content)
    if m:
        result["avg_batch_occupancy"] = float(m.group(1))

    m = re.search(r"Total Steps:\s+(\d+)", content)
    if m:
        result["total_steps"] = int(m.group(1))

    # Also try JSON output
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        try:
            block = json.loads(match.group(1))
            if block.get("instance_id") == "cluster":
                if "avg_batch_occupancy" in block:
                    result["avg_batch_occupancy"] = block["avg_batch_occupancy"]
                if "total_steps" in block:
                    result["total_steps"] = block["total_steps"]
        except json.JSONDecodeError:
            continue

    return result


def analyze(results_dir):
    """Main analysis function."""
    results_path = Path(results_dir)
    configs = ["small-batch", "large-batch", "elastic", "elastic-adm", "fast-lane"]
    seeds = ["42", "123", "456"]

    # Collect metrics per config
    config_data = defaultdict(list)  # config -> list of seed dicts

    for config in configs:
        for seed in seeds:
            label = f"{config}_s{seed}"
            filepath = results_path / f"{label}.txt"

            if not filepath.exists():
                print(f"WARNING: missing {filepath}", file=sys.stderr)
                continue

            if check_for_timeout(str(filepath)):
                print(f"  SKIP: {label} (timeout/error)", file=sys.stderr)
                continue

            base = parse_blis_output(str(filepath))
            slo = parse_per_slo_metrics(str(filepath))
            occ = parse_batch_occupancy(str(filepath))

            entry = {
                "seed": seed,
                "ttft_p99": base["ttft_p99"],
                "e2e_p99": base["e2e_p99"],
                "throughput": base["throughput"],
                "completed": base["completed"],
                "preemption_count": base["preemption_count"],
                "avg_batch_occupancy": occ["avg_batch_occupancy"],
                "total_steps": occ["total_steps"],
            }

            # Per-SLO critical TTFT
            if "critical" in slo:
                entry["critical_ttft_p99"] = slo["critical"]["ttft_p99"]
                entry["critical_ttft_mean"] = slo["critical"]["ttft_mean"]
            else:
                entry["critical_ttft_p99"] = base["ttft_p99"]
                entry["critical_ttft_mean"] = base["ttft_mean"]

            config_data[config].append(entry)

    # Print summary table
    print("=" * 110)
    print("  DUAL-OBJECTIVE COMPARISON: Critical TTFT P99 (SLO) vs Batch Occupancy (GPU Utilization)")
    print("=" * 110)
    print()

    header = f"{'Config':<16} {'Crit TTFT P99':>14} {'Crit TTFT Mean':>15} {'Batch Occ':>10} {'Throughput':>11} {'Preemptions':>12} {'Completed':>10}"
    print(header)
    print("-" * len(header))

    summary = {}
    for config in configs:
        entries = config_data.get(config, [])
        if not entries:
            print(f"{config:<16} {'N/A':>14} {'N/A':>15} {'N/A':>10} {'N/A':>11} {'N/A':>12} {'N/A':>10}")
            continue

        # Compute means across seeds
        n = len(entries)
        avg_crit_p99 = sum(e["critical_ttft_p99"] for e in entries) / n
        avg_crit_mean = sum(e["critical_ttft_mean"] for e in entries) / n
        avg_occ = sum(e["avg_batch_occupancy"] for e in entries) / n
        avg_tput = sum(e["throughput"] for e in entries) / n
        total_preempt = sum(e["preemption_count"] for e in entries) / n
        avg_completed = sum(e["completed"] for e in entries) / n

        summary[config] = {
            "critical_ttft_p99": avg_crit_p99,
            "critical_ttft_mean": avg_crit_mean,
            "avg_batch_occupancy": avg_occ,
            "throughput": avg_tput,
            "preemption_count": total_preempt,
            "completed": avg_completed,
        }

        print(
            f"{config:<16} {avg_crit_p99:>13.2f}ms {avg_crit_mean:>14.2f}ms {avg_occ:>9.4f} {avg_tput:>10.1f}/s {total_preempt:>11.0f} {avg_completed:>9.0f}"
        )

    print()

    # Key comparisons
    if "elastic" in summary and "small-batch" in summary:
        print("=== Key Comparison: Elastic vs Small-batch ===")
        e = summary["elastic"]
        s = summary["small-batch"]
        if s["critical_ttft_p99"] > 0:
            ttft_ratio = e["critical_ttft_p99"] / s["critical_ttft_p99"]
            print(f"  Critical TTFT P99 ratio: {ttft_ratio:.2f}x (elastic/small-batch)")
        if s["avg_batch_occupancy"] > 0:
            occ_ratio = e["avg_batch_occupancy"] / s["avg_batch_occupancy"]
            print(f"  Batch occupancy ratio: {occ_ratio:.2f}x (elastic/small-batch)")
        print()

    if "elastic" in summary and "large-batch" in summary:
        print("=== Key Comparison: Elastic vs Large-batch ===")
        e = summary["elastic"]
        l = summary["large-batch"]
        if l["critical_ttft_p99"] > 0:
            ttft_ratio = e["critical_ttft_p99"] / l["critical_ttft_p99"]
            print(f"  Critical TTFT P99 ratio: {ttft_ratio:.2f}x (elastic/large-batch)")
        if l["avg_batch_occupancy"] > 0:
            occ_ratio = e["avg_batch_occupancy"] / l["avg_batch_occupancy"]
            print(f"  Batch occupancy ratio: {occ_ratio:.2f}x (elastic/large-batch)")
        print()

    if "elastic" in summary and "fast-lane" in summary:
        print("=== Key Comparison: Elastic vs Fast-lane ===")
        e = summary["elastic"]
        f = summary["fast-lane"]
        if f["critical_ttft_p99"] > 0:
            ttft_ratio = e["critical_ttft_p99"] / f["critical_ttft_p99"]
            print(f"  Critical TTFT P99 ratio: {ttft_ratio:.2f}x (elastic/fast-lane)")
        print()

    # Per-seed detail
    print("=== Per-Seed Detail ===")
    print()
    for config in configs:
        entries = config_data.get(config, [])
        if not entries:
            continue
        print(f"  {config}:")
        for e in entries:
            print(
                f"    seed={e['seed']}: crit_ttft_p99={e['critical_ttft_p99']:.2f}ms, "
                f"occ={e['avg_batch_occupancy']:.4f}, tput={e['throughput']:.1f}/s, "
                f"preemptions={e['preemption_count']}, completed={e['completed']}"
            )
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)
    analyze(sys.argv[1])
