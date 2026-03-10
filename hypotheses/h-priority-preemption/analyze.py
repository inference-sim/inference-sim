#!/usr/bin/env python3
"""Analyze H-Priority-Preemption: Strategy Evolution Iteration 3.

Usage: python3 analyze.py <results_dir>

Parses BLIS output and per-SLO metrics to evaluate:
  H-main:               T3 vs B2 critical TTFT P99 improvement (>50%)
  H-zero-sum:           Cluster P99 within 20% of B2
  H-control-negative:   <5% difference with uniform SLO
"""

import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

# Add shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout


def parse_per_slo_metrics(filepath):
    """Parse per-SLO-class metrics from BLIS output.

    Returns dict: {slo_class: {"ttft_mean": float, "ttft_p99": float,
                                "e2e_mean": float, "e2e_p99": float, "n": int}}
    """
    result = {}
    path = Path(filepath)
    if not path.exists():
        return result

    content = path.read_text()

    # Parse "=== Per-SLO Metrics ===" section
    slo_section = re.search(
        r"=== Per-SLO Metrics ===\s*\n(.*?)(?:\n===|\Z)", content, re.DOTALL
    )
    if not slo_section:
        return result

    section_text = slo_section.group(1)

    # Parse each SLO class block
    class_pattern = re.compile(
        r"^\s+(\S+):\s*\n"
        r"\s+TTFT:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)\s*\n"
        r"\s+E2E:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)",
        re.MULTILINE,
    )

    for m in class_pattern.finditer(section_text):
        cls = m.group(1)
        result[cls] = {
            "ttft_mean": float(m.group(2)),
            "ttft_p99": float(m.group(3)),
            "ttft_n": int(m.group(4)),
            "e2e_mean": float(m.group(5)),
            "e2e_p99": float(m.group(6)),
            "e2e_n": int(m.group(7)),
        }

    return result


def pct_change(baseline, treatment):
    """Compute percentage change: (treatment - baseline) / baseline * 100.
    Negative means treatment is better (lower latency)."""
    if baseline == 0:
        return 0.0
    return (treatment - baseline) / baseline * 100.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    seeds = [42, 123, 456]

    print("=" * 72)
    print("  Strategy Evolution Iteration 3: Priority-Based Preemption Analysis")
    print("=" * 72)
    print()

    # -- H-main: T3 vs B2 ---------------------------------------------------
    print("=== H-main: T3 (priority preemption) vs B2 (baseline) ===")
    print()

    for rate_label in ["80", "120"]:
        print(f"  --- Rate: {rate_label}% capacity ---")
        crit_improvements = []
        shed_degradations = []
        cluster_changes = []
        preemption_counts = []

        for seed in seeds:
            b2_file = results_dir / f"B2_{rate_label}pct_s{seed}.txt"
            t3_file = results_dir / f"T3_{rate_label}pct_s{seed}.txt"

            b2 = parse_blis_output(b2_file)
            t3 = parse_blis_output(t3_file)
            b2_slo = parse_per_slo_metrics(b2_file)
            t3_slo = parse_per_slo_metrics(t3_file)

            if b2["timed_out"] or t3["timed_out"]:
                print(f"    seed={seed}: SKIPPED (timeout)")
                continue

            # Critical TTFT P99
            b2_crit_p99 = b2_slo.get("critical", {}).get("ttft_p99", 0)
            t3_crit_p99 = t3_slo.get("critical", {}).get("ttft_p99", 0)
            crit_change = pct_change(b2_crit_p99, t3_crit_p99)
            crit_improvements.append(-crit_change)  # positive = improvement

            # Sheddable TTFT P99
            b2_shed_p99 = b2_slo.get("sheddable", {}).get("ttft_p99", 0)
            t3_shed_p99 = t3_slo.get("sheddable", {}).get("ttft_p99", 0)
            shed_change = pct_change(b2_shed_p99, t3_shed_p99)
            shed_degradations.append(shed_change)  # positive = degradation

            # Cluster TTFT P99
            cluster_change = pct_change(b2["ttft_p99"], t3["ttft_p99"])
            cluster_changes.append(cluster_change)

            # Preemption count
            t3_preemptions = t3["preemption_count"]
            preemption_counts.append(t3_preemptions)

            print(f"    seed={seed}:")
            print(f"      Critical TTFT P99: B2={b2_crit_p99:.1f}ms, T3={t3_crit_p99:.1f}ms ({crit_change:+.1f}%)")
            print(f"      Sheddable TTFT P99: B2={b2_shed_p99:.1f}ms, T3={t3_shed_p99:.1f}ms ({shed_change:+.1f}%)")
            print(f"      Cluster  TTFT P99: B2={b2['ttft_p99']:.1f}ms, T3={t3['ttft_p99']:.1f}ms ({cluster_change:+.1f}%)")
            print(f"      Throughput: T3={t3['throughput']:.1f}, B2={b2['throughput']:.1f} rps")
            print(f"      Preemptions: T3={t3_preemptions}, B2={b2['preemption_count']}")
            print(f"      Completed: T3={t3['completed']}, B2={b2['completed']}")

        if crit_improvements:
            avg_crit = mean(crit_improvements)
            avg_shed = mean(shed_degradations)
            avg_cluster = mean(cluster_changes)
            avg_preemptions = mean(preemption_counts)
            print()
            print(f"    Mean critical TTFT P99 improvement: {avg_crit:+.1f}%")
            print(f"    Mean sheddable TTFT P99 degradation: {avg_shed:+.1f}%")
            print(f"    Mean cluster  TTFT P99 change: {avg_cluster:+.1f}%")
            print(f"    Mean preemption count: {avg_preemptions:.0f}")
            if rate_label == "120":
                print()
                if avg_crit > 50:
                    print(f"    H-main VERDICT: SUPPORTED ({avg_crit:.1f}% > 50% threshold)")
                elif avg_crit > 20:
                    print(f"    H-main VERDICT: PARTIAL ({avg_crit:.1f}% > 20% but < 50%)")
                else:
                    print(f"    H-main VERDICT: NOT SUPPORTED ({avg_crit:.1f}% < 20%)")
        print()

    # -- H-zero-sum: Cluster-wide impact -------------------------------------
    print("=== H-zero-sum: Cluster-wide P99 impact ===")
    print()

    cluster_120_changes = []
    for seed in seeds:
        b2_file = results_dir / f"B2_120pct_s{seed}.txt"
        t3_file = results_dir / f"T3_120pct_s{seed}.txt"
        b2 = parse_blis_output(b2_file)
        t3 = parse_blis_output(t3_file)

        if b2["timed_out"] or t3["timed_out"]:
            continue

        change = pct_change(b2["ttft_p99"], t3["ttft_p99"])
        cluster_120_changes.append(change)
        print(f"    seed={seed}: Cluster TTFT P99 B2={b2['ttft_p99']:.1f}ms T3={t3['ttft_p99']:.1f}ms ({change:+.1f}%)")

    if cluster_120_changes:
        avg = mean(cluster_120_changes)
        print()
        if abs(avg) < 20:
            print(f"    H-zero-sum VERDICT: SUPPORTED (|{avg:+.1f}%| < 20%)")
        else:
            print(f"    H-zero-sum VERDICT: NOT SUPPORTED (|{avg:+.1f}%| >= 20%)")
    print()

    # -- H-control-negative: Uniform SLO -----------------------------------
    print("=== H-control-negative: Uniform SLO (all 'standard') at 120% ===")
    print()

    uniform_diffs = []
    for seed in seeds:
        b2_file = results_dir / f"B2_uniform_120pct_s{seed}.txt"
        t3_file = results_dir / f"T3_uniform_120pct_s{seed}.txt"
        b2 = parse_blis_output(b2_file)
        t3 = parse_blis_output(t3_file)

        if b2["timed_out"] or t3["timed_out"]:
            print(f"    seed={seed}: SKIPPED (timeout)")
            continue

        diff = abs(pct_change(b2["ttft_p99"], t3["ttft_p99"]))
        uniform_diffs.append(diff)
        t3_preemptions = t3["preemption_count"]
        print(f"    seed={seed}: B2 TTFT P99={b2['ttft_p99']:.1f}ms, T3 TTFT P99={t3['ttft_p99']:.1f}ms (diff={diff:.1f}%), preemptions={t3_preemptions}")

    if uniform_diffs:
        avg_diff = mean(uniform_diffs)
        print()
        if avg_diff < 5:
            print(f"    H-control-negative VERDICT: SUPPORTED (mean diff {avg_diff:.1f}% < 5%)")
        else:
            print(f"    H-control-negative VERDICT: NOT SUPPORTED (mean diff {avg_diff:.1f}% >= 5%)")
    print()

    # -- Summary Table -------------------------------------------------------
    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    print()
    print("  Arm                      | Prediction              | Result")
    print("  -------------------------|-----------------------  |--------")

    # Recompute for summary
    crit_120 = []
    cluster_120 = []
    shed_120 = []
    for seed in seeds:
        b2 = parse_blis_output(results_dir / f"B2_120pct_s{seed}.txt")
        t3 = parse_blis_output(results_dir / f"T3_120pct_s{seed}.txt")
        b2_slo = parse_per_slo_metrics(results_dir / f"B2_120pct_s{seed}.txt")
        t3_slo = parse_per_slo_metrics(results_dir / f"T3_120pct_s{seed}.txt")
        if b2["timed_out"] or t3["timed_out"]:
            continue
        b2_c = b2_slo.get("critical", {}).get("ttft_p99", 0)
        t3_c = t3_slo.get("critical", {}).get("ttft_p99", 0)
        b2_s = b2_slo.get("sheddable", {}).get("ttft_p99", 0)
        t3_s = t3_slo.get("sheddable", {}).get("ttft_p99", 0)
        if b2_c > 0:
            crit_120.append(-(t3_c - b2_c) / b2_c * 100)
        cluster_120.append(pct_change(b2["ttft_p99"], t3["ttft_p99"]))
        if b2_s > 0:
            shed_120.append((t3_s - b2_s) / b2_s * 100)

    if crit_120:
        v = mean(crit_120)
        s = "SUPPORTED" if v > 50 else ("PARTIAL" if v > 20 else "NOT SUPPORTED")
        print(f"  H-main                   | >50% crit P99 imp    | {v:+.1f}% [{s}]")
    if cluster_120:
        v = mean(cluster_120)
        s = "SUPPORTED" if abs(v) < 20 else "NOT SUPPORTED"
        print(f"  H-zero-sum               | |cluster P99| < 20%  | {v:+.1f}% [{s}]")
    if shed_120:
        v = mean(shed_120)
        s = "OK" if v < 100 else "CONCERN"
        print(f"  Sheddable degradation    | <100% (2x)           | {v:+.1f}% [{s}]")
    if uniform_diffs:
        v = mean(uniform_diffs)
        s = "SUPPORTED" if v < 5 else "NOT SUPPORTED"
        print(f"  H-control-negative       | <5% diff uniform     | {v:.1f}% [{s}]")

    print()


if __name__ == "__main__":
    main()
