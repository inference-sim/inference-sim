#!/usr/bin/env python3
"""Analyze H-Heterogeneous-Pools: Strategy Evolution Iteration 5.

Usage: python3 analyze.py <results_dir>

Compares heterogeneous pool isolation vs shared-pool approaches:
  Sim A: Fast Lane (1 inst, critical only, maxRun=8)
  Sim B: Bulk Pool (3 inst, std+shed, maxRun=64)
  Sim C: Shared Baseline (4 inst, all SLO, maxRun=32)
  Sim D: Compound (4 inst, admission+preemption)

Hypotheses:
  H-main:       Fast lane critical TTFT P99 < 100ms (vs 500-1100ms shared)
  H-throughput:  Total throughput (A+B) within 10% of C (no sacrifice from splitting)
  H-bulk:       Standard/sheddable P99 in bulk pool within 50% of shared baseline
"""

import json
import os
import re
import sys
from pathlib import Path
from statistics import mean, stdev

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

    print("=" * 80)
    print("  Strategy Evolution Iteration 5: Heterogeneous Instance Pools")
    print("=" * 80)
    print()

    # -- Load all data ---------------------------------------------------------

    data = {}  # config -> seed -> {"cluster": dict, "slo": dict}
    for cfg in ["A", "B", "C", "D"]:
        data[cfg] = {}
        for seed in seeds:
            fname = results_dir / f"{cfg}_s{seed}.txt"
            cluster = parse_blis_output(fname)
            slo = parse_per_slo_metrics(fname)
            data[cfg][seed] = {"cluster": cluster, "slo": slo}

    # -- Raw Metrics Table -----------------------------------------------------

    print("=== Raw Metrics (all seeds) ===")
    print()
    print(f"  {'Config':<6} {'Seed':>4}  {'TTFT P99':>10}  {'TTFT mean':>10}  "
          f"{'E2E P99':>10}  {'Throughput':>10}  {'Completed':>9}  {'Rejected':>8}")
    print(f"  {'-'*6} {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*8}")

    for cfg in ["A", "B", "C", "D"]:
        for seed in seeds:
            d = data[cfg][seed]
            if d["cluster"]["timed_out"]:
                print(f"  {cfg:<6} {seed:>4}  {'TIMEOUT':>10}")
                continue
            c = d["cluster"]
            print(f"  {cfg:<6} {seed:>4}  {c['ttft_p99']:>10.1f}  {c['ttft_mean']:>10.1f}  "
                  f"{c['e2e_p99']:>10.1f}  {c['throughput']:>10.2f}  "
                  f"{c['completed']:>9}  {c['rejected']:>8}")
        print()

    # -- Per-SLO Metrics Table -------------------------------------------------

    print("=" * 80)
    print("=== Per-SLO Metrics (all seeds) ===")
    print()
    print(f"  {'Config':<6} {'Seed':>4}  {'SLO':>10}  {'TTFT mean':>10}  {'TTFT P99':>10}  "
          f"{'E2E mean':>10}  {'E2E P99':>10}  {'N':>6}")
    print(f"  {'-'*6} {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

    slo_order = ["critical", "standard", "sheddable"]
    for cfg in ["A", "B", "C", "D"]:
        for seed in seeds:
            d = data[cfg][seed]
            if d["cluster"]["timed_out"]:
                print(f"  {cfg:<6} {seed:>4}  {'TIMEOUT':>10}")
                continue
            for slo_cls in slo_order:
                slo = d["slo"].get(slo_cls, {})
                if slo:
                    print(f"  {cfg:<6} {seed:>4}  {slo_cls:>10}  "
                          f"{slo['ttft_mean']:>10.1f}  {slo['ttft_p99']:>10.1f}  "
                          f"{slo['e2e_mean']:>10.1f}  {slo['e2e_p99']:>10.1f}  "
                          f"{slo.get('ttft_n', 0):>6}")
        print()

    # =========================================================================
    # H-main: Fast lane critical TTFT P99 < 100ms
    # =========================================================================

    print("=" * 80)
    print("=== H-main: Fast Lane Critical TTFT P99 ===")
    print("  Prediction: Fast lane (Sim A) critical TTFT P99 < 100ms")
    print("  Comparison: vs shared baseline (Sim C) and compound (Sim D)")
    print()

    a_crit_p99s = []
    c_crit_p99s = []
    d_crit_p99s = []
    improvement_ratios = []

    for seed in seeds:
        a = data["A"][seed]
        c = data["C"][seed]
        d = data["D"][seed]

        if any(x["cluster"]["timed_out"] for x in [a, c, d]):
            print(f"  seed={seed}: SKIPPED (timeout)")
            continue

        # Sim A: single-instance → all requests are "critical" (from critical-only workload)
        # Cluster-level TTFT P99 IS the critical TTFT P99 for Sim A
        a_crit = a["slo"].get("critical", {}).get("ttft_p99", a["cluster"]["ttft_p99"])
        c_crit = c["slo"].get("critical", {}).get("ttft_p99", 0)
        d_crit = d["slo"].get("critical", {}).get("ttft_p99", 0)

        a_crit_p99s.append(a_crit)
        c_crit_p99s.append(c_crit)
        d_crit_p99s.append(d_crit)

        ratio_vs_c = c_crit / a_crit if a_crit > 0 else float("inf")
        ratio_vs_d = d_crit / a_crit if a_crit > 0 else float("inf")
        improvement_ratios.append(ratio_vs_c)

        print(f"  seed={seed}:")
        print(f"    A (fast lane)  critical TTFT P99 = {a_crit:>10.1f} ms")
        print(f"    C (shared)     critical TTFT P99 = {c_crit:>10.1f} ms  ({ratio_vs_c:.1f}x worse)")
        print(f"    D (compound)   critical TTFT P99 = {d_crit:>10.1f} ms  ({ratio_vs_d:.1f}x worse)")
        print()

    if a_crit_p99s:
        avg_a = mean(a_crit_p99s)
        avg_c = mean(c_crit_p99s)
        avg_d = mean(d_crit_p99s)
        avg_ratio = mean(improvement_ratios)

        print(f"  Mean fast lane critical TTFT P99:  {avg_a:.1f} ms")
        print(f"  Mean shared critical TTFT P99:     {avg_c:.1f} ms")
        print(f"  Mean compound critical TTFT P99:   {avg_d:.1f} ms")
        print(f"  Mean improvement ratio (C/A):      {avg_ratio:.1f}x")
        print()

        if avg_a < 100:
            print(f"  H-main VERDICT: SUPPORTED (fast lane = {avg_a:.1f}ms < 100ms; "
                  f"{avg_ratio:.1f}x better than shared)")
        else:
            print(f"  H-main VERDICT: NOT SUPPORTED (fast lane = {avg_a:.1f}ms >= 100ms)")
    print()

    # =========================================================================
    # H-throughput: Total throughput (A+B) within 10% of C
    # =========================================================================

    print("=" * 80)
    print("=== H-throughput: Total Throughput (A+B) vs C ===")
    print("  Prediction: Combined throughput within 10% of shared baseline")
    print()

    throughput_diffs = []

    for seed in seeds:
        a = data["A"][seed]
        b = data["B"][seed]
        c = data["C"][seed]

        if any(x["cluster"]["timed_out"] for x in [a, b, c]):
            print(f"  seed={seed}: SKIPPED (timeout)")
            continue

        a_tput = a["cluster"]["throughput"]
        b_tput = b["cluster"]["throughput"]
        c_tput = c["cluster"]["throughput"]
        combined = a_tput + b_tput

        # Also compare completed requests
        a_comp = a["cluster"]["completed"]
        b_comp = b["cluster"]["completed"]
        c_comp = c["cluster"]["completed"]
        combined_comp = a_comp + b_comp

        diff_pct = abs(pct_change(c_tput, combined))
        throughput_diffs.append(diff_pct)

        print(f"  seed={seed}:")
        print(f"    A throughput = {a_tput:.2f} rps  (completed: {a_comp})")
        print(f"    B throughput = {b_tput:.2f} rps  (completed: {b_comp})")
        print(f"    A+B combined = {combined:.2f} rps  (completed: {combined_comp})")
        print(f"    C throughput = {c_tput:.2f} rps  (completed: {c_comp})")
        print(f"    Difference:    {pct_change(c_tput, combined):+.1f}%")
        print()

    if throughput_diffs:
        avg_diff = mean(throughput_diffs)
        print(f"  Mean absolute throughput difference: {avg_diff:.1f}%")
        print()
        if avg_diff < 10:
            print(f"  H-throughput VERDICT: SUPPORTED (diff={avg_diff:.1f}% < 10%)")
        else:
            print(f"  H-throughput VERDICT: NOT SUPPORTED (diff={avg_diff:.1f}% >= 10%)")
    print()

    # =========================================================================
    # H-bulk: Bulk pool standard/sheddable P99 within 50% of shared baseline
    # =========================================================================

    print("=" * 80)
    print("=== H-bulk: Bulk Pool Metrics vs Shared Baseline ===")
    print("  Prediction: Std/shed TTFT P99 in bulk pool within 50% of shared")
    print()

    bulk_diffs_std = []
    bulk_diffs_shed = []

    for seed in seeds:
        b = data["B"][seed]
        c = data["C"][seed]

        if any(x["cluster"]["timed_out"] for x in [b, c]):
            print(f"  seed={seed}: SKIPPED (timeout)")
            continue

        b_std = b["slo"].get("standard", {}).get("ttft_p99", 0)
        b_shed = b["slo"].get("sheddable", {}).get("ttft_p99", 0)
        c_std = c["slo"].get("standard", {}).get("ttft_p99", 0)
        c_shed = c["slo"].get("sheddable", {}).get("ttft_p99", 0)

        diff_std = pct_change(c_std, b_std) if c_std > 0 else 0
        diff_shed = pct_change(c_shed, b_shed) if c_shed > 0 else 0

        if c_std > 0:
            bulk_diffs_std.append(abs(diff_std))
        if c_shed > 0:
            bulk_diffs_shed.append(abs(diff_shed))

        print(f"  seed={seed}:")
        print(f"    B (bulk) standard TTFT P99  = {b_std:>10.1f} ms  (vs C: {diff_std:+.1f}%)")
        print(f"    C (shared) standard TTFT P99 = {c_std:>10.1f} ms")
        print(f"    B (bulk) sheddable TTFT P99  = {b_shed:>10.1f} ms  (vs C: {diff_shed:+.1f}%)")
        print(f"    C (shared) sheddable TTFT P99 = {c_shed:>10.1f} ms")
        print()

    if bulk_diffs_std or bulk_diffs_shed:
        avg_std = mean(bulk_diffs_std) if bulk_diffs_std else 0
        avg_shed = mean(bulk_diffs_shed) if bulk_diffs_shed else 0
        max_diff = max(avg_std, avg_shed)

        print(f"  Mean |diff| standard:  {avg_std:.1f}%")
        print(f"  Mean |diff| sheddable: {avg_shed:.1f}%")
        print()
        if max_diff < 50:
            print(f"  H-bulk VERDICT: SUPPORTED (max diff={max_diff:.1f}% < 50%)")
        else:
            print(f"  H-bulk VERDICT: NOT SUPPORTED (max diff={max_diff:.1f}% >= 50%)")
    print()

    # =========================================================================
    # Cross-strategy comparison: Fast Lane vs Compound vs Shared
    # =========================================================================

    print("=" * 80)
    print("=== Cross-Strategy Critical TTFT P99 Comparison ===")
    print("  The key question: Is physical isolation worth more than queue management?")
    print()

    for seed in seeds:
        a = data["A"][seed]
        c = data["C"][seed]
        d = data["D"][seed]

        if any(x["cluster"]["timed_out"] for x in [a, c, d]):
            print(f"  seed={seed}: SKIPPED (timeout)")
            continue

        a_crit = a["slo"].get("critical", {}).get("ttft_p99", a["cluster"]["ttft_p99"])
        c_crit = c["slo"].get("critical", {}).get("ttft_p99", 0)
        d_crit = d["slo"].get("critical", {}).get("ttft_p99", 0)

        # Rank from best to worst
        strategies = [
            ("A (Fast Lane)", a_crit),
            ("C (Shared)", c_crit),
            ("D (Compound)", d_crit),
        ]
        strategies.sort(key=lambda x: x[1])

        print(f"  seed={seed} ranking:")
        for rank, (name, val) in enumerate(strategies, 1):
            print(f"    {rank}. {name:<20} = {val:.1f} ms")
        print()

    # =========================================================================
    # Preemption and Conservation Summary
    # =========================================================================

    print("=" * 80)
    print("=== Preemption and Conservation Summary ===")
    print()
    print(f"  {'Config':<6} {'Seed':>4}  {'Preemptions':>12}  {'Preempt Rate':>13}  "
          f"{'Rejected':>8}  {'Completed':>9}  {'Injected':>8}")
    print(f"  {'-'*6} {'-'*4}  {'-'*12}  {'-'*13}  {'-'*8}  {'-'*9}  {'-'*8}")

    for cfg in ["A", "B", "C", "D"]:
        for seed in seeds:
            d = data[cfg][seed]
            if d["cluster"]["timed_out"]:
                print(f"  {cfg:<6} {seed:>4}  {'TIMEOUT':>12}")
                continue
            c = d["cluster"]
            print(f"  {cfg:<6} {seed:>4}  {c['preemption_count']:>12}  "
                  f"{c['preemption_rate']:>12.4f}  "
                  f"{c['rejected']:>8}  {c['completed']:>9}  {c['injected']:>8}")
        print()

    # =========================================================================
    # Summary Table
    # =========================================================================

    print("=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"  {'Hypothesis':<30} | {'Prediction':<30} | {'Result':<35}")
    print(f"  {'-'*30}-+-{'-'*30}-+-{'-'*35}")

    # H-main
    if a_crit_p99s:
        avg_a = mean(a_crit_p99s)
        avg_ratio = mean(improvement_ratios)
        s = "SUPPORTED" if avg_a < 100 else "NOT SUPPORTED"
        print(f"  {'H-main (Isolation)':<30} | {'Crit P99 < 100ms':<30} | "
              f"{avg_a:.1f}ms, {avg_ratio:.1f}x vs shared [{s}]")

    # H-throughput
    if throughput_diffs:
        avg_diff = mean(throughput_diffs)
        s = "SUPPORTED" if avg_diff < 10 else "NOT SUPPORTED"
        print(f"  {'H-throughput':<30} | {'(A+B) within 10% of C':<30} | "
              f"diff={avg_diff:.1f}% [{s}]")

    # H-bulk
    if bulk_diffs_std or bulk_diffs_shed:
        max_d = max(
            mean(bulk_diffs_std) if bulk_diffs_std else 0,
            mean(bulk_diffs_shed) if bulk_diffs_shed else 0,
        )
        s = "SUPPORTED" if max_d < 50 else "NOT SUPPORTED"
        print(f"  {'H-bulk':<30} | {'Bulk P99 within 50% of C':<30} | "
              f"max_diff={max_d:.1f}% [{s}]")

    print()


if __name__ == "__main__":
    main()
