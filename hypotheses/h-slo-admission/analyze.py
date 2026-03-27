#!/usr/bin/env python3
"""Analyze H-SLO-Admission: Strategy Evolution Iteration 2.

Usage: python3 analyze.py <results_dir>

Parses BLIS output and per-SLO metrics to evaluate:
  H-main:               T2 vs B2 critical TTFT P99 improvement
  H-zero-sum-broken:    Cluster P99 improvement (non-zero-sum)
  H-control-negative:   <5% difference with uniform SLO
  H-threshold-sensitivity: >15% improvement across thresholds
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

    Note: Per-SLO output from BLIS is in TICKS (microseconds). We convert to ms
    here to match cluster JSON output (which uses _ms suffix fields).
    """
    result = {}
    path = Path(filepath)
    if not path.exists():
        return result

    content = path.read_text()

    # Parse "=== Per-SLO Metrics ===" section
    # Format:
    #   === Per-SLO Metrics ===
    #     critical:
    #       TTFT: mean=1234.56 p99=5678.90 (n=300)
    #       E2E:  mean=2345.67 p99=6789.01 (n=300)
    #     standard:
    #       ...
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
        # Convert ticks (us) to ms
        result[cls] = {
            "ttft_mean": float(m.group(2)) / 1000.0,
            "ttft_p99": float(m.group(3)) / 1000.0,
            "ttft_n": int(m.group(4)),
            "e2e_mean": float(m.group(5)) / 1000.0,
            "e2e_p99": float(m.group(6)) / 1000.0,
            "e2e_n": int(m.group(7)),
        }

    return result


def parse_rejected(filepath):
    """Parse rejected request count from BLIS output."""
    path = Path(filepath)
    if not path.exists():
        return 0
    content = path.read_text()
    m = re.search(r"Rejected Requests:\s*(\d+)", content)
    if m:
        return int(m.group(1))
    return 0


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
    print("  Strategy Evolution Iteration 2: SLO-Gated Admission Analysis")
    print("=" * 72)
    print()

    # ── H-main: T2 vs B2 ────────────────────────────────────────────────
    print("=== H-main: T2 (slo-gated) vs B2 (always-admit) ===")
    print()

    for rate_label in ["80", "120"]:
        print(f"  --- Rate: {rate_label}% capacity ---")
        crit_improvements = []
        cluster_improvements = []

        for seed in seeds:
            b2_file = results_dir / f"B2_{rate_label}pct_s{seed}.txt"
            t2_file = results_dir / f"T2_{rate_label}pct_s{seed}.txt"

            b2 = parse_blis_output(b2_file)
            t2 = parse_blis_output(t2_file)
            b2_slo = parse_per_slo_metrics(b2_file)
            t2_slo = parse_per_slo_metrics(t2_file)
            t2_rejected = parse_rejected(t2_file)
            b2_rejected = parse_rejected(b2_file)

            if b2["timed_out"] or t2["timed_out"]:
                print(f"    seed={seed}: SKIPPED (timeout)")
                continue

            # Critical TTFT P99
            b2_crit_p99 = b2_slo.get("critical", {}).get("ttft_p99", 0)
            t2_crit_p99 = t2_slo.get("critical", {}).get("ttft_p99", 0)
            crit_change = pct_change(b2_crit_p99, t2_crit_p99)
            crit_improvements.append(-crit_change)  # positive = improvement

            # Cluster TTFT P99
            cluster_change = pct_change(b2["ttft_p99"], t2["ttft_p99"])
            cluster_improvements.append(-cluster_change)

            # Rejection info
            t2_total = t2["completed"] + t2_rejected
            t2_reject_pct = (t2_rejected / t2_total * 100) if t2_total > 0 else 0

            print(f"    seed={seed}:")
            print(f"      Critical TTFT P99: B2={b2_crit_p99:.1f}ms, T2={t2_crit_p99:.1f}ms ({crit_change:+.1f}%)")
            print(f"      Cluster  TTFT P99: B2={b2['ttft_p99']:.1f}ms, T2={t2['ttft_p99']:.1f}ms ({cluster_change:+.1f}%)")
            print(f"      Rejected: T2={t2_rejected} ({t2_reject_pct:.1f}%), B2={b2_rejected}")
            print(f"      Completed: T2={t2['completed']}, B2={b2['completed']}")
            print(f"      Throughput: T2={t2['throughput']:.1f}, B2={b2['throughput']:.1f} rps")

        if crit_improvements:
            avg_crit = mean(crit_improvements)
            avg_cluster = mean(cluster_improvements)
            print()
            print(f"    Mean critical TTFT P99 improvement: {avg_crit:+.1f}%")
            print(f"    Mean cluster  TTFT P99 improvement: {avg_cluster:+.1f}%")
            if rate_label == "120":
                print()
                if avg_crit > 20:
                    print(f"    H-main VERDICT: SUPPORTED (>{avg_crit:.1f}% > 20% threshold)")
                else:
                    print(f"    H-main VERDICT: NOT SUPPORTED ({avg_crit:.1f}% < 20% threshold)")
        print()

    # ── H-zero-sum-broken: Non-zero-sum verification ────────────────────
    print("=== H-zero-sum-broken: Non-zero-sum verification ===")
    print()
    print("  Uses H-main 120% data. Checking cluster-wide P99 improvement.")
    print()

    cluster_120_improvements = []
    for seed in seeds:
        b2_file = results_dir / f"B2_120pct_s{seed}.txt"
        t2_file = results_dir / f"T2_120pct_s{seed}.txt"
        b2 = parse_blis_output(b2_file)
        t2 = parse_blis_output(t2_file)

        if b2["timed_out"] or t2["timed_out"]:
            continue

        change = pct_change(b2["ttft_p99"], t2["ttft_p99"])
        cluster_120_improvements.append(-change)
        print(f"    seed={seed}: Cluster TTFT P99 B2={b2['ttft_p99']:.1f}ms T2={t2['ttft_p99']:.1f}ms ({change:+.1f}%)")

    if cluster_120_improvements:
        avg = mean(cluster_120_improvements)
        print()
        if avg > 5:
            print(f"    H-zero-sum-broken VERDICT: SUPPORTED ({avg:+.1f}% > 5% improvement)")
        elif avg > 0:
            print(f"    H-zero-sum-broken VERDICT: PARTIAL ({avg:+.1f}% positive but < 5%)")
        else:
            print(f"    H-zero-sum-broken VERDICT: NOT SUPPORTED ({avg:+.1f}% -- no improvement)")
    print()

    # ── H-control-negative: Uniform SLO ─────────────────────────────────
    print("=== H-control-negative: Uniform SLO (all 'standard') at 120% ===")
    print()

    uniform_diffs = []
    for seed in seeds:
        b2_file = results_dir / f"B2_uniform_120pct_s{seed}.txt"
        t2_file = results_dir / f"T2_uniform_120pct_s{seed}.txt"
        b2 = parse_blis_output(b2_file)
        t2 = parse_blis_output(t2_file)
        t2_rejected = parse_rejected(t2_file)

        if b2["timed_out"] or t2["timed_out"]:
            print(f"    seed={seed}: SKIPPED (timeout)")
            continue

        diff = abs(pct_change(b2["ttft_p99"], t2["ttft_p99"]))
        uniform_diffs.append(diff)
        print(f"    seed={seed}: B2 TTFT P99={b2['ttft_p99']:.1f}ms, T2 TTFT P99={t2['ttft_p99']:.1f}ms (diff={diff:.1f}%), rejected={t2_rejected}")

    if uniform_diffs:
        avg_diff = mean(uniform_diffs)
        print()
        if avg_diff < 5:
            print(f"    H-control-negative VERDICT: SUPPORTED (mean diff {avg_diff:.1f}% < 5%)")
        else:
            print(f"    H-control-negative VERDICT: NOT SUPPORTED (mean diff {avg_diff:.1f}% >= 5%)")
    print()

    # ── H-threshold-sensitivity: Thresholds [50, 100, 200] ──────────────
    print("=== H-threshold-sensitivity: Queue thresholds [50, 100, 200] at 120% ===")
    print()

    thresholds = [50, 100, 200]
    for thresh in thresholds:
        improvements = []
        reject_rates = []
        print(f"  --- Threshold = {thresh} ---")

        for seed in seeds:
            b2_file = results_dir / f"B2_120pct_s{seed}.txt"
            t2_file = results_dir / f"T2_thresh{thresh}_120pct_s{seed}.txt"
            b2 = parse_blis_output(b2_file)
            t2 = parse_blis_output(t2_file)
            b2_slo = parse_per_slo_metrics(b2_file)
            t2_slo = parse_per_slo_metrics(t2_file)
            t2_rejected = parse_rejected(t2_file)

            if b2["timed_out"] or t2["timed_out"]:
                print(f"    seed={seed}: SKIPPED (timeout)")
                continue

            b2_crit_p99 = b2_slo.get("critical", {}).get("ttft_p99", 0)
            t2_crit_p99 = t2_slo.get("critical", {}).get("ttft_p99", 0)
            change = pct_change(b2_crit_p99, t2_crit_p99)
            improvements.append(-change)

            t2_total = t2["completed"] + t2_rejected
            reject_pct = (t2_rejected / t2_total * 100) if t2_total > 0 else 0
            reject_rates.append(reject_pct)

            print(f"    seed={seed}: Crit P99 B2={b2_crit_p99:.1f} T2={t2_crit_p99:.1f} ({change:+.1f}%), rejected={t2_rejected} ({reject_pct:.1f}%)")

        if improvements:
            avg_imp = mean(improvements)
            avg_rej = mean(reject_rates)
            print(f"    Mean improvement: {avg_imp:+.1f}%, Mean rejection rate: {avg_rej:.1f}%")
            if avg_imp > 15:
                print(f"    VERDICT: SUPPORTED (>{avg_imp:.1f}% > 15%)")
            else:
                print(f"    VERDICT: NOT SUPPORTED ({avg_imp:.1f}% < 15%)")
        print()

    # ── ED-3 Precondition Check ─────────────────────────────────────────
    print("=== ED-3 Precondition: Sheddable rejection rate > 5% at 120% ===")
    print()
    for seed in seeds:
        t2_file = results_dir / f"T2_120pct_s{seed}.txt"
        t2 = parse_blis_output(t2_file)
        t2_rejected = parse_rejected(t2_file)
        if t2["timed_out"]:
            print(f"    seed={seed}: SKIPPED (timeout)")
            continue
        total = t2["completed"] + t2_rejected
        reject_pct = (t2_rejected / total * 100) if total > 0 else 0
        status = "OK" if reject_pct > 5 else "FAIL"
        print(f"    seed={seed}: rejected={t2_rejected}/{total} ({reject_pct:.1f}%) [{status}]")
    print()

    # ── Summary Table ───────────────────────────────────────────────────
    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    print()
    print("  Arm                      | Prediction         | Result")
    print("  -------------------------|--------------------|--------")

    # Recompute for summary
    crit_120 = []
    cluster_120 = []
    for seed in seeds:
        b2 = parse_blis_output(results_dir / f"B2_120pct_s{seed}.txt")
        t2 = parse_blis_output(results_dir / f"T2_120pct_s{seed}.txt")
        b2_slo = parse_per_slo_metrics(results_dir / f"B2_120pct_s{seed}.txt")
        t2_slo = parse_per_slo_metrics(results_dir / f"T2_120pct_s{seed}.txt")
        if b2["timed_out"] or t2["timed_out"]:
            continue
        b2_c = b2_slo.get("critical", {}).get("ttft_p99", 0)
        t2_c = t2_slo.get("critical", {}).get("ttft_p99", 0)
        if b2_c > 0:
            crit_120.append(-(t2_c - b2_c) / b2_c * 100)
        cluster_120.append(-pct_change(b2["ttft_p99"], t2["ttft_p99"]))

    if crit_120:
        v = mean(crit_120)
        s = "SUPPORTED" if v > 20 else "NOT SUPPORTED"
        print(f"  H-main                   | >20% crit P99 imp | {v:+.1f}% [{s}]")
    if cluster_120:
        v = mean(cluster_120)
        s = "SUPPORTED" if v > 5 else ("PARTIAL" if v > 0 else "NOT SUPPORTED")
        print(f"  H-zero-sum-broken        | >5% cluster imp   | {v:+.1f}% [{s}]")
    if uniform_diffs:
        v = mean(uniform_diffs)
        s = "SUPPORTED" if v < 5 else "NOT SUPPORTED"
        print(f"  H-control-negative       | <5% diff uniform  | {v:.1f}% [{s}]")

    # threshold sensitivity: check all three thresholds
    all_above_15 = True
    thresh_results = []
    for thresh in thresholds:
        imps = []
        for seed in seeds:
            b2 = parse_blis_output(results_dir / f"B2_120pct_s{seed}.txt")
            t2 = parse_blis_output(results_dir / f"T2_thresh{thresh}_120pct_s{seed}.txt")
            b2_slo = parse_per_slo_metrics(results_dir / f"B2_120pct_s{seed}.txt")
            t2_slo = parse_per_slo_metrics(results_dir / f"T2_thresh{thresh}_120pct_s{seed}.txt")
            if b2["timed_out"] or t2["timed_out"]:
                continue
            b2_c = b2_slo.get("critical", {}).get("ttft_p99", 0)
            t2_c = t2_slo.get("critical", {}).get("ttft_p99", 0)
            if b2_c > 0:
                imps.append(-(t2_c - b2_c) / b2_c * 100)
        if imps:
            avg = mean(imps)
            thresh_results.append((thresh, avg))
            if avg <= 15:
                all_above_15 = False

    if thresh_results:
        parts = ", ".join(f"t={t}:{v:+.1f}%" for t, v in thresh_results)
        s = "SUPPORTED" if all_above_15 else "NOT SUPPORTED"
        print(f"  H-threshold-sensitivity  | >15% all thresholds| {parts} [{s}]")

    print()


if __name__ == "__main__":
    main()
