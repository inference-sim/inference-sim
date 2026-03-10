#!/usr/bin/env python3
"""Analyze H-Compound-Strategy: Strategy Evolution Iteration 4.

Usage: python3 analyze.py <results_dir>

Parses BLIS output and per-SLO metrics to evaluate:
  H-main (Dominance):       T4 reduces critical TTFT P99 by >25% over B2
  H-super-additivity:       (B2 - T4) > (B2 - T2) + (B2 - T3)
  H-cluster-health:         T4 produces best cluster-wide TTFT P99
  H-control-negative:       T4-uniform <5% difference from B2
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
    configs = ["B2", "T2", "T3", "T4"]

    print("=" * 76)
    print("  Strategy Evolution Iteration 4: Compound Strategy Analysis")
    print("=" * 76)
    print()

    # -- Load all data ---------------------------------------------------------

    data = {}  # config -> seed -> {"cluster": dict, "slo": dict, "rejected": int}
    for cfg in configs + ["T4uniform"]:
        data[cfg] = {}
        for seed in seeds:
            fname = results_dir / f"{cfg}_s{seed}.txt"
            cluster = parse_blis_output(fname)
            slo = parse_per_slo_metrics(fname)
            rejected = parse_rejected(fname)
            data[cfg][seed] = {"cluster": cluster, "slo": slo, "rejected": rejected}

    # -- Per-config summary table ----------------------------------------------

    print("=== Per-Config Metrics (all seeds) ===")
    print()
    print(f"  {'Config':<12} {'Seed':>4}  {'Crit P99':>10}  {'Std P99':>10}  "
          f"{'Shed P99':>10}  {'Cluster P99':>12}  {'Rejected':>8}  {'Completed':>9}")
    print(f"  {'-'*12} {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*9}")

    for cfg in configs:
        for seed in seeds:
            d = data[cfg][seed]
            if d["cluster"]["timed_out"]:
                print(f"  {cfg:<12} {seed:>4}  {'TIMEOUT':>10}")
                continue
            crit_p99 = d["slo"].get("critical", {}).get("ttft_p99", 0)
            std_p99 = d["slo"].get("standard", {}).get("ttft_p99", 0)
            shed_p99 = d["slo"].get("sheddable", {}).get("ttft_p99", 0)
            clust_p99 = d["cluster"]["ttft_p99"]
            rej = d["rejected"]
            comp = d["cluster"]["completed"]
            print(f"  {cfg:<12} {seed:>4}  {crit_p99:>10.1f}  {std_p99:>10.1f}  "
                  f"{shed_p99:>10.1f}  {clust_p99:>12.1f}  {rej:>8}  {comp:>9}")
        print()

    # -- H-main: T4 vs B2 dominance -------------------------------------------

    print("=" * 76)
    print("=== H-main (Dominance): T4 vs B2 critical TTFT P99 ===")
    print()

    t4_vs_b2_crit = []
    t4_vs_t2_crit = []
    t4_vs_t3_crit = []

    for seed in seeds:
        b2 = data["B2"][seed]
        t2 = data["T2"][seed]
        t3 = data["T3"][seed]
        t4 = data["T4"][seed]

        if any(d["cluster"]["timed_out"] for d in [b2, t2, t3, t4]):
            print(f"  seed={seed}: SKIPPED (timeout)")
            continue

        b2_crit = b2["slo"].get("critical", {}).get("ttft_p99", 0)
        t2_crit = t2["slo"].get("critical", {}).get("ttft_p99", 0)
        t3_crit = t3["slo"].get("critical", {}).get("ttft_p99", 0)
        t4_crit = t4["slo"].get("critical", {}).get("ttft_p99", 0)

        imp_vs_b2 = pct_change(b2_crit, t4_crit)  # negative = better
        imp_vs_t2 = pct_change(t2_crit, t4_crit)
        imp_vs_t3 = pct_change(t3_crit, t4_crit)

        t4_vs_b2_crit.append(-imp_vs_b2)  # positive = improvement
        t4_vs_t2_crit.append(-imp_vs_t2)
        t4_vs_t3_crit.append(-imp_vs_t3)

        print(f"  seed={seed}:")
        print(f"    B2 crit P99 = {b2_crit:.1f} ms")
        print(f"    T2 crit P99 = {t2_crit:.1f} ms (vs B2: {pct_change(b2_crit, t2_crit):+.1f}%)")
        print(f"    T3 crit P99 = {t3_crit:.1f} ms (vs B2: {pct_change(b2_crit, t3_crit):+.1f}%)")
        print(f"    T4 crit P99 = {t4_crit:.1f} ms (vs B2: {imp_vs_b2:+.1f}%)")
        print()

    if t4_vs_b2_crit:
        avg_vs_b2 = mean(t4_vs_b2_crit)
        avg_vs_t2 = mean(t4_vs_t2_crit)
        avg_vs_t3 = mean(t4_vs_t3_crit)
        print(f"  Mean T4 improvement over B2: {avg_vs_b2:+.1f}%")
        print(f"  Mean T4 improvement over T2: {avg_vs_t2:+.1f}%")
        print(f"  Mean T4 improvement over T3: {avg_vs_t3:+.1f}%")
        print()
        if avg_vs_b2 > 25:
            print(f"  H-main VERDICT: SUPPORTED (T4 vs B2 = {avg_vs_b2:+.1f}% > 25%)")
        else:
            print(f"  H-main VERDICT: NOT SUPPORTED (T4 vs B2 = {avg_vs_b2:+.1f}% <= 25%)")
    print()

    # -- H-super-additivity: compound > sum of parts --------------------------

    print("=" * 76)
    print("=== H-super-additivity: (B2 - T4) vs (B2 - T2) + (B2 - T3) ===")
    print()

    super_additive_results = []

    for seed in seeds:
        b2 = data["B2"][seed]
        t2 = data["T2"][seed]
        t3 = data["T3"][seed]
        t4 = data["T4"][seed]

        if any(d["cluster"]["timed_out"] for d in [b2, t2, t3, t4]):
            print(f"  seed={seed}: SKIPPED (timeout)")
            continue

        b2_crit = b2["slo"].get("critical", {}).get("ttft_p99", 0)
        t2_crit = t2["slo"].get("critical", {}).get("ttft_p99", 0)
        t3_crit = t3["slo"].get("critical", {}).get("ttft_p99", 0)
        t4_crit = t4["slo"].get("critical", {}).get("ttft_p99", 0)

        compound_effect = b2_crit - t4_crit          # total improvement (ms)
        admission_effect = b2_crit - t2_crit          # admission-only (ms)
        preemption_effect = b2_crit - t3_crit         # preemption-only (ms)
        sum_of_parts = admission_effect + preemption_effect
        interaction = compound_effect - sum_of_parts  # positive = super-additive

        is_super = compound_effect > sum_of_parts

        super_additive_results.append({
            "seed": seed,
            "compound": compound_effect,
            "admission": admission_effect,
            "preemption": preemption_effect,
            "sum_parts": sum_of_parts,
            "interaction": interaction,
            "is_super": is_super,
        })

        print(f"  seed={seed}:")
        print(f"    B2 crit P99 = {b2_crit:.1f} ms")
        print(f"    Compound effect (B2-T4)     = {compound_effect:+.1f} ms")
        print(f"    Admission effect (B2-T2)    = {admission_effect:+.1f} ms")
        print(f"    Preemption effect (B2-T3)   = {preemption_effect:+.1f} ms")
        print(f"    Sum of parts                = {sum_of_parts:+.1f} ms")
        print(f"    Interaction term            = {interaction:+.1f} ms "
              f"({'SUPER-ADDITIVE' if is_super else 'SUB-ADDITIVE'})")
        print()

    if super_additive_results:
        n_super = sum(1 for r in super_additive_results if r["is_super"])
        avg_interaction = mean(r["interaction"] for r in super_additive_results)
        avg_compound = mean(r["compound"] for r in super_additive_results)
        avg_sum = mean(r["sum_parts"] for r in super_additive_results)

        print(f"  Super-additive in {n_super}/{len(super_additive_results)} seeds")
        print(f"  Mean compound effect:   {avg_compound:+.1f} ms")
        print(f"  Mean sum of parts:      {avg_sum:+.1f} ms")
        print(f"  Mean interaction term:  {avg_interaction:+.1f} ms")
        print()

        if n_super >= 2:
            print(f"  H-super-additivity VERDICT: SUPPORTED ({n_super}/3 seeds super-additive)")
        elif n_super >= 1:
            print(f"  H-super-additivity VERDICT: PARTIAL ({n_super}/3 seeds super-additive)")
        else:
            if avg_compound > avg_sum:
                print(f"  H-super-additivity VERDICT: BORDERLINE (mean is super-additive but no seed majority)")
            else:
                print(f"  H-super-additivity VERDICT: NOT SUPPORTED (sub-additive: mechanisms partially substitute)")
    print()

    # -- H-cluster-health: T4 produces best cluster P99 -----------------------

    print("=" * 76)
    print("=== H-cluster-health: Best cluster-wide TTFT P99 ===")
    print()

    for seed in seeds:
        cluster_p99s = {}
        any_timeout = False
        for cfg in configs:
            d = data[cfg][seed]
            if d["cluster"]["timed_out"]:
                any_timeout = True
                break
            cluster_p99s[cfg] = d["cluster"]["ttft_p99"]

        if any_timeout:
            print(f"  seed={seed}: SKIPPED (timeout)")
            continue

        best_cfg = min(cluster_p99s, key=cluster_p99s.get)
        print(f"  seed={seed}: ", end="")
        parts = [f"{cfg}={v:.1f}ms" for cfg, v in sorted(cluster_p99s.items())]
        print(", ".join(parts))
        print(f"    Best = {best_cfg} ({cluster_p99s[best_cfg]:.1f} ms)")

    # Count how often T4 is best
    t4_best_count = 0
    total_valid = 0
    for seed in seeds:
        if any(data[cfg][seed]["cluster"]["timed_out"] for cfg in configs):
            continue
        total_valid += 1
        p99s = {cfg: data[cfg][seed]["cluster"]["ttft_p99"] for cfg in configs}
        if min(p99s, key=p99s.get) == "T4":
            t4_best_count += 1

    print()
    if total_valid > 0:
        if t4_best_count == total_valid:
            print(f"  H-cluster-health VERDICT: SUPPORTED (T4 best in all {total_valid} seeds)")
        elif t4_best_count > 0:
            print(f"  H-cluster-health VERDICT: PARTIAL (T4 best in {t4_best_count}/{total_valid} seeds)")
        else:
            print(f"  H-cluster-health VERDICT: NOT SUPPORTED (T4 never best)")
    print()

    # -- H-control-negative: T4-uniform vs B2 ---------------------------------

    print("=" * 76)
    print("=== H-control-negative: T4-uniform vs B2 (uniform SLO) ===")
    print()

    # For uniform SLO, we compare T4-uniform against a B2 uniform baseline.
    # Since B2 uniform isn't separately run, we compare T4-uniform cluster P99
    # against B2 mixed cluster P99 as a sanity check. The key test: with uniform
    # SLO, the SLO-gated admission and priority preemption should have <5% effect.
    # We compare T4-uniform against B2 (mixed) — the mechanisms should be inert.

    uniform_diffs = []
    for seed in seeds:
        b2 = data["B2"][seed]
        t4u = data["T4uniform"][seed]

        if b2["cluster"]["timed_out"] or t4u["cluster"]["timed_out"]:
            print(f"  seed={seed}: SKIPPED (timeout)")
            continue

        # Cluster P99 comparison
        b2_p99 = b2["cluster"]["ttft_p99"]
        t4u_p99 = t4u["cluster"]["ttft_p99"]
        diff = abs(pct_change(b2_p99, t4u_p99))
        uniform_diffs.append(diff)

        t4u_rej = t4u["rejected"]

        print(f"  seed={seed}: B2 cluster P99={b2_p99:.1f}ms, T4-uniform P99={t4u_p99:.1f}ms "
              f"(diff={diff:.1f}%), rejected={t4u_rej}")

    if uniform_diffs:
        avg_diff = mean(uniform_diffs)
        print()
        if avg_diff < 5:
            print(f"  H-control-negative VERDICT: SUPPORTED (mean diff {avg_diff:.1f}% < 5%)")
        else:
            print(f"  H-control-negative VERDICT: NOT SUPPORTED (mean diff {avg_diff:.1f}% >= 5%)")
    print()

    # -- Per-SLO-class detailed breakdown --------------------------------------

    print("=" * 76)
    print("=== Per-SLO-Class Detailed Breakdown (mean across seeds) ===")
    print()

    slo_classes = ["critical", "standard", "sheddable"]
    metrics_names = ["ttft_mean", "ttft_p99", "e2e_mean", "e2e_p99"]

    for slo_cls in slo_classes:
        print(f"  --- {slo_cls} ---")
        print(f"  {'Config':<8}  {'TTFT mean':>10}  {'TTFT P99':>10}  "
              f"{'E2E mean':>10}  {'E2E P99':>10}  {'N':>6}")

        for cfg in configs:
            vals = {m: [] for m in metrics_names}
            ns = []
            for seed in seeds:
                d = data[cfg][seed]
                if d["cluster"]["timed_out"]:
                    continue
                slo = d["slo"].get(slo_cls, {})
                if slo:
                    for m in metrics_names:
                        vals[m].append(slo.get(m, 0))
                    ns.append(slo.get("ttft_n", 0))

            if vals["ttft_mean"]:
                avgs = {m: mean(vals[m]) for m in metrics_names}
                avg_n = mean(ns) if ns else 0
                print(f"  {cfg:<8}  {avgs['ttft_mean']:>10.1f}  {avgs['ttft_p99']:>10.1f}  "
                      f"{avgs['e2e_mean']:>10.1f}  {avgs['e2e_p99']:>10.1f}  {avg_n:>6.0f}")
            else:
                print(f"  {cfg:<8}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>6}")
        print()

    # -- Preemption and Rejection Summary --------------------------------------

    print("=" * 76)
    print("=== Preemption and Rejection Summary ===")
    print()
    print(f"  {'Config':<8}  {'Seed':>4}  {'Preemptions':>12}  "
          f"{'Preempt Rate':>13}  {'Rejected':>8}  {'Completed':>9}")

    for cfg in configs:
        for seed in seeds:
            d = data[cfg][seed]
            if d["cluster"]["timed_out"]:
                print(f"  {cfg:<8}  {seed:>4}  {'TIMEOUT':>12}")
                continue
            preempt_count = d["cluster"]["preemption_count"]
            preempt_rate = d["cluster"]["preemption_rate"]
            rej = d["rejected"]
            comp = d["cluster"]["completed"]
            print(f"  {cfg:<8}  {seed:>4}  {preempt_count:>12}  "
                  f"{preempt_rate:>12.4f}  {rej:>8}  {comp:>9}")
    print()

    # -- Summary Table ---------------------------------------------------------

    print("=" * 76)
    print("  SUMMARY TABLE")
    print("=" * 76)
    print()
    print(f"  {'Arm':<26} | {'Prediction':<25} | {'Result':<30}")
    print(f"  {'-'*26}-+-{'-'*25}-+-{'-'*30}")

    # H-main
    if t4_vs_b2_crit:
        v = mean(t4_vs_b2_crit)
        s = "SUPPORTED" if v > 25 else "NOT SUPPORTED"
        print(f"  {'H-main (Dominance)':<26} | {'>25% crit P99 imp':<25} | {v:+.1f}% [{s}]")

    # H-super-additivity
    if super_additive_results:
        n_super = sum(1 for r in super_additive_results if r["is_super"])
        avg_int = mean(r["interaction"] for r in super_additive_results)
        if n_super >= 2:
            s = "SUPPORTED"
        elif n_super >= 1:
            s = "PARTIAL"
        else:
            s = "NOT SUPPORTED"
        print(f"  {'H-super-additivity':<26} | {'compound > sum parts':<25} | "
              f"{n_super}/3 seeds, int={avg_int:+.1f}ms [{s}]")

    # H-cluster-health
    if total_valid > 0:
        if t4_best_count == total_valid:
            s = "SUPPORTED"
        elif t4_best_count > 0:
            s = "PARTIAL"
        else:
            s = "NOT SUPPORTED"
        print(f"  {'H-cluster-health':<26} | {'T4 best cluster P99':<25} | "
              f"{t4_best_count}/{total_valid} seeds [{s}]")

    # H-control-negative
    if uniform_diffs:
        v = mean(uniform_diffs)
        s = "SUPPORTED" if v < 5 else "NOT SUPPORTED"
        print(f"  {'H-control-negative':<26} | {'<5% diff uniform SLO':<25} | {v:.1f}% [{s}]")

    print()


if __name__ == "__main__":
    main()
