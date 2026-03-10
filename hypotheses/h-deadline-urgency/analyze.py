#!/usr/bin/env python3
"""Analysis script for Strategy Evolution: Deadline-Aware SLO Scheduling.

Parses BLIS output from 6 hypothesis arms and produces per-arm verdicts.

Arms:
  H-main:         Treatment vs B0/B1/B2 at 30%, 80%, 120% capacity
  H-ablation:     Treatment (differentiated) vs uniform-deadline at 80%, 120%
  H-zero-sum:     Cluster-wide side-effect measurement at 120% (uses H-main data)
  H-control-neg:  Uniform-SLO vs differentiated-SLO at 30%
  H-robustness:   Treatment vs B2 at CV=1.5, 2.0, 3.5 at 120%
  H-single-turn:  Treatment vs B2 on single-turn workload at 120%

Output format reference:
  - Per-SLO metrics from BLIS stdout:
      === Per-SLO Metrics ===
        <class>:
          TTFT: mean=<val> p99=<val> (n=<count>)
          E2E:  mean=<val> p99=<val> (n=<count>)
  - Cluster JSON: "instance_id": "cluster" block with ttft_p99_ms, e2e_mean_ms, etc.
"""

import re
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout

SEEDS = [42, 123, 456]
SLO_CLASSES = ["critical", "standard", "sheddable"]
SLO_FRACTIONS = {"critical": 0.2, "standard": 0.4, "sheddable": 0.4}


# ── Per-SLO metric parsing ──────────────────────────────────────────────────

def parse_per_slo_metrics(filepath):
    """Parse Per-SLO Metrics from BLIS stdout.

    Expected format (cmd/root.go):
        === Per-SLO Metrics ===
          <class>:
            TTFT: mean=<val> p99=<val> (n=<count>)
            E2E:  mean=<val> p99=<val> (n=<count>)

    Returns dict of class -> {ttft_mean, ttft_p99, e2e_mean, e2e_p99, ttft_n, e2e_n}
    """
    path = Path(filepath)
    if not path.exists():
        return {}
    content = path.read_text()
    result = {}

    slo_block = re.search(r"=== Per-SLO Metrics ===(.*?)(?:===|\Z)", content, re.DOTALL)
    if not slo_block:
        return result

    block_text = slo_block.group(1)
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


# ── Utility functions ────────────────────────────────────────────────────────

def pct_diff(baseline, treatment):
    """Percentage change: (treatment - baseline) / baseline * 100.
    Positive = treatment is higher (worse for latency). Negative = improvement.
    """
    if baseline == 0:
        return float("inf") if treatment != 0 else 0.0
    return (treatment - baseline) / baseline * 100.0


def mean(vals):
    """Safe mean."""
    return sum(vals) / len(vals) if vals else 0.0


def verdict(condition, label_true="CONFIRMED", label_false="REFUTED"):
    """Return verdict string."""
    return label_true if condition else label_false


def safe_parse(filepath):
    """Parse BLIS output, returning None on timeout/error."""
    if check_for_timeout(filepath):
        return None
    return parse_blis_output(filepath)


# ── H-main analysis ─────────────────────────────────────────────────────────

def analyze_h_main(results_dir):
    """H-main: Treatment vs B0/B1/B2 at 30%, 80%, 120%."""
    print("=" * 90)
    print("  H-main: Deadline-aware urgency mechanism")
    print("=" * 90)
    print()

    d = Path(results_dir) / "h-main"

    for rate_label in [30, 80, 120]:
        print(f"  --- Rate: {rate_label}% capacity ---")
        print(f"  {'Config':<12} {'Seed':>6} {'Crit TTFT P99':>15} {'Crit TTFT Mean':>16} "
              f"{'Cluster TTFT P99':>18} {'Throughput':>12}")
        print(f"  {'-'*12} {'-'*6} {'-'*15} {'-'*16} {'-'*18} {'-'*12}")

        for config_name in ["b0", "b1", "b2", "treatment"]:
            for seed in SEEDS:
                f = d / f"{config_name}_r{rate_label}_s{seed}.txt"
                cluster = safe_parse(f)
                slo = parse_per_slo_metrics(f)
                crit = slo.get("critical", {})

                crit_p99 = crit.get("ttft_p99", 0)
                crit_mean = crit.get("ttft_mean", 0)
                cl_p99 = cluster.get("ttft_p99", 0) if cluster else 0
                tput = cluster.get("throughput", 0) if cluster else 0

                print(f"  {config_name:<12} {seed:>6} {crit_p99:>15.2f} {crit_mean:>16.2f} "
                      f"{cl_p99:>18.2f} {tput:>12.2f}")
        print()

    # Compute improvement percentages (Treatment vs B2, Treatment vs B1) at each rate
    print("  --- Improvement Summary (negative = Treatment is better) ---")
    print(f"  {'Rate':>6} {'Metric':<20} {'vs B2 (mean)':>14} {'vs B1 (mean)':>14} {'vs B0 (mean)':>14}")
    print(f"  {'-'*6} {'-'*20} {'-'*14} {'-'*14} {'-'*14}")

    improvement_vs_b2_120 = []  # For pass/fail
    improvement_vs_b1_120 = []

    for rate_label in [30, 80, 120]:
        diffs_vs_b2 = []
        diffs_vs_b1 = []
        diffs_vs_b0 = []

        for seed in SEEDS:
            slo_t = parse_per_slo_metrics(d / f"treatment_r{rate_label}_s{seed}.txt")
            slo_b2 = parse_per_slo_metrics(d / f"b2_r{rate_label}_s{seed}.txt")
            slo_b1 = parse_per_slo_metrics(d / f"b1_r{rate_label}_s{seed}.txt")
            slo_b0 = parse_per_slo_metrics(d / f"b0_r{rate_label}_s{seed}.txt")

            crit_t = slo_t.get("critical", {}).get("ttft_p99", 0)
            crit_b2 = slo_b2.get("critical", {}).get("ttft_p99", 0)
            crit_b1 = slo_b1.get("critical", {}).get("ttft_p99", 0)
            crit_b0 = slo_b0.get("critical", {}).get("ttft_p99", 0)

            if crit_b2 > 0:
                diffs_vs_b2.append(pct_diff(crit_b2, crit_t))
            if crit_b1 > 0:
                diffs_vs_b1.append(pct_diff(crit_b1, crit_t))
            if crit_b0 > 0:
                diffs_vs_b0.append(pct_diff(crit_b0, crit_t))

        mean_vs_b2 = mean(diffs_vs_b2)
        mean_vs_b1 = mean(diffs_vs_b1)
        mean_vs_b0 = mean(diffs_vs_b0)

        print(f"  {rate_label:>5}% {'Crit TTFT P99':<20} {mean_vs_b2:>+13.1f}% {mean_vs_b1:>+13.1f}% {mean_vs_b0:>+13.1f}%")

        if rate_label == 120:
            improvement_vs_b2_120 = diffs_vs_b2
            improvement_vs_b1_120 = diffs_vs_b1

    # Also show mean TTFT improvements
    print()
    for rate_label in [30, 80, 120]:
        diffs_vs_b2 = []
        diffs_vs_b1 = []
        for seed in SEEDS:
            slo_t = parse_per_slo_metrics(d / f"treatment_r{rate_label}_s{seed}.txt")
            slo_b2 = parse_per_slo_metrics(d / f"b2_r{rate_label}_s{seed}.txt")
            slo_b1 = parse_per_slo_metrics(d / f"b1_r{rate_label}_s{seed}.txt")

            crit_t_mean = slo_t.get("critical", {}).get("ttft_mean", 0)
            crit_b2_mean = slo_b2.get("critical", {}).get("ttft_mean", 0)
            crit_b1_mean = slo_b1.get("critical", {}).get("ttft_mean", 0)

            if crit_b2_mean > 0:
                diffs_vs_b2.append(pct_diff(crit_b2_mean, crit_t_mean))
            if crit_b1_mean > 0:
                diffs_vs_b1.append(pct_diff(crit_b1_mean, crit_t_mean))

        print(f"  {rate_label:>5}% {'Crit TTFT Mean':<20} {mean(diffs_vs_b2):>+13.1f}% {mean(diffs_vs_b1):>+13.1f}%")

    # Throughput check (Treatment vs B1 at 120%)
    tput_diffs = []
    for seed in SEEDS:
        cl_t = safe_parse(d / f"treatment_r120_s{seed}.txt")
        cl_b1 = safe_parse(d / f"b1_r120_s{seed}.txt")
        if cl_t and cl_b1 and cl_b1["throughput"] > 0:
            tput_diffs.append(pct_diff(cl_b1["throughput"], cl_t["throughput"]))

    mean_tput_diff = mean(tput_diffs)
    print(f"\n  Throughput change (Treatment vs B1) at 120%: {mean_tput_diff:+.1f}%")

    # ED-2 vanishing point check at 30%
    diffs_30 = []
    for seed in SEEDS:
        slo_t = parse_per_slo_metrics(d / f"treatment_r30_s{seed}.txt")
        slo_b0 = parse_per_slo_metrics(d / f"b0_r30_s{seed}.txt")
        crit_t = slo_t.get("critical", {}).get("ttft_p99", 0)
        crit_b0 = slo_b0.get("critical", {}).get("ttft_p99", 0)
        if crit_b0 > 0:
            diffs_30.append(abs(pct_diff(crit_b0, crit_t)))

    vanish_ok = mean(diffs_30) < 5.0 if diffs_30 else False
    print(f"  ED-2 vanishing point (30%): mean |diff| vs B0 = {mean(diffs_30):.1f}% (threshold <5%): "
          f"{verdict(vanish_ok, 'PASS', 'FAIL')}")

    # Verdicts
    print()
    # Primary: >15% improvement over B2 at 120% (negative pct_diff = improvement)
    mean_imp_b2 = mean(improvement_vs_b2_120) if improvement_vs_b2_120 else 0
    primary_pass = mean_imp_b2 < -15.0
    print(f"  PRIMARY (vs B2 at 120%): mean change = {mean_imp_b2:+.1f}% "
          f"(need < -15%): {verdict(primary_pass)}")

    # Secondary: >20% improvement over B1 at 120%
    mean_imp_b1 = mean(improvement_vs_b1_120) if improvement_vs_b1_120 else 0
    secondary_pass = mean_imp_b1 < -20.0
    print(f"  SECONDARY (vs B1 at 120%): mean change = {mean_imp_b1:+.1f}% "
          f"(need < -20%): {verdict(secondary_pass)}")

    # Throughput: <5% change
    tput_pass = abs(mean_tput_diff) < 5.0
    print(f"  THROUGHPUT (vs B1 at 120%): change = {mean_tput_diff:+.1f}% "
          f"(need |change| < 5%): {verdict(tput_pass, 'PASS', 'FAIL')}")

    # Overall
    if primary_pass and secondary_pass and tput_pass:
        h_main_verdict = "CONFIRMED"
    elif primary_pass or secondary_pass:
        h_main_verdict = "PARTIALLY_CONFIRMED"
    else:
        h_main_verdict = "REFUTED"
    print(f"\n  H-main VERDICT: {h_main_verdict}")
    print()
    return h_main_verdict


# ── H-ablation analysis ─────────────────────────────────────────────────────

def analyze_h_ablation(results_dir):
    """H-ablation: Differentiated vs uniform deadline."""
    print("=" * 90)
    print("  H-ablation: Per-class deadline differentiation")
    print("=" * 90)
    print()

    d = Path(results_dir) / "h-ablation"

    print(f"  {'Rate':>6} {'Seed':>6} {'Diff Crit P99':>15} {'Uniform Crit P99':>18} {'Degradation%':>14}")
    print(f"  {'-'*6} {'-'*6} {'-'*15} {'-'*18} {'-'*14}")

    all_degradations = {}
    for rate_label in [80, 120]:
        degradations = []
        for seed in SEEDS:
            slo_diff = parse_per_slo_metrics(d / f"diff_r{rate_label}_s{seed}.txt")
            slo_uniform = parse_per_slo_metrics(d / f"uniform_dl_r{rate_label}_s{seed}.txt")

            diff_p99 = slo_diff.get("critical", {}).get("ttft_p99", 0)
            uniform_p99 = slo_uniform.get("critical", {}).get("ttft_p99", 0)

            deg = pct_diff(diff_p99, uniform_p99) if diff_p99 > 0 else 0
            degradations.append(deg)

            print(f"  {rate_label:>5}% {seed:>6} {diff_p99:>15.2f} {uniform_p99:>18.2f} {deg:>+13.1f}%")

        all_degradations[rate_label] = degradations
        print(f"  {rate_label:>5}% {'mean':>6} {'':>15} {'':>18} {mean(degradations):>+13.1f}%")
        print()

    # Verdict: uniform deadlines degrade critical TTFT P99 by >15%
    # Check at both rates; primary is 80% (where deadline differentiation has max leverage)
    deg_80 = mean(all_degradations.get(80, [0]))
    deg_120 = mean(all_degradations.get(120, [0]))

    pass_80 = deg_80 > 15.0
    pass_120 = deg_120 > 15.0

    print(f"  80% rate: mean degradation = {deg_80:+.1f}% (need >15%): {verdict(pass_80)}")
    print(f"  120% rate: mean degradation = {deg_120:+.1f}% (need >15%): {verdict(pass_120)}")

    if pass_80 and pass_120:
        v = "CONFIRMED"
    elif pass_80 or pass_120:
        v = "PARTIALLY_CONFIRMED"
    else:
        v = "REFUTED"
    print(f"\n  H-ablation VERDICT: {v}")
    print()
    return v


# ── H-zero-sum analysis ─────────────────────────────────────────────────────

def analyze_h_zero_sum(results_dir):
    """H-zero-sum: Cluster-wide side-effect at 120% (uses H-main data)."""
    print("=" * 90)
    print("  H-zero-sum: Cluster-wide side-effect at 120%")
    print("=" * 90)
    print()

    d = Path(results_dir) / "h-main"

    # Cluster TTFT P99 for each config at 120%
    print(f"  {'Config':<12} {'Seed':>6} {'Cluster TTFT P99':>18} {'vs B0 %':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*18} {'-'*10}")

    b0_p99s = []
    treatment_p99s = []
    b2_p99s = []

    for config_name in ["b0", "b1", "b2", "treatment"]:
        for seed in SEEDS:
            cl = safe_parse(d / f"{config_name}_r120_s{seed}.txt")
            p99 = cl["ttft_p99"] if cl else 0

            if config_name == "b0":
                b0_p99s.append(p99)
            elif config_name == "treatment":
                treatment_p99s.append(p99)
            elif config_name == "b2":
                b2_p99s.append(p99)

            b0_ref = safe_parse(d / f"b0_r120_s{seed}.txt")
            b0_val = b0_ref["ttft_p99"] if b0_ref else 0
            deg = pct_diff(b0_val, p99) if b0_val > 0 else 0

            print(f"  {config_name:<12} {seed:>6} {p99:>18.2f} {deg:>+9.1f}%")
        print()

    # Sub-prediction A: Treatment degradation < B2 degradation
    treatment_degs = []
    b2_degs = []
    for i, seed in enumerate(SEEDS):
        if b0_p99s[i] > 0:
            treatment_degs.append(pct_diff(b0_p99s[i], treatment_p99s[i]))
            b2_degs.append(pct_diff(b0_p99s[i], b2_p99s[i]))

    mean_t_deg = mean(treatment_degs)
    mean_b2_deg = mean(b2_degs)

    # Absolute bound: <40% degradation vs B0
    abs_pass = mean_t_deg < 40.0
    # Comparative: Treatment degradation < B2 degradation
    comp_pass = mean_t_deg < mean_b2_deg

    print(f"  Treatment cluster P99 degradation vs B0: {mean_t_deg:+.1f}%")
    print(f"  B2 cluster P99 degradation vs B0:        {mean_b2_deg:+.1f}%")
    print(f"  Absolute bound (<40%): {verdict(abs_pass, 'PASS', 'FAIL')}")
    print(f"  Comparative (Treatment < B2): {verdict(comp_pass, 'PASS', 'FAIL')}")
    print()

    # Sub-prediction B: Weighted zero-sum check
    # |sum(fraction_i * (mean_TTFT_treatment_i - mean_TTFT_B0_i) / mean_TTFT_B0_i)| < 0.10
    zs_values = []
    for seed in SEEDS:
        slo_t = parse_per_slo_metrics(d / f"treatment_r120_s{seed}.txt")
        slo_b0 = parse_per_slo_metrics(d / f"b0_r120_s{seed}.txt")

        weighted_sum = 0.0
        valid = True
        for cls in SLO_CLASSES:
            t_mean = slo_t.get(cls, {}).get("ttft_mean", 0)
            b0_mean = slo_b0.get(cls, {}).get("ttft_mean", 0)
            if b0_mean <= 0:
                valid = False
                break
            weighted_sum += SLO_FRACTIONS[cls] * (t_mean - b0_mean) / b0_mean

        if valid:
            zs_values.append(weighted_sum)

    mean_zs = mean(zs_values)
    zs_pass = abs(mean_zs) < 0.10

    print(f"  Weighted zero-sum index: {mean_zs:+.4f} (threshold |x| < 0.10): "
          f"{verdict(zs_pass, 'PASS', 'FAIL')}")

    # Overall verdict
    if abs_pass and comp_pass and zs_pass:
        v = "CONFIRMED"
    elif abs_pass and (comp_pass or zs_pass):
        v = "PARTIALLY_CONFIRMED"
    else:
        v = "REFUTED"
    print(f"\n  H-zero-sum VERDICT: {v}")
    print()
    return v


# ── H-control-negative analysis ─────────────────────────────────────────────

def analyze_h_control_neg(results_dir):
    """H-control-neg: Uniform SLO vs differentiated at 30%."""
    print("=" * 90)
    print("  H-control-negative: Mechanism specificity at 30%")
    print("=" * 90)
    print()

    d = Path(results_dir) / "h-control-neg"

    # Compare cluster-level metrics (since uniform-SLO has only 'standard' class,
    # per-SLO comparison is not meaningful; use cluster TTFT P99)
    print(f"  {'Seed':>6} {'Uniform TTFT P99':>18} {'Diff TTFT P99':>15} {'Diff%':>10}")
    print(f"  {'-'*6} {'-'*18} {'-'*15} {'-'*10}")

    diffs = []
    for seed in SEEDS:
        cl_uniform = safe_parse(d / f"uniform_slo_s{seed}.txt")
        cl_diff = safe_parse(d / f"diff_slo_s{seed}.txt")

        uniform_p99 = cl_uniform["ttft_p99"] if cl_uniform else 0
        diff_p99 = cl_diff["ttft_p99"] if cl_diff else 0

        pct = pct_diff(uniform_p99, diff_p99) if uniform_p99 > 0 else 0
        diffs.append(abs(pct))

        print(f"  {seed:>6} {uniform_p99:>18.2f} {diff_p99:>15.2f} {pct:>+9.1f}%")

    mean_abs_diff = mean(diffs)
    equiv_pass = mean_abs_diff < 5.0

    # Also check per-SLO for the differentiated config
    print()
    print("  Per-SLO detail (differentiated config only):")
    for seed in SEEDS:
        slo_diff = parse_per_slo_metrics(d / f"diff_slo_s{seed}.txt")
        for cls in SLO_CLASSES:
            m = slo_diff.get(cls, {})
            print(f"    seed={seed} {cls:<12} TTFT: mean={m.get('ttft_mean',0):.2f} "
                  f"p99={m.get('ttft_p99',0):.2f} (n={m.get('ttft_n',0)})")
    print()

    print(f"  Mean |difference|: {mean_abs_diff:.1f}% (threshold <5%): {verdict(equiv_pass, 'PASS', 'FAIL')}")

    v = "CONFIRMED" if equiv_pass else "REFUTED"
    print(f"\n  H-control-negative VERDICT: {v}")
    print()
    return v


# ── H-robustness analysis ───────────────────────────────────────────────────

def analyze_h_robustness(results_dir):
    """H-robustness: Treatment vs B2 at CV=1.5, 2.0, 3.5 at 120%."""
    print("=" * 90)
    print("  H-robustness: Burst intensity scaling at 120%")
    print("=" * 90)
    print()

    d = Path(results_dir) / "h-robustness"

    cv_labels = [("1.5", "1p5"), ("2.0", "2p0"), ("3.5", "3p5")]

    print(f"  {'CV':>5} {'Seed':>6} {'Treatment Crit P99':>20} {'B2 Crit P99':>15} {'Improvement%':>14}")
    print(f"  {'-'*5} {'-'*6} {'-'*20} {'-'*15} {'-'*14}")

    cv_improvements = {}
    for cv_str, cv_file in cv_labels:
        improvements = []
        for seed in SEEDS:
            slo_t = parse_per_slo_metrics(d / f"treatment_cv{cv_file}_s{seed}.txt")
            slo_b2 = parse_per_slo_metrics(d / f"b2_cv{cv_file}_s{seed}.txt")

            t_p99 = slo_t.get("critical", {}).get("ttft_p99", 0)
            b2_p99 = slo_b2.get("critical", {}).get("ttft_p99", 0)

            imp = pct_diff(b2_p99, t_p99) if b2_p99 > 0 else 0
            improvements.append(imp)

            print(f"  {cv_str:>5} {seed:>6} {t_p99:>20.2f} {b2_p99:>15.2f} {imp:>+13.1f}%")

        cv_improvements[cv_str] = improvements
        print(f"  {cv_str:>5} {'mean':>6} {'':>20} {'':>15} {mean(improvements):>+13.1f}%")
        print()

    # Verdict: maintain >15% improvement across all CV values
    # (negative pct_diff = improvement, so we check < -15%)
    all_pass = True
    any_pass = False
    for cv_str, imps in cv_improvements.items():
        m = mean(imps)
        ok = m < -15.0
        if ok:
            any_pass = True
        else:
            all_pass = False
        print(f"  CV={cv_str}: mean improvement = {m:+.1f}% (need < -15%): {verdict(ok)}")

    if all_pass:
        v = "CONFIRMED"
    elif any_pass:
        v = "PARTIALLY_CONFIRMED"
    else:
        v = "REFUTED"
    print(f"\n  H-robustness VERDICT: {v}")
    print()
    return v


# ── H-single-turn analysis ──────────────────────────────────────────────────

def analyze_h_single_turn(results_dir):
    """H-single-turn: Treatment vs B2 on single-turn at 120%."""
    print("=" * 90)
    print("  H-single-turn: Multi-turn confound isolation at 120%")
    print("=" * 90)
    print()

    d = Path(results_dir) / "h-single-turn"

    print(f"  {'Seed':>6} {'Treatment Crit P99':>20} {'B2 Crit P99':>15} {'Improvement%':>14}")
    print(f"  {'-'*6} {'-'*20} {'-'*15} {'-'*14}")

    improvements = []
    for seed in SEEDS:
        slo_t = parse_per_slo_metrics(d / f"treatment_s{seed}.txt")
        slo_b2 = parse_per_slo_metrics(d / f"b2_s{seed}.txt")

        t_p99 = slo_t.get("critical", {}).get("ttft_p99", 0)
        b2_p99 = slo_b2.get("critical", {}).get("ttft_p99", 0)

        imp = pct_diff(b2_p99, t_p99) if b2_p99 > 0 else 0
        improvements.append(imp)

        print(f"  {seed:>6} {t_p99:>20.2f} {b2_p99:>15.2f} {imp:>+13.1f}%")

    mean_imp = mean(improvements)
    print(f"  {'mean':>6} {'':>20} {'':>15} {mean_imp:>+13.1f}%")
    print()

    # Verdict: >10% improvement (negative pct_diff = improvement)
    ok = mean_imp < -10.0
    print(f"  Mean improvement: {mean_imp:+.1f}% (need < -10%): {verdict(ok)}")

    v = "CONFIRMED" if ok else "REFUTED"
    print(f"\n  H-single-turn VERDICT: {v}")
    print()
    return v


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(verdicts):
    """Print final summary table."""
    print("=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print()
    print(f"  {'Arm':<25} {'Verdict':<25} {'Key Metric':<30}")
    print(f"  {'-'*25} {'-'*25} {'-'*30}")
    for arm, v, metric in verdicts:
        print(f"  {arm:<25} {v:<25} {metric:<30}")
    print()

    confirmed = sum(1 for _, v, _ in verdicts if v == "CONFIRMED")
    partial = sum(1 for _, v, _ in verdicts if v == "PARTIALLY_CONFIRMED")
    refuted = sum(1 for _, v, _ in verdicts if v == "REFUTED")
    print(f"  Confirmed: {confirmed}/{len(verdicts)}  "
          f"Partially: {partial}/{len(verdicts)}  "
          f"Refuted: {refuted}/{len(verdicts)}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        print("Usage: analyze.py <results_dir>", file=sys.stderr)
        print("  results_dir should contain h-main/, h-ablation/, h-control-neg/,", file=sys.stderr)
        print("  h-robustness/, h-single-turn/ subdirectories", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]
    if not Path(results_dir).is_dir():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    verdicts = []

    v = analyze_h_main(results_dir)
    verdicts.append(("H-main", v, "Crit TTFT P99 vs B2/B1"))

    v = analyze_h_ablation(results_dir)
    verdicts.append(("H-ablation-deadline", v, "Crit TTFT P99 degradation"))

    v = analyze_h_zero_sum(results_dir)
    verdicts.append(("H-zero-sum", v, "Cluster TTFT P99 vs B0"))

    v = analyze_h_control_neg(results_dir)
    verdicts.append(("H-control-negative", v, "<5% diff at 30%"))

    v = analyze_h_robustness(results_dir)
    verdicts.append(("H-robustness-burst", v, "Crit TTFT P99 vs B2 per CV"))

    v = analyze_h_single_turn(results_dir)
    verdicts.append(("H-single-turn-control", v, "Crit TTFT P99 vs B2"))

    print_summary(verdicts)


if __name__ == "__main__":
    main()
