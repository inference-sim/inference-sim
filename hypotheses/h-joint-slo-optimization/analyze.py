#!/usr/bin/env python3
"""
analyze.py — Statistical analysis for h-joint-slo-optimization.

USAGE
  python3 analyze.py <iteration> <results_dir>

  python3 analyze.py iter0  results/     # baseline tables
  python3 analyze.py iter1  results/     # composition + ablations
  python3 analyze.py iter2  results/     # SLO-priority preemption
  python3 analyze.py iter3  results/     # tiered-LRU KV eviction
  python3 analyze.py iter4  results/     # tier-budget + fraction sweep
  python3 analyze.py all    results/     # all iterations in sequence

OUTPUT
  Markdown tables ready to paste into FINDINGS.md.

PARSING
  Uses hypotheses/lib/analyze_helpers.py (parse_blis_output) and
  h-compound-strategy's parse_per_slo_metrics pattern for per-SLO P99.
  BLIS per-SLO output is in ticks (microseconds); converted to ms here.
"""

import re
import sys
import statistics
from pathlib import Path

# Add shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout  # noqa: E402

SEEDS = [42, 123, 456]


# ── Per-SLO parser (matches actual blis output format) ───────────────────────

def parse_per_slo(filepath: str) -> dict[str, dict]:
    """Parse '=== Per-SLO Metrics ===' section from blis output.

    Returns {slo_class: {"ttft_p99_ms": float, "ttft_mean_ms": float, "n": int}}
    BLIS outputs ticks (microseconds); we divide by 1000 to get ms.
    Returns empty dict if section is missing or file is missing/timed-out.
    """
    result: dict[str, dict] = {}
    path = Path(filepath)
    if not path.exists():
        return result
    content = path.read_text()
    slo_section = re.search(
        r"=== Per-SLO Metrics ===\s*\n(.*?)(?:\n===|\Z)", content, re.DOTALL
    )
    if not slo_section:
        return result
    class_pattern = re.compile(
        r"^\s+(\S+):\s*\n"
        r"\s+TTFT:\s+mean=([0-9.]+)\s+p99=([0-9.]+)\s+\(n=(\d+)\)",
        re.MULTILINE,
    )
    for m in class_pattern.finditer(slo_section.group(1)):
        cls, mean_us, p99_us, n = m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))
        result[cls] = {
            "ttft_mean_ms": mean_us / 1000.0,
            "ttft_p99_ms":  p99_us  / 1000.0,
            "n": n,
        }
    return result


# ── Aggregation helpers ───────────────────────────────────────────────────────

def load_arm(arm_dir: str) -> list[tuple[int, dict, dict]]:
    """Load all seed results for an arm. Returns [(seed, cluster_metrics, slo_metrics)]."""
    results = []
    for seed in SEEDS:
        p = Path(arm_dir) / f"seed{seed}.txt"
        cluster = parse_blis_output(str(p))
        slo     = parse_per_slo(str(p))
        results.append((seed, cluster, slo))
    return results


def critical_p99s(arm_results: list) -> list[float]:
    return [slo.get("critical", {}).get("ttft_p99_ms", float("nan"))
            for _, _, slo in arm_results]


def goodput(arm_results: list, cls: str) -> list[float]:
    """Goodput = completed requests for that SLO class / injected for that class."""
    # BLIS doesn't report per-class injected separately; use n from per-SLO as a proxy.
    return [slo.get(cls, {}).get("n", float("nan"))
            for _, _, slo in arm_results]


def cache_hit_rates(arm_results: list) -> list[float]:
    return [c.get("cache_hit_rate", float("nan")) for _, c, _ in arm_results]


def ms(x: float) -> str:
    return "—" if x != x else f"{x:.1f}"  # nan check


def pct(x: float) -> str:
    return "—" if x != x else f"{x:+.1f}%"


def mean_pm_std(vals: list[float]) -> str:
    clean = [v for v in vals if v == v]
    if not clean:
        return "—"
    m = statistics.mean(clean)
    s = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return f"{m:.1f}±{s:.1f}"


def improvement(baseline: list[float], treatment: list[float]) -> list[float]:
    """% reduction in P99 (positive = better). Per-seed."""
    return [
        (b - t) / b * 100 if b == b and t == t and b != 0 else float("nan")
        for b, t in zip(baseline, treatment)
    ]


def print_table(headers: list, rows: list) -> None:
    def fmt(v) -> str:
        if isinstance(v, float):
            return "—" if v != v else f"{v:.1f}"
        return str(v) if v is not None else "—"
    widths = [max(len(str(h)), max((len(fmt(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    print("| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |")
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(v).ljust(w) for v, w in zip(row, widths)) + " |")
    print()


# ── Per-iteration analyses ────────────────────────────────────────────────────

def analyze_iter0(results_dir: str) -> None:
    print("## Iteration 0: Baseline\n")
    arm = load_arm(f"{results_dir}/iter0/compound")
    rows = []
    p99s, hr_vals = [], []
    for seed, cluster, slo in arm:
        p99 = slo.get("critical", {}).get("ttft_p99_ms", float("nan"))
        hr  = cluster.get("cache_hit_rate", float("nan"))
        pc  = cluster.get("preemption_count", 0)
        std_n   = slo.get("standard",  {}).get("n", "—")
        shed_n  = slo.get("sheddable", {}).get("n", "—")
        crit_n  = slo.get("critical",  {}).get("n", "—")
        rows.append([seed, ms(p99), std_n, shed_n, crit_n, pc, f"{hr:.3f}" if hr == hr else "—"])
        if p99 == p99: p99s.append(p99)
        if hr  == hr:  hr_vals.append(hr)
    p99_m = statistics.mean(p99s) if p99s else float("nan")
    p99_s = statistics.stdev(p99s) if len(p99s) > 1 else 0.0
    rows.append(["Mean±σ", f"{p99_m:.1f}±{p99_s:.1f}", "—", "—", "—", "—", "—"])
    print_table(["Seed", "Critical P99 (ms)", "Std n", "Shed n", "Crit n",
                 "Preemptions", "KV Hit Rate"], rows)
    print(f"**Reference:** critical_p99_baseline = {p99_m:.1f} ms\n")


def analyze_iter1(results_dir: str) -> None:
    print("## Iteration 1: Joint Composition\n")
    compound  = load_arm(f"{results_dir}/iter1/compound")
    defaults  = load_arm(f"{results_dir}/iter1/blis-defaults")
    cp99s = critical_p99s(compound)
    dp99s = critical_p99s(defaults)
    imps  = improvement(dp99s, cp99s)

    print("### H-main: Compound vs BLIS defaults\n")
    rows = [(s, ms(c), ms(d), pct(i)) for s, c, d, i in zip(SEEDS, cp99s, dp99s, imps)]
    rows.append(["Mean±σ", mean_pm_std(cp99s), mean_pm_std(dp99s), mean_pm_std(imps)])
    print_table(["Seed", "Compound P99 (ms)", "Default P99 (ms)", "Improvement"], rows)
    imp_m = statistics.mean([v for v in imps if v == v]) if any(v == v for v in imps) else float("nan")
    verdict = ("✅ CONFIRMED" if imp_m > 40
               else ("⚠️  BELOW 40% (but >15%)" if imp_m > 15
               else "❌ BELOW 15% MINIMUM"))
    print(f"**H-main verdict:** {verdict} (mean improvement={imp_m:.1f}%, threshold >40%)\n")

    print("### H-ablation: Component contributions\n")
    ablations = [
        ("abl-routing",    "Round-robin routing",  15),
        ("abl-scheduling", "FCFS scheduler",        20),
        ("abl-admission",  "Always-admit",          30),
    ]
    rows = []
    for subdir, label, threshold in ablations:
        abl = load_arm(f"{results_dir}/iter1/{subdir}")
        ap99s = critical_p99s(abl)
        # Degradation = how much worse than compound (positive = ablation is worse)
        degs = improvement(cp99s, ap99s)  # positive means abl has higher P99 than compound
        deg_m = statistics.mean([v for v in degs if v == v]) if any(v == v for v in degs) else float("nan")
        # Negate: we want "ablation is X% worse than compound"
        deg_m = -deg_m  # now positive means compound beats ablation
        fast_fail = " → FAST-FAIL" if abs(deg_m) < 5 else ""
        verdict_sym = "✅" if deg_m >= threshold else "❌"
        rows.append([label, mean_pm_std(ap99s), f"{deg_m:+.1f}%", f">{threshold}%",
                     f"{verdict_sym}{fast_fail}"])
    print_table(["Arm", "Ablation P99 (ms)", "Degradation vs Compound", "Threshold", "Pass?"], rows)


def analyze_iter2(results_dir: str) -> None:
    print("## Iteration 2: SLO-Priority Preemption\n")
    treatment = load_arm(f"{results_dir}/iter2/treatment")
    ablation  = load_arm(f"{results_dir}/iter2/ablation")
    tp99s = critical_p99s(treatment)
    ap99s = critical_p99s(ablation)
    imps  = improvement(ap99s, tp99s)

    print("### H-main\n")
    rows = [(s, ms(t), ms(a), pct(i)) for s, t, a, i in zip(SEEDS, tp99s, ap99s, imps)]
    rows.append(["Mean±σ", mean_pm_std(tp99s), mean_pm_std(ap99s), mean_pm_std(imps)])
    print_table(["Seed", "SLO-priority P99 (ms)", "LIFO (ablation) P99 (ms)", "Improvement"], rows)
    imp_m = statistics.mean([v for v in imps if v == v]) if any(v == v for v in imps) else float("nan")
    verdict = "✅ CONFIRMED" if imp_m > 15 else ("⚠️  MARGINAL" if imp_m > 5 else "❌ FAST-FAIL (<5%)")
    print(f"**H-main verdict:** {verdict} (mean improvement={imp_m:.1f}%, threshold >15%)\n")

    print("### H-zero-sum: Standard request count\n")
    t_std = goodput(treatment, "standard")
    a_std = goodput(ablation,  "standard")
    rows = [(s, ts, as_) for s, ts, as_ in zip(SEEDS, t_std, a_std)]
    rows.append(["Total", sum(v for v in t_std if str(v) != "nan"),
                 sum(v for v in a_std if str(v) != "nan")])
    print_table(["Seed", "Std completed (treatment)", "Std completed (LIFO)"], rows)
    print("*(Goodput floor: treatment std count ≥ 85% of ablation std count)*\n")


def analyze_iter3(results_dir: str) -> None:
    print("## Iteration 3: Tiered-LRU KV Eviction\n")
    treatment = load_arm(f"{results_dir}/iter3/treatment")
    ablation  = load_arm(f"{results_dir}/iter3/ablation")
    tp99s = critical_p99s(treatment)
    ap99s = critical_p99s(ablation)
    imps  = improvement(ap99s, tp99s)

    print("### H-main: P99\n")
    rows = [(s, ms(t), ms(a), pct(i)) for s, t, a, i in zip(SEEDS, tp99s, ap99s, imps)]
    rows.append(["Mean±σ", mean_pm_std(tp99s), mean_pm_std(ap99s), mean_pm_std(imps)])
    print_table(["Seed", "Tiered-LRU P99 (ms)", "Single-list P99 (ms)", "Improvement"], rows)
    imp_m = statistics.mean([v for v in imps if v == v]) if any(v == v for v in imps) else float("nan")
    verdict = "✅ CONFIRMED" if imp_m > 15 else ("⚠️  MARGINAL" if imp_m > 5 else "❌ FAST-FAIL")
    print(f"**H-main verdict:** {verdict} (mean improvement={imp_m:.1f}%, threshold >15%)\n")

    print("### Cache hit rate\n")
    t_hr = cache_hit_rates(treatment)
    a_hr = cache_hit_rates(ablation)
    rows = [(s, f"{t:.3f}" if t==t else "—", f"{a:.3f}" if a==a else "—")
            for s, t, a in zip(SEEDS, t_hr, a_hr)]
    print_table(["Seed", "Tiered-LRU hit rate", "Single-list hit rate"], rows)


def analyze_iter4(results_dir: str) -> None:
    print("## Iteration 4: Tier-Budget Batch Formation\n")
    treatment = load_arm(f"{results_dir}/iter4/treatment")
    abl_equal = load_arm(f"{results_dir}/iter4/abl-equal-share")

    # Baseline for iter4 = iter2 treatment
    iter2 = load_arm(f"{results_dir}/iter2/treatment")
    i2p99s = critical_p99s(iter2)
    tp99s  = critical_p99s(treatment)
    imps   = improvement(i2p99s, tp99s)

    print("### H-main: f_c=0.50 vs Iter 2 compound\n")
    rows = [(s, ms(t), ms(b), pct(i)) for s, t, b, i in zip(SEEDS, tp99s, i2p99s, imps)]
    rows.append(["Mean±σ", mean_pm_std(tp99s), mean_pm_std(i2p99s), mean_pm_std(imps)])
    print_table(["Seed", "Tier-budget P99 (ms)", "Iter2 compound P99 (ms)", "Improvement"], rows)
    imp_m = statistics.mean([v for v in imps if v == v]) if any(v == v for v in imps) else float("nan")
    verdict = "✅ CONFIRMED" if imp_m > 10 else ("⚠️  MARGINAL" if imp_m > 5 else "❌ FAST-FAIL")
    print(f"**H-main verdict:** {verdict} (mean improvement={imp_m:.1f}%, threshold >10%)\n")

    print("### H-ablation: f_c=0.50 vs equal-share f_c=0.333\n")
    ap99s = critical_p99s(abl_equal)
    degs  = improvement(tp99s, ap99s)  # positive = equal-share is worse
    rows = [(s, ms(t), ms(a), pct(d)) for s, t, a, d in zip(SEEDS, tp99s, ap99s, degs)]
    rows.append(["Mean±σ", mean_pm_std(tp99s), mean_pm_std(ap99s), mean_pm_std(degs)])
    print_table(["Seed", "f_c=0.50 P99 (ms)", "f_c=0.333 P99 (ms)", "Degradation (equal-share)"], rows)

    print("### H-robustness: fraction sweep\n")
    sweep = {
        "0.20": "iter4/sweep-fc020",
        "0.30": "iter4/sweep-fc030",
        "0.40": "iter4/sweep-fc040",
        "0.50": "iter4/sweep-fc050",
        "0.60": "iter4/sweep-fc060",
        "0.70": "iter4/sweep-fc070",
    }
    rows = []
    prev = float("nan")
    for fc, subdir in sweep.items():
        arm = load_arm(f"{results_dir}/{subdir}")
        p99s = critical_p99s(arm)
        pm = statistics.mean([v for v in p99s if v == v]) if any(v == v for v in p99s) else float("nan")
        mono = "✅" if prev != prev or pm <= prev else "❌ non-monotone"
        rows.append([fc, ms(pm), mono])
        prev = pm
    print_table(["f_c", "Critical P99 (ms)", "Monotone?"], rows)


def analyze_all(results_dir: str) -> None:
    for fn in [analyze_iter0, analyze_iter1, analyze_iter2, analyze_iter3, analyze_iter4]:
        fn(results_dir)
        print("---\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 analyze.py <iter0|iter1|iter2|iter3|iter4|all> <results_dir>")
        sys.exit(1)
    cmd, results_dir = sys.argv[1], sys.argv[2]
    dispatch = {
        "iter0": analyze_iter0,
        "iter1": analyze_iter1,
        "iter2": analyze_iter2,
        "iter3": analyze_iter3,
        "iter4": analyze_iter4,
        "all":   analyze_all,
    }
    if cmd not in dispatch:
        print(f"Unknown command: {cmd}. Use: {', '.join(dispatch)}")
        sys.exit(1)
    dispatch[cmd](results_dir)
