#!/usr/bin/env python3
"""
analyze.py — Statistical analysis for h-joint-slo-optimization experiments.

USAGE
  python3 analyze.py iter0                Print Iter 0 baseline table
  python3 analyze.py iter1                H-main + ablation results for Iter 1
  python3 analyze.py iter2                SLO-priority preemption analysis
  python3 analyze.py iter3                Tiered-LRU analysis
  python3 analyze.py iter4                Tier-budget analysis + fraction sweep
  python3 analyze.py all                  All iterations in sequence

OUTPUT
  Markdown tables suitable for pasting directly into FINDINGS.md.

ASSUMPTIONS
  Results are in results/iter{N}/{label}/seed{42,123,456}.json
  Each JSON file has the structure produced by `blis run --output`:
    {
      "completed_requests": N,
      "injected_requests": N,
      "timed_out_requests": N,
      "slo_metrics": {
        "ttft_p99_ms_by_class": {"critical": F, "standard": F, "sheddable": F},
        "goodput_by_class": {"critical": F, "standard": F, "sheddable": F}
      },
      "kv_metrics": {
        "cache_hit_rate": F,
        "preemption_count": N
      }
    }

  If your blis build uses different field names, update FIELD_MAP below.
"""

import json
import sys
import statistics
from pathlib import Path

# ── Field mapping (adjust if blis output format differs) ────────────────────
def get_critical_p99(d: dict) -> float | None:
    """Extract critical TTFT P99 from result JSON."""
    try:
        return d["slo_metrics"]["ttft_p99_ms_by_class"]["critical"]
    except (KeyError, TypeError):
        pass
    # Fallback: flat structure
    return d.get("critical_ttft_p99_ms") or d.get("ttft_p99_ms")

def get_goodput(d: dict, cls: str) -> float | None:
    try:
        return d["slo_metrics"]["goodput_by_class"][cls]
    except (KeyError, TypeError):
        return d.get(f"{cls}_goodput")

def get_kv_hit_rate(d: dict) -> float | None:
    try:
        return d["kv_metrics"]["cache_hit_rate"]
    except (KeyError, TypeError):
        return d.get("kv_cache_hit_rate")

def get_preemption_count(d: dict) -> int:
    try:
        return d["kv_metrics"]["preemption_count"]
    except (KeyError, TypeError):
        return d.get("preemption_count", 0)

# ── Utilities ────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
SEEDS = [42, 123, 456]

def load_results(subpath: str) -> list[dict]:
    """Load all seed results from results/<subpath>/seed{N}.json."""
    results = []
    for seed in SEEDS:
        p = RESULTS_DIR / subpath / f"seed{seed}.json"
        if p.exists():
            with open(p) as f:
                results.append(json.load(f))
        else:
            print(f"  WARNING: missing {p}", file=sys.stderr)
    return results

def mean_std(values: list[float | None]) -> tuple[float, float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)

def pct_change(baseline: float, treatment: float) -> float:
    if baseline == 0:
        return float("nan")
    return (baseline - treatment) / baseline * 100  # positive = improvement

def fmt(v) -> str:
    if v is None or (isinstance(v, float) and (v != v)):  # nan check
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)

def print_table(headers: list[str], rows: list[list]) -> None:
    widths = [max(len(str(h)), max((len(fmt(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    hdr = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(fmt(v).ljust(w) for v, w in zip(row, widths)) + " |")
    print()

# ── Per-iteration analyses ────────────────────────────────────────────────────

def analyze_iter0():
    print("## Iteration 0: Baseline\n")
    results = load_results("iter0")
    if not results:
        print("No results found.\n"); return

    rows = []
    p99s, std_goods, shed_goods = [], [], []
    for r, seed in zip(results, SEEDS):
        p99 = get_critical_p99(r)
        sg  = get_goodput(r, "standard")
        shg = get_goodput(r, "sheddable")
        pc  = get_preemption_count(r)
        hr  = get_kv_hit_rate(r)
        rows.append([seed, p99, sg, shg, pc, hr])
        if p99 is not None: p99s.append(p99)
        if sg  is not None: std_goods.append(sg)
        if shg is not None: shed_goods.append(shg)

    p99_m, p99_s = mean_std(p99s)
    sg_m,  sg_s  = mean_std(std_goods)
    shg_m, shg_s = mean_std(shed_goods)
    rows.append(["Mean±σ", f"{p99_m:.1f}±{p99_s:.1f}", f"{sg_m:.1f}±{sg_s:.1f}",
                 f"{shg_m:.1f}±{shg_s:.1f}", "—", "—"])

    print_table(["Seed", "Critical P99 (ms)", "Std Goodput", "Shed Goodput",
                 "Preemptions", "KV Hit Rate"], rows)
    print(f"Reference values: critical_p99_baseline={p99_m:.1f}ms "
          f"std_goodput_baseline={sg_m:.1f} shed_goodput_baseline={shg_m:.1f}\n")


def analyze_iter1():
    print("## Iteration 1: Joint Composition\n")
    compound  = load_results("iter1/compound")
    defaults  = load_results("iter1/blis-defaults")

    if not compound or not defaults:
        print("Missing compound or blis-defaults results.\n"); return

    print("### H-main: Compound vs BLIS defaults\n")
    rows = []
    for seed, c, d in zip(SEEDS, compound, defaults):
        cp99 = get_critical_p99(c)
        dp99 = get_critical_p99(d)
        imp  = pct_change(dp99, cp99) if cp99 and dp99 else None
        rows.append([seed, cp99, dp99, imp])
    cp99s = [get_critical_p99(r) for r in compound if get_critical_p99(r)]
    dp99s = [get_critical_p99(r) for r in defaults if get_critical_p99(r)]
    cm, cs = mean_std(cp99s); dm, ds = mean_std(dp99s)
    rows.append(["Mean±σ", f"{cm:.1f}±{cs:.1f}", f"{dm:.1f}±{ds:.1f}",
                 f"{pct_change(dm, cm):.1f}%"])
    print_table(["Seed", "Compound P99 (ms)", "Default P99 (ms)", "Improvement %"], rows)
    imp_m = pct_change(dm, cm)
    verdict = "✅ CONFIRMED" if imp_m > 40 else ("⚠️ BELOW 40% THRESHOLD" if imp_m > 15 else "❌ BELOW 15% MIN")
    print(f"H-main verdict: {verdict} (improvement={imp_m:.1f}%, threshold >40%)\n")

    print("### H-ablation: Component contributions\n")
    ablations = [
        ("abl-routing",     "Round-robin routing", 15),
        ("abl-scheduling",  "FCFS scheduler",      20),
        ("abl-nochunk",     "All tiers chunked",   10),
        ("abl-admission",   "Always-admit",        30),
    ]
    rows = []
    for subdir, label, threshold in ablations:
        abl_results = load_results(f"iter1/{subdir}")
        if not abl_results:
            rows.append([label, "—", "—", "—", f"> {threshold}%", "?"]);  continue
        abl_p99s = [get_critical_p99(r) for r in abl_results if get_critical_p99(r)]
        am, _ = mean_std(abl_p99s)
        degradation = pct_change(cm, am)  # how much worse vs compound
        fast_fail = "DROP" if degradation < 5 else ""
        verdict_sym = "✅" if degradation >= threshold else "❌"
        rows.append([label, f"{am:.1f}", f"{degradation:.1f}%",
                     f"> {threshold}%", f"{verdict_sym} {fast_fail}"])
    print_table(["Arm", "Ablation P99 (ms)", "Degradation", "Threshold", "Pass?"], rows)


def analyze_iter2():
    print("## Iteration 2: SLO-Priority Preemption\n")
    treatment = load_results("iter2/treatment")
    ablation  = load_results("iter2/ablation")

    if not treatment or not ablation:
        print("Missing treatment or ablation results.\n"); return

    print("### H-main\n")
    rows = []
    for seed, t, a in zip(SEEDS, treatment, ablation):
        tp99 = get_critical_p99(t)
        ap99 = get_critical_p99(a)
        imp  = pct_change(ap99, tp99) if tp99 and ap99 else None
        rows.append([seed, tp99, ap99, imp])
    tp99s = [get_critical_p99(r) for r in treatment if get_critical_p99(r)]
    ap99s = [get_critical_p99(r) for r in ablation  if get_critical_p99(r)]
    tm, ts = mean_std(tp99s); am, as_ = mean_std(ap99s)
    imp_m = pct_change(am, tm)
    rows.append(["Mean±σ", f"{tm:.1f}±{ts:.1f}", f"{am:.1f}±{as_:.1f}", f"{imp_m:.1f}%"])
    print_table(["Seed", "SLO-priority P99", "LIFO (ablation) P99", "Improvement %"], rows)
    verdict = "✅ CONFIRMED" if imp_m > 15 else ("⚠️ MARGINAL" if imp_m > 5 else "❌ FAST-FAIL")
    print(f"H-main verdict: {verdict} (improvement={imp_m:.1f}%, threshold >15%)\n")

    print("### H-zero-sum: Standard goodput\n")
    rows = []
    for seed, t, a in zip(SEEDS, treatment, ablation):
        tsg = get_goodput(t, "standard")
        asg = get_goodput(a, "standard")
        deg = pct_change(asg, tsg) if tsg and asg else None  # positive = degradation
        rows.append([seed, tsg, asg, deg])
    tsgs = [get_goodput(r, "standard") for r in treatment if get_goodput(r, "standard")]
    asgs = [get_goodput(r, "standard") for r in ablation  if get_goodput(r, "standard")]
    tgm, _ = mean_std(tsgs); agm, _ = mean_std(asgs)
    deg_m = pct_change(agm, tgm)
    rows.append(["Mean", tgm, agm, f"{deg_m:.1f}%"])
    print_table(["Seed", "Std Goodput (treatment)", "Std Goodput (ablation)", "Degradation %"], rows)
    v2 = "✅ NON-ZERO-SUM" if deg_m <= 20 else "❌ ZERO-SUM (>20% degradation)"
    print(f"H-zero-sum verdict: {v2}\n")


def analyze_iter3():
    print("## Iteration 3: Tiered LRU KV Eviction\n")
    treatment = load_results("iter3/treatment")
    ablation  = load_results("iter3/ablation")

    if not treatment or not ablation:
        print("Missing treatment or ablation results.\n"); return

    print("### H-main: P99\n")
    tp99s = [get_critical_p99(r) for r in treatment if get_critical_p99(r)]
    ap99s = [get_critical_p99(r) for r in ablation  if get_critical_p99(r)]
    tm, ts = mean_std(tp99s); am, _ = mean_std(ap99s)
    imp_m = pct_change(am, tm)
    print_table(["", "Tiered-LRU P99 (ms)", "Single-list P99 (ms)", "Improvement %"],
                [["Mean", f"{tm:.1f}±{ts:.1f}", f"{am:.1f}", f"{imp_m:.1f}%"]])
    verdict = "✅ CONFIRMED" if imp_m > 15 else ("⚠️ MARGINAL" if imp_m > 5 else "❌ FAST-FAIL")
    print(f"H-main verdict: {verdict} (improvement={imp_m:.1f}%, threshold >15%)\n")

    print("### Cache hit rate\n")
    thrs = [get_kv_hit_rate(r) for r in treatment if get_kv_hit_rate(r)]
    ahrs = [get_kv_hit_rate(r) for r in ablation  if get_kv_hit_rate(r)]
    if thrs and ahrs:
        thrm, _ = mean_std(thrs); ahrm, _ = mean_std(ahrs)
        hr_imp = pct_change(ahrm, thrm)
        print_table(["", "Tiered-LRU hit rate", "Single-list hit rate", "Improvement %"],
                    [["Mean", f"{thrm:.3f}", f"{ahrm:.3f}", f"{hr_imp:.1f}%"]])
        verdict = "✅ CONFIRMED" if hr_imp > 10 else "❌ BELOW 10% THRESHOLD"
        print(f"Hit rate verdict: {verdict}\n")


def analyze_iter4():
    print("## Iteration 4: Tier-Budget Batch Formation\n")
    treatment = load_results("iter4/treatment")
    ablation  = load_results("iter4/ablation-equal-share")

    if not treatment or not ablation:
        print("Missing treatment or ablation results.\n"); return

    print("### H-main: f_c=0.50 vs Iter 3 compound\n")
    # Note: Iter 3 treatment is the baseline for Iter 4
    iter3 = load_results("iter3/treatment")
    if not iter3:
        print("  (Iter 3 results not found — cannot compute improvement over Iter 3)\n")
    else:
        i3p99s = [get_critical_p99(r) for r in iter3 if get_critical_p99(r)]
        i3m, _ = mean_std(i3p99s)
        tp99s = [get_critical_p99(r) for r in treatment if get_critical_p99(r)]
        tm, ts = mean_std(tp99s)
        imp_m = pct_change(i3m, tm)
        print_table(["", "Tier-budget P99 (ms)", "Iter3 compound P99 (ms)", "Improvement %"],
                    [["Mean", f"{tm:.1f}±{ts:.1f}", f"{i3m:.1f}", f"{imp_m:.1f}%"]])
        verdict = "✅ CONFIRMED" if imp_m > 10 else ("⚠️ MARGINAL" if imp_m > 5 else "❌ FAST-FAIL")
        print(f"H-main verdict: {verdict} (improvement={imp_m:.1f}%, threshold >10%)\n")

    print("### H-ablation: f_c=0.50 vs f_c=0.333 (equal-share)\n")
    tp99s = [get_critical_p99(r) for r in treatment if get_critical_p99(r)]
    ap99s = [get_critical_p99(r) for r in ablation  if get_critical_p99(r)]
    tm, ts = mean_std(tp99s); am, _ = mean_std(ap99s)
    deg_m = pct_change(am, tm)  # how much worse with equal-share
    print_table(["", "f_c=0.50 P99", "f_c=0.333 P99", "Degradation (equal-share) %"],
                [["Mean", f"{tm:.1f}", f"{am:.1f}", f"{deg_m:.1f}%"]])
    verdict = "✅ CONFIRMED" if deg_m > 8 else "❌ FRACTION-INSENSITIVE"
    print(f"H-ablation verdict: {verdict} (degradation={deg_m:.1f}%, threshold >8%)\n")

    print("### H-robustness: fraction sweep\n")
    sweep_dirs = {
        "0.20": "iter4/sweep-fc020",
        "0.30": "iter4/sweep-fc030",
        "0.40": "iter4/sweep-fc040",
        "0.50": "iter4/sweep-fc050",
        "0.60": "iter4/sweep-fc060",
        "0.70": "iter4/sweep-fc070",
    }
    rows = []
    prev_p99 = None
    for fc, subdir in sweep_dirs.items():
        res = load_results(subdir)
        if not res:
            rows.append([fc, "—", "—", "—"]); continue
        p99s = [get_critical_p99(r) for r in res if get_critical_p99(r)]
        sgs  = [get_goodput(r, "standard") for r in res if get_goodput(r, "standard")]
        pm, _ = mean_std(p99s); sgm, _ = mean_std(sgs)
        mono = "✅" if prev_p99 is None or pm <= prev_p99 else "❌ non-monotone"
        rows.append([fc, f"{pm:.1f}", f"{sgm:.3f}", mono])
        prev_p99 = pm
    print_table(["f_c", "Critical P99 (ms)", "Std Goodput", "Monotone?"], rows)


def analyze_all():
    analyze_iter0()
    analyze_iter1()
    analyze_iter2()
    analyze_iter3()
    analyze_iter4()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    dispatch = {
        "iter0": analyze_iter0,
        "iter1": analyze_iter1,
        "iter2": analyze_iter2,
        "iter3": analyze_iter3,
        "iter4": analyze_iter4,
        "all":   analyze_all,
    }
    if cmd not in dispatch:
        print("Usage: python3 analyze.py {iter0|iter1|iter2|iter3|iter4|all}")
        sys.exit(1)
    dispatch[cmd]()
