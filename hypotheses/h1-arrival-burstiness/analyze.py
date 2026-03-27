#!/usr/bin/env python3
"""
analyze.py — Arrival Burstiness Experiment (H1)

Loads the 18 result JSON files, computes paired comparisons between smooth and bursty
conditions at each rate level, runs statistical tests, compares against Kingman's
queueing theory prediction, and prints a full report.

Usage: python3 analyze.py [--results-dir output/]
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional
import math

_scipy_stats = None
HAS_SCIPY = False
try:
    from scipy import stats as _scipy_stats  # type: ignore[import-untyped]
    HAS_SCIPY = True
except ImportError:
    print("WARNING: scipy not available — skipping paired t-tests. Install with: pip install scipy", file=sys.stderr)


# ─── Data loading ────────────────────────────────────────────────────────────

RATES  = [5, 12, 18, 21]
# Observed saturation throughput for qwen3-14b/H100 with this workload profile.
# Determined by running at 50+ req/s (above saturation) and reading responses_per_sec.
SATURATION_RPS = 22.5
SEEDS  = [42, 123, 456]
CONDITIONS = ["smooth", "bursty"]

METRICS = [
    ("ttft_mean_ms",           "TTFT Mean (ms)"),
    ("ttft_p99_ms",            "TTFT p99  (ms)"),
    ("e2e_mean_ms",            "E2E Mean  (ms)"),
    ("e2e_p99_ms",             "E2E p99   (ms)"),
    ("scheduling_delay_p99_ms","Sched Delay p99 (ms)"),
    ("responses_per_sec",      "Throughput (req/s)"),
    ("completed_requests",     "Completed Requests"),
    ("timed_out_requests",     "Timed-Out Requests"),
]


def load_results(results_dir: Path) -> dict:
    """Load all result files into a nested dict: results[condition][rate][seed] = dict."""
    data = {c: {r: {} for r in RATES} for c in CONDITIONS}
    missing = []

    for condition in CONDITIONS:
        for rate in RATES:
            for seed in SEEDS:
                path = results_dir / f"{condition}_rate{rate}_seed{seed}.json"
                if path.exists():
                    with open(path) as f:
                        data[condition][rate][seed] = json.load(f)
                else:
                    missing.append(str(path))

    if missing:
        print(f"\nWARNING: {len(missing)} result file(s) not found:")
        for p in missing:
            print(f"  {p}")
        print()

    return data


# ─── Statistical helpers ─────────────────────────────────────────────────────

def mean(values: list) -> Optional[float]:
    return sum(values) / len(values) if values else None


def stdev(values: list) -> Optional[float]:
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def paired_ttest(a: list, b: list):
    """Returns (t_stat, p_value) or (None, None) if unavailable."""
    if not HAS_SCIPY or _scipy_stats is None or len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return None, None
    try:
        result = _scipy_stats.ttest_rel(b, a)  # b - a: does bursty > smooth?
        return result.statistic, result.pvalue
    except Exception:
        return None, None


def kingman_ratio(cv_a: float, cv_s: float = 1.5) -> float:
    """
    Kingman's G/G/1 prediction for the ratio E[W_bursty] / E[W_smooth].
    cv_s=1.5 is a reasonable estimate for LLM inference service time variance
    (prefill dominated by input length variance + decode by output length).
    """
    return (cv_a**2 + cv_s**2) / (1.0 + cv_s**2)


# ─── Formatting ──────────────────────────────────────────────────────────────

def fmt_val(v: Optional[float], decimals: int = 1) -> str:
    if v is None:
        return "  N/A  "
    return f"{v:{8}.{decimals}f}"


def fmt_ratio(ratio: Optional[float]) -> str:
    if ratio is None:
        return "  N/A  "
    sign = "+" if ratio >= 1.0 else ""
    pct = (ratio - 1.0) * 100
    return f"{sign}{pct:+6.1f}%"


def fmt_pval(p: Optional[float]) -> str:
    if p is None:
        return "  N/A  "
    if p < 0.001:
        return "p<0.001***"
    if p < 0.01:
        return f"p={p:.3f} **"
    if p < 0.05:
        return f"p={p:.3f}  *"
    return f"p={p:.3f}   "


# ─── Main analysis ───────────────────────────────────────────────────────────

def analyze(results_dir: Path):
    data = load_results(results_dir)

    print("=" * 78)
    print(" H1: ARRIVAL BURSTINESS EXPERIMENT — ANALYSIS REPORT")
    print(" Smooth (Poisson CV=1)  vs  Bursty (Gamma CV=3)")
    print(" Model: qwen/qwen3-14b  |  Instances: 1  |  Seeds: 3")
    print("=" * 78)

    # ── Per-rate summary tables ──────────────────────────────────────────────
    rate_summaries = {}  # rate -> {metric -> {smooth_mean, bursty_mean, ratio, p}}

    for rate in RATES:
        print(f"\n{'─'*78}")
        print(f" RATE = {rate} req/s")
        print(f"{'─'*78}")

        smooth_rps = [data["smooth"][rate][s].get("responses_per_sec", 0) for s in SEEDS if s in data["smooth"][rate]]
        bursty_rps = [data["bursty"][rate][s].get("responses_per_sec", 0) for s in SEEDS if s in data["bursty"][rate]]

        smooth_completed = [data["smooth"][rate][s].get("completed_requests", 0) for s in SEEDS if s in data["smooth"][rate]]
        bursty_completed = [data["bursty"][rate][s].get("completed_requests", 0) for s in SEEDS if s in data["bursty"][rate]]

        print(f"\n  Completed requests:  smooth={smooth_completed}  bursty={bursty_completed}")
        if smooth_rps and bursty_rps:
            print(f"  Actual throughput:   smooth={[f'{v:.1f}' for v in smooth_rps]} req/s  bursty={[f'{v:.1f}' for v in bursty_rps]} req/s")

        # True utilization = nominal_rate / saturation_rate
        util_est = rate / SATURATION_RPS
        print(f"  True utilization: ρ = {rate}/{SATURATION_RPS} ≈ {util_est:.2f}")

        print()
        print(f"  {'Metric':<28} {'Smooth (mean±σ)':<22} {'Bursty (mean±σ)':<22} {'Ratio':>8}  {'p-value':>12}")
        print(f"  {'-'*28} {'-'*22} {'-'*22} {'-'*8}  {'-'*12}")

        rate_summaries[rate] = {}

        for key, label in METRICS:
            smooth_vals = [data["smooth"][rate][s].get(key, 0) for s in SEEDS if s in data["smooth"][rate] and key in data["smooth"][rate][s]]
            bursty_vals = [data["bursty"][rate][s].get(key, 0) for s in SEEDS if s in data["bursty"][rate] and key in data["bursty"][rate][s]]

            if not smooth_vals or not bursty_vals:
                continue

            sm = mean(smooth_vals)
            ss = stdev(smooth_vals)
            bm = mean(bursty_vals)
            bs = stdev(bursty_vals)

            ratio = (bm / sm) if (sm is not None and bm is not None and sm > 0) else None
            _, p_val = paired_ttest(smooth_vals, bursty_vals)

            ss_str = f"±{ss:.1f}" if ss is not None else ""
            bs_str = f"±{bs:.1f}" if bs is not None else ""

            smooth_str = f"{sm:.1f}{ss_str}" if sm is not None else "N/A"
            bursty_str = f"{bm:.1f}{bs_str}" if bm is not None else "N/A"
            ratio_str  = fmt_ratio(ratio)
            p_str      = fmt_pval(p_val)

            print(f"  {label:<28} {smooth_str:<22} {bursty_str:<22} {ratio_str:>8}  {p_str:>12}")

            rate_summaries[rate][key] = dict(
                smooth_mean=sm, smooth_std=ss,
                bursty_mean=bm, bursty_std=bs,
                ratio=ratio, p_val=p_val
            )

    # ── Utilization trend analysis ───────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(" UTILIZATION TREND: Does effect size grow with rate?")
    print(f"{'─'*78}")
    print()
    print(f" {'Rate':>6}  {'TTFT p99 Ratio':>14}  {'E2E p99 Ratio':>13}  {'SchedDelay p99 Ratio':>20}  {'Kingman CV=3':>12}")
    print(f" {'-'*6}  {'-'*14}  {'-'*13}  {'-'*20}  {'-'*12}")

    kingman_pred = kingman_ratio(cv_a=3.0, cv_s=1.5)

    for rate in RATES:
        s = rate_summaries.get(rate, {})

        ttft_r  = s.get("ttft_p99_ms",            {}).get("ratio")
        e2e_r   = s.get("e2e_p99_ms",             {}).get("ratio")
        sched_r = s.get("scheduling_delay_p99_ms", {}).get("ratio")

        def fmt(r):
            if r is None:
                return "N/A".rjust(10)
            return f"{r:.2f}x ({fmt_ratio(r).strip()})".rjust(14)

        print(f" {rate:>6}  {fmt(ttft_r)}  {fmt(e2e_r)}  {fmt(sched_r).rjust(20)}  {kingman_pred:.2f}x (predicted)")

    print()
    print(f" Kingman's G/G/1 prediction (CV_a=3, CV_s=1.5): {kingman_pred:.2f}x higher queueing delay")
    print(f" Note: Kingman applies to queueing delay; TTFT/E2E include base service time,")
    print(f"       so the measured ratio will be lower (especially at low utilization).")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(" VERDICT")
    print(f"{'='*78}")

    # Collect TTFT p99 ratios across rates
    ttft_p99_ratios = [
        rate_summaries.get(r, {}).get("ttft_p99_ms", {}).get("ratio")
        for r in RATES
    ]
    ttft_p99_ratios = [r for r in ttft_p99_ratios if r is not None]

    # Check: does the hypothesis hold?
    # Also collect TTFT mean ratios (more stable than p99 for 3-seed experiments)
    ttft_mean_ratios = [
        rate_summaries.get(r, {}).get("ttft_mean_ms", {}).get("ratio")
        for r in RATES
    ]
    ttft_mean_ratios = [r for r in ttft_mean_ratios if r is not None]

    if ttft_p99_ratios:
        # Hypothesis confirmed if ALL ratios are well above 20% threshold at ρ >= 0.5
        high_util_rates = [r for r in RATES if r / SATURATION_RPS >= 0.5]
        high_util_p99_ratios = [
            rate_summaries.get(r, {}).get("ttft_p99_ms", {}).get("ratio")
            for r in high_util_rates
        ]
        high_util_p99_ratios = [r for r in high_util_p99_ratios if r is not None]

        all_above_threshold = all(r >= 1.20 for r in high_util_p99_ratios) if high_util_p99_ratios else False
        mean_monotone = all(ttft_mean_ratios[i] <= ttft_mean_ratios[i+1] for i in range(len(ttft_mean_ratios)-1)) if len(ttft_mean_ratios) > 1 else True
        any_affected = any(r >= 1.20 for r in ttft_p99_ratios)

        if all_above_threshold:
            verdict_str = "CONFIRMED"
        elif not any_affected:
            verdict_str = "REFUTED"
        else:
            verdict_str = "PARTIALLY CONFIRMED"

        print(f"\n  Verdict: {verdict_str}")
        print(f"  TTFT p99 ratios (bursty/smooth):  {[f'{r:.2f}x' for r in ttft_p99_ratios]}")
        print(f"  TTFT mean ratios (bursty/smooth): {[f'{r:.2f}x' for r in ttft_mean_ratios]}")
        if len(ttft_mean_ratios) > 1:
            print(f"  TTFT mean effect is {'monotonically increasing ✓' if mean_monotone else 'NOT monotonically increasing'} with rate")
        print()

    # Summary narrative
    print("  Summary:")
    print("  - Bursty arrivals (Gamma CV=3) should produce higher TTFT/E2E vs smooth (Poisson)")
    print("  - The magnitude should grow with utilization (Kingman's prediction)")
    print("  - At low utilization (ρ≪1), queueing delays are negligible for both conditions")
    print("  - The effect becomes pronounced as ρ approaches saturation")
    print()
    print("  See FINDINGS.md for full interpretation and citations.")
    print(f"{'='*78}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze H1 arrival burstiness experiment results")
    parser.add_argument("--results-dir", default="output", help="Directory containing result JSON files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        print("Run ./run.sh first to generate results.", file=sys.stderr)
        sys.exit(1)

    analyze(results_dir)
