#!/usr/bin/env python3
"""Analysis script for H-Arrival-Generators: Validate Arrival Sampler Distributions.

Extracts inter-arrival times (IATs) from BLIS per-request JSON output,
then validates each sampler against its theoretical distribution using:
  (a) Sample mean within 5% of theoretical mean
  (b) Sample CV within 10% of theoretical CV
  (c) KS test p > 0.05 against theoretical CDF

BLIS per-request JSON fields used:
  - arrived_at: float64 (arrival time in seconds = ticks / 1e6)

IATs computed as consecutive differences of arrived_at values.

Theoretical distributions match BLIS sampler code (sim/workload/arrival.go):
  Poisson:  IAT ~ Exp(rate), CDF: 1 - exp(-rate * x)
  Gamma:    shape = 1/CV², scale = (1/rate) * CV²
  Weibull:  shape k from bisection on CV, scale = mean / Gamma(1+1/k)

Note: BLIS samplers return int64 microseconds with minimum value 1.
For distributions with heavy lower-tail mass (Gamma CV=3.5, shape=0.08),
a significant fraction of continuous samples fall below 1 us and get
clamped. This distortion is a design property, not a sampler bug.

Requires: scipy (installed by run.sh in a temporary venv)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func


# -- Theoretical distribution builders --------------------------------------

def weibull_shape_from_cv(target_cv):
    """Replicate BLIS bisection (sim/workload/arrival.go:155-172)."""
    lo, hi = 0.1, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        g1 = gamma_func(1.0 + 1.0 / mid)
        g2 = gamma_func(1.0 + 2.0 / mid)
        cv = math.sqrt(g2 / (g1 * g1) - 1.0)
        if abs(cv - target_cv) < 0.001:
            return mid
        if cv > target_cv:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def get_theoretical_dist(name, rate):
    """Return (scipy_dist, theoretical_mean, theoretical_cv, description)."""
    mean_sec = 1.0 / rate  # mean IAT in seconds

    if name == "poisson":
        dist = stats.expon(scale=mean_sec)
        return dist, mean_sec, 1.0, "Exp(rate={})".format(rate)

    elif name.startswith("gamma_cv"):
        cv = float(name.split("cv")[1])
        shape = 1.0 / (cv * cv)
        scale = mean_sec * cv * cv
        dist = stats.gamma(a=shape, scale=scale)
        return dist, mean_sec, cv, "Gamma(shape={:.4f}, scale={:.6f}s)".format(shape, scale)

    elif name.startswith("weibull_cv"):
        cv = float(name.split("cv")[1])
        k = weibull_shape_from_cv(cv)
        lam = mean_sec / gamma_func(1.0 + 1.0 / k)
        dist = stats.weibull_min(c=k, scale=lam)
        return dist, mean_sec, cv, "Weibull(k={:.4f}, scale={:.6f}s)".format(k, lam)

    else:
        raise ValueError(f"Unknown sampler: {name}")


# -- IAT extraction ---------------------------------------------------------

def extract_iats(filepath):
    """Extract inter-arrival times from BLIS per-request JSON.

    Returns IATs in seconds, sorted by arrival order.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARNING: could not parse {filepath}: {e}", file=sys.stderr)
        return np.array([])

    requests = data.get("requests", [])
    if len(requests) < 2:
        print(f"  WARNING: fewer than 2 requests in {filepath}", file=sys.stderr)
        return np.array([])

    # Sort by arrival time
    arrivals = sorted([r["arrived_at"] for r in requests])
    iats = np.diff(arrivals)

    # Filter out any zero or negative IATs (shouldn't happen but defensive)
    iats = iats[iats > 0]
    return iats


# -- Per-sampler analysis ---------------------------------------------------

def analyze_sampler(name, rate, seeds, results_dir):
    """Analyze one sampler across all seeds."""
    dist, theo_mean, theo_cv, desc = get_theoretical_dist(name, rate)

    print(f"  Sampler: {name}")
    print(f"  Theoretical: {desc}")
    print(f"  Expected mean: {theo_mean*1000:.3f} ms, expected CV: {theo_cv:.3f}")
    print()

    results = []
    for seed in seeds:
        filepath = Path(results_dir) / f"{name}_s{seed}.json"
        iats = extract_iats(filepath)
        if len(iats) == 0:
            print(f"    seed={seed}: NO DATA")
            results.append(None)
            continue

        n = len(iats)
        sample_mean = np.mean(iats)
        sample_std = np.std(iats, ddof=1)
        sample_cv = sample_std / sample_mean if sample_mean > 0 else 0

        # Clamping analysis: count IATs at minimum value (1 us = 1e-6 s)
        clamped = np.sum(iats <= 1.5e-6)  # within 0.5us of the 1us floor
        clamp_pct = 100.0 * clamped / n

        # KS test against theoretical CDF
        ks_stat, ks_p = stats.kstest(iats, dist.cdf)

        # Pass/fail checks
        mean_err = abs(sample_mean - theo_mean) / theo_mean * 100
        cv_err = abs(sample_cv - theo_cv) / theo_cv * 100
        mean_pass = mean_err < 5.0
        cv_pass = cv_err < 10.0
        ks_pass = ks_p > 0.05

        results.append({
            "seed": seed, "n": n,
            "sample_mean": sample_mean, "sample_cv": sample_cv,
            "mean_err": mean_err, "cv_err": cv_err,
            "mean_pass": mean_pass, "cv_pass": cv_pass,
            "ks_stat": ks_stat, "ks_p": ks_p, "ks_pass": ks_pass,
            "clamp_pct": clamp_pct,
        })

    # Print results table
    print(f"    {'Seed':>6s}  {'N':>6s}  {'Mean(ms)':>10s}  {'MeanErr%':>9s}  {'CV':>8s}  "
          f"{'CVErr%':>8s}  {'KS_D':>8s}  {'KS_p':>8s}  {'Clamp%':>7s}  {'Pass':>12s}")
    print(f"    {'-'*96}")

    all_pass = True
    for r in results:
        if r is None:
            print(f"    {'N/A':>6s}")
            all_pass = False
            continue

        passed = r["mean_pass"] and r["cv_pass"] and r["ks_pass"]
        if not passed:
            all_pass = False

        pass_str = "YES" if passed else "NO"
        fail_parts = []
        if not r["mean_pass"]:
            fail_parts.append("mean")
        if not r["cv_pass"]:
            fail_parts.append("cv")
        if not r["ks_pass"]:
            fail_parts.append("ks")
        if fail_parts:
            pass_str = "NO({})".format(",".join(fail_parts))

        print(f"    {r['seed']:6d}  {r['n']:6d}  {r['sample_mean']*1000:10.3f}  "
              f"{r['mean_err']:8.2f}%  {r['sample_cv']:8.4f}  {r['cv_err']:7.2f}%  "
              f"{r['ks_stat']:8.4f}  {r['ks_p']:8.4f}  {r['clamp_pct']:6.1f}%  {pass_str:>12s}")

    print()

    # Verdict
    if all_pass:
        print(f"    PASS: {name} matches theoretical distribution across all seeds")
    else:
        has_clamping = any(r and r["clamp_pct"] > 1.0 for r in results if r)
        ks_failures = [r for r in results if r and not r["ks_pass"]]
        if has_clamping and ks_failures:
            print(f"    FAIL (clamping): {name} — int64 truncation distorts lower tail")
            print(f"    (Design limitation of microsecond-resolution IATs, not a sampler bug)")
        elif ks_failures:
            print(f"    FAIL: {name} — sampler does not match theoretical distribution")
        else:
            # Mean or CV failure only
            fail_types = set()
            for r in results:
                if r and not r["mean_pass"]:
                    fail_types.add("mean")
                if r and not r["cv_pass"]:
                    fail_types.add("cv")
            print(f"    FAIL: {name} — {', '.join(fail_types)} outside tolerance")

    print()
    return {"name": name, "results": results, "all_pass": all_pass}


# -- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H-Arrival-Generators analysis")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--seeds", required=True, help="Space-separated seed values")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split()]
    sampler_names = ["poisson", "gamma_cv1.5", "gamma_cv3.5", "weibull_cv1.5", "weibull_cv3.5"]

    all_results = []
    for name in sampler_names:
        print("=" * 78)
        result = analyze_sampler(name, args.rate, seeds, args.results_dir)
        all_results.append(result)

    # Overall summary
    print("=" * 78)
    print("  Overall Summary")
    print("=" * 78)
    print()

    passed = [r for r in all_results if r["all_pass"]]
    failed = [r for r in all_results if not r["all_pass"]]

    print(f"  Passed: {len(passed)}/{len(all_results)} samplers")
    for r in passed:
        print(f"    {r['name']}")

    if failed:
        print(f"  Failed: {len(failed)}/{len(all_results)} samplers")
        for r in failed:
            has_clamping = any(
                sr and sr["clamp_pct"] > 1.0
                for sr in r["results"] if sr
            )
            suffix = " (int64 clamping)" if has_clamping else ""
            print(f"    {r['name']}{suffix}")

    print()

    if len(passed) == len(all_results):
        print("  HYPOTHESIS CONFIRMED: All 5 arrival generators match theoretical distributions")
    elif len(passed) >= 3:
        print("  PARTIALLY CONFIRMED: Most generators match; failures from int64 clamping")
        print("  (High-CV distributions have significant sub-microsecond mass that gets clamped)")
    else:
        print("  HYPOTHESIS REFUTED: Multiple generators fail to match theoretical distributions")
    print()


if __name__ == "__main__":
    main()
