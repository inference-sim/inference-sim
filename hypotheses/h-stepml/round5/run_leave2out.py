#!/usr/bin/env python3
"""Round 5 Supplementary: Leave-Two-Out (L2O) validation of power law formulas.

Tests whether the power law holds when trained on only 2 models and predicting 2.
With 4 models, there are C(4,2) = 6 folds.

This is a harder test than LOMO (train on 3, predict 1) because:
- 2-parameter power law from 2 points has 0 degrees of freedom (exactly determined)
- The fit is pure interpolation/extrapolation with no error averaging
"""

import json
import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.optimize import curve_fit

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_SCRIPT_DIR, "..", "shared")
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _SHARED_DIR)

from data_loader import load_all_experiments
from validate_r5 import (
    list_experiments, validate_all, normalize_model, load_ground_truth,
)

# ─── Model Metadata ───

MODEL_METADATA = {
    "llama-2-7b": {"params_b": 7.0, "tp": 1, "active_b": 7.0},
    "codellama-34b": {"params_b": 34.0, "tp": 2, "active_b": 34.0},
    "llama-2-70b": {"params_b": 70.0, "tp": 4, "active_b": 70.0},
    "mixtral-8x7b-v0-1": {"params_b": 46.7, "tp": 2, "active_b": 12.9},
}

CANONICAL_MODELS = list(MODEL_METADATA.keys())

R4_COEFFICIENTS = {
    "llama-2-7b": {"alpha": [27129, 0, 0], "beta": [9741, 0.30, 13.6]},
    "codellama-34b": {"alpha": [47618, 0, 0], "beta": [14196, 0.0, 25.8]},
    "llama-2-70b": {"alpha": [78888, 0, 0], "beta": [17992, 1.22, 35.2]},
    "mixtral-8x7b-v0-1": {"alpha": [62767, 0, 0], "beta": [18921, 0.69, 8.8]},
}


def power_law(x, a, b):
    return a * np.power(x, b)


def fit_power_law_2pt(x_vals, y_vals):
    """Fit a*x^b from exactly 2 points (analytically determined)."""
    x1, x2 = x_vals
    y1, y2 = y_vals
    if x1 <= 0 or x2 <= 0 or y1 <= 0 or y2 <= 0 or x1 == x2:
        # Fallback to curve_fit
        try:
            popt, _ = curve_fit(power_law, x_vals, y_vals, p0=[5000, 0.3], maxfev=10000)
            return popt
        except RuntimeError:
            return [y_vals.mean(), 0.0]
    b = np.log(y2 / y1) / np.log(x2 / x1)
    a = y1 / (x1 ** b)
    return [a, b]


def compute_target_steps_and_ttfts():
    """Per-model: target_step = (E2E - TTFT) / output_len, and mean TTFT."""
    model_steps = defaultdict(list)
    model_ttfts = defaultdict(list)
    for exp in list_experiments():
        gt = load_ground_truth(exp["dirpath"])
        model = normalize_model(exp["model"])
        e2e_us = gt["e2e_mean_s"] * 1e6
        ttft_us = gt["ttft_mean_s"] * 1e6
        output_len = gt["output_len_mean"]
        if output_len > 0:
            model_steps[model].append((e2e_us - ttft_us) / output_len)
        if ttft_us > 0:
            model_ttfts[model].append(ttft_us)
    targets = {m: float(np.mean(v)) for m, v in model_steps.items()}
    ttfts = {m: float(np.mean(v)) for m, v in model_ttfts.items()}
    return targets, ttfts


def compute_step_data_stats(step_df):
    """Per-model average batch composition from step data."""
    import pandas as pd
    stats = {}
    for model in step_df["model"].unique():
        mdf = step_df[step_df["model"] == model]
        decode = mdf.get("batch.decode_tokens", pd.Series(0, index=mdf.index)).fillna(0).astype(float)
        dur = mdf["step.duration_us"].dropna().astype(float)
        stats[normalize_model(model)] = {
            "avg_decode_batch": float(decode.mean()),
            "avg_dur": float(dur.mean()),
        }
    return stats


# ─── Idea 1: Analytical Overhead (power law on R4 coefficients) ───

def idea1_l2o_predict(train_models, all_models):
    """Fit power laws from 2 train models, predict all."""
    params = np.array([MODEL_METADATA[m]["params_b"] for m in train_models])
    beta0s = np.array([R4_COEFFICIENTS[m]["beta"][0] for m in train_models])
    alpha0s = np.array([R4_COEFFICIENTS[m]["alpha"][0] for m in train_models])
    ppg = np.array([MODEL_METADATA[m]["active_b"] / MODEL_METADATA[m]["tp"] for m in train_models])
    beta2s = np.array([R4_COEFFICIENTS[m]["beta"][2] for m in train_models])

    b0_popt = fit_power_law_2pt(params, beta0s)
    a0_popt = fit_power_law_2pt(params, alpha0s)
    b2_popt = fit_power_law_2pt(ppg, beta2s)

    coeffs = {}
    for m in all_models:
        meta = MODEL_METADATA[m]
        ppg_m = meta["active_b"] / meta["tp"]
        coeffs[m] = {
            "alpha": [float(power_law(meta["params_b"], *a0_popt)), 0, 0],
            "beta": [float(power_law(meta["params_b"], *b0_popt)), 0.0,
                     float(power_law(ppg_m, *b2_popt))],
        }
    return coeffs, b0_popt, a0_popt, b2_popt


# ─── Idea 2: Hybrid (metadata overhead + step-derived beta2) ───

def idea2_l2o_predict(train_models, all_models, target_steps, ttfts, step_stats):
    """Fit from 2 train models using E2E-derived targets."""
    params = np.array([MODEL_METADATA[m]["params_b"] for m in train_models])
    targets = np.array([target_steps[m] for m in train_models])
    oh_popt = fit_power_law_2pt(params, targets)

    # beta2 from step data
    ppg_list, b2_list = [], []
    for m in train_models:
        if m in step_stats and step_stats[m]["avg_decode_batch"] > 0:
            ppg_list.append(MODEL_METADATA[m]["active_b"] / MODEL_METADATA[m]["tp"])
            b2_list.append(step_stats[m]["avg_dur"] / step_stats[m]["avg_decode_batch"])
    b2_popt = fit_power_law_2pt(np.array(ppg_list), np.array(b2_list)) if len(ppg_list) == 2 else [5.0, 0.5]

    # alpha0
    a0_params = np.array([MODEL_METADATA[m]["params_b"] for m in train_models if m in ttfts])
    a0_vals = np.array([ttfts[m] for m in train_models if m in ttfts])
    a0_popt = fit_power_law_2pt(a0_params, a0_vals) if len(a0_vals) == 2 else [10000, 0.4]

    coeffs = {}
    for m in all_models:
        meta = MODEL_METADATA[m]
        ppg_m = meta["active_b"] / meta["tp"]
        overhead = float(power_law(meta["params_b"], *oh_popt))
        beta2 = float(power_law(ppg_m, *b2_popt))
        avg_batch = step_stats.get(m, {}).get("avg_decode_batch", 30)
        beta0 = max(overhead - beta2 * avg_batch, 1000)
        alpha0 = float(power_law(meta["params_b"], *a0_popt))
        coeffs[m] = {"alpha": [alpha0, 0, 0], "beta": [beta0, 0.0, beta2]}
    return coeffs, oh_popt, b2_popt, a0_popt


def main():
    t_start = time.time()
    experiments = list_experiments()
    print(f"Found {len(experiments)} experiments\n")

    print("Loading step data...")
    step_df = load_all_experiments()
    print(f"  {len(step_df)} steps\n")

    target_steps, ttfts = compute_target_steps_and_ttfts()
    step_stats = compute_step_data_stats(step_df)

    all_folds = list(combinations(CANONICAL_MODELS, 2))
    print(f"Leave-Two-Out: {len(all_folds)} folds (C(4,2) = 6)\n")

    results = {"idea1_l2o": {}, "idea2_l2o": {}}

    # ── IDEA 1: L2O ──
    print("=" * 70)
    print("IDEA 1: Analytical Overhead — Leave-Two-Out")
    print("=" * 70)

    for train_pair in all_folds:
        holdout = [m for m in CANONICAL_MODELS if m not in train_pair]
        fold_label = f"train=[{','.join(t.split('-')[0] for t in train_pair)}]"
        holdout_label = f"holdout=[{','.join(h.split('-')[0] for h in holdout)}]"

        coeffs, b0p, a0p, b2p = idea1_l2o_predict(list(train_pair), CANONICAL_MODELS)
        print(f"\n  {fold_label} → {holdout_label}")
        print(f"    beta0={b0p[0]:.0f}*x^{b0p[1]:.3f}, beta2={b2p[0]:.2f}*x^{b2p[1]:.3f}")

        for m in holdout:
            c = coeffs[m]
            actual = R4_COEFFICIENTS[m]
            print(f"    {m}: beta0={c['beta'][0]:.0f} (R4={actual['beta'][0]}), "
                  f"beta2={c['beta'][2]:.1f} (R4={actual['beta'][2]}), "
                  f"alpha0={c['alpha'][0]:.0f} (R4={actual['alpha'][0]})")

        fold_results = validate_all(experiments, coeffs, fold_label)
        fold_key = "+".join(sorted(train_pair))

        for h in holdout:
            h_exps = [r for r in fold_results if r["status"] == "ok"
                      and normalize_model(r["model"]) == h]
            if h_exps:
                e2e = np.mean([r["e2e_error"] for r in h_exps]) * 100
                below = sum(1 for r in h_exps if r["e2e_error"] < 0.10)
                status = "PASS" if e2e < 80 else "FAIL"
                print(f"    → {h}: E2E={e2e:.1f}% ({below}/{len(h_exps)} <10%) [{status}]")
                results["idea1_l2o"][f"{fold_key}→{h}"] = {
                    "train": list(train_pair), "holdout": h,
                    "mean_e2e_pct": e2e, "below_10_pct": below,
                    "total": len(h_exps), "pass": e2e < 80,
                }

    # ── IDEA 2: L2O ──
    print("\n" + "=" * 70)
    print("IDEA 2: Normalized Features — Leave-Two-Out")
    print("=" * 70)

    for train_pair in all_folds:
        holdout = [m for m in CANONICAL_MODELS if m not in train_pair]
        fold_label = f"train=[{','.join(t.split('-')[0] for t in train_pair)}]"
        holdout_label = f"holdout=[{','.join(h.split('-')[0] for h in holdout)}]"

        coeffs, ohp, b2p, a0p = idea2_l2o_predict(
            list(train_pair), CANONICAL_MODELS, target_steps, ttfts, step_stats)
        print(f"\n  {fold_label} → {holdout_label}")
        print(f"    overhead={ohp[0]:.0f}*x^{ohp[1]:.3f}, beta2={b2p[0]:.2f}*x^{b2p[1]:.3f}")

        for m in holdout:
            c = coeffs[m]
            actual = R4_COEFFICIENTS[m]
            print(f"    {m}: beta0={c['beta'][0]:.0f} (R4={actual['beta'][0]}), "
                  f"beta2={c['beta'][2]:.1f} (R4={actual['beta'][2]})")

        fold_results = validate_all(experiments, coeffs, fold_label)
        fold_key = "+".join(sorted(train_pair))

        for h in holdout:
            h_exps = [r for r in fold_results if r["status"] == "ok"
                      and normalize_model(r["model"]) == h]
            if h_exps:
                e2e = np.mean([r["e2e_error"] for r in h_exps]) * 100
                below = sum(1 for r in h_exps if r["e2e_error"] < 0.10)
                status = "PASS" if e2e < 80 else "FAIL"
                print(f"    → {h}: E2E={e2e:.1f}% ({below}/{len(h_exps)} <10%) [{status}]")
                results["idea2_l2o"][f"{fold_key}→{h}"] = {
                    "train": list(train_pair), "holdout": h,
                    "mean_e2e_pct": e2e, "below_10_pct": below,
                    "total": len(h_exps), "pass": e2e < 80,
                }

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("LEAVE-TWO-OUT SUMMARY")
    print("=" * 70)

    for idea_key, idea_label in [("idea1_l2o", "Idea 1"), ("idea2_l2o", "Idea 2")]:
        folds = results[idea_key]
        if not folds:
            continue
        e2es = [f["mean_e2e_pct"] for f in folds.values()]
        passes = sum(1 for f in folds.values() if f["pass"])
        print(f"\n  {idea_label}:")
        print(f"    Mean holdout E2E: {np.mean(e2es):.1f}%")
        print(f"    Median holdout E2E: {np.median(e2es):.1f}%")
        print(f"    Max holdout E2E: {np.max(e2es):.1f}%")
        print(f"    Folds passing <80%: {passes}/{len(folds)}")
        print(f"    Folds passing <50%: {sum(1 for e in e2es if e < 50)}/{len(folds)}")
        print(f"    Folds passing <25%: {sum(1 for e in e2es if e < 25)}/{len(folds)}")

    # Compare to LOMO (train on 3)
    print("\n  Comparison: LOMO (train 3) vs L2O (train 2)")
    print(f"  {'Metric':<30s} {'LOMO Idea1':>12s} {'L2O Idea1':>12s} {'LOMO Idea2':>12s} {'L2O Idea2':>12s}")
    lomo_i1 = [25.5, 13.7, 13.9, 13.2]  # From round5_results.json
    lomo_i2 = [24.0, 9.6, 13.1, 10.6]
    l2o_i1 = [f["mean_e2e_pct"] for f in results["idea1_l2o"].values()]
    l2o_i2 = [f["mean_e2e_pct"] for f in results["idea2_l2o"].values()]
    print(f"  {'Mean holdout E2E':<30s} {np.mean(lomo_i1):>11.1f}% {np.mean(l2o_i1):>11.1f}% "
          f"{np.mean(lomo_i2):>11.1f}% {np.mean(l2o_i2):>11.1f}%")
    print(f"  {'Max holdout E2E':<30s} {np.max(lomo_i1):>11.1f}% {np.max(l2o_i1):>11.1f}% "
          f"{np.max(lomo_i2):>11.1f}% {np.max(l2o_i2):>11.1f}%")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")

    output_path = os.path.join(_SCRIPT_DIR, "round5_l2o_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
