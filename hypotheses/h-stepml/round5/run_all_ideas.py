#!/usr/bin/env python3
"""Round 5: Unified Cross-Model Latency Prediction — All 3 Ideas.

Tests whether a SINGLE model/formula can predict per-model BlackboxLatencyModel
coefficients from model architecture metadata, matching R4's per-model 5.7% E2E.

Ideas:
  1. Analytical Overhead: Power law meta-regression on R4 coefficients
  2. Normalized Features: Unified Ridge on normalized step data
  3. Hierarchical Two-Stage: Shared delta + metadata overhead
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_SCRIPT_DIR, "..", "shared")
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _SHARED_DIR)

from data_loader import load_all_experiments, parse_experiment_metadata
from validate_r5 import (
    DATA_ROOT, list_experiments, validate_all, normalize_model,
    load_ground_truth,
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


# ─── Compute E2E-derived step times from lifecycle data ───

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
    stats = {}
    for model in step_df["model"].unique():
        mdf = step_df[step_df["model"] == model]
        decode = mdf.get("batch.decode_tokens", pd.Series(0, index=mdf.index)).fillna(0).astype(float)
        dur = mdf["step.duration_us"].dropna().astype(float)
        stats[normalize_model(model)] = {
            "avg_decode_batch": float(decode.mean()),
            "avg_dur": float(dur.mean()),
            "n_steps": len(mdf),
        }
    return stats


# ═══════════════════════════════════════════════════════
# IDEA 1: Analytical Overhead Model
# ═══════════════════════════════════════════════════════

def idea1_fit(train_models):
    """Fit power laws for beta0, alpha0, beta2 from training models' R4 coefficients."""
    params = np.array([MODEL_METADATA[m]["params_b"] for m in train_models])
    beta0s = np.array([R4_COEFFICIENTS[m]["beta"][0] for m in train_models])
    alpha0s = np.array([R4_COEFFICIENTS[m]["alpha"][0] for m in train_models])
    ppg = np.array([MODEL_METADATA[m]["active_b"] / MODEL_METADATA[m]["tp"] for m in train_models])
    beta2s = np.array([R4_COEFFICIENTS[m]["beta"][2] for m in train_models])

    try:
        b0_popt, _ = curve_fit(power_law, params, beta0s, p0=[5000, 0.3], maxfev=10000)
    except RuntimeError:
        lp, lb = np.log(params), np.log(beta0s)
        fit = np.polyfit(lp, lb, 1)
        b0_popt = [np.exp(fit[1]), fit[0]]

    try:
        a0_popt, _ = curve_fit(power_law, params, alpha0s, p0=[10000, 0.4], maxfev=10000)
    except RuntimeError:
        lp, la = np.log(params), np.log(alpha0s)
        fit = np.polyfit(lp, la, 1)
        a0_popt = [np.exp(fit[1]), fit[0]]

    try:
        b2_popt, _ = curve_fit(power_law, ppg, beta2s, p0=[5, 0.5], maxfev=10000)
    except RuntimeError:
        lp, lb = np.log(ppg), np.log(np.maximum(beta2s, 0.1))
        fit = np.polyfit(lp, lb, 1)
        b2_popt = [np.exp(fit[1]), fit[0]]

    return b0_popt, a0_popt, b2_popt


def idea1_predict(model, b0_popt, a0_popt, b2_popt):
    meta = MODEL_METADATA[model]
    ppg = meta["active_b"] / meta["tp"]
    return {
        "alpha": [float(power_law(meta["params_b"], *a0_popt)), 0, 0],
        "beta": [float(power_law(meta["params_b"], *b0_popt)), 0.0, float(power_law(ppg, *b2_popt))],
    }


# ═══════════════════════════════════════════════════════
# IDEA 2: Normalized Features
# ═══════════════════════════════════════════════════════

def idea2_fit(train_models, target_steps, ttfts, step_stats):
    """Train unified model using E2E-derived target step times.

    Instead of training a Ridge on step-level data (which captures only GPU time),
    directly fit a metadata formula that reproduces the E2E-derived targets.
    """
    # Fit overhead = a * params^b from target_steps
    params = np.array([MODEL_METADATA[m]["params_b"] for m in train_models])
    targets = np.array([target_steps[m] for m in train_models])

    try:
        oh_popt, _ = curve_fit(power_law, params, targets, p0=[5000, 0.3], maxfev=10000)
    except RuntimeError:
        lp, lt = np.log(params), np.log(targets)
        fit = np.polyfit(lp, lt, 1)
        oh_popt = [np.exp(fit[1]), fit[0]]

    # Compute beta2 from step data: beta2 = avg_dur / avg_decode_batch
    beta2_estimates = {}
    for m in train_models:
        if m in step_stats:
            avg_batch = step_stats[m]["avg_decode_batch"]
            avg_dur = step_stats[m]["avg_dur"]
            if avg_batch > 0:
                beta2_estimates[m] = avg_dur / avg_batch

    # Fit beta2 = c * (active_params/tp)^d
    if beta2_estimates:
        ppg = np.array([MODEL_METADATA[m]["active_b"] / MODEL_METADATA[m]["tp"]
                       for m in beta2_estimates])
        b2s = np.array(list(beta2_estimates.values()))
        try:
            b2_popt, _ = curve_fit(power_law, ppg, b2s, p0=[5, 0.5], maxfev=10000)
        except RuntimeError:
            lp, lb = np.log(ppg), np.log(np.maximum(b2s, 0.1))
            fit = np.polyfit(lp, lb, 1)
            b2_popt = [np.exp(fit[1]), fit[0]]
    else:
        b2_popt = [5.0, 0.5]

    # Fit alpha0 = e * params^f
    a0_params = np.array([MODEL_METADATA[m]["params_b"] for m in train_models if m in ttfts])
    a0_vals = np.array([ttfts[m] for m in train_models if m in ttfts])
    try:
        a0_popt, _ = curve_fit(power_law, a0_params, a0_vals, p0=[10000, 0.4], maxfev=10000)
    except RuntimeError:
        a0_popt = [10000, 0.4]

    return oh_popt, b2_popt, a0_popt


def idea2_predict(model, oh_popt, b2_popt, a0_popt, step_stats):
    meta = MODEL_METADATA[model]
    ppg = meta["active_b"] / meta["tp"]
    overhead = float(power_law(meta["params_b"], *oh_popt))
    beta2 = float(power_law(ppg, *b2_popt))

    # beta0 = overhead - beta2 * avg_decode_batch (so that at avg batch, step = overhead)
    avg_batch = step_stats.get(model, {}).get("avg_decode_batch", 30)
    beta0 = overhead - beta2 * avg_batch
    beta0 = max(beta0, 1000)

    alpha0 = float(power_law(meta["params_b"], *a0_popt))

    return {"alpha": [alpha0, 0, 0], "beta": [beta0, 0.0, beta2]}


# ═══════════════════════════════════════════════════════
# IDEA 3: Hierarchical Two-Stage
# ═══════════════════════════════════════════════════════

def idea3_fit(train_models, target_steps, ttfts, step_df):
    """Stage 1: shared delta Ridge on step residuals. Stage 2: metadata overhead."""
    # Stage 2: overhead from metadata
    params = np.array([MODEL_METADATA[m]["params_b"] for m in train_models])
    targets = np.array([target_steps[m] for m in train_models])

    try:
        oh_popt, _ = curve_fit(power_law, params, targets, p0=[5000, 0.3], maxfev=10000)
    except RuntimeError:
        lp, lt = np.log(params), np.log(targets)
        fit = np.polyfit(lp, lt, 1)
        oh_popt = [np.exp(fit[1]), fit[0]]

    # Stage 1: shared delta model
    train_df = step_df[step_df["model"].apply(normalize_model).isin(train_models)].copy()
    train_df["model_norm"] = train_df["model"].apply(normalize_model)

    # Scale step data to cycle time: scale = target_step / mean_gpu_duration
    model_mean_dur = train_df.groupby("model_norm")["step.duration_us"].mean()
    scale_factors = {}
    for m in train_models:
        if m in target_steps and m in model_mean_dur.index:
            scale_factors[m] = target_steps[m] / model_mean_dur[m]

    train_df["scale"] = train_df["model_norm"].map(scale_factors)
    train_df = train_df.dropna(subset=["scale"])
    train_df["cycle_est"] = train_df["step.duration_us"].astype(float) * train_df["scale"]
    train_df["model_mean"] = train_df["model_norm"].map(target_steps)
    train_df["delta"] = train_df["cycle_est"] - train_df["model_mean"]

    decode_col = "batch.decode_tokens" if "batch.decode_tokens" in train_df.columns else "batch.scheduled_tokens"
    prefill_col = "batch.prefill_tokens"

    X = train_df[[decode_col, prefill_col]].fillna(0).astype(float).values
    y = train_df["delta"].values
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    ridge = Ridge(alpha=10.0)
    ridge.fit(X, y)

    # Fit alpha0
    a0_params = np.array([MODEL_METADATA[m]["params_b"] for m in train_models if m in ttfts])
    a0_vals = np.array([ttfts[m] for m in train_models if m in ttfts])
    try:
        a0_popt, _ = curve_fit(power_law, a0_params, a0_vals, p0=[10000, 0.4], maxfev=10000)
    except RuntimeError:
        a0_popt = [10000, 0.4]

    return oh_popt, ridge, a0_popt


def idea3_predict(model, oh_popt, ridge, a0_popt):
    meta = MODEL_METADATA[model]
    overhead = float(power_law(meta["params_b"], *oh_popt))
    beta0 = overhead + ridge.intercept_
    beta0 = max(beta0, 1000)
    beta2 = ridge.coef_[0]  # decode coefficient
    beta1 = ridge.coef_[1]  # prefill coefficient
    alpha0 = float(power_law(meta["params_b"], *a0_popt))
    return {"alpha": [alpha0, 0, 0], "beta": [beta0, beta1, beta2]}


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    t_start = time.time()
    experiments = list_experiments()
    print(f"Found {len(experiments)} experiments\n")

    print("Loading step data...")
    step_df = load_all_experiments()
    print(f"  {len(step_df)} steps from {step_df['model'].nunique()} models\n")

    print("Computing target step times and TTFTs from lifecycle data...")
    target_steps, ttfts = compute_target_steps_and_ttfts()
    step_stats = compute_step_data_stats(step_df)
    for m in CANONICAL_MODELS:
        ts = target_steps.get(m, 0)
        print(f"  {m}: target_step={ts:.0f}μs, TTFT={ttfts.get(m, 0)/1000:.1f}ms")

    all_results = {}

    # ── R4 Control ──
    print("\n" + "=" * 70)
    print("CONTROL: R4 exact coefficients")
    print("=" * 70)
    ctrl = validate_all(experiments, R4_COEFFICIENTS, "R4-control")
    ctrl_ok = [r for r in ctrl if r["status"] == "ok"]
    mean_e2e = np.mean([r["e2e_error"] for r in ctrl_ok]) * 100 if ctrl_ok else float("inf")
    below_10 = sum(1 for r in ctrl_ok if r["e2e_error"] < 0.10)
    print(f"\n  R4 Control: Mean E2E = {mean_e2e:.1f}%, {below_10}/{len(ctrl_ok)} <10%")
    all_results["control"] = ctrl

    # ── IDEA 1: Analytical Overhead ──
    print("\n" + "=" * 70)
    print("IDEA 1: Analytical Overhead Model (meta-regression on R4 coefficients)")
    print("=" * 70)

    # H1: Full training
    b0p, a0p, b2p = idea1_fit(CANONICAL_MODELS)
    print(f"  Fits: beta0={b0p[0]:.0f}*x^{b0p[1]:.3f}, alpha0={a0p[0]:.0f}*x^{a0p[1]:.3f}, "
          f"beta2={b2p[0]:.2f}*x^{b2p[1]:.3f}")

    i1_coeffs = {m: idea1_predict(m, b0p, a0p, b2p) for m in CANONICAL_MODELS}
    for m in CANONICAL_MODELS:
        c = i1_coeffs[m]
        actual = R4_COEFFICIENTS[m]
        print(f"  {m}: beta0={c['beta'][0]:.0f} (actual {actual['beta'][0]:.0f}), "
              f"beta2={c['beta'][2]:.1f} (actual {actual['beta'][2]:.1f})")

    i1_full = validate_all(experiments, i1_coeffs, "I1-full")
    i1_ok = [r for r in i1_full if r["status"] == "ok"]
    mean_e2e = np.mean([r["e2e_error"] for r in i1_ok]) * 100 if i1_ok else float("inf")
    below_10 = sum(1 for r in i1_ok if r["e2e_error"] < 0.10)
    print(f"\n  Idea 1 H1: Mean E2E = {mean_e2e:.1f}%, {below_10}/{len(i1_ok)} <10%")
    all_results["idea1_full"] = i1_full

    # H2: LOMO
    print("\n  --- Idea 1 LOMO ---")
    i1_lomo = {}
    for holdout in CANONICAL_MODELS:
        train = [m for m in CANONICAL_MODELS if m != holdout]
        b0p, a0p, b2p = idea1_fit(train)
        fold_coeffs = {m: idea1_predict(m, b0p, a0p, b2p) for m in CANONICAL_MODELS}
        pred = fold_coeffs[holdout]
        actual = R4_COEFFICIENTS[holdout]
        print(f"  Holdout {holdout}: beta0={pred['beta'][0]:.0f} (actual {actual['beta'][0]:.0f})")

        fold_results = validate_all(experiments, fold_coeffs, f"I1-lomo-{holdout}")
        holdout_exps = [r for r in fold_results if r["status"] == "ok"
                       and normalize_model(r["model"]) == holdout]
        if holdout_exps:
            fold_e2e = np.mean([r["e2e_error"] for r in holdout_exps]) * 100
            print(f"    → LOMO E2E = {fold_e2e:.1f}% ({'PASS' if fold_e2e < 80 else 'FAIL'})")
            i1_lomo[holdout] = fold_e2e

    all_results["idea1_lomo"] = i1_lomo

    # ── IDEA 2: Normalized Features ──
    print("\n" + "=" * 70)
    print("IDEA 2: Normalized Features (metadata overhead + step-derived beta2)")
    print("=" * 70)

    oh2, b2_2, a0_2 = idea2_fit(CANONICAL_MODELS, target_steps, ttfts, step_stats)
    i2_coeffs = {m: idea2_predict(m, oh2, b2_2, a0_2, step_stats) for m in CANONICAL_MODELS}
    for m in CANONICAL_MODELS:
        c = i2_coeffs[m]
        print(f"  {m}: beta0={c['beta'][0]:.0f}, beta2={c['beta'][2]:.1f}")

    i2_full = validate_all(experiments, i2_coeffs, "I2-full")
    i2_ok = [r for r in i2_full if r["status"] == "ok"]
    mean_e2e = np.mean([r["e2e_error"] for r in i2_ok]) * 100 if i2_ok else float("inf")
    below_10 = sum(1 for r in i2_ok if r["e2e_error"] < 0.10)
    print(f"\n  Idea 2 H1: Mean E2E = {mean_e2e:.1f}%, {below_10}/{len(i2_ok)} <10%")
    all_results["idea2_full"] = i2_full

    # H2: LOMO
    print("\n  --- Idea 2 LOMO ---")
    i2_lomo = {}
    for holdout in CANONICAL_MODELS:
        train = [m for m in CANONICAL_MODELS if m != holdout]
        train_targets = {m: target_steps[m] for m in train if m in target_steps}
        train_ttfts = {m: ttfts[m] for m in train if m in ttfts}
        oh2_f, b2_f, a0_f = idea2_fit(train, train_targets, train_ttfts, step_stats)
        fold_coeffs = {m: idea2_predict(m, oh2_f, b2_f, a0_f, step_stats) for m in CANONICAL_MODELS}

        fold_results = validate_all(experiments, fold_coeffs, f"I2-lomo-{holdout}")
        holdout_exps = [r for r in fold_results if r["status"] == "ok"
                       and normalize_model(r["model"]) == holdout]
        if holdout_exps:
            fold_e2e = np.mean([r["e2e_error"] for r in holdout_exps]) * 100
            print(f"    Holdout {holdout}: LOMO E2E = {fold_e2e:.1f}% ({'PASS' if fold_e2e < 80 else 'FAIL'})")
            i2_lomo[holdout] = fold_e2e

    all_results["idea2_lomo"] = i2_lomo

    # ── IDEA 3: Hierarchical Two-Stage ──
    print("\n" + "=" * 70)
    print("IDEA 3: Hierarchical Two-Stage (shared delta + metadata overhead)")
    print("=" * 70)

    oh3, ridge3, a0_3 = idea3_fit(CANONICAL_MODELS, target_steps, ttfts, step_df)
    print(f"  Delta Ridge: intercept={ridge3.intercept_:.2f}, "
          f"decode={ridge3.coef_[0]:.4f}, prefill={ridge3.coef_[1]:.4f}")

    i3_coeffs = {m: idea3_predict(m, oh3, ridge3, a0_3) for m in CANONICAL_MODELS}
    for m in CANONICAL_MODELS:
        c = i3_coeffs[m]
        print(f"  {m}: beta0={c['beta'][0]:.0f}, beta1={c['beta'][1]:.2f}, beta2={c['beta'][2]:.2f}")

    i3_full = validate_all(experiments, i3_coeffs, "I3-full")
    i3_ok = [r for r in i3_full if r["status"] == "ok"]
    mean_e2e = np.mean([r["e2e_error"] for r in i3_ok]) * 100 if i3_ok else float("inf")
    below_10 = sum(1 for r in i3_ok if r["e2e_error"] < 0.10)
    print(f"\n  Idea 3 H1: Mean E2E = {mean_e2e:.1f}%, {below_10}/{len(i3_ok)} <10%")
    all_results["idea3_full"] = i3_full

    # H2: LOMO
    print("\n  --- Idea 3 LOMO ---")
    i3_lomo = {}
    for holdout in CANONICAL_MODELS:
        train = [m for m in CANONICAL_MODELS if m != holdout]
        train_targets = {m: target_steps[m] for m in train if m in target_steps}
        train_ttfts = {m: ttfts[m] for m in train if m in ttfts}
        oh3_f, ridge3_f, a0_f = idea3_fit(train, train_targets, train_ttfts, step_df)
        fold_coeffs = {m: idea3_predict(m, oh3_f, ridge3_f, a0_f) for m in CANONICAL_MODELS}

        fold_results = validate_all(experiments, fold_coeffs, f"I3-lomo-{holdout}")
        holdout_exps = [r for r in fold_results if r["status"] == "ok"
                       and normalize_model(r["model"]) == holdout]
        if holdout_exps:
            fold_e2e = np.mean([r["e2e_error"] for r in holdout_exps]) * 100
            print(f"    Holdout {holdout}: LOMO E2E = {fold_e2e:.1f}% ({'PASS' if fold_e2e < 80 else 'FAIL'})")
            i3_lomo[holdout] = fold_e2e

    all_results["idea3_lomo"] = i3_lomo

    # ── GRAND SUMMARY ──
    print("\n" + "=" * 70)
    print("GRAND SUMMARY")
    print("=" * 70)

    for key, label in [("control", "R4 Control"), ("idea1_full", "Idea 1"),
                       ("idea2_full", "Idea 2"), ("idea3_full", "Idea 3")]:
        results = all_results.get(key, [])
        ok = [r for r in results if r["status"] == "ok"]
        if ok:
            mean_e2e = np.mean([r["e2e_error"] for r in ok]) * 100
            below_10 = sum(1 for r in ok if r["e2e_error"] < 0.10)
            print(f"  {label:20s}: Mean E2E = {mean_e2e:6.1f}%, {below_10}/{len(ok)} <10%")

    print("\n  LOMO (per fold, holdout model E2E %):")
    print(f"  {'Holdout':<20s} {'Idea 1':>10s} {'Idea 2':>10s} {'Idea 3':>10s}")
    for m in CANONICAL_MODELS:
        vals = []
        for key in ["idea1_lomo", "idea2_lomo", "idea3_lomo"]:
            v = all_results.get(key, {}).get(m)
            vals.append(f"{v:.1f}%" if v is not None else "N/A")
        print(f"  {m:<20s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save all results
    output_path = os.path.join(_SCRIPT_DIR, "round5_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
