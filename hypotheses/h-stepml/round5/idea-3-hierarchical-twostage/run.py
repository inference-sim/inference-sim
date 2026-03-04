#!/usr/bin/env python3
"""Idea 3: Hierarchical Two-Stage Model (Shared Physics + Model-Specific Intercept).

Stage 1: Shared delta model — learns batch-composition → step-time variation
          from pooled step data across ALL models (one Ridge, one set of weights).
Stage 2: Metadata-derived overhead — predicts the absolute base overhead from
          model architecture metadata (parameter count, TP degree, MoE flag).

Prediction: step_time = metadata_overhead(model) + delta(decode, prefill)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "shared")
sys.path.insert(0, _SHARED_DIR)

from data_loader import load_all_experiments, parse_experiment_metadata, DEFAULT_DATA_ROOT
from validate_blis import (
    DEFAULT_BINARY,
    parse_experiment_dir,
    load_ground_truth_metrics,
    load_exp_config,
    extract_kv_blocks_from_vllm_log,
    extract_cpu_kv_blocks_from_vllm_log,
    load_profile,
    build_workload_spec,
    run_blis,
    compute_error,
)

OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_METADATA = {
    "llama-2-7b": {"params_b": 7.0, "tp": 1, "is_moe": False, "active_params_b": 7.0},
    "codellama-34b": {"params_b": 34.0, "tp": 2, "is_moe": False, "active_params_b": 34.0},
    "llama-2-70b": {"params_b": 70.0, "tp": 4, "is_moe": False, "active_params_b": 70.0},
    "llama-2-70b-hf": {"params_b": 70.0, "tp": 4, "is_moe": False, "active_params_b": 70.0},
    "mixtral-8x7b-v0-1": {"params_b": 46.7, "tp": 2, "is_moe": True, "active_params_b": 12.9},
}

MODEL_ALIASES = {"llama-2-70b-hf": "llama-2-70b"}
CANONICAL_MODELS = ["llama-2-7b", "codellama-34b", "llama-2-70b", "mixtral-8x7b-v0-1"]


def normalize_model(name):
    return MODEL_ALIASES.get(name, name)


# ─── Stage 2: Metadata-derived overhead ───

def compute_target_step_times(data_root):
    """Per-model mean step time from E2E decomposition."""
    model_data = {}
    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        summary_path = os.path.join(dirpath, "results", "summary_lifecycle_metrics.json")
        if not os.path.isfile(summary_path):
            continue

        meta = parse_experiment_metadata(dirname)
        model = normalize_model(meta["model"])
        gt = load_ground_truth_metrics(dirpath)
        e2e_us = gt["e2e_mean_s"] * 1e6
        ttft_us = gt["ttft_mean_s"] * 1e6
        output_len = gt["output_len_mean"]

        if output_len > 0:
            target = (e2e_us - ttft_us) / output_len
            model_data.setdefault(model, []).append(target)

    result = {}
    for model, values in model_data.items():
        result[model] = float(np.mean(values))
        print(f"  {model}: target_step = {result[model]:.0f}μs")
    return result


def power_law(x, a, b):
    return a * np.power(x, b)


def fit_overhead_from_metadata(train_models, target_steps):
    """Fit metadata_overhead = a * (params_b)^b from training models' target step times."""
    params = np.array([MODEL_METADATA[m]["params_b"] for m in train_models])
    targets = np.array([target_steps[m] for m in train_models])

    try:
        popt, _ = curve_fit(power_law, params, targets, p0=[5000, 0.3], maxfev=10000)
        a, b = popt
        print(f"  Overhead fit: {a:.0f} * (params)^{b:.3f}")
    except RuntimeError:
        log_p = np.log(params)
        log_t = np.log(targets)
        fit = np.polyfit(log_p, log_t, 1)
        a, b = np.exp(fit[1]), fit[0]
        print(f"  Overhead fit (fallback): {a:.0f} * (params)^{b:.3f}")

    return lambda p: a * p**b


def fit_alpha0_from_metadata(train_models, data_root):
    """Fit alpha0 = a * (params)^b from training models' TTFT data."""
    params = []
    alpha0s = []
    for model in train_models:
        meta = MODEL_METADATA[model]
        params.append(meta["params_b"])
        # Get mean TTFT for this model
        ttft_values = []
        for dirname in sorted(os.listdir(data_root)):
            dirpath = os.path.join(data_root, dirname)
            if not os.path.isdir(dirpath):
                continue
            summary_path = os.path.join(dirpath, "results", "summary_lifecycle_metrics.json")
            if not os.path.isfile(summary_path):
                continue
            exp_meta = parse_experiment_metadata(dirname)
            if normalize_model(exp_meta["model"]) != model:
                continue
            gt = load_ground_truth_metrics(dirpath)
            ttft_values.append(gt["ttft_mean_s"] * 1e6)
        if ttft_values:
            alpha0s.append(float(np.mean(ttft_values)))

    if len(params) != len(alpha0s):
        return lambda p: 27129 * (p / 7) ** 0.45

    params = np.array(params)
    alpha0s = np.array(alpha0s)

    try:
        popt, _ = curve_fit(power_law, params, alpha0s, p0=[10000, 0.4], maxfev=10000)
        a, b = popt
    except RuntimeError:
        log_p = np.log(params)
        log_a = np.log(alpha0s)
        fit = np.polyfit(log_p, log_a, 1)
        a, b = np.exp(fit[1]), fit[0]

    return lambda p: a * p**b


# ─── Stage 1: Shared delta model ───

def train_shared_delta(step_df, target_steps, train_models):
    """Train shared Ridge on step-time residuals (deviation from model mean).

    For each step:
      - Compute model's mean step time (from E2E decomposition)
      - Residual = (step.duration_us scaled to cycle time) - model_mean
      - The scaling uses: cycle_time_est = step.duration_us * (target_step / mean_gpu_duration)

    Train Ridge on (decode_tokens, prefill_tokens) → residual across ALL training models.
    """
    train_df = step_df[step_df["model"].apply(normalize_model).isin(train_models)].copy()
    train_df["model_norm"] = train_df["model"].apply(normalize_model)

    # Compute scaling factor per model: target_step / mean(step.duration_us)
    model_mean_dur = train_df.groupby("model_norm")["step.duration_us"].mean()

    scale_factors = {}
    for m in train_models:
        if m in target_steps and m in model_mean_dur.index:
            scale_factors[m] = target_steps[m] / model_mean_dur[m]

    train_df["scale_factor"] = train_df["model_norm"].map(scale_factors)
    train_df = train_df.dropna(subset=["scale_factor"])

    # Scaled step time (cycle time estimate)
    train_df["cycle_time_est"] = train_df["step.duration_us"].astype(float) * train_df["scale_factor"]

    # Per-model mean cycle time (= target_step)
    train_df["model_mean_cycle"] = train_df["model_norm"].map(target_steps)

    # Residual (delta)
    train_df["delta"] = train_df["cycle_time_est"] - train_df["model_mean_cycle"]

    # Features
    decode_col = "batch.decode_tokens" if "batch.decode_tokens" in train_df.columns else "batch.scheduled_tokens"
    prefill_col = "batch.prefill_tokens"

    X = train_df[[decode_col, prefill_col]].fillna(0).astype(float).values
    y = train_df["delta"].values

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    ridge = Ridge(alpha=10.0)
    ridge.fit(X, y)

    print(f"  Delta Ridge: intercept={ridge.intercept_:.2f}, "
          f"coeff_decode={ridge.coef_[0]:.4f}, coeff_prefill={ridge.coef_[1]:.4f}")
    print(f"  Mean |delta|={np.mean(np.abs(y)):.0f}μs, std={np.std(y):.0f}μs")

    return ridge


def coefficients_from_twostage(model_name, overhead_fn, alpha0_fn, delta_ridge):
    """Convert two-stage model to BlackboxLatencyModel coefficients.

    step_time = overhead(params) + ridge.intercept + ridge.coef[0]*decode + ridge.coef[1]*prefill
    = (overhead + ridge.intercept) + ridge.coef[1]*prefill + ridge.coef[0]*decode
    = beta0 + beta1*prefill + beta2*decode
    """
    meta = MODEL_METADATA[model_name]
    overhead = overhead_fn(meta["params_b"])
    alpha0 = alpha0_fn(meta["params_b"])

    beta0 = overhead + delta_ridge.intercept_
    beta1 = delta_ridge.coef_[1]  # prefill coefficient
    beta2 = delta_ridge.coef_[0]  # decode coefficient

    return {"alpha0": alpha0, "beta0": beta0, "beta1": beta1, "beta2": beta2}


# ─── BLIS Validation ───

def run_validation(coefficients, label, data_root=DEFAULT_DATA_ROOT):
    import tempfile
    import yaml

    binary = DEFAULT_BINARY
    if not os.path.isfile(binary):
        import subprocess
        repo_root = os.path.abspath(os.path.join(_SHARED_DIR, "..", "..", ".."))
        subprocess.run(["go", "build", "-o", binary, "main.go"], cwd=repo_root, check=True)

    results = []
    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        summary_path = os.path.join(dirpath, "results", "summary_lifecycle_metrics.json")
        if not os.path.isfile(summary_path):
            continue

        try:
            meta = parse_experiment_dir(dirname)
        except ValueError:
            continue

        model_name = meta["model"]
        norm_model = normalize_model(model_name)
        if norm_model not in coefficients:
            continue

        coeffs = coefficients[norm_model]
        alpha = [coeffs["alpha0"], 0, 0]
        beta = [coeffs["beta0"], coeffs["beta1"], coeffs["beta2"]]

        gt = load_ground_truth_metrics(dirpath)
        exp_config = load_exp_config(dirpath)
        total_kv_blocks = extract_kv_blocks_from_vllm_log(dirpath)
        cpu_kv_blocks = extract_cpu_kv_blocks_from_vllm_log(dirpath)

        if total_kv_blocks is None:
            continue

        try:
            profile = load_profile(dirpath)
        except Exception:
            continue

        workload_spec = build_workload_spec(profile, gt)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(workload_spec, f, default_flow_style=False)
            spec_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            results_path = f.name

        try:
            blis = run_blis(
                binary=binary, workload_spec_path=spec_path, exp_config=exp_config,
                total_kv_blocks=total_kv_blocks, alpha_coeffs=alpha, beta_coeffs=beta,
                results_path=results_path, cpu_kv_blocks=cpu_kv_blocks,
            )
        finally:
            os.unlink(spec_path)
            if os.path.exists(results_path):
                os.unlink(results_path)

        if blis is None:
            results.append({"experiment": dirname, "model": model_name, "workload": meta["workload"],
                          "label": label, "status": "blis_failed"})
            continue

        e2e_err = compute_error(blis["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
        ttft_err = compute_error(blis["ttft_mean_ms"], gt["ttft_mean_s"] * 1000)
        itl_err = compute_error(blis["itl_mean_ms"], gt["itl_mean_s"] * 1000)

        print(f"  [{label}] {dirname}: E2E={e2e_err*100:.1f}%")

        results.append({
            "experiment": dirname, "model": model_name, "workload": meta["workload"],
            "label": label, "status": "ok",
            "gt_e2e_ms": gt["e2e_mean_s"] * 1000, "blis_e2e_ms": blis["e2e_mean_ms"],
            "e2e_error": e2e_err, "ttft_error": ttft_err, "itl_error": itl_err,
            "alpha0": coeffs["alpha0"], "beta0": coeffs["beta0"],
            "beta1": coeffs["beta1"], "beta2": coeffs["beta2"],
        })

    return pd.DataFrame(results)


# ─── Main ───

def main():
    print("Loading step data...")
    step_df = load_all_experiments()
    print(f"  Loaded {len(step_df)} steps")

    print("\nComputing target step times from E2E...")
    target_steps = compute_target_step_times(DEFAULT_DATA_ROOT)

    # ── H1: Full training ──
    print("\n=== H1: Two-Stage Model (all models) ===\n")

    print("Stage 2: Fitting overhead from metadata...")
    overhead_fn = fit_overhead_from_metadata(CANONICAL_MODELS, target_steps)
    alpha0_fn = fit_alpha0_from_metadata(CANONICAL_MODELS, DEFAULT_DATA_ROOT)

    for m in CANONICAL_MODELS:
        pred_oh = overhead_fn(MODEL_METADATA[m]["params_b"])
        actual = target_steps.get(m, 0)
        print(f"  {m}: predicted_overhead={pred_oh:.0f}, target_step={actual:.0f}, "
              f"err={abs(pred_oh-actual)/max(actual,1)*100:.1f}%")

    print("\nStage 1: Training shared delta model...")
    delta_ridge = train_shared_delta(step_df, target_steps, CANONICAL_MODELS)

    h1_coefficients = {}
    print("\nPredicted coefficients:")
    for m in CANONICAL_MODELS:
        coeffs = coefficients_from_twostage(m, overhead_fn, alpha0_fn, delta_ridge)
        h1_coefficients[m] = coeffs
        print(f"  {m}: alpha0={coeffs['alpha0']:.0f}, beta0={coeffs['beta0']:.0f}, "
              f"beta1={coeffs['beta1']:.2f}, beta2={coeffs['beta2']:.2f}")

    print("\nRunning BLIS validation...")
    h1_results = run_validation(h1_coefficients, "h1-twostage")

    # ── H2: LOMO ──
    print("\n=== H2: LOMO ===\n")
    lomo_results = []
    for holdout in CANONICAL_MODELS:
        train_models = [m for m in CANONICAL_MODELS if m != holdout]
        print(f"\n--- Holdout: {holdout} ---")

        train_targets = {m: target_steps[m] for m in train_models if m in target_steps}

        oh_fn = fit_overhead_from_metadata(train_models, train_targets)
        a0_fn = fit_alpha0_from_metadata(train_models, DEFAULT_DATA_ROOT)

        try:
            fold_delta = train_shared_delta(step_df, train_targets, train_models)
        except Exception as e:
            print(f"  Training failed: {e}")
            continue

        fold_coefficients = {}
        for m in CANONICAL_MODELS:
            fold_coefficients[m] = coefficients_from_twostage(m, oh_fn, a0_fn, fold_delta)

        pred = fold_coefficients[holdout]
        actual_target = target_steps.get(holdout, 0)
        print(f"  Predicted {holdout}: beta0={pred['beta0']:.0f} (target_step={actual_target:.0f})")

        fold_df = run_validation(fold_coefficients, f"lomo-{holdout}")
        fold_df["holdout_model"] = holdout
        lomo_results.append(fold_df)

    lomo_df = pd.concat(lomo_results, ignore_index=True) if lomo_results else pd.DataFrame()

    # ── Summary ──
    print("\n=== Results Summary ===\n")

    h1_ok = h1_results[h1_results["status"] == "ok"]
    if len(h1_ok) > 0:
        mean_e2e = h1_ok["e2e_error"].mean() * 100
        below_10 = (h1_ok["e2e_error"] < 0.10).sum()
        print(f"H1 (two-stage): Mean E2E = {mean_e2e:.1f}%, {below_10}/{len(h1_ok)} <10%")

    if len(lomo_df) > 0:
        lomo_ok = lomo_df[lomo_df["status"] == "ok"]
        for holdout in CANONICAL_MODELS:
            fold = lomo_ok[lomo_ok["holdout_model"] == holdout]
            holdout_exps = fold[fold["model"].apply(normalize_model) == holdout]
            if len(holdout_exps) > 0:
                fold_e2e = holdout_exps["e2e_error"].mean() * 100
                print(f"LOMO holdout={holdout}: E2E={fold_e2e:.1f}% ({'PASS' if fold_e2e < 80 else 'FAIL'})")

    if len(h1_ok) > 0:
        for model in h1_ok["model"].apply(normalize_model).unique():
            model_exps = h1_ok[h1_ok["model"].apply(normalize_model) == model]
            if len(model_exps) > 1:
                e2e_range = (model_exps["e2e_error"].max() - model_exps["e2e_error"].min()) * 100
                print(f"LOWO {model}: workload range = {e2e_range:.1f}pp")

    all_results = pd.concat([h1_results, lomo_df], ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, "idea3_results.csv")
    all_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
