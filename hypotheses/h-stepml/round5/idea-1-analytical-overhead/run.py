#!/usr/bin/env python3
"""Idea 1: Analytical Overhead Model with Metadata-Derived Coefficients.

Derives all BlackboxLatencyModel coefficients from model architecture metadata
using R4's 4 calibrated coefficient sets as training data for a meta-regression.

No step-level data or lifecycle data needed — pure arithmetic on 4 data points.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "shared")
sys.path.insert(0, _SHARED_DIR)

from validate_blis import (
    DEFAULT_DATA_ROOT,
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

# ─── R4 Production Coefficients (training data for meta-regression) ───

MODEL_METADATA = {
    "llama-2-7b": {"params_b": 7.0, "tp": 1, "is_moe": False, "active_params_b": 7.0},
    "codellama-34b": {"params_b": 34.0, "tp": 2, "is_moe": False, "active_params_b": 34.0},
    "llama-2-70b": {"params_b": 70.0, "tp": 4, "is_moe": False, "active_params_b": 70.0},
    "mixtral-8x7b-v0-1": {"params_b": 46.7, "tp": 2, "is_moe": True, "active_params_b": 12.9},
}

R4_COEFFICIENTS = {
    "llama-2-7b": {"alpha0": 27129, "beta0": 9741, "beta1": 0.30, "beta2": 13.6},
    "codellama-34b": {"alpha0": 47618, "beta0": 14196, "beta1": 0.00, "beta2": 25.8},
    "llama-2-70b": {"alpha0": 78888, "beta0": 17992, "beta1": 1.22, "beta2": 35.2},
    "mixtral-8x7b-v0-1": {"alpha0": 62767, "beta0": 18921, "beta1": 0.69, "beta2": 8.8},
}

# Model name normalization (experiment dirs → coefficient keys)
MODEL_ALIASES = {
    "llama-2-70b-hf": "llama-2-70b",
}


def normalize_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


# ─── Meta-regression: fit power laws ───

def power_law(x, a, b):
    """y = a * x^b"""
    return a * np.power(x, b)


def fit_beta0(models: list[str]) -> callable:
    """Fit beta0 = a * (params/1e9)^b from training models."""
    params = []
    beta0s = []
    for m in models:
        meta = MODEL_METADATA[m]
        params.append(meta["params_b"])
        beta0s.append(R4_COEFFICIENTS[m]["beta0"])

    params = np.array(params)
    beta0s = np.array(beta0s)

    try:
        popt, _ = curve_fit(power_law, params, beta0s, p0=[5000, 0.3], maxfev=10000)
        a, b = popt
        print(f"  beta0 fit: {a:.0f} * (params)^{b:.3f}")
    except RuntimeError:
        # Fallback: log-linear fit
        log_p = np.log(params)
        log_b = np.log(beta0s)
        b_fit = np.polyfit(log_p, log_b, 1)
        a = np.exp(b_fit[1])
        b = b_fit[0]
        print(f"  beta0 fit (fallback): {a:.0f} * (params)^{b:.3f}")

    return lambda p: a * p**b


def fit_alpha0(models: list[str]) -> callable:
    """Fit alpha0 = a * (params/1e9)^b from training models."""
    params = []
    alpha0s = []
    for m in models:
        meta = MODEL_METADATA[m]
        params.append(meta["params_b"])
        alpha0s.append(R4_COEFFICIENTS[m]["alpha0"])

    params = np.array(params)
    alpha0s = np.array(alpha0s)

    try:
        popt, _ = curve_fit(power_law, params, alpha0s, p0=[10000, 0.4], maxfev=10000)
        a, b = popt
        print(f"  alpha0 fit: {a:.0f} * (params)^{b:.3f}")
    except RuntimeError:
        log_p = np.log(params)
        log_a = np.log(alpha0s)
        b_fit = np.polyfit(log_p, log_a, 1)
        a = np.exp(b_fit[1])
        b = b_fit[0]
        print(f"  alpha0 fit (fallback): {a:.0f} * (params)^{b:.3f}")

    return lambda p: a * p**b


def fit_beta2(models: list[str]) -> callable:
    """Fit beta2 = a * (params_per_gpu)^b from training models.

    params_per_gpu = total_params / tp for dense, active_params / tp for MoE.
    """
    ppg = []
    beta2s = []
    for m in models:
        meta = MODEL_METADATA[m]
        params_per_gpu = meta["active_params_b"] / meta["tp"]
        ppg.append(params_per_gpu)
        beta2s.append(R4_COEFFICIENTS[m]["beta2"])

    ppg = np.array(ppg)
    beta2s = np.array(beta2s)

    try:
        popt, _ = curve_fit(power_law, ppg, beta2s, p0=[5, 0.5], maxfev=10000)
        a, b = popt
        print(f"  beta2 fit: {a:.2f} * (params_per_gpu)^{b:.3f}")
    except RuntimeError:
        log_p = np.log(ppg)
        log_b = np.log(np.maximum(beta2s, 0.1))
        b_fit = np.polyfit(log_p, log_b, 1)
        a = np.exp(b_fit[1])
        b = b_fit[0]
        print(f"  beta2 fit (fallback): {a:.2f} * (params_per_gpu)^{b:.3f}")

    return lambda p: a * p**b


def predict_coefficients(model_name: str, beta0_fn, alpha0_fn, beta2_fn) -> dict:
    """Predict coefficients for a model using meta-regression functions."""
    meta = MODEL_METADATA[model_name]
    params_per_gpu = meta["active_params_b"] / meta["tp"]

    return {
        "alpha0": float(alpha0_fn(meta["params_b"])),
        "beta0": float(beta0_fn(meta["params_b"])),
        "beta1": 0.0,  # negligible
        "beta2": float(beta2_fn(params_per_gpu)),
    }


# ─── BLIS Validation ───

def run_validation(
    coefficients: dict[str, dict],
    label: str,
    data_root: str = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    """Run BLIS validation with predicted coefficients for each experiment."""
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
        norm_model = normalize_model_name(model_name)

        if norm_model not in coefficients:
            print(f"  SKIP {dirname}: no coefficients for {norm_model}")
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
                binary=binary,
                workload_spec_path=spec_path,
                exp_config=exp_config,
                total_kv_blocks=total_kv_blocks,
                alpha_coeffs=alpha,
                beta_coeffs=beta,
                results_path=results_path,
                cpu_kv_blocks=cpu_kv_blocks,
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

        print(f"  [{label}] {dirname}: E2E={e2e_err*100:.1f}%, TTFT={ttft_err*100:.1f}%, ITL={itl_err*100:.1f}%")

        results.append({
            "experiment": dirname, "model": model_name, "workload": meta["workload"],
            "label": label, "status": "ok",
            "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
            "blis_e2e_ms": blis["e2e_mean_ms"],
            "e2e_error": e2e_err,
            "ttft_error": ttft_err,
            "itl_error": itl_err,
            "alpha0": coeffs["alpha0"], "beta0": coeffs["beta0"],
            "beta1": coeffs["beta1"], "beta2": coeffs["beta2"],
        })

    return pd.DataFrame(results)


# ─── Main Experiment ───

def main():
    all_models = list(MODEL_METADATA.keys())

    # ── H1: Full training (all 4 models) ──
    print("\n=== H1: Meta-regression on ALL 4 models ===\n")
    beta0_fn = fit_beta0(all_models)
    alpha0_fn = fit_alpha0(all_models)
    beta2_fn = fit_beta2(all_models)

    full_coefficients = {}
    print("\nPredicted vs R4 coefficients:")
    for m in all_models:
        pred = predict_coefficients(m, beta0_fn, alpha0_fn, beta2_fn)
        actual = R4_COEFFICIENTS[m]
        full_coefficients[m] = pred
        print(f"  {m}:")
        for k in ["alpha0", "beta0", "beta1", "beta2"]:
            err = abs(pred[k] - actual[k]) / max(abs(actual[k]), 1) * 100
            print(f"    {k}: pred={pred[k]:.1f}, actual={actual[k]:.1f}, err={err:.1f}%")

    print("\nRunning BLIS validation (all models, meta-regression)...")
    h1_results = run_validation(full_coefficients, "h1-full")

    # Also run with R4's exact coefficients as control
    print("\n=== Control: R4 exact coefficients ===\n")
    control_results = run_validation(R4_COEFFICIENTS, "control-r4")

    # ── H2: LOMO ──
    print("\n=== H2: LOMO (Leave-One-Model-Out) ===\n")
    lomo_results = []
    for holdout in all_models:
        train_models = [m for m in all_models if m != holdout]
        print(f"\n--- LOMO fold: holdout={holdout}, train={train_models} ---")

        b0_fn = fit_beta0(train_models)
        a0_fn = fit_alpha0(train_models)
        b2_fn = fit_beta2(train_models)

        # Predict coefficients for ALL models (including holdout)
        fold_coefficients = {}
        for m in all_models:
            fold_coefficients[m] = predict_coefficients(m, b0_fn, a0_fn, b2_fn)

        pred = fold_coefficients[holdout]
        actual = R4_COEFFICIENTS[holdout]
        print(f"  Predicted {holdout}: alpha0={pred['alpha0']:.0f} (actual {actual['alpha0']:.0f}), "
              f"beta0={pred['beta0']:.0f} (actual {actual['beta0']:.0f}), "
              f"beta2={pred['beta2']:.1f} (actual {actual['beta2']:.1f})")

        fold_df = run_validation(fold_coefficients, f"lomo-holdout-{holdout}")
        fold_df["holdout_model"] = holdout
        lomo_results.append(fold_df)

    lomo_df = pd.concat(lomo_results, ignore_index=True) if lomo_results else pd.DataFrame()

    # ── H3: LOWO ──
    print("\n=== H3: LOWO (Leave-One-Workload-Out) ===\n")
    print("  Coefficients are metadata-derived → workload-invariant by design.")
    print("  LOWO uses same coefficients as H1. Measuring per-workload variance.\n")
    # LOWO is inherent — same coefficients for all workloads. We just analyze H1 results by workload.

    # ── Output ──
    print("\n=== Results Summary ===\n")

    # H1 summary
    h1_ok = h1_results[h1_results["status"] == "ok"]
    if len(h1_ok) > 0:
        mean_e2e = h1_ok["e2e_error"].mean() * 100
        below_10 = (h1_ok["e2e_error"] < 0.10).sum()
        print(f"H1 (all models, meta-regression): Mean E2E = {mean_e2e:.1f}%, {below_10}/{len(h1_ok)} <10%")

    # Control summary
    ctrl_ok = control_results[control_results["status"] == "ok"]
    if len(ctrl_ok) > 0:
        mean_e2e = ctrl_ok["e2e_error"].mean() * 100
        below_10 = (ctrl_ok["e2e_error"] < 0.10).sum()
        print(f"Control (R4 exact): Mean E2E = {mean_e2e:.1f}%, {below_10}/{len(ctrl_ok)} <10%")

    # LOMO summary
    if len(lomo_df) > 0:
        lomo_ok = lomo_df[lomo_df["status"] == "ok"]
        # Per-fold: only count holdout model's experiments
        for holdout in all_models:
            fold = lomo_ok[(lomo_ok["holdout_model"] == holdout)]
            holdout_exps = fold[fold["model"].apply(normalize_model_name) == holdout]
            if len(holdout_exps) > 0:
                fold_e2e = holdout_exps["e2e_error"].mean() * 100
                print(f"LOMO holdout={holdout}: Mean E2E = {fold_e2e:.1f}% ({'PASS' if fold_e2e < 80 else 'FAIL'})")

    # LOWO: analyze H1 by workload
    if len(h1_ok) > 0:
        for model in h1_ok["model"].apply(normalize_model_name).unique():
            model_exps = h1_ok[h1_ok["model"].apply(normalize_model_name) == model]
            if len(model_exps) > 1:
                e2e_range = (model_exps["e2e_error"].max() - model_exps["e2e_error"].min()) * 100
                print(f"LOWO {model}: workload range = {e2e_range:.1f}pp")

    # Save results
    all_results = pd.concat([h1_results, control_results, lomo_df], ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, "idea1_results.csv")
    all_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Save coefficients
    coeff_path = os.path.join(OUTPUT_DIR, "predicted_coefficients.json")
    with open(coeff_path, "w") as f:
        json.dump({"h1_full": full_coefficients, "r4_actual": R4_COEFFICIENTS}, f, indent=2)
    print(f"Coefficients saved to {coeff_path}")


if __name__ == "__main__":
    main()
