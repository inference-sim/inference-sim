#!/usr/bin/env python3
"""Idea 2: Normalized Feature Space with Scale-Invariant Regression.

Trains a SINGLE Ridge regression across all models on normalized features.
Step times are normalized by metadata-derived overhead, removing 3+ OOM scale
variation. Features capture relative batch composition (fullness ratios).

Training uses E2E-derived target step times (from lifecycle data) as the
"true" step cycle time, NOT raw step.duration_us (which is GPU-only).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "shared")
sys.path.insert(0, _SHARED_DIR)

from data_loader import load_all_experiments, load_lifecycle_data, parse_experiment_metadata, DEFAULT_DATA_ROOT
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

# ─── Model metadata ───

MODEL_METADATA = {
    "llama-2-7b": {"params_b": 7.0, "tp": 1, "is_moe": False, "active_params_b": 7.0},
    "codellama-34b": {"params_b": 34.0, "tp": 2, "is_moe": False, "active_params_b": 34.0},
    "llama-2-70b": {"params_b": 70.0, "tp": 4, "is_moe": False, "active_params_b": 70.0},
    "llama-2-70b-hf": {"params_b": 70.0, "tp": 4, "is_moe": False, "active_params_b": 70.0},
    "mixtral-8x7b-v0-1": {"params_b": 46.7, "tp": 2, "is_moe": True, "active_params_b": 12.9},
}

MODEL_ALIASES = {"llama-2-70b-hf": "llama-2-70b"}
CANONICAL_MODELS = ["llama-2-7b", "codellama-34b", "llama-2-70b", "mixtral-8x7b-v0-1"]

# vLLM config (from exp-config.yaml defaults)
DEFAULT_MAX_NUM_SEQS = 128
DEFAULT_MAX_BATCHED_TOKENS = 2048


def normalize_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def metadata_overhead(params_b: float) -> float:
    """Estimate step overhead from model params using power law.

    Based on R4: beta0 ≈ 9741 × (params/7)^0.30
    """
    return 9741.0 * (params_b / 7.0) ** 0.30


# ─── Compute E2E-derived target step times per model ───

def compute_target_step_times(data_root: str) -> dict:
    """Compute per-model target step time from lifecycle E2E data.

    target_step = (mean_E2E - mean_TTFT) / mean_output_len
    This is the step time that reproduces observed E2E in BLIS.
    """
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
            target_step = (e2e_us - ttft_us) / output_len
            model_data.setdefault(model, []).append(target_step)

    result = {}
    for model, values in model_data.items():
        result[model] = float(np.mean(values))
        print(f"  {model}: target_step = {result[model]:.0f}μs")

    return result


# ─── Train unified normalized Ridge ───

def train_normalized_ridge(
    step_df: pd.DataFrame,
    target_steps: dict,
    train_models: list[str],
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    max_batched_tokens: int = DEFAULT_MAX_BATCHED_TOKENS,
) -> tuple:
    """Train a single Ridge on normalized features across models.

    For each step in the training data:
    - target = target_step_time / overhead_est  (normalized step time)
    - features = [decode_tokens/max_seqs, prefill_tokens/max_batched_tokens]

    Returns (ridge_model, base_overhead_multiplier) — the overhead multiplier
    calibrates the metadata formula to match E2E-derived targets.
    """
    # Filter to training models
    train_df = step_df[step_df["model"].apply(normalize_model).isin(train_models)].copy()

    # Compute overhead estimate per model
    train_df["overhead_est"] = train_df["model"].apply(
        lambda m: metadata_overhead(MODEL_METADATA.get(m, MODEL_METADATA.get(normalize_model(m), {})).get("params_b", 7.0))
    )

    # Compute calibration multiplier: target_step / overhead_est per model
    # This scales the metadata formula to match E2E-derived reality
    calibration_multipliers = {}
    for model in train_models:
        if model in target_steps:
            oh_est = metadata_overhead(MODEL_METADATA[model]["params_b"])
            calibration_multipliers[model] = target_steps[model] / oh_est
            print(f"  {model}: overhead_est={oh_est:.0f}, target={target_steps[model]:.0f}, "
                  f"multiplier={calibration_multipliers[model]:.3f}")

    # Use mean multiplier across training models
    mean_mult = np.mean(list(calibration_multipliers.values()))
    print(f"  Mean calibration multiplier: {mean_mult:.3f}")

    # Create features
    decode_col = "batch.decode_tokens" if "batch.decode_tokens" in train_df.columns else "batch.scheduled_tokens"
    prefill_col = "batch.prefill_tokens"

    train_df["norm_decode"] = train_df[decode_col].fillna(0).astype(float) / max_num_seqs
    train_df["norm_prefill"] = train_df.get(prefill_col, pd.Series(0, index=train_df.index)).fillna(0).astype(float) / max_batched_tokens

    # Target: target_step / (overhead_est * mean_mult)  [should be ~1.0 + small variation]
    train_df["calibrated_overhead"] = train_df["overhead_est"] * mean_mult
    # Use step.duration_us scaled by per-model ratio as target for per-step variation
    # But the absolute level comes from target_step
    train_df["model_norm"] = train_df["model"].apply(normalize_model)
    train_df["target_step"] = train_df["model_norm"].map(target_steps)
    train_df = train_df.dropna(subset=["target_step"])

    # Normalized target: how much does this step deviate from the model's mean?
    # We use: normalized_step = step.duration_us / mean(step.duration_us_for_model) * (target_step / calibrated_overhead)
    # This preserves the relative variation from step data but scales to E2E-derived absolute level
    model_mean_dur = train_df.groupby("model_norm")["step.duration_us"].transform("mean")
    train_df["norm_target"] = (train_df["step.duration_us"].astype(float) / model_mean_dur) * (train_df["target_step"] / train_df["calibrated_overhead"])

    X = train_df[["norm_decode", "norm_prefill"]].values
    y = train_df["norm_target"].values

    # Remove NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    ridge = Ridge(alpha=10.0)
    ridge.fit(X, y)

    print(f"  Ridge: intercept={ridge.intercept_:.4f}, "
          f"w_decode={ridge.coef_[0]:.4f}, w_prefill={ridge.coef_[1]:.4f}")

    return ridge, mean_mult


def predict_coefficients(
    model_name: str,
    ridge,
    calibration_mult: float,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    max_batched_tokens: int = DEFAULT_MAX_BATCHED_TOKENS,
    target_steps: dict = None,
) -> dict:
    """Convert unified Ridge weights to per-model BlackboxLatencyModel coefficients.

    step_time = overhead * mult * (intercept + w1*decode/max_seqs + w2*prefill/max_tokens)
    = overhead*mult*intercept + overhead*mult*w1/max_seqs * decode + overhead*mult*w2/max_tokens * prefill
    = beta0 + beta2*decode + beta1*prefill
    """
    meta = MODEL_METADATA[model_name]
    overhead_est = metadata_overhead(meta["params_b"])
    scaled_overhead = overhead_est * calibration_mult

    beta0 = scaled_overhead * ridge.intercept_
    beta2 = scaled_overhead * ridge.coef_[0] / max_num_seqs
    beta1 = scaled_overhead * ridge.coef_[1] / max_batched_tokens

    # For alpha0, use target_steps data if available, else scale from metadata
    if target_steps and model_name in target_steps:
        # Use lifecycle TTFT data (this is "training data", not held-out)
        alpha0 = _compute_ttft(model_name)
    else:
        # Metadata-only: scale from 7B
        alpha0 = 27129 * (meta["params_b"] / 7.0) ** 0.45

    return {"alpha0": alpha0, "beta0": beta0, "beta1": beta1, "beta2": beta2}


def _compute_ttft(model_name: str) -> float:
    """Compute mean TTFT from lifecycle data for a model."""
    data_root = DEFAULT_DATA_ROOT
    ttft_values = []
    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        summary_path = os.path.join(dirpath, "results", "summary_lifecycle_metrics.json")
        if not os.path.isfile(summary_path):
            continue

        meta = parse_experiment_metadata(dirname)
        if normalize_model(meta["model"]) != model_name:
            continue

        gt = load_ground_truth_metrics(dirpath)
        ttft_values.append(gt["ttft_mean_s"] * 1e6)

    return float(np.mean(ttft_values)) if ttft_values else 27129 * (MODEL_METADATA.get(model_name, {}).get("params_b", 7) / 7) ** 0.45


# ─── BLIS Validation (shared with Idea 1) ───

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

        print(f"  [{label}] {dirname}: E2E={e2e_err*100:.1f}%, TTFT={ttft_err*100:.1f}%")

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
    print(f"  Loaded {len(step_df)} steps from {step_df['model'].nunique()} models")

    print("\nComputing E2E-derived target step times...")
    target_steps = compute_target_step_times(DEFAULT_DATA_ROOT)

    # ── H1: Unified Ridge (all models) ──
    print("\n=== H1: Unified Normalized Ridge (all models) ===\n")
    ridge, cal_mult = train_normalized_ridge(step_df, target_steps, CANONICAL_MODELS)

    h1_coefficients = {}
    print("\nPredicted coefficients:")
    for m in CANONICAL_MODELS:
        coeffs = predict_coefficients(m, ridge, cal_mult, target_steps=target_steps)
        h1_coefficients[m] = coeffs
        print(f"  {m}: alpha0={coeffs['alpha0']:.0f}, beta0={coeffs['beta0']:.0f}, "
              f"beta1={coeffs['beta1']:.2f}, beta2={coeffs['beta2']:.1f}")

    print("\nRunning BLIS validation...")
    h1_results = run_validation(h1_coefficients, "h1-unified")

    # ── H2: LOMO ──
    print("\n=== H2: LOMO ===\n")
    lomo_results = []
    for holdout in CANONICAL_MODELS:
        train_models = [m for m in CANONICAL_MODELS if m != holdout]
        print(f"\n--- Holdout: {holdout} ---")

        # Compute target steps only for training models
        train_targets = {m: target_steps[m] for m in train_models if m in target_steps}

        try:
            fold_ridge, fold_mult = train_normalized_ridge(step_df, train_targets, train_models)
        except Exception as e:
            print(f"  Training failed: {e}")
            continue

        fold_coefficients = {}
        for m in CANONICAL_MODELS:
            ts = train_targets if m != holdout else None
            fold_coefficients[m] = predict_coefficients(m, fold_ridge, fold_mult, target_steps=ts)

        pred = fold_coefficients[holdout]
        print(f"  Holdout {holdout}: beta0={pred['beta0']:.0f}, beta2={pred['beta2']:.1f}")

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
        print(f"H1 (unified Ridge): Mean E2E = {mean_e2e:.1f}%, {below_10}/{len(h1_ok)} <10%")

    if len(lomo_df) > 0:
        lomo_ok = lomo_df[lomo_df["status"] == "ok"]
        for holdout in CANONICAL_MODELS:
            fold = lomo_ok[lomo_ok["holdout_model"] == holdout]
            holdout_exps = fold[fold["model"].apply(normalize_model) == holdout]
            if len(holdout_exps) > 0:
                fold_e2e = holdout_exps["e2e_error"].mean() * 100
                print(f"LOMO holdout={holdout}: E2E={fold_e2e:.1f}% ({'PASS' if fold_e2e < 80 else 'FAIL'})")

    # LOWO
    if len(h1_ok) > 0:
        for model in h1_ok["model"].apply(normalize_model).unique():
            model_exps = h1_ok[h1_ok["model"].apply(normalize_model) == model]
            if len(model_exps) > 1:
                e2e_range = (model_exps["e2e_error"].max() - model_exps["e2e_error"].min()) * 100
                print(f"LOWO {model}: workload range = {e2e_range:.1f}pp")

    all_results = pd.concat([h1_results, lomo_df], ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, "idea2_results.csv")
    all_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
