#!/usr/bin/env python3
"""Run H1 (FairBatching 3-Coefficient) and H2 (Feature Scaling) experiments.

Loads step-level data with KV features, trains per-model models using:
  H1: OLS with FairBatching formulation (a + b*new_tokens + c*kv_sum)
  H2: Ridge with StandardScaler / log-transform of KV features

Evaluates on temporal test split (last 20% of steps per experiment).

Data source: BLIS-research/eval/ground_truth/ (10 experiments)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

# Add shared/ to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "shared"))
sys.path.insert(0, _SHARED_DIR)

from data_loader import load_all_experiments, parse_experiment_metadata
from evaluation import compute_mape, compute_mspe, compute_p99_error, compute_pearson_r
from lifecycle_kv_extractor import extract_all_experiments_kv_features
from splits import temporal_split

# Ground truth data location
DATA_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..",
        "..", "..", "BLIS-research", "eval", "ground_truth"
    )
)
# If relative path doesn't work (worktree), fall back to absolute
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/BLIS-research/eval/ground_truth"

TARGET_COL = "step.duration_us"

# Output directories
H1_OUTPUT = os.path.join(_SCRIPT_DIR, "h1-fairbatching-formulation", "output")
H2_OUTPUT = os.path.join(_SCRIPT_DIR, "h2-feature-scaling", "output")


def load_data_with_kv():
    """Load all experiments with KV features extracted from lifecycle data."""
    print(f"Loading data from: {DATA_ROOT}")
    df = extract_all_experiments_kv_features(DATA_ROOT)
    print(f"Loaded {len(df)} steps from {df['experiment_id'].nunique()} experiments")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"KV features present: kv_sum range [{df['kv_sum'].min():.0f}, {df['kv_sum'].max():.0f}]")

    # Derive canonical model+tp key from experiment metadata
    df["model_tp"] = df["model"] + "_tp" + df["tp"].astype(str)

    # Create combined new_tokens feature (prefill + decode)
    df["new_tokens"] = df["batch.prefill_tokens"].fillna(0) + df["batch.decode_tokens"].fillna(0)

    return df


def evaluate_model(predicted, actual, label=""):
    """Compute standard metrics."""
    mape = compute_mape(predicted, actual)
    mspe = compute_mspe(predicted, actual)
    r = compute_pearson_r(predicted, actual)
    p99 = compute_p99_error(predicted, actual)
    return {"label": label, "mape": mape, "mspe": mspe, "pearson_r": r, "p99_error": p99, "n": len(actual)}


def get_overhead_floor(model_tp):
    """Get per-model overhead floor from Round 2 calibrated artifacts."""
    artifact_dir = "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/BLIS-research/hypotheses/h-stepml/round2/idea-2-regime-ensemble/h3-secondary-method-calibration/output/calibrated_artifacts"

    fname = f"{model_tp}_regime.json"
    path = os.path.join(artifact_dir, fname)

    if os.path.isfile(path):
        with open(path) as f:
            art = json.load(f)
        return art.get("step_overhead_us", 0)

    # Fallback: try LOMO artifacts
    lomo_dir = artifact_dir.replace("h3-secondary-method-calibration", "h4-model-generalization").replace("calibrated_artifacts", "lomo_artifacts")
    path2 = os.path.join(lomo_dir, fname)
    if os.path.isfile(path2):
        with open(path2) as f:
            art = json.load(f)
        return art.get("step_overhead_us", 0)

    print(f"  WARNING: No overhead artifact found for {model_tp}")
    return 0


def apply_overhead_floor(predictions, overhead):
    """Apply overhead floor: step_cycle = max(overhead, prediction)."""
    if overhead > 0:
        return np.maximum(predictions, overhead)
    return predictions


# ---------------------------------------------------------------------------
# H1: FairBatching 3-Coefficient Formulation
# ---------------------------------------------------------------------------

def run_h1(df, splits):
    """H1: FairBatching formulation — OLS with new_tokens + kv_sum."""
    print("\n" + "="*70)
    print("H1: FairBatching 3-Coefficient Formulation")
    print("="*70)

    results = []
    coefficients = {}

    # Get unique model+tp combinations
    model_tps = sorted(df["model_tp"].unique())

    for model_tp in model_tps:
        print(f"\n--- Model: {model_tp} ---")
        mask = df["model_tp"] == model_tp
        model_df = df[mask]

        # Get temporal split indices for this model's experiments
        train_mask = mask & df.index.isin(splits["train"])
        test_mask = mask & df.index.isin(splits["test"])

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"  SKIP: no train or test data")
            continue

        overhead = get_overhead_floor(model_tp)
        print(f"  Overhead floor: {overhead:.1f} µs")
        print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

        y_train = train_df[TARGET_COL].values.astype(np.float64)
        y_test = test_df[TARGET_COL].values.astype(np.float64)

        # --- Variant A: 3-coeff (a + b*new_tokens + c*kv_sum) ---
        X_train_3 = train_df[["new_tokens", "kv_sum"]].values.astype(np.float64)
        X_test_3 = test_df[["new_tokens", "kv_sum"]].values.astype(np.float64)

        ols_3 = LinearRegression()
        ols_3.fit(X_train_3, y_train)
        pred_3 = apply_overhead_floor(ols_3.predict(X_test_3), overhead)
        metrics_3 = evaluate_model(pred_3, y_test, f"{model_tp}_3coeff")
        results.append(metrics_3)
        print(f"  3-coeff OLS: MAPE={metrics_3['mape']:.1f}%, r={metrics_3['pearson_r']:.3f}")
        print(f"    intercept={ols_3.intercept_:.3f}, b_new_tokens={ols_3.coef_[0]:.6f}, c_kv_sum={ols_3.coef_[1]:.6f}")

        # --- Variant B: 4-coeff (a + b*prefill + c*decode + d*kv_sum) ---
        feat_4 = ["batch.prefill_tokens", "batch.decode_tokens", "kv_sum"]
        X_train_4 = train_df[feat_4].values.astype(np.float64)
        X_test_4 = test_df[feat_4].values.astype(np.float64)

        ols_4 = LinearRegression()
        ols_4.fit(X_train_4, y_train)
        pred_4 = apply_overhead_floor(ols_4.predict(X_test_4), overhead)
        metrics_4 = evaluate_model(pred_4, y_test, f"{model_tp}_4coeff")
        results.append(metrics_4)
        print(f"  4-coeff OLS: MAPE={metrics_4['mape']:.1f}%, r={metrics_4['pearson_r']:.3f}")
        print(f"    intercept={ols_4.intercept_:.3f}, b_prefill={ols_4.coef_[0]:.6f}, c_decode={ols_4.coef_[1]:.6f}, d_kv_sum={ols_4.coef_[2]:.6f}")

        # --- Variant C: 4-coeff with regime split ---
        # Decode-only: prefill_tokens == 0
        decode_train = train_df[train_df["batch.prefill_tokens"] == 0]
        decode_test = test_df[test_df["batch.prefill_tokens"] == 0]
        mixed_train = train_df[train_df["batch.prefill_tokens"] > 0]
        mixed_test = test_df[test_df["batch.prefill_tokens"] > 0]

        pred_regime = np.zeros(len(test_df))

        if len(decode_train) > 5 and len(decode_test) > 0:
            X_d_train = decode_train[["batch.decode_tokens", "kv_sum"]].values.astype(np.float64)
            X_d_test = decode_test[["batch.decode_tokens", "kv_sum"]].values.astype(np.float64)
            ols_d = LinearRegression()
            ols_d.fit(X_d_train, decode_train[TARGET_COL].values.astype(np.float64))
            pred_d = ols_d.predict(X_d_test)
            pred_regime[test_df["batch.prefill_tokens"] == 0] = pred_d

        if len(mixed_train) > 5 and len(mixed_test) > 0:
            X_m_train = mixed_train[feat_4].values.astype(np.float64)
            X_m_test = mixed_test[feat_4].values.astype(np.float64)
            ols_m = LinearRegression()
            ols_m.fit(X_m_train, mixed_train[TARGET_COL].values.astype(np.float64))
            pred_m = ols_m.predict(X_m_test)
            pred_regime[test_df["batch.prefill_tokens"] > 0] = pred_m

        pred_regime = apply_overhead_floor(pred_regime, overhead)
        metrics_regime = evaluate_model(pred_regime, y_test, f"{model_tp}_regime_kv")
        results.append(metrics_regime)
        print(f"  Regime+KV OLS: MAPE={metrics_regime['mape']:.1f}%, r={metrics_regime['pearson_r']:.3f}")

        # --- Round 2 baseline: 2-coeff without KV ---
        X_train_2 = train_df[["batch.prefill_tokens", "batch.decode_tokens"]].values.astype(np.float64)
        X_test_2 = test_df[["batch.prefill_tokens", "batch.decode_tokens"]].values.astype(np.float64)
        ols_2 = LinearRegression()
        ols_2.fit(X_train_2, y_train)
        pred_2 = apply_overhead_floor(ols_2.predict(X_test_2), overhead)
        metrics_2 = evaluate_model(pred_2, y_test, f"{model_tp}_2coeff_baseline")
        results.append(metrics_2)
        print(f"  2-coeff baseline: MAPE={metrics_2['mape']:.1f}%, r={metrics_2['pearson_r']:.3f}")

        # Store best coefficients
        best_mape = min(metrics_3["mape"], metrics_4["mape"], metrics_regime["mape"])
        if best_mape == metrics_3["mape"]:
            coefficients[model_tp] = {
                "variant": "3coeff",
                "intercept": float(ols_3.intercept_),
                "new_tokens": float(ols_3.coef_[0]),
                "kv_sum": float(ols_3.coef_[1]),
                "test_mape": metrics_3["mape"],
                "overhead_us": overhead,
            }
        elif best_mape == metrics_4["mape"]:
            coefficients[model_tp] = {
                "variant": "4coeff",
                "intercept": float(ols_4.intercept_),
                "prefill_tokens": float(ols_4.coef_[0]),
                "decode_tokens": float(ols_4.coef_[1]),
                "kv_sum": float(ols_4.coef_[2]),
                "test_mape": metrics_4["mape"],
                "overhead_us": overhead,
            }
        else:
            coefficients[model_tp] = {
                "variant": "regime_kv",
                "test_mape": metrics_regime["mape"],
                "overhead_us": overhead,
            }

    # Summary
    print("\n" + "="*70)
    print("H1 SUMMARY")
    print("="*70)

    # Aggregate by variant
    for variant in ["3coeff", "4coeff", "regime_kv", "2coeff_baseline"]:
        variant_results = [r for r in results if variant in r["label"]]
        if variant_results:
            mapes = [r["mape"] for r in variant_results]
            print(f"  {variant}: mean MAPE = {np.mean(mapes):.1f}%, "
                  f"median = {np.median(mapes):.1f}%, "
                  f"range = [{min(mapes):.1f}%, {max(mapes):.1f}%]")

    # Save results
    os.makedirs(H1_OUTPUT, exist_ok=True)
    with open(os.path.join(H1_OUTPUT, "h1_results.json"), "w") as f:
        json.dump({"results": results, "coefficients": coefficients}, f, indent=2)

    return results, coefficients


# ---------------------------------------------------------------------------
# H2: Feature Scaling Comparison
# ---------------------------------------------------------------------------

def run_h2(df, splits):
    """H2: Feature scaling — StandardScaler, log-transform, combined."""
    print("\n" + "="*70)
    print("H2: Feature Scaling Comparison")
    print("="*70)

    results = []

    # Feature sets for Ridge
    kv_features = ["batch.prefill_tokens", "batch.decode_tokens",
                   "batch.num_prefill_reqs", "batch.num_decode_reqs",
                   "kv_sum", "kv_max", "kv_mean", "kv_std"]

    no_kv_features = ["batch.prefill_tokens", "batch.decode_tokens",
                      "batch.num_prefill_reqs", "batch.num_decode_reqs"]

    model_tps = sorted(df["model_tp"].unique())

    for model_tp in model_tps:
        print(f"\n--- Model: {model_tp} ---")
        mask = df["model_tp"] == model_tp

        train_mask = mask & df.index.isin(splits["train"])
        test_mask = mask & df.index.isin(splits["test"])

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        overhead = get_overhead_floor(model_tp)
        y_train = train_df[TARGET_COL].values.astype(np.float64)
        y_test = test_df[TARGET_COL].values.astype(np.float64)

        # --- Variant 1: Raw Ridge with KV (Round 2 reproduction) ---
        avail_kv = [f for f in kv_features if f in train_df.columns]
        X_train_raw = train_df[avail_kv].fillna(0).values.astype(np.float64)
        X_test_raw = test_df[avail_kv].fillna(0).values.astype(np.float64)

        ridge_raw = Ridge(alpha=1.0)
        ridge_raw.fit(X_train_raw, y_train)
        pred_raw = apply_overhead_floor(ridge_raw.predict(X_test_raw), overhead)
        metrics_raw = evaluate_model(pred_raw, y_test, f"{model_tp}_raw_ridge_kv")
        results.append(metrics_raw)
        print(f"  Raw Ridge+KV: MAPE={metrics_raw['mape']:.1f}%")

        # --- Variant 2: StandardScaler + Ridge with KV ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        ridge_scaled = Ridge(alpha=1.0)
        ridge_scaled.fit(X_train_scaled, y_train)
        pred_scaled = apply_overhead_floor(ridge_scaled.predict(X_test_scaled), overhead)
        metrics_scaled = evaluate_model(pred_scaled, y_test, f"{model_tp}_scaler_ridge_kv")
        results.append(metrics_scaled)
        print(f"  StandardScaler+Ridge+KV: MAPE={metrics_scaled['mape']:.1f}%")

        # --- Variant 3: Log-transform KV features + Ridge ---
        train_log = train_df.copy()
        test_log = test_df.copy()
        kv_log_cols = ["kv_sum", "kv_max", "kv_mean", "kv_std"]
        for col in kv_log_cols:
            if col in train_log.columns:
                train_log[col] = np.log1p(train_log[col].fillna(0).values.astype(np.float64))
                test_log[col] = np.log1p(test_log[col].fillna(0).values.astype(np.float64))

        avail_kv_log = [f for f in kv_features if f in train_log.columns]
        X_train_log = train_log[avail_kv_log].fillna(0).values.astype(np.float64)
        X_test_log = test_log[avail_kv_log].fillna(0).values.astype(np.float64)

        ridge_log = Ridge(alpha=1.0)
        ridge_log.fit(X_train_log, y_train)
        pred_log = apply_overhead_floor(ridge_log.predict(X_test_log), overhead)
        metrics_log = evaluate_model(pred_log, y_test, f"{model_tp}_log_ridge_kv")
        results.append(metrics_log)
        print(f"  Log-transform+Ridge+KV: MAPE={metrics_log['mape']:.1f}%")

        # --- Variant 4: StandardScaler + log-transform combined ---
        scaler_log = StandardScaler()
        X_train_log_scaled = scaler_log.fit_transform(X_train_log)
        X_test_log_scaled = scaler_log.transform(X_test_log)

        ridge_log_scaled = Ridge(alpha=1.0)
        ridge_log_scaled.fit(X_train_log_scaled, y_train)
        pred_log_scaled = apply_overhead_floor(ridge_log_scaled.predict(X_test_log_scaled), overhead)
        metrics_log_scaled = evaluate_model(pred_log_scaled, y_test, f"{model_tp}_log_scaler_ridge_kv")
        results.append(metrics_log_scaled)
        print(f"  Log+Scaler+Ridge+KV: MAPE={metrics_log_scaled['mape']:.1f}%")

        # --- Variant 5: No-KV Ridge baseline ---
        avail_no_kv = [f for f in no_kv_features if f in train_df.columns]
        X_train_no = train_df[avail_no_kv].fillna(0).values.astype(np.float64)
        X_test_no = test_df[avail_no_kv].fillna(0).values.astype(np.float64)

        ridge_no = Ridge(alpha=1.0)
        ridge_no.fit(X_train_no, y_train)
        pred_no = apply_overhead_floor(ridge_no.predict(X_test_no), overhead)
        metrics_no = evaluate_model(pred_no, y_test, f"{model_tp}_no_kv_ridge")
        results.append(metrics_no)
        print(f"  No-KV Ridge baseline: MAPE={metrics_no['mape']:.1f}%")

        # Ridge coefficient comparison
        print(f"\n  Ridge coefficient magnitudes:")
        print(f"    Raw:    {[f'{c:.4f}' for c in ridge_raw.coef_]}")
        print(f"    Scaled: {[f'{c:.4f}' for c in ridge_scaled.coef_]}")
        print(f"    Log:    {[f'{c:.4f}' for c in ridge_log.coef_]}")

    # Summary
    print("\n" + "="*70)
    print("H2 SUMMARY")
    print("="*70)

    for variant in ["raw_ridge_kv", "scaler_ridge_kv", "log_ridge_kv", "log_scaler_ridge_kv", "no_kv_ridge"]:
        variant_results = [r for r in results if variant in r["label"]]
        if variant_results:
            mapes = [r["mape"] for r in variant_results]
            print(f"  {variant}: mean MAPE = {np.mean(mapes):.1f}%, "
                  f"median = {np.median(mapes):.1f}%, "
                  f"range = [{min(mapes):.1f}%, {max(mapes):.1f}%]")

    # Save results
    os.makedirs(H2_OUTPUT, exist_ok=True)
    with open(os.path.join(H2_OUTPUT, "h2_results.json"), "w") as f:
        json.dump({"results": results}, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Per-experiment analysis for both H1 and H2
# ---------------------------------------------------------------------------

def run_per_experiment_analysis(df, splits):
    """Run the best H1 and H2 variants per experiment for detailed analysis."""
    print("\n" + "="*70)
    print("PER-EXPERIMENT ANALYSIS (Best Variants)")
    print("="*70)

    experiments = sorted(df["experiment_id"].unique())
    per_exp_results = []

    for exp_id in experiments:
        mask = df["experiment_id"] == exp_id
        model_tp = df[mask]["model_tp"].iloc[0]

        train_mask = mask & df.index.isin(splits["train"])
        test_mask = mask & df.index.isin(splits["test"])

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            continue

        overhead = get_overhead_floor(model_tp)
        y_test = test_df[TARGET_COL].values.astype(np.float64)

        # Train per-model (using ALL training data for this model, not just this experiment)
        model_mask = df["model_tp"] == model_tp
        full_train = df[model_mask & df.index.isin(splits["train"])]
        y_full_train = full_train[TARGET_COL].values.astype(np.float64)

        # Best H1 variant: 4-coeff OLS
        X_tr = full_train[["batch.prefill_tokens", "batch.decode_tokens", "kv_sum"]].values.astype(np.float64)
        X_te = test_df[["batch.prefill_tokens", "batch.decode_tokens", "kv_sum"]].values.astype(np.float64)
        ols = LinearRegression()
        ols.fit(X_tr, y_full_train)
        pred = apply_overhead_floor(ols.predict(X_te), overhead)

        mape = compute_mape(pred, y_test)
        mspe = compute_mspe(pred, y_test)
        r = compute_pearson_r(pred, y_test)

        per_exp_results.append({
            "experiment": exp_id,
            "model_tp": model_tp,
            "model": df[mask]["model"].iloc[0],
            "workload": df[mask]["workload"].iloc[0],
            "n_test": len(test_df),
            "mape": mape,
            "mspe": mspe,
            "pearson_r": r,
            "overhead_us": overhead,
        })

        print(f"  {exp_id}: MAPE={mape:.1f}%, MSPE={mspe:.1f}%, r={r:.3f}")

    return per_exp_results


def main():
    # Load data
    df = load_data_with_kv()

    # Apply temporal split
    splits = temporal_split(df)
    print(f"\nTemporal split: train={len(splits['train'])}, "
          f"valid={len(splits['valid'])}, test={len(splits['test'])}")

    # Run H1
    h1_results, h1_coefficients = run_h1(df, splits)

    # Run H2
    h2_results = run_h2(df, splits)

    # Per-experiment analysis
    per_exp = run_per_experiment_analysis(df, splits)

    # Save per-experiment results
    with open(os.path.join(H1_OUTPUT, "per_experiment.json"), "w") as f:
        json.dump(per_exp, f, indent=2)

    # Print overall winner comparison
    print("\n" + "="*70)
    print("OVERALL COMPARISON: H1 vs H2 vs Baselines")
    print("="*70)

    # Collect mean MAPEs per variant across models
    all_results = h1_results + h2_results
    variant_mapes = {}
    for r in all_results:
        # Extract variant from label
        parts = r["label"].split("_")
        # model_tp is first 2-3 parts, variant is rest
        if "3coeff" in r["label"]:
            variant = "H1: 3-coeff OLS"
        elif "4coeff" in r["label"]:
            variant = "H1: 4-coeff OLS"
        elif "regime_kv" in r["label"]:
            variant = "H1: Regime+KV OLS"
        elif "2coeff_baseline" in r["label"]:
            variant = "Baseline: 2-coeff OLS"
        elif "raw_ridge_kv" in r["label"]:
            variant = "H2: Raw Ridge+KV"
        elif "scaler_ridge_kv" in r["label"]:
            variant = "H2: StandardScaler+Ridge+KV"
        elif "log_scaler_ridge_kv" in r["label"]:
            variant = "H2: Log+Scaler+Ridge+KV"
        elif "log_ridge_kv" in r["label"]:
            variant = "H2: Log-transform+Ridge+KV"
        elif "no_kv_ridge" in r["label"]:
            variant = "Baseline: No-KV Ridge"
        else:
            variant = r["label"]

        variant_mapes.setdefault(variant, []).append(r["mape"])

    print(f"\n{'Variant':<40} {'Mean MAPE':<12} {'Median':<12} {'Min':<10} {'Max':<10}")
    print("-" * 84)
    for variant, mapes in sorted(variant_mapes.items(), key=lambda x: np.mean(x[1])):
        print(f"  {variant:<38} {np.mean(mapes):>8.1f}%  {np.median(mapes):>8.1f}%  "
              f"{min(mapes):>6.1f}%  {max(mapes):>6.1f}%")

    # Determine best overall
    best_variant = min(variant_mapes.items(), key=lambda x: np.mean(x[1]))
    print(f"\nBest variant: {best_variant[0]} (mean MAPE = {np.mean(best_variant[1]):.1f}%)")

    # Save comprehensive summary
    summary = {
        "variant_comparison": {
            k: {"mean_mape": float(np.mean(v)), "median_mape": float(np.median(v)),
                "min_mape": float(min(v)), "max_mape": float(max(v)), "n_models": len(v)}
            for k, v in variant_mapes.items()
        },
        "best_variant": best_variant[0],
        "best_mean_mape": float(np.mean(best_variant[1])),
        "h1_coefficients": h1_coefficients,
        "per_experiment": per_exp,
    }

    with open(os.path.join(H1_OUTPUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(H2_OUTPUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResults saved to h1-fairbatching-formulation/output/ and h2-feature-scaling/output/")


if __name__ == "__main__":
    main()
