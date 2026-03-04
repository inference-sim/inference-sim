#!/usr/bin/env python3
"""Run H4 (LOWO) and H5 (LOMO) generalization experiments for Idea 2.

H4: Leave-One-Workload-Out — train FairBatching 3-coeff OLS on 2 workloads,
    evaluate on held-out 3rd workload (per model).
H5: Leave-One-Model-Out — train pooled OLS on 3 model groups,
    evaluate on held-out 4th model.

Data source: BLIS-research/eval/ground_truth/ (10 experiments)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Add shared/ to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "shared"))
sys.path.insert(0, _SHARED_DIR)

from data_loader import load_all_experiments, parse_experiment_metadata
from evaluation import compute_mape, compute_pearson_r
from lifecycle_kv_extractor import extract_all_experiments_kv_features
from splits import leave_one_model_out, leave_one_workload_out, temporal_split

# Ground truth data location
DATA_ROOT = "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/BLIS-research/eval/ground_truth"
TARGET_COL = "step.duration_us"

# Output directories
H4_OUTPUT = os.path.join(_SCRIPT_DIR, "h4-lowo-generalization", "output")
H5_OUTPUT = os.path.join(_SCRIPT_DIR, "h5-lomo-generalization", "output")


def get_overhead_floor(model_tp):
    """Get per-model overhead floor from Round 2 calibrated artifacts."""
    artifact_dir = "/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/BLIS-research/hypotheses/h-stepml/round2/idea-2-regime-ensemble/h3-secondary-method-calibration/output/calibrated_artifacts"
    fname = f"{model_tp}_regime.json"
    path = os.path.join(artifact_dir, fname)
    if os.path.isfile(path):
        with open(path) as f:
            art = json.load(f)
        return art.get("step_overhead_us", 0)
    # Fallback for llama-2-70b-hf variant
    if "70b-hf" in model_tp:
        alt = model_tp.replace("70b-hf", "70b")
        path2 = os.path.join(artifact_dir, f"{alt}_regime.json")
        if os.path.isfile(path2):
            with open(path2) as f:
                art = json.load(f)
            return art.get("step_overhead_us", 0)
    print(f"  WARNING: No overhead artifact found for {model_tp}")
    return 0


def apply_overhead_floor(predictions, overhead):
    if overhead > 0:
        return np.maximum(predictions, overhead)
    return predictions


def load_data():
    """Load all experiments with KV features."""
    print(f"Loading data from: {DATA_ROOT}")
    df = extract_all_experiments_kv_features(DATA_ROOT)
    print(f"Loaded {len(df)} steps from {df['experiment_id'].nunique()} experiments")

    df["model_tp"] = df["model"] + "_tp" + df["tp"].astype(str)
    df["new_tokens"] = df["batch.prefill_tokens"].fillna(0) + df["batch.decode_tokens"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# H4: Leave-One-Workload-Out
# ---------------------------------------------------------------------------
def run_h4_lowo(df):
    """H4: LOWO — train on 2 workloads, test on held-out 3rd (per model)."""
    print("\n" + "=" * 70)
    print("H4: LEAVE-ONE-WORKLOAD-OUT (LOWO) GENERALIZATION")
    print("=" * 70)

    # Use temporal split to get train portion only (avoid test leakage)
    splits = temporal_split(df)
    train_idx = set(splits["train"].tolist())

    # LOWO folds
    lowo_folds = leave_one_workload_out(df)
    model_tps = sorted(df["model_tp"].unique())

    all_results = []

    for fold in lowo_folds:
        holdout_wl = fold["holdout_workload"]
        print(f"\n--- Holdout workload: {holdout_wl} ---")

        for model_tp in model_tps:
            model_mask = df["model_tp"] == model_tp

            # Check this model has the held-out workload
            model_workloads = df[model_mask]["workload"].unique()
            if holdout_wl not in model_workloads:
                continue
            if len(model_workloads) < 2:
                print(f"  {model_tp}: SKIP (only {len(model_workloads)} workloads)")
                continue

            # Train: this model's data from non-held-out workloads, temporal train portion
            fold_train_mask = model_mask & df.index.isin(fold["train"]) & df.index.isin(train_idx)
            # Test: this model's held-out workload, temporal test portion
            fold_test_mask = model_mask & df.index.isin(fold["test"]) & df.index.isin(splits["test"])

            train_df = df[fold_train_mask]
            test_df = df[fold_test_mask]

            if len(train_df) < 10 or len(test_df) < 5:
                print(f"  {model_tp}: SKIP (train={len(train_df)}, test={len(test_df)})")
                continue

            overhead = get_overhead_floor(model_tp)
            y_train = train_df[TARGET_COL].values.astype(np.float64)
            y_test = test_df[TARGET_COL].values.astype(np.float64)

            # 3-coeff OLS: a + b*new_tokens + c*kv_sum
            X_train = train_df[["new_tokens", "kv_sum"]].values.astype(np.float64)
            X_test = test_df[["new_tokens", "kv_sum"]].values.astype(np.float64)

            ols = LinearRegression()
            ols.fit(X_train, y_train)
            pred = apply_overhead_floor(ols.predict(X_test), overhead)

            mape = compute_mape(pred, y_test)
            r = compute_pearson_r(pred, y_test)

            # In-distribution comparison: train on ALL workloads for this model
            id_train_mask = model_mask & df.index.isin(train_idx)
            id_test_mask = model_mask & (df["workload"] == holdout_wl) & df.index.isin(splits["test"])
            X_id_train = df[id_train_mask][["new_tokens", "kv_sum"]].values.astype(np.float64)
            y_id_train = df[id_train_mask][TARGET_COL].values.astype(np.float64)
            ols_id = LinearRegression()
            ols_id.fit(X_id_train, y_id_train)
            pred_id = apply_overhead_floor(ols_id.predict(X_test), overhead)
            mape_id = compute_mape(pred_id, y_test)

            result = {
                "model_tp": model_tp,
                "holdout_workload": holdout_wl,
                "train_workloads": [w for w in model_workloads if w != holdout_wl],
                "n_train": len(train_df),
                "n_test": len(test_df),
                "lowo_mape": float(mape),
                "lowo_pearson_r": float(r),
                "in_dist_mape": float(mape_id),
                "mape_degradation_pp": float(mape - mape_id),
                "intercept": float(ols.intercept_),
                "coef_new_tokens": float(ols.coef_[0]),
                "coef_kv_sum": float(ols.coef_[1]),
                "overhead_us": overhead,
            }
            all_results.append(result)

            print(f"  {model_tp}: LOWO MAPE={mape:.1f}% (in-dist={mape_id:.1f}%, "
                  f"degradation={mape - mape_id:+.1f}pp)")

    # Summary
    print("\n" + "=" * 70)
    print("H4 LOWO SUMMARY")
    print("=" * 70)

    if all_results:
        mapes = [r["lowo_mape"] for r in all_results]
        degr = [r["mape_degradation_pp"] for r in all_results]
        print(f"  Mean LOWO MAPE: {np.mean(mapes):.1f}%")
        print(f"  Median LOWO MAPE: {np.median(mapes):.1f}%")
        print(f"  Range: [{min(mapes):.1f}%, {max(mapes):.1f}%]")
        print(f"  Mean degradation vs in-dist: {np.mean(degr):+.1f}pp")
        print(f"  R2 LOWO baseline: 117.4%")
        print(f"  Supported threshold: <70%")

        # Per-model summary
        for mt in sorted(set(r["model_tp"] for r in all_results)):
            mt_results = [r for r in all_results if r["model_tp"] == mt]
            mt_mapes = [r["lowo_mape"] for r in mt_results]
            print(f"  {mt}: mean LOWO MAPE = {np.mean(mt_mapes):.1f}% "
                  f"(folds: {[f'{m:.1f}%' for m in mt_mapes]})")
    else:
        print("  No results produced!")

    os.makedirs(H4_OUTPUT, exist_ok=True)
    summary = {
        "folds": all_results,
        "aggregate": {
            "mean_lowo_mape": float(np.mean(mapes)) if all_results else None,
            "median_lowo_mape": float(np.median(mapes)) if all_results else None,
            "mean_degradation_pp": float(np.mean(degr)) if all_results else None,
            "n_folds": len(all_results),
            "r2_baseline_lowo": 117.4,
            "supported_threshold": 70.0,
        },
    }
    with open(os.path.join(H4_OUTPUT, "h4_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {H4_OUTPUT}")

    return summary


# ---------------------------------------------------------------------------
# H5: Leave-One-Model-Out
# ---------------------------------------------------------------------------
def run_h5_lomo(df):
    """H5: LOMO — train on 3 model groups, test on held-out 4th."""
    print("\n" + "=" * 70)
    print("H5: LEAVE-ONE-MODEL-OUT (LOMO) GENERALIZATION")
    print("=" * 70)

    # Use temporal split to get train portion only
    splits = temporal_split(df)
    train_idx = set(splits["train"].tolist())

    # LOMO folds
    lomo_folds = leave_one_model_out(df)  # Note: this normalizes model names in-place

    all_results = []

    for fold in lomo_folds:
        holdout_model = fold["holdout_model"]
        print(f"\n--- Holdout model: {holdout_model} ---")

        # Train data: non-held-out models, temporal train portion
        fold_train_mask = df.index.isin(fold["train"]) & df.index.isin(train_idx)
        fold_test_mask = df.index.isin(fold["test"]) & df.index.isin(splits["test"])

        train_df = df[fold_train_mask]
        test_df = df[fold_test_mask]

        if len(train_df) < 10 or len(test_df) < 5:
            print(f"  SKIP: insufficient data (train={len(train_df)}, test={len(test_df)})")
            continue

        y_train = train_df[TARGET_COL].values.astype(np.float64)
        y_test = test_df[TARGET_COL].values.astype(np.float64)

        # 3-coeff OLS on pooled training data
        X_train = train_df[["new_tokens", "kv_sum"]].values.astype(np.float64)
        X_test = test_df[["new_tokens", "kv_sum"]].values.astype(np.float64)

        ols = LinearRegression()
        ols.fit(X_train, y_train)
        pred_raw = ols.predict(X_test)

        # Apply held-out model's overhead floor
        # Find the model_tp values for the held-out model
        holdout_model_tps = sorted(test_df["model_tp"].unique())
        per_model_tp_results = []

        for mt in holdout_model_tps:
            mt_mask = test_df["model_tp"] == mt
            mt_test = test_df[mt_mask]
            mt_pred_raw = pred_raw[mt_mask.values]
            overhead = get_overhead_floor(mt)
            mt_pred = apply_overhead_floor(mt_pred_raw, overhead)
            mt_actual = mt_test[TARGET_COL].values.astype(np.float64)

            mt_mape = compute_mape(mt_pred, mt_actual)
            mt_r = compute_pearson_r(mt_pred, mt_actual)

            # In-distribution: per-model training baseline
            mt_all_train = df[(df["model_tp"] == mt) & df.index.isin(train_idx)]
            if len(mt_all_train) >= 5:
                X_id = mt_all_train[["new_tokens", "kv_sum"]].values.astype(np.float64)
                y_id = mt_all_train[TARGET_COL].values.astype(np.float64)
                ols_id = LinearRegression()
                ols_id.fit(X_id, y_id)
                pred_id = apply_overhead_floor(ols_id.predict(test_df[mt_mask][["new_tokens", "kv_sum"]].values.astype(np.float64)), overhead)
                mape_id = compute_mape(pred_id, mt_actual)
            else:
                mape_id = None

            per_model_tp_results.append({
                "model_tp": mt,
                "lomo_mape": float(mt_mape),
                "lomo_pearson_r": float(mt_r),
                "in_dist_mape": float(mape_id) if mape_id is not None else None,
                "n_test": len(mt_test),
                "overhead_us": overhead,
            })

            id_str = f"in-dist={mape_id:.1f}%" if mape_id is not None else "no in-dist baseline"
            print(f"  {mt}: LOMO MAPE={mt_mape:.1f}% ({id_str})")

        # Overall fold stats
        fold_mapes = [r["lomo_mape"] for r in per_model_tp_results]
        mean_fold_mape = float(np.mean(fold_mapes))
        print(f"  Fold mean LOMO MAPE: {mean_fold_mape:.1f}%")

        fold_result = {
            "holdout_model": holdout_model,
            "train_models": sorted(set(train_df["model"].unique())),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mean_lomo_mape": mean_fold_mape,
            "per_model_tp": per_model_tp_results,
            "pooled_intercept": float(ols.intercept_),
            "pooled_coef_new_tokens": float(ols.coef_[0]),
            "pooled_coef_kv_sum": float(ols.coef_[1]),
        }
        all_results.append(fold_result)

    # Summary
    print("\n" + "=" * 70)
    print("H5 LOMO SUMMARY")
    print("=" * 70)

    if all_results:
        fold_mapes = [r["mean_lomo_mape"] for r in all_results]
        print(f"  Mean LOMO MAPE (across folds): {np.mean(fold_mapes):.1f}%")
        print(f"  Per-fold: {[f'{m:.1f}%' for m in fold_mapes]}")
        print(f"  R1 LOMO baseline: 2,559.7%")
        print(f"  R2 LOMO baseline: 108.6%")
        print(f"  Supported threshold: <80%")

        # Which model is hardest?
        all_mt_results = [r for fold in all_results for r in fold["per_model_tp"]]
        for mt in sorted(set(r["model_tp"] for r in all_mt_results)):
            mt_r = [r for r in all_mt_results if r["model_tp"] == mt]
            mt_m = [r["lomo_mape"] for r in mt_r]
            print(f"  {mt}: LOMO MAPE = {np.mean(mt_m):.1f}%")

        # Coefficient comparison
        print(f"\n  Pooled coefficients across folds:")
        for fold in all_results:
            print(f"    Holdout {fold['holdout_model']}: "
                  f"intercept={fold['pooled_intercept']:.1f}, "
                  f"new_tokens={fold['pooled_coef_new_tokens']:.4f}, "
                  f"kv_sum={fold['pooled_coef_kv_sum']:.6f}")
    else:
        print("  No results produced!")
        fold_mapes = []

    os.makedirs(H5_OUTPUT, exist_ok=True)
    summary = {
        "folds": all_results,
        "aggregate": {
            "mean_lomo_mape": float(np.mean(fold_mapes)) if fold_mapes else None,
            "per_fold_mapes": [float(m) for m in fold_mapes],
            "n_folds": len(all_results),
            "r1_baseline_lomo": 2559.7,
            "r2_baseline_lomo": 108.6,
            "supported_threshold": 80.0,
        },
    }
    with open(os.path.join(H5_OUTPUT, "h5_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {H5_OUTPUT}")

    return summary


def main():
    df = load_data()
    h4_summary = run_h4_lowo(df)
    h5_summary = run_h5_lomo(df)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if h4_summary["aggregate"]["mean_lowo_mape"] is not None:
        print(f"  H4 LOWO: {h4_summary['aggregate']['mean_lowo_mape']:.1f}% "
              f"(threshold <70%, R2 baseline 117.4%)")
    if h5_summary["aggregate"]["mean_lomo_mape"] is not None:
        print(f"  H5 LOMO: {h5_summary['aggregate']['mean_lomo_mape']:.1f}% "
              f"(threshold <80%, R2 baseline 108.6%)")


if __name__ == "__main__":
    main()
