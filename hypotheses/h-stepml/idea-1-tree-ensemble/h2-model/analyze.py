"""Evaluate XGBoost per-experiment models against baselines.

Computes per-step MAPE, MSPE, Pearson r, p99 error, and workload-level
E2E mean error for each experiment. Compares against blackbox and Ridge baselines.

Success criteria (from HYPOTHESIS.md):
  Per-step MAPE < 15% AND workload-level E2E mean error < 10% on >= 12/16 experiments.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from evaluation import compute_mape, compute_mspe, compute_p99_error, compute_pearson_r
from baselines import BlackboxBaseline
from splits import temporal_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H1_DIR = os.path.join(SCRIPT_DIR, "..", "h1-features")
TARGET_COL = "step.duration_us"


def main():
    # Load XGBoost predictions
    pred_path = os.path.join(SCRIPT_DIR, "output", "predictions.csv")
    preds_df = pd.read_csv(pred_path)

    # Load features for baseline computation
    feat_path = os.path.join(H1_DIR, "output", "features.csv")
    feat_df = pd.read_csv(feat_path)

    # --- Per-experiment evaluation ---
    results = []

    for exp_id in sorted(preds_df["experiment_id"].unique()):
        exp_preds = preds_df[preds_df["experiment_id"] == exp_id]
        model_name = exp_preds["model"].iloc[0]
        workload = exp_preds["workload"].iloc[0]

        xgb_pred = exp_preds["predicted_us"].values
        xgb_actual = exp_preds[TARGET_COL].values

        xgb_mape = compute_mape(xgb_pred, xgb_actual)
        xgb_mspe = compute_mspe(xgb_pred, xgb_actual)
        xgb_r = compute_pearson_r(xgb_pred, xgb_actual)
        xgb_p99 = compute_p99_error(xgb_pred, xgb_actual)

        # Blackbox baseline on same test set
        exp_feat = feat_df[feat_df["experiment_id"] == exp_id]
        exp_split = temporal_split(exp_feat)
        exp_train = exp_feat.loc[exp_split["train"]]
        exp_test = exp_feat.loc[exp_split["test"]]

        bb = BlackboxBaseline().fit(exp_train)
        bb_pred = bb.predict(exp_test)
        bb_actual = exp_test[TARGET_COL].values
        bb_mape = compute_mape(bb_pred, bb_actual)
        bb_mspe = compute_mspe(bb_pred, bb_actual)

        results.append({
            "experiment_id": exp_id,
            "model": model_name,
            "workload": workload,
            "xgb_mape": xgb_mape,
            "xgb_mspe": xgb_mspe,
            "xgb_r": xgb_r,
            "xgb_p99": xgb_p99,
            "bb_mape": bb_mape,
            "bb_mspe": bb_mspe,
            "n_test": len(xgb_actual),
        })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(SCRIPT_DIR, "output", "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)

    # --- Print results ---
    print("=" * 90)
    print("EVALUATION RESULTS: XGBoost (per-experiment) vs Blackbox Baseline")
    print("=" * 90)

    print(f"\n{'Model':22s} {'Workload':12s} {'XGB MAPE':>10s} {'BB MAPE':>10s} {'Δ':>8s} {'XGB MSPE':>10s} {'XGB r':>8s}")
    print("-" * 90)

    results_sorted = results_df.sort_values("xgb_mape")
    for _, row in results_sorted.iterrows():
        delta = row["bb_mape"] - row["xgb_mape"]
        print(f"{row['model']:22s} {row['workload']:12s} "
              f"{row['xgb_mape']:>9.1f}% {row['bb_mape']:>9.1f}% {delta:>+7.1f} "
              f"{row['xgb_mspe']:>+9.1f}% {row['xgb_r']:>7.3f}")

    # --- Aggregate metrics ---
    avg_xgb = results_df["xgb_mape"].mean()
    avg_bb = results_df["bb_mape"].mean()
    median_xgb = results_df["xgb_mape"].median()

    exps_under_15 = (results_df["xgb_mape"] < 15).sum()
    exps_under_25 = (results_df["xgb_mape"] < 25).sum()
    exps_under_30 = (results_df["xgb_mape"] < 30).sum()

    print(f"\n--- Aggregate ---")
    print(f"  Average XGBoost MAPE:  {avg_xgb:.1f}%  (blackbox: {avg_bb:.1f}%)")
    print(f"  Median XGBoost MAPE:   {median_xgb:.1f}%")
    print(f"  Experiments < 15% MAPE: {exps_under_15}/16")
    print(f"  Experiments < 25% MAPE: {exps_under_25}/16")
    print(f"  Experiments < 30% MAPE: {exps_under_30}/16")
    print(f"  Improvement over BB:   {avg_bb - avg_xgb:+.1f} pp average MAPE")

    # --- Hypothesis Verdict ---
    print("\n" + "=" * 90)
    print("HYPOTHESIS VERDICT")
    print("=" * 90)

    if exps_under_15 >= 12:
        print(f"  HYPOTHESIS STRONGLY SUPPORTED: {exps_under_15}/16 experiments < 15% MAPE")
        print(f"  XGBoost with physics-informed features achieves the target.")
        print(f"  Proceed to h3-generalization.")
    elif exps_under_25 >= 12:
        print(f"  HYPOTHESIS PARTIALLY SUPPORTED: {exps_under_25}/16 experiments < 25% MAPE")
        print(f"  XGBoost improves significantly over blackbox but doesn't fully meet <15% target.")
        print(f"  Proceed to h3-generalization with caution.")
    elif avg_xgb < avg_bb:
        print(f"  HYPOTHESIS WEAKLY SUPPORTED: XGBoost ({avg_xgb:.1f}%) beats blackbox ({avg_bb:.1f}%)")
        print(f"  But only {exps_under_25}/16 experiments < 25% MAPE.")
    else:
        print(f"  HYPOTHESIS NOT SUPPORTED: XGBoost ({avg_xgb:.1f}%) does not beat blackbox ({avg_bb:.1f}%)")

    # Worst experiments
    worst = results_df.nlargest(3, "xgb_mape")
    print(f"\n  Worst 3 experiments:")
    for _, row in worst.iterrows():
        print(f"    {row['model']:22s} {row['workload']:12s}  MAPE={row['xgb_mape']:.1f}%  MSPE={row['xgb_mspe']:+.1f}%")


if __name__ == "__main__":
    main()
