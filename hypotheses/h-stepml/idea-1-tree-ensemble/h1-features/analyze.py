"""Evaluate Ridge regression with 30 features against baselines.

Computes per-step MAPE, MSPE, Pearson r, and p99 error for:
  - Global Ridge (30 features)
  - Per-experiment Ridge (30 features)
  - Blackbox baseline (2 features)
  - Naive mean baseline

Success criteria (from HYPOTHESIS.md):
  Per-step MAPE < 25% on the held-out 20% test set, averaged across 16 experiments.
  Short-circuit: global MAPE > 30% → drop idea.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from evaluation import compute_mape, compute_mspe, compute_p99_error, compute_pearson_r
from baselines import BlackboxBaseline, NaiveMeanBaseline
from splits import temporal_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_COL = "step.duration_us"


def main():
    # Load predictions
    pred_path = os.path.join(SCRIPT_DIR, "output", "predictions.csv")
    preds_df = pd.read_csv(pred_path)

    # Load original features for baseline computation
    feat_path = os.path.join(SCRIPT_DIR, "output", "features.csv")
    feat_df = pd.read_csv(feat_path)

    # --- Global Ridge evaluation ---
    global_preds = preds_df[preds_df["model_type"] == "global_ridge_30f"]
    actual = global_preds[TARGET_COL].values
    predicted = global_preds["predicted_us"].values

    global_mape = compute_mape(predicted, actual)
    global_mspe = compute_mspe(predicted, actual)
    global_r = compute_pearson_r(predicted, actual)
    global_p99 = compute_p99_error(predicted, actual)

    # --- Baselines on same test set ---
    split = temporal_split(feat_df)
    train_df = feat_df.iloc[split["train"]]
    test_df = feat_df.iloc[split["test"]]

    bb = BlackboxBaseline().fit(train_df)
    bb_preds = bb.predict(test_df)
    bb_actual = test_df[TARGET_COL].values

    bb_mape = compute_mape(bb_preds, bb_actual)
    bb_mspe = compute_mspe(bb_preds, bb_actual)
    bb_r = compute_pearson_r(bb_preds, bb_actual)
    bb_p99 = compute_p99_error(bb_preds, bb_actual)

    nm = NaiveMeanBaseline().fit(train_df)
    nm_preds = nm.predict(test_df)
    nm_mape = compute_mape(nm_preds, bb_actual)

    # --- Per-experiment breakdown ---
    per_exp_results = []
    per_exp_preds = preds_df[preds_df["model_type"] == "per_experiment_ridge_30f"]

    for exp_id in sorted(feat_df["experiment_id"].unique()):
        exp_data = feat_df[feat_df["experiment_id"] == exp_id]
        model_name = exp_data["model"].iloc[0]
        workload = exp_data["workload"].iloc[0]

        # Per-experiment split
        exp_split = temporal_split(exp_data)
        exp_train = exp_data.loc[exp_split["train"]]
        exp_test = exp_data.loc[exp_split["test"]]

        if len(exp_test) < 5:
            continue

        # Per-experiment Ridge predictions
        exp_pred_rows = per_exp_preds[per_exp_preds["experiment_id"] == exp_id]
        if len(exp_pred_rows) > 0:
            pe_pred = exp_pred_rows["predicted_us"].values
            pe_actual = exp_pred_rows[TARGET_COL].values
            pe_mape = compute_mape(pe_pred, pe_actual)
            pe_r = compute_pearson_r(pe_pred, pe_actual)
        else:
            pe_mape = float("nan")
            pe_r = float("nan")

        # Global Ridge on this experiment's test set
        gl_rows = global_preds[global_preds["experiment_id"] == exp_id]
        if len(gl_rows) > 0:
            gl_pred = gl_rows["predicted_us"].values
            gl_actual = gl_rows[TARGET_COL].values
            gl_mape = compute_mape(gl_pred, gl_actual)
        else:
            gl_mape = float("nan")

        # Blackbox on this experiment
        bb_exp = BlackboxBaseline().fit(exp_train)
        bb_exp_pred = bb_exp.predict(exp_test)
        bb_exp_actual = exp_test[TARGET_COL].values
        bb_exp_mape = compute_mape(bb_exp_pred, bb_exp_actual)

        per_exp_results.append({
            "experiment_id": exp_id,
            "model": model_name,
            "workload": workload,
            "ridge_30f_per_exp_mape": pe_mape,
            "ridge_30f_per_exp_r": pe_r,
            "ridge_30f_global_mape": gl_mape,
            "blackbox_mape": bb_exp_mape,
        })

    per_exp_df = pd.DataFrame(per_exp_results)
    per_exp_path = os.path.join(SCRIPT_DIR, "output", "evaluation_results.csv")
    per_exp_df.to_csv(per_exp_path, index=False)

    # --- Print Results ---
    print("=" * 80)
    print("EVALUATION RESULTS: 30-Feature Ridge vs Baselines")
    print("=" * 80)

    print(f"\n--- Global Metrics (pooled test set, n={len(actual)}) ---")
    print(f"  {'Model':35s} {'MAPE':>8s} {'MSPE':>8s} {'r':>8s} {'p99':>8s}")
    print(f"  {'Ridge (30 features)':35s} {global_mape:>7.1f}% {global_mspe:>+7.1f}% {global_r:>7.3f} {global_p99:>7.1f}%")
    print(f"  {'Blackbox (2 features)':35s} {bb_mape:>7.1f}% {bb_mspe:>+7.1f}% {bb_r:>7.3f} {bb_p99:>7.1f}%")
    print(f"  {'Naive mean':35s} {nm_mape:>7.1f}%")

    improvement = bb_mape - global_mape
    print(f"\n  Improvement over blackbox: {improvement:+.1f} pp MAPE")

    print(f"\n--- Per-Experiment Breakdown ---")
    print(f"  {'Model':20s} {'Workload':12s} {'Ridge30f':>10s} {'Blackbox':>10s} {'Δ':>8s}")

    per_exp_df_sorted = per_exp_df.sort_values("ridge_30f_per_exp_mape")
    for _, row in per_exp_df_sorted.iterrows():
        ridge_m = row["ridge_30f_per_exp_mape"]
        bb_m = row["blackbox_mape"]
        delta = bb_m - ridge_m
        ridge_str = f"{ridge_m:.1f}%" if not np.isnan(ridge_m) else "N/A"
        print(f"  {row['model']:20s} {row['workload']:12s} {ridge_str:>10s} {bb_m:>9.1f}% {delta:>+7.1f}")

    avg_ridge = per_exp_df["ridge_30f_per_exp_mape"].mean()
    avg_bb = per_exp_df["blackbox_mape"].mean()

    print(f"\n  Average per-experiment MAPE:")
    print(f"    Ridge 30f: {avg_ridge:.1f}%")
    print(f"    Blackbox:  {avg_bb:.1f}%")

    # --- Hypothesis Verdict ---
    print("\n" + "=" * 80)
    print("HYPOTHESIS VERDICT")
    print("=" * 80)

    # Short-circuit check
    if global_mape > 30:
        print(f"  SHORT-CIRCUIT: Global MAPE ({global_mape:.1f}%) exceeds 30% threshold.")
        print(f"  The 30-feature set does not justify tree ensembles.")
        print(f"  Recommendation: Drop Idea 1 or revisit feature engineering.")
    elif avg_ridge < 25:
        print(f"  HYPOTHESIS SUPPORTED: Average per-experiment MAPE ({avg_ridge:.1f}%) < 25%")
        print(f"  The physics-informed feature set significantly outperforms blackbox.")
        print(f"  Proceed to h2-model (XGBoost).")
    else:
        print(f"  HYPOTHESIS WEAKLY SUPPORTED: Average MAPE ({avg_ridge:.1f}%) is 25-30%")
        print(f"  Features show improvement but not convincingly < 25%.")
        print(f"  Proceed to h2-model with caution.")


if __name__ == "__main__":
    main()
