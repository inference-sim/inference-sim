"""Train Ridge regression on 30 physics-informed features.

Trains both:
  1. Global model (all experiments pooled)
  2. Per-experiment models (16 separate regressions)

Uses temporal 60/20/20 split from shared infrastructure.
Saves predictions to output/predictions.csv.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from splits import temporal_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import feature column names from engineer_features
sys.path.insert(0, SCRIPT_DIR)
from engineer_features import FEATURE_COLS

TARGET_COL = "step.duration_us"


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, Ridge, StandardScaler]:
    """Train Ridge on train_df, predict on test_df. Returns (train_preds, test_preds, model, scaler)."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    y_train = train_df[TARGET_COL].values

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    return model.predict(X_train), model.predict(X_test), model, scaler


def main():
    input_path = os.path.join(SCRIPT_DIR, "output", "features.csv")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}", file=sys.stderr)

    # Replace any NaN/inf in features with 0
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Temporal split
    split = temporal_split(df)
    train_idx = split["train"]
    valid_idx = split["valid"]
    test_idx = split["test"]

    train_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[valid_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print(f"  Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}", file=sys.stderr)

    # --- Global model ---
    print("\nTraining global Ridge (30 features)...", file=sys.stderr)
    _, test_preds, model, scaler = train_and_predict(train_df, test_df, FEATURE_COLS)

    # Save global predictions
    global_results = test_df[["experiment_id", "model", "workload", "step.id", TARGET_COL]].copy()
    global_results["predicted_us"] = test_preds
    global_results["model_type"] = "global_ridge_30f"

    # Feature importance (standardized coefficients)
    coef_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_),
    }).sort_values("abs_coefficient", ascending=False)

    coef_path = os.path.join(SCRIPT_DIR, "output", "feature_importance.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"  Top 5 features by |coefficient|:", file=sys.stderr)
    for _, row in coef_df.head(5).iterrows():
        print(f"    {row['feature']:40s}  coef={row['coefficient']:+.4f}", file=sys.stderr)

    # --- Per-experiment models ---
    print("\nTraining per-experiment Ridge models...", file=sys.stderr)
    per_exp_parts = []
    per_exp_summaries = []

    for exp_id in sorted(df["experiment_id"].unique()):
        exp_df = df[df["experiment_id"] == exp_id]
        exp_split = temporal_split(exp_df)

        exp_train = exp_df.loc[exp_split["train"]]
        exp_test = exp_df.loc[exp_split["test"]]

        if len(exp_train) < 10 or len(exp_test) < 5:
            print(f"  Skipping {exp_id}: insufficient data", file=sys.stderr)
            continue

        _, exp_preds, exp_model, _ = train_and_predict(exp_train, exp_test, FEATURE_COLS)

        exp_results = exp_test[["experiment_id", "model", "workload", "step.id", TARGET_COL]].copy()
        exp_results["predicted_us"] = exp_preds
        exp_results["model_type"] = "per_experiment_ridge_30f"
        per_exp_parts.append(exp_results)

        per_exp_summaries.append({
            "experiment_id": exp_id,
            "model": exp_df["model"].iloc[0],
            "workload": exp_df["workload"].iloc[0],
            "n_train": len(exp_train),
            "n_test": len(exp_test),
        })

    # Combine all predictions
    all_preds = pd.concat([global_results] + per_exp_parts, ignore_index=True)
    pred_path = os.path.join(SCRIPT_DIR, "output", "predictions.csv")
    all_preds.to_csv(pred_path, index=False)
    print(f"\n  Saved {len(all_preds)} predictions to {pred_path}", file=sys.stderr)

    # Save per-experiment summary
    summary_df = pd.DataFrame(per_exp_summaries)
    summary_path = os.path.join(SCRIPT_DIR, "output", "per_experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
