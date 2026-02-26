"""Leave-one-model-out and leave-one-workload-out cross-validation.

For each fold:
  1. Train XGBoost on all data EXCEPT the held-out model/workload
  2. Evaluate on the held-out model/workload
  3. Use temporal split within training data for validation (bigger valid set)

Reports per-step MAPE for each held-out group, per-experiment breakdown,
and feature importance stability across folds.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from splits import leave_one_model_out, leave_one_workload_out
from evaluation import compute_mape, compute_mspe, compute_pearson_r, compute_p99_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H1_DIR = os.path.join(SCRIPT_DIR, "..", "h1-features")

sys.path.insert(0, H1_DIR)
from engineer_features import FEATURE_COLS

TARGET_COL = "step.duration_us"

# Best hyperparams from h2 (most experiments picked depth=4, n_est=100)
# Use a slightly larger model for generalization since training set is bigger
XGBOOST_PARAMS = {
    "max_depth": 6,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    valid_frac: float = 0.25,
) -> tuple[np.ndarray, XGBRegressor, float]:
    """Train XGBoost with validation split for early stopping.

    Uses last `valid_frac` of training data (temporally) as validation
    for early stopping. This gives a bigger validation set than 20%.
    """
    # Sort training data by experiment_id then step.id for temporal ordering
    train_sorted = train_df.sort_values(["experiment_id", "step.id"])
    n = len(train_sorted)
    n_valid = int(n * valid_frac)
    n_train = n - n_valid

    actual_train = train_sorted.iloc[:n_train]
    actual_valid = train_sorted.iloc[n_train:]

    X_train = actual_train[feature_cols].values
    y_train = actual_train[TARGET_COL].values
    X_valid = actual_valid[feature_cols].values
    y_valid = actual_valid[TARGET_COL].values
    X_test = test_df[feature_cols].values

    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    valid_preds = model.predict(X_valid)
    valid_mape = compute_mape(valid_preds, y_valid)

    test_preds = model.predict(X_test)
    return test_preds, model, valid_mape


def run_lomo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-model-out cross-validation."""
    folds = leave_one_model_out(df)
    all_predictions = []
    fold_summaries = []
    fold_importances = []

    for fold in folds:
        holdout = fold["holdout_model"]
        train_data = df.loc[fold["train"]].copy()
        test_data = df.loc[fold["test"]].copy()

        # Replace NaN/inf in features
        train_data[FEATURE_COLS] = train_data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_data[FEATURE_COLS] = test_data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"  LOMO holdout={holdout:25s} train={len(train_data):>6d} test={len(test_data):>6d}...",
              end="", file=sys.stderr)

        preds, model, valid_mape = train_and_evaluate(train_data, test_data, FEATURE_COLS)

        actual = test_data[TARGET_COL].values
        test_mape = compute_mape(preds, actual)
        test_mspe = compute_mspe(preds, actual)
        test_r = compute_pearson_r(preds, actual)

        print(f"  valid={valid_mape:.1f}%  test={test_mape:.1f}%", file=sys.stderr)

        # Save predictions
        pred_df = test_data[["experiment_id", "model", "workload", "step.id", TARGET_COL]].copy()
        pred_df["predicted_us"] = preds
        pred_df["cv_type"] = "lomo"
        pred_df["holdout"] = holdout
        all_predictions.append(pred_df)

        fold_summaries.append({
            "cv_type": "lomo",
            "holdout": holdout,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "valid_mape": valid_mape,
            "test_mape": test_mape,
            "test_mspe": test_mspe,
            "test_r": test_r,
        })

        # Feature importance for stability analysis
        importance = model.feature_importances_
        for feat, imp in zip(FEATURE_COLS, importance):
            fold_importances.append({
                "cv_type": "lomo",
                "holdout": holdout,
                "feature": feat,
                "importance": imp,
            })

        # Per-experiment breakdown within this fold
        for exp_id, exp_group in test_data.groupby("experiment_id"):
            exp_preds = preds[test_data.index.get_indexer(exp_group.index)]
            exp_actual = exp_group[TARGET_COL].values
            exp_mape = compute_mape(exp_preds, exp_actual)
            fold_summaries.append({
                "cv_type": "lomo_per_exp",
                "holdout": holdout,
                "experiment_id": exp_id,
                "model": exp_group["model"].iloc[0],
                "workload": exp_group["workload"].iloc[0],
                "n_test": len(exp_group),
                "test_mape": exp_mape,
                "test_mspe": compute_mspe(exp_preds, exp_actual),
                "test_r": compute_pearson_r(exp_preds, exp_actual) if len(exp_group) > 2 else float("nan"),
            })

    return pd.DataFrame(fold_summaries), pd.DataFrame(fold_importances)


def run_lowo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-workload-out cross-validation."""
    folds = leave_one_workload_out(df)
    all_predictions = []
    fold_summaries = []
    fold_importances = []

    for fold in folds:
        holdout = fold["holdout_workload"]
        train_data = df.loc[fold["train"]].copy()
        test_data = df.loc[fold["test"]].copy()

        train_data[FEATURE_COLS] = train_data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_data[FEATURE_COLS] = test_data[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"  LOWO holdout={holdout:25s} train={len(train_data):>6d} test={len(test_data):>6d}...",
              end="", file=sys.stderr)

        preds, model, valid_mape = train_and_evaluate(train_data, test_data, FEATURE_COLS)

        actual = test_data[TARGET_COL].values
        test_mape = compute_mape(preds, actual)
        test_mspe = compute_mspe(preds, actual)
        test_r = compute_pearson_r(preds, actual)

        print(f"  valid={valid_mape:.1f}%  test={test_mape:.1f}%", file=sys.stderr)

        pred_df = test_data[["experiment_id", "model", "workload", "step.id", TARGET_COL]].copy()
        pred_df["predicted_us"] = preds
        pred_df["cv_type"] = "lowo"
        pred_df["holdout"] = holdout
        all_predictions.append(pred_df)

        fold_summaries.append({
            "cv_type": "lowo",
            "holdout": holdout,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "valid_mape": valid_mape,
            "test_mape": test_mape,
            "test_mspe": test_mspe,
            "test_r": test_r,
        })

        for feat, imp in zip(FEATURE_COLS, model.feature_importances_):
            fold_importances.append({
                "cv_type": "lowo",
                "holdout": holdout,
                "feature": feat,
                "importance": imp,
            })

        # Per-experiment breakdown
        for exp_id, exp_group in test_data.groupby("experiment_id"):
            exp_preds = preds[test_data.index.get_indexer(exp_group.index)]
            exp_actual = exp_group[TARGET_COL].values
            exp_mape = compute_mape(exp_preds, exp_actual)
            fold_summaries.append({
                "cv_type": "lowo_per_exp",
                "holdout": holdout,
                "experiment_id": exp_id,
                "model": exp_group["model"].iloc[0],
                "workload": exp_group["workload"].iloc[0],
                "n_test": len(exp_group),
                "test_mape": exp_mape,
                "test_mspe": compute_mspe(exp_preds, exp_actual),
                "test_r": compute_pearson_r(exp_preds, exp_actual) if len(exp_group) > 2 else float("nan"),
            })

    return pd.DataFrame(fold_summaries), pd.DataFrame(fold_importances)


def main():
    feat_path = os.path.join(H1_DIR, "output", "features.csv")
    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} rows from {feat_path}", file=sys.stderr)

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- LOMO ---
    print("\n=== Leave-One-Model-Out ===", file=sys.stderr)
    lomo_summary, lomo_importance = run_lomo(df)

    # --- LOWO ---
    print("\n=== Leave-One-Workload-Out ===", file=sys.stderr)
    lowo_summary, lowo_importance = run_lowo(df)

    # Save results
    summary = pd.concat([lomo_summary, lowo_summary], ignore_index=True)
    summary_path = os.path.join(SCRIPT_DIR, "output", "cv_summary.csv")
    summary.to_csv(summary_path, index=False)

    importance = pd.concat([lomo_importance, lowo_importance], ignore_index=True)
    importance_path = os.path.join(SCRIPT_DIR, "output", "feature_importance.csv")
    importance.to_csv(importance_path, index=False)

    # Save predictions
    print(f"\n  Saved summaries to {summary_path}", file=sys.stderr)
    print(f"  Saved feature importance to {importance_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
