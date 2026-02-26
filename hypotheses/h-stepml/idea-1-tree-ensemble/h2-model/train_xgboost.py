"""Train XGBoost on 30 physics-informed features from h1.

Strategy: Per-experiment XGBoost models (16 separate) since the global model
fails due to 3-order-of-magnitude step time range across models.

Hyperparameter search: small grid on validation split, final eval on test split.
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from splits import temporal_split
from evaluation import compute_mape

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H1_DIR = os.path.join(SCRIPT_DIR, "..", "h1-features")

# Reuse feature column list from h1
sys.path.insert(0, H1_DIR)
from engineer_features import FEATURE_COLS

TARGET_COL = "step.duration_us"

# Hyperparameter grid (compact for speed — ~12 combos per experiment)
PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "n_estimators": [100, 300],
    "learning_rate": [0.05, 0.1],
}


def grid_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[XGBRegressor, dict, float]:
    """Simple grid search over PARAM_GRID. Returns (best_model, best_params, valid_mape)."""
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_valid = valid_df[feature_cols].values
    y_valid = valid_df[TARGET_COL].values

    best_mape = float("inf")
    best_model = None
    best_params = {}

    for depth in PARAM_GRID["max_depth"]:
        for n_est in PARAM_GRID["n_estimators"]:
            for lr in PARAM_GRID["learning_rate"]:
                model = XGBRegressor(
                    max_depth=depth,
                    n_estimators=n_est,
                    learning_rate=lr,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False,
                )
                preds = model.predict(X_valid)
                mape = compute_mape(preds, y_valid)

                if mape < best_mape:
                    best_mape = mape
                    best_model = model
                    best_params = {
                        "max_depth": depth,
                        "n_estimators": n_est,
                        "learning_rate": lr,
                    }

    return best_model, best_params, best_mape


def main():
    # Load features from h1
    feat_path = os.path.join(H1_DIR, "output", "features.csv")
    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} rows from {feat_path}", file=sys.stderr)

    # Replace NaN/inf
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Per-experiment XGBoost ---
    print("\nTraining per-experiment XGBoost models...", file=sys.stderr)
    all_predictions = []
    experiment_summaries = []

    for exp_id in sorted(df["experiment_id"].unique()):
        exp_df = df[df["experiment_id"] == exp_id]
        model_name = exp_df["model"].iloc[0]
        workload = exp_df["workload"].iloc[0]

        exp_split = temporal_split(exp_df)
        train = exp_df.loc[exp_split["train"]]
        valid = exp_df.loc[exp_split["valid"]]
        test = exp_df.loc[exp_split["test"]]

        if len(train) < 20 or len(test) < 10:
            print(f"  Skipping {exp_id}: insufficient data", file=sys.stderr)
            continue

        print(f"  {model_name:25s} {workload:12s} (train={len(train)}, valid={len(valid)}, test={len(test)})...",
              end="", file=sys.stderr)

        best_model, best_params, valid_mape = grid_search(train, valid, FEATURE_COLS)

        # Final evaluation on test set
        test_preds = best_model.predict(test[FEATURE_COLS].values)
        test_mape = compute_mape(test_preds, test[TARGET_COL].values)

        print(f"  valid={valid_mape:.1f}%  test={test_mape:.1f}%  params={best_params}", file=sys.stderr)

        # Save predictions
        pred_df = test[["experiment_id", "model", "workload", "step.id", TARGET_COL]].copy()
        pred_df["predicted_us"] = test_preds
        pred_df["model_type"] = "xgboost_per_exp"
        all_predictions.append(pred_df)

        # Feature importance for this experiment
        importance = best_model.feature_importances_
        top_features = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])[:5]

        experiment_summaries.append({
            "experiment_id": exp_id,
            "model": model_name,
            "workload": workload,
            "n_train": len(train),
            "n_test": len(test),
            "valid_mape": valid_mape,
            "test_mape": test_mape,
            "best_max_depth": best_params["max_depth"],
            "best_n_estimators": best_params["n_estimators"],
            "best_learning_rate": best_params["learning_rate"],
            "top_feature_1": top_features[0][0] if len(top_features) > 0 else "",
            "top_feature_2": top_features[1][0] if len(top_features) > 1 else "",
            "top_feature_3": top_features[2][0] if len(top_features) > 2 else "",
        })

    # Save all predictions
    preds_df = pd.concat(all_predictions, ignore_index=True)
    pred_path = os.path.join(SCRIPT_DIR, "output", "predictions.csv")
    preds_df.to_csv(pred_path, index=False)
    print(f"\n  Saved {len(preds_df)} predictions to {pred_path}", file=sys.stderr)

    # Save experiment summaries
    summary_df = pd.DataFrame(experiment_summaries)
    summary_path = os.path.join(SCRIPT_DIR, "output", "experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Save best params as JSON for reproducibility
    params_path = os.path.join(SCRIPT_DIR, "output", "best_params.json")
    with open(params_path, "w") as f:
        json.dump(experiment_summaries, f, indent=2)


if __name__ == "__main__":
    main()
