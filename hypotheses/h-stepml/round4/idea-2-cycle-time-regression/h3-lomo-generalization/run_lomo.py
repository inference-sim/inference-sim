"""H3: Leave-One-Model-Out (LOMO) cross-validation for FairBatching regression.

4-fold CV: train on 3 models, predict held-out 4th.
Target: per-step MAPE < 80% per fold.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))

from data_loader import DEFAULT_DATA_ROOT
from evaluation import compute_mape, compute_pearson_r
from lifecycle_kv_extractor import extract_all_experiments_kv_features
from splits import leave_one_model_out


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    new_tokens = (
        df["batch.prefill_tokens"].fillna(0).values
        + df["batch.decode_tokens"].fillna(0).values
    ).astype(float)
    kv_sum = df["kv_sum"].fillna(0).values.astype(float) if "kv_sum" in df.columns else np.zeros(len(df))
    return np.column_stack([new_tokens, kv_sum])


def main():
    parser = argparse.ArgumentParser(description="H3: LOMO Cross-Validation")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("H3: Leave-One-Model-Out (LOMO) Cross-Validation")
    print("=" * 70)

    # Load data with KV features
    print("\nLoading data...")
    try:
        df = extract_all_experiments_kv_features(args.data_root)
    except Exception as e:
        print(f"KV extraction failed ({e}), falling back")
        from data_loader import load_all_experiments
        df = load_all_experiments(args.data_root)
        df["kv_sum"] = 0.0

    print(f"Loaded {len(df)} steps, models: {sorted(df['model'].unique())}")

    # Get LOMO folds
    folds = leave_one_model_out(df)
    print(f"LOMO folds: {len(folds)}")

    results = []
    target_col = "step.duration_us"

    for fold in folds:
        holdout = fold["holdout_model"]
        train_idx = fold["train"]
        test_idx = fold["test"]

        X_train = prepare_features(df.iloc[train_idx])
        y_train = df.iloc[train_idx][target_col].values.astype(float)
        X_test = prepare_features(df.iloc[test_idx])
        y_test = df.iloc[test_idx][target_col].values.astype(float)

        # Train global model on training models
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        pred_test = reg.predict(X_test)
        test_mape = compute_mape(pred_test, y_test)
        test_r = compute_pearson_r(pred_test, y_test) if len(pred_test) > 2 else 0

        result = {
            "holdout_model": holdout,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_mape": test_mape,
            "test_r": test_r,
            "intercept": float(reg.intercept_),
            "coeff_new_tokens": float(reg.coef_[0]),
            "coeff_kv_sum": float(reg.coef_[1]),
        }
        results.append(result)

        print(f"\n  Fold: holdout={holdout}")
        print(f"    Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        print(f"    Test MAPE: {test_mape:.1f}% (target <80%)")
        print(f"    Test Pearson r: {test_r:.3f}")
        print(f"    Coefficients: intercept={reg.intercept_:.1f}, "
              f"new_tokens={reg.coef_[0]:.4f}, kv_sum={reg.coef_[1]:.6f}")

    # Save results
    with open(os.path.join(args.output_dir, "lomo_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(results).to_csv(
        os.path.join(args.output_dir, "lomo_results.csv"), index=False
    )

    # Summary
    print("\n" + "=" * 70)
    print("LOMO SUMMARY")
    print("=" * 70)
    print(f"\n{'Holdout Model':<25} {'MAPE':>8} {'r':>7} {'Samples':>8} {'Status':>10}")
    print("-" * 65)
    for r in results:
        status = "PASS" if r["test_mape"] < 80 else "FAIL"
        print(f"{r['holdout_model']:<25} {r['test_mape']:>7.1f}% {r['test_r']:>7.3f} {r['test_samples']:>8} {status:>10}")

    mean_mape = np.mean([r["test_mape"] for r in results])
    max_mape = max(r["test_mape"] for r in results)
    print("-" * 65)
    print(f"{'MEAN':<25} {mean_mape:>7.1f}%")
    print(f"{'MAX':<25} {max_mape:>7.1f}%")
    print(f"\nVERDICT:")
    print(f"  Mean LOMO MAPE: {mean_mape:.1f}%")
    print(f"  Max fold MAPE: {max_mape:.1f}%")
    all_pass = all(r["test_mape"] < 80 for r in results)
    any_above_150 = any(r["test_mape"] > 150 for r in results)
    print(f"  All folds <80%: {'YES' if all_pass else 'NO'} (target)")
    print(f"  Any fold >150%: {'YES — REFUTED' if any_above_150 else 'NO'} (refutation)")


if __name__ == "__main__":
    main()
