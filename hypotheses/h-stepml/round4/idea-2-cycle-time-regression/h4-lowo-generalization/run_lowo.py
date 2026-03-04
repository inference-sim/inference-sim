"""H4: Leave-One-Workload-Out (LOWO) cross-validation for FairBatching regression.

3-fold CV: train on 2 workloads, predict held-out 3rd.
Target: per-step MAPE < 50% per fold.
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
from splits import leave_one_workload_out


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    new_tokens = (
        df["batch.prefill_tokens"].fillna(0).values
        + df["batch.decode_tokens"].fillna(0).values
    ).astype(float)
    kv_sum = df["kv_sum"].fillna(0).values.astype(float) if "kv_sum" in df.columns else np.zeros(len(df))
    return np.column_stack([new_tokens, kv_sum])


def main():
    parser = argparse.ArgumentParser(description="H4: LOWO Cross-Validation")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("H4: Leave-One-Workload-Out (LOWO) Cross-Validation")
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

    print(f"Loaded {len(df)} steps, workloads: {sorted(df['workload'].unique())}")

    # Get LOWO folds
    folds = leave_one_workload_out(df)
    print(f"LOWO folds: {len(folds)}")

    results = []
    target_col = "step.duration_us"

    for fold in folds:
        holdout = fold["holdout_workload"]
        train_idx = fold["train"]
        test_idx = fold["test"]

        X_train = prepare_features(df.iloc[train_idx])
        y_train = df.iloc[train_idx][target_col].values.astype(float)
        X_test = prepare_features(df.iloc[test_idx])
        y_test = df.iloc[test_idx][target_col].values.astype(float)

        # Per-model training within the LOWO fold
        per_model_results = []
        for model_name in sorted(df.iloc[train_idx]["model"].unique()):
            model_train_mask = df.iloc[train_idx]["model"].values == model_name
            model_test_mask = df.iloc[test_idx]["model"].values == model_name

            if np.sum(model_train_mask) < 10 or np.sum(model_test_mask) < 5:
                continue

            reg = LinearRegression()
            reg.fit(X_train[model_train_mask], y_train[model_train_mask])
            pred = reg.predict(X_test[model_test_mask])
            mape = compute_mape(pred, y_test[model_test_mask])
            per_model_results.append({"model": model_name, "mape": mape, "n": int(np.sum(model_test_mask))})

        # Also train global model
        reg_global = LinearRegression()
        reg_global.fit(X_train, y_train)
        pred_global = reg_global.predict(X_test)
        global_mape = compute_mape(pred_global, y_test)
        global_r = compute_pearson_r(pred_global, y_test) if len(pred_global) > 2 else 0

        # Use per-model weighted average if available
        if per_model_results:
            total_n = sum(r["n"] for r in per_model_results)
            weighted_mape = sum(r["mape"] * r["n"] for r in per_model_results) / total_n if total_n > 0 else global_mape
        else:
            weighted_mape = global_mape

        result = {
            "holdout_workload": holdout,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "global_mape": global_mape,
            "per_model_weighted_mape": weighted_mape,
            "global_r": global_r,
            "per_model_results": per_model_results,
        }
        results.append(result)

        print(f"\n  Fold: holdout={holdout}")
        print(f"    Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        print(f"    Global MAPE: {global_mape:.1f}%")
        print(f"    Per-model weighted MAPE: {weighted_mape:.1f}% (target <50%)")
        print(f"    Pearson r: {global_r:.3f}")
        for pmr in per_model_results:
            print(f"      {pmr['model']}: MAPE={pmr['mape']:.1f}% (n={pmr['n']})")

    # Save results
    with open(os.path.join(args.output_dir, "lowo_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    summary_rows = [{
        "holdout_workload": r["holdout_workload"],
        "train_samples": r["train_samples"],
        "test_samples": r["test_samples"],
        "global_mape": r["global_mape"],
        "per_model_weighted_mape": r["per_model_weighted_mape"],
        "global_r": r["global_r"],
    } for r in results]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "lowo_results.csv"), index=False
    )

    # Summary
    print("\n" + "=" * 70)
    print("LOWO SUMMARY")
    print("=" * 70)
    print(f"\n{'Holdout Workload':<20} {'Global MAPE':>12} {'Per-Model MAPE':>15} {'r':>7} {'Status':>10}")
    print("-" * 70)
    for r in results:
        status = "PASS" if r["per_model_weighted_mape"] < 50 else "FAIL"
        print(f"{r['holdout_workload']:<20} {r['global_mape']:>11.1f}% {r['per_model_weighted_mape']:>14.1f}% {r['global_r']:>7.3f} {status:>10}")

    mean_mape = np.mean([r["per_model_weighted_mape"] for r in results])
    max_mape = max(r["per_model_weighted_mape"] for r in results)
    print("-" * 70)
    print(f"{'MEAN':<20} {'':>12} {mean_mape:>14.1f}%")
    print(f"{'MAX':<20} {'':>12} {max_mape:>14.1f}%")
    print(f"\nVERDICT:")
    print(f"  Mean LOWO per-model MAPE: {mean_mape:.1f}%")
    print(f"  Max fold MAPE: {max_mape:.1f}%")
    all_pass = all(r["per_model_weighted_mape"] < 50 for r in results)
    any_above_100 = any(r["per_model_weighted_mape"] > 100 for r in results)
    print(f"  All folds <50%: {'YES' if all_pass else 'NO'} (target)")
    print(f"  Any fold >100%: {'YES — REFUTED' if any_above_100 else 'NO'} (refutation)")


if __name__ == "__main__":
    main()
