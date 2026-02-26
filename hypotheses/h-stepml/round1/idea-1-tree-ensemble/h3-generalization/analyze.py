"""Analyze LOMO and LOWO cross-validation results.

Reports:
  - Per-fold MAPE, MSPE, Pearson r
  - Per-experiment breakdown within each fold
  - Feature importance stability across folds
  - Hypothesis verdict against 20%/25% thresholds
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def print_fold_table(summary: pd.DataFrame, cv_type: str, label: str):
    """Print a formatted table for one CV type."""
    folds = summary[summary["cv_type"] == cv_type]
    if len(folds) == 0:
        return

    print(f"\n--- {label} ---")
    print(f"  {'Holdout':25s} {'n_train':>8s} {'n_test':>8s} {'Valid%':>8s} {'Test%':>8s} {'MSPE':>8s} {'r':>8s}")
    print("  " + "-" * 75)

    for _, row in folds.sort_values("test_mape").iterrows():
        holdout = row.get("holdout", "?")
        valid_s = f"{row['valid_mape']:.1f}%" if not np.isnan(row.get("valid_mape", float("nan"))) else "N/A"
        mspe_s = f"{row['test_mspe']:+.1f}%" if not np.isnan(row.get("test_mspe", float("nan"))) else "N/A"
        r_s = f"{row['test_r']:.3f}" if not np.isnan(row.get("test_r", float("nan"))) else "N/A"
        print(f"  {holdout:25s} {int(row.get('n_train', 0)):>8d} {int(row['n_test']):>8d} "
              f"{valid_s:>8s} {row['test_mape']:>7.1f}% {mspe_s:>8s} {r_s:>8s}")

    avg_mape = folds["test_mape"].mean()
    max_mape = folds["test_mape"].max()
    worst = folds.loc[folds["test_mape"].idxmax(), "holdout"]
    print(f"\n  Average MAPE: {avg_mape:.1f}%  |  Max: {max_mape:.1f}% ({worst})")


def print_per_experiment_breakdown(summary: pd.DataFrame, cv_type: str, label: str):
    """Print per-experiment breakdown for a CV type."""
    per_exp = summary[summary["cv_type"] == cv_type]
    if len(per_exp) == 0:
        return

    print(f"\n--- {label}: Per-Experiment Breakdown ---")
    print(f"  {'Holdout':15s} {'Model':22s} {'Workload':12s} {'MAPE':>8s} {'MSPE':>8s}")
    print("  " + "-" * 70)

    for _, row in per_exp.sort_values("test_mape").iterrows():
        holdout = str(row.get("holdout", "?"))[:15]
        model = str(row.get("model", "?"))
        workload = str(row.get("workload", "?"))
        mspe_s = f"{row['test_mspe']:+.1f}%" if not np.isnan(row.get("test_mspe", float("nan"))) else "N/A"
        print(f"  {holdout:15s} {model:22s} {workload:12s} {row['test_mape']:>7.1f}% {mspe_s:>8s}")


def analyze_feature_stability(importance: pd.DataFrame, cv_type: str, label: str):
    """Analyze feature importance stability across folds."""
    cv_imp = importance[importance["cv_type"] == cv_type]
    if len(cv_imp) == 0:
        return

    # Rank features within each fold
    ranked = cv_imp.copy()
    ranked["rank"] = ranked.groupby("holdout")["importance"].rank(ascending=False)

    # Average rank and rank stability (std of rank across folds)
    stability = ranked.groupby("feature").agg(
        avg_importance=("importance", "mean"),
        avg_rank=("rank", "mean"),
        rank_std=("rank", "std"),
    ).sort_values("avg_rank")

    print(f"\n--- {label}: Feature Importance Stability (top 10) ---")
    print(f"  {'Feature':40s} {'Avg Imp':>10s} {'Avg Rank':>10s} {'Rank StdDev':>12s}")
    print("  " + "-" * 75)

    for feat, row in stability.head(10).iterrows():
        std_s = f"{row['rank_std']:.1f}" if not np.isnan(row["rank_std"]) else "N/A"
        print(f"  {feat:40s} {row['avg_importance']:>10.4f} {row['avg_rank']:>10.1f} {std_s:>12s}")

    # Overall stability metric: average rank std across top 10 features
    top10_std = stability.head(10)["rank_std"].mean()
    print(f"\n  Top-10 avg rank StdDev: {top10_std:.1f} (lower = more stable)")


def main():
    summary_path = os.path.join(SCRIPT_DIR, "output", "cv_summary.csv")
    summary = pd.read_csv(summary_path)

    importance_path = os.path.join(SCRIPT_DIR, "output", "feature_importance.csv")
    importance = pd.read_csv(importance_path)

    print("=" * 90)
    print("GENERALIZATION RESULTS: Leave-One-Out Cross-Validation")
    print("=" * 90)

    # LOMO results
    print_fold_table(summary, "lomo", "Leave-One-Model-Out (LOMO)")
    print_per_experiment_breakdown(summary, "lomo_per_exp", "LOMO")

    # LOWO results
    print_fold_table(summary, "lowo", "Leave-One-Workload-Out (LOWO)")
    print_per_experiment_breakdown(summary, "lowo_per_exp", "LOWO")

    # Feature stability
    analyze_feature_stability(importance, "lomo", "LOMO")
    analyze_feature_stability(importance, "lowo", "LOWO")

    # --- Hypothesis Verdict ---
    print("\n" + "=" * 90)
    print("HYPOTHESIS VERDICT")
    print("=" * 90)

    lomo_folds = summary[summary["cv_type"] == "lomo"]
    lowo_folds = summary[summary["cv_type"] == "lowo"]

    lomo_max = lomo_folds["test_mape"].max()
    lomo_avg = lomo_folds["test_mape"].mean()
    lowo_max = lowo_folds["test_mape"].max()
    lowo_avg = lowo_folds["test_mape"].mean()

    lomo_all_under_25 = (lomo_folds["test_mape"] < 25).all()
    lomo_all_under_20 = (lomo_folds["test_mape"] < 20).all()
    lowo_all_under_25 = (lowo_folds["test_mape"] < 25).all()
    lowo_all_under_20 = (lowo_folds["test_mape"] < 20).all()

    print(f"\n  LOMO: avg={lomo_avg:.1f}%  max={lomo_max:.1f}%  all<20%: {'✓' if lomo_all_under_20 else '✗'}  all<25%: {'✓' if lomo_all_under_25 else '✗'}")
    print(f"  LOWO: avg={lowo_avg:.1f}%  max={lowo_max:.1f}%  all<20%: {'✓' if lowo_all_under_20 else '✗'}  all<25%: {'✓' if lowo_all_under_25 else '✗'}")

    # MoE-specific check
    mixtral_fold = lomo_folds[lomo_folds["holdout"].str.contains("mixtral", case=False, na=False)]
    if len(mixtral_fold) > 0:
        mixtral_mape = mixtral_fold.iloc[0]["test_mape"]
        mixtral_ok = mixtral_mape < 25
        print(f"  MoE transfer (Mixtral held out): {mixtral_mape:.1f}% {'(< 25% ✓)' if mixtral_ok else '(>= 25% ✗)'}")

    if lomo_all_under_20 and lowo_all_under_20:
        print(f"\n  HYPOTHESIS STRONGLY SUPPORTED: all folds < 20% MAPE")
        print(f"  XGBoost generalizes well across both models and workloads.")
    elif lomo_all_under_25 and lowo_all_under_25:
        print(f"\n  HYPOTHESIS SUPPORTED: all folds < 25% MAPE (relaxed threshold)")
        print(f"  XGBoost generalizes adequately. Per-experiment tuning may improve further.")
    elif lomo_avg < 30 and lowo_avg < 30:
        print(f"\n  HYPOTHESIS PARTIALLY SUPPORTED: average MAPE < 30% but some folds exceed 25%")
        print(f"  Generalization is moderate. Consider per-model or per-workload models.")
    else:
        print(f"\n  HYPOTHESIS NOT SUPPORTED: generalization is weak.")
        print(f"  The model has memorized configuration-specific patterns.")

    # Recommendation
    print(f"\n  RECOMMENDATION:")
    if lomo_avg < lowo_avg:
        print(f"    Model generalization ({lomo_avg:.1f}%) is better than workload generalization ({lowo_avg:.1f}%).")
        print(f"    → Per-workload models may be more effective than per-model models.")
    else:
        print(f"    Workload generalization ({lowo_avg:.1f}%) is better than model generalization ({lomo_avg:.1f}%).")
        print(f"    → Per-model models may be more effective than per-workload models.")


if __name__ == "__main__":
    main()
