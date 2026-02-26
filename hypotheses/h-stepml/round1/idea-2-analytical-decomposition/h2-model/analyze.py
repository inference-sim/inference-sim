"""Evaluate learned correction factor models against baselines.

Compares 9-parameter global and 36-parameter per-model variants.
Reports per-step MAPE, MSPE, Pearson r, and parameter efficiency.

Success criteria (from HYPOTHESIS.md):
  Workload-level E2E mean error < 10% on >= 12/16 experiments AND |MSPE| < 5%.
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
    # Load predictions
    pred_path = os.path.join(SCRIPT_DIR, "output", "predictions.csv")
    preds_df = pd.read_csv(pred_path)

    # Load h1 components for baseline computation
    comp_path = os.path.join(H1_DIR, "output", "step_components.csv")
    comp_df = pd.read_csv(comp_path)

    # --- Per-experiment evaluation ---
    results = []

    for exp_id in sorted(comp_df["experiment_id"].unique()):
        exp_comp = comp_df[comp_df["experiment_id"] == exp_id]
        model_name = exp_comp["model"].iloc[0]
        workload = exp_comp["workload"].iloc[0]

        # Blackbox baseline
        exp_split = temporal_split(exp_comp)
        exp_train = exp_comp.loc[exp_split["train"]]
        exp_test = exp_comp.loc[exp_split["test"]]

        if len(exp_test) < 5:
            continue

        bb = BlackboxBaseline().fit(exp_train)
        bb_pred = bb.predict(exp_test)
        bb_actual = exp_test[TARGET_COL].values
        bb_mape = compute_mape(bb_pred, bb_actual)

        row = {
            "experiment_id": exp_id,
            "model": model_name,
            "workload": workload,
            "bb_mape": bb_mape,
            "n_test": len(exp_test),
        }

        for variant in ["9param_global", "36param_per_model"]:
            vpreds = preds_df[
                (preds_df["experiment_id"] == exp_id) & (preds_df["model_type"] == variant)
            ]
            if len(vpreds) == 0:
                row[f"{variant}_mape"] = float("nan")
                row[f"{variant}_mspe"] = float("nan")
                row[f"{variant}_r"] = float("nan")
                continue

            pred = vpreds["predicted_us"].values
            actual = vpreds[TARGET_COL].values
            row[f"{variant}_mape"] = compute_mape(pred, actual)
            row[f"{variant}_mspe"] = compute_mspe(pred, actual)
            row[f"{variant}_r"] = compute_pearson_r(pred, actual)

        results.append(row)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(SCRIPT_DIR, "output", "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)

    # --- Print results ---
    print("=" * 100)
    print("EVALUATION RESULTS: Learned Correction Factors vs Blackbox Baseline")
    print("=" * 100)

    print(f"\n{'Model':22s} {'Workload':12s} {'9-param':>10s} {'36-param':>10s} {'BB':>10s} "
          f"{'9p MSPE':>10s} {'36p MSPE':>10s} {'36p r':>8s}")
    print("-" * 100)

    for _, row in results_df.sort_values("36param_per_model_mape").iterrows():
        m9 = row["9param_global_mape"]
        m36 = row["36param_per_model_mape"]
        mspe9 = row["9param_global_mspe"]
        mspe36 = row["36param_per_model_mspe"]
        r36 = row["36param_per_model_r"]

        m9_s = f"{m9:.1f}%" if not np.isnan(m9) else "N/A"
        m36_s = f"{m36:.1f}%" if not np.isnan(m36) else "N/A"
        mspe9_s = f"{mspe9:+.1f}%" if not np.isnan(mspe9) else "N/A"
        mspe36_s = f"{mspe36:+.1f}%" if not np.isnan(mspe36) else "N/A"
        r36_s = f"{r36:.3f}" if not np.isnan(r36) else "N/A"

        print(f"{row['model']:22s} {row['workload']:12s} "
              f"{m9_s:>10s} {m36_s:>10s} {row['bb_mape']:>9.1f}% "
              f"{mspe9_s:>10s} {mspe36_s:>10s} {r36_s:>8s}")

    # --- Aggregate ---
    print(f"\n--- Aggregate ---")
    for variant, label in [("9param_global", "9-param global"), ("36param_per_model", "36-param per-model")]:
        col = f"{variant}_mape"
        mspe_col = f"{variant}_mspe"
        avg = results_df[col].mean()
        median = results_df[col].median()
        avg_mspe = results_df[mspe_col].mean()
        under_15 = (results_df[col] < 15).sum()
        under_25 = (results_df[col] < 25).sum()
        print(f"  {label}:  avg MAPE={avg:.1f}%  median={median:.1f}%  avg MSPE={avg_mspe:+.1f}%  "
              f"<15%: {under_15}/16  <25%: {under_25}/16")

    avg_bb = results_df["bb_mape"].mean()
    print(f"  Blackbox:         avg MAPE={avg_bb:.1f}%")

    # --- Parameter efficiency ---
    avg_9 = results_df["9param_global_mape"].mean()
    avg_36 = results_df["36param_per_model_mape"].mean()
    if avg_36 > 0:
        print(f"\n  Parameter efficiency: 9-param ({avg_9:.1f}%) vs 36-param ({avg_36:.1f}%)")
        print(f"    Improvement from 4x more parameters: {avg_9 - avg_36:.1f} pp MAPE")

    # --- Hypothesis Verdict ---
    print("\n" + "=" * 100)
    print("HYPOTHESIS VERDICT")
    print("=" * 100)

    best_variant = "36param_per_model"
    best_mape_col = f"{best_variant}_mape"
    best_mspe_col = f"{best_variant}_mspe"

    best_avg_mape = results_df[best_mape_col].mean()
    best_avg_mspe = abs(results_df[best_mspe_col].mean())
    exps_under_10_mape = (results_df[best_mape_col] < 10).sum()
    exps_under_15_mape = (results_df[best_mape_col] < 15).sum()
    mspe_ok = best_avg_mspe < 5

    print(f"  Best variant: {best_variant}")
    print(f"  Average MAPE: {best_avg_mape:.1f}%")
    print(f"  Average |MSPE|: {best_avg_mspe:.1f}% ({'< 5% ✓' if mspe_ok else '>= 5% ✗'})")
    print(f"  Experiments < 10% MAPE: {exps_under_10_mape}/16")
    print(f"  Experiments < 15% MAPE: {exps_under_15_mape}/16")

    if exps_under_15_mape >= 12 and mspe_ok:
        print(f"\n  HYPOTHESIS SUPPORTED: analytical decomposition + corrections work.")
        print(f"  Proceed to h3-generalization.")
    elif best_avg_mape < avg_bb:
        print(f"\n  HYPOTHESIS PARTIALLY SUPPORTED: improves over blackbox ({avg_bb:.1f}%)")
        print(f"  but doesn't fully meet targets. Investigate worst experiments.")
    else:
        print(f"\n  HYPOTHESIS NOT SUPPORTED.")


if __name__ == "__main__":
    main()
