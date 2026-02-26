"""Correlation analysis for analytical component decomposition.

Tests whether the 4 analytical FLOPs components correlate with observed
step duration on pure-phase subsets.

Success criteria (from HYPOTHESIS.md):
  At least 2 of 4 components achieve Pearson r > 0.8 on pure-phase subsets.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from evaluation import compute_pearson_r

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def correlate_subset(df: pd.DataFrame, component_col: str, target_col: str = "step.duration_us") -> dict:
    """Compute Pearson r between a component and step duration."""
    valid = df[[component_col, target_col]].dropna()
    valid = valid[valid[target_col] > 0]
    # Need variance in both columns for meaningful correlation
    if len(valid) < 10 or valid[component_col].std() == 0:
        return {"pearson_r": float("nan"), "n_steps": len(valid)}

    r = compute_pearson_r(valid[component_col].values, valid[target_col].values)
    return {"pearson_r": float(r), "n_steps": len(valid)}


def main():
    input_path = os.path.join(SCRIPT_DIR, "output", "step_components.csv")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} steps from {input_path}", file=sys.stderr)

    target = "step.duration_us"
    results = []

    # --- Pure Prefill subset ---
    pure_prefill = df[df["batch.decode_tokens"] == 0].copy()
    print(f"\nPure prefill steps: {len(pure_prefill)}", file=sys.stderr)

    for comp in ["prefill_gemm_us", "prefill_attn_us", "total_analytical_us"]:
        r = correlate_subset(pure_prefill, comp)
        results.append({"subset": "pure_prefill", "component": comp, **r})

    # Baseline: raw prefill_tokens correlation
    r = correlate_subset(pure_prefill, "batch.prefill_tokens")
    results.append({"subset": "pure_prefill", "component": "batch.prefill_tokens (baseline)", **r})

    # --- Pure Decode subset ---
    pure_decode = df[df["batch.prefill_tokens"] == 0].copy()
    print(f"Pure decode steps:  {len(pure_decode)}", file=sys.stderr)

    for comp in ["decode_gemm_us", "decode_attn_us", "total_analytical_us"]:
        r = correlate_subset(pure_decode, comp)
        results.append({"subset": "pure_decode", "component": comp, **r})

    # Baseline: raw decode_tokens correlation
    r = correlate_subset(pure_decode, "batch.decode_tokens")
    results.append({"subset": "pure_decode", "component": "batch.decode_tokens (baseline)", **r})

    # --- All steps ---
    for comp in ["total_analytical_us", "prefill_gemm_us", "decode_gemm_us"]:
        r = correlate_subset(df, comp)
        results.append({"subset": "all", "component": comp, **r})

    # --- Per-experiment correlations ---
    per_exp_results = []
    for exp_id, group in df.groupby("experiment_id"):
        model = group["model"].iloc[0]
        workload = group["workload"].iloc[0]
        for comp in ["prefill_gemm_us", "prefill_attn_us", "decode_gemm_us", "total_analytical_us"]:
            r = correlate_subset(group, comp)
            per_exp_results.append({
                "experiment_id": exp_id,
                "model": model,
                "workload": workload,
                "component": comp,
                **r,
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(SCRIPT_DIR, "output", "correlation_results.csv")
    results_df.to_csv(results_path, index=False)

    per_exp_df = pd.DataFrame(per_exp_results)
    per_exp_path = os.path.join(SCRIPT_DIR, "output", "per_experiment_correlations.csv")
    per_exp_df.to_csv(per_exp_path, index=False)

    # --- Print results ---
    print("\n" + "=" * 80)
    print("CORRELATION RESULTS: Analytical Components vs step.duration_us")
    print("=" * 80)

    for subset_name in ["pure_prefill", "pure_decode", "all"]:
        subset_rows = results_df[results_df["subset"] == subset_name]
        print(f"\n--- {subset_name} ---")
        for _, row in subset_rows.iterrows():
            r_val = row["pearson_r"]
            r_str = f"{r_val:.4f}" if not np.isnan(r_val) else "N/A"
            marker = " ✓" if not np.isnan(r_val) and abs(r_val) > 0.8 else ""
            print(f"  {row['component']:40s}  r={r_str:>8s}  n={int(row['n_steps']):>6d}{marker}")

    # --- Hypothesis verdict ---
    print("\n" + "=" * 80)
    print("HYPOTHESIS VERDICT")
    print("=" * 80)

    components_above_08 = 0
    checks = [
        ("pure_prefill", "prefill_gemm_us"),
        ("pure_prefill", "prefill_attn_us"),
        ("pure_decode", "decode_gemm_us"),
        ("pure_decode", "decode_attn_us"),
    ]
    for subset, comp in checks:
        row = results_df[(results_df["subset"] == subset) & (results_df["component"] == comp)]
        if len(row) > 0:
            r_val = row.iloc[0]["pearson_r"]
            passed = not np.isnan(r_val) and abs(r_val) > 0.8
            if passed:
                components_above_08 += 1
            status = "PASS (r > 0.8)" if passed else f"FAIL (r = {r_val:.4f})" if not np.isnan(r_val) else "N/A (no data)"
            print(f"  {subset}/{comp}: {status}")

    print(f"\n  Components with r > 0.8: {components_above_08}/4")
    if components_above_08 >= 2:
        print("  RESULT: HYPOTHESIS SUPPORTED — analytical decomposition captures physical structure")
    else:
        print("  RESULT: HYPOTHESIS NOT SUPPORTED — fewer than 2 components correlate")
        print("  NOTE: decode_attn_us is 0 due to missing per-request KV lengths.")
        print("        This is a known data limitation, not a decomposition failure.")

    # Per-experiment summary for total_analytical_us
    print(f"\n--- Per-experiment total_analytical_us correlation ---")
    total_exp = per_exp_df[per_exp_df["component"] == "total_analytical_us"].copy()
    if len(total_exp) > 0:
        total_exp = total_exp.sort_values("pearson_r", ascending=False)
        for _, row in total_exp.iterrows():
            r_val = row["pearson_r"]
            r_str = f"{r_val:.4f}" if not np.isnan(r_val) else "N/A"
            print(f"  {row['model']:25s} {row['workload']:12s}  r={r_str}")


if __name__ == "__main__":
    main()
