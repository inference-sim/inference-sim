#!/usr/bin/env python3
"""Idea 1, H5: Leave-One-Workload-Out (LOWO) cross-validation.

Per-model 2-regime piecewise-linear trained on 3/4 workloads, tested on
the held-out workload. Reports per-step MAPE per (model, workload) fold
and runs BLIS E2E.

Usage:
    python3 lowo_cv.py [--output-dir output]
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add shared infrastructure to path
SHARED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_loader import DEFAULT_DATA_ROOT
from evaluation import compute_mape, compute_pearson_r
from lifecycle_kv_extractor import extract_all_experiments_kv_features
from splits import temporal_split

# Reuse h1's definitions
H1_DIR = Path(__file__).resolve().parent.parent / "h1-piecewise-steptime"
sys.path.insert(0, str(H1_DIR))
from train_piecewise import (
    DECODE_FEATURES, MIXED_FEATURES, TARGET,
    REGIME_DECODE_ONLY, REGIME_MIXED_BATCH,
    add_interaction_features, classify_regimes,
    train_regime_model, evaluate_model,
    export_piecewise_artifact, normalize_model_key,
    compute_step_overhead,
)

from validate_blis import (
    build_workload_spec, extract_kv_blocks_from_vllm_log,
    load_exp_config, load_ground_truth_metrics, load_profile,
    parse_experiment_dir, run_blis,
)


# Model family grouping (same as h4)
MODEL_FAMILIES = {
    "7b": ["llama-2-7b_tp1"],
    "70b": ["llama-2-70b_tp4", "llama-2-70b-hf_tp4"],
    "34b": ["codellama-34b_tp2"],
    "mixtral": ["mixtral-8x7b-v0-1_tp2"],
}


def get_family(model_key: str) -> str:
    for family, keys in MODEL_FAMILIES.items():
        if model_key in keys:
            return family
    return model_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "output"))
    parser.add_argument("--binary", default=str(Path(__file__).resolve().parents[4] / "simulation_worker"))
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    print("Step 1: Loading data with KV features...", flush=True)
    df = extract_all_experiments_kv_features()
    df = add_interaction_features(df)
    df["model_key"] = df.apply(
        lambda r: normalize_model_key(r["model"], r["tp"]), axis=1
    )
    df["family"] = df["model_key"].map(get_family)
    df["regime"] = classify_regimes(df)
    print(f"  Loaded {len(df)} steps, {df['experiment_id'].nunique()} experiments")

    workloads = sorted(df["workload"].unique())
    families = sorted(MODEL_FAMILIES.keys())
    print(f"  Workloads: {workloads}")
    print(f"  Families: {families}")

    # -----------------------------------------------------------------------
    # Step 2: Compute overhead
    # -----------------------------------------------------------------------
    print("\nStep 2: Computing per-model overhead...", flush=True)
    model_overheads = compute_step_overhead(df)
    for mk, oh in sorted(model_overheads.items()):
        print(f"  {mk}: {oh:.0f} us")

    # -----------------------------------------------------------------------
    # Step 3: LOWO cross-validation (per model x per workload)
    # -----------------------------------------------------------------------
    print(f"\nStep 3: LOWO {len(workloads)}-fold CV per model")
    print("=" * 70, flush=True)

    lowo_results = []
    blis_results = []

    for family in families:
        family_keys = MODEL_FAMILIES[family]
        family_mask = df["family"] == family
        family_df = df[family_mask]

        print(f"\n  Model family: {family.upper()} ({len(family_df)} steps)")

        for held_out_wl in workloads:
            train_wls = [w for w in workloads if w != held_out_wl]
            train_mask = family_df["workload"].isin(train_wls)
            test_mask = family_df["workload"] == held_out_wl
            train_df = family_df[train_mask]
            test_df = family_df[test_mask]

            if len(train_df) < 100 or len(test_df) == 0:
                print(f"    Hold out {held_out_wl}: SKIP (train={len(train_df)}, test={len(test_df)})")
                continue

            print(f"    Hold out {held_out_wl}: train={len(train_df)}, test={len(test_df)}")

            # Temporal split within training data
            split = temporal_split(train_df)
            ridge_train = train_df.loc[split["train"]]

            # Train piecewise models
            regime_models = {}
            for regime_name in [REGIME_DECODE_ONLY, REGIME_MIXED_BATCH]:
                features = DECODE_FEATURES if regime_name == REGIME_DECODE_ONLY else MIXED_FEATURES
                regime_train = ridge_train[ridge_train["regime"] == regime_name]

                if len(regime_train) < 30:
                    continue

                ridge, _ = train_regime_model(regime_train, features)
                regime_models[regime_name] = (ridge, features)

            # Evaluate on held-out workload
            all_preds = []
            all_actuals = []
            for regime_name, (ridge_model, features) in regime_models.items():
                regime_test = test_df[test_df["regime"] == regime_name]
                if len(regime_test) == 0:
                    continue

                metrics = evaluate_model(ridge_model, regime_test, features)
                lowo_results.append({
                    "family": family,
                    "held_out_workload": held_out_wl,
                    "regime": regime_name,
                    "mape": metrics["mape"],
                    "pearson_r": metrics["pearson_r"],
                    "n_test": metrics["n_samples"],
                })

                X = regime_test[features].fillna(0).values
                y = regime_test[TARGET].values
                pred = ridge_model.predict(X)
                all_preds.extend(pred.tolist())
                all_actuals.extend(y.tolist())

            if all_preds:
                overall_mape = compute_mape(np.array(all_preds), np.array(all_actuals))
                lowo_results.append({
                    "family": family,
                    "held_out_workload": held_out_wl,
                    "regime": "ALL",
                    "mape": overall_mape,
                    "n_test": len(all_preds),
                })
                print(f"      MAPE={overall_mape:.1f}% ({len(all_preds)} steps)")

            # -----------------------------------------------------------
            # BLIS E2E on held-out workload
            # -----------------------------------------------------------
            artifact_dir = os.path.join(output_dir, "lowo_artifacts")
            os.makedirs(artifact_dir, exist_ok=True)

            family_overhead = np.mean([
                v for k, v in model_overheads.items()
                if k in family_keys
            ])

            for model_key in family_keys:
                artifact_path = os.path.join(
                    artifact_dir, f"{model_key}_{held_out_wl}_piecewise.json"
                )
                export_piecewise_artifact(
                    regime_models,
                    secondary_constants={"output_token": 0, "scheduling": 0, "preemption": 0},
                    output_path=artifact_path,
                    step_overhead_us=family_overhead,
                )

            # Find held-out experiments
            for dirname in sorted(os.listdir(args.data_root)):
                dirpath = os.path.join(args.data_root, dirname)
                if not os.path.isdir(dirpath):
                    continue
                summary_path = os.path.join(dirpath, "results", "summary_lifecycle_metrics.json")
                if not os.path.isfile(summary_path):
                    continue

                try:
                    meta = parse_experiment_dir(dirname)
                except ValueError:
                    continue

                model_key = f"{meta['model']}_tp{meta['tp']}"
                if get_family(model_key) != family or meta["workload"] != held_out_wl:
                    continue

                artifact_path = os.path.join(
                    artifact_dir, f"{model_key}_{held_out_wl}_piecewise.json"
                )
                if not os.path.isfile(artifact_path):
                    continue

                gt = load_ground_truth_metrics(dirpath)
                exp_config = load_exp_config(dirpath)
                total_kv_blocks = extract_kv_blocks_from_vllm_log(dirpath)
                if total_kv_blocks is None:
                    continue

                try:
                    profile = load_profile(dirpath)
                except Exception:
                    continue

                workload_spec = build_workload_spec(profile, gt)

                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    yaml.dump(workload_spec, f, default_flow_style=False)
                    spec_path = f.name

                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                    results_json_path = f.name

                try:
                    blis_metrics = run_blis(
                        binary=args.binary,
                        workload_spec_path=spec_path,
                        exp_config=exp_config,
                        total_kv_blocks=total_kv_blocks,
                        alpha_coeffs=[0, 0, 0],
                        beta_coeffs=[0, 0, 0],
                        results_path=results_json_path,
                        stepml_model_path=artifact_path,
                    )
                finally:
                    os.unlink(spec_path)
                    if os.path.exists(results_json_path):
                        os.unlink(results_json_path)

                if blis_metrics is None:
                    print(f"        BLIS {dirname}: FAILED")
                    continue

                gt_e2e = gt["e2e_mean_s"] * 1000
                gt_itl = gt["itl_mean_s"] * 1000
                e2e_err = abs(blis_metrics["e2e_mean_ms"] - gt_e2e) / gt_e2e * 100
                itl_err = abs(blis_metrics["itl_mean_ms"] - gt_itl) / gt_itl * 100 if gt_itl > 0 else 0

                blis_results.append({
                    "family": family,
                    "held_out_workload": held_out_wl,
                    "experiment": dirname,
                    "gt_e2e_ms": gt_e2e,
                    "blis_e2e_ms": blis_metrics["e2e_mean_ms"],
                    "e2e_error_pct": e2e_err,
                    "gt_itl_ms": gt_itl,
                    "blis_itl_ms": blis_metrics["itl_mean_ms"],
                    "itl_error_pct": itl_err,
                })

                print(f"        BLIS {dirname}: E2E={e2e_err:.1f}%, ITL={itl_err:.1f}%")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  LOWO CROSS-VALIDATION RESULTS (2-REGIME PIECEWISE)")
    print("=" * 70)

    lowo_df = pd.DataFrame(lowo_results)
    lowo_df.to_csv(os.path.join(output_dir, "lowo_per_step_results.csv"), index=False)

    overall = lowo_df[lowo_df["regime"] == "ALL"]

    print(f"\n  Per-model LOWO MAPE:")
    for family in families:
        fam = overall[overall["family"] == family]
        if not fam.empty:
            avg = fam["mape"].mean()
            worst = fam.loc[fam["mape"].idxmax()]
            print(f"    {family}: avg={avg:.1f}%, "
                  f"worst={worst['held_out_workload']} ({worst['mape']:.1f}%)")

    grand_avg = overall["mape"].mean()
    max_mape = overall["mape"].max()
    worst_row = overall.loc[overall["mape"].idxmax()]

    print(f"\n  Grand average LOWO MAPE: {grand_avg:.1f}%")
    print(f"  Worst fold: {worst_row['family']}/{worst_row['held_out_workload']} ({max_mape:.1f}%)")

    # Per held-out workload
    print(f"\n  Per held-out workload (avg across models):")
    for wl in workloads:
        wl_rows = overall[overall["held_out_workload"] == wl]
        if not wl_rows.empty:
            print(f"    {wl}: avg MAPE={wl_rows['mape'].mean():.1f}% "
                  f"(range: {wl_rows['mape'].min():.1f}-{wl_rows['mape'].max():.1f}%)")

    # BLIS E2E
    if blis_results:
        blis_df = pd.DataFrame(blis_results)
        blis_df.to_csv(os.path.join(output_dir, "lowo_blis_e2e_results.csv"), index=False)
        print(f"\n  BLIS E2E on held-out workloads:")
        for wl in workloads:
            wl_df = blis_df[blis_df["held_out_workload"] == wl]
            if not wl_df.empty:
                print(f"    {wl}: mean E2E={wl_df['e2e_error_pct'].mean():.1f}%, "
                      f"mean ITL={wl_df['itl_error_pct'].mean():.1f}%")

    # Hypothesis evaluation
    print(f"\n  {'='*70}")
    if grand_avg < 40:
        print(f"  HYPOTHESIS SUPPORTED: Grand avg LOWO MAPE = {grand_avg:.1f}% < 40%")
    else:
        print(f"  HYPOTHESIS REFUTED: Grand avg LOWO MAPE = {grand_avg:.1f}% >= 40%")

    if max_mape > 60:
        print(f"  WARNING: ({worst_row['family']}/{worst_row['held_out_workload']}) "
              f"MAPE = {max_mape:.1f}% > 60% single-fold threshold")

    # Check "general" workload specifically
    general_rows = overall[overall["held_out_workload"] == "general"]
    if not general_rows.empty:
        general_max = general_rows["mape"].max()
        if general_max > 50:
            worst_gen = general_rows.loc[general_rows["mape"].idxmax(), "family"]
            print(f"  WARNING: {worst_gen}/general MAPE = {general_max:.1f}% > 50%")

    round1_baseline = 109.7
    improvement = round1_baseline / grand_avg if grand_avg > 0 else float("inf")
    print(f"  Round 1 LOWO baseline: {round1_baseline:.1f}% -> {improvement:.1f}x improvement")
    print(f"  {'='*70}")

    # Save summary
    summary = {
        "grand_avg_lowo_mape": grand_avg,
        "max_lowo_mape": max_mape,
        "worst_fold": f"{worst_row['family']}/{worst_row['held_out_workload']}",
        "round1_baseline_mape": round1_baseline,
        "improvement_factor": improvement,
        "hypothesis_status": "supported" if grand_avg < 40 else "refuted",
        "n_folds": len(overall),
    }
    with open(os.path.join(output_dir, "lowo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
