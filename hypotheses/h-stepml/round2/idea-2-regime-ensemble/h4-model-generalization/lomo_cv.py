#!/usr/bin/env python3
"""Idea 2, H4: Leave-One-Model-Out (LOMO) cross-validation.

Trains regime Ridge on 3 of 4 model families, tests on the held-out model.
Reports per-step MAPE per fold and runs BLIS E2E on held-out experiments.

Usage:
    python3 lomo_cv.py [--output-dir output] [--threshold 256]
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Add shared infrastructure to path
SHARED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_loader import DEFAULT_DATA_ROOT
from evaluation import compute_mape, compute_mspe, compute_pearson_r, compute_p99_error
from lifecycle_kv_extractor import extract_all_experiments_kv_features
from splits import temporal_split

# Reuse h1's feature definitions and helpers
H1_DIR = Path(__file__).resolve().parent.parent / "h1-kv-regime-models"
sys.path.insert(0, str(H1_DIR))
from train_regime_ridge import (
    DECODE_FEATURES, MIXED_FEATURES, TARGET,
    REGIME_DECODE_ONLY, REGIME_MIXED_LIGHT, REGIME_MIXED_HEAVY,
    add_interaction_features, classify_regimes,
    train_regime_model, evaluate_model,
    export_regime_artifact, normalize_model_key,
    compute_step_overhead,
    _FEATURE_NAME_MAP,
)

# BLIS validation imports
from validate_blis import (
    build_workload_spec, extract_kv_blocks_from_vllm_log,
    load_exp_config, load_ground_truth_metrics, load_profile,
    parse_experiment_dir, run_blis,
)

# ---------------------------------------------------------------------------
# Model family grouping
# ---------------------------------------------------------------------------
# Group 70b and 70b-hf together for LOMO purposes
MODEL_FAMILIES = {
    "7b": ["llama-2-7b_tp1"],
    "70b": ["llama-2-70b_tp4", "llama-2-70b-hf_tp4"],
    "34b": ["codellama-34b_tp2"],
    "mixtral": ["mixtral-8x7b-v0-1_tp2"],
}


def get_family(model_key: str) -> str:
    """Map a model_key to its family name."""
    for family, keys in MODEL_FAMILIES.items():
        if model_key in keys:
            return family
    return model_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "output"))
    parser.add_argument("--threshold", type=int, default=256)
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
    df["regime"] = classify_regimes(df, threshold=args.threshold)
    print(f"  Loaded {len(df)} steps, {df['experiment_id'].nunique()} experiments")
    print(f"  Model families: {df['family'].value_counts().to_dict()}")

    # -----------------------------------------------------------------------
    # Step 2: Compute overhead (needed for BLIS artifacts)
    # -----------------------------------------------------------------------
    print("\nStep 2: Computing per-model overhead...", flush=True)
    model_overheads = compute_step_overhead(df)
    for mk, oh in sorted(model_overheads.items()):
        print(f"  {mk}: {oh:.0f} µs")

    # -----------------------------------------------------------------------
    # Step 3: LOMO cross-validation
    # -----------------------------------------------------------------------
    print(f"\nStep 3: LOMO 4-fold cross-validation (threshold={args.threshold})")
    print("=" * 70, flush=True)

    families = sorted(MODEL_FAMILIES.keys())
    lomo_results = []
    blis_results = []

    for fold_idx, held_out_family in enumerate(families):
        train_families = [f for f in families if f != held_out_family]
        held_out_keys = MODEL_FAMILIES[held_out_family]

        train_mask = df["family"].isin(train_families)
        test_mask = df["family"] == held_out_family
        train_df = df[train_mask]
        test_df = df[test_mask]

        print(f"\n  Fold {fold_idx+1}: Hold out {held_out_family.upper()} "
              f"(train={len(train_df)}, test={len(test_df)})")

        # Temporal split WITHIN training data for Ridge alpha tuning
        split = temporal_split(train_df)
        ridge_train = train_df.loc[split["train"]]

        # Train regime models on pooled training data
        regime_models = {}
        for regime_name in [REGIME_DECODE_ONLY, REGIME_MIXED_LIGHT, REGIME_MIXED_HEAVY]:
            features = DECODE_FEATURES if regime_name == REGIME_DECODE_ONLY else MIXED_FEATURES
            regime_train = ridge_train[ridge_train["regime"] == regime_name]

            if len(regime_train) < 50:
                print(f"    {regime_name}: SKIP ({len(regime_train)} samples)")
                continue

            ridge, info = train_regime_model(
                regime_train, features, use_log_target=False
            )
            regime_models[regime_name] = (ridge, features)

        # Evaluate on held-out model
        for regime_name in [REGIME_DECODE_ONLY, REGIME_MIXED_LIGHT, REGIME_MIXED_HEAVY]:
            if regime_name not in regime_models:
                continue

            ridge_model, features = regime_models[regime_name]
            regime_test = test_df[test_df["regime"] == regime_name]

            if len(regime_test) == 0:
                continue

            metrics = evaluate_model(
                ridge_model, regime_test, features, use_log_target=False
            )

            lomo_results.append({
                "fold": fold_idx + 1,
                "held_out": held_out_family,
                "regime": regime_name,
                "mape": metrics["mape"],
                "mspe": metrics["mspe"],
                "pearson_r": metrics["pearson_r"],
                "p99_error": metrics["p99_error"],
                "n_test": metrics["n_samples"],
                "mean_predicted": metrics["mean_predicted"],
                "mean_actual": metrics["mean_actual"],
            })

            print(f"    {regime_name}: MAPE={metrics['mape']:.1f}%, "
                  f"r={metrics['pearson_r']:.3f}, n={metrics['n_samples']}")

        # Overall MAPE for this fold (all regimes combined)
        all_preds = []
        all_actuals = []
        for regime_name, (ridge_model, features) in regime_models.items():
            regime_test = test_df[test_df["regime"] == regime_name]
            if len(regime_test) == 0:
                continue
            X = regime_test[features].fillna(0).values
            y = regime_test[TARGET].values
            pred = ridge_model.predict(X)
            all_preds.extend(pred.tolist())
            all_actuals.extend(y.tolist())

        if all_preds:
            overall_mape = compute_mape(np.array(all_preds), np.array(all_actuals))
            print(f"    OVERALL: MAPE={overall_mape:.1f}% ({len(all_preds)} steps)")
            lomo_results.append({
                "fold": fold_idx + 1,
                "held_out": held_out_family,
                "regime": "ALL",
                "mape": overall_mape,
                "n_test": len(all_preds),
            })

        # -------------------------------------------------------------------
        # Export LOMO artifact for BLIS E2E validation
        # -------------------------------------------------------------------
        # Use the average overhead of training models for the held-out model
        train_overheads = [
            v for k, v in model_overheads.items()
            if get_family(k) in train_families
        ]
        avg_overhead = float(np.mean(train_overheads)) if train_overheads else 0

        artifact_dir = os.path.join(output_dir, "lomo_artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        for held_key in held_out_keys:
            artifact_path = os.path.join(artifact_dir, f"{held_key}_regime.json")
            export_regime_artifact(
                regime_models, args.threshold,
                secondary_constants={"output_token": 0, "scheduling": 0, "preemption": 0},
                output_path=artifact_path,
                step_overhead_us=avg_overhead,
                log_target=False,
            )

        # -------------------------------------------------------------------
        # BLIS E2E on held-out experiments
        # -------------------------------------------------------------------
        print(f"\n    BLIS E2E on held-out {held_out_family} experiments:")
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
            if get_family(model_key) != held_out_family:
                continue

            artifact_path = os.path.join(artifact_dir, f"{model_key}_regime.json")
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
                print(f"      {dirname}: BLIS FAILED")
                continue

            gt_e2e = gt["e2e_mean_s"] * 1000
            gt_itl = gt["itl_mean_s"] * 1000
            e2e_err = abs(blis_metrics["e2e_mean_ms"] - gt_e2e) / gt_e2e * 100
            itl_err = abs(blis_metrics["itl_mean_ms"] - gt_itl) / gt_itl * 100 if gt_itl > 0 else 0

            blis_results.append({
                "fold": fold_idx + 1,
                "held_out": held_out_family,
                "experiment": dirname,
                "workload": meta["workload"],
                "gt_e2e_ms": gt_e2e,
                "blis_e2e_ms": blis_metrics["e2e_mean_ms"],
                "e2e_error_pct": e2e_err,
                "gt_itl_ms": gt_itl,
                "blis_itl_ms": blis_metrics["itl_mean_ms"],
                "itl_error_pct": itl_err,
            })

            print(f"      {dirname}: E2E={e2e_err:.1f}%, ITL={itl_err:.1f}%")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  LOMO CROSS-VALIDATION RESULTS")
    print("=" * 70)

    lomo_df = pd.DataFrame(lomo_results)
    lomo_df.to_csv(os.path.join(output_dir, "lomo_per_step_results.csv"), index=False)

    # Per-fold overall MAPE
    overall = lomo_df[lomo_df["regime"] == "ALL"]
    print(f"\n  Per-fold overall MAPE:")
    for _, row in overall.iterrows():
        print(f"    Fold {int(row['fold'])} (hold out {row['held_out']}): "
              f"MAPE={row['mape']:.1f}%, n={int(row['n_test'])}")

    avg_mape = overall["mape"].mean()
    max_mape = overall["mape"].max()
    worst_fold = overall.loc[overall["mape"].idxmax(), "held_out"]

    print(f"\n  Average LOMO MAPE: {avg_mape:.1f}%")
    print(f"  Worst fold: {worst_fold} ({max_mape:.1f}%)")

    # Per-regime MAPE
    print(f"\n  Per-regime MAPE (averaged across folds):")
    for regime in [REGIME_DECODE_ONLY, REGIME_MIXED_LIGHT, REGIME_MIXED_HEAVY]:
        regime_rows = lomo_df[lomo_df["regime"] == regime]
        if not regime_rows.empty:
            print(f"    {regime}: {regime_rows['mape'].mean():.1f}% "
                  f"(range: {regime_rows['mape'].min():.1f}-{regime_rows['mape'].max():.1f}%)")

    # BLIS E2E results
    if blis_results:
        blis_df = pd.DataFrame(blis_results)
        blis_df.to_csv(os.path.join(output_dir, "lomo_blis_e2e_results.csv"), index=False)

        print(f"\n  BLIS E2E on held-out models:")
        for family in families:
            fam_df = blis_df[blis_df["held_out"] == family]
            if not fam_df.empty:
                print(f"    {family}: mean E2E={fam_df['e2e_error_pct'].mean():.1f}%, "
                      f"mean ITL={fam_df['itl_error_pct'].mean():.1f}%")

        print(f"\n  Overall BLIS E2E: {blis_df['e2e_error_pct'].mean():.1f}%")
        print(f"  Overall BLIS ITL: {blis_df['itl_error_pct'].mean():.1f}%")

    # Hypothesis evaluation
    print(f"\n  {'='*70}")
    if avg_mape < 80:
        print(f"  HYPOTHESIS SUPPORTED: Avg LOMO MAPE = {avg_mape:.1f}% < 80%")
    else:
        print(f"  HYPOTHESIS REFUTED: Avg LOMO MAPE = {avg_mape:.1f}% >= 80%")

    if max_mape > 150:
        print(f"  WARNING: Fold '{worst_fold}' MAPE = {max_mape:.1f}% > 150% threshold")

    round1_baseline = 2559.7
    improvement = round1_baseline / avg_mape if avg_mape > 0 else float("inf")
    print(f"  Round 1 LOMO baseline: {round1_baseline:.1f}% → {improvement:.1f}x improvement")
    print(f"  {'='*70}")

    # Save summary
    summary = {
        "avg_lomo_mape": avg_mape,
        "max_lomo_mape": max_mape,
        "worst_fold": worst_fold,
        "round1_baseline_mape": round1_baseline,
        "improvement_factor": improvement,
        "hypothesis_status": "supported" if avg_mape < 80 else "refuted",
        "n_folds": len(overall),
    }
    with open(os.path.join(output_dir, "lomo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
