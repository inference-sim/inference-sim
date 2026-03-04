#!/usr/bin/env python3
"""Train piecewise-linear StepTime with 2 regimes and KV features.

Idea 1, H1: Piecewise-linear model with decode-only vs mixed-batch regimes.
Each regime uses a specific feature set including ProgressIndex-derived KV
features. Trains per-model Ridge (with CV-tuned alpha), exports StepML
artifacts compatible with the Go evaluator.

Usage: python3 train_piecewise.py [--output-dir OUTPUT_DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Add shared infrastructure to path
SHARED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_loader import DEFAULT_DATA_ROOT, load_all_experiments
from evaluation import compute_mape, compute_mspe, compute_pearson_r, compute_p99_error
from lifecycle_kv_extractor import extract_all_experiments_kv_features
from splits import temporal_split


# ---------------------------------------------------------------------------
# Regime classification (2 regimes: decode-only vs mixed-batch)
# ---------------------------------------------------------------------------

REGIME_DECODE_ONLY = "decode_only"
REGIME_MIXED_BATCH = "mixed_batch"


def classify_regimes(df: pd.DataFrame) -> pd.Series:
    """Classify each step into decode-only or mixed-batch regime."""
    prefill = df["batch.prefill_tokens"].fillna(0).astype(int)
    regime = pd.Series(REGIME_DECODE_ONLY, index=df.index)
    regime[prefill > 0] = REGIME_MIXED_BATCH
    return regime


# ---------------------------------------------------------------------------
# Feature engineering (per HYPOTHESIS.md specification)
# ---------------------------------------------------------------------------

# Decode-only regime: 4 features
DECODE_FEATURES = [
    "batch.decode_tokens",
    "kv_mean",
    "kv_max",
    "kv_sum",
]

# Mixed-batch regime: 4 features
MIXED_FEATURES = [
    "batch.prefill_tokens",
    "batch.decode_tokens",
    "prefill_x_decode",
    "kv_sum",
]

TARGET = "step.duration_us"


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived interaction features."""
    df = df.copy()
    df["prefill_x_decode"] = (
        df["batch.prefill_tokens"].fillna(0) * df["batch.decode_tokens"].fillna(0)
    )
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_regime_model(
    train_df: pd.DataFrame, features: list[str], target: str = TARGET,
) -> tuple[Ridge, dict]:
    """Train Ridge with CV-tuned alpha on a single regime.

    Uses raw linear targets (no log transform) for BLIS E2E compatibility.
    The overhead floor in the Go evaluator handles small-batch accuracy.
    """
    X = train_df[features].fillna(0).values
    y = train_df[target].values

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    grid = GridSearchCV(
        Ridge(), {"alpha": alphas}, cv=5, scoring="neg_mean_absolute_error"
    )
    grid.fit(X, y)
    best = grid.best_estimator_

    return best, {"alpha": grid.best_params_["alpha"], "n_samples": len(X)}


def evaluate_model(
    model: Ridge, df: pd.DataFrame, features: list[str], target: str = TARGET,
) -> dict:
    """Evaluate a trained model on a DataFrame."""
    X = df[features].fillna(0).values
    y = df[target].values
    pred = model.predict(X)

    return {
        "mape": compute_mape(pred, y),
        "mspe": compute_mspe(pred, y),
        "pearson_r": compute_pearson_r(pred, y),
        "p99_error": compute_p99_error(pred, y),
        "n_samples": len(y),
        "mean_predicted": float(np.mean(pred)),
        "mean_actual": float(np.mean(y)),
    }


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

_FEATURE_NAME_MAP = {
    "batch.prefill_tokens": "prefill_tokens",
    "batch.decode_tokens": "decode_tokens",
    "kv_sum": "kv_sum",
    "kv_max": "kv_max",
    "kv_mean": "kv_mean",
    "prefill_x_decode": "prefill_x_decode",
}


def _ridge_to_linear_model(ridge: Ridge, features: list[str]) -> dict:
    """Convert a sklearn Ridge model to Go LinearModel JSON format."""
    coeffs = {}
    for feat, coeff in zip(features, ridge.coef_):
        go_name = _FEATURE_NAME_MAP.get(feat, feat)
        if abs(coeff) > 1e-10:
            coeffs[go_name] = float(coeff)
    return {
        "model_type": "linear",
        "intercept": float(ridge.intercept_),
        "feature_coefficients": coeffs,
    }


def export_piecewise_artifact(
    models: dict[str, tuple],
    secondary_constants: dict,
    output_path: str,
    step_overhead_us: float = 0.0,
):
    """Export a 2-regime piecewise-linear StepML artifact JSON."""
    regimes = []

    # Regime 1: decode-only (prefill_tokens == 0)
    if REGIME_DECODE_ONLY in models:
        ridge, features = models[REGIME_DECODE_ONLY]
        regimes.append({
            "name": "decode_only",
            "condition": {"feature": "prefill_tokens", "op": "==", "value": 0},
            "model": _ridge_to_linear_model(ridge, features),
        })

    # Regime 2: mixed-batch (prefill_tokens > 0) — fallback
    if REGIME_MIXED_BATCH in models:
        ridge, features = models[REGIME_MIXED_BATCH]
        regimes.append({
            "name": "mixed_batch",
            "condition": None,  # fallback
            "model": _ridge_to_linear_model(ridge, features),
        })

    artifact = {
        "version": 2,
        "step_time_regimes": regimes,
        "step_overhead_us": step_overhead_us,
        "output_token_processing_time_us": secondary_constants.get("output_token", 0.0),
        "scheduling_processing_time_us": secondary_constants.get("scheduling", 0.0),
        "preemption_processing_time_us": secondary_constants.get("preemption", 0.0),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)


# ---------------------------------------------------------------------------
# Step overhead computation (same as idea-2)
# ---------------------------------------------------------------------------


def normalize_model_key(model: str, tp: int) -> str:
    return f"{model}_tp{tp}"


def compute_step_overhead(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-model per-step overhead from ground truth ITL."""
    overheads: dict[str, list[float]] = {}

    for exp_id in sorted(df["experiment_id"].unique()):
        if "reasoning" in exp_id:
            continue

        sub = df[df["experiment_id"] == exp_id]
        model = sub.iloc[0]["model"]
        tp = int(sub.iloc[0]["tp"])
        model_key = normalize_model_key(model, tp)

        exp_dir = os.path.join(DEFAULT_DATA_ROOT, exp_id)
        gt_path = os.path.join(exp_dir, "results", "summary_lifecycle_metrics.json")
        if not os.path.isfile(gt_path):
            continue
        with open(gt_path) as f:
            gt = json.load(f)

        successes = gt.get("successes", {})
        itl_mean_s = successes.get("latency", {}).get(
            "inter_token_latency", {}
        ).get("mean", 0)
        if itl_mean_s <= 0:
            continue

        itl_mean_us = itl_mean_s * 1e6
        compute_mean = sub["step.duration_us"].mean()
        overhead = itl_mean_us - compute_mean

        if overhead > 0:
            if model_key not in overheads:
                overheads[model_key] = []
            overheads[model_key].append(overhead)

    return {mk: float(np.mean(vals)) for mk, vals in overheads.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "output"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load data with KV features
    # -----------------------------------------------------------------------
    print("Step 1: Loading data with KV features...", flush=True)
    df = extract_all_experiments_kv_features()
    print(f"  Loaded {len(df)} steps from {df['experiment_id'].nunique()} experiments")

    df = add_interaction_features(df)
    df["model_key"] = df.apply(
        lambda r: normalize_model_key(r["model"], r["tp"]), axis=1
    )

    # -----------------------------------------------------------------------
    # Step 1b: Compute per-model step overhead
    # -----------------------------------------------------------------------
    print("Step 1b: Computing per-model step overhead...", flush=True)
    model_overheads = compute_step_overhead(df)
    for mk, oh in sorted(model_overheads.items()):
        print(f"  {mk}: {oh:.0f} us")

    # -----------------------------------------------------------------------
    # Step 2: Classify regimes (2 regimes)
    # -----------------------------------------------------------------------
    print("Step 2: Classifying regimes (decode-only / mixed-batch)...", flush=True)
    df["regime"] = classify_regimes(df)
    regime_counts = df["regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(df)
        print(f"  {regime}: {count} steps ({pct:.1f}%)")

    # -----------------------------------------------------------------------
    # Step 3: Train per-model piecewise-linear Ridge models
    # -----------------------------------------------------------------------
    print("Step 3: Training per-model piecewise-linear Ridge models...", flush=True)
    results = []
    all_model_keys = sorted(df["model_key"].unique())

    for model_key in all_model_keys:
        model_mask = df["model_key"] == model_key
        model_df = df[model_mask]
        model_split = temporal_split(model_df)
        model_train = model_df.loc[model_split["train"]]
        model_valid = model_df.loc[model_split["valid"]]
        model_test = model_df.loc[model_split["test"]]

        regime_models = {}

        for regime_name in [REGIME_DECODE_ONLY, REGIME_MIXED_BATCH]:
            features = DECODE_FEATURES if regime_name == REGIME_DECODE_ONLY else MIXED_FEATURES
            regime_train = model_train[model_train["regime"] == regime_name]
            regime_test = model_test[model_test["regime"] == regime_name]

            if len(regime_train) < 50:
                print(f"  {model_key}/{regime_name}: SKIP (only {len(regime_train)} training samples)")
                continue

            ridge, train_info = train_regime_model(regime_train, features)
            regime_models[regime_name] = (ridge, features)

            if len(regime_test) > 0:
                test_metrics = evaluate_model(ridge, regime_test, features)
                results.append({
                    "model_key": model_key,
                    "regime": regime_name,
                    "n_train": train_info["n_samples"],
                    "alpha": train_info["alpha"],
                    "test_mape": test_metrics["mape"],
                    "test_mspe": test_metrics["mspe"],
                    "test_r": test_metrics["pearson_r"],
                    "test_p99": test_metrics["p99_error"],
                    "test_n": test_metrics["n_samples"],
                })
                print(
                    f"  {model_key}/{regime_name}: MAPE={test_metrics['mape']:.1f}%, "
                    f"r={test_metrics['pearson_r']:.3f}, "
                    f"n_train={train_info['n_samples']}, n_test={test_metrics['n_samples']}, "
                    f"alpha={train_info['alpha']}"
                )

        # Export piecewise artifact
        overhead = model_overheads.get(model_key, 0.0)
        artifact_path = os.path.join(output_dir, "artifacts", f"{model_key}_piecewise.json")
        export_piecewise_artifact(
            regime_models,
            secondary_constants={"output_token": 0, "scheduling": 0, "preemption": 0},
            output_path=artifact_path,
            step_overhead_us=overhead,
        )

    # -----------------------------------------------------------------------
    # Step 4: Blackbox baseline comparison
    # -----------------------------------------------------------------------
    print("\nStep 4: Blackbox baseline comparison...", flush=True)
    bb_features = ["batch.prefill_tokens", "batch.decode_tokens"]
    for model_key in all_model_keys:
        model_mask = df["model_key"] == model_key
        model_df = df[model_mask]
        model_split = temporal_split(model_df)
        model_train = model_df.loc[model_split["train"]]
        model_test = model_df.loc[model_split["test"]]

        bb_ridge, _ = train_regime_model(model_train, bb_features)
        bb_metrics = evaluate_model(bb_ridge, model_test, bb_features)
        results.append({
            "model_key": model_key,
            "regime": "blackbox_baseline",
            "n_train": len(model_train),
            "alpha": 0,
            "test_mape": bb_metrics["mape"],
            "test_mspe": bb_metrics["mspe"],
            "test_r": bb_metrics["pearson_r"],
            "test_p99": bb_metrics["p99_error"],
            "test_n": bb_metrics["n_samples"],
        })
        print(
            f"  {model_key}/blackbox: MAPE={bb_metrics['mape']:.1f}%, "
            f"r={bb_metrics['pearson_r']:.3f}"
        )

    # -----------------------------------------------------------------------
    # Step 5: KV feature ablation
    # -----------------------------------------------------------------------
    print("\nStep 5: KV feature ablation (piecewise WITHOUT KV features)...", flush=True)
    no_kv_decode = ["batch.decode_tokens"]
    no_kv_mixed = ["batch.prefill_tokens", "batch.decode_tokens", "prefill_x_decode"]
    for model_key in all_model_keys:
        model_mask = df["model_key"] == model_key
        model_df = df[model_mask]
        model_split = temporal_split(model_df)
        model_train = model_df.loc[model_split["train"]]
        model_test = model_df.loc[model_split["test"]]

        for regime_name in [REGIME_DECODE_ONLY, REGIME_MIXED_BATCH]:
            features = no_kv_decode if regime_name == REGIME_DECODE_ONLY else no_kv_mixed
            regime_train = model_train[model_train["regime"] == regime_name]
            regime_test = model_test[model_test["regime"] == regime_name]

            if len(regime_train) < 50 or len(regime_test) < 10:
                continue

            ridge, _ = train_regime_model(regime_train, features)
            metrics = evaluate_model(ridge, regime_test, features)
            results.append({
                "model_key": model_key,
                "regime": f"{regime_name}_no_kv",
                "n_train": len(regime_train),
                "alpha": 0,
                "test_mape": metrics["mape"],
                "test_mspe": metrics["mspe"],
                "test_r": metrics["pearson_r"],
                "test_p99": metrics["p99_error"],
                "test_n": metrics["n_samples"],
            })

    # -----------------------------------------------------------------------
    # Step 6: Save results
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "piecewise_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # -----------------------------------------------------------------------
    # Step 7: Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIECEWISE-LINEAR STEPTIME RESULTS (2 REGIMES)")
    print("=" * 70)

    # Per-model aggregate (piecewise regimes)
    regime_results = results_df[
        results_df["regime"].isin([REGIME_DECODE_ONLY, REGIME_MIXED_BATCH])
    ]
    if len(regime_results) > 0:
        print("\n  --- Per-Model Piecewise MAPE (test set) ---")
        for model_key in all_model_keys:
            model_regime = regime_results[regime_results["model_key"] == model_key]
            if len(model_regime) > 0:
                total_n = model_regime["test_n"].sum()
                weighted_mape = (model_regime["test_mape"] * model_regime["test_n"]).sum() / total_n
                print(f"  {model_key}: weighted MAPE = {weighted_mape:.1f}%")
                for _, row in model_regime.iterrows():
                    print(
                        f"    {row['regime']}: MAPE={row['test_mape']:.1f}%, "
                        f"r={row['test_r']:.3f}, n={row['test_n']}"
                    )

        overall_n = regime_results["test_n"].sum()
        overall_mape = (regime_results["test_mape"] * regime_results["test_n"]).sum() / overall_n
        overall_r = regime_results.apply(
            lambda r: r["test_r"] * r["test_n"], axis=1
        ).sum() / overall_n
        print(f"\n  Overall weighted MAPE (piecewise): {overall_mape:.1f}%")
        print(f"  Overall weighted Pearson r:        {overall_r:.3f}")

    # Blackbox baseline
    bb_results = results_df[results_df["regime"] == "blackbox_baseline"]
    if len(bb_results) > 0:
        bb_n = bb_results["test_n"].sum()
        bb_mape = (bb_results["test_mape"] * bb_results["test_n"]).sum() / bb_n
        print(f"  Overall weighted MAPE (blackbox):  {bb_mape:.1f}%")

    # KV ablation
    no_kv_results = results_df[results_df["regime"].str.endswith("_no_kv")]
    if len(no_kv_results) > 0:
        no_kv_n = no_kv_results["test_n"].sum()
        no_kv_mape = (no_kv_results["test_mape"] * no_kv_results["test_n"]).sum() / no_kv_n
        print(f"  Overall weighted MAPE (no KV):     {no_kv_mape:.1f}%")
        kv_improvement = no_kv_mape - overall_mape
        kv_pct = (kv_improvement / no_kv_mape) * 100 if no_kv_mape > 0 else 0
        print(f"  KV feature contribution: {kv_improvement:.1f} pp ({kv_pct:.1f}%)")

    # Hypothesis evaluation
    print(f"\n  {'='*70}")
    if overall_mape < 30:
        print(f"  HYPOTHESIS SUPPORTED: Overall MAPE = {overall_mape:.1f}% < 30%")
    else:
        print(f"  HYPOTHESIS REFUTED: Overall MAPE = {overall_mape:.1f}% >= 30%")

    if overall_r < 0.5:
        print(f"  WARNING: Pearson r = {overall_r:.3f} < 0.5 threshold")

    if len(no_kv_results) > 0 and kv_pct < 10:
        print(f"  WARNING: KV contribution = {kv_pct:.1f}% < 10% threshold")

    print(f"\n  Artifact directory: {os.path.join(output_dir, 'artifacts')}")
    print("=" * 70)

    # Save summary
    summary = {
        "overall_mape": overall_mape,
        "overall_pearson_r": overall_r,
        "blackbox_mape": bb_mape if len(bb_results) > 0 else None,
        "no_kv_mape": no_kv_mape if len(no_kv_results) > 0 else None,
        "kv_contribution_pp": kv_improvement if len(no_kv_results) > 0 else None,
        "kv_contribution_pct": kv_pct if len(no_kv_results) > 0 else None,
        "hypothesis_status": "supported" if overall_mape < 30 else "refuted",
        "model_overheads": model_overheads,
    }
    with open(os.path.join(output_dir, "piecewise_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
