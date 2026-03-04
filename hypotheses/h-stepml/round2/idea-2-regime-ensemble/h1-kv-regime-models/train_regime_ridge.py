#!/usr/bin/env python3
"""Train regime-specific Ridge regression with ProgressIndex-derived KV features.

Idea 2, H1: 3-regime ensemble (decode-only, mixed-light, mixed-heavy) per model.
Exports StepML artifacts compatible with the Go evaluator's regime dispatch.

Usage: python3 train_regime_ridge.py [--output-dir OUTPUT_DIR] [--threshold THRESHOLD]
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

from data_loader import load_all_experiments
from evaluation import compute_mape, compute_mspe, compute_pearson_r, compute_p99_error
from lifecycle_kv_extractor import extract_all_experiments_kv_features
from splits import temporal_split


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

REGIME_DECODE_ONLY = "decode_only"
REGIME_MIXED_LIGHT = "mixed_light"
REGIME_MIXED_HEAVY = "mixed_heavy"


def classify_regimes(df: pd.DataFrame, threshold: int = 256) -> pd.Series:
    """Classify each step into a batch-composition regime."""
    prefill = df["batch.prefill_tokens"].fillna(0).astype(int)
    regime = pd.Series(REGIME_DECODE_ONLY, index=df.index)
    regime[prefill > 0] = REGIME_MIXED_LIGHT
    regime[prefill >= threshold] = REGIME_MIXED_HEAVY
    return regime


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

DECODE_FEATURES = [
    "batch.decode_tokens",
    "batch.num_decode_reqs",
    "kv_sum",
    "kv_max",
    "kv_mean",
    "kv_std",
    "decode_x_kv_mean",
]

MIXED_FEATURES = [
    "batch.prefill_tokens",
    "batch.decode_tokens",
    "batch.num_prefill_reqs",
    "batch.num_decode_reqs",
    "kv_sum",
    "kv_max",
    "prefill_x_decode",
    "prefill_sq",
]

TARGET = "step.duration_us"


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived interaction features matching Go's extractBatchFeatures."""
    df = df.copy()
    df["prefill_x_decode"] = (
        df["batch.prefill_tokens"].fillna(0) * df["batch.decode_tokens"].fillna(0)
    )
    df["decode_x_kv_mean"] = (
        df["batch.num_decode_reqs"].fillna(0) * df["kv_mean"].fillna(0)
    )
    df["prefill_sq"] = df["batch.prefill_tokens"].fillna(0) ** 2
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_regime_model(
    train_df: pd.DataFrame, features: list[str], target: str = TARGET,
    use_log_target: bool = True,
) -> tuple[Ridge, dict]:
    """Train Ridge with CV-tuned alpha on a single regime.

    When use_log_target=True, trains on log(step_time) to prevent negative
    predictions. Predictions must be exponentiated before use.

    IMPORTANT: use_log_target=False is required when training on cycle_time
    (compute + overhead) because the overhead is additive but expm1 makes
    predictions multiplicative, causing exponential amplification of per-request
    coefficients at large batch sizes.
    """
    X = train_df[features].fillna(0).values
    y = train_df[target].values

    if use_log_target:
        y = np.log1p(np.maximum(y, 1))  # log(1 + max(y, 1))

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    grid = GridSearchCV(
        Ridge(), {"alpha": alphas}, cv=5, scoring="neg_mean_absolute_error"
    )
    grid.fit(X, y)
    best = grid.best_estimator_

    return best, {"alpha": grid.best_params_["alpha"], "n_samples": len(X),
                  "log_target": use_log_target}


def evaluate_model(
    model: Ridge, df: pd.DataFrame, features: list[str], target: str = TARGET,
    use_log_target: bool = True,
) -> dict:
    """Evaluate a trained model on a DataFrame."""
    X = df[features].fillna(0).values
    y = df[target].values
    pred = model.predict(X)

    if use_log_target:
        # Exponentiate predictions back to original scale
        pred = np.expm1(pred)  # exp(pred) - 1
        pred = np.maximum(pred, 1)  # ensure positive

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

# Mapping from Python feature column names to Go feature names
_FEATURE_NAME_MAP = {
    "batch.prefill_tokens": "prefill_tokens",
    "batch.decode_tokens": "decode_tokens",
    "batch.num_prefill_reqs": "num_prefill_reqs",
    "batch.num_decode_reqs": "num_decode_reqs",
    "batch.scheduled_tokens": "scheduled_tokens",
    "kv_sum": "kv_sum",
    "kv_max": "kv_max",
    "kv_mean": "kv_mean",
    "kv_std": "kv_std",
    "prefill_x_decode": "prefill_x_decode",
    "decode_x_kv_mean": "decode_x_kv_mean",
    "prefill_sq": "prefill_sq",
}


def export_regime_artifact(
    models: dict[str, tuple],
    threshold: int,
    secondary_constants: dict,
    output_path: str,
    step_overhead_us: float = 0.0,
    log_target: bool = True,
):
    """Export a regime-based StepML artifact JSON for the Go evaluator.

    models values can be (Ridge, features) or (Ridge, features, use_log).
    When use_log is present, it overrides the global log_target for that regime.
    """
    regimes = []

    def _unpack(entry):
        if len(entry) == 3:
            return entry[0], entry[1], entry[2]
        return entry[0], entry[1], log_target

    # Regime 1: decode-only (prefill_tokens == 0)
    if REGIME_DECODE_ONLY in models:
        ridge, features, use_log = _unpack(models[REGIME_DECODE_ONLY])
        regimes.append({
            "name": "decode_only",
            "condition": {"feature": "prefill_tokens", "op": "==", "value": 0},
            "model": _ridge_to_linear_model(ridge, features, log_target=use_log),
        })

    # Regime 2: mixed-light (0 < prefill_tokens < threshold)
    if REGIME_MIXED_LIGHT in models:
        ridge, features, use_log = _unpack(models[REGIME_MIXED_LIGHT])
        regimes.append({
            "name": "mixed_light",
            "condition": {"feature": "prefill_tokens", "op": "<", "value": threshold},
            "model": _ridge_to_linear_model(ridge, features, log_target=use_log),
        })

    # Regime 3: mixed-heavy (fallback — prefill_tokens >= threshold)
    if REGIME_MIXED_HEAVY in models:
        ridge, features, use_log = _unpack(models[REGIME_MIXED_HEAVY])
        regimes.append({
            "name": "mixed_heavy",
            "condition": None,  # fallback
            "model": _ridge_to_linear_model(ridge, features, log_target=use_log),
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


def _ridge_to_linear_model(ridge: Ridge, features: list[str],
                           log_target: bool = True) -> dict:
    """Convert a sklearn Ridge model to Go LinearModel JSON format."""
    coeffs = {}
    for feat, coeff in zip(features, ridge.coef_):
        go_name = _FEATURE_NAME_MAP.get(feat, feat)
        if abs(coeff) > 1e-10:  # skip near-zero coefficients
            coeffs[go_name] = float(coeff)
    model = {
        "model_type": "linear",
        "intercept": float(ridge.intercept_),
        "feature_coefficients": coeffs,
    }
    if log_target:
        model["output_transform"] = "expm1"
    return model


# ---------------------------------------------------------------------------
# Also export a single (non-regime) model for comparison
# ---------------------------------------------------------------------------

ALL_FEATURES = [
    "batch.prefill_tokens",
    "batch.decode_tokens",
    "batch.num_prefill_reqs",
    "batch.num_decode_reqs",
    "kv_sum",
    "kv_max",
    "kv_mean",
    "kv_std",
]


def export_single_artifact(
    ridge: Ridge,
    features: list[str],
    secondary_constants: dict,
    output_path: str,
    step_overhead_us: float = 0.0,
    log_target: bool = True,
):
    """Export a single (non-regime) StepML artifact for comparison."""
    artifact = {
        "version": 1,
        "step_time": _ridge_to_linear_model(ridge, features, log_target=log_target),
        "step_overhead_us": step_overhead_us,
        "output_token_processing_time_us": secondary_constants.get("output_token", 0.0),
        "scheduling_processing_time_us": secondary_constants.get("scheduling", 0.0),
        "preemption_processing_time_us": secondary_constants.get("preemption", 0.0),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def compute_step_overhead(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-model per-step overhead from ground truth ITL.

    The training data's step.duration_us only captures GPU compute time.
    The real step cycle includes scheduling, sync, and CPU-side overhead.

    We derive the overhead from ground truth inter-token latency (ITL),
    which equals the full step cycle time in continuous batching:
        overhead = GT_ITL - mean_compute_time_per_step

    Returns a dict mapping model_key → overhead in µs.
    """
    from data_loader import DEFAULT_DATA_ROOT

    overheads: dict[str, list[float]] = {}

    for exp_id in sorted(df["experiment_id"].unique()):
        # Skip reasoning workloads (different dynamics — overhead is
        # negative due to large-batch compute dominating)
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

        # Ground truth ITL = step cycle time in continuous batching
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


def normalize_model_key(model: str, tp: int) -> str:
    """Create a model+TP key matching the baseline convention."""
    return f"{model}_tp{tp}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "output"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=256,
        help="Mixed-light/heavy regime boundary (prefill tokens)",
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
        print(f"  {mk}: {oh:.0f} µs")

    # -----------------------------------------------------------------------
    # Step 2: Temporal split (per-model to avoid cross-model interference)
    # -----------------------------------------------------------------------
    print("Step 2: Per-model temporal split (60/20/20)...", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Classify regimes
    # -----------------------------------------------------------------------
    print(f"Step 3: Classifying regimes (threshold={args.threshold})...", flush=True)
    df["regime"] = classify_regimes(df, threshold=args.threshold)
    regime_counts = df["regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(df)
        print(f"  {regime}: {count} steps ({pct:.1f}%)")

    # -----------------------------------------------------------------------
    # Step 4: Train per-model regime-specific Ridge models
    # -----------------------------------------------------------------------
    # Step 3b: Augment target with per-model overhead (cycle time = compute + overhead)
    # -----------------------------------------------------------------------
    print("Step 3b: Augmenting target with per-model overhead...", flush=True)
    CYCLE_TARGET = "step.cycle_time_us"
    df[CYCLE_TARGET] = df[TARGET].astype(float)  # start with compute time (float for overhead addition)
    for model_key, overhead in model_overheads.items():
        mask = df["model_key"] == model_key
        df.loc[mask, CYCLE_TARGET] = df.loc[mask, TARGET] + overhead
        n = mask.sum()
        print(f"  {model_key}: +{overhead:.0f} µs overhead applied to {n} steps")

    # -----------------------------------------------------------------------
    print("Step 4: Training per-model regime-specific Ridge models...", flush=True)
    results = []
    all_model_keys = sorted(df["model_key"].unique())

    for model_key in all_model_keys:
        model_mask = df["model_key"] == model_key
        model_df = df[model_mask]
        # Per-model temporal split avoids cross-model interference
        # when experiments are added/removed from ground truth.
        model_split = temporal_split(model_df)
        model_train = model_df.loc[model_split["train"]]
        model_valid = model_df.loc[model_split["valid"]]
        model_test = model_df.loc[model_split["test"]]

        regime_models = {}

        for regime_name in [REGIME_DECODE_ONLY, REGIME_MIXED_LIGHT, REGIME_MIXED_HEAVY]:
            features = DECODE_FEATURES if regime_name == REGIME_DECODE_ONLY else MIXED_FEATURES
            regime_train = model_train[model_train["regime"] == regime_name]
            regime_test = model_test[model_test["regime"] == regime_name]

            if len(regime_train) < 50:
                print(f"  {model_key}/{regime_name}: SKIP (only {len(regime_train)} training samples)")
                continue

            # ALL regimes: raw linear (no log transform).
            # Log-space (expm1) is unsafe for BLIS E2E because KV features
            # and prefill tokens cause exponential blowup at large values.
            # The overhead floor handles small-batch accuracy; raw linear
            # provides correct batch-size scaling for large batches.
            ridge, train_info = train_regime_model(
                regime_train, features, use_log_target=False
            )
            regime_models[regime_name] = (ridge, features)

            if len(regime_test) > 0:
                test_metrics = evaluate_model(
                    ridge, regime_test, features, use_log_target=False
                )
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

        # Export regime artifact: raw linear compute + overhead floor/cap in Go
        overhead = model_overheads.get(model_key, 0.0)
        artifact_path = os.path.join(output_dir, "artifacts", f"{model_key}_regime.json")
        export_regime_artifact(
            regime_models, args.threshold,
            secondary_constants={"output_token": 0, "scheduling": 0, "preemption": 0},
            output_path=artifact_path,
            step_overhead_us=overhead,
            log_target=False,
        )

        # Also train and export a single (non-regime) model for comparison
        single_ridge, single_info = train_regime_model(
            model_train, ALL_FEATURES, use_log_target=False
        )
        single_test_metrics = evaluate_model(
            single_ridge, model_test, ALL_FEATURES, use_log_target=False
        )
        results.append({
            "model_key": model_key,
            "regime": "single_global",
            "n_train": single_info["n_samples"],
            "alpha": single_info["alpha"],
            "test_mape": single_test_metrics["mape"],
            "test_mspe": single_test_metrics["mspe"],
            "test_r": single_test_metrics["pearson_r"],
            "test_p99": single_test_metrics["p99_error"],
            "test_n": single_test_metrics["n_samples"],
        })
        print(
            f"  {model_key}/single_global: MAPE={single_test_metrics['mape']:.1f}%, "
            f"r={single_test_metrics['pearson_r']:.3f}"
        )

        single_path = os.path.join(output_dir, "artifacts", f"{model_key}_single.json")
        export_single_artifact(
            single_ridge, ALL_FEATURES,
            secondary_constants={"output_token": 0, "scheduling": 0, "preemption": 0},
            output_path=single_path,
            step_overhead_us=overhead,
            log_target=False,
        )

    # -----------------------------------------------------------------------
    # Step 5: Blackbox baseline comparison
    # -----------------------------------------------------------------------
    print("\nStep 5: Blackbox baseline comparison...", flush=True)
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
    # Step 6: KV feature ablation
    # -----------------------------------------------------------------------
    print("\nStep 6: KV feature ablation (regime models WITHOUT KV features)...", flush=True)
    no_kv_decode = ["batch.decode_tokens", "batch.num_decode_reqs"]
    no_kv_mixed = [
        "batch.prefill_tokens", "batch.decode_tokens",
        "batch.num_prefill_reqs", "batch.num_decode_reqs",
        "prefill_x_decode", "prefill_sq",
    ]
    for model_key in all_model_keys:
        model_mask = df["model_key"] == model_key
        model_df = df[model_mask]
        model_split = temporal_split(model_df)
        model_train = model_df.loc[model_split["train"]]
        model_test = model_df.loc[model_split["test"]]

        for regime_name in [REGIME_DECODE_ONLY, REGIME_MIXED_LIGHT, REGIME_MIXED_HEAVY]:
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
    # Step 7: Save results
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "regime_ridge_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # -----------------------------------------------------------------------
    # Step 8: Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  REGIME-SPECIFIC RIDGE RESULTS SUMMARY")
    print("=" * 70)

    # Per-model aggregate (regime-specific)
    regime_results = results_df[
        results_df["regime"].isin([REGIME_DECODE_ONLY, REGIME_MIXED_LIGHT, REGIME_MIXED_HEAVY])
    ]
    if len(regime_results) > 0:
        print("\n  --- Per-Model Regime MAPE (test set) ---")
        for model_key in all_model_keys:
            model_regime = regime_results[regime_results["model_key"] == model_key]
            if len(model_regime) > 0:
                # Weighted average MAPE by sample count
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
        print(f"\n  Overall weighted MAPE (regime models): {overall_mape:.1f}%")

    # Single-model comparison
    single_results = results_df[results_df["regime"] == "single_global"]
    if len(single_results) > 0:
        single_n = single_results["test_n"].sum()
        single_mape = (single_results["test_mape"] * single_results["test_n"]).sum() / single_n
        print(f"  Overall weighted MAPE (single model):  {single_mape:.1f}%")

    # Blackbox baseline
    bb_results = results_df[results_df["regime"] == "blackbox_baseline"]
    if len(bb_results) > 0:
        bb_n = bb_results["test_n"].sum()
        bb_mape = (bb_results["test_mape"] * bb_results["test_n"]).sum() / bb_n
        print(f"  Overall weighted MAPE (blackbox):       {bb_mape:.1f}%")

    # KV ablation
    no_kv_results = results_df[results_df["regime"].str.endswith("_no_kv")]
    if len(no_kv_results) > 0:
        no_kv_n = no_kv_results["test_n"].sum()
        no_kv_mape = (no_kv_results["test_mape"] * no_kv_results["test_n"]).sum() / no_kv_n
        print(f"  Overall weighted MAPE (no KV features): {no_kv_mape:.1f}%")
        kv_improvement = no_kv_mape - overall_mape
        print(f"  KV feature contribution: {kv_improvement:.1f} pp improvement")

    print("\n  Artifact directory:", os.path.join(output_dir, "artifacts"))
    print("=" * 70)


if __name__ == "__main__":
    main()
