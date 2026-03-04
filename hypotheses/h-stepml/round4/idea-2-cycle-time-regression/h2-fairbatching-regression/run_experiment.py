"""H2: FairBatching Cycle-Time Regression → BLIS E2E Validation.

Trains per-model FairBatching-style OLS regression:
    cycle_time = a + b*new_tokens + c*kv_sum

Uses step.duration_us as training target (GPU compute) with overhead floor
calibrated from H1's ITI-derived cycle times where available, falling back
to R2's empirically-derived per-model overhead values.

Exports StepML JSON artifacts and validates via BLIS trace replay.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))

from data_loader import DEFAULT_DATA_ROOT, load_all_experiments
from evaluation import compute_mape, compute_pearson_r
from lifecycle_kv_extractor import extract_all_experiments_kv_features
from splits import temporal_split

# Per-model overhead floors from R2 + H1 ITI-derived cycle times
# H1 found median cycle times: 70B ~9,670us, 34B ~7,149us
# R2 established: 7B=3,897us, 34B=6,673us, 70B=8,029-8,203us, Mixtral=9,125us
# We use H1-confirmed values where available, R2 values otherwise
OVERHEAD_FLOORS = {
    "llama-2-7b": 3897,
    "codellama-34b": 7149,   # H1-confirmed (was 6,673 in R2)
    "llama-2-70b": 9670,     # H1-confirmed (was 8,029-8,203 in R2)
    "llama-2-70b-hf": 9670,  # Same model, different HF variant
    "mixtral-8x7b-v0-1": 9125,
}

# FairBatching features: new_tokens (prefill+decode) and kv_sum
# This is the 3-coefficient formulation from Patel et al., 2025


def get_model_key(model_name: str) -> str:
    """Normalize model name for overhead floor lookup."""
    model_lower = model_name.lower()
    for key in OVERHEAD_FLOORS:
        if key in model_lower:
            return key
    return model_name


def prepare_features(df: pd.DataFrame, use_kv: bool = True) -> np.ndarray:
    """Extract FairBatching features from step data."""
    new_tokens = (
        df["batch.prefill_tokens"].fillna(0).values
        + df["batch.decode_tokens"].fillna(0).values
    ).astype(float)

    if use_kv and "kv_sum" in df.columns:
        kv_sum = df["kv_sum"].fillna(0).values.astype(float)
    else:
        kv_sum = np.zeros(len(df))

    return np.column_stack([new_tokens, kv_sum])


def train_per_model_regression(
    df: pd.DataFrame, use_kv: bool = True
) -> dict:
    """Train per-model FairBatching regression.

    Returns dict mapping model_key -> {model, intercept, coefficients, metrics}.
    """
    splits = temporal_split(df)
    train_idx = splits["train"]
    valid_idx = splits["valid"]
    test_idx = splits["test"]

    models = {}

    for model_name in sorted(df["model"].unique()):
        model_key = get_model_key(model_name)
        model_mask = df["model"] == model_name

        # Training data
        train_mask = np.isin(df.index, train_idx) & model_mask.values
        valid_mask = np.isin(df.index, valid_idx) & model_mask.values
        test_mask = np.isin(df.index, test_idx) & model_mask.values

        train_df = df[train_mask]
        valid_df = df[valid_mask]
        test_df = df[test_mask]

        if len(train_df) < 10:
            print(f"  SKIP {model_name}: only {len(train_df)} training samples")
            continue

        X_train = prepare_features(train_df, use_kv)
        y_train = train_df["step.duration_us"].values.astype(float)

        X_valid = prepare_features(valid_df, use_kv)
        y_valid = valid_df["step.duration_us"].values.astype(float)

        X_test = prepare_features(test_df, use_kv)
        y_test = test_df["step.duration_us"].values.astype(float)

        # Train OLS
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        # Predictions
        pred_train = reg.predict(X_train)
        pred_valid = reg.predict(X_valid)
        pred_test = reg.predict(X_test)

        # Metrics
        train_mape = compute_mape(pred_train, y_train)
        valid_mape = compute_mape(pred_valid, y_valid)
        test_mape = compute_mape(pred_test, y_test)

        train_r = compute_pearson_r(pred_train, y_train) if len(pred_train) > 2 else 0
        valid_r = compute_pearson_r(pred_valid, y_valid) if len(pred_valid) > 2 else 0
        test_r = compute_pearson_r(pred_test, y_test) if len(pred_test) > 2 else 0

        overhead = OVERHEAD_FLOORS.get(model_key, 5000)

        # Also train decode-only and mixed-batch separately
        decode_mask_train = train_df["batch.prefill_tokens"].fillna(0).values == 0
        mixed_mask_train = ~decode_mask_train

        decode_model = None
        mixed_model = None

        if np.sum(decode_mask_train) > 10:
            decode_model = LinearRegression()
            decode_model.fit(X_train[decode_mask_train], y_train[decode_mask_train])

        if np.sum(mixed_mask_train) > 10:
            mixed_model = LinearRegression()
            mixed_model.fit(X_train[mixed_mask_train], y_train[mixed_mask_train])

        models[model_key] = {
            "model_name": model_name,
            "model_key": model_key,
            "regression": reg,
            "decode_regression": decode_model,
            "mixed_regression": mixed_model,
            "overhead_us": overhead,
            "intercept": float(reg.intercept_),
            "coeff_new_tokens": float(reg.coef_[0]),
            "coeff_kv_sum": float(reg.coef_[1]) if use_kv else 0.0,
            "train_samples": len(train_df),
            "valid_samples": len(valid_df),
            "test_samples": len(test_df),
            "train_mape": train_mape,
            "valid_mape": valid_mape,
            "test_mape": test_mape,
            "train_r": train_r,
            "valid_r": valid_r,
            "test_r": test_r,
        }

        print(f"\n  {model_name} (key={model_key}):")
        print(f"    Intercept: {reg.intercept_:.1f}")
        print(f"    Coeff new_tokens: {reg.coef_[0]:.4f}")
        if use_kv:
            print(f"    Coeff kv_sum: {reg.coef_[1]:.6f}")
        print(f"    Overhead floor: {overhead} us")
        print(f"    Train MAPE: {train_mape:.1f}%, Valid MAPE: {valid_mape:.1f}%, Test MAPE: {test_mape:.1f}%")
        print(f"    Train r: {train_r:.3f}, Valid r: {valid_r:.3f}, Test r: {test_r:.3f}")
        print(f"    Samples: {len(train_df)} train, {len(valid_df)} valid, {len(test_df)} test")

    return models


def export_stepml_artifact(
    models: dict, output_path: str, use_regimes: bool = True
) -> str:
    """Export trained models to a StepML JSON artifact.

    Creates a per-model artifact since each model has different coefficients
    and overhead values. Returns the path pattern.
    """
    artifacts = {}

    for model_key, model_info in models.items():
        if use_regimes and model_info.get("decode_regression") is not None:
            # Regime-based artifact
            regimes = []

            # Decode-only regime
            decode_reg = model_info["decode_regression"]
            regimes.append({
                "name": "decode_only",
                "condition": {"feature": "prefill_tokens", "op": "==", "value": 0},
                "model": {
                    "model_type": "linear",
                    "intercept": float(decode_reg.intercept_),
                    "feature_coefficients": {
                        "decode_tokens": float(decode_reg.coef_[0]),
                        "kv_sum": float(decode_reg.coef_[1]),
                    },
                },
            })

            # Mixed-batch regime (fallback)
            mixed_reg = model_info.get("mixed_regression")
            if mixed_reg is not None:
                regimes.append({
                    "name": "mixed",
                    "condition": None,
                    "model": {
                        "model_type": "linear",
                        "intercept": float(mixed_reg.intercept_),
                        "feature_coefficients": {
                            "scheduled_tokens": float(mixed_reg.coef_[0]),
                            "kv_sum": float(mixed_reg.coef_[1]),
                        },
                    },
                })
            else:
                # Fallback to global model
                reg = model_info["regression"]
                regimes.append({
                    "name": "mixed",
                    "condition": None,
                    "model": {
                        "model_type": "linear",
                        "intercept": float(reg.intercept_),
                        "feature_coefficients": {
                            "scheduled_tokens": float(reg.coef_[0]),
                            "kv_sum": float(reg.coef_[1]),
                        },
                    },
                })

            artifact = {
                "version": 2,
                "step_time_regimes": regimes,
                "step_overhead_us": float(model_info["overhead_us"]),
                "output_token_processing_time_us": 0,
                "scheduling_processing_time_us": 0,
                "preemption_processing_time_us": 0,
            }
        else:
            # Single-model artifact
            reg = model_info["regression"]
            artifact = {
                "version": 1,
                "step_time": {
                    "model_type": "linear",
                    "intercept": float(reg.intercept_),
                    "feature_coefficients": {
                        "scheduled_tokens": float(reg.coef_[0]),
                        "kv_sum": float(reg.coef_[1]),
                    },
                },
                "step_overhead_us": float(model_info["overhead_us"]),
                "output_token_processing_time_us": 0,
                "scheduling_processing_time_us": 0,
                "preemption_processing_time_us": 0,
            }

        artifact_path = os.path.join(
            output_path, f"stepml_{model_key}.json"
        )
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)
        artifacts[model_key] = artifact_path
        print(f"  Exported: {artifact_path}")

    return artifacts


def run_blis_validation(artifacts: dict, output_dir: str, data_root: str) -> list:
    """Run BLIS validation for each experiment using the corresponding model artifact."""
    import subprocess
    import tempfile
    import yaml

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))
    from validate_blis import (
        build_workload_spec,
        compute_error,
        extract_cpu_kv_blocks_from_vllm_log,
        extract_kv_blocks_from_vllm_log,
        load_exp_config,
        load_ground_truth_metrics,
        load_profile,
        parse_blis_stdout,
        parse_experiment_dir,
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    binary = os.path.join(repo_root, "simulation_worker")

    # Build binary if needed
    if not os.path.isfile(binary):
        print("Building simulation_worker...")
        build_result = subprocess.run(
            ["go", "build", "-o", binary, "main.go"],
            cwd=repo_root, capture_output=True, text=True,
        )
        if build_result.returncode != 0:
            print(f"Build failed: {build_result.stderr}")
            return []

    results = []

    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        summary_path = os.path.join(dirpath, "results", "summary_lifecycle_metrics.json")
        if not os.path.isfile(summary_path):
            continue

        try:
            meta = parse_experiment_dir(dirname)
        except ValueError:
            continue

        model_key = get_model_key(meta["model"])
        if model_key not in artifacts:
            print(f"  SKIP {dirname}: no artifact for model {model_key}")
            continue

        artifact_path = artifacts[model_key]

        gt = load_ground_truth_metrics(dirpath)
        exp_config = load_exp_config(dirpath)
        total_kv_blocks = extract_kv_blocks_from_vllm_log(dirpath)
        if total_kv_blocks is None:
            continue
        cpu_kv_blocks = extract_cpu_kv_blocks_from_vllm_log(dirpath)

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

        model_name = exp_config.get("model", "unknown")
        tp = exp_config.get("tensor_parallelism", 1)
        max_model_len = exp_config.get("max_model_len", 4096)
        max_num_seqs = exp_config.get("max_num_seqs", 128)
        max_num_batched_tokens = exp_config.get("max_num_batched_tokens", 2048)

        cmd = [
            binary, "run",
            "--model", model_name,
            "--workload-spec", spec_path,
            "--tp", str(tp),
            "--max-model-len", str(max_model_len),
            "--max-num-running-reqs", str(max_num_seqs),
            "--max-num-scheduled-tokens", str(max_num_batched_tokens),
            "--total-kv-blocks", str(total_kv_blocks),
            "--block-size-in-tokens", "16",
            "--alpha-coeffs=0,0,0",
            "--beta-coeffs=0,0,0",
            "--stepml-model", artifact_path,
            "--results-path", results_json_path,
            "--log", "error",
        ]

        if cpu_kv_blocks > 0:
            cmd.extend(["--kv-cpu-blocks", str(cpu_kv_blocks)])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                cwd=os.path.dirname(binary),
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {dirname}")
            continue
        except FileNotFoundError:
            print(f"  Binary not found: {binary}")
            return results
        finally:
            os.unlink(spec_path)
            if os.path.exists(results_json_path):
                os.unlink(results_json_path)

        if result.returncode != 0:
            print(f"  BLIS FAILED ({dirname}): {result.stderr[:300]}")
            results.append({
                "experiment": dirname, "model": meta["model"],
                "workload": meta["workload"], "tp": meta["tp"],
                "status": "blis_failed",
            })
            continue

        blis_metrics = parse_blis_stdout(result.stdout)
        if blis_metrics is None:
            results.append({
                "experiment": dirname, "model": meta["model"],
                "workload": meta["workload"], "tp": meta["tp"],
                "status": "parse_failed",
            })
            continue

        e2e_error = compute_error(blis_metrics["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
        ttft_error = compute_error(blis_metrics["ttft_mean_ms"], gt["ttft_mean_s"] * 1000)
        itl_error = compute_error(blis_metrics["itl_mean_ms"], gt["itl_mean_s"] * 1000)

        print(f"  {dirname}: E2E={e2e_error*100:.1f}% TTFT={ttft_error*100:.1f}% ITL={itl_error*100:.1f}%")

        results.append({
            "experiment": dirname,
            "model": meta["model"],
            "workload": meta["workload"],
            "tp": meta["tp"],
            "status": "ok",
            "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
            "gt_ttft_ms": gt["ttft_mean_s"] * 1000,
            "gt_itl_ms": gt["itl_mean_s"] * 1000,
            "blis_e2e_ms": blis_metrics["e2e_mean_ms"],
            "blis_ttft_ms": blis_metrics["ttft_mean_ms"],
            "blis_itl_ms": blis_metrics["itl_mean_ms"],
            "e2e_error": e2e_error,
            "ttft_error": ttft_error,
            "itl_error": itl_error,
            "completed": blis_metrics.get("completed_requests", 0),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="H2: FairBatching Cycle-Time Regression")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-blis", action="store_true", help="Skip BLIS validation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("H2: FairBatching Cycle-Time Regression")
    print("=" * 70)

    # Step 1: Load data with KV features
    print("\nStep 1: Loading data with KV features...")
    try:
        df = extract_all_experiments_kv_features(args.data_root)
    except Exception as e:
        print(f"KV extraction failed ({e}), falling back to basic data")
        df = load_all_experiments(args.data_root)
        df["kv_sum"] = 0.0

    print(f"  Loaded {len(df)} steps across {df['experiment_id'].nunique()} experiments")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Workloads: {sorted(df['workload'].unique())}")

    # Step 2: Train per-model regression
    print("\nStep 2: Training per-model FairBatching regression...")
    models = train_per_model_regression(df, use_kv=True)

    # Save training results
    training_summary = {}
    for key, info in models.items():
        training_summary[key] = {
            k: v for k, v in info.items()
            if k not in ("regression", "decode_regression", "mixed_regression")
        }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(training_summary, f, indent=2)

    # Step 3: Export StepML artifacts
    print("\nStep 3: Exporting StepML artifacts...")
    artifacts = export_stepml_artifact(models, args.output_dir)

    # Step 4: BLIS validation
    if not args.skip_blis:
        print("\nStep 4: Running BLIS validation...")
        blis_results = run_blis_validation(artifacts, args.output_dir, args.data_root)

        # Save BLIS results
        with open(os.path.join(args.output_dir, "blis_results.json"), "w") as f:
            json.dump(blis_results, f, indent=2)

        blis_df = pd.DataFrame(blis_results)
        blis_df.to_csv(os.path.join(args.output_dir, "blis_results.csv"), index=False)

        # Summary
        ok = [r for r in blis_results if r.get("status") == "ok"]
        if ok:
            e2e_errors = [r["e2e_error"] for r in ok]
            ttft_errors = [r["ttft_error"] for r in ok]
            itl_errors = [r["itl_error"] for r in ok]

            print("\n" + "=" * 70)
            print("BLIS E2E VALIDATION RESULTS")
            print("=" * 70)
            print(f"\n{'Experiment':<50} {'E2E%':>7} {'TTFT%':>7} {'ITL%':>7}")
            print("-" * 75)
            for r in ok:
                print(f"{r['experiment']:<50} {r['e2e_error']*100:>6.1f}% {r['ttft_error']*100:>6.1f}% {r['itl_error']*100:>6.1f}%")
            print("-" * 75)
            mean_e2e = np.mean(e2e_errors) * 100
            mean_ttft = np.mean(ttft_errors) * 100
            mean_itl = np.mean(itl_errors) * 100
            print(f"{'MEAN':<50} {mean_e2e:>6.1f}% {mean_ttft:>6.1f}% {mean_itl:>6.1f}%")

            print(f"\nVERDICT:")
            print(f"  Mean E2E error: {mean_e2e:.1f}% (target <25%)")
            print(f"  Mean ITL error: {mean_itl:.1f}% (target <20%)")
            e2e_below_25 = sum(1 for e in e2e_errors if e < 0.25)
            print(f"  Experiments <25% E2E: {e2e_below_25}/{len(ok)}")
    else:
        print("\nStep 4: BLIS validation SKIPPED (--skip-blis)")

    print("\n=== H2 Complete ===")


if __name__ == "__main__":
    main()
