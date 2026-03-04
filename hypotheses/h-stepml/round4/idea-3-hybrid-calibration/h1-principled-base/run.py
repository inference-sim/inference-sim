#!/usr/bin/env python3
"""H1: Stage 1 Principled Base — Overhead-Scaled Step Time + TTFT Corrections.

Key insight from R3: BLIS simulates a "faster universe" (~40% of real time).
The root cause is that step.duration_us measures only GPU forward-pass time,
while real step cycle time includes CPU scheduling overhead, CUDA sync, and
memory management. The overhead floor (max(overhead, compute)) is the right
mechanism but needs calibration from lifecycle data.

This script:
1. Computes per-model median ITL from lifecycle data (the true step cycle time)
2. Computes per-model median step.duration_us (GPU-only time)
3. Derives overhead_multiplier = median_ITL / median_duration (typically ~2-3x)
4. Trains per-model regime regression on SCALED step times (duration * multiplier)
5. Applies TTFT additive corrections from lifecycle data
6. Exports StepML artifacts and runs BLIS validation
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Add shared infrastructure to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IDEA_DIR = os.path.dirname(_SCRIPT_DIR)
_SHARED_DIR = os.path.join(_IDEA_DIR, "..", "..", "shared")
sys.path.insert(0, _SHARED_DIR)

from data_loader import (
    load_all_experiments,
    load_lifecycle_data,
    parse_experiment_metadata,
    DEFAULT_DATA_ROOT,
)
from evaluation import compute_mape
from validate_blis import DEFAULT_BINARY

OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Compute per-model ITL from lifecycle data
# ---------------------------------------------------------------------------
def compute_per_model_itl(data_root: str) -> dict:
    """Compute per-model median inter-token latency from lifecycle data.

    ITL = median of (output_token_times[i+1] - output_token_times[i]) for all
    requests in all experiments for each model. This is the true step cycle time
    as seen by the client (GPU compute + CPU overhead + CUDA sync).
    """
    model_itls = {}  # {model: [itl_values]}

    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        lifecycle_path = os.path.join(
            dirpath, "results", "per_request_lifecycle_metrics.json"
        )
        if not os.path.isfile(lifecycle_path):
            continue

        meta = parse_experiment_metadata(dirname)
        model = meta["model"]

        try:
            lifecycle_df = load_lifecycle_data(dirpath)
        except Exception:
            continue

        itls = []
        for _, row in lifecycle_df.iterrows():
            times = row.get("output_token_times", [])
            if not times or len(times) < 2:
                continue
            for i in range(1, len(times)):
                iti_s = times[i] - times[i - 1]
                if iti_s > 0:
                    itls.append(iti_s * 1_000_000)  # convert to us

        if itls:
            model_itls.setdefault(model, []).extend(itls)

    result = {}
    for model, itls in model_itls.items():
        arr = np.array(itls)
        result[model] = {
            "median_itl_us": float(np.median(arr)),
            "mean_itl_us": float(np.mean(arr)),
            "p10_itl_us": float(np.percentile(arr, 10)),
            "p90_itl_us": float(np.percentile(arr, 90)),
            "n_tokens": len(arr),
        }
        print(
            f"  {model}: median_ITL={result[model]['median_itl_us']:.0f}us, "
            f"mean_ITL={result[model]['mean_itl_us']:.0f}us, "
            f"p10={result[model]['p10_itl_us']:.0f}us, "
            f"p90={result[model]['p90_itl_us']:.0f}us, n={len(arr)}"
        )

    return result


# ---------------------------------------------------------------------------
# Step 2: Compute overhead multiplier per model
# ---------------------------------------------------------------------------
def compute_overhead_multipliers(
    step_df: pd.DataFrame, itl_data: dict
) -> dict:
    """Compute per-model overhead multiplier = median_ITL / median_step_duration.

    This captures the ratio of real cycle time (ITL) to GPU-only time (duration).
    """
    multipliers = {}
    for model in sorted(step_df["model"].unique()):
        mdf = step_df[step_df["model"] == model]
        dur = mdf["step.duration_us"].dropna()
        if len(dur) == 0:
            continue

        median_dur = float(dur.median())
        itl_info = itl_data.get(model)
        if itl_info is None:
            # Try normalized name
            for k, v in itl_data.items():
                if k.replace("-hf", "") == model.replace("-hf", ""):
                    itl_info = v
                    break

        if itl_info is None:
            print(f"  {model}: no ITL data, using 2.0x default multiplier")
            multipliers[model] = {
                "multiplier": 2.0,
                "median_dur_us": median_dur,
                "median_itl_us": median_dur * 2.0,
            }
            continue

        median_itl = itl_info["median_itl_us"]
        mult = median_itl / median_dur if median_dur > 0 else 2.0

        # Clamp to reasonable range [1.2, 10.0]
        mult = max(1.2, min(10.0, mult))

        multipliers[model] = {
            "multiplier": float(mult),
            "median_dur_us": float(median_dur),
            "median_itl_us": float(median_itl),
        }
        print(
            f"  {model}: multiplier={mult:.2f}x "
            f"(median_ITL={median_itl:.0f}us / median_dur={median_dur:.0f}us)"
        )

    return multipliers


# ---------------------------------------------------------------------------
# Step 3: Train per-model regression on overhead-scaled step times
# ---------------------------------------------------------------------------
def train_per_model_regression(
    step_df: pd.DataFrame, multipliers: dict
) -> dict:
    """Train per-model regime regression on overhead-scaled step times.

    target = step.duration_us * overhead_multiplier (the estimated cycle time)
    features = [scheduled_tokens, num_decode_reqs]
    """
    models = {}

    for model_name in sorted(step_df["model"].unique()):
        mdf = step_df[step_df["model"] == model_name].copy()
        mult_info = multipliers.get(model_name, {"multiplier": 2.0})
        mult = mult_info["multiplier"]

        # Compute scaled target
        mdf = mdf.dropna(subset=["step.duration_us"])
        mdf["scaled_duration_us"] = mdf["step.duration_us"].astype(float) * mult

        # Feature columns
        if "batch.scheduled_tokens" in mdf.columns:
            mdf["new_tokens"] = mdf["batch.scheduled_tokens"].astype(float)
        else:
            mdf["new_tokens"] = (
                mdf.get("batch.prefill_tokens", pd.Series(0, index=mdf.index)).astype(float)
                + mdf.get("batch.decode_tokens", pd.Series(0, index=mdf.index)).astype(float)
            )

        mdf["n_decode_reqs"] = mdf.get(
            "batch.num_decode_reqs", pd.Series(0, index=mdf.index)
        ).astype(float)

        # Regime separation
        if "batch.prefill_tokens" in mdf.columns:
            decode_only = mdf[mdf["batch.prefill_tokens"] == 0]
            mixed = mdf[mdf["batch.prefill_tokens"] > 0]
        else:
            decode_only = mdf
            mixed = mdf.iloc[0:0]

        regimes = {"multiplier": mult}

        # Train decode-only
        if len(decode_only) >= 10:
            X = decode_only[["new_tokens", "n_decode_reqs"]].values
            y = decode_only["scaled_duration_us"].values
            reg = Ridge(alpha=10.0)
            reg.fit(X, y)
            pred = reg.predict(X)
            mape = compute_mape(pred, y)
            regimes["decode_only"] = {
                "intercept": float(reg.intercept_),
                "coeff_new_tokens": float(reg.coef_[0]),
                "coeff_n_decode_reqs": float(reg.coef_[1]),
                "mape": mape,
                "n": len(decode_only),
                "mean_target": float(y.mean()),
            }
            print(
                f"  {model_name} decode-only: intercept={reg.intercept_:.0f}, "
                f"b_tokens={reg.coef_[0]:.2f}, b_reqs={reg.coef_[1]:.2f}, "
                f"MAPE={mape:.1f}%, n={len(decode_only)}"
            )

        # Train mixed
        if len(mixed) >= 10:
            X = mixed[["new_tokens", "n_decode_reqs"]].values
            y = mixed["scaled_duration_us"].values
            reg = Ridge(alpha=10.0)
            reg.fit(X, y)
            pred = reg.predict(X)
            mape = compute_mape(pred, y)
            regimes["mixed"] = {
                "intercept": float(reg.intercept_),
                "coeff_new_tokens": float(reg.coef_[0]),
                "coeff_n_decode_reqs": float(reg.coef_[1]),
                "mape": mape,
                "n": len(mixed),
                "mean_target": float(y.mean()),
            }
            print(
                f"  {model_name} mixed:       intercept={reg.intercept_:.0f}, "
                f"b_tokens={reg.coef_[0]:.2f}, b_reqs={reg.coef_[1]:.2f}, "
                f"MAPE={mape:.1f}%, n={len(mixed)}"
            )

        # Global
        X = mdf[["new_tokens", "n_decode_reqs"]].values
        y = mdf["scaled_duration_us"].values
        reg = Ridge(alpha=10.0)
        reg.fit(X, y)
        pred = reg.predict(X)
        mape = compute_mape(pred, y)
        regimes["global"] = {
            "intercept": float(reg.intercept_),
            "coeff_new_tokens": float(reg.coef_[0]),
            "coeff_n_decode_reqs": float(reg.coef_[1]),
            "mape": mape,
            "n": len(mdf),
            "mean_target": float(y.mean()),
        }

        models[model_name] = regimes

    return models


# ---------------------------------------------------------------------------
# Step 4: TTFT corrections from lifecycle data
# ---------------------------------------------------------------------------
def estimate_ttft_corrections(data_root: str) -> dict:
    """Per-model TTFT correction = observed mean TTFT from lifecycle data."""
    corrections = {}
    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        lifecycle_path = os.path.join(
            dirpath, "results", "per_request_lifecycle_metrics.json"
        )
        if not os.path.isfile(lifecycle_path):
            continue

        meta = parse_experiment_metadata(dirname)
        model = meta["model"]

        try:
            lifecycle_df = load_lifecycle_data(dirpath)
        except Exception:
            continue

        ttft_values = []
        for _, row in lifecycle_df.iterrows():
            times = row.get("output_token_times", [])
            start = row.get("start_time", 0)
            if times and len(times) > 0 and start > 0:
                ttft_s = times[0] - start
                if ttft_s > 0:
                    ttft_values.append(ttft_s)

        if ttft_values:
            corrections.setdefault(model, []).extend(ttft_values)

    result = {}
    for model, values in corrections.items():
        mean_ttft_us = float(np.mean(values)) * 1_000_000
        result[model] = mean_ttft_us
        print(f"  TTFT correction for {model}: {mean_ttft_us:.0f}us ({mean_ttft_us/1000:.1f}ms)")

    return result


# ---------------------------------------------------------------------------
# Step 5: Export StepML artifacts
# ---------------------------------------------------------------------------
def export_stepml_artifact(
    model_name: str, regimes: dict, ttft_correction_us: float, output_path: str
) -> dict:
    """Export per-model StepML JSON artifact."""
    mult = regimes.get("multiplier", 2.0)

    # Use decode-only intercept as overhead floor
    if "decode_only" in regimes:
        d = regimes["decode_only"]
        overhead_us = max(d["intercept"], 1000)
    elif "global" in regimes:
        overhead_us = max(regimes["global"]["intercept"], 1000)
    else:
        overhead_us = 4000

    step_time_regimes = []

    if "decode_only" in regimes:
        d = regimes["decode_only"]
        step_time_regimes.append(
            {
                "name": "decode_only",
                "condition": {"feature": "prefill_tokens", "op": "==", "value": 0},
                "model": {
                    "model_type": "linear",
                    "intercept": d["intercept"],
                    "feature_coefficients": {
                        "decode_tokens": d["coeff_new_tokens"],
                        "num_decode_reqs": d["coeff_n_decode_reqs"],
                    },
                },
            }
        )

    if "mixed" in regimes:
        m = regimes["mixed"]
        step_time_regimes.append(
            {
                "name": "mixed",
                "condition": None,
                "model": {
                    "model_type": "linear",
                    "intercept": m["intercept"],
                    "feature_coefficients": {
                        "scheduled_tokens": m["coeff_new_tokens"],
                        "num_decode_reqs": m["coeff_n_decode_reqs"],
                    },
                },
            }
        )

    if not step_time_regimes:
        g = regimes["global"]
        step_time_regimes.append(
            {
                "name": "global",
                "condition": None,
                "model": {
                    "model_type": "linear",
                    "intercept": g["intercept"],
                    "feature_coefficients": {
                        "scheduled_tokens": g["coeff_new_tokens"],
                        "num_decode_reqs": g["coeff_n_decode_reqs"],
                    },
                },
            }
        )

    artifact = {
        "version": 2,
        "step_time_regimes": step_time_regimes,
        "step_overhead_us": overhead_us,
        "output_token_processing_time_us": 0,
        "scheduling_processing_time_us": 0,
        "preemption_processing_time_us": 0,
    }

    if ttft_correction_us > 0:
        artifact["queueing_time"] = {
            "model_type": "linear",
            "intercept": ttft_correction_us,
            "feature_coefficients": {},
        }

    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(
        f"  {model_name}: overhead={overhead_us:.0f}us, "
        f"TTFT={ttft_correction_us:.0f}us, mult={mult:.2f}x"
    )
    return artifact


# ---------------------------------------------------------------------------
# Step 6: BLIS validation
# ---------------------------------------------------------------------------
def run_blis_validation_per_model(
    data_root: str, artifacts: dict, output_csv: str
) -> pd.DataFrame:
    """Run BLIS validation with per-model StepML artifacts."""
    from validate_blis import (
        parse_experiment_dir,
        load_ground_truth_metrics,
        load_exp_config,
        extract_kv_blocks_from_vllm_log,
        extract_cpu_kv_blocks_from_vllm_log,
        load_profile,
        build_workload_spec,
        run_blis,
        compute_error,
    )
    import tempfile
    import yaml

    binary = DEFAULT_BINARY
    if not os.path.isfile(binary):
        import subprocess
        repo_root = os.path.abspath(os.path.join(_SHARED_DIR, "..", "..", ".."))
        subprocess.run(["go", "build", "-o", binary, "main.go"], cwd=repo_root, check=True)

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

        model_name = meta["model"]

        # Find matching artifact
        artifact_path = None
        for art_model, art_path in artifacts.items():
            if art_model == model_name:
                artifact_path = art_path
                break
        if artifact_path is None:
            for art_model, art_path in artifacts.items():
                if art_model.replace("-hf", "") == model_name.replace("-hf", ""):
                    artifact_path = art_path
                    break
        if artifact_path is None:
            results.append({"experiment": dirname, "model": model_name, "workload": meta["workload"], "status": "no_artifact"})
            continue

        gt = load_ground_truth_metrics(dirpath)
        exp_config = load_exp_config(dirpath)
        total_kv_blocks = extract_kv_blocks_from_vllm_log(dirpath)
        cpu_kv_blocks = extract_cpu_kv_blocks_from_vllm_log(dirpath)

        if total_kv_blocks is None:
            results.append({"experiment": dirname, "model": model_name, "workload": meta["workload"], "status": "no_kv_blocks"})
            continue

        try:
            profile = load_profile(dirpath)
        except Exception:
            results.append({"experiment": dirname, "model": model_name, "workload": meta["workload"], "status": "no_profile"})
            continue

        workload_spec = build_workload_spec(profile, gt)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(workload_spec, f, default_flow_style=False)
            spec_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            results_json_path = f.name

        try:
            blis_metrics = run_blis(
                binary=binary,
                workload_spec_path=spec_path,
                exp_config=exp_config,
                total_kv_blocks=total_kv_blocks,
                alpha_coeffs=[0, 0, 0],
                beta_coeffs=[0, 0, 0],
                results_path=results_json_path,
                stepml_model_path=artifact_path,
                cpu_kv_blocks=cpu_kv_blocks,
            )
        finally:
            os.unlink(spec_path)
            if os.path.exists(results_json_path):
                os.unlink(results_json_path)

        if blis_metrics is None:
            results.append({"experiment": dirname, "model": model_name, "workload": meta["workload"], "status": "blis_failed"})
            continue

        e2e_err = compute_error(blis_metrics["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
        ttft_err = compute_error(blis_metrics["ttft_mean_ms"], gt["ttft_mean_s"] * 1000)
        itl_err = compute_error(blis_metrics["itl_mean_ms"], gt["itl_mean_s"] * 1000)

        print(f"  {dirname}: E2E={e2e_err*100:.1f}%, TTFT={ttft_err*100:.1f}%, ITL={itl_err*100:.1f}%")

        results.append({
            "experiment": dirname,
            "model": model_name,
            "workload": meta["workload"],
            "tp": meta["tp"],
            "status": "ok",
            "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
            "gt_ttft_ms": gt["ttft_mean_s"] * 1000,
            "gt_itl_ms": gt["itl_mean_s"] * 1000,
            "blis_e2e_ms": blis_metrics["e2e_mean_ms"],
            "blis_ttft_ms": blis_metrics["ttft_mean_ms"],
            "blis_itl_ms": blis_metrics["itl_mean_ms"],
            "e2e_error": e2e_err,
            "ttft_error": ttft_err,
            "itl_error": itl_err,
            "blis_completed": blis_metrics["completed_requests"],
            "gt_requests": gt["num_requests"],
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("H1: Stage 1 Principled Base — Overhead-Scaled + TTFT Corrections")
    print("=" * 70)

    data_root = DEFAULT_DATA_ROOT

    # Step 1: Compute ITL from lifecycle data
    print("\n--- Step 1: Per-model ITL from lifecycle data ---")
    itl_data = compute_per_model_itl(data_root)

    # Step 2: Load step data and compute multipliers
    print("\n--- Step 2: Compute overhead multipliers ---")
    step_df = load_all_experiments(data_root)
    multipliers = compute_overhead_multipliers(step_df, itl_data)

    # Step 3: Train regression on scaled step times
    print("\n--- Step 3: Train per-model regression on scaled step times ---")
    models = train_per_model_regression(step_df, multipliers)

    with open(os.path.join(OUTPUT_DIR, "regression_results.json"), "w") as f:
        json.dump(models, f, indent=2, default=str)

    # Step 4: TTFT corrections
    print("\n--- Step 4: TTFT corrections ---")
    ttft_corrections = estimate_ttft_corrections(data_root)

    # Step 5: Export artifacts
    print("\n--- Step 5: Export StepML artifacts ---")
    artifact_dir = os.path.join(OUTPUT_DIR, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    artifacts = {}
    for model_name, regimes in models.items():
        ttft_us = 0
        for ttft_model, ttft_val in ttft_corrections.items():
            if ttft_model == model_name or ttft_model.replace("-hf", "") == model_name.replace("-hf", ""):
                ttft_us = ttft_val
                break

        artifact_path = os.path.join(artifact_dir, f"{model_name}.json")
        export_stepml_artifact(model_name, regimes, ttft_us, artifact_path)
        artifacts[model_name] = artifact_path

    # Step 6: BLIS validation
    print("\n--- Step 6: BLIS validation ---")
    results_csv = os.path.join(OUTPUT_DIR, "blis_validation.csv")
    results_df = run_blis_validation_per_model(data_root, artifacts, results_csv)

    # Summary
    ok = results_df[results_df["status"] == "ok"]
    if len(ok) > 0:
        mean_e2e = ok["e2e_error"].mean() * 100
        mean_ttft = ok["ttft_error"].mean() * 100
        mean_itl = ok["itl_error"].mean() * 100

        print("\n" + "=" * 70)
        print("STAGE 1 RESULTS SUMMARY")
        print("=" * 70)
        print(f"Experiments: {len(ok)}")
        print(f"Mean E2E:  {mean_e2e:.1f}% (target <30%)")
        print(f"Mean TTFT: {mean_ttft:.1f}%")
        print(f"Mean ITL:  {mean_itl:.1f}% (target <20%)")
        below_10 = (ok["e2e_error"] < 0.10).sum()
        print(f"E2E < 10%: {below_10}/{len(ok)}")

        print(f"\n{'Experiment':<55} {'E2E%':>7} {'TTFT%':>7} {'ITL%':>7}")
        print("-" * 80)
        for _, row in ok.iterrows():
            print(
                f"{row['experiment']:<55} {row['e2e_error']*100:>7.1f} "
                f"{row['ttft_error']*100:>7.1f} {row['itl_error']*100:>7.1f}"
            )
        print("-" * 80)
        print(f"{'MEAN':<55} {mean_e2e:>7.1f} {mean_ttft:>7.1f} {mean_itl:>7.1f}")

        # Save summary
        summary = {
            "mean_e2e_error": mean_e2e,
            "mean_ttft_error": mean_ttft,
            "mean_itl_error": mean_itl,
            "n_experiments": len(ok),
            "n_below_10_e2e": int(below_10),
            "per_experiment": ok[
                ["experiment", "model", "workload", "e2e_error", "ttft_error", "itl_error",
                 "gt_e2e_ms", "blis_e2e_ms", "gt_itl_ms", "blis_itl_ms"]
            ].to_dict("records"),
            "multipliers": multipliers,
            "regression_results": {k: {rk: rv for rk, rv in v.items() if rk != "multiplier"} for k, v in models.items()},
            "ttft_corrections": {k: float(v) for k, v in ttft_corrections.items()},
            "itl_data": itl_data,
        }
        with open(os.path.join(OUTPUT_DIR, "h1_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
    else:
        print("\nERROR: No experiments completed successfully!")
        sys.exit(1)


if __name__ == "__main__":
    main()
