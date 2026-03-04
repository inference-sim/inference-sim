#!/usr/bin/env python3
"""Joint Bayesian optimization of all 5 LatencyModel methods.

Idea 1, H2: Uses h1's piecewise-linear StepTime as a warm start, then
optimizes StepTime coefficients + secondary method constants jointly using
BLIS E2E error as the objective function. The GP-based BO uses
scikit-optimize.

Usage:
    python3 bo_calibrate.py --h1-artifact-dir DIR [--output-dir DIR]
        [--max-evals 200] [--binary PATH]
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

# Add shared infrastructure to path
SHARED_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_loader import DEFAULT_DATA_ROOT
from validate_blis import (
    build_workload_spec,
    extract_kv_blocks_from_vllm_log,
    load_exp_config,
    load_ground_truth_metrics,
    load_profile,
    parse_experiment_dir,
    run_blis,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECONDARY_BOUNDS = {
    "output_token_processing_time_us": (0.0, 5000.0),
    "scheduling_processing_time_us": (0.0, 5000.0),
    "preemption_processing_time_us": (0.0, 5000.0),
}

OVERHEAD_BOUNDS = (500.0, 15000.0)  # step_overhead_us


# ---------------------------------------------------------------------------
# Artifact manipulation
# ---------------------------------------------------------------------------


def load_h1_artifact(path: str) -> dict:
    """Load h1's piecewise-linear artifact as the starting point."""
    with open(path) as f:
        return json.load(f)


def extract_bo_params(artifact: dict) -> list[float]:
    """Extract tunable parameters from an artifact as a flat list.

    Order:
    1. step_overhead_us
    2. output_token_processing_time_us
    3. scheduling_processing_time_us
    4. preemption_processing_time_us

    StepTime coefficients are kept fixed from h1 (warm start).
    This reduces the BO search space to 4 dimensions.
    """
    return [
        artifact.get("step_overhead_us", 0.0),
        artifact.get("output_token_processing_time_us", 0.0),
        artifact.get("scheduling_processing_time_us", 0.0),
        artifact.get("preemption_processing_time_us", 0.0),
    ]


def inject_bo_params(artifact: dict, params: list[float]) -> dict:
    """Inject BO-proposed parameters back into an artifact."""
    a = artifact.copy()
    a["step_overhead_us"] = params[0]
    a["output_token_processing_time_us"] = params[1]
    a["scheduling_processing_time_us"] = params[2]
    a["preemption_processing_time_us"] = params[3]
    return a


# ---------------------------------------------------------------------------
# BO objective
# ---------------------------------------------------------------------------


def evaluate_blis_e2e(
    artifact: dict,
    experiments: list[dict],
    binary: str,
) -> tuple[float, dict]:
    """Run BLIS on all experiments for a model and return mean E2E error.

    Returns (mean_e2e_error_pct, detailed_results_dict).
    """
    errors = []
    details = {}

    for exp in experiments:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(artifact, f, indent=2)
            artifact_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(exp["workload_spec"], f, default_flow_style=False)
            spec_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            results_path = f.name

        try:
            blis_metrics = run_blis(
                binary=binary,
                workload_spec_path=spec_path,
                exp_config=exp["exp_config"],
                total_kv_blocks=exp["total_kv_blocks"],
                alpha_coeffs=[0, 0, 0],
                beta_coeffs=[0, 0, 0],
                results_path=results_path,
                stepml_model_path=artifact_path,
            )
        finally:
            for p in [artifact_path, spec_path]:
                if os.path.exists(p):
                    os.unlink(p)
            if os.path.exists(results_path):
                os.unlink(results_path)

        if blis_metrics is None:
            errors.append(200.0)  # penalty for failed runs
            details[exp["dirname"]] = {"status": "failed"}
            continue

        gt_e2e = exp["gt"]["e2e_mean_s"] * 1000
        gt_itl = exp["gt"]["itl_mean_s"] * 1000
        gt_ttft = exp["gt"]["ttft_mean_s"] * 1000

        e2e_err = abs(blis_metrics["e2e_mean_ms"] - gt_e2e) / gt_e2e * 100
        itl_err = abs(blis_metrics["itl_mean_ms"] - gt_itl) / gt_itl * 100 if gt_itl > 0 else 0
        ttft_err = abs(blis_metrics["ttft_mean_ms"] - gt_ttft) / gt_ttft * 100 if gt_ttft > 0 else 0

        errors.append(e2e_err)
        details[exp["dirname"]] = {
            "e2e_err": e2e_err,
            "itl_err": itl_err,
            "ttft_err": ttft_err,
            "blis_e2e_ms": blis_metrics["e2e_mean_ms"],
            "gt_e2e_ms": gt_e2e,
        }

    mean_err = float(np.mean(errors)) if errors else 200.0
    return mean_err, details


# ---------------------------------------------------------------------------
# Experiment loading
# ---------------------------------------------------------------------------


def load_experiments_for_model(data_root: str, model_key: str) -> list[dict]:
    """Load all experiments matching a model key."""
    experiments = []

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

        exp_model_key = f"{meta['model']}_tp{meta['tp']}"
        # Match by model key, also handle -hf variants
        if exp_model_key != model_key and exp_model_key.replace("-hf_", "_") != model_key:
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

        experiments.append({
            "dirname": dirname,
            "meta": meta,
            "gt": gt,
            "exp_config": exp_config,
            "total_kv_blocks": total_kv_blocks,
            "workload_spec": workload_spec,
            "workload": meta["workload"],
        })

    return experiments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h1-artifact-dir",
        required=True,
        help="Directory with h1's piecewise artifacts (e.g., h1-piecewise-steptime/output/artifacts)",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "output"))
    parser.add_argument("--binary", default=str(Path(__file__).resolve().parents[4] / "simulation_worker"))
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-evals", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Discover model artifacts from h1
    # -----------------------------------------------------------------------
    print("Step 1: Loading h1 artifacts...", flush=True)
    artifact_files = {}
    for fname in sorted(os.listdir(args.h1_artifact_dir)):
        if fname.endswith("_piecewise.json"):
            model_key = fname.replace("_piecewise.json", "")
            artifact_files[model_key] = os.path.join(args.h1_artifact_dir, fname)
            print(f"  Found: {model_key}")

    if not artifact_files:
        print("ERROR: No h1 artifacts found. Run h1 first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: BO per model
    # -----------------------------------------------------------------------
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        print("ERROR: scikit-optimize not installed. Run: pip install scikit-optimize")
        sys.exit(1)

    all_results = []

    for model_key, artifact_path in artifact_files.items():
        print(f"\n{'='*70}")
        print(f"  BO for {model_key} (max {args.max_evals} evaluations)")
        print(f"{'='*70}", flush=True)

        base_artifact = load_h1_artifact(artifact_path)
        experiments = load_experiments_for_model(args.data_root, model_key)

        if not experiments:
            print(f"  No experiments found for {model_key}, skipping.")
            continue

        print(f"  Found {len(experiments)} experiments: "
              f"{[e['workload'] for e in experiments]}")

        # Define search space (4 dimensions)
        space = [
            Real(*OVERHEAD_BOUNDS, name="step_overhead_us"),
            Real(*SECONDARY_BOUNDS["output_token_processing_time_us"], name="output_token"),
            Real(*SECONDARY_BOUNDS["scheduling_processing_time_us"], name="scheduling"),
            Real(*SECONDARY_BOUNDS["preemption_processing_time_us"], name="preemption"),
        ]

        # Initial point from h1
        x0 = extract_bo_params(base_artifact)
        print(f"  Warm start: overhead={x0[0]:.0f}, "
              f"out_tok={x0[1]:.0f}, sched={x0[2]:.0f}, preempt={x0[3]:.0f}")

        eval_count = [0]
        best_error = [float("inf")]

        def objective(params):
            eval_count[0] += 1
            trial_artifact = inject_bo_params(base_artifact, params)
            mean_err, _ = evaluate_blis_e2e(trial_artifact, experiments, args.binary)

            if mean_err < best_error[0]:
                best_error[0] = mean_err

            if eval_count[0] % 10 == 0 or eval_count[0] <= 5:
                print(f"  [{eval_count[0]:3d}/{args.max_evals}] "
                      f"E2E={mean_err:.1f}% (best={best_error[0]:.1f}%)")

            return mean_err

        result = gp_minimize(
            objective,
            space,
            x0=x0,
            n_calls=args.max_evals,
            n_initial_points=min(20, args.max_evals // 3),
            random_state=42,
            verbose=False,
        )

        # Best parameters
        best_params = result.x
        best_artifact = inject_bo_params(base_artifact, best_params)

        print(f"\n  Best E2E error: {result.fun:.1f}%")
        print(f"  Best params: overhead={best_params[0]:.0f}, "
              f"out_tok={best_params[1]:.0f}, "
              f"sched={best_params[2]:.0f}, "
              f"preempt={best_params[3]:.0f}")

        # Final evaluation with best params
        final_err, final_details = evaluate_blis_e2e(
            best_artifact, experiments, args.binary
        )

        print(f"\n  Final per-experiment results:")
        for dirname, det in sorted(final_details.items()):
            if det.get("status") == "failed":
                print(f"    {dirname}: FAILED")
            else:
                print(f"    {dirname}: E2E={det['e2e_err']:.1f}%, "
                      f"ITL={det['itl_err']:.1f}%, TTFT={det['ttft_err']:.1f}%")

        # Save optimized artifact
        opt_artifact_path = os.path.join(
            args.output_dir, "artifacts", f"{model_key}_bo_optimized.json"
        )
        os.makedirs(os.path.dirname(opt_artifact_path), exist_ok=True)
        with open(opt_artifact_path, "w") as f:
            json.dump(best_artifact, f, indent=2)

        # Track results
        for dirname, det in final_details.items():
            if det.get("status") != "failed":
                all_results.append({
                    "model_key": model_key,
                    "experiment": dirname,
                    "e2e_error_pct": det["e2e_err"],
                    "itl_error_pct": det["itl_err"],
                    "ttft_error_pct": det["ttft_err"],
                    "blis_e2e_ms": det["blis_e2e_ms"],
                    "gt_e2e_ms": det["gt_e2e_ms"],
                    "bo_overhead": best_params[0],
                    "bo_output_token": best_params[1],
                    "bo_scheduling": best_params[2],
                    "bo_preemption": best_params[3],
                })

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    import pandas as pd

    print(f"\n{'='*70}")
    print("  JOINT BAYESIAN OPTIMIZATION RESULTS")
    print(f"{'='*70}")

    if not all_results:
        print("  No results!")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, "bo_results.csv"), index=False)

    mean_e2e = df["e2e_error_pct"].mean()
    mean_itl = df["itl_error_pct"].mean()
    mean_ttft = df["ttft_error_pct"].mean()
    below_10 = (df["e2e_error_pct"] < 10).sum()
    below_15 = (df["e2e_error_pct"] < 15).sum()

    print(f"\n  Experiments: {len(df)}")
    print(f"  Mean E2E error:  {mean_e2e:.1f}%")
    print(f"  Mean TTFT error: {mean_ttft:.1f}%")
    print(f"  Mean ITL error:  {mean_itl:.1f}%")
    print(f"  E2E < 15%: {below_15}/{len(df)}")
    print(f"  E2E < 10%: {below_10}/{len(df)}")

    # Hypothesis evaluation
    print(f"\n  {'='*70}")
    if mean_e2e < 15:
        print(f"  HYPOTHESIS SUPPORTED: Mean E2E = {mean_e2e:.1f}% < 15%")
    else:
        print(f"  HYPOTHESIS REFUTED: Mean E2E = {mean_e2e:.1f}% >= 15%")

    if mean_ttft > 50:
        print(f"  WARNING: Mean TTFT error = {mean_ttft:.1f}% > 50% (compensating errors?)")
    if mean_itl > 50:
        print(f"  WARNING: Mean ITL error = {mean_itl:.1f}% > 50% (compensating errors?)")

    print(f"  {'='*70}")

    # Save summary
    summary = {
        "mean_e2e_error_pct": mean_e2e,
        "mean_ttft_error_pct": mean_ttft,
        "mean_itl_error_pct": mean_itl,
        "experiments_below_15pct": int(below_15),
        "experiments_below_10pct": int(below_10),
        "total_experiments": len(df),
        "max_evals_per_model": args.max_evals,
        "hypothesis_status": "supported" if mean_e2e < 15 else "refuted",
    }
    with open(os.path.join(args.output_dir, "bo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
