"""Establish baselines for StepML research.

Runs two baselines through BLIS E2E validation and measures E2E/TTFT/ITL error
against ground truth.  These are the accuracy bars that StepML models must beat.

Baseline 1 — Roofline (analytical):
    Uses the analytical FLOPs/bandwidth roofline model.  Zero calibration data
    needed — step time is estimated from model architecture (HuggingFace
    config.json) and GPU specs (hardware_config.json).

Baseline 2 — Per-model linear regression:
    Trains a separate 3-coefficient linear regression per model+TP on
    ground-truth step traces (step_time = beta0 + beta1*prefill + beta2*decode),
    pooling all workloads for a given model.  Requires training data.

Usage:
    python establish_baseline.py [--data-root PATH] [--output-dir PATH]
    python establish_baseline.py --skip-linear              # roofline only
    python establish_baseline.py --skip-roofline            # linear only
    python establish_baseline.py --hardware A100            # non-default GPU
"""

import argparse
import csv as csv_mod
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Add shared/ to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import yaml

from baselines import BlackboxBaseline, check_r4_gate
from data_loader import load_all_experiments, load_experiment_steps, parse_experiment_metadata
from evaluation import compute_mape, compute_mspe, compute_pearson_r, compute_p99_error
from validate_blis import (
    build_workload_spec,
    compute_error,
    extract_cpu_kv_blocks_from_vllm_log,
    extract_kv_blocks_from_vllm_log,
    load_exp_config,
    load_ground_truth_metrics,
    load_profile,
    parse_experiment_dir,
    run_blis,
)

_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(_REPO_ROOT, "eval", "ground_truth")
_DEFAULT_WORKERS = min((os.cpu_count() or 4) // 2, 4)


def _build_canonical_model_map(data_root: str) -> dict:
    """Build a map from experiment_id to canonical (model, tp) from exp-config.yaml.

    Directory names have inconsistencies (e.g. "llama-2-70b" vs "llama-2-70b-hf"
    for the same model).  exp-config.yaml has the authoritative HF model name.
    """
    mapping = {}
    for dirname in sorted(os.listdir(data_root)):
        config_path = os.path.join(data_root, dirname, "exp-config.yaml")
        if not os.path.isfile(config_path):
            continue
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        # Canonical model name from HF (e.g. "meta-llama/Llama-2-70b-hf")
        model = cfg.get("model", "")
        tp = cfg.get("tensor_parallelism", 1)
        mapping[dirname] = (model, tp)
    return mapping


def train_per_model_coefficients(data_root: str) -> dict:
    """Train a linear regression per model+TP and return fitted coefficients.

    Groups experiments by canonical (model, tp) from exp-config.yaml, trains
    one linear regression per group, and returns the coefficients.

    Returns:
        dict keyed by (model, tp) with values:
        {
            "beta0": float, "beta1": float, "beta2": float,
            "n_steps": int, "n_experiments": int,
            "train_mape": float, "train_r": float,
            "experiments": [list of experiment_ids],
        }
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    # Build canonical model map from exp-config.yaml
    canon_map = _build_canonical_model_map(data_root)

    all_steps = load_all_experiments(data_root)
    if all_steps.empty:
        raise ValueError(f"No step data found in {data_root}")

    # Replace parsed model/tp with canonical values from exp-config.yaml
    all_steps["canonical_model"] = all_steps["experiment_id"].map(
        lambda eid: canon_map.get(eid, ("unknown", 0))[0]
    )
    all_steps["canonical_tp"] = all_steps["experiment_id"].map(
        lambda eid: canon_map.get(eid, ("unknown", 0))[1]
    )

    # Group by canonical model+tp
    results = {}
    for (model, tp), group_df in all_steps.groupby(["canonical_model", "canonical_tp"]):
        tp_val = int(tp)
        baseline = BlackboxBaseline().fit(group_df)
        coeffs = baseline.coefficients

        predicted = baseline.predict(group_df)
        actual = group_df["step.duration_us"].values

        train_mape = compute_mape(predicted, actual)
        train_r = compute_pearson_r(predicted, actual)
        train_mspe = compute_mspe(predicted, actual)

        experiments = sorted(group_df["experiment_id"].unique().tolist())

        results[(model, tp_val)] = {
            "beta0": coeffs["beta0"],
            "beta1": coeffs["beta1"],
            "beta2": coeffs["beta2"],
            "n_steps": len(group_df),
            "n_experiments": len(experiments),
            "train_mape": train_mape,
            "train_mspe": train_mspe,
            "train_r": train_r,
            "experiments": experiments,
        }

    return results


def _ensure_binary() -> str:
    """Return path to the simulation_worker binary, building if needed."""
    binary = os.path.join(_REPO_ROOT, "simulation_worker")
    if not os.path.isfile(binary):
        print("Building simulation_worker...", file=sys.stderr)
        subprocess.run(
            ["go", "build", "-o", binary, "main.go"],
            cwd=_REPO_ROOT, check=True,
        )
    return binary


def _run_single_experiment(
    binary: str,
    dirname: str,
    dirpath: str,
    alpha_coeffs: list[float],
    beta_coeffs: list[float],
    roofline: bool = False,
    hardware: str | None = None,
    extra_fields: dict | None = None,
) -> dict:
    """Run BLIS for a single experiment and return a result row.

    This is the unit of work dispatched to the thread pool.
    """
    try:
        meta = parse_experiment_dir(dirname)
    except ValueError:
        return {"experiment": dirname, "status": "parse_failed"}

    gt = load_ground_truth_metrics(dirpath)
    exp_config = load_exp_config(dirpath)
    total_kv_blocks = extract_kv_blocks_from_vllm_log(dirpath)
    if total_kv_blocks is None:
        return {
            "experiment": dirname, "model": meta["model"],
            "workload": meta["workload"], "tp": meta["tp"],
            "status": "no_kv_blocks",
        }
    cpu_kv_blocks = extract_cpu_kv_blocks_from_vllm_log(dirpath)

    try:
        profile = load_profile(dirpath)
    except Exception:
        return {
            "experiment": dirname, "model": meta["model"],
            "workload": meta["workload"], "tp": meta["tp"],
            "status": "no_profile",
        }

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
            alpha_coeffs=alpha_coeffs,
            beta_coeffs=beta_coeffs,
            results_path=results_json_path,
            roofline=roofline,
            hardware=hardware,
            cpu_kv_blocks=cpu_kv_blocks,
        )
    finally:
        os.unlink(spec_path)
        if os.path.exists(results_json_path):
            os.unlink(results_json_path)

    if blis_metrics is None:
        return {
            "experiment": dirname, "model": meta["model"],
            "workload": meta["workload"], "tp": meta["tp"],
            "status": "blis_failed",
        }

    e2e_err = compute_error(blis_metrics["e2e_mean_ms"], gt["e2e_mean_s"] * 1000)
    ttft_err = compute_error(blis_metrics["ttft_mean_ms"], gt["ttft_mean_s"] * 1000)
    itl_err = compute_error(blis_metrics["itl_mean_ms"], gt["itl_mean_s"] * 1000)

    marker = " <10%" if e2e_err < 0.10 else ""
    print(f"  {dirname}: GT={gt['e2e_mean_s']*1000:.0f}ms  "
          f"BLIS={blis_metrics['e2e_mean_ms']:.0f}ms  "
          f"E2E={e2e_err*100:.1f}%{marker}")

    row = {
        "experiment": dirname, "model": meta["model"],
        "workload": meta["workload"], "tp": meta["tp"],
        "status": "ok",
        "gt_e2e_ms": gt["e2e_mean_s"] * 1000,
        "gt_ttft_ms": gt["ttft_mean_s"] * 1000,
        "gt_itl_ms": gt["itl_mean_s"] * 1000,
        "blis_e2e_ms": blis_metrics["e2e_mean_ms"],
        "blis_ttft_ms": blis_metrics["ttft_mean_ms"],
        "blis_itl_ms": blis_metrics["itl_mean_ms"],
        "e2e_error": e2e_err, "ttft_error": ttft_err, "itl_error": itl_err,
        "blis_completed": blis_metrics["completed_requests"],
        "gt_requests": gt["num_requests"],
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _write_results_csv(rows: list[dict], output_csv: str) -> None:
    """Write result rows to CSV."""
    if not rows:
        return
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(output_csv, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _discover_experiments(data_root: str) -> list[tuple[str, str]]:
    """Return list of (dirname, dirpath) for valid experiment directories."""
    experiments = []
    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        summary_path = os.path.join(dirpath, "results", "summary_lifecycle_metrics.json")
        if not os.path.isfile(summary_path):
            continue
        experiments.append((dirname, dirpath))
    return experiments


def run_blis_validation(
    coefficients: dict,
    data_root: str,
    output_csv: str,
    max_workers: int = _DEFAULT_WORKERS,
) -> str:
    """Run BLIS for each experiment using its model+TP-specific coefficients.

    All experiments run in parallel via ThreadPoolExecutor.

    Returns path to the output CSV.
    """
    # Build experiment → coefficients lookup
    exp_to_coeffs = {}
    for (_model, _tp), coeffs in coefficients.items():
        for exp_id in coeffs["experiments"]:
            exp_to_coeffs[exp_id] = coeffs

    binary = _ensure_binary()
    experiments = _discover_experiments(data_root)

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for dirname, dirpath in experiments:
            coeffs = exp_to_coeffs.get(dirname)
            if coeffs is None:
                continue
            beta = [coeffs["beta0"], coeffs["beta1"], coeffs["beta2"]]
            extra = {
                "beta0": coeffs["beta0"], "beta1": coeffs["beta1"],
                "beta2": coeffs["beta2"], "train_mape": coeffs["train_mape"],
            }
            fut = pool.submit(
                _run_single_experiment,
                binary=binary, dirname=dirname, dirpath=dirpath,
                alpha_coeffs=[0.0, 0.0, 0.0], beta_coeffs=beta,
                extra_fields=extra,
            )
            futures[fut] = dirname

        all_rows = []
        for fut in as_completed(futures):
            all_rows.append(fut.result())

    # Sort by experiment name for deterministic output
    all_rows.sort(key=lambda r: r["experiment"])
    _write_results_csv(all_rows, output_csv)
    return output_csv


def run_roofline_validation(
    data_root: str,
    output_csv: str,
    hardware: str = "H100",
    max_workers: int = _DEFAULT_WORKERS,
) -> str:
    """Run BLIS in roofline mode for all experiments in parallel.

    The roofline model is analytical — no training step.  It estimates step
    time from FLOPs/bandwidth using HuggingFace model config + GPU hardware
    specs.  Zero calibration data needed.

    Returns path to the output CSV.
    """
    print(f"\n{'='*60}")
    print(f"  Roofline baseline: hardware={hardware}")
    print(f"  (analytical FLOPs/bandwidth — no trained coefficients)")
    print(f"{'='*60}")

    binary = _ensure_binary()
    experiments = _discover_experiments(data_root)

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for dirname, dirpath in experiments:
            fut = pool.submit(
                _run_single_experiment,
                binary=binary, dirname=dirname, dirpath=dirpath,
                alpha_coeffs=[0, 0, 0], beta_coeffs=[0, 0, 0],
                roofline=True, hardware=hardware,
            )
            futures[fut] = dirname

        all_rows = []
        for fut in as_completed(futures):
            all_rows.append(fut.result())

    all_rows.sort(key=lambda r: r["experiment"])
    _write_results_csv(all_rows, output_csv)
    return output_csv


def print_roofline_summary(output_csv: str):
    """Print a human-readable summary of the roofline baseline results."""
    if not os.path.isfile(output_csv):
        print("No roofline results to summarize.")
        return

    df = pd.read_csv(output_csv)
    ok = df[df["status"] == "ok"]

    if ok.empty:
        print("\nNo successful roofline experiments. Check BLIS binary and data.")
        return

    print(f"\n{'='*70}")
    print("  ROOFLINE BASELINE RESULTS (analytical — no training)")
    print(f"{'='*70}")

    print(f"\n  Experiments: {len(ok)} completed, "
          f"{len(df) - len(ok)} skipped/failed")

    print(f"\n  {'Experiment':<55s} {'E2E%':>6s} {'TTFT%':>6s} {'ITL%':>6s}")
    print(f"  {'-'*55} {'-'*6} {'-'*6} {'-'*6}")
    for _, row in ok.iterrows():
        e2e_pct = row["e2e_error"] * 100
        ttft_pct = row["ttft_error"] * 100
        itl_pct = row["itl_error"] * 100
        marker = " <10%" if e2e_pct < 10 else ""
        print(f"  {row['experiment']:<55s} "
              f"{e2e_pct:>5.1f}% {ttft_pct:>5.1f}% {itl_pct:>5.1f}%{marker}")

    e2e_errors = ok["e2e_error"].values * 100
    ttft_errors = ok["ttft_error"].values * 100
    itl_errors = ok["itl_error"].values * 100

    print(f"\n  Mean E2E error:  {e2e_errors.mean():.1f}%")
    print(f"  Mean TTFT error: {ttft_errors.mean():.1f}%")
    print(f"  Mean ITL error:  {itl_errors.mean():.1f}%")

    passing = (e2e_errors < 10).sum()
    print(f"\n  E2E < 10%: {passing}/{len(ok)} experiments")

    # Save summary JSON
    summary = {
        "model": "roofline (analytical)",
        "e2e_validation": {
            "mean_e2e_error_pct": float(e2e_errors.mean()),
            "mean_ttft_error_pct": float(ttft_errors.mean()),
            "mean_itl_error_pct": float(itl_errors.mean()),
            "experiments_below_10pct": int(passing),
            "total_experiments": len(ok),
        },
    }
    summary_path = output_csv.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON: {summary_path}")


def print_summary(output_csv: str, coefficients: dict):
    """Print a human-readable summary of the baseline results."""
    if not os.path.isfile(output_csv):
        print("No results to summarize.")
        return

    df = pd.read_csv(output_csv)
    ok = df[df["status"] == "ok"]

    if ok.empty:
        print("\nNo successful experiments. Check BLIS binary and data.")
        return

    print(f"\n{'='*70}")
    print("  PER-MODEL LINEAR REGRESSION BASELINE RESULTS")
    print(f"{'='*70}")

    print(f"\n  Experiments: {len(ok)} completed, "
          f"{len(df) - len(ok)} skipped/failed")

    # Per-step training metrics
    print(f"\n  --- Per-Step Training Metrics (linear regression on step traces) ---")
    for (model, tp), coeffs in sorted(coefficients.items()):
        print(f"  {model} TP={tp}: "
              f"MAPE={coeffs['train_mape']:.1f}%, "
              f"MSPE={coeffs['train_mspe']:+.1f}%, "
              f"r={coeffs['train_r']:.3f}, "
              f"n_steps={coeffs['n_steps']:,}")
        print(f"    beta = [{coeffs['beta0']:.2f}, "
              f"{coeffs['beta1']:.4f}, {coeffs['beta2']:.4f}]")

    # E2E validation metrics
    print(f"\n  --- BLIS E2E Validation (simulator vs ground truth) ---")
    print(f"  {'Experiment':<55s} {'E2E%':>6s} {'TTFT%':>6s} {'ITL%':>6s}")
    print(f"  {'-'*55} {'-'*6} {'-'*6} {'-'*6}")
    for _, row in ok.iterrows():
        e2e_pct = row["e2e_error"] * 100
        ttft_pct = row["ttft_error"] * 100
        itl_pct = row["itl_error"] * 100
        marker = " <10%" if e2e_pct < 10 else ""
        print(f"  {row['experiment']:<55s} "
              f"{e2e_pct:>5.1f}% {ttft_pct:>5.1f}% {itl_pct:>5.1f}%{marker}")

    e2e_errors = ok["e2e_error"].values * 100
    ttft_errors = ok["ttft_error"].values * 100
    itl_errors = ok["itl_error"].values * 100

    print(f"\n  Mean E2E error:  {e2e_errors.mean():.1f}%")
    print(f"  Mean TTFT error: {ttft_errors.mean():.1f}%")
    print(f"  Mean ITL error:  {itl_errors.mean():.1f}%")

    passing = (e2e_errors < 10).sum()
    print(f"\n  E2E < 10%: {passing}/{len(ok)} experiments")

    # R4 gate check
    print(f"\n  --- R4 Gate (is linear regression already good enough?) ---")
    r4 = check_r4_gate(e2e_errors.mean())
    print(f"  {r4['message']}")

    # Save summary JSON
    summary = {
        "per_model_coefficients": {
            f"{m}_tp{tp}": {k: v for k, v in coeffs.items()
                            if k != "experiments"}
            for (m, tp), coeffs in coefficients.items()
        },
        "e2e_validation": {
            "mean_e2e_error_pct": float(e2e_errors.mean()),
            "mean_ttft_error_pct": float(ttft_errors.mean()),
            "mean_itl_error_pct": float(itl_errors.mean()),
            "experiments_below_10pct": int(passing),
            "total_experiments": len(ok),
        },
        "r4_gate": r4,
    }
    summary_path = output_csv.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Establish per-model linear regression baseline"
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Path to eval/ground_truth/",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_SCRIPT_DIR, "baseline_results"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--hardware",
        default="H100",
        help="GPU type for roofline baseline (default: H100)",
    )
    parser.add_argument(
        "--skip-roofline",
        action="store_true",
        help="Skip the roofline baseline (only run linear regression)",
    )
    parser.add_argument(
        "--skip-linear",
        action="store_true",
        help="Skip the linear regression baseline (only run roofline)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help=f"Max parallel BLIS runs (default: {_DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Baseline 1: Roofline (analytical, no training) ---
    if not args.skip_roofline:
        roofline_csv = os.path.join(args.output_dir, "roofline_baseline.csv")

        print(f"Step 1: Running roofline baseline "
              f"(hardware={args.hardware}, workers={args.workers})...")
        run_roofline_validation(
            args.data_root, roofline_csv, args.hardware,
            max_workers=args.workers,
        )

        print_roofline_summary(roofline_csv)

    # --- Baseline 2: Per-model linear regression ---
    if not args.skip_linear:
        linear_csv = os.path.join(args.output_dir, "per_model_linear_baseline.csv")

        step_num = "2" if not args.skip_roofline else "1"
        print(f"\nStep {step_num}: Training per-model linear regression on step traces...")
        coefficients = train_per_model_coefficients(args.data_root)
        print(f"  Trained {len(coefficients)} model+TP groups")

        print(f"\nStep {int(step_num)+1}: Running BLIS E2E validation "
              f"(workers={args.workers})...")
        run_blis_validation(
            coefficients, args.data_root, linear_csv,
            max_workers=args.workers,
        )

        print_summary(linear_csv, coefficients)


if __name__ == "__main__":
    main()
