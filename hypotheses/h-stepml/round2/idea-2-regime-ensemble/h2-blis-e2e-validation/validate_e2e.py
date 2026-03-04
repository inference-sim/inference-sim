#!/usr/bin/env python3
"""Idea 2, H2: BLIS E2E validation with per-model regime StepML artifacts.

For each ground-truth experiment, selects the matching model artifact
(regime or single) and runs BLIS to compute E2E, TTFT, ITL errors.

Usage: python3 validate_e2e.py --artifact-dir DIR [--mode regime|single] [--output-dir DIR]
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

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


def _model_key_from_experiment(meta: dict) -> str:
    """Convert experiment metadata to artifact filename key."""
    return f"{meta['model']}_tp{meta['tp']}"


def _find_artifact(artifact_dir: str, model_key: str, mode: str) -> str | None:
    """Find the StepML artifact for a given model key and mode."""
    suffix = "_regime.json" if mode == "regime" else "_single.json"
    path = os.path.join(artifact_dir, model_key + suffix)
    if os.path.isfile(path):
        return path

    # Try normalizing model key (strip -hf suffix)
    normalized = model_key.replace("-hf_", "_")
    path = os.path.join(artifact_dir, normalized + suffix)
    if os.path.isfile(path):
        return path

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory containing per-model StepML artifacts",
    )
    parser.add_argument(
        "--mode",
        choices=["regime", "single"],
        default="regime",
        help="Which artifact type to use (default: regime)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "output"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--binary",
        default=str(Path(__file__).resolve().parents[4] / "simulation_worker"),
        help="Path to BLIS binary",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Path to ground truth data",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel BLIS runs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    skipped = []

    for dirname in sorted(os.listdir(args.data_root)):
        dirpath = os.path.join(args.data_root, dirname)
        if not os.path.isdir(dirpath):
            continue

        summary_path = os.path.join(
            dirpath, "results", "summary_lifecycle_metrics.json"
        )
        if not os.path.isfile(summary_path):
            continue

        try:
            meta = parse_experiment_dir(dirname)
        except ValueError:
            continue

        model_key = _model_key_from_experiment(meta)
        artifact_path = _find_artifact(args.artifact_dir, model_key, args.mode)

        if artifact_path is None:
            print(f"  {dirname}: SKIP (no artifact for {model_key})")
            skipped.append(dirname)
            continue

        gt = load_ground_truth_metrics(dirpath)
        exp_config = load_exp_config(dirpath)
        total_kv_blocks = extract_kv_blocks_from_vllm_log(dirpath)
        if total_kv_blocks is None:
            print(f"  {dirname}: SKIP (no KV blocks)")
            skipped.append(dirname)
            continue

        try:
            profile = load_profile(dirpath)
        except Exception:
            print(f"  {dirname}: SKIP (no profile)")
            skipped.append(dirname)
            continue

        workload_spec = build_workload_spec(profile, gt)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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
            print(f"  {dirname}: TIMEOUT or FAIL")
            skipped.append(dirname)
            continue

        gt_e2e_ms = gt["e2e_mean_s"] * 1000
        gt_ttft_ms = gt["ttft_mean_s"] * 1000
        gt_itl_ms = gt["itl_mean_s"] * 1000

        e2e_err = abs(blis_metrics["e2e_mean_ms"] - gt_e2e_ms) / gt_e2e_ms * 100
        ttft_err = abs(blis_metrics["ttft_mean_ms"] - gt_ttft_ms) / gt_ttft_ms * 100 if gt_ttft_ms > 0 else 0
        itl_err = abs(blis_metrics["itl_mean_ms"] - gt_itl_ms) / gt_itl_ms * 100 if gt_itl_ms > 0 else 0

        results.append({
            "experiment": dirname,
            "model": meta["model"],
            "workload": meta["workload"],
            "tp": meta["tp"],
            "model_key": model_key,
            "artifact": os.path.basename(artifact_path),
            "gt_e2e_ms": gt_e2e_ms,
            "blis_e2e_ms": blis_metrics["e2e_mean_ms"],
            "e2e_error_pct": e2e_err,
            "gt_ttft_ms": gt_ttft_ms,
            "blis_ttft_ms": blis_metrics["ttft_mean_ms"],
            "ttft_error_pct": ttft_err,
            "gt_itl_ms": gt_itl_ms,
            "blis_itl_ms": blis_metrics["itl_mean_ms"],
            "itl_error_pct": itl_err,
        })

        print(
            f"  {dirname}: GT={gt_e2e_ms:.0f}ms  BLIS={blis_metrics['e2e_mean_ms']:.0f}ms  "
            f"E2E={e2e_err:.1f}%  TTFT={ttft_err:.1f}%  ITL={itl_err:.1f}%"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    import pandas as pd

    if not results:
        print("\nNo experiments completed successfully!")
        return

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"e2e_validation_{args.mode}.csv")
    df.to_csv(csv_path, index=False)

    mean_e2e = df["e2e_error_pct"].mean()
    mean_ttft = df["ttft_error_pct"].mean()
    mean_itl = df["itl_error_pct"].mean()
    below_10 = (df["e2e_error_pct"] < 10).sum()

    print(f"\n{'='*70}")
    print(f"  BLIS E2E VALIDATION — {args.mode.upper()} MODEL")
    print(f"{'='*70}")
    print(f"\n  Experiments: {len(df)} completed, {len(skipped)} skipped")
    print(f"\n  {'Experiment':<55s} {'E2E%':>6s} {'TTFT%':>7s} {'ITL%':>6s}")
    print(f"  {'-'*55} {'-'*6} {'-'*7} {'-'*6}")
    for _, row in df.iterrows():
        print(
            f"  {row['experiment']:<55s} {row['e2e_error_pct']:>5.1f}% "
            f"{row['ttft_error_pct']:>6.1f}% {row['itl_error_pct']:>5.1f}%"
        )
    print(f"\n  Mean E2E error:  {mean_e2e:.1f}%")
    print(f"  Mean TTFT error: {mean_ttft:.1f}%")
    print(f"  Mean ITL error:  {mean_itl:.1f}%")
    print(f"\n  E2E < 10%: {below_10}/{len(df)} experiments")
    print(f"\n  Results: {csv_path}")
    print(f"{'='*70}")

    # Save summary JSON
    summary = {
        "mode": args.mode,
        "n_completed": len(df),
        "n_skipped": len(skipped),
        "mean_e2e_error_pct": mean_e2e,
        "mean_ttft_error_pct": mean_ttft,
        "mean_itl_error_pct": mean_itl,
        "experiments_below_10pct": int(below_10),
    }
    summary_path = os.path.join(args.output_dir, f"e2e_summary_{args.mode}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
