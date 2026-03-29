#!/usr/bin/env python3
"""
Run ablation experiments for iteration 2 hypothesis validation.

This script tests H-ablation-1 (β₇ importance) and H-ablation-2 (β₈ importance)
by removing each term and re-optimizing to measure the impact.
"""

import sys
import json
import yaml
import subprocess
import shutil
from pathlib import Path

def run_ablation(ablation_name: str, beta_index: int, num_trials: int = 50):
    """Run ablation by removing a specific beta term.

    Args:
        ablation_name: Name of the ablation (e.g., "ablation_beta7")
        beta_index: Index of beta term to remove (0-indexed)
        num_trials: Number of optimization trials

    Returns:
        dict: Optimization results
    """
    iter_dir = Path(__file__).parent.parent
    ablations_dir = Path(__file__).parent
    training_dir = iter_dir.parent

    # Load original bounds
    with open(iter_dir / "coefficient_bounds.yaml") as f:
        bounds = yaml.safe_load(f)

    # Create modified bounds with the target beta fixed to 0
    modified_bounds = bounds.copy()
    original_beta_bounds = modified_bounds["beta_bounds"].copy()
    original_beta_initial = modified_bounds["beta_initial"].copy()

    # Fix the target beta to 0 (set bounds to [0, 0])
    modified_bounds["beta_bounds"][beta_index] = [0.0, 0.0]
    modified_bounds["beta_initial"][beta_index] = 0.0

    # Save modified bounds to ablations directory
    modified_bounds_path = ablations_dir / f"{ablation_name}_bounds.yaml"
    with open(modified_bounds_path, "w") as f:
        yaml.dump(modified_bounds, f, default_flow_style=False)

    print(f"Running {ablation_name}...")
    print(f"  Beta[{beta_index}] fixed to 0.0 (removed)")
    print(f"  Original bounds: {original_beta_bounds[beta_index]}")
    print(f"  Optimization trials: {num_trials}")

    # Need to temporarily replace coefficient_bounds.yaml to run ablation
    original_bounds_path = iter_dir / "coefficient_bounds.yaml"
    backup_bounds_path = iter_dir / "coefficient_bounds.yaml.backup"

    try:
        # Backup original bounds
        shutil.copy(original_bounds_path, backup_bounds_path)

        # Replace with modified bounds
        shutil.copy(modified_bounds_path, original_bounds_path)

        # Run optimization via CLI
        cmd = [
            "python3",
            str(training_dir / "inner_loop_optimize.py"),
            "--iteration", "2",
            "--n-trials", str(num_trials),
            "--no-detailed-eval"
        ]

        result = subprocess.run(cmd, cwd=str(training_dir), capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ERROR: Optimization failed")
            print(result.stderr)
            raise RuntimeError(f"Ablation {ablation_name} failed")

        # Read results from iter2 directory
        results_path = iter_dir / "inner_loop_results.json"
        with open(results_path) as f:
            results = json.load(f)

        # Save to ablations directory
        ablation_results_path = ablations_dir / f"{ablation_name}_results.json"
        with open(ablation_results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Results saved to: {ablation_results_path}")
        print(f"  Overall loss: {results['loss']['overall_loss']:.2f}%")
        print(f"  TTFT RMSE: {results['loss']['ttft_rmse']:.2f}%")
        print(f"  E2E RMSE: {results['loss']['e2e_rmse']:.2f}%")
        print()

        return results

    finally:
        # Always restore original bounds
        if backup_bounds_path.exists():
            shutil.copy(backup_bounds_path, original_bounds_path)
            backup_bounds_path.unlink()


def main():
    ablations_dir = Path(__file__).parent

    print("="*80)
    print("Iteration 2: Ablation Experiments")
    print("="*80)
    print()

    # Load baseline results for comparison
    iter_dir = ablations_dir.parent
    with open(iter_dir / "inner_loop_results.json") as f:
        baseline = json.load(f)

    print("Baseline (full model):")
    print(f"  Overall loss: {baseline['loss']['overall_loss']:.2f}%")
    print(f"  TTFT RMSE: {baseline['loss']['ttft_rmse']:.2f}%")
    print(f"  E2E RMSE: {baseline['loss']['e2e_rmse']:.2f}%")
    print()

    # H-ablation-1: Remove β₇ (very long context overhead)
    # β₇ is at index 7 in the beta array
    print("H-ablation-1: Testing β₇ (very long context overhead) importance")
    print("-" * 80)
    ablation1_results = run_ablation("ablation_beta7", beta_index=7, num_trials=50)

    # H-ablation-2: Remove β₈ (per-request decode overhead)
    # β₈ is at index 8 in the beta array
    print("H-ablation-2: Testing β₈ (per-request decode overhead) importance")
    print("-" * 80)
    ablation2_results = run_ablation("ablation_beta8", beta_index=8, num_trials=50)

    # Compute deltas
    print("="*80)
    print("Ablation Summary")
    print("="*80)
    print()

    print("H-ablation-1 (β₇ removed):")
    print(f"  Overall loss delta: +{ablation1_results['loss']['overall_loss'] - baseline['loss']['overall_loss']:.2f}%")
    print(f"  TTFT RMSE delta: +{ablation1_results['loss']['ttft_rmse'] - baseline['loss']['ttft_rmse']:.2f}%")
    print(f"  E2E RMSE delta: +{ablation1_results['loss']['e2e_rmse'] - baseline['loss']['e2e_rmse']:.2f}%")
    print()

    print("H-ablation-2 (β₈ removed):")
    print(f"  Overall loss delta: +{ablation2_results['loss']['overall_loss'] - baseline['loss']['overall_loss']:.2f}%")
    print(f"  TTFT RMSE delta: +{ablation2_results['loss']['ttft_rmse'] - baseline['loss']['ttft_rmse']:.2f}%")
    print(f"  E2E RMSE delta: +{ablation2_results['loss']['e2e_rmse'] - baseline['loss']['e2e_rmse']:.2f}%")
    print()

    # Save summary
    summary = {
        "baseline": baseline["loss"],
        "ablation_beta7": {
            "loss": ablation1_results["loss"],
            "delta": {
                "overall_loss": ablation1_results['loss']['overall_loss'] - baseline['loss']['overall_loss'],
                "ttft_rmse": ablation1_results['loss']['ttft_rmse'] - baseline['loss']['ttft_rmse'],
                "e2e_rmse": ablation1_results['loss']['e2e_rmse'] - baseline['loss']['e2e_rmse']
            }
        },
        "ablation_beta8": {
            "loss": ablation2_results["loss"],
            "delta": {
                "overall_loss": ablation2_results['loss']['overall_loss'] - baseline['loss']['overall_loss'],
                "ttft_rmse": ablation2_results['loss']['ttft_rmse'] - baseline['loss']['ttft_rmse'],
                "e2e_rmse": ablation2_results['loss']['e2e_rmse'] - baseline['loss']['e2e_rmse']
            }
        }
    }

    summary_path = ablations_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")
    print()
    print("Next step: Use these results to complete iter2-HYPOTHESIS-validation.md")


if __name__ == "__main__":
    main()
