#!/usr/bin/env python3
"""
Run single ablation experiment by temporarily swapping bounds file.
Based on iter1/ablations/run_ablation.py pattern.
"""

import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path


def run_ablation(iteration: int, beta_index: int, beta_name: str, n_trials: int = 30):
    """Run ablation experiment with temporary bounds file swap."""

    # Get training root
    training_root = Path(__file__).parent.parent.parent
    iteration_dir = training_root / f"iterations/iter{iteration}"
    ablations_dir = iteration_dir / "ablations"

    # Paths
    original_bounds = iteration_dir / "coefficient_bounds.yaml"
    ablation_bounds = ablations_dir / f"ablation_beta{beta_index}_bounds.yaml"
    backup_bounds = iteration_dir / "coefficient_bounds.yaml.backup"

    # Create ablation bounds if it doesn't exist
    if not ablation_bounds.exists():
        import yaml
        with open(original_bounds) as f:
            bounds = yaml.safe_load(f)

        # Fix the target beta to 0
        bounds["beta_bounds"][beta_index] = [0.0, 0.0]
        bounds["beta_initial"][beta_index] = 0.0

        with open(ablation_bounds, "w") as f:
            yaml.dump(bounds, f, default_flow_style=False)

        print(f"Created ablation bounds: {ablation_bounds}")

    print(f"\n{'='*60}")
    print(f"Running Ablation: β{beta_index} ({beta_name})")
    print(f"Iteration: {iteration}")
    print(f"Trials: {n_trials}")
    print(f"{'='*60}\n")

    try:
        # Backup original bounds
        print(f"1. Backing up original bounds...")
        shutil.copy(original_bounds, backup_bounds)

        # Swap in ablation bounds
        print(f"2. Installing ablation bounds...")
        shutil.copy(ablation_bounds, original_bounds)

        # Run optimization with reduced trials
        print(f"3. Running optimization...")
        result = subprocess.run(
            [
                sys.executable,
                "inner_loop_optimize.py",
                "--iteration", str(iteration),
                "--n-trials", str(n_trials),
                "--seed", "42",  # Same seed for reproducibility
                "--no-detailed-eval"  # Skip detailed eval for speed
            ],
            cwd=training_root,
            capture_output=False,  # Show output in real-time
            text=True
        )

        if result.returncode != 0:
            print(f"Error: Optimization failed with return code {result.returncode}")
            return result.returncode

        # Move results to ablations directory
        print(f"4. Moving results to ablations directory...")
        results_file = iteration_dir / "inner_loop_results.json"
        ablation_results = ablations_dir / f"ablation_beta{beta_index}_results.json"

        if results_file.exists():
            shutil.move(results_file, ablation_results)
            print(f"   Saved: {ablation_results}")
        else:
            print(f"Warning: Results file not found: {results_file}")

        print(f"\n✅ Ablation 'β{beta_index}' complete!\n")
        return 0

    finally:
        # Always restore original bounds
        print(f"5. Restoring original bounds...")
        if backup_bounds.exists():
            shutil.copy(backup_bounds, original_bounds)
            backup_bounds.unlink()  # Remove backup
            print(f"   Restored: {original_bounds}")


def main():
    parser = argparse.ArgumentParser(description="Run single ablation experiment")
    parser.add_argument("--iteration", type=int, default=2, help="Iteration number")
    parser.add_argument("--beta-index", type=int, required=True,
                       help="Beta coefficient index to ablate (0-indexed)")
    parser.add_argument("--beta-name", type=str, required=True,
                       help="Name of beta coefficient (for logging)")
    parser.add_argument("--n-trials", type=int, default=30,
                       help="Number of optimization trials (default: 30)")

    args = parser.parse_args()

    sys.exit(run_ablation(args.iteration, args.beta_index, args.beta_name, args.n_trials))


if __name__ == "__main__":
    main()
