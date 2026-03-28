#!/usr/bin/env python3
"""
Run ablation experiment by temporarily swapping bounds file.

Usage:
    python run_ablation.py --iteration 1 --ablation chunking --n-trials 50
"""

import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path


def run_ablation(iteration: int, ablation_name: str, n_trials: int = 50):
    """Run ablation experiment with temporary bounds file swap."""

    # Get training root (3 levels up from ablations: ablations -> iter1 -> iterations -> training)
    training_root = Path(__file__).parent.parent.parent
    iteration_dir = training_root / f"iterations/iter{iteration}"
    ablations_dir = iteration_dir / "ablations"

    # Paths
    original_bounds = iteration_dir / "coefficient_bounds.yaml"
    ablation_bounds = ablations_dir / f"ablation_no_{ablation_name}_bounds.yaml"
    backup_bounds = iteration_dir / "coefficient_bounds.yaml.backup"

    # Verify files exist
    if not original_bounds.exists():
        print(f"Error: Original bounds not found: {original_bounds}")
        return 1

    if not ablation_bounds.exists():
        print(f"Error: Ablation bounds not found: {ablation_bounds}")
        return 1

    print(f"\n{'='*60}")
    print(f"Running Ablation: {ablation_name}")
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
        print(f"3. Running optimization (this may take a while)...")
        result = subprocess.run(
            [
                sys.executable,
                "inner_loop_optimize.py",
                "--iteration", str(iteration),
                "--n-trials", str(n_trials),
                "--seed", "42"  # Same seed for reproducibility
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
        ablation_results = ablations_dir / f"ablation_no_{ablation_name}_results.json"

        if results_file.exists():
            shutil.move(results_file, ablation_results)
            print(f"   Saved: {ablation_results}")
        else:
            print(f"Warning: Results file not found: {results_file}")

        print(f"\n✅ Ablation '{ablation_name}' complete!\n")
        return 0

    finally:
        # Always restore original bounds
        print(f"5. Restoring original bounds...")
        if backup_bounds.exists():
            shutil.copy(backup_bounds, original_bounds)
            backup_bounds.unlink()  # Remove backup
            print(f"   Restored: {original_bounds}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiment")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument("--ablation", type=str, required=True,
                       choices=["chunking", "tp_comm", "kv_mgmt"],
                       help="Ablation to run (chunking, tp_comm, or kv_mgmt)")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of optimization trials (default: 50)")

    args = parser.parse_args()

    sys.exit(run_ablation(args.iteration, args.ablation, args.n_trials))


if __name__ == "__main__":
    main()
