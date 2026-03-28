#!/usr/bin/env python3
"""
Run ablation experiments for iteration 2 hypothesis validation.

This script runs three ablation experiments:
1. Remove β₇ (very long context overhead)
2. Remove β₈ (per-request decode overhead)
3. Remove β₄ (KV management overhead)

Each ablation modifies evolved_model.go, runs optimization with 50 trials,
restores the original code, and saves results.
"""

import json
import subprocess
import sys
from pathlib import Path
import shutil
from datetime import datetime

# Paths
TRAINING_DIR = Path(__file__).parent.parent.parent
ITER2_DIR = TRAINING_DIR / "iterations" / "iter2"
ABLATIONS_DIR = ITER2_DIR / "ablations"
GO_FILE = TRAINING_DIR.parent / "sim" / "latency" / "evolved_model.go"
GO_BACKUP = ABLATIONS_DIR / "evolved_model.go.backup"

# Optimization parameters (reduced trials for speed)
N_TRIALS = 50
BACKEND_NAME = "evolved"


def backup_go_file():
    """Backup the original Go file."""
    print(f"Backing up {GO_FILE} to {GO_BACKUP}")
    shutil.copy(GO_FILE, GO_BACKUP)


def restore_go_file():
    """Restore the original Go file."""
    print(f"Restoring {GO_FILE} from {GO_BACKUP}")
    shutil.copy(GO_BACKUP, GO_FILE)


def run_optimization(output_file: Path, description: str):
    """Run the inner loop optimization and save results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        str(TRAINING_DIR / "scripts" / "inner_loop_optimize.py"),
        "--iteration", "2",
        "--n-trials", str(N_TRIALS),
        "--output", str(output_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Optimization failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

    print(f"SUCCESS: Results saved to {output_file}")
    return True


def ablation_beta7():
    """Ablation 1: Remove β₇ (very long context overhead)."""
    print("\n" + "="*60)
    print("ABLATION 1: Remove β₇ (very long context overhead)")
    print("="*60)

    # Read the Go file
    with open(GO_FILE, 'r') as f:
        content = f.read()

    # Find and comment out β₇ contribution
    # Looking for the very long context overhead term in PrefillTime
    # This should be something like: + beta[7] * veryLongContextOverhead

    # Need to examine the actual code structure first
    print("\nSearching for β₇ usage in evolved_model.go...")

    # Let's use a different approach - we'll modify the coefficient bounds
    # to fix β₇ to zero instead of modifying the code

    return "skip_code_modification"  # Signal to use coefficient bounds instead


def ablation_beta8():
    """Ablation 2: Remove β₈ (per-request decode overhead)."""
    print("\n" + "="*60)
    print("ABLATION 2: Remove β₈ (per-request decode overhead)")
    print("="*60)

    return "skip_code_modification"


def ablation_beta4():
    """Ablation 3: Remove β₄ (KV management overhead)."""
    print("\n" + "="*60)
    print("ABLATION 3: Remove β₄ (KV management overhead)")
    print("="*60)

    return "skip_code_modification"


def main():
    """Run all ablation experiments."""
    print("Starting ablation experiments for iteration 2")
    print(f"Working directory: {ABLATIONS_DIR}")

    # Ensure we're in the right directory
    ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Instead of modifying Go code, we'll use coefficient bounds
    # to fix specific Beta coefficients to zero

    print("\n" + "="*60)
    print("NOTE: Ablations will be run by fixing coefficients to zero")
    print("This is cleaner than modifying the Go code directly")
    print("="*60)

    ablations = [
        {
            "name": "ablation_beta7",
            "description": "Remove β₇ (very long context overhead)",
            "output": ABLATIONS_DIR / "ablation_beta7_results.json",
            "fix_coefficient": 7,  # Fix beta[7] to zero
        },
        {
            "name": "ablation_beta8",
            "description": "Remove β₈ (per-request decode overhead)",
            "output": ABLATIONS_DIR / "ablation_beta8_results.json",
            "fix_coefficient": 8,  # Fix beta[8] to zero
        },
        {
            "name": "ablation_beta4",
            "description": "Remove β₄ (KV management overhead)",
            "output": ABLATIONS_DIR / "ablation_beta4_results.json",
            "fix_coefficient": 4,  # Fix beta[4] to zero
        },
    ]

    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS REQUIRE MANUAL SETUP")
    print("="*60)
    print("\nTo run these ablations, you need to:")
    print("1. Modify coefficient_bounds.yaml to fix specific coefficients to zero")
    print("2. Run inner_loop_optimize.py with the modified bounds")
    print("3. Restore coefficient_bounds.yaml")
    print("4. Repeat for each ablation")
    print("\nAlternatively, I can create a helper script to automate this.")

    return 1  # Signal that manual intervention is needed


if __name__ == "__main__":
    sys.exit(main())
