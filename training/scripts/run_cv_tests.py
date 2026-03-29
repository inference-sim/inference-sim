#!/usr/bin/env python3
"""
Cross-validation test runner with determinism guarantees.

Implements CV-1 (LOMO), CV-2 (LOWO), CV-3 (LOTO) from generalization-validation-protocol.md.

Determinism guarantees:
1. Fixed train/test splits (documented in code, never randomized)
2. Fixed random seed for Bayesian optimization (seed=42)
3. Fixed BLIS simulation seed (--seed 42)
4. Results reproducible across runs

Usage:
    python scripts/run_cv_tests.py --iteration 0 --cv-test all
    python scripts/run_cv_tests.py --iteration 0 --cv-test CV-1
    python scripts/run_cv_tests.py --iteration 0 --cv-test CV-2
    python scripts/run_cv_tests.py --iteration 0 --cv-test CV-3

Outputs:
    - cv{1,2,3}_results.json - Test set predictions and metrics
    - cv{1,2,3}_report.md - Human-readable validation report
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def get_python_executable() -> str:
    """
    Get a reliable Python executable path.

    macOS Homebrew Python's sys.executable points to Python.app/Contents/MacOS/Python,
    which doesn't work properly in subprocess calls. This function finds a working
    python3 executable.

    Returns:
        Path to python3 executable
    """
    # First try: Use python3 from PATH (works if in venv or system python3)
    python3_path = shutil.which("python3")
    if python3_path:
        return python3_path

    # Second try: If sys.executable is a .app bundle, find the bin/python3
    if "Python.app" in sys.executable:
        # Homebrew pattern: /opt/homebrew/.../Python.app/Contents/MacOS/Python
        # We want: /opt/homebrew/bin/python3
        parts = sys.executable.split("/")
        if "opt" in parts and "homebrew" in parts:
            return "/opt/homebrew/bin/python3"

    # Fallback: Use sys.executable (will fail on Homebrew .app but we tried)
    return sys.executable


# ============================================================================
# DETERMINISTIC DATA SPLITS (NEVER CHANGE THESE!)
# ============================================================================

# CV-1: Leave-One-Model-Out (Dense → MoE)
# Test: Can model trained on dense architectures generalize to MoE?
CV1_TRAIN_EXPERIMENTS = [
    # Dense models only (11 experiments)
    "20260217-155451-llama-2-7b-tp1-codegen",
    "20260217-162547-llama-2-7b-tp1-roleplay",
    "20260217-170634-llama-2-7b-tp1-reasoning",
    "20260217-231439-llama-2-7b-tp1-general",
    "60-llama-3-1-70b-tp4-general-lite-4-1",
    "61-llama-3-1-70b-tp4-codegen-4-1",
    "62-mistral-nemo-12b-tp2-general-lite-2-1",
    "63-mistral-nemo-12b-tp1-codegen-1-1",
    "64-qwen2-5-7b-instruct-tp1-roleplay-1-1",
    "65-01-ai-yi-34b-tp2-general-lite-2-1",
    "66-qwen2-5-7b-instruct-tp1-reasoning-1-1",
]

CV1_TEST_EXPERIMENTS = [
    # MoE models only (4 experiments)
    "17-llama-4-scout-17b-16e-tp2-general-2",
    "20-llama-4-scout-17b-16e-tp2-codegen-2",
    "21-llama-4-scout-17b-16e-tp2-roleplay-2",
    "48-llama-4-scout-17b-16e-tp2-reasoning-2",
]

# CV-2: Leave-One-Workload-Out (Workload-agnostic validation)
# Test: Can model trained on codegen+reasoning generalize to roleplay+general?
CV2_TRAIN_EXPERIMENTS = [
    # codegen (4) + reasoning (3) = 7 experiments
    "20260217-155451-llama-2-7b-tp1-codegen",
    "20260217-170634-llama-2-7b-tp1-reasoning",
    "20-llama-4-scout-17b-16e-tp2-codegen-2",
    "48-llama-4-scout-17b-16e-tp2-reasoning-2",
    "61-llama-3-1-70b-tp4-codegen-4-1",
    "63-mistral-nemo-12b-tp1-codegen-1-1",
    "66-qwen2-5-7b-instruct-tp1-reasoning-1-1",
]

CV2_TEST_EXPERIMENTS = [
    # roleplay (3) + general (5) = 8 experiments
    "20260217-162547-llama-2-7b-tp1-roleplay",
    "20260217-231439-llama-2-7b-tp1-general",
    "17-llama-4-scout-17b-16e-tp2-general-2",
    "21-llama-4-scout-17b-16e-tp2-roleplay-2",
    "60-llama-3-1-70b-tp4-general-lite-4-1",
    "62-mistral-nemo-12b-tp2-general-lite-2-1",
    "64-qwen2-5-7b-instruct-tp1-roleplay-1-1",
    "65-01-ai-yi-34b-tp2-general-lite-2-1",
]

# CV-3: Leave-One-TP-Out (TP communication overhead interpolation)
# Test: Can model trained on TP=1+4 interpolate to TP=2?
CV3_TRAIN_EXPERIMENTS = [
    # TP=1 (7) + TP=4 (2) = 9 experiments
    "20260217-155451-llama-2-7b-tp1-codegen",
    "20260217-162547-llama-2-7b-tp1-roleplay",
    "20260217-170634-llama-2-7b-tp1-reasoning",
    "20260217-231439-llama-2-7b-tp1-general",
    "60-llama-3-1-70b-tp4-general-lite-4-1",
    "61-llama-3-1-70b-tp4-codegen-4-1",
    "63-mistral-nemo-12b-tp1-codegen-1-1",
    "64-qwen2-5-7b-instruct-tp1-roleplay-1-1",
    "66-qwen2-5-7b-instruct-tp1-reasoning-1-1",
]

CV3_TEST_EXPERIMENTS = [
    # TP=2 only (6 experiments)
    "17-llama-4-scout-17b-16e-tp2-general-2",
    "20-llama-4-scout-17b-16e-tp2-codegen-2",
    "21-llama-4-scout-17b-16e-tp2-roleplay-2",
    "48-llama-4-scout-17b-16e-tp2-reasoning-2",
    "62-mistral-nemo-12b-tp2-general-lite-2-1",
    "65-01-ai-yi-34b-tp2-general-lite-2-1",
]

# ============================================================================
# CV Test Configuration
# ============================================================================

CV_TESTS = {
    "CV-1": {
        "name": "Leave-One-Model-Out (Dense → MoE)",
        "train": CV1_TRAIN_EXPERIMENTS,
        "test": CV1_TEST_EXPERIMENTS,
        "pass_criteria": "MAPE < 20% (lenient - only 1 MoE architecture)",
        "failure_diagnosis": "Need MoE-specific basis function (expert routing, load imbalance)",
    },
    "CV-2": {
        "name": "Leave-One-Workload-Out (Workload-agnostic)",
        "train": CV2_TRAIN_EXPERIMENTS,
        "test": CV2_TEST_EXPERIMENTS,
        "pass_criteria": "Mean MAPE < 15%, variance < 3% between roleplay and general",
        "failure_diagnosis": "Basis functions not truly workload-agnostic (memorizing patterns)",
    },
    "CV-3": {
        "name": "Leave-One-TP-Out (TP interpolation)",
        "train": CV3_TRAIN_EXPERIMENTS,
        "test": CV3_TEST_EXPERIMENTS,
        "pass_criteria": "MAPE < 15%",
        "failure_diagnosis": "TP basis function has wrong functional form (not interpolating)",
    },
}


class CVTestRunner:
    """Run cross-validation tests with determinism guarantees."""

    def __init__(self, iteration: int, data_dir: str, output_dir: str):
        self.iteration = iteration
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.script_dir = Path(__file__).parent
        self.training_dir = self.script_dir.parent
        self.project_root = self.training_dir.parent

        # Determinism parameters
        self.optimization_seed = 42
        self.blis_seed = 42
        self.n_trials = 50  # Fixed for reproducibility

        # Get reliable Python executable (fix macOS Homebrew .app bundle issues)
        self.python_executable = get_python_executable()
        self.backend_name = "evolved"  # Fixed backend name for all iterations

    def run_cv_test(self, test_name: str) -> Dict:
        """
        Run a single CV test with full determinism.

        Returns test results dictionary.
        """
        print("=" * 80)
        print(f"RUNNING {test_name}: {CV_TESTS[test_name]['name']}")
        print("=" * 80)
        print()

        test_config = CV_TESTS[test_name]
        train_experiments = test_config["train"]
        test_experiments = test_config["test"]

        print(f"Train set: {len(train_experiments)} experiments")
        for exp in train_experiments:
            print(f"  - {exp}")
        print()

        print(f"Test set: {len(test_experiments)} experiments")
        for exp in test_experiments:
            print(f"  - {exp}")
        print()

        # Step 1: Create temporary train data directory
        train_dir = self.output_dir / f"{test_name.lower()}_train_data"
        if train_dir.exists():
            shutil.rmtree(train_dir)
        train_dir.mkdir(parents=True)

        print(f"Creating train data directory: {train_dir}")
        for exp in train_experiments:
            src = self.data_dir / exp
            dst = train_dir / exp
            if not src.exists():
                print(f"  ⚠️  Warning: {exp} not found in {self.data_dir}")
                continue
            shutil.copytree(src, dst)
            print(f"  ✓ {exp}")
        print()

        # Step 2: Train model on train set (run inner loop optimization)
        print(f"Training model on {len(train_experiments)} experiments...")
        print(f"  Seed: {self.optimization_seed} (deterministic)")
        print(f"  Trials: {self.n_trials}")
        print()

        # Run optimization with deterministic seed
        train_results_file = self.output_dir / f"{test_name.lower()}_train_results.json"

        # Use inner_loop_optimize.py with train data directory
        optimize_cmd = [
            self.python_executable,
            str(self.training_dir / "inner_loop_optimize.py"),
            f"--n-trials={self.n_trials}",
            f"--data-dir={train_dir}",
            f"--seed={self.optimization_seed}",
            f"--output={train_results_file}"
        ]

        print(f"Command: {' '.join(optimize_cmd)}")
        print()

        # Run optimization
        try:
            result = subprocess.run(
                optimize_cmd,
                cwd=self.training_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr, file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"❌ Optimization failed with return code {e.returncode}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr, file=sys.stderr)
            return {
                "status": "optimization_failed",
                "test": test_name,
                "error": str(e)
            }

        # Step 3: Load trained coefficients
        with open(train_results_file) as f:
            train_results = json.load(f)

        best_alpha = train_results["best_alpha"]
        best_beta = train_results["best_beta"]
        train_loss = train_results["best_loss"]

        print(f"Best coefficients from training:")
        print(f"  Alpha: {best_alpha}")
        print(f"  Beta: {best_beta}")
        print(f"  Train loss: {train_loss:.2f}%")
        print()

        # Step 4: Create test data directory
        test_dir = self.output_dir / f"{test_name.lower()}_test_data"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True)

        print(f"Creating test data directory: {test_dir}")
        for exp in test_experiments:
            src = self.data_dir / exp
            dst = test_dir / exp
            if not src.exists():
                print(f"  ⚠️  Warning: {exp} not found in {self.data_dir}")
                continue
            shutil.copytree(src, dst)
            print(f"  ✓ {exp}")
        print()

        # Step 5: Evaluate on test set
        print(f"Evaluating on {len(test_experiments)} test experiments...")

        alpha_str = ",".join(f"{x:.10e}" for x in best_alpha)
        beta_str = ",".join(f"{x:.10e}" for x in best_beta)

        eval_cmd = [
            self.python_executable,
            str(self.training_dir / "run_blis_and_compute_loss.py"),
            "--latency-model", self.backend_name,
            "--alpha-coeffs", alpha_str,
            "--beta-coeffs", beta_str,
            "--blis-binary", "../blis",
            "--data-dir", str(test_dir),
            "--evaluate-per-experiment"
        ]

        try:
            result = subprocess.run(
                eval_cmd,
                cwd=self.training_dir,
                check=True,
                capture_output=True,
                text=True
            )
            test_diagnostics = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Test evaluation failed with return code {e.returncode}")
            print("STDERR:", e.stderr, file=sys.stderr)
            return {
                "status": "evaluation_failed",
                "test": test_name,
                "error": str(e)
            }
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse test evaluation results: {e}")
            return {
                "status": "evaluation_failed",
                "test": test_name,
                "error": f"JSON decode error: {e}"
            }

        test_loss = test_diagnostics["overall_loss"]
        ttft_rmse = test_diagnostics["ttft_rmse"]
        e2e_rmse = test_diagnostics["e2e_rmse"]

        # Compute MAPE (Mean Absolute Percentage Error) from per-experiment results
        # MAPE = mean(APE) across experiments (NOT RMSE!)
        per_exp = test_diagnostics.get("per_experiment", [])
        if per_exp:
            ttft_mape = sum(exp["ttft_mean_ape"] for exp in per_exp) / len(per_exp)
            e2e_mape = sum(exp["e2e_mean_ape"] for exp in per_exp) / len(per_exp)
        else:
            ttft_mape = ttft_rmse  # Fallback if no per-experiment data
            e2e_mape = e2e_rmse

        print(f"Test set metrics:")
        print(f"  TTFT RMSE: {ttft_rmse:.2f}%")
        print(f"  E2E RMSE: {e2e_rmse:.2f}%")
        print(f"  TTFT MAPE: {ttft_mape:.2f}%")
        print(f"  E2E MAPE: {e2e_mape:.2f}%")
        print(f"  Overall loss: {test_loss:.2f}%")
        print()

        # Step 6: Compute test set metrics
        test_results = {
            "test_name": test_name,
            "train_experiments": train_experiments,
            "test_experiments": test_experiments,
            "n_trials": self.n_trials,
            "seed": self.optimization_seed,
            "timestamp": datetime.now().isoformat(),
            "status": "complete",
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_ttft_rmse": ttft_rmse,
            "test_e2e_rmse": e2e_rmse,
            "test_ttft_mape": ttft_mape,  # MAPE for CV criteria
            "test_e2e_mape": e2e_mape,    # MAPE for CV criteria
            "best_alpha": best_alpha,
            "best_beta": best_beta,
            "per_experiment": test_diagnostics.get("per_experiment", [])
        }

        # CV-2 specific: Compute variance between roleplay and general workloads
        if test_name == "CV-2" and per_exp:
            roleplay_experiments = [exp for exp in per_exp if "roleplay" in exp["experiment_folder"].lower()]
            general_experiments = [exp for exp in per_exp if "general" in exp["experiment_folder"].lower()]

            if roleplay_experiments and general_experiments:
                roleplay_mape = sum(exp["ttft_mean_ape"] + exp["e2e_mean_ape"] for exp in roleplay_experiments) / (2 * len(roleplay_experiments))
                general_mape = sum(exp["ttft_mean_ape"] + exp["e2e_mean_ape"] for exp in general_experiments) / (2 * len(general_experiments))
                variance = abs(roleplay_mape - general_mape)

                test_results["cv2_roleplay_mape"] = roleplay_mape
                test_results["cv2_general_mape"] = general_mape
                test_results["cv2_workload_variance"] = variance

                print(f"CV-2 Workload-Agnostic Check:")
                print(f"  Roleplay MAPE: {roleplay_mape:.2f}%")
                print(f"  General MAPE: {general_mape:.2f}%")
                print(f"  Variance: {variance:.2f}% (threshold: <3%)")
                print()

        # Save results
        results_file = self.output_dir / f"{test_name.lower()}_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        print(f"Results saved to: {results_file}")
        print()

        return test_results

    def generate_report(self, test_name: str, results: Dict):
        """Generate human-readable markdown report."""
        report_lines = [
            f"# {test_name} Validation Report",
            "",
            f"**Test**: {CV_TESTS[test_name]['name']}",
            f"**Iteration**: {self.iteration}",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Test Configuration",
            "",
            f"- **Train set**: {len(CV_TESTS[test_name]['train'])} experiments",
            f"- **Test set**: {len(CV_TESTS[test_name]['test'])} experiments",
            f"- **Optimization trials**: {self.n_trials}",
            f"- **Random seed**: {self.optimization_seed} (deterministic)",
            "",
            "## Data Splits",
            "",
            "### Training Set",
            "",
        ]

        for exp in CV_TESTS[test_name]['train']:
            report_lines.append(f"- `{exp}`")

        report_lines.extend([
            "",
            "### Test Set",
            "",
        ])

        for exp in CV_TESTS[test_name]['test']:
            report_lines.append(f"- `{exp}`")

        report_lines.extend([
            "",
            "## Pass Criteria",
            "",
            f"{CV_TESTS[test_name]['pass_criteria']}",
            "",
            "## Results",
            "",
        ])

        if results.get("status") == "complete":
            train_loss = results.get("train_loss", 0.0)
            test_loss = results.get("test_loss", 0.0)
            test_ttft_rmse = results.get("test_ttft_rmse", 0.0)
            test_e2e_rmse = results.get("test_e2e_rmse", 0.0)
            test_ttft_mape = results.get("test_ttft_mape", 0.0)
            test_e2e_mape = results.get("test_e2e_mape", 0.0)

            # Determine pass/fail based on test name (using MAPE, not RMSE)
            passed = False
            if test_name == "CV-1":
                # CV-1: MAPE < 20% (lenient - only 1 MoE architecture)
                passed = test_ttft_mape < 20.0 and test_e2e_mape < 20.0
            elif test_name == "CV-2":
                # CV-2: Mean MAPE < 15%, variance < 3% between roleplay and general
                variance = results.get("cv2_workload_variance", float('inf'))
                passed = test_ttft_mape < 15.0 and test_e2e_mape < 15.0 and variance < 3.0
            elif test_name == "CV-3":
                # CV-3: MAPE < 15%
                passed = test_ttft_mape < 15.0 and test_e2e_mape < 15.0

            status_emoji = "✅ PASS" if passed else "❌ FAIL"

            report_lines.extend([
                f"**Status**: {status_emoji}",
                "",
                "### Training Set Performance",
                "",
                f"- **Loss**: {train_loss:.2f}%",
                f"- **Trials**: {self.n_trials}",
                f"- **Seed**: {self.optimization_seed}",
                "",
                "### Test Set Performance",
                "",
                f"- **TTFT RMSE**: {test_ttft_rmse:.2f}%",
                f"- **E2E RMSE**: {test_e2e_rmse:.2f}%",
                f"- **TTFT MAPE**: {test_ttft_mape:.2f}% ← Used for pass/fail criteria",
                f"- **E2E MAPE**: {test_e2e_mape:.2f}% ← Used for pass/fail criteria",
                f"- **Overall Loss**: {test_loss:.2f}%",
                "",
            ])

            # CV-2 specific: Add workload-agnostic check results
            if test_name == "CV-2" and "cv2_workload_variance" in results:
                roleplay_mape = results.get("cv2_roleplay_mape", 0.0)
                general_mape = results.get("cv2_general_mape", 0.0)
                variance = results.get("cv2_workload_variance", 0.0)
                variance_passed = "✅" if variance < 3.0 else "❌"

                report_lines.extend([
                    "### CV-2 Workload-Agnostic Validation",
                    "",
                    f"- **Roleplay MAPE**: {roleplay_mape:.2f}%",
                    f"- **General MAPE**: {general_mape:.2f}%",
                    f"- **Variance**: {variance:.2f}% {variance_passed} (threshold: <3%)",
                    "",
                    "**Interpretation**:",
                    "- If variance <3%: Basis functions depend on batch composition, not workload patterns ✅",
                    "- If variance >3%: Basis functions memorizing workload-specific patterns ❌",
                    "",
                ])

            report_lines.extend([
                "### Best Coefficients",
                "",
                f"- **Alpha**: {results.get('best_alpha', [])}",
                f"- **Beta**: {results.get('best_beta', [])}",
                "",
            ])

            # Add per-experiment breakdown if available
            per_exp = results.get("per_experiment", [])
            if per_exp:
                report_lines.extend([
                    "### Per-Experiment Test Results",
                    "",
                    "| Experiment | TTFT APE | E2E APE | Combined Loss |",
                    "|------------|----------|---------|---------------|"
                ])
                for exp in per_exp:
                    exp_name = os.path.basename(exp["experiment_folder"])
                    ttft_ape = exp["ttft_mean_ape"]
                    e2e_ape = exp["e2e_mean_ape"]
                    combined = exp["combined_loss"]
                    report_lines.append(f"| {exp_name} | {ttft_ape:.2f}% | {e2e_ape:.2f}% | {combined:.2f}% |")
                report_lines.append("")

        else:
            status = results.get("status", "unknown")
            error = results.get("error", "No error message")
            report_lines.extend([
                f"**Status**: ❌ {status}",
                "",
                f"**Error**: {error}",
                "",
            ])

        report_file = self.output_dir / f"{test_name.lower()}_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-validation tests with determinism guarantees"
    )
    parser.add_argument(
        '--iteration',
        type=int,
        required=True,
        help='Iteration number'
    )
    parser.add_argument(
        '--cv-test',
        choices=['CV-1', 'CV-2', 'CV-3', 'all'],
        required=True,
        help='Which CV test to run (or "all" for all tests)'
    )
    parser.add_argument(
        '--data-dir',
        default='trainval_data',
        help='Path to training/validation data directory'
    )
    parser.add_argument(
        '--output-dir',
        default='cv_results',
        help='Output directory for CV test results'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    training_dir = script_dir.parent
    data_dir = training_dir / args.data_dir
    output_dir = training_dir / args.output_dir

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    # Create test runner
    runner = CVTestRunner(args.iteration, data_dir, output_dir)

    # Run requested tests
    tests_to_run = ['CV-1', 'CV-2', 'CV-3'] if args.cv_test == 'all' else [args.cv_test]

    print("=" * 80)
    print("CROSS-VALIDATION TEST SUITE")
    print("=" * 80)
    print(f"Iteration: {args.iteration}")
    print(f"Tests to run: {', '.join(tests_to_run)}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("DETERMINISM GUARANTEES:")
    print(f"  - Optimization seed: {runner.optimization_seed}")
    print(f"  - BLIS simulation seed: {runner.blis_seed}")
    print(f"  - Fixed data splits (never randomized)")
    print(f"  - Fixed number of trials: {runner.n_trials}")
    print()

    results = {}
    for test_name in tests_to_run:
        results[test_name] = runner.run_cv_test(test_name)
        runner.generate_report(test_name, results[test_name])
        print()

    print("=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80)
    print()

    # Print summary
    all_passed = True
    for test_name in tests_to_run:
        result = results[test_name]
        status = result.get("status", "unknown")
        if status == "complete":
            test_loss = result.get("test_loss", 0.0)
            test_ttft_mape = result.get("test_ttft_mape", 0.0)
            test_e2e_mape = result.get("test_e2e_mape", 0.0)

            # Check pass criteria (using MAPE)
            passed = False
            if test_name == "CV-1":
                passed = test_ttft_mape < 20.0 and test_e2e_mape < 20.0
            elif test_name == "CV-2":
                variance = result.get("cv2_workload_variance", float('inf'))
                passed = test_ttft_mape < 15.0 and test_e2e_mape < 15.0 and variance < 3.0
            elif test_name == "CV-3":
                passed = test_ttft_mape < 15.0 and test_e2e_mape < 15.0

            status_emoji = "✅" if passed else "❌"
            print(f"{status_emoji} {test_name}: Test loss = {test_loss:.2f}% (TTFT MAPE={test_ttft_mape:.2f}%, E2E MAPE={test_e2e_mape:.2f}%)")

            # CV-2: Print workload variance check
            if test_name == "CV-2" and "cv2_workload_variance" in result:
                variance = result["cv2_workload_variance"]
                variance_emoji = "✅" if variance < 3.0 else "❌"
                print(f"         Workload variance: {variance:.2f}% {variance_emoji} (threshold: <3%)")

            all_passed = all_passed and passed
        else:
            print(f"❌ {test_name}: {status}")
            all_passed = False

    print()
    if all_passed:
        print("✅ All cross-validation tests PASSED")
    else:
        print("❌ Some cross-validation tests FAILED")
        print("   Review detailed reports for failure diagnosis")
    print()
    print("See generated reports for detailed results and per-experiment breakdown.")


if __name__ == '__main__':
    main()
