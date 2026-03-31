#!/usr/bin/env python3
"""
Inner Loop Optimizer for Agentic Latency Training

Reads outer loop deliverables (manifest, bounds), compiles BLIS,
and runs Bayesian optimization to find optimal (α, β) coefficients.
"""

import os
import sys
import json
import subprocess
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import yaml
import optuna
from optuna.trial import Trial


class InnerLoopOptimizer:
    """
    Inner loop Bayesian optimization for latency model coefficients.

    Responsibilities:
    1. Setup: Verify outer loop deliverables and compile BLIS
    2. Optimize: Run Bayesian optimization over coefficient space
    3. Evaluate: Post-convergence detailed evaluation
    """

    def __init__(self,
                 iteration: int,
                 n_trials: int = 1000,
                 timeout_per_trial: int = 120,
                 data_dir: str = "trainval_data",
                 seed: int = 42):
        """
        Initialize inner loop optimizer.

        Args:
            iteration: Iteration number (used to locate files in iterations/iter{N}/)
            n_trials: Number of Bayesian optimization trials (default: 1000)
            timeout_per_trial: Timeout in seconds for each BLIS run
            data_dir: Directory containing ground-truth experiments (for CV: subset of trainval_data)
            seed: Random seed for Bayesian optimization (determinism guarantee)
        """
        self.iteration = iteration
        self.iteration_dir = f"iterations/iter{iteration}"
        self.manifest_path = os.path.join(self.iteration_dir, "iteration_manifest.yaml")
        self.bounds_path = os.path.join(self.iteration_dir, "coefficient_bounds.yaml")
        self.n_trials = n_trials
        self.timeout_per_trial = timeout_per_trial
        self.data_dir = data_dir
        self.seed = seed

        self.backend_name = None
        self.bounds = None
        self.error_log = []  # Track failed trials

        # Get reliable Python executable (fix macOS Homebrew .app bundle issues)
        self.python_executable = self._get_python_executable()

    def _get_python_executable(self) -> str:
        """
        Get a reliable Python executable path.

        macOS Homebrew Python's sys.executable points to Python.app/Contents/MacOS/Python,
        which doesn't work properly in subprocess calls. This method finds a working
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

    def setup(self) -> None:
        """
        Setup phase: Verify deliverables, compile BLIS, load bounds.

        Raises:
            FileNotFoundError: If required files are missing
            RuntimeError: If compilation fails
        """
        print("=" * 70)
        print("INNER LOOP SETUP")
        print("=" * 70)

        # 0. Verify run_blis_and_compute_loss.py exists
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.run_blis_script = os.path.join(self.script_dir, "run_blis_and_compute_loss.py")
        if not os.path.exists(self.run_blis_script):
            raise FileNotFoundError(
                f"run_blis_and_compute_loss.py not found at {self.run_blis_script}"
            )

        # 1. Read iteration manifest
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"Iteration manifest not found: {self.manifest_path}\n"
                f"Outer loop must provide this file."
            )

        with open(self.manifest_path) as f:
            manifest = yaml.safe_load(f)

        self.backend_name = manifest["latency_backend_name"]
        self.iteration = manifest["iteration"]
        modified_files = manifest["modified_files"]
        reasoning = manifest["reasoning"]

        print(f"\nIteration: {self.iteration}")
        print(f"Backend: {self.backend_name}")
        print(f"Reasoning: {reasoning}")
        print(f"Modified files: {modified_files}")

        # 2. Verify declared files exist (relative to project root)
        print("\nVerifying declared files...")
        self.project_root = os.path.dirname(self.script_dir)
        for filepath in modified_files:
            full_path = os.path.join(self.project_root, filepath)
            if not os.path.exists(full_path):
                raise FileNotFoundError(
                    f"Agent declared {filepath} but file not found at {full_path}.\n"
                    f"Check outer loop output."
                )
            print(f"  ✓ {filepath}")

        # 2.5. Run pre-flight validation (checks backend registration and integration)
        print("\nRunning pre-flight validation...")
        validate_script = os.path.join(self.script_dir, "scripts", "validate_backend.py")
        if os.path.exists(validate_script):
            try:
                result = subprocess.run(
                    [sys.executable, validate_script, self.backend_name, '--iteration', str(self.iteration)],
                    cwd=self.script_dir,
                    capture_output=True,
                    timeout=60
                )
                # Print validation output
                print(result.stdout.decode('utf-8', errors='replace'))
                if result.returncode != 0:
                    print(result.stderr.decode('utf-8', errors='replace'))
                    raise RuntimeError(
                        f"Pre-flight validation failed for backend '{self.backend_name}'.\n"
                        f"Fix the integration issues listed above before retrying."
                    )
            except subprocess.TimeoutExpired:
                print("  ⚠ Validation script timed out - skipping (proceeding with caution)")
        else:
            print(f"  ⚠ Validation script not found at {validate_script} - skipping")

        # 3. Load coefficient bounds
        if not os.path.exists(self.bounds_path):
            raise FileNotFoundError(
                f"Coefficient bounds not found: {self.bounds_path}\n"
                f"Outer loop must provide this file."
            )

        with open(self.bounds_path) as f:
            self.bounds = yaml.safe_load(f)

        n_alpha = len(self.bounds["alpha_bounds"])
        n_beta = len(self.bounds["beta_bounds"])

        if n_alpha != 3:
            raise ValueError(
                f"Expected exactly 3 alpha bounds, got {n_alpha}.\n"
                f"Alpha must be [α₀, α₁, α₂]."
            )

        # Validate non-negative bounds (BLIS requirement: all coefficients >= 0)
        for i, (low, high) in enumerate(self.bounds["alpha_bounds"]):
            if low < 0:
                raise ValueError(
                    f"Alpha bound {i} has negative lower bound: {low}. "
                    f"All alpha coefficients must be non-negative."
                )
        for i, (low, high) in enumerate(self.bounds["beta_bounds"]):
            if low < 0:
                raise ValueError(
                    f"Beta bound {i} has negative lower bound: {low}. "
                    f"All beta coefficients must be non-negative."
                )

        print(f"\nCoefficient bounds loaded:")
        print(f"  Alpha: {n_alpha} coefficients")
        print(f"  Beta: {n_beta} coefficients")

        print(f"\nPython executable: {self.python_executable}")

        # 4. Compile BLIS binary with new latency backend
        print("\nCompiling BLIS...")
        compile_start = time.time()

        try:
            result = subprocess.run(
                ["go", "build", "-o", "blis", "main.go"],
                capture_output=True,
                check=True,
                timeout=60,
                cwd=self.project_root  # Compile from project root
            )
            compile_time = time.time() - compile_start
            print(f"  ✓ Compilation successful ({compile_time:.1f}s)")

        except subprocess.CalledProcessError as e:
            print("\n" + "=" * 70)
            print("COMPILATION FAILED")
            print("=" * 70)
            print("\nSTDOUT:")
            print(e.stdout.decode())
            print("\nSTDERR:")
            print(e.stderr.decode())
            print("=" * 70)
            raise RuntimeError(
                f"Agent generated invalid Go code. See compilation errors above."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Compilation timeout (>60s). Check for infinite loops.")

        # 5. Verify binary exists
        self.blis_binary = os.path.join(self.project_root, "blis")
        if not os.path.exists(self.blis_binary):
            raise RuntimeError(f"BLIS binary not found at {self.blis_binary} after compilation.")

        print(f"\nSetup complete. Ready to optimize {self.backend_name}.")
        print("=" * 70)

    def objective_function(self, alpha: List[float], beta: List[float]) -> float:
        """
        Evaluate loss for given coefficients.

        Args:
            alpha: Alpha coefficients [α₀, α₁, α₂]
            beta: Beta coefficients [β₁, ..., βₙ]

        Returns:
            Loss value (RMSE[APE(TTFT)] + RMSE[APE(E2E)])

        Raises:
            RuntimeError: If evaluation fails
        """
        # Format coefficients as comma-separated strings
        alpha_str = ",".join(f"{x:.10e}" for x in alpha)
        beta_str = ",".join(f"{x:.10e}" for x in beta)

        # Run run_blis_and_compute_loss.py with injected coefficients
        try:
            result = subprocess.run([
                self.python_executable,  # Use reliable Python interpreter
                self.run_blis_script,  # Absolute path to script
                "--latency-model", self.backend_name,
                "--alpha-coeffs", alpha_str,
                "--beta-coeffs", beta_str,
                "--blis-binary", "../blis",
                "--data-dir", self.data_dir
            ],
            capture_output=True,
            check=True,
            timeout=self.timeout_per_trial,
            text=True,
            cwd=self.script_dir  # Run from training/ directory
            )

            # Parse loss from JSON output
            output = json.loads(result.stdout)
            loss = output["overall_loss"]

            return loss

        except subprocess.CalledProcessError as e:
            # Evaluation crashed - return penalty loss
            error_msg = f"Crash: alpha={alpha}, beta={beta}, stderr={e.stderr[:200] if e.stderr else 'N/A'}"
            self.error_log.append(error_msg)
            print(f"\n⚠ {error_msg}")
            return 1e6  # Penalty loss

        except subprocess.TimeoutExpired:
            # Timeout - bad coefficients causing hang
            error_msg = f"Timeout: alpha={alpha}, beta={beta}"
            self.error_log.append(error_msg)
            print(f"\n⚠ {error_msg}")
            return 1e6  # Penalty loss

        except (json.JSONDecodeError, KeyError) as e:
            # Malformed output
            error_msg = f"Parse error: {e}"
            self.error_log.append(error_msg)
            print(f"\n⚠ {error_msg}")
            return 1e6  # Penalty loss

    def optuna_objective(self, trial: Trial) -> float:
        """
        Optuna trial objective function.

        Args:
            trial: Optuna trial object

        Returns:
            Loss value for this trial
        """
        # Sample alpha coefficients (always 3)
        alpha = [
            trial.suggest_float(f"alpha_{i}", *self.bounds["alpha_bounds"][i])
            for i in range(3)
        ]

        # Sample beta coefficients (variable count)
        beta = [
            trial.suggest_float(f"beta_{i}", *self.bounds["beta_bounds"][i])
            for i in range(len(self.bounds["beta_bounds"]))
        ]

        return self.objective_function(alpha, beta)

    def optimize(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization to find optimal coefficients.

        Returns:
            dict with keys:
                - best_alpha: Optimal alpha coefficients
                - best_beta: Optimal beta coefficients
                - best_loss: Final loss value
                - n_trials: Number of trials run
                - optimization_time: Wall-clock time in seconds
        """
        print("\n" + "=" * 70)
        print(f"BAYESIAN OPTIMIZATION ({self.n_trials} trials)")
        print(f"Random seed: {self.seed} (deterministic)")
        print("=" * 70)

        # Create Optuna study with deterministic seed
        study = optuna.create_study(
            direction="minimize",
            study_name=f"iter{self.iteration}_{self.backend_name}",
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )

        # Enqueue initial trial if suggested values provided (warm-start optimization)
        if "alpha_initial" in self.bounds and "beta_initial" in self.bounds:
            alpha_init = self.bounds["alpha_initial"]
            beta_init = self.bounds["beta_initial"]

            # Validate initial values length
            if len(alpha_init) == 3 and len(beta_init) == len(self.bounds["beta_bounds"]):
                initial_params = {}
                for i in range(3):
                    initial_params[f"alpha_{i}"] = alpha_init[i]
                for i in range(len(beta_init)):
                    initial_params[f"beta_{i}"] = beta_init[i]

                study.enqueue_trial(initial_params)
                print(f"  Warm-start: Enqueued initial trial with suggested values")
            else:
                print(f"  Warning: Initial values have wrong length, skipping warm-start")

        # Run optimization
        opt_start = time.time()
        study.optimize(
            self.optuna_objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        opt_time = time.time() - opt_start

        # Extract best parameters
        best_params = study.best_params
        best_alpha = [best_params[f"alpha_{i}"] for i in range(3)]
        best_beta = [best_params[f"beta_{i}"] for i in range(len(self.bounds["beta_bounds"]))]
        best_loss = study.best_value
        actual_trials = len(study.trials)

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"\nBest loss: {best_loss:.6f}")
        print(f"Best alpha: {best_alpha}")
        print(f"Best beta: {best_beta}")
        print(f"Trials completed: {actual_trials}/{self.n_trials}")
        print(f"Optimization time: {opt_time:.1f}s")
        print(f"Time per trial: {opt_time / actual_trials:.2f}s")

        return {
            "best_alpha": best_alpha,
            "best_beta": best_beta,
            "best_loss": best_loss,
            "n_trials": actual_trials,
            "optimization_time": opt_time
        }

    def evaluate_detailed(self, alpha: List[float], beta: List[float]) -> Dict[str, Any]:
        """
        Post-convergence detailed evaluation with per-experiment metrics.

        Args:
            alpha: Optimal alpha coefficients
            beta: Optimal beta coefficients

        Returns:
            dict with detailed per-experiment diagnostics
        """
        print("\n" + "=" * 70)
        print("POST-CONVERGENCE EVALUATION")
        print("=" * 70)

        alpha_str = ",".join(f"{x:.10e}" for x in alpha)
        beta_str = ",".join(f"{x:.10e}" for x in beta)

        # Run with --evaluate-per-experiment flag
        result = subprocess.run([
            self.python_executable,  # Use reliable Python interpreter
            self.run_blis_script,  # Absolute path to script
            "--latency-model", self.backend_name,
            "--alpha-coeffs", alpha_str,
            "--beta-coeffs", beta_str,
            "--blis-binary", "../blis",
            "--data-dir", self.data_dir,
            "--evaluate-per-experiment"  # Detailed diagnostics
        ],
        capture_output=True,
        check=True,
        timeout=self.timeout_per_trial * 2,  # Detailed eval may be slower
        text=True,
        cwd=self.script_dir  # Run from training/ directory
        )

        diagnostics = json.loads(result.stdout)

        print("\nPer-experiment metrics:")
        if "per_experiment" in diagnostics:
            for exp in diagnostics["per_experiment"]:
                exp_name = exp.get("experiment_folder", "unknown")
                ttft_ape = exp.get("ttft_mean_ape", 0.0)
                e2e_ape = exp.get("e2e_mean_ape", 0.0)
                print(f"  {exp_name}: TTFT APE={ttft_ape:.3f}%, E2E APE={e2e_ape:.3f}%")

        return diagnostics

    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save optimization results to JSON file in iteration directory.

        Args:
            results: Results dictionary from optimize()

        Output JSON structure:
        {
          "iteration": N,
          "backend_name": "evolved",
          "timestamp": "ISO-8601 timestamp",

          # Optimization metadata
          "optimization": {
            "n_trials": int,  # Number of trials run
            "optimization_time_seconds": float,
            "num_errors": int  # Failed trials (penalty loss)
          },

          # Best coefficients found
          "best_params": {
            "alpha": [α₀, α₁, α₂],  # Request-level: fixed overhead, per-input-token, per-output-token
            "beta": [β₀, β₁, ..., βₙ]  # Step-level: basis function coefficients
          },

          # Overall loss (RMSE across experiments)
          "loss": {
            "overall_loss": float,  # Sum of ttft_rmse + e2e_rmse
            "ttft_rmse": float,     # RMSE[APE(mean_TTFT_per_exp)] - root mean square of APE across 15 experiments
            "e2e_rmse": float       # RMSE[APE(mean_E2E_per_exp)] - root mean square of APE across 15 experiments
          },

          # Per-experiment diagnostics (only if detailed eval ran)
          "per_experiment_results": [
            {
              "experiment_folder": str,
              "model": str,
              "workload": str,
              "ttft_mean_ape": float,  # Absolute percentage error for TTFT mean (this experiment only)
              "e2e_mean_ape": float,   # Absolute percentage error for E2E mean (this experiment only)
              "latency_ape": {...},    # Detailed latency APEs (mean/p90/p99 for TTFT/E2E/ITL)
              "throughput_ape": {...}  # Throughput APEs
            },
            ...
          ],

          # Error log (failed trials)
          "error_log": [str, ...]
        }
        """
        output_path = os.path.join(self.iteration_dir, "inner_loop_results.json")

        # Restructure for clarity
        detailed_diagnostics = results.get("detailed_diagnostics", {})

        output = {
            "iteration": self.iteration,
            "backend_name": self.backend_name,
            "timestamp": datetime.now().isoformat(),

            # Optimization metadata
            "optimization": {
                "n_trials": results["n_trials"],
                "optimization_time_seconds": results["optimization_time"],
                "num_errors": len(self.error_log)
            },

            # Best parameters
            "best_params": {
                "alpha": results["best_alpha"],
                "beta": results["best_beta"]
            },

            # Overall loss (RMSE across experiments)
            "loss": {
                "overall_loss": results["best_loss"],
                "ttft_rmse": detailed_diagnostics.get("ttft_rmse"),
                "e2e_rmse": detailed_diagnostics.get("e2e_rmse")
            },

            # Per-experiment results (if available)
            "per_experiment_results": detailed_diagnostics.get("per_experiment", []),

            # Error log
            "error_log": self.error_log
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        if self.error_log:
            print(f"⚠ {len(self.error_log)} trials failed (see error_log in results)")


def main():
    """Main entry point for inner loop optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inner loop Bayesian optimization for latency coefficients"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
        help="Iteration number (reads files from iterations/iter{N}/)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1000,
        help="Number of Bayesian optimization trials (default: 1000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per trial in seconds"
    )
    parser.add_argument(
        "--no-detailed-eval",
        action="store_true",
        help="Skip post-convergence detailed evaluation"
    )
    parser.add_argument(
        "--data-dir",
        default="trainval_data",
        help="Directory containing ground-truth experiments (default: trainval_data)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Bayesian optimization (default: 42, for determinism)"
    )

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = InnerLoopOptimizer(
        iteration=args.iteration,
        n_trials=args.n_trials,
        timeout_per_trial=args.timeout,
        data_dir=args.data_dir,
        seed=args.seed
    )

    # Phase 1: Setup
    try:
        optimizer.setup()
    except Exception as e:
        print(f"\n❌ Setup failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Phase 2: Optimize
    try:
        results = optimizer.optimize()
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Phase 3: Detailed evaluation (optional)
    if not args.no_detailed_eval:
        try:
            diagnostics = optimizer.evaluate_detailed(
                results["best_alpha"],
                results["best_beta"]
            )
            results["detailed_diagnostics"] = diagnostics
        except Exception as e:
            print(f"\n⚠ Detailed evaluation failed: {e}", file=sys.stderr)
            print("Continuing with basic results...")

    # Save results
    optimizer.save_results(results)

    print("\n" + "=" * 70)
    print("INNER LOOP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
