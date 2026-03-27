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
                 manifest_path: str = "iteration_manifest.yaml",
                 bounds_path: str = "coefficient_bounds.yaml",
                 n_trials: int = 50,
                 timeout_per_trial: int = 120):
        """
        Initialize inner loop optimizer.

        Args:
            manifest_path: Path to iteration manifest from outer loop
            bounds_path: Path to coefficient bounds specification
            n_trials: Number of Bayesian optimization trials
            timeout_per_trial: Timeout in seconds for each BLIS run
        """
        self.manifest_path = manifest_path
        self.bounds_path = bounds_path
        self.n_trials = n_trials
        self.timeout_per_trial = timeout_per_trial

        self.backend_name = None
        self.bounds = None
        self.iteration = None
        self.error_log = []  # Track failed trials

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

        # 2. Verify declared files exist
        print("\nVerifying declared files...")
        for filepath in modified_files:
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Agent declared {filepath} but file not found.\n"
                    f"Check outer loop output."
                )
            print(f"  ✓ {filepath}")

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

        # 4. Compile BLIS binary with new latency backend
        print("\nCompiling BLIS...")
        compile_start = time.time()

        try:
            result = subprocess.run(
                ["go", "build", "-o", "blis", "main.go"],
                capture_output=True,
                check=True,
                timeout=60
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
        if not os.path.exists("./blis"):
            raise RuntimeError("BLIS binary not found after compilation.")

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
                sys.executable,  # Use same Python interpreter
                self.run_blis_script,  # Absolute path to script
                "--latency-model", self.backend_name,
                "--alpha-coeffs", alpha_str,
                "--beta-coeffs", beta_str,
                "--blis-binary", "../blis",
                "--data-dir", "trainval_data"
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
                - n_trials: Number of trials run (actual, may be less than requested if converged early)
                - optimization_time: Wall-clock time in seconds
                - converged_early: Whether optimization stopped early due to convergence
        """
        print("\n" + "=" * 70)
        print(f"BAYESIAN OPTIMIZATION (up to {self.n_trials} trials)")
        print("Early stopping: >1% improvement required in 50-trial window")
        print("=" * 70)

        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            study_name=f"iter{self.iteration}_{self.backend_name}",
            sampler=optuna.samplers.TPESampler(seed=42)
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

        # Convergence callback: stop if no >1% improvement in last 50 trials
        converged_early = [False]  # Mutable container for callback

        def convergence_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            """Stop if best loss hasn't improved >1% in last 50 trials."""
            n = len(study.trials)
            if n <= 50:
                return  # Need more than 50 trials to check 50-trial window

            # Get best loss from all trials up to 50 trials ago (trials 0 to n-51)
            trials_before_window = study.trials[:n-50]
            best_loss_50_ago = min(t.value for t in trials_before_window if t.value is not None)

            # Get current best loss
            current_best = study.best_value

            # Calculate improvement
            improvement = (best_loss_50_ago - current_best) / best_loss_50_ago

            if improvement <= 0.01:  # ≤1% improvement
                print(f"\n[Convergence] Stopping at trial {n}/{self.n_trials}")
                print(f"[Convergence] Best loss 50 trials ago: {best_loss_50_ago:.6f}")
                print(f"[Convergence] Current best loss: {current_best:.6f}")
                print(f"[Convergence] Improvement: {improvement*100:.2f}% (threshold: >1.00%)")
                converged_early[0] = True
                study.stop()

        # Run optimization
        opt_start = time.time()
        study.optimize(
            self.optuna_objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            callbacks=[convergence_callback]
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
        if converged_early[0]:
            print(f"Status: Converged early (no >1% improvement in last 50 trials)")
        else:
            print(f"Status: Completed all requested trials")
        print(f"Optimization time: {opt_time:.1f}s")
        print(f"Time per trial: {opt_time / actual_trials:.2f}s")

        return {
            "best_alpha": best_alpha,
            "best_beta": best_beta,
            "best_loss": best_loss,
            "n_trials": actual_trials,
            "optimization_time": opt_time,
            "converged_early": converged_early[0]
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
            sys.executable,  # Use same Python interpreter
            self.run_blis_script,  # Absolute path to script
            "--latency-model", self.backend_name,
            "--alpha-coeffs", alpha_str,
            "--beta-coeffs", beta_str,
            "--blis-binary", "../blis",
            "--data-dir", "trainval_data",
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

    def save_results(self, results: Dict[str, Any], output_path: str = "inner_loop_results.json") -> None:
        """
        Save optimization results to JSON file.

        Args:
            results: Results dictionary from optimize()
            output_path: Output file path
        """
        results["timestamp"] = datetime.now().isoformat()
        results["iteration"] = self.iteration
        results["backend_name"] = self.backend_name
        results["num_errors"] = len(self.error_log)
        results["error_log"] = self.error_log  # Include failed trials

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

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
        "--manifest",
        default="iteration_manifest.yaml",
        help="Path to iteration manifest from outer loop"
    )
    parser.add_argument(
        "--bounds",
        default="coefficient_bounds.yaml",
        help="Path to coefficient bounds specification"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Bayesian optimization trials"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per trial in seconds"
    )
    parser.add_argument(
        "--output",
        default="inner_loop_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--no-detailed-eval",
        action="store_true",
        help="Skip post-convergence detailed evaluation"
    )

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = InnerLoopOptimizer(
        manifest_path=args.manifest,
        bounds_path=args.bounds,
        n_trials=args.n_trials,
        timeout_per_trial=args.timeout
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
    optimizer.save_results(results, args.output)

    print("\n" + "=" * 70)
    print("INNER LOOP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
