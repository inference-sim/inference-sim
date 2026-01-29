#!/usr/bin/env python3
"""
BLIS Evaluator

Evaluates BLIS roofline model predictions against ground truth data from
vLLM experiments. Runs BLIS simulations for each experiment/QPS point and
compares predicted metrics against server-side measurements.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any


class BLISEvaluator:
    def __init__(self, ground_truth_path: str, blis_binary_path: str = None, verbose: bool = False):
        """
        Initialize evaluator with paths to ground truth data and BLIS binary.

        Args:
            ground_truth_path: Path to combined_ground_truth.json
            blis_binary_path: Path to simulation_worker binary (default: {cwd}/simulation_worker)
            verbose: Enable verbose output for debugging
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.verbose = verbose

        # Default to current directory / simulation_worker
        if blis_binary_path is None:
            self.blis_binary_path = Path.cwd() / "simulation_worker"
        else:
            self.blis_binary_path = Path(blis_binary_path)

        # Load ground truth data
        with open(self.ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)

        # Verify binary exists
        if not self.blis_binary_path.exists():
            raise FileNotFoundError(f"BLIS binary not found: {self.blis_binary_path}")

        print(f"Loaded {len(self.ground_truth['experiments'])} experiments from {ground_truth_path}")

    def build_blis_command(self, experiment: Dict, qps_sweep: Dict) -> List[str]:
        """
        Build BLIS command from experiment configuration and QPS point.

        Args:
            experiment: Experiment dict from combined_ground_truth.json
            qps_sweep: Single QPS sweep point with metrics

        Returns:
            Command list for subprocess execution
        """
        model_name = experiment["model"]
        vllm_config = experiment["vllm_config"]
        workload_config = experiment["workload_config"]

        # Derive model config folder: model_configs/{model.split("/")[1].lower()}
        model_folder_name = model_name.split("/")[1].lower()
        model_config_folder = f"model_configs/{model_folder_name}"

        # Build command
        cmd = [
            str(self.blis_binary_path),
            "run",
            "--model", model_name,
            "--tp", str(vllm_config["tensor_parallelism"]),
            "--hardware", "H100",
            "--vllm-version", "vllm/vllm-openai:v0.8.4",
            "--model-config-folder", model_config_folder,
            "--hardware-config", "hardware_config.json",
            "--total-kv-blocks", str(experiment["total_kv_blocks"]),
            "--max-num-running-reqs", str(vllm_config["max_num_seqs"]),
            "--max-num-scheduled-tokens", str(vllm_config["max_num_batched_tokens"]),
            "--max-model-len", str(vllm_config["max_model_len"]),
            "--workload", "distribution",
            "--rate", str(qps_sweep["qps"]),
            "--max-prompts", str(workload_config["max_requests"]),
            "--prefix-tokens", str(workload_config["data"]["prefix_tokens"]),
            "--prompt-tokens", str(workload_config["data"]["prompt_tokens"]),
            "--prompt-tokens-stdev", str(workload_config["data"]["prompt_tokens_stdev"]),
            "--prompt-tokens-min", str(workload_config["data"]["prompt_tokens_min"]),
            "--prompt-tokens-max", str(workload_config["data"]["prompt_tokens_max"]),
            "--output-tokens", str(workload_config["data"]["output_tokens"]),
            "--output-tokens-stdev", str(workload_config["data"]["output_tokens_stdev"]),
            "--output-tokens-min", str(workload_config["data"]["output_tokens_min"]),
            "--output-tokens-max", str(workload_config["data"]["output_tokens_max"]),
        ]

        return cmd

    def run_blis(self, command: List[str]) -> Dict[str, Any]:
        """
        Execute BLIS via subprocess and capture stdout.

        Args:
            command: Command list for subprocess

        Returns:
            Parsed metrics dict from BLIS JSON output
        """
        # print(" ".join(list(map(str, command))))
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=True
            )

            # Debug: Show first 500 chars of stdout if verbose
            stdout = result.stdout.strip()
            if not stdout:
                print(f"ERROR: BLIS produced empty output")
                print(f"STDERR: {result.stderr}")
                raise ValueError("Empty BLIS output")

            # Try to parse JSON from stdout
            # BLIS might output logs before JSON, so try to find JSON block
            try:
                metrics = json.loads(stdout)
                return metrics
            except json.JSONDecodeError:
                # Try to extract JSON if there's text before it
                json_start = stdout.find('{')
                if json_start != -1:
                    json_str = stdout[json_start:]
                    metrics = json.loads(json_str)
                    return metrics
                else:
                    raise

        except subprocess.TimeoutExpired:
            print(f"ERROR: BLIS execution timed out")
            raise
        except subprocess.CalledProcessError as e:
            print(f"ERROR: BLIS execution failed with return code {e.returncode}")
            print(f"STDERR: {e.stderr}")
            if e.stdout:
                print(f"STDOUT (first 500 chars): {e.stdout[:500]}")
            raise
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse BLIS output as JSON")
            print(f"JSON decode error: {e}")
            print(f"STDOUT (first 1000 chars):")
            print(result.stdout[:1000])
            print(f"\nSTDERR:")
            print(result.stderr)
            raise

    def compare_metrics(self, ground_truth: Dict, predicted: Dict) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-metric errors.

        Args:
            ground_truth: Ground truth metrics dict
            predicted: BLIS predicted metrics dict

        Returns:
            Dict with error structure for each metric
        """
        metrics = ["ttft_mean_ms", "ttft_p90_ms", "itl_mean_ms", "e2e_mean_ms", "e2e_p90_ms"]
        errors = {}

        for metric in metrics:
            gt_value = ground_truth[metric]
            pred_value = predicted[metric]

            absolute_error = abs(pred_value - gt_value)
            percentage_error = absolute_error / gt_value * 100

            errors[metric] = {
                "absolute": absolute_error,
                "percentage": percentage_error
            }

        return errors

    def evaluate_qps_point(self, experiment: Dict, qps_sweep: Dict) -> Dict[str, Any]:
        """
        Evaluate a single QPS point for an experiment.

        Args:
            experiment: Experiment configuration
            qps_sweep: QPS sweep point with ground truth metrics

        Returns:
            Evaluation result with ground truth, predictions, and errors
        """
        # Build and run BLIS command
        command = self.build_blis_command(experiment, qps_sweep)

        if self.verbose:
            print(f"  Evaluating QPS {qps_sweep['qps']:.3f}")
            print(f"  Command: {' '.join(command)}")

        # Run BLIS and handle errors
        try:
            blis_output = self.run_blis(command)
        except Exception as e:
            if not self.verbose:
                print(f"  ERROR at QPS {qps_sweep['qps']:.3f}")
            print(f"  Failed command: {' '.join(command)}")
            raise

        # Extract predicted metrics (assuming BLIS outputs same metric names)
        predicted_metrics = blis_output  # May need adjustment based on actual BLIS output format

        # Compare metrics
        errors = self.compare_metrics(qps_sweep, predicted_metrics)

        result = {
            "qps": qps_sweep["qps"],
            "ground_truth": {
                "ttft_mean_ms": qps_sweep["ttft_mean_ms"],
                "ttft_p90_ms": qps_sweep["ttft_p90_ms"],
                "itl_mean_ms": qps_sweep["itl_mean_ms"],
                "e2e_mean_ms": qps_sweep["e2e_mean_ms"],
                "e2e_p90_ms": qps_sweep["e2e_p90_ms"]
            },
            "blis_predicted": predicted_metrics,
            "errors": errors
        }

        return result

    def evaluate_experiment(self, experiment: Dict) -> Dict[str, Any]:
        """
        Evaluate all QPS points for a single experiment.

        Args:
            experiment: Experiment configuration from ground truth

        Returns:
            Experiment evaluation with per-QPS results and aggregated percentage errors
        """
        exp_name = experiment["experiment_name"]

        qps_evaluations = []

        # Initialize sum accumulator for percentage errors only
        sum_percentage_errors = {
            "ttft_mean_ms": 0.0,
            "ttft_p90_ms": 0.0,
            "itl_mean_ms": 0.0,
            "e2e_mean_ms": 0.0,
            "e2e_p90_ms": 0.0
        }

        # Evaluate each QPS point
        for qps_sweep in experiment["qps_sweeps"]:
            try:
                qps_result = self.evaluate_qps_point(experiment, qps_sweep)
                qps_evaluations.append(qps_result)

                # Accumulate percentage errors
                for metric in sum_percentage_errors.keys():
                    sum_percentage_errors[metric] += qps_result["errors"][metric]["percentage"]

            except Exception as e:
                if self.verbose:
                    print(f"  ERROR: Failed to evaluate QPS {qps_sweep['qps']}: {e}")
                continue

        result = {
            "experiment_name": exp_name,
            "model": experiment["model"],
            "app_type": experiment["vllm_config"]["app"],
            "num_qps_points": len(qps_evaluations),
            "sum_percentage_errors": sum_percentage_errors
        }

        return result

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Process all experiments and aggregate results.

        Returns:
            Complete evaluation report with summary and per-experiment details
        """
        all_experiment_results = []

        # Process each experiment
        for experiment in self.ground_truth["experiments"]:
            try:
                exp_result = self.evaluate_experiment(experiment)
                all_experiment_results.append(exp_result)
            except Exception as e:
                if self.verbose:
                    print(f"ERROR: Failed to evaluate experiment {experiment['experiment_name']}: {e}")
                continue

        # Aggregate across all experiments
        total_qps_points = sum(exp["num_qps_points"] for exp in all_experiment_results)

        sum_percentage_errors = {
            "ttft_mean_ms": 0.0,
            "ttft_p90_ms": 0.0,
            "itl_mean_ms": 0.0,
            "e2e_mean_ms": 0.0,
            "e2e_p90_ms": 0.0
        }

        # Aggregate by workload type
        by_workload = {"train": [], "prefill": []}

        for exp_result in all_experiment_results:
            # Add to overall sum
            for metric in sum_percentage_errors.keys():
                sum_percentage_errors[metric] += exp_result["sum_percentage_errors"][metric]

            # Categorize by workload type
            app_type = exp_result["app_type"]
            if "train" in app_type:
                by_workload["train"].append(exp_result)
            else:
                by_workload["prefill"].append(exp_result)

        # Aggregate by model
        by_model = {"codellama": [], "llama-2": []}
        for exp_result in all_experiment_results:
            model_name = exp_result["model"].lower()
            if "codellama" in model_name:
                by_model["codellama"].append(exp_result)
            elif "llama" in model_name:
                by_model["llama-2"].append(exp_result)

        # Compute aggregations for workload types
        by_workload_agg = {}
        for workload_type, experiments in by_workload.items():
            if experiments:
                agg_pct = {metric: sum(e["sum_percentage_errors"][metric] for e in experiments)
                          for metric in sum_percentage_errors.keys()}
                by_workload_agg[workload_type] = {
                    "sum_percentage_errors": agg_pct
                }

        # Compute aggregations for model types
        by_model_agg = {}
        for model_type, experiments in by_model.items():
            if experiments:
                agg_pct = {metric: sum(e["sum_percentage_errors"][metric] for e in experiments)
                          for metric in sum_percentage_errors.keys()}
                by_model_agg[model_type] = {
                    "sum_percentage_errors": agg_pct
                }

        # Build final result
        evaluation_result = {
            "evaluation_summary": {
                "total_experiments": len(all_experiment_results),
                "total_qps_points": total_qps_points,
                "sum_percentage_errors": sum_percentage_errors,
                "by_workload_type": by_workload_agg,
                "by_model": by_model_agg
            },
            "experiments": all_experiment_results
        }

        return evaluation_result

    def print_results(self, results: Dict[str, Any]):
        """
        Print evaluation results to terminal - percentage errors only.

        Args:
            results: Evaluation results dict
        """
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS - Sum of % Errors")
        print("=" * 80)

        # Per-experiment results
        for exp in results["experiments"]:
            print(f"\n{exp['experiment_name']}:")
            for metric, error in exp["sum_percentage_errors"].items():
                print(f"  {metric:20s}: {error:8.2f}%")

        print("\n" + "=" * 80)

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to JSON file (optional).

        Args:
            results: Evaluation results dict
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate BLIS predictions against ground truth")
    parser.add_argument(
        "--ground-truth",
        default="eval/combined_ground_truth.json",
        help="Path to combined ground truth JSON (default: eval/combined_ground_truth.json)"
    )
    parser.add_argument(
        "--blis-binary",
        default=None,
        help="Path to BLIS binary (default: {cwd}/simulation_worker)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file instead of printing to terminal"
    )
    parser.add_argument(
        "--output",
        default="eval/evaluation_results.json",
        help="Path to output results JSON (only used with --save)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = BLISEvaluator(args.ground_truth, args.blis_binary, verbose=args.verbose)
    results = evaluator.evaluate_all()

    # Print or save results
    if args.save:
        evaluator.save_results(results, args.output)
    else:
        evaluator.print_results(results)


if __name__ == "__main__":
    main()
