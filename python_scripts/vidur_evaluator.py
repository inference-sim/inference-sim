"""
Vidur Runner and Evaluator - Run Vidur simulations with configuration

This module provides a Vidur-specific interface matching the BLIS runner API.

Key components:
- run_vidur(): Run single simulation at specific QPS (equivalent to run_blis)
- VidurEvaluator: Evaluates Vidur predictions against ground truth data
"""
import json
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import glob


def generate_synthetic_trace(config: Dict, output_path: str, qps: float):
    """
    Generate a synthetic trace file from workload parameters with constant inter-arrival times.

    Creates a CSV file with format: arrived_at,num_prefill_tokens,num_decode_tokens

    Args:
        config: Config with workload parameters (prompt_tokens, output_tokens, etc.)
        output_path: Path to write trace CSV
        qps: Queries per second (for constant inter-arrival time)
    """
    import numpy as np

    num_requests = config.get('num_requests', 1000)
    inter_arrival_time = 1.0 / qps  # Constant spacing

    # Get workload parameters with defaults
    prompt_mean = config.get('prompt_tokens', 512)
    prompt_std = config.get('prompt_tokens_stdev', 100)
    prompt_min = config.get('prompt_tokens_min', 1)
    prompt_max = config.get('prompt_tokens_max', 2048)

    output_mean = config.get('output_tokens', 128)
    output_std = config.get('output_tokens_stdev', 50)
    output_min = config.get('output_tokens_min', 1)
    output_max = config.get('output_tokens_max', 512)

    # Generate arrival times (constant spacing)
    arrival_times = np.arange(num_requests) * inter_arrival_time

    # Generate token counts from normal distribution, clipped to min/max
    np.random.seed(42)  # Reproducibility
    prompt_tokens = np.random.normal(prompt_mean, prompt_std, num_requests)
    prompt_tokens = np.clip(prompt_tokens, prompt_min, prompt_max).astype(int)

    output_tokens = np.random.normal(output_mean, output_std, num_requests)
    output_tokens = np.clip(output_tokens, output_min, output_max).astype(int)

    # Write to CSV with arrival times (Vidur format)
    with open(output_path, 'w') as f:
        f.write('arrived_at,num_prefill_tokens,num_decode_tokens\n')
        for i in range(num_requests):
            f.write(f'{arrival_times[i]:.6f},{prompt_tokens[i]},{output_tokens[i]}\n')


def parse_request_metrics_csv(csv_path: str) -> Optional[Dict]:
    """
    Parse Vidur's request_metrics.csv and extract key metrics.

    Columns in request_metrics.csv:
    - prefill_time_execution_plus_preemption: TTFT (execution time only, excludes scheduling delay) in seconds
    - decode_time_execution_plus_preemption_normalized: TPOT/ITL in seconds per token
    - request_e2e_time: E2E latency in seconds
    - request_scheduling_delay: Scheduling delay in seconds

    Args:
        csv_path: Path to request_metrics.csv

    Returns:
        Dictionary with metrics in BLIS-compatible format (milliseconds)
    """
    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"Warning: Empty CSV file: {csv_path}", file=sys.stderr)
            return None

        # Extract metrics and convert seconds to milliseconds for compatibility with BLIS
        metrics = {
            # TTFT (Time to First Token) - use execution time only
            'ttft_mean_ms': df['prefill_time_execution_plus_preemption'].mean() * 1000,
            'ttft_p90_ms': df['prefill_time_execution_plus_preemption'].quantile(0.90) * 1000,
            'ttft_p95_ms': df['prefill_time_execution_plus_preemption'].quantile(0.95) * 1000,
            'ttft_p99_ms': df['prefill_time_execution_plus_preemption'].quantile(0.99) * 1000,

            # ITL/TPOT (Inter-Token Latency / Time Per Output Token)
            'itl_mean_ms': df['decode_time_execution_plus_preemption_normalized'].mean() * 1000,
            'itl_p90_ms': df['decode_time_execution_plus_preemption_normalized'].quantile(0.90) * 1000,
            'itl_p95_ms': df['decode_time_execution_plus_preemption_normalized'].quantile(0.95) * 1000,
            'itl_p99_ms': df['decode_time_execution_plus_preemption_normalized'].quantile(0.99) * 1000,

            # E2E Latency
            'e2e_mean_ms': df['request_e2e_time'].mean() * 1000,
            'e2e_p90_ms': df['request_e2e_time'].quantile(0.90) * 1000,
            'e2e_p95_ms': df['request_e2e_time'].quantile(0.95) * 1000,
            'e2e_p99_ms': df['request_e2e_time'].quantile(0.99) * 1000,

            # Scheduling Delay
            'scheduling_delay_mean_ms': df['request_scheduling_delay'].mean() * 1000,
            'scheduling_delay_p90_ms': df['request_scheduling_delay'].quantile(0.90) * 1000,
            'scheduling_delay_p95_ms': df['request_scheduling_delay'].quantile(0.95) * 1000,
            'scheduling_delay_p99_ms': df['request_scheduling_delay'].quantile(0.99) * 1000,

            # Request stats
            'total_requests': len(df),
            'completed_requests': len(df),
            'failed_requests': 0,  # Vidur doesn't track failures in same way
        }

        # Calculate throughput (requests per second)
        if 'completed_at' in df.columns and 'arrived_at' in df.columns:
            total_duration = df['completed_at'].max() - df['arrived_at'].min()
            if total_duration > 0:
                metrics['responses_per_sec'] = len(df) / total_duration

            # Token throughput
            if 'request_num_tokens' in df.columns:
                total_tokens = df['request_num_tokens'].sum()
                metrics['tokens_per_sec'] = total_tokens / total_duration

        # Validate metrics - check if we got valid latency data
        # If all key metrics are 0 or NaN, the simulation didn't produce valid results
        import math
        key_metrics = ['e2e_p95_ms', 'ttft_p90_ms', 'itl_mean_ms']
        valid_data = False
        for metric_key in key_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                if not math.isnan(value) and value > 0:
                    valid_data = True
                    break

        if not valid_data:
            print(f"Warning: Invalid metrics (all zeros or NaN) in {csv_path}", file=sys.stderr)
            return None

        return metrics

    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error parsing CSV {csv_path}: {e}", file=sys.stderr)
        return None


def run_vidur(
    config: Dict,
    qps: float,
    trace_file: Optional[str] = None,
    num_requests: Optional[int] = None,
    timeout: int = 300,
    verbose: bool = True
) -> Dict:
    """
    Run Vidur simulation with given configuration and QPS.

    Args:
        config: Configuration dictionary containing:
            - model: Model identifier
            - hardware: Hardware type (e.g., 'H100')
            - tp: Tensor parallelism degree
            - batch_size: Maximum number of concurrent requests
            - max_scheduled_tokens: Maximum tokens per iteration
            - max_model_len: Maximum sequence length
            - total_kv_blocks: Total KV cache blocks (from ground truth)
            - num_requests: Number of requests to simulate (optional, default: 1000)
        qps: Queries per second (arrival rate)
        trace_file: Optional path to trace file (CSV with arrived_at,num_prefill_tokens,num_decode_tokens)
        num_requests: Number of requests to simulate (optional, overrides config value)
        timeout: Timeout in seconds (default: 300)
        verbose: If True, print detailed simulation info (default: True)

    Returns:
        Dictionary with simulation results including:
        - ttft_p90_ms: 90th percentile time to first token
        - itl_p95_ms: 95th percentile inter-token latency
        - e2e_p90_ms: 90th percentile end-to-end latency
        - etc.
        Or None if simulation failed
    """
    # Get num_requests from config if not provided as argument
    if num_requests is None:
        num_requests = config.get('num_requests', 1000)

    # Create local tmp directory in current path
    local_tmp = Path('./tmp').resolve()  # Use absolute path
    local_tmp.mkdir(exist_ok=True)

    # Create unique subdirectory for this simulation (include PID for parallel safety)
    timestamp = int(time.time() * 1000000)  # microsecond precision
    pid = os.getpid()
    temp_dir = local_tmp / f'vidur_sim_{timestamp}_{pid}'
    temp_dir.mkdir(exist_ok=True)
    output_dir = str(temp_dir)  # Already absolute since local_tmp is absolute

    # Map hardware names
    device_map = {
        'H100': 'h100',
        'A100': 'a100',
        'A40': 'a40',
    }
    device = device_map.get(config['hardware'], config['hardware'].lower())

    # Generate trace file if not provided
    if not trace_file:
        actual_trace_file = os.path.join(output_dir, f'synthetic_trace_qps{qps}.csv')
        # Update config num_requests for trace generation
        trace_config = {**config, 'num_requests': num_requests}
        generate_synthetic_trace(trace_config, actual_trace_file, qps=qps)
    else:
        # Convert trace_file to absolute path if relative
        actual_trace_file = str(Path(trace_file).resolve())

    # Check if Vidur is installed as a package or available in subdirectory
    vidur_dir = Path(__file__).parent / 'vidur'
    vidur_main = vidur_dir / 'vidur' / 'main.py'

    if not vidur_main.exists():
        print(f"Error: Vidur not found at {vidur_main}", file=sys.stderr)
        print(f"Please ensure vidur/ subdirectory is present", file=sys.stderr)
        return None

    # Try to import vidur to see if Vidur is properly installed
    vidur_installed = False
    try:
        import vidur.main
        vidur_installed = True
    except ImportError:
        pass

    # Initialize environment variable (will be set if vidur not installed)
    run_env = None

    # Build vidur.main command with explicit arguments
    if vidur_installed:
        base_cmd = [sys.executable, '-m', 'vidur.main']
    else:
        if verbose:
            print(f"Note: Vidur not installed as package, running from {vidur_dir}")

        run_env = os.environ.copy()
        pythonpath = str(vidur_dir)
        if 'PYTHONPATH' in run_env:
            pythonpath = f"{pythonpath}:{run_env['PYTHONPATH']}"
        run_env['PYTHONPATH'] = pythonpath

        base_cmd = [sys.executable, str(vidur_main)]

    # Get total_kv_blocks from config (from ground truth)
    total_kv_blocks = config['total_kv_blocks']

    if verbose:
        print(f"  Using total_kv_blocks: {total_kv_blocks:,}")

    # Build full command with all required arguments
    cmd = base_cmd + [
        # Model and hardware
        '--replica_config_model_name', config['model'],
        '--replica_config_device', device,
        '--replica_config_tensor_parallel_size', str(config.get('tp', 1)),
        '--cluster_config_num_replicas', '1',

        # Scheduler
        '--replica_scheduler_config_type', 'vllm',
        '--vllm_scheduler_config_max_tokens_in_batch', str(config.get('max_scheduled_tokens', 8192)),
        '--vllm_scheduler_config_num_blocks', str(total_kv_blocks),
        '--vllm_scheduler_config_watermark_blocks_fraction', '0.0',

        # Random Forest execution time predictor
        '--random_forrest_execution_time_predictor_config_prediction_max_batch_size', str(config.get('batch_size', 256)),

        # Request generator - use trace_replay for both lengths AND arrival times
        '--length_generator_config_type', 'trace',
        '--request_generator_config_type', 'trace_replay',
        '--trace_request_generator_config_trace_file', actual_trace_file,
        '--trace_request_generator_config_max_tokens', str(config.get('max_model_len', 4096)),

        # Output
        '--metrics_config_output_dir', output_dir,

        # Disable unnecessary features for speed
        '--no-metrics_config_save_table_to_wandb',
        '--no-metrics_config_store_plots',
        '--no-metrics_config_enable_chrome_trace',
    ]

    if verbose:
        print(f"\nRunning Vidur simulation:")
        print(f"  QPS: {qps}")
        print(f"  Num Requests: {num_requests}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Max Scheduled Tokens: {config['max_scheduled_tokens']}")
        print(f"  Max Model Length: {config['max_model_len']}")
        print(f"  Total KV Blocks: {total_kv_blocks:,}")
        print(f"  Command: {' '.join(cmd)}\n")

    # Run Vidur simulation
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            cwd=vidur_dir,  # Run from vidur directory to access ./data/profiling
            env=run_env
        )

        if result.returncode != 0:
            if verbose:
                print(f"Vidur simulation failed with return code {result.returncode}", file=sys.stderr)
                print(f"STDERR: {result.stderr}", file=sys.stderr)
            return None

    except subprocess.TimeoutExpired:
        print(f"Vidur simulation timed out after {timeout}s", file=sys.stderr)
        # Clean up temporary directory
        try:
            shutil.rmtree(str(temp_dir))
        except:
            pass
        return None
    except Exception as e:
        print(f"Error running Vidur simulation: {e}", file=sys.stderr)
        # Clean up temporary directory
        try:
            shutil.rmtree(str(temp_dir))
        except:
            pass
        return None

    # Parse results from output directory
    # Find request_metrics.csv in timestamped subdirectory
    glob_pattern = f"{output_dir}/*/request_metrics.csv"
    metrics_files = glob.glob(glob_pattern)

    if not metrics_files:
        print(f"Error: No request_metrics.csv found in {output_dir}", file=sys.stderr)
        # Clean up temporary directory
        try:
            shutil.rmtree(str(temp_dir))
        except:
            pass
        return None

    # Parse the metrics
    metrics_file = metrics_files[0]
    metrics = parse_request_metrics_csv(metrics_file)

    if metrics:
        metrics['qps'] = qps

    # Clean up temporary directory after parsing results
    try:
        shutil.rmtree(str(temp_dir))
    except Exception as e:
        if verbose:
            print(f"Warning: Could not clean up {temp_dir}: {e}", file=sys.stderr)

    return metrics


class VidurEvaluator:
    """
    Evaluates Vidur predictions against ground truth data from
    vLLM experiments. Runs Vidur simulations for each experiment/QPS point and
    compares predicted metrics against server-side measurements.
    """

    def __init__(self, ground_truth_path: str, verbose: bool = False):
        """
        Initialize evaluator with path to ground truth data.

        Args:
            ground_truth_path: Path to combined_ground_truth.json
            verbose: Enable verbose output for debugging
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.verbose = verbose

        # Load ground truth data
        with open(self.ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)

        print(f"Loaded {len(self.ground_truth['experiments'])} experiments from {ground_truth_path}")

    def build_vidur_config(self, experiment: Dict, qps_sweep: Dict) -> Dict:
        """
        Build Vidur config from experiment configuration and QPS point.

        Args:
            experiment: Experiment dict from combined_ground_truth.json
            qps_sweep: Single QPS sweep point with metrics

        Returns:
            Config dict for run_vidur()
        """
        model_name = experiment["model"]
        vllm_config = experiment["vllm_config"]
        workload_config = experiment["workload_config"]

        config = {
            # Model and hardware
            "model": model_name,
            "hardware": "H100",
            "tp": vllm_config["tensor_parallelism"],

            # Scheduler config
            "batch_size": vllm_config["max_num_seqs"],
            "max_scheduled_tokens": vllm_config["max_num_batched_tokens"],
            "max_model_len": vllm_config["max_model_len"],
            "total_kv_blocks": experiment["total_kv_blocks"],

            # Workload config
            "num_requests": workload_config["max_requests"],
            "prompt_tokens": workload_config["data"]["prompt_tokens"],
            "prompt_tokens_stdev": workload_config["data"]["prompt_tokens_stdev"],
            "prompt_tokens_min": workload_config["data"]["prompt_tokens_min"],
            "prompt_tokens_max": workload_config["data"]["prompt_tokens_max"],
            "output_tokens": workload_config["data"]["output_tokens"],
            "output_tokens_stdev": workload_config["data"]["output_tokens_stdev"],
            "output_tokens_min": workload_config["data"]["output_tokens_min"],
            "output_tokens_max": workload_config["data"]["output_tokens_max"],
        }

        return config

    def compare_metrics(self, ground_truth: Dict, predicted: Dict) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-metric errors.

        Args:
            ground_truth: Ground truth metrics dict
            predicted: Vidur predicted metrics dict

        Returns:
            Dict with error structure for each metric
        """
        metrics = ["ttft_mean_ms", "ttft_p90_ms", "itl_mean_ms", "e2e_mean_ms", "e2e_p90_ms"]
        errors = {}

        for metric in metrics:
            gt_value = ground_truth[metric]
            pred_value = predicted.get(metric, 0.0)

            if gt_value == 0:
                percentage_error = 0.0 if pred_value == 0 else float('inf')
            else:
                absolute_error = abs(pred_value - gt_value)
                percentage_error = absolute_error / gt_value * 100

            errors[metric] = {
                "absolute": abs(pred_value - gt_value),
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
        # Build config for Vidur
        config = self.build_vidur_config(experiment, qps_sweep)
        qps = qps_sweep["qps"]

        if self.verbose:
            print(f"  Evaluating QPS {qps:.3f}")
            print(f"  Config: {config}")

        # Run Vidur simulation
        try:
            vidur_output = run_vidur(
                config=config,
                qps=qps,
                num_requests=config["num_requests"],
                verbose=self.verbose
            )
        except Exception as e:
            if not self.verbose:
                print(f"  ERROR at QPS {qps:.3f}")
            print(f"  Exception: {e}")
            raise

        if vidur_output is None:
            raise RuntimeError(f"Vidur simulation returned None for QPS {qps}")

        # Compare metrics
        errors = self.compare_metrics(qps_sweep, vidur_output)

        result = {
            "qps": qps,
            "ground_truth": {
                "ttft_mean_ms": qps_sweep["ttft_mean_ms"],
                "ttft_p90_ms": qps_sweep["ttft_p90_ms"],
                "itl_mean_ms": qps_sweep["itl_mean_ms"],
                "e2e_mean_ms": qps_sweep["e2e_mean_ms"],
                "e2e_p90_ms": qps_sweep["e2e_p90_ms"]
            },
            "vidur_predicted": {
                "ttft_mean_ms": vidur_output.get("ttft_mean_ms", 0.0),
                "ttft_p90_ms": vidur_output.get("ttft_p90_ms", 0.0),
                "itl_mean_ms": vidur_output.get("itl_mean_ms", 0.0),
                "e2e_mean_ms": vidur_output.get("e2e_mean_ms", 0.0),
                "e2e_p90_ms": vidur_output.get("e2e_p90_ms", 0.0)
            },
            "errors": errors
        }

        return result

    def evaluate_experiment(self, experiment: Dict) -> Dict[str, Any]:
        """
        Evaluate all QPS points for a single experiment.

        Args:
            experiment: Experiment configuration from ground truth

        Returns:
            Experiment evaluation with per-QPS results and mean percentage errors
        """
        exp_name = experiment["experiment_name"]

        qps_evaluations = []
        failed_qps_points = []

        # Initialize accumulator for percentage errors
        total_percentage_errors = {
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
                for metric in total_percentage_errors.keys():
                    total_percentage_errors[metric] += qps_result["errors"][metric]["percentage"]

            except Exception as e:
                # Track failed QPS points (don't include in results)
                failed_qps_points.append({
                    "qps": qps_sweep["qps"],
                    "error": str(e)
                })
                if self.verbose:
                    print(f"  ERROR: Failed to evaluate QPS {qps_sweep['qps']}: {e}")
                continue

        # Calculate mean percentage errors (only from successful runs)
        num_qps = len(qps_evaluations)
        num_failed = len(failed_qps_points)
        mean_percentage_errors = {
            metric: total / num_qps if num_qps > 0 else 0.0
            for metric, total in total_percentage_errors.items()
        }

        result = {
            "experiment_name": exp_name,
            "model": experiment["model"],
            "app_type": experiment["vllm_config"]["app"],
            "num_qps_points": num_qps,
            "num_failed_qps_points": num_failed,
            "failed_qps_points": failed_qps_points,
            "mean_percentage_errors": mean_percentage_errors
        }

        return result

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Process all experiments and aggregate results.

        Returns:
            Complete evaluation report with summary and per-experiment details
        """
        all_experiment_results = []
        failed_experiments = []

        # Process each experiment
        for experiment in self.ground_truth["experiments"]:
            print(f"\nEvaluating experiment: {experiment['experiment_name']}")
            try:
                exp_result = self.evaluate_experiment(experiment)
                all_experiment_results.append(exp_result)
            except Exception as e:
                error_msg = str(e)
                print(f"ERROR: Failed to evaluate experiment {experiment['experiment_name']}: {error_msg}")
                failed_experiments.append({
                    "experiment_name": experiment["experiment_name"],
                    "model": experiment["model"],
                    "error": error_msg
                })
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                continue

        # Aggregate across all experiments
        total_qps_points = sum(exp["num_qps_points"] for exp in all_experiment_results)
        total_failed_qps_points = sum(exp.get("num_failed_qps_points", 0) for exp in all_experiment_results)
        num_experiments = len(all_experiment_results)

        # Calculate overall mean by averaging experiment means
        mean_percentage_errors = {
            "ttft_mean_ms": 0.0,
            "ttft_p90_ms": 0.0,
            "itl_mean_ms": 0.0,
            "e2e_mean_ms": 0.0,
            "e2e_p90_ms": 0.0
        }

        # Aggregate by workload type
        by_workload = {}

        for exp_result in all_experiment_results:
            # Add to overall mean accumulator
            for metric in mean_percentage_errors.keys():
                mean_percentage_errors[metric] += exp_result["mean_percentage_errors"][metric]

            # Categorize by workload type
            app_type = exp_result["app_type"]
            if app_type not in by_workload:
                by_workload[app_type] = []
            by_workload[app_type].append(exp_result)

        # Compute overall mean
        if num_experiments > 0:
            mean_percentage_errors = {
                metric: total / num_experiments
                for metric, total in mean_percentage_errors.items()
            }

        # Aggregate by model (dynamically extract model family from name)
        by_model = {}
        for exp_result in all_experiment_results:
            model_name = exp_result["model"]
            # Extract model family: use the part after "/" or the whole name
            if "/" in model_name:
                model_key = model_name.split("/")[1].split("-")[0].lower()
            else:
                model_key = model_name.split("-")[0].lower()

            if model_key not in by_model:
                by_model[model_key] = []
            by_model[model_key].append(exp_result)

        # Compute aggregations for workload types (mean of means)
        by_workload_agg = {}
        for workload_type, experiments in by_workload.items():
            if experiments:
                num_exp = len(experiments)
                agg_pct = {metric: sum(e["mean_percentage_errors"][metric] for e in experiments) / num_exp
                          for metric in mean_percentage_errors.keys()}
                by_workload_agg[workload_type] = {
                    "num_experiments": num_exp,
                    "mean_percentage_errors": agg_pct
                }

        # Compute aggregations for model types (mean of means)
        by_model_agg = {}
        for model_type, experiments in by_model.items():
            if experiments:
                num_exp = len(experiments)
                agg_pct = {metric: sum(e["mean_percentage_errors"][metric] for e in experiments) / num_exp
                          for metric in mean_percentage_errors.keys()}
                by_model_agg[model_type] = {
                    "num_experiments": num_exp,
                    "mean_percentage_errors": agg_pct
                }

        # Build final result
        evaluation_result = {
            "evaluation_summary": {
                "total_experiments": num_experiments,
                "failed_experiments": len(failed_experiments),
                "total_qps_points": total_qps_points,
                "total_failed_qps_points": total_failed_qps_points,
                "mean_percentage_errors": mean_percentage_errors,
                "by_workload_type": by_workload_agg,
                "by_model": by_model_agg
            },
            "experiments": all_experiment_results,
            "failed": failed_experiments
        }

        return evaluation_result

    def print_results(self, results: Dict[str, Any]):
        """
        Print evaluation results to terminal - percentage errors only.

        Args:
            results: Evaluation results dict
        """
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS - Mean % Error across QPS points")
        print("=" * 80)

        # Per-experiment results
        for exp in results["experiments"]:
            failed_count = exp.get("num_failed_qps_points", 0)
            failed_suffix = f" ({failed_count} QPS points failed)" if failed_count > 0 else ""
            print(f"\n{exp['experiment_name']}:{failed_suffix}")
            for metric, error in exp["mean_percentage_errors"].items():
                print(f"  {metric:20s}: {error:8.2f}%")

        # Failed experiments (entire experiments that failed)
        if results.get("failed"):
            print("\n" + "-" * 80)
            print(f"FAILED EXPERIMENTS ({len(results['failed'])} total)")
            print("-" * 80)
            for failed in results["failed"]:
                print(f"\n  {failed['experiment_name']}:")
                print(f"    Model: {failed['model']}")
                print(f"    Error: {failed['error']}")

        # Summary of failed QPS points
        summary = results.get("evaluation_summary", {})
        total_failed_qps = summary.get("total_failed_qps_points", 0)
        total_qps = summary.get("total_qps_points", 0)
        if total_failed_qps > 0:
            print("\n" + "-" * 80)
            print(f"FAILED QPS CONFIGURATIONS: {total_failed_qps} out of {total_qps + total_failed_qps} total")
            print("-" * 80)
            for exp in results["experiments"]:
                failed_points = exp.get("failed_qps_points", [])
                if failed_points:
                    print(f"\n  {exp['experiment_name']}:")
                    for fp in failed_points:
                        print(f"    QPS {fp['qps']:.3f}: {fp['error']}")

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

    parser = argparse.ArgumentParser(description="Evaluate Vidur predictions against ground truth")
    parser.add_argument(
        "--ground-truth",
        default="eval/combined_ground_truth.json",
        help="Path to combined ground truth JSON (default: eval/combined_ground_truth.json)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file instead of printing to terminal"
    )
    parser.add_argument(
        "--output",
        default="eval/vidur_evaluation_results.json",
        help="Path to output results JSON (only used with --save)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = VidurEvaluator(args.ground_truth, verbose=args.verbose)
    results = evaluator.evaluate_all()

    # Print or save results
    if args.save:
        evaluator.save_results(results, args.output)
    else:
        evaluator.print_results(results)


if __name__ == "__main__":
    main()
