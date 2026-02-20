#!/usr/bin/env python3
"""
BLIS Ground Truth Aggregation Script

This script aggregates BLIS postprocessing results from all ground truth
experiments into a single combined JSON file. It extracts:
- Model and vLLM configuration
- Workload configuration
- Total KV blocks
- Per-QPS server-side metrics (TTFT, ITL, E2E latencies)
"""

import json
import os
import re
import sys
import yaml
import pandas as pd
from pathlib import Path


# ============================================================================
# Utility Functions
# ============================================================================

def read_traces_jsonl(filepath):
    """
    vLLM traces is a JSONL file.
    This function loads raw data from traces file.
    """
    traces_raw_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            json_object = json.loads(line.strip())
            traces_raw_data.append(json_object)
    return traces_raw_data


def get_server_side_metrics_from_traces(traces_raw_data):
    """
    Fetch server-side info from traces file per request.
    We currently fetch vLLM requestID, input, output tokens,
    e2e latency, waiting, prefill and decode time per request.
    """
    all_requests = []
    for data in traces_raw_data:
        for resourceSpan in data["resourceSpans"]:
            for scopeSpan in resourceSpan["scopeSpans"]:
                for span in scopeSpan["spans"]:
                    request = {}
                    for attribute in span["attributes"]:
                        if attribute["key"] == "gen_ai.request.id":
                            request["request_id"] = attribute["value"]["stringValue"].rsplit("-0", 1)[0]
                        if attribute["key"] == "gen_ai.usage.prompt_tokens":
                            request["input_tokens"] = int(attribute["value"]["intValue"])
                        if attribute["key"] == "gen_ai.usage.completion_tokens":
                            request["output_tokens"] = int(attribute["value"]["intValue"])
                        if attribute["key"] == "gen_ai.latency.e2e":
                            request["e2e_latency"] = attribute["value"]["doubleValue"]
                        if attribute["key"] == "gen_ai.latency.time_to_first_token":
                            request["ttft"] = attribute["value"]["doubleValue"]
                        if attribute["key"] == "gen_ai.latency.time_in_queue":
                            request["queued_time"] = attribute["value"]["doubleValue"]
                        if attribute["key"] == "gen_ai.latency.time_in_model_prefill":
                            request["prefill_time"] = attribute["value"]["doubleValue"]
                        if attribute["key"] == "gen_ai.latency.time_in_model_decode":
                            request["decode_time"] = attribute["value"]["doubleValue"]
                    all_requests.append(request)
    return all_requests


def get_GuideLLM_rps_list(guidellm_results):
    """
    Read GuideLLM results file and extract list of
    uniformly spaced constant RPS values.
    """
    rps_list = []
    profile = guidellm_results["benchmarks"][0]["config"]["profile"]
    for strategies in profile["completed_strategies"]:
        if strategies["type_"] == "constant":
            rps_list.append(float(strategies["rate"]))
    return rps_list


def get_sweep_info(guidellm_results, rps_list):
    """
    Get details about each GuideLLM sweep trial (unique RPS).
    Details include: constant rps value and response-ids (vLLM requestIDs)
    """
    sweep_info = []
    for rps_idx, rps in enumerate(rps_list):
        current_sweep = {}
        current_sweep["rps"] = rps
        current_sweep["requestIDs"] = []
        # exclude synchronous(idx: 0) and throughput(idx: 1)
        all_requests = guidellm_results["benchmarks"][rps_idx + 2]["requests"]
        for req in all_requests["successful"]:
            current_sweep["requestIDs"].append(req["response_id"])
        sweep_info.append(current_sweep)
    return sweep_info


def extract_total_kv_blocks(log_file_path):
    """
    Finds the value of "Total KV blocks" for input to the simulator during testing.
    Extracts the number following 'num_gpu_blocks is:' from vllm's server log file.
    """
    total_kv_blocks = 0
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                match = re.search(r'num_gpu_blocks is:\s*(\d+)', line)
                if match:
                    total_kv_blocks += int(match.group(1))
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
    return total_kv_blocks


# ============================================================================
# Configuration Loading Functions
# ============================================================================

def load_yaml_config(filepath):
    """Load and parse a YAML configuration file."""
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filepath}: {e}")
        return None


def load_json_file(filepath):
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {filepath}: {e}")
        return None


# ============================================================================
# Experiment Processing
# ============================================================================

def process_experiment(exp_folder_path, exp_name):
    """
    Process a single ground truth experiment folder and extract all metrics.

    Args:
        exp_folder_path: Path to the experiment folder
        exp_name: Name of the experiment

    Returns:
        Dictionary containing experiment data, or None if processing failed
    """
    print(f"Processing experiment: {exp_name}")

    # Define file paths
    exp_config_path = os.path.join(exp_folder_path, "exp-config.yaml")
    profile_path = os.path.join(exp_folder_path, "profile.yaml")
    guidellm_results_path = os.path.join(exp_folder_path, "guidellm-results.json")
    vllm_log_path = os.path.join(exp_folder_path, "vllm.log")
    traces_path = os.path.join(exp_folder_path, "traces.json")

    # Load configurations
    vllm_config = load_yaml_config(exp_config_path)
    if vllm_config is None:
        print(f"  Skipping {exp_name}: Failed to load exp-config.yaml")
        return None

    workload_config = load_yaml_config(profile_path)
    if workload_config is None:
        print(f"  Skipping {exp_name}: Failed to load profile.yaml")
        return None

    guidellm_results = load_json_file(guidellm_results_path)
    if guidellm_results is None:
        print(f"  Skipping {exp_name}: Failed to load guidellm-results.json")
        return None

    # Extract total KV blocks from vLLM logs
    total_kv_blocks = extract_total_kv_blocks(vllm_log_path)
    if total_kv_blocks == 0:
        print(f"  Warning: Could not extract total_kv_blocks from {vllm_log_path}")

    # Get RPS list and sweep information from GuideLLM results (for request IDs)
    try:
        rps_list = get_GuideLLM_rps_list(guidellm_results)
        sweep_info = get_sweep_info(guidellm_results, rps_list)
    except Exception as e:
        print(f"  Skipping {exp_name}: Failed to extract sweep info: {e}")
        return None

    # Read traces and get server-side metrics
    try:
        traces_raw_data = read_traces_jsonl(traces_path)
        all_requests = get_server_side_metrics_from_traces(traces_raw_data)
        requests_df = pd.DataFrame(all_requests)
    except Exception as e:
        print(f"  Skipping {exp_name}: Failed to process traces: {e}")
        return None

    # Build QPS sweeps array with server-side latency metrics from traces
    qps_sweeps = []
    for sweep in sweep_info:
        # Filter trace data by request IDs for this sweep
        benchmark_request_ids = sweep["requestIDs"]
        benchmark_df = requests_df[requests_df["request_id"].isin(benchmark_request_ids)].copy()

        if len(benchmark_df) == 0:
            print(f"  Warning: No trace data found for sweep at RPS {sweep['rps']}")
            continue

        # Calculate server-side metrics from traces
        qps_data = {
            "qps": sweep["rps"],
            "ttft_mean_ms": benchmark_df["ttft"].mean() * 1e3,
            "ttft_p90_ms": benchmark_df["ttft"].quantile(0.9) * 1e3,
            "itl_mean_ms": (benchmark_df["prefill_time"] + benchmark_df["decode_time"]).sum() / benchmark_df["output_tokens"].sum() * 1e3,
            "e2e_mean_ms": benchmark_df["e2e_latency"].mean() * 1e3,
            "e2e_p90_ms": benchmark_df["e2e_latency"].quantile(0.9) * 1e3
        }
        qps_sweeps.append(qps_data)

    # For train experiments, sample only min, max, and middle QPS
    if vllm_config.get("app") == "train" and len(qps_sweeps) > 3:
        print(f"  Sampling 3 QPS values (min, middle, max) for train experiment")
        middle_idx = len(qps_sweeps) // 2
        qps_sweeps = [qps_sweeps[0], qps_sweeps[middle_idx], qps_sweeps[-1]]

    # Construct experiment object
    experiment_data = {
        "experiment_name": exp_name,
        "model": vllm_config.get("model", "unknown"),
        "vllm_config": {
            "tensor_parallelism": vllm_config.get("tensor_parallelism", 1),
            "max_model_len": vllm_config.get("max_model_len"),
            "max_num_batched_tokens": vllm_config.get("max_num_batched_tokens"),
            "max_num_seqs": vllm_config.get("max_num_seqs"),
            "app": vllm_config.get("app", "unknown")
        },
        "workload_config": {
            "rate_type": workload_config.get("rate-type", "sweep"),
            "max_requests": workload_config.get("max-requests"),
            "rate": workload_config.get("rate"),
            "data": workload_config.get("data", {})
        },
        "total_kv_blocks": total_kv_blocks,
        "qps_sweeps": qps_sweeps
    }

    print(f"  Successfully processed {exp_name} with {len(qps_sweeps)} QPS sweeps")
    return experiment_data


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main aggregation function."""
    # Define base path
    base_path = Path(__file__).resolve().parent.parent / "eval" / "ground_truth"

    print(f"Starting ground truth aggregation...")
    print(f"Base path: {base_path}\n")

    # Check if base path exists
    if not base_path.exists():
        print(f"Error: Ground truth directory not found: {base_path}")
        sys.exit(1)

    # Get all subdirectories in ground_truth folder
    experiment_folders = [d.name for d in base_path.iterdir() if d.is_dir()]

    if len(experiment_folders) == 0:
        print(f"Error: No experiment folders found in {base_path}")
        sys.exit(1)

    print(f"Found {len(experiment_folders)} experiment folder(s)\n")

    # Process each experiment
    all_experiments = []
    for exp_folder in sorted(experiment_folders):
        exp_path = base_path / exp_folder
        exp_data = process_experiment(exp_path, exp_folder)
        if exp_data is not None:
            all_experiments.append(exp_data)

    # Check if we successfully processed any experiments
    if len(all_experiments) == 0:
        print("\nError: No experiments were successfully processed.")
        sys.exit(1)

    # Build final output structure
    output_data = {
        "experiments": all_experiments
    }

    # Write to output file
    output_path = base_path.parent / "combined_ground_truth.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSuccess! Combined ground truth data written to: {output_path}")
        print(f"Total experiments processed: {len(all_experiments)}")
    except Exception as e:
        print(f"\nError writing output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
