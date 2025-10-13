import argparse
import copy
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading

import pandas as pd

from arrival_times_generation import add_arrival_delta, generate_arrival_times
from experiment_constants import *
from generate_random_prompts import generate_synthetic_requests

GO_BINARY_NAME = "simulation_worker"
# SIMULATION_BASE_DIR=/Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/inference-sim
BASE_DIR = os.getenv("SIMULATION_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)
DEFAULT_OUTPUT_DIR = "results/sweep_params"
DEFAULT_DATA_FOLDER = "data/"
DATASET_NAME = "sharegpt"

metrics_lock = threading.Lock()

def parse_metrics_to_json(stdout, mode, filename, model_name, request_rate, spec, prefix_hit_ratio, chunk_size, dataset_name):
    """
    Reads text from standard input, parses key-value metrics,
    and prints a single JSON object to standard output.
    """

    metrics_data = {}
    metric_pattern = re.compile(r'^\s*(.+?)\s*:\s*([\d\.]+)')
    metrics_data["model"] = model_name
    metrics_data["request_rate"] = request_rate
    if mode == "train":
        metrics_data["spec"] = spec
    metrics_data["prefix_ratio"] = prefix_hit_ratio
    metrics_data["chunk_size"] = chunk_size
    metrics_data["dataset"] = dataset_name

    for line in stdout.split('\n'):
        match = metric_pattern.search(line)
        if match:
            key = match.group(1).strip()
            key = key.rstrip(":")
            value_str = match.group(2)

            try:
                if '.' in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
                metrics_data[key] = value
            except ValueError:
                continue
    
    with open(filename, 'w+') as f:
        json.dump(metrics_data, f, indent=4)
    return metrics_data

def run_go_binary(thread_id, mode, arguments, model_name, spec, prefix_ratio, chunk_size, results, output_dir=DEFAULT_OUTPUT_DIR):
    print(' '.join(arguments))
    result = subprocess.run(
        [GO_BINARY_PATH] + arguments,
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8'
    )
    # print(result.stdout, flush=True)
    with metrics_lock:
        request_rate = int(float(arguments[2])*1e6)
        if mode == "train":
            output_filename = f"{output_dir}/exp_{request_rate}r_{spec}_{prefix_ratio}_{chunk_size}_{DATASET_NAME}.json"
        else:
            output_filename = f"{output_dir}/exp_{request_rate}r_{prefix_ratio}_{chunk_size}_{DATASET_NAME}.json"
        if result.stderr:
            print(
                f"[Thread {thread_id}] Go binary error output:\n{result.stderr}")
        else:
            metrics = parse_metrics_to_json(result.stdout, mode, output_filename, model_name, request_rate, spec, prefix_ratio, chunk_size, DATASET_NAME)
            results.append(pd.DataFrame([metrics]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Go binary with sweep over different parameters")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output results from simulation"
    )

    parser.add_argument(
        "--horizon",
        type=str,
        default="922337203685477580",
        help="Horizon in micosec(ticks) the simulation runs for at max"
    )

    parser.add_argument(
        '--mode',
        type=str,
        default="test",
        help="BLIS mode - train/test"
    )

    args = parser.parse_args()
    model_name = MODEL.split("/")[-1].replace(".", "_")
    output_dir = os.path.join(args.output_dir, model_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("--- Starting request rate sweep ---")

    if not os.path.exists(GO_BINARY_PATH):
        print(f"Error: Go binary not found at '{GO_BINARY_PATH}'.")

    if args.mode == "test":
        data_root_folder = f'{DEFAULT_DATA_FOLDER}/test/scenario4'
        os.makedirs(data_root_folder, exist_ok=True)
        for rr in REQUEST_RATES:
            for prefix_hit_ratio in PREFIX_HIT_RATIOS:
                generate_synthetic_requests(SEED, MODEL, NUM_PROMPTS, prefix_hit_ratio, 
                                                       INFERENCE_SPECS["INPUT_LEN_MEAN"], INFERENCE_SPECS["OUTPUT_LEN_MEAN"], 
                                                       rr, data_root_folder)

    args_template = [
        "run",
        "--rate", "0.000034",
        "--max-num-running-reqs", "8192",
        "--total-kv-blocks", "5000",
        "--max-num-scheduled-tokens", "256",
        "--block-size-in-tokens", "16",
        "--horizon", args.horizon,
        "--regression-coeffs", "1.17167255e-02,1.69822525e-05,1.86698155e-04",
        "--requests-file-path", "data/output_tokens_2025-06-30_arrivaldeltas.json",
        "--long-prefill-token-threshold", "256",
        "--queuing-coeffs", "1000, 1000",
        "--finished-coeffs", "1000, 1000",
        "--log", "warn"
    ]
    
    tasks = []
    thread_id = 1
    all_metrics_filepath = f"{args.output_dir}/simulator_{args.mode}_results.csv"
    all_metrics = []
    if args.mode == "test":
        SPECS = ['N/A']

    for rr in REQUEST_RATES:
        for spec in SPECS:
            for chunk_size in CHUNK_SIZES:
                for prefix_hit_ratio in PREFIX_HIT_RATIOS:
                    if args.mode == "train":
                        requests_folder = f"{BASE_DIR}/data/{args.mode}/scenario4/{model_name}/{spec.lower()}/chunk_size_{chunk_size}/rr_{rr}/prefix_{prefix_hit_ratio}"
                    else:
                        requests_folder = f"{BASE_DIR}/data/{args.mode}/scenario4/{model_name}/rr_{rr}/prefix_{prefix_hit_ratio}"
                    current_args = copy.deepcopy(args_template)
                    current_args[2] = str(rr / 1e6)
                    current_args[4] = str(MAX_NUM_SEQS)
                    current_args[6] = str(TOTAL_KV_BLOCKS[model_name])
                    current_args[8] = str(MAX_NUM_BATCHED_TOKENS)
                    current_args[10] = str(BLOCK_SIZE)
                    current_args[14] = ','.join(list(map(str, BETA_COEFFS[model_name])))
                    current_args[16] = os.path.join(requests_folder, "detailed_results_test_tokenized.json")
                    current_args[18] = str(chunk_size)
                    current_args[20] = ','.join(list(map(str, QUEUING_COEFFS[model_name])))
                    current_args[22] = ','.join(list(map(str, FINISHED_COEFFS[model_name])))

                    tasks.append({"thread_id": thread_id, "mode": args.mode, "args": current_args,
                                "output_dir": output_dir, "spec": spec, "prefix_ratio": prefix_hit_ratio, "chunk_size": chunk_size, "model": model_name, "results": all_metrics})
                    thread_id += 1

    threads = []
    for task in tasks:
        thread = threading.Thread(
            target=run_go_binary,
            args=(task["thread_id"], task["mode"], task["args"], task["model"], task["spec"], task["prefix_ratio"], task["chunk_size"], task["results"], task["output_dir"])
        )
        threads.append(thread)
        thread.start()  # Start the thread

    # Wait for all threads to complete.
    for thread in threads:
        thread.join()

    if os.path.exists(all_metrics_filepath):
        df = pd.read_csv(all_metrics_filepath)
        new_metrics = pd.concat(all_metrics, ignore_index=True)
        final_df = pd.concat([df, new_metrics], ignore_index=True)
    else:
        final_df = pd.concat(all_metrics, ignore_index=True)
    final_df.to_csv(all_metrics_filepath, index=False)
    print("--- All Go binary executions completed ---")
