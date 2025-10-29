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

from generate_random_prompts import generate_synthetic_requests

GO_BINARY_NAME = "simulation_worker"
# e.g., SIMULATION_BASE_DIR=/Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/inference-sim
BASE_DIR = os.getenv("SIMULATION_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)


DEFAULT_OUTPUT_DIR =  os.path.join(BASE_DIR, "results/sweep_params") 
DEFAULT_DATA_FOLDER = os.path.join(BASE_DIR, "data") 

metrics_lock = threading.Lock()

def parse_metrics_to_json(stdout, filename, model_name, request_rate, spec, mbnt):
    """
    Reads text from standard input, parses key-value metrics,
    and prints a single JSON object to standard output.
    """

    metrics_data = {}
    metric_pattern = re.compile(r'^\s*(.+?)\s*:\s*([\d\.]+)')
    metrics_data["model"] = model_name
    metrics_data["request_rate"] = request_rate
    metrics_data["spec"] = spec
    metrics_data["mbnt"] = mbnt

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

def run_go_binary(thread_id, arguments, model_name, spec, mbnt, results, output_dir=DEFAULT_OUTPUT_DIR):
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
        request_rate = round(float(arguments[2])*1e6, 2)
        request_rate = f"{float(request_rate):.2f}"
        output_filename = f"{output_dir}/exp_{args.mode}_{request_rate}r_{spec}_{mbnt}.json"
        
        if result.stderr:
            print(
                f"[Thread {thread_id}] Go binary error output:\n{result.stderr}")
        metrics = parse_metrics_to_json(result.stdout, output_filename, model_name, request_rate, spec, mbnt)
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
        default="inference",
        help="BLIS mode - test/inference"
    )

    args = parser.parse_args()
    if args.mode == "test":
        from experiment_constants_test import *
    if args.mode == "inference":
        from experiment_constants_inference import *
    model_name = MODEL.split("/")[-1].replace(".", "_")
    output_dir = os.path.join(args.output_dir, model_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("--- Starting request rate sweep ---")

    if not os.path.exists(GO_BINARY_PATH):
        print(f"Error: Go binary not found at '{GO_BINARY_PATH}'.")

    if args.mode == "inference":
        # generate data on the fly for inference
        data_root_folder = f'{DEFAULT_DATA_FOLDER}/inference/scenario4'
        os.makedirs(data_root_folder, exist_ok=True)
        for rr in REQUEST_RATES:
            for prefix_hit_ratio in PREFIX_HIT_RATIOS:
                generate_synthetic_requests(SEED, MODEL, NUM_PROMPTS, prefix_hit_ratio, 
                                                       DATAGEN_SPECS["INPUT_LEN_MEAN"], DATAGEN_SPECS["OUTPUT_LEN_MEAN"], 
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
        "--long-prefill-token-threshold", "0",
        "--queuing-coeffs", "1000, 1000",
        "--finished-coeffs", "1000, 1000",
        "--log", "warn"
    ]
    
    tasks = []
    thread_id = 1
    all_metrics_filepath = f"{args.output_dir}/simulator_{args.mode}_results.csv"
    all_metrics = []
    request_rates = []
    if args.mode == "inference":
        SPECS = ['synthetic']
    if args.mode == "test":
        PREFIX_HIT_RATIOS = [0.1] # default value for exactly 1 loop execution

    for spec in SPECS:
        request_rates = []
        if isinstance(REQUEST_RATES, dict):
            request_rates = REQUEST_RATES[spec]
        else:
            request_rates = REQUEST_RATES
        for rr in request_rates:
            rr = f"{float(rr):.2f}"
            for mbnt in MAX_NUM_BATCHED_TOKENS:
                for prefix_hit_rate in PREFIX_HIT_RATIOS:
                    if args.mode == "test":
                        requests_folder = f"{BASE_DIR}/data/{args.mode}/scenario4/{model_name}/{spec.lower()}/mbnt_{mbnt}/rr_{rr}"
                    elif args.mode == "inference":
                        requests_folder = f"{BASE_DIR}/data/{args.mode}/scenario4/{model_name}/{spec.lower()}/rr_{rr}/prefix_{prefix_hit_rate}"
                    current_args = copy.deepcopy(args_template)
                    current_args[2] = str(float(rr) / 1e6)
                    current_args[6] = str(TOTAL_KV_BLOCKS[model_name])
                    current_args[8] = str(mbnt)
                    current_args[10] = str(BLOCK_SIZE)
                    current_args[14] = ','.join(list(map(str, BETA_COEFFS[model_name])))
                    current_args[16] = os.path.join(requests_folder, f"detailed_results_{args.mode}_tokenized.json")
                    current_args[20] = ','.join(list(map(str, QUEUING_COEFFS[model_name])))
                    current_args[22] = ','.join(list(map(str, FINISHED_COEFFS[model_name])))

                    tasks.append({"thread_id": thread_id, "mode": args.mode, "args": current_args,
                                "output_dir": output_dir, "spec": spec, "mbnt": mbnt, "model": model_name, "results": all_metrics})
                    thread_id += 1

    threads = []
    for task in tasks:
        thread = threading.Thread(
            target=run_go_binary,
            args=(task["thread_id"], task["args"], task["model"], task["spec"], task["mbnt"], task["results"], task["output_dir"])
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
