import argparse
import copy
import itertools
import os
import shutil
import subprocess
import threading

from arrival_times_generation import add_arrival_delta, generate_arrival_times
from experiment_constants import *

GO_BINARY_NAME = "simulation_worker"

GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)
DEFAULT_OUTPUT_DIR = "results/sweep_params"
DATASET_NAME = "sharegpt"
TEMPERATURE = 0.0

print_lock = threading.Lock()


def save_results(filename, output, arguments):
    with open(filename, "w") as f:
        f.write(' '.join(arguments))
        f.write("\n\n")
        f.write(output)


def run_go_binary(thread_id, arguments, num_requests, output_dir=DEFAULT_OUTPUT_DIR):
    result = subprocess.run(
        [GO_BINARY_PATH] + arguments,
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8'
    )
    # print(result.stdout, flush=True)
    with print_lock:
        request_rate = int(float(arguments[2])*1e6)
        long_prefill_token_threshold = int(arguments[26])
        max_num_scheduled_token = int(arguments[8])
        output_filename = f"{output_dir}/exp_{num_requests}p_{request_rate}r_{TEMPERATURE}t_{max_num_scheduled_token}mbt_{long_prefill_token_threshold}lpt_{DATASET_NAME}.txt"
        save_results(output_filename, result.stdout, arguments)
        if result.stderr:
            print(
                f"[Thread {thread_id}] Go binary error output:\n{result.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Go binary with sweep over different parameters")

    parser.add_argument(
        "--model",
        type=str,
        help="LLM name"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output results from simulation"
    )

    parser.add_argument(
        "--regression_coeffs", # in the order [intercept, gamma1, gamma2]
        type=str,
        default="1.17167255e-02,1.69822525e-05,1.86698155e-04",
    )

    parser.add_argument(
        "--horizon",
        type=str,
        default="922337203685477580",
        help="Horizon in micosec(ticks) the simulation runs for at max"
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("--- Starting request rate sweep ---")

    if not os.path.exists(GO_BINARY_PATH):
        print(f"Error: Go binary not found at '{GO_BINARY_PATH}'.")

    args_template = [
        "run",
        "--rate", "0.000034",
        "--max-num-running-reqs", args.max_num_seqs,
        "--total-kv-blocks", args.total_kv_blocks,
        "--max-num-scheduled-tokens", "256",
        "--block-size-in-tokens", args.block_size,
        "--horizon", args.horizon,
        "--regression-coeffs", args.regression_coeffs,
        "--requests-file-path", "data/output_tokens_2025-06-30_arrivaldeltas.json",
        "--long-prefill-token-threshold", "16",
    ]
    
    tasks = []
    thread_id = 1
    model_name = args.model.split("/")[-1].replace(".", "_")

    for rr in REQUEST_RATES:
        for spec in SPECS:
            for chunk_size in CHUNK_SIZES:
                for prefix_hit_ratio in PREFIX_HIT_RATIOS:
                    requests_folder = f"data/scenario4/{model_name}/{spec.lower()}/chunk_size_{chunk_size}/rr_{rr}/prefix_{prefix_hit_ratio}"
                    current_args = copy.deepcopy(args_template)
                    current_args[2] = str(rr / 1e6)
                    current_args[4] = str(MAX_NUM_SEQS)
                    current_args[6] = str(TOTAL_KV_BLOCKS[model_name])
                    current_args[8] = str(MAX_NUM_BATCHED_TOKENS)
                    current_args[10] = str(BLOCK_SIZE)
                    current_args[16] = os.path.join(requests_folder, "detailed_results_test_tokenized.json")
                    current_args[18] = str(chunk_size)

                    tasks.append({"thread_id": thread_id, "args": current_args,
                                "num_requests": NUM_PROMPTS, "output_dir": output_dir})
                    thread_id += 1

                threads = []
                for task in tasks:
                    thread = threading.Thread(
                        target=run_go_binary,
                        args=(task["thread_id"], task["args"], task["num_requests"], task["output_dir"])
                    )
                    threads.append(thread)
                    thread.start()  # Start the thread

                # Wait for all threads to complete.
                for thread in threads:
                    thread.join()

    print("--- All Go binary executions completed ---")
