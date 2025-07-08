import argparse
import copy
import itertools
import os
import shutil
import subprocess
import threading

from arrival_times_generation import add_arrival_delta, generate_arrival_times

GO_BINARY_NAME = "simulation_worker"

GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)
OUTPUT_DIR = "results/sweep_params"
DATASET_NAME = "sharegpt"
TEMPERATURE = 0.0

print_lock = threading.Lock()


def save_results(filename, output, arguments):
    with open(filename, "w") as f:
        f.write(' '.join(arguments))
        f.write("\n\n")
        f.write(output)


def run_go_binary(thread_id, arguments, num_requests):
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
        output_filename = f"{OUTPUT_DIR}/exp_{num_requests}p_{request_rate}r_{TEMPERATURE}t_{max_num_scheduled_token}mbt_{long_prefill_token_threshold}lpt_{DATASET_NAME}.txt"
        save_results(output_filename, result.stdout, arguments)
        if result.stderr:
            print(
                f"[Thread {thread_id}] Go binary error output:\n{result.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Go binary with sweep over different parameters")

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random generation seed"
    )

    parser.add_argument(
        "--input_filename",
        type=str,
        help="Input ShareGPT tokenized requests filename"
    )

    parser.add_argument(
        "--num_requests",
        type=int,
        nargs='+',
        default=[400],
        help="Number of requests to process"
    )

    parser.add_argument(
        "--regression_coeffs",
        type=str,
        default="3.38283913e-05,9.82346868e-06,-3.11237143e-06,1.50291993e-03,4.24173346e-08,-1.06897441e-07,1.92844617e-07,2.60430816e-05,-7.72212201e-09,2.67059068e-08,7.20303280e-06,-1.06904337e-08,-1.05254706e-05,-9.19828725e-04,0.005708624032334771",
    )
    parser.add_argument(
        "--schedule_time",
        type=str,
        default="544",
    )
    parser.add_argument(
        "--update_time",
        type=str,
        default="80",
    )
    parser.add_argument(
        "--queue_overhead_time",
        type=str,
        default="1000",
    )
    parser.add_argument(
        "--vllm_overhead_time",
        type=str,
        default="6000",
    )
    parser.add_argument(
        "--rates",
        type=int,
        nargs='+',
        default=[32],
        help='An optional list of request arrival rates. Defaults to [32]'
    )

    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        nargs='+',
        default=[256],
        help='An optional list of max_num_scheduled_tokens. Defaults to [256]'
    )

    parser.add_argument(
        "--long_prefill_token_thresholds",
        type=int,
        nargs='+',
        default=[128],
        help='An optional list of long-prefill-token-thresholds. Defaults to [128]'
    )

    args = parser.parse_args()
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("--- Starting request rate sweep ---")

    if not os.path.exists(GO_BINARY_PATH):
        print(f"Error: Go binary not found at '{GO_BINARY_PATH}'.")

    args_template = [
        "run",
        "--rate", "0.000034",
        "--max-num-running-reqs", "256",
        "--total-kv-blocks", "94060",
        "--max-num-scheduled-tokens", "256",
        "--block-size-in-tokens", "16",
        "--horizon", "10000000000",
        "--regression-coeffs", args.regression_coeffs,
        "--requests-file-path", "data/output_tokens_2025-06-30_arrivaldeltas.json",
        "--schedule-time", args.schedule_time,
        "--update-time", args.update_time,
        "--queue-overhead-time", args.queue_overhead_time,
        "--vllm-overhead-time", args.vllm_overhead_time,
        "--long-prefill-token-threshold", "32",
    ]

    rates = args.rates
    num_requests = args.num_requests
    input_filename_root = args.input_filename.split(".json")[0]

    # timestamp tokenized requests file with arrival deltas based on num_requests and request_rate.
    for n in num_requests:
        for rate in rates:
            timestamped_filename = f"{input_filename_root}_arrivaldeltas_n={n}_rr={rate}.json"
            if os.path.exists(timestamped_filename):
                continue
            inter_arrival_times = list(
                generate_arrival_times(n - 1, rate, seed=args.seed))
            add_arrival_delta(args.input_filename,
                              inter_arrival_times, n, timestamped_filename)

    max_num_scheduled_tokens = args.max_num_batched_tokens
    long_prefill_token_thresholds = args.long_prefill_token_thresholds

    tasks = []
    thread_id = 1
    for n, rate, max_num_token, threshold in itertools.product(num_requests, rates, max_num_scheduled_tokens, long_prefill_token_thresholds):
        current_args = copy.deepcopy(args_template)
        current_args[2] = str(rate / 1e6)
        current_args[16] = f"{input_filename_root}_arrivaldeltas_n={n}_rr={rate}.json"
        current_args[8] = str(max_num_token)
        current_args[26] = str(threshold)

        tasks.append({"thread_id": thread_id, "args": current_args,
                     "num_requests": n})
        thread_id += 1

    threads = []
    for task in tasks:
        thread = threading.Thread(
            target=run_go_binary,
            args=(task["thread_id"], task["args"], task["num_requests"])
        )
        threads.append(thread)
        thread.start()  # Start the thread

    # Wait for all threads to complete.
    for thread in threads:
        thread.join()

    print("--- All Go binary executions completed ---")
