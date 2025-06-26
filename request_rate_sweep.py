import subprocess
import threading
import os
import shutil
import sys
import platform

GO_BINARY_NAME = "simulation_worker"

GO_BINARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), GO_BINARY_NAME)
OUTPUT_DIR = "results/sweep_request_rate"

print_lock = threading.Lock()

def save_results(filename, output):
    with open (filename, "w+") as f:
        f.write(output)

def run_go_binary(thread_id, arguments):
    result = subprocess.run(
        [GO_BINARY_PATH] + arguments,
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8'
    )
    with print_lock:
        request_rate = int(float(arguments[2])*1e6)
        output_filename = f"{OUTPUT_DIR}/output_rr={request_rate}.txt"
        save_results(output_filename, result.stdout)
        if result.stderr:
            print(f"[Thread {thread_id}] Go binary error output:\n{result.stderr}")

if __name__ == "__main__":
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
        "--max-num-scheduled-tokens", "2048",
        "--block-size-in-tokens", "16",
        "--horizon", "10000000000",
        "--regression-coeffs", "7.00524792e-05,1.95214710e-08,1.09916161e-08,2.84607896e-03,0.004977576410358687",
        "--requests-file-path", "output_tokens_2025-06-26_tokenized.json",
    ]

    rates = [25, 30, 32, 34, 35, 38]

    tasks = []
    for idx, rate in enumerate(rates):
        tasks.append({"thread_id": idx+1, "args": args_template[:2] + [str(rate/1e6)] + args_template[3:]})

    threads = []
    for task in tasks:
        thread = threading.Thread(
            target=run_go_binary,
            args=(task["thread_id"], task["args"])
        )
        threads.append(thread)
        thread.start() # Start the thread

    # Wait for all threads to complete.
    for thread in threads:
        thread.join()

    print("--- All Go binary executions completed ---")