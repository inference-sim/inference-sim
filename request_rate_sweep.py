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
        "--regression-coeffs", "3.32362194e-05,1.08420984e-05,-4.50691525e-06,1.48830841e-03,4.37062592e-08,-1.08944527e-07,1.94495889e-07,2.62577887e-05,-7.53014907e-09,2.51275601e-08,7.01248262e-06,-9.23311658e-09,-1.00026507e-05,-9.31907150e-04,0.0057138352285822655",
        "--requests-file-path", "data/output_tokens_2025-06-30_arrivaldeltas.json",
        "--log", "warn"
    ]

    rates = [32]

    tasks = []
    for idx, rate in enumerate(rates):
        args_template[16] = f"data/output_tokens_2025-06-30_arrivaldeltas_rr={rate}.json"
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