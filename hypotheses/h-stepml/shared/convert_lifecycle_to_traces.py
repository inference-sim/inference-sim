"""Convert ground-truth per_request_lifecycle_metrics.json to BLIS trace CSV.

The simulator's legacy trace replay (--workload traces) expects a 5-column CSV:
    arrival_time,request_id,model,"[token_ids]","[token_ids]"

This script reads the lifecycle JSON and generates synthetic token IDs of the
correct lengths, with relative arrival times.
"""

import json
import os
import random


def convert_lifecycle_to_csv(
    lifecycle_path: str,
    output_csv_path: str,
    model_name: str = "model",
    seed: int = 42,
) -> dict:
    """Convert a per_request_lifecycle_metrics.json to BLIS trace CSV.

    Args:
        lifecycle_path: Path to per_request_lifecycle_metrics.json.
        output_csv_path: Path to write the output CSV.
        model_name: Model name for the CSV (unused by simulator but required).
        seed: RNG seed for synthetic token ID generation.

    Returns:
        dict with: num_requests, min_arrival_s, max_arrival_s, duration_s
    """
    with open(lifecycle_path, "r") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"Empty lifecycle data: {lifecycle_path}")

    rng = random.Random(seed)

    # Find the earliest start_time to compute relative arrivals
    base_time = min(entry["start_time"] for entry in data)

    rows = []
    for i, entry in enumerate(data):
        info = entry.get("info", {})
        input_count = info.get("input_tokens", 0)
        output_count = info.get("output_tokens", 0)

        # Relative arrival time in seconds
        arrival_s = entry["start_time"] - base_time

        # Generate synthetic token IDs (simulator only cares about array lengths)
        input_tokens = [rng.randint(0, 31999) for _ in range(input_count)]
        output_tokens = [rng.randint(0, 31999) for _ in range(output_count)]

        rows.append((arrival_s, i, model_name, input_tokens, output_tokens))

    # Sort by arrival time (should already be sorted, but ensure)
    rows.sort(key=lambda x: x[0])

    with open(output_csv_path, "w") as f:
        f.write("arrival_time,request_id,model,prefill_tokens,decode_tokens\n")
        for arrival_s, req_id, model, inp, out in rows:
            f.write(
                f"{arrival_s:.6f},request_{req_id},{model},"
                f'"{json.dumps(inp)}","{json.dumps(out)}"\n'
            )

    return {
        "num_requests": len(rows),
        "min_arrival_s": rows[0][0],
        "max_arrival_s": rows[-1][0],
        "duration_s": rows[-1][0] - rows[0][0],
    }


def convert_experiment(
    experiment_dir: str,
    output_dir: str,
    seed: int = 42,
) -> str:
    """Convert a single experiment's lifecycle data to trace CSV.

    Args:
        experiment_dir: Path to the experiment directory.
        output_dir: Directory to write the trace CSV.
        seed: RNG seed.

    Returns:
        Path to the generated CSV file.
    """
    lifecycle_path = os.path.join(
        experiment_dir, "results", "per_request_lifecycle_metrics.json"
    )
    if not os.path.isfile(lifecycle_path):
        raise FileNotFoundError(f"No lifecycle data: {lifecycle_path}")

    exp_name = os.path.basename(experiment_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{exp_name}.csv")

    # Read exp-config for model name
    config_path = os.path.join(experiment_dir, "exp-config.yaml")
    model_name = "model"
    if os.path.isfile(config_path):
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_name = config.get("model", "model")

    convert_lifecycle_to_csv(lifecycle_path, csv_path, model_name, seed)
    return csv_path
