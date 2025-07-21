# generate a set of n arrival times a r req/sec with a poisson distribution
import argparse
import json
import numpy as np

def generate_arrival_times(num_reqs, arrival_rate, burstiness = 1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    theta = 1 / (arrival_rate * burstiness)
    inter_arrival_times = np.random.gamma(shape=burstiness, scale=theta, size=num_reqs) * 1e6

    inter_arrival_times = np.array([0] + list(inter_arrival_times)).astype(int)  # Ensure the first arrival time is 0
    
    return inter_arrival_times

import json

def add_arrival_delta(json_filepath, arrival_deltas_list, num_requests, output_filepath=None):
    """
    Adds an 'arrival delta' field to each entry in a JSON file.

    Args:
        json_filepath (str): The path to the input JSON file.
        arrival_deltas_list (list): A list of values to be added as 'arrival delta'.
                                    The length of this list must match the number
                                    of entries in the JSON file.
        output_filepath (str, optional): The path to save the modified JSON file.
                                         If None, the original file will be overwritten.
                                         Defaults to None.
    """
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_filepath}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_filepath}'. Please check the file format.")
        return

    data = data[:num_requests]
    if len(data) != len(arrival_deltas_list):
        print(f"Warning: The number of entries in the JSON file ({len(data)}) "
              f"does not match the number of arrival delta values ({len(arrival_deltas_list)}). "
              f"Some entries might not get an 'arrival delta' or there might be unused values.")

    for i, entry in enumerate(data):
        if i < len(arrival_deltas_list):
            entry["arrivalDelta"] = int(arrival_deltas_list[i])

    if output_filepath is None:
        output_filepath = json_filepath

    try:
        with open(output_filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully added 'arrival delta' to entries and saved to '{output_filepath}'.")
    except IOError as e:
        print(f"Error writing to file '{output_filepath}': {e}")

def main():
    parser = argparse.ArgumentParser(
            description="A request arrival times generator for Gamma/Exponential Distribution"
            )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random generation seed"
    )

    parser.add_argument(
        "--input_filename",
        type=str,
        help="Input ShareGPT requests filename"
    )

    parser.add_argument(
        "--num_requests",
        type=int,
        help="Number of requests to process"
    )

    args = parser.parse_args()
    rates = [4, 16, 30, 64]
    for rate in rates:
        output_filename = f"data/output_tokens_2025-07-07_arrivaldeltas_rr={rate}.json"
        inter_arrival_times = list(generate_arrival_times(args.num_requests - 1, rate, seed = args.seed))
        add_arrival_delta(args.input_filename, inter_arrival_times, args.num_requests, output_filename)


if __name__=="__main__":
    main()

