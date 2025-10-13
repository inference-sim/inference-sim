import argparse
import json
import os

from transformers import AutoTokenizer
from experiment_constants import *

def main():
    parser = argparse.ArgumentParser(
        description="A tokenizer for JSON requests in the ShareGPT format."
    )

    parser.add_argument(
        "--results_path",
        type=str,
        help="Path for the scenario results folder"
    )

    args = parser.parse_args()

    # Load the tokenizer for given model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model_name = MODEL.split("/")[-1].replace(".", "_")
    results_dir = args.results_path

    for rr in REQUEST_RATES:
        for spec in SPECS:
            for chunk_size in CHUNK_SIZES:
                for prefix_hit_ratio in PREFIX_HIT_RATIOS:
                    spec_small = spec.lower()
                    input_folder = f"{results_dir}/{model_name}/{spec_small}/chunk_size_{chunk_size}/rr_{rr}/prefix_{prefix_hit_ratio}"
                    output_folder = f"data/train/scenario4/{model_name}/{spec_small}/chunk_size_{chunk_size}/rr_{rr}/prefix_{prefix_hit_ratio}"
                    if os.path.isdir(input_folder):
                        for input_dirpath, _, input_filenames in os.walk(input_folder):
                            for input_filename in input_filenames:
                                if input_filename == "detailed_results_test.json":
                                    full_path = os.path.join(input_dirpath, input_filename)
                                    try:
                                        with open(full_path, 'r', encoding='utf-8') as f:
                                            data = json.load(f)

                                        all_data = {}
                                        all_data["num_prompts"] = data["num_prompts"]
                                        all_data["request_rate"] = data["request_rate"]
                                        all_data["prompts"] = []
                                        start_timestamp = 0
                                        # final start timestamp for time.monotonic(). Every other timestamp will be a delta wrt this timestamp
                                        for event in data["prompts"][0]["events"]:
                                            if event["event_type"] == "SERVER_HIT":
                                                start_timestamp = event["timestamp"]
                                        for prompt in data["prompts"]:
                                            if prompt["error"]=="":
                                                input_text = prompt["input_text"]
                                                tokenized_input_text = tokenizer(input_text).input_ids
                                                prompt["input_text"] = tokenized_input_text

                                                output_text = prompt["generated_text"]
                                                tokenized_output_text = tokenizer(output_text).input_ids
                                                if len(tokenized_output_text) != prompt["output_len"]:
                                                    print(len(tokenized_output_text), prompt["output_len"])
                                                prompt["generated_text"] = tokenized_output_text
                                                for event in prompt["events"]:
                                                    if event["event_type"] == "SERVER_HIT":
                                                        prompt["arrival_time"] = int((event["timestamp"] - start_timestamp)*1e6)
                                                        break
                                                prompt.pop('events', None)
                                                prompt.pop('error', None)

                                                all_data["prompts"].append(prompt)

                                        print("Num tokenized requests:", len(all_data["prompts"]))
                                        os.makedirs(output_folder, exist_ok=True)
                                        output_filename = "detailed_results_test_tokenized.json"
                                        output_filepath = os.path.join(output_folder, output_filename)
                                        with open(output_filepath, 'w', encoding='utf-8') as f:
                                            json.dump(all_data, f, indent=2)

                                    except FileNotFoundError:
                                        print(f"Error: The file at '{full_path}' was not found.")


if __name__ == "__main__":
    main()
