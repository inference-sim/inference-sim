import argparse
import json
import os

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="A tokenizer for JSON requests in the ShareGPT format."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="LLM name for loading appropriate tokenizer"
    )

    args = parser.parse_args()

    # Load the tokenizer for given model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model_name = args.model_name.split("/")[-1].replace(".", "_")

    for rr in REQUEST_RATES:
        for spec in SPECS:
            for chunk_size in CHUNK_SIZES:
                for prefix_hit_ratio in PREFIX_HIT_RATIOS:
                    spec_small = spec.lower()
                    input_folder = f"../vllm-data-collection/results_new/scenario4/{model_name}/{spec_small}/chunk_size_{chunk_size}/rr_{rr}/prefix_{prefix_hit_ratio}"
                    output_folder = f"data/scenario4/{model_name}/{spec_small}/chunk_size_{chunk_size}/rr_{rr}/prefix_{prefix_hit_ratio}"
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
                                        for prompt in data["prompts"]:

                                            input_text = prompt["input_text"]
                                            tokenized_input_text = tokenizer.encode(input_text, return_tensors="np").tolist()[0]
                                            prompt["input_text"] = tokenized_input_text

                                            output_text = prompt["generated_text"]
                                            tokenized_output_text = tokenizer.encode(output_text, return_tensors="np").tolist()[0]
                                            prompt["generated_text"] = tokenized_output_text

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
    CHUNK_SIZES = [256, 2048]
    REQUEST_RATES = [5]
    PREFIX_HIT_RATIOS = [0.3, 0.6]
    SPECS = ["LL"]
    main()
