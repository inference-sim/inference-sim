import argparse
import json
import os

from transformers import AutoTokenizer
from experiment_configs_scenario5 import config

parser = argparse.ArgumentParser(
    description="A tokenizer for JSON requests."
)

parser.add_argument(
    "--results_path",
    type=str,
    help="Path for the scenario results folder"
)

parser.add_argument(
    "--mode",
    type=str,
    help="train/test"
)

args = parser.parse_args()

# Load the tokenizer for given model
tokenizer = AutoTokenizer.from_pretrained(config["MODEL"], trust_remote_code=True, revision=None)
model_name = config["MODEL"].split("/")[-1].replace(".", "_")
results_dir = args.results_path

def get_unsaturated_exps(model_name):
    # find only unsaturated configs to train over
    unsaturated_exps = []
    results_folder = f"../vllm-data-collection/scenario4/results_server_side/{model_name}/train"
    if os.path.isdir(results_folder):
        for _, _, filenames in os.walk(results_folder):
            for filename in filenames:
                rr = filename.split("_")[1][:-1]
                spec = filename.split("_")[2]
                mbnt = filename.split("_")[3].split(".")[0]
                unsaturated_exps.append({"rr": rr, "spec": spec, "mbnt": mbnt})
    return unsaturated_exps

unsatured_exps = get_unsaturated_exps(model_name)
for exp in unsatured_exps:
    spec = exp["spec"]
    rr = exp["rr"]
    mbnt = exp["mbnt"]
    rr = f"{float(rr):.2f}"
    input_folder = f"{results_dir}/{model_name}/{args.mode}/{spec}/mbnt_{mbnt}/rr_{rr}"
    output_folder = f"data/{args.mode}/scenario4/{model_name}/{spec}/mbnt_{mbnt}/rr_{rr}"
    print(input_folder)
    if os.path.isdir(input_folder):
        for input_dirpath, _, input_filenames in os.walk(input_folder):
            for input_filename in input_filenames:
                if input_filename == f"detailed_results_{args.mode}.json":
                    full_path = os.path.join(input_dirpath, input_filename)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        all_data = {}
                        all_data["num_prompts"] = data["num_prompts"]
                        all_data["request_rate"] = data["request_rate"]
                        all_data["prompts"] = []
                        start_timestamp = 0
                        total_input_tokens = 0
                        total_output_tokens = 0
                        # final start timestamp for time.monotonic(). Every other timestamp will be a delta wrt this timestamp
                        for event in data["prompts"][0]["events"]:
                            if event["event_type"] == "SERVER_HIT":
                                start_timestamp = event["timestamp"]
                        for prompt in data["prompts"]:
                            if prompt["error"]=="":
                                input_text = prompt["input_text"]
                                tokenized_input_text = tokenizer(input_text).input_ids
                                total_input_tokens += len(tokenized_input_text)
                                prompt["input_text"] = tokenized_input_text
                                # if len(tokenized_input_text) != prompt["input_len"]:
                                #     print("input mismatch:", len(tokenized_input_text), prompt["input_len"])

                                output_text = prompt["generated_text"]
                                tokenized_output_text = tokenizer(output_text).input_ids
                                total_output_tokens += len(tokenized_output_text)
                                # if len(tokenized_output_text) != prompt["output_len"]:
                                #     print("output mismatch:", len(tokenized_output_text), prompt["output_len"])
                                
                                prompt["generated_text"] = tokenized_output_text
                                for event in prompt["events"]:
                                    if event["event_type"] == "SERVER_HIT":
                                        prompt["arrival_time"] = int((event["timestamp"] - start_timestamp)*1e6)
                                        break
                                prompt.pop('events', None)
                                prompt.pop('error', None)

                                all_data["prompts"].append(prompt)

                        print("Num tokenized requests:", len(all_data["prompts"]))
                        print("Input", total_input_tokens, data["total_input_tokens"])
                        print("Output", total_output_tokens, data["total_output_tokens"])
                        os.makedirs(output_folder, exist_ok=True)
                        output_filename = f"detailed_results_{args.mode}_tokenized.json"
                        output_filepath = os.path.join(output_folder, output_filename)
                        with open(output_filepath, 'w', encoding='utf-8') as f:
                            json.dump(all_data, f, indent=2)

                    except FileNotFoundError:
                        print(f"Error: The file at '{full_path}' was not found.")