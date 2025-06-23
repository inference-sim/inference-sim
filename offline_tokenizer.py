import argparse
import json

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
    parser.add_argument(
        "--input_filepath",
        type=str,
        help="JSON filepath with text inputs and outputs for requests"
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        help="JSON filepath to store tokenized inputs and outputs for request"
    )

    args = parser.parse_args()

    # Load the tokenizer for given model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    try:
        with open(args.input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_conversations = []

        for entry in data:
            conversations = entry["conversations"]
            
            # Process only the first two turns (human-gpt)
            if len(conversations) > 1 and conversations[0]["from"] == "human" and conversations[1]["from"] == "gpt":
                human_value = conversations[0]["value"]
                tokenized_human_value = tokenizer.encode(human_value, return_tensors="np").tolist()[0]
                conversations[0]["value"] = tokenized_human_value
            
                gpt_value = conversations[1]["value"]
                tokenized_gpt_value = tokenizer.encode(gpt_value, return_tensors="np").tolist()[0]
                conversations[1]["value"] = tokenized_gpt_value

                conversation_obj = {"ID": entry["id"], "conversations": []}
                conversation_obj["conversations"].append(conversations[0])
                conversation_obj["conversations"].append(conversations[1])
                all_conversations.append(conversation_obj)

        print ("Num tokenized requests:", len(all_conversations))
        with open(args.output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2)

    except FileNotFoundError:
        print(f"Error: The file at '{args.input_filepath}' was not found.")

if __name__=="__main__":
    main()