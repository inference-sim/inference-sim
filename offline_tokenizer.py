from transformers import AutoTokenizer
import json

# Load the tokenizer for facebook/opt-125m
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
filepath = "ShareGPT_V3_unfiltered_cleaned_split.json"
output_filepath = "ShareGPT_V3_tokenized.json"

try:
    with open(filepath, 'r', encoding='utf-8') as f:
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

    print (len(all_conversations))
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, indent=2)

except FileNotFoundError:
    print(f"Error: The file at '{filepath}' was not found.")