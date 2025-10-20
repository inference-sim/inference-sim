import os
import json

import matplotlib.pyplot as plt
import pandas as pd

SATURATION_PERCENTAGE = 0.9

def get_metrics_from_file(folder, filepath):
    full_path = os.path.join(folder, filepath)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        mean_e2e = data["Mean E2E(ms)"]
        median_e2e = data["Median E2E(ms)"]
        p99_e2e = data["P99 E2E(ms)"]
        throughput = data["Request throughput (req/s)"]
        mean_active_steps = data["Mean Active Steps"]
        
    except FileNotFoundError:
        print(f"Error: The file at '{full_path}' was not found.")
    return mean_e2e, median_e2e, p99_e2e, throughput, mean_active_steps

def plot_vllm_vs_sim(data_df, groupby = ["model"]):
    grouped_df = data_df.groupby(groupby).mean(numeric_only=True)
    print(grouped_df)
    overall_errors = {}
    for group_idx in grouped_df.index:
        print(group_idx)
        plot_title = group_idx
        for metric in metrics:
            overall_errors[metric] = grouped_df.loc[group_idx, metric]
        plt.figure(figsize=(10, 6))
        colors = ['orange', 'red', 'green', 'blue', 'purple']
        plt.bar(list(overall_errors.keys()), list(overall_errors.values()), label=list(overall_errors.keys()), color=colors)
        
        plt.title(f'Percentage error - vllm vs sim - {plot_title}')
        plt.xlabel("Metrics")
        plt.ylabel("Error %")
        plt.legend()
        plots_folder = f"analysis_results_{mode}"
        os.makedirs(plots_folder, exist_ok=True)
        plt.savefig(f'{plots_folder}/{plot_title}_error.png')

def aggregate_results():
    all_data = []
    for model_name in models:
        row = {}
        for spec in specs[model_name]:
            for rr in REQUEST_RATES:
                for chunk_size in CHUNK_SIZES:
                    if mode == "val":
                        PREFIX_HIT_RATIOS = [0]
                    for prefix_hit_ratio in PREFIX_HIT_RATIOS:
                        if mode == "train":
                            row = {"model": model_name, "spec": spec, "rr": rr, "chunk_size": chunk_size, "prefix_ratio": prefix_hit_ratio}
                            vllm_filename = f"vllm_{rr}r_{spec}_{prefix_hit_ratio}_{chunk_size}_sharegpt.json"
                            sim_filename = f"exp_{rr}r_{spec}_{prefix_hit_ratio}_{chunk_size}_sharegpt.json"
                        else:
                            row = {"model": model_name, "spec": spec, "rr": rr, "chunk_size": chunk_size}
                            vllm_filename = f"vllm_{rr}r_{spec}_{chunk_size}_sharegpt.json"
                            sim_filename = f"exp_val_{rr}r_{spec}_{chunk_size}_sharegpt.json"
                        vllm_results_folder = f"../vllm-data-collection/scenario4/results_server_side/{model_name}"
                        vllm_metrics = get_metrics_from_file(vllm_results_folder, vllm_filename)
                        if vllm_metrics[3] >= SATURATION_PERCENTAGE * rr:
                            print(f"{spec}, {rr}, {chunk_size}")
                            sim_results_folder = f"results/sweep_params/{model_name}"
                            sim_metrics = get_metrics_from_file(sim_results_folder, sim_filename)
                            for idx, metric in enumerate(metrics):
                                mape = abs(sim_metrics[idx] - vllm_metrics[idx])/vllm_metrics[idx] * 100
                                row[metric] = mape
                            all_data.append(row)
    return pd.DataFrame(all_data)

mode = "val"
if mode == "val":
    from experiment_constants_val import *
else:
    from experiment_constants import *
models = ["Qwen2_5-7B"]
metrics = ["mean_e2e_error", "median_e2e_error", "p99_e2e_error", "throughput_error"]
specs = {
    "Qwen2_5-7B": ["Summarization", "Chatbot", "Classification"],
    # "Qwen3-14B": ["LL", "LH"]
}
all_data = aggregate_results()
plot_vllm_vs_sim(all_data, groupby=["model", "rr"])

