import json
import os 
import re
import subprocess

def get_unsaturated_exps(results_folder):
    # find only unsaturated configs to train over
    unsaturated_exps = []
    if os.path.isdir(results_folder):
        for _, _, filenames in os.walk(results_folder):
            for filename in filenames:
                rr = filename.split("_")[1][:-1]
                spec = filename.split("_")[2]
                mbnt = filename.split("_")[3].split(".")[0]
                unsaturated_exps.append({"rr": rr, "spec": spec, "mbnt": mbnt})
    return unsaturated_exps

def parse_vllm_metrics_to_json(folder, filepath):
    full_path = os.path.join(folder, filepath)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
        
    except FileNotFoundError:
        print(f"Error: The file at '{full_path}' was not found.")
        return None
    
def run_go_binary(arguments, go_binary_path, model_name, spec, mbnt, request_rate):
    result = subprocess.run(
        [go_binary_path] + arguments,
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8'
    )

    
    if result.stderr:
        print(
            f"Go binary error output:\n{result.stderr}")
    sim_metrics = parse_sim_metrics_to_json(result.stdout, model_name, spec, mbnt, request_rate)
    return sim_metrics
    
def parse_sim_metrics_to_json(stdout, model_name, spec, mbnt, request_rate):
    """
    Reads text from standard input, parses key-value metrics,
    and prints a single JSON object to standard output.
    """

    metrics_data = {}
    metric_pattern = re.compile(r'^\s*(.+?)\s*:\s*([\d\.]+)')
    metrics_data["model"] = model_name
    metrics_data["request_rate"] = request_rate
    metrics_data["spec"] = spec
    metrics_data["mbnt"] = mbnt

    for line in stdout.split('\n'):
        match = metric_pattern.search(line)
        if match:
            key = match.group(1).strip()
            key = key.rstrip(":")
            value_str = match.group(2)

            try:
                if '.' in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
                metrics_data[key] = value
            except ValueError:
                continue
    return metrics_data