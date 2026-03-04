"""Generate BLIS workload-spec YAMLs and run commands for each experiment.

Reads profile.yaml, exp-config.yaml, and traces.json from each experiment
to produce:
  1. An inference-perf workload-spec YAML for `blis convert inference-perf`
  2. The `blis run` command with all matching flags
  3. Ground truth extraction from summary_lifecycle_metrics.json

Output: training/replay_data/<experiment>.yaml  (workload spec)
        training/replay_data/<experiment>_ground_truth.json

Usage:
    python3 training/generate_replay_specs.py
"""

from __future__ import annotations

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from split import EXPERIMENTS, ExperimentMeta, experiment_dir, config_json_path


def extract_kv_blocks_total(traces_path: str) -> int:
    """Extract kv.blocks_total_gpu from the first BATCH_SUMMARY event."""
    with open(traces_path) as f:
        for line in f:
            export = json.loads(line)
            for rs in export.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    if ss["scope"]["name"] == "vllm.scheduler.step":
                        for span in ss["spans"]:
                            for event in span["events"]:
                                if event["name"] == "step.BATCH_SUMMARY":
                                    attrs = {
                                        a["key"]: a["value"]
                                        for a in event["attributes"]
                                    }
                                    if "kv.blocks_total_gpu" in attrs:
                                        return int(
                                            attrs["kv.blocks_total_gpu"]["intValue"]
                                        )
    raise ValueError(f"No kv.blocks_total_gpu found in {traces_path}")


def count_requests(results_dir: str) -> tuple[int, int]:
    """Count success and failure requests from per_request_lifecycle_metrics.json."""
    path = os.path.join(results_dir, "per_request_lifecycle_metrics.json")
    with open(path) as f:
        data = json.load(f)
    n_ok = sum(1 for r in data if r.get("error") is None or r.get("error") == "")
    n_err = len(data) - n_ok
    return n_ok, n_err


def generate_spec(exp: ExperimentMeta, repo_root: str) -> dict:
    """Generate workload spec + run command for one experiment."""
    exp_path = experiment_dir(exp, repo_root)
    results_dir = os.path.join(exp_path, "results")

    # Load profile
    with open(os.path.join(exp_path, "profile.yaml")) as f:
        raw = f.read().strip()
    profile = json.loads(raw) if raw.startswith("{") else yaml.safe_load(raw)

    # Load exp-config
    with open(os.path.join(exp_path, "exp-config.yaml")) as f:
        exp_config = yaml.safe_load(f)

    # Extract KV blocks from traces
    kv_blocks = extract_kv_blocks_total(os.path.join(exp_path, "traces.json"))

    # Extract shared_prefix config
    sp = profile["data"]["shared_prefix"]
    stages = profile["load"]["stages"]

    # Build inference-perf spec YAML
    inf_perf_spec = {
        "version": "2",
        "inference_perf": {
            "stages": [{"rate": s["rate"], "duration": s["duration"]} for s in stages],
            "shared_prefix": {
                "num_unique_system_prompts": sp["num_unique_system_prompts"],
                "num_users_per_system_prompt": sp["num_users_per_system_prompt"],
                "system_prompt_len": sp["system_prompt_len"],
                "question_len": sp["question_len"],
                "output_len": sp["output_len"],
                # IMPORTANT: inference-perf's enable_multi_turn_chat controls the chat
                # template format, NOT context accumulation. Real requests have constant
                # input tokens (~system_prompt_len + question_len). BLIS's reasoning
                # multi-turn with ContextGrowth:"accumulate" is a different semantic.
                # Set to false so BLIS generates independent requests matching real data.
                "enable_multi_turn_chat": False,
            },
        },
    }

    # Compute total requests from real data
    n_ok, n_err = count_requests(results_dir)
    total_requests = n_ok + n_err

    # Compute horizon from stages
    total_duration_s = sum(s["duration"] for s in stages)
    horizon_us = total_duration_s * 1_000_000

    # Load summary for ground truth
    with open(os.path.join(results_dir, "summary_lifecycle_metrics.json")) as f:
        summary = json.load(f)

    succ = summary["successes"]
    ground_truth = {
        "experiment": exp.dir_name,
        "model_id": exp.model_id,
        "model_short": exp.model_short,
        "profile": exp.profile,
        "split": exp.split.value,
        "success_count": succ["count"],
        "failure_count": summary["failures"]["count"],
        "ttft": {
            "mean_ms": succ["latency"]["time_to_first_token"]["mean"] * 1000,
            "p50_ms": succ["latency"]["time_to_first_token"]["median"] * 1000,
            "p90_ms": succ["latency"]["time_to_first_token"]["p90"] * 1000,
            "p95_ms": succ["latency"]["time_to_first_token"]["p95"] * 1000,
            "p99_ms": succ["latency"]["time_to_first_token"]["p99"] * 1000,
        },
        "e2e": {
            "mean_ms": succ["latency"]["request_latency"]["mean"] * 1000,
            "p50_ms": succ["latency"]["request_latency"]["median"] * 1000,
            "p90_ms": succ["latency"]["request_latency"]["p90"] * 1000,
            "p95_ms": succ["latency"]["request_latency"]["p95"] * 1000,
            "p99_ms": succ["latency"]["request_latency"]["p99"] * 1000,
        },
        "throughput": {
            "requests_per_sec": succ["throughput"]["requests_per_sec"],
            "output_tokens_per_sec": succ["throughput"]["output_tokens_per_sec"],
            "total_tokens_per_sec": succ["throughput"]["total_tokens_per_sec"],
        },
        "config": {
            "model": exp_config["model"],
            "tensor_parallelism": exp_config["tensor_parallelism"],
            "max_num_seqs": exp_config["max_num_seqs"],
            "max_num_batched_tokens": exp_config["max_num_batched_tokens"],
            "kv_blocks_total_gpu": kv_blocks,
            "block_size": 16,
            "total_requests": total_requests,
            "horizon_us": horizon_us,
        },
    }

    # Build blis run command
    blis_cmd = (
        f"./blis run"
        f" --model {exp_config['model']}"
        f" --latency-model crossmodel"
        f" --hardware H100"
        f" --tp {exp_config['tensor_parallelism']}"
        f" --total-kv-blocks {kv_blocks}"
        f" --block-size-in-tokens 16"
        f" --max-num-running-reqs {exp_config['max_num_seqs']}"
        f" --max-num-scheduled-tokens {exp_config['max_num_batched_tokens']}"
        f" --num-instances 1"
        f" --num-requests {total_requests}"
        f" --horizon {horizon_us}"
    )

    return {
        "spec": inf_perf_spec,
        "ground_truth": ground_truth,
        "blis_cmd": blis_cmd,
    }


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(repo_root, "training", "replay_data")
    os.makedirs(output_dir, exist_ok=True)

    for exp in EXPERIMENTS:
        print(f"Generating spec for {exp.dir_name} ({exp.split.value})...")
        try:
            result = generate_spec(exp, repo_root)

            # Write workload spec YAML
            spec_path = os.path.join(output_dir, f"{exp.dir_name}.yaml")
            with open(spec_path, "w") as f:
                yaml.dump(result["spec"], f, default_flow_style=False)

            # Write ground truth JSON
            gt_path = os.path.join(output_dir, f"{exp.dir_name}_ground_truth.json")
            with open(gt_path, "w") as f:
                json.dump(result["ground_truth"], f, indent=2)

            gt = result["ground_truth"]
            print(f"  {gt['model_short']}/{gt['profile']}: "
                  f"{gt['success_count']}ok+{gt['failure_count']}err, "
                  f"KV={gt['config']['kv_blocks_total_gpu']}, "
                  f"TTFT p99={gt['ttft']['p99_ms']:.1f}ms")
            print(f"  cmd: {result['blis_cmd']}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
