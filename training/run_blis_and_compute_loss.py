#!/usr/bin/env python3
"""Standalone BLIS runner — run BLIS binary against ground-truth experiments.

This script runs BLIS simulations in parallel against ground-truth experiments
and computes loss metrics: RMSE[APE(TTFT mean)] + RMSE[APE(E2E mean)].

Output:
    - Silent operation (no progress messages)
    - Loss metrics output as JSON to stdout
    - Without --evaluate-per-experiment: returns overall loss only
    - With --evaluate-per-experiment: includes per-experiment breakdown

Usage:
    # Basic usage - returns overall loss as JSON
    python run_blis_and_compute_loss.py --latency-model roofline

    # With per-experiment breakdown
    python run_blis_and_compute_loss.py --latency-model roofline --evaluate-per-experiment

    # Custom configuration
    python run_blis_and_compute_loss.py \\
        --data-dir vllm_data/ground_truth \\
        --output-dir results \\
        --latency-model roofline \\
        --max-workers 8 \\
        --evaluate-per-experiment

Example Output (without --evaluate-per-experiment):
    {
      "ttft_rmse": 12.34,
      "e2e_rmse": 15.67,
      "overall_loss": 28.01,
      "num_experiments": 10,
      "num_succeeded": 10,
      "num_failed": 0
    }

Example Output (with --evaluate-per-experiment):
    {
      "ttft_rmse": 12.34,
      "e2e_rmse": 15.67,
      "overall_loss": 28.01,
      "num_experiments": 10,
      "num_succeeded": 10,
      "num_failed": 0,
      "per_experiment": [
        {
          "experiment_folder": "/path/to/exp1",
          "model": "qwen/qwen3-14b",
          "workload": "chatbot",
          "ttft_mean_ape": 10.5,
          "e2e_mean_ape": 12.3,
          "combined_loss": 22.8,
          "wall_clock_seconds": 45.2,
          "latency_ape": {
            "e2e": {"mean": 12.3, "p90": 15.4, "p99": 18.2},
            "ttft": {"mean": 10.5, "p90": 13.1, "p99": 16.8},
            "itl": {"mean": 8.7}
          },
          "throughput_ape": {
            "input_tokens_per_sec": 5.2,
            "output_tokens_per_sec": 7.8,
            "requests_per_sec": 6.1
          }
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import yaml

# Add experiment package to path
sys.path.insert(0, os.path.dirname(__file__))

from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)
from experiment.ground_truth import discover_experiments, parse_experiment
from experiment.metrics import ErrorRecord, RuntimeRecord, compute_errors


def normalize_hardware(hardware: str) -> str:
    """Normalize hardware names for BLIS."""
    if hardware == "A100-80GB":
        return "A100-80"
    return hardware


def write_workload_spec(experiment: Experiment, output_path: str) -> str:
    """Generate BLIS WorkloadSpec YAML from experiment profile config."""
    stages_config = experiment.profile_config["load"]["stages"]
    data_config = experiment.profile_config.get("data", {})
    sp = data_config.get("shared_prefix", data_config)

    total_requests = sum(round(s["rate"] * s["duration"]) for s in stages_config)

    spec = {
        "version": "2",
        "seed": 42,
        "num_requests": total_requests,
        "inference_perf": {
            "stages": [
                {"rate": float(s["rate"]), "duration": int(s["duration"])}
                for s in stages_config
            ],
            "shared_prefix": {
                "num_unique_system_prompts": int(sp.get("num_unique_system_prompts", 1)),
                "num_users_per_system_prompt": int(sp.get("num_users_per_system_prompt", 1)),
                "system_prompt_len": int(sp.get("system_prompt_len", 0)),
                "question_len": int(sp.get("question_len", 512)),
                "output_len": int(sp.get("output_len", 512)),
                "enable_multi_turn_chat": bool(sp.get("enable_multi_turn_chat", False)),
            },
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        yaml.dump(spec, fh, default_flow_style=False)
    return output_path


def build_blis_command(
    blis_binary: str,
    experiment: Experiment,
    workload_spec: str,
    results_path: str,
    latency_model: str,
) -> list[str]:
    """Build BLIS command-line arguments."""
    return [
        blis_binary,
        "run",
        "--model",
        experiment.model,
        "--tp",
        str(experiment.tp),
        "--hardware",
        normalize_hardware(experiment.hardware),
        "--max-num-running-reqs",
        str(experiment.max_num_seqs),
        "--max-num-scheduled-tokens",
        str(experiment.max_num_batched_tokens),
        "--total-kv-blocks",
        str(experiment.total_kv_blocks),
        "--kv-cpu-blocks",
        str(experiment.cpu_kv_blocks),
        "--kv-offload-threshold",
        "0.9",
        "--kv-transfer-bandwidth",
        "0.2",
        "--latency-model",
        latency_model,
        "--seed",
        "42",
        "--workload-spec",
        workload_spec,
        "--results-path",
        results_path,
    ]


def split_requests_by_stage(requests: list[dict], stages_config: list[dict]) -> list[list[dict]]:
    """Split requests into per-stage buckets by arrival time."""
    boundaries: list[float] = []
    cumulative = 0.0
    for s in stages_config:
        cumulative += s.get("duration", 0)
        boundaries.append(cumulative)

    stage_buckets: list[list[dict]] = [[] for _ in stages_config]
    for req in requests:
        for i, boundary in enumerate(boundaries):
            if req["arrived_at"] <= boundary or i == len(boundaries) - 1:
                stage_buckets[i].append(req)
                break
    return stage_buckets


def compute_stage_from_bucket(bucket: list[dict], stage_index: int, stage_cfg: dict) -> StageMetrics:
    """Compute percentile metrics for a single stage bucket."""
    required_keys = {"e2e_ms", "ttft_ms", "itl_ms", "num_prefill_tokens", "num_decode_tokens", "arrived_at"}
    valid = [r for r in bucket if required_keys.issubset(r)]

    zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
    if not valid:
        return StageMetrics(
            stage_index=stage_index,
            rate=float(stage_cfg.get("rate", 0)),
            duration=float(stage_cfg.get("duration", 0)),
            num_requests=0,
            e2e=zero_lat,
            ttft=zero_lat,
            itl=zero_lat,
            throughput=ThroughputMetrics(0, 0, 0),
        )

    e2e_vals = np.array([r["e2e_ms"] for r in valid])
    ttft_vals = np.array([r["ttft_ms"] for r in valid])
    itl_vals = np.array([r["itl_ms"] for r in valid])

    dur = max(1.0, stage_cfg.get("duration", 0))
    return StageMetrics(
        stage_index=stage_index,
        rate=float(stage_cfg.get("rate", 0)),
        duration=float(stage_cfg.get("duration", 0)),
        num_requests=len(valid),
        e2e=LatencyDistribution(
            mean=float(np.mean(e2e_vals)),
            p90=float(np.percentile(e2e_vals, 90)),
            p99=float(np.percentile(e2e_vals, 99)),
        ),
        ttft=LatencyDistribution(
            mean=float(np.mean(ttft_vals)),
            p90=float(np.percentile(ttft_vals, 90)),
            p99=float(np.percentile(ttft_vals, 99)),
        ),
        itl=LatencyDistribution(
            mean=float(np.mean(itl_vals)),
            p90=float(np.percentile(itl_vals, 90)),
            p99=float(np.percentile(itl_vals, 99)),
        ),
        throughput=ThroughputMetrics(
            input_tokens_per_sec=sum(r["num_prefill_tokens"] for r in valid) / dur,
            output_tokens_per_sec=sum(r["num_decode_tokens"] for r in valid) / dur,
            requests_per_sec=len(valid) / dur,
        ),
    )


def parse_blis_results(results_path: str, experiment: Experiment) -> SimulatorResult:
    """Parse BLIS JSON output into SimulatorResult."""
    with open(results_path) as fh:
        data = json.load(fh)

    stages_config = experiment.profile_config["load"]["stages"]
    total_duration = sum(s["duration"] for s in stages_config)

    # Summary metrics
    summary = StageMetrics(
        stage_index=-1,
        rate=0.0,
        duration=0.0,
        num_requests=data.get("completed_requests", 0),
        e2e=LatencyDistribution(
            mean=data.get("e2e_mean_ms", 0.0),
            p90=data.get("e2e_p90_ms", 0.0),
            p99=data.get("e2e_p99_ms", 0.0),
        ),
        ttft=LatencyDistribution(
            mean=data.get("ttft_mean_ms", 0.0),
            p90=data.get("ttft_p90_ms", 0.0),
            p99=data.get("ttft_p99_ms", 0.0),
        ),
        itl=LatencyDistribution(
            mean=data.get("itl_mean_ms", 0.0),
            p90=data.get("itl_p90_ms", 0.0),
            p99=data.get("itl_p99_ms", 0.0),
        ),
        throughput=ThroughputMetrics(
            input_tokens_per_sec=data.get("total_input_tokens", 0) / max(1.0, total_duration),
            output_tokens_per_sec=data.get("tokens_per_sec", 0),
            requests_per_sec=data.get("responses_per_sec", 0),
        ),
    )

    # Per-stage metrics
    raw_requests = data.get("requests", [])
    stage_buckets = split_requests_by_stage(raw_requests, stages_config)
    stages = [
        compute_stage_from_bucket(bucket, i, stages_config[i])
        for i, bucket in enumerate(stage_buckets)
    ]

    return SimulatorResult(
        adapter_name="blis",
        experiment_folder=experiment.folder,
        stages=stages,
        summary=summary,
    )


def run_blis_on_experiment(
    blis_binary: str, experiment: Experiment, latency_model: str
) -> SimulatorResult:
    """Run BLIS binary on a single experiment and return results."""
    blis_binary_abs = os.path.abspath(blis_binary)
    blis_dir = os.path.dirname(blis_binary_abs)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write workload spec
        spec_path = os.path.join(tmpdir, "workload_spec.yaml")
        write_workload_spec(experiment, spec_path)

        # Prepare results file
        results_path = os.path.join(tmpdir, "results.json")

        # Build command
        cmd = build_blis_command(blis_binary_abs, experiment, spec_path, results_path, latency_model)

        # Run BLIS
        result = subprocess.run(cmd, capture_output=True, cwd=blis_dir)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"BLIS failed (rc={result.returncode}): {stderr}")

        # Parse results
        return parse_blis_results(results_path, experiment)


@dataclass
class ExperimentLoss:
    """Per-experiment loss breakdown with full error metrics."""
    experiment_folder: str
    model: str
    workload: str
    ttft_mean_ape: float
    e2e_mean_ape: float
    combined_loss: float
    wall_clock_seconds: float
    # Latency errors (APE - Absolute Percentage Error), None if data unavailable
    e2e_p90_ape: float | None
    e2e_p99_ape: float | None
    ttft_p90_ape: float | None
    ttft_p99_ape: float | None
    itl_mean_ape: float | None
    # Throughput errors (APE), None if data unavailable
    input_tokens_per_sec_ape: float | None
    output_tokens_per_sec_ape: float | None
    requests_per_sec_ape: float | None


def compute_experiment_loss(
    error_records: list[ErrorRecord], runtime_seconds: float, result: SimulatorResult, experiment: Experiment
) -> ExperimentLoss | None:
    """Extract APE for all metrics from error records and compute throughput APE.

    Returns None if the required metrics are missing.
    """
    if not error_records:
        return None

    # Find summary metrics (stage_index == -1)
    summary_records = [r for r in error_records if r.stage_index == -1]

    # Build a map of metric_name -> APE
    ape_map = {rec.metric_name: rec.mape for rec in summary_records}

    # Extract required metrics for overall loss calculation
    ttft_mean_ape = ape_map.get("ttft_mean")
    e2e_mean_ape = ape_map.get("e2e_mean")

    if ttft_mean_ape is None or e2e_mean_ape is None:
        return None

    # For per-experiment, combine as sum
    combined = ttft_mean_ape + e2e_mean_ape

    # Helper to get APE with None if missing (indicates no data available)
    def get_ape(metric_name: str) -> float | None:
        return ape_map.get(metric_name)

    # Compute throughput APE from ground truth and predicted
    from experiment.metrics import compute_mape

    gt_throughput = experiment.summary.throughput
    pred_throughput = result.summary.throughput

    # Throughput APE might be inf if actual is 0, return None in that case
    def safe_mape(pred: float, actual: float) -> float | None:
        result = compute_mape(pred, actual)
        return None if result == float('inf') or result == float('-inf') else result

    input_tps_ape = safe_mape(pred_throughput.input_tokens_per_sec, gt_throughput.input_tokens_per_sec)
    output_tps_ape = safe_mape(pred_throughput.output_tokens_per_sec, gt_throughput.output_tokens_per_sec)
    rps_ape = safe_mape(pred_throughput.requests_per_sec, gt_throughput.requests_per_sec)

    return ExperimentLoss(
        experiment_folder=error_records[0].experiment_folder,
        model=error_records[0].model,
        workload=error_records[0].workload,
        ttft_mean_ape=ttft_mean_ape,
        e2e_mean_ape=e2e_mean_ape,
        combined_loss=combined,
        wall_clock_seconds=runtime_seconds,
        # Latency errors (APE) - None means data unavailable
        e2e_p90_ape=get_ape("e2e_p90"),
        e2e_p99_ape=get_ape("e2e_p99"),
        ttft_p90_ape=get_ape("ttft_p90"),
        ttft_p99_ape=get_ape("ttft_p99"),
        itl_mean_ape=get_ape("itl_mean"),
        # Throughput errors (APE) - None means data unavailable
        input_tokens_per_sec_ape=input_tps_ape,
        output_tokens_per_sec_ape=output_tps_ape,
        requests_per_sec_ape=rps_ape,
    )


def compute_overall_loss(experiment_losses: list[ExperimentLoss]) -> tuple[float, float, float]:
    """Compute overall loss: RMSE[APE(TTFT)] + RMSE[APE(E2E)].

    Returns:
        (ttft_rmse, e2e_rmse, combined_loss)
    """
    if not experiment_losses:
        return 0.0, 0.0, 0.0

    ttft_apes = np.array([loss.ttft_mean_ape for loss in experiment_losses])
    e2e_apes = np.array([loss.e2e_mean_ape for loss in experiment_losses])

    ttft_rmse = float(np.sqrt(np.mean(ttft_apes ** 2)))
    e2e_rmse = float(np.sqrt(np.mean(e2e_apes ** 2)))
    combined = ttft_rmse + e2e_rmse

    return ttft_rmse, e2e_rmse, combined


def discover_experiment_dirs(base_dir: str) -> list[str]:
    """Find experiment directories without requiring manifest."""
    import re
    # Match both patterns:
    # - Timestamp-based: YYYYMMDD-HHMMSS-model-tpN-workload
    # - ID-based: ID-model-tpN-workload-variant (with hyphens in suffix)
    pattern = re.compile(r"^\d+-.*-tp\d+-[\w-]+$")
    results = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and pattern.match(entry):
            results.append(os.path.abspath(full_path))
    results.sort()
    return results


def run_single_experiment_wrapper(
    args: tuple[int, int, Experiment, str, str]
) -> tuple[Experiment, list[ErrorRecord], RuntimeRecord | None, SimulatorResult | None, Experiment | None, Exception | None]:
    """Wrapper for running a single experiment (used by parallel execution)."""
    i, total, exp, blis_binary, latency_model = args

    try:
        t0 = time.perf_counter()
        result = run_blis_on_experiment(blis_binary, exp, latency_model)
        elapsed = time.perf_counter() - t0

        # Compute errors
        records = compute_errors(exp, result)

        # Track runtime
        runtime_record = RuntimeRecord(
            simulator="blis",
            experiment_folder=exp.folder,
            model=exp.model,
            workload=exp.workload,
            wall_clock_seconds=elapsed,
            exp_id=exp.exp_id,
            hardware=exp.hardware,
            dp=exp.dp,
            cpu_offload=exp.cpu_offload,
            gpu_mem_util=exp.gpu_mem_util,
            precision=exp.precision,
            tp=exp.tp,
            max_num_batched_tokens=exp.max_num_batched_tokens,
        )

        return exp, records, runtime_record, result, exp, None

    except Exception as exc:
        return exp, [], None, None, None, exc


@dataclass
class LossOutput:
    """Loss calculation output."""
    ttft_rmse: float
    e2e_rmse: float
    overall_loss: float
    num_experiments: int
    num_succeeded: int
    num_failed: int
    per_experiment: list[dict] | None = None


def run_evaluation(
    data_dir: str,
    blis_binary: str,
    output_dir: str,
    latency_model: str,
    max_workers: int = 4,
    evaluate_per_experiment: bool = False,
) -> LossOutput | None:
    """Run BLIS evaluation pipeline and return loss metrics."""

    # Try manifest-driven discovery first, fall back to directory scanning
    try:
        discovered = discover_experiments(data_dir)
        use_manifest = True
    except (FileNotFoundError, ValueError):
        # No manifest - discover from directories
        dirs = discover_experiment_dirs(data_dir)
        discovered = [(None, d) for d in dirs]
        use_manifest = False

    if not discovered:
        return None

    # Parse experiments
    experiments = []
    for manifest_entry, dir_path in discovered:
        try:
            experiments.append(parse_experiment(dir_path, manifest_entry=manifest_entry))
        except Exception:
            pass  # Skip failed experiments silently

    if not experiments:
        return None

    # Run BLIS on experiments in parallel
    error_records = []
    runtime_records = []
    experiment_losses = []
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        futures = {
            executor.submit(
                run_single_experiment_wrapper,
                (i, len(experiments), exp, blis_binary, latency_model)
            ): exp
            for i, exp in enumerate(experiments, 1)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            exp, records, runtime_record, result, ground_truth_exp, exc = future.result()

            if exc is None:
                error_records.extend(records)
                if runtime_record:
                    runtime_records.append(runtime_record)

                # Compute per-experiment loss
                runtime_seconds = runtime_record.wall_clock_seconds if runtime_record else 0.0
                if result and ground_truth_exp:
                    exp_loss = compute_experiment_loss(records, runtime_seconds, result, ground_truth_exp)
                    if exp_loss:
                        experiment_losses.append(exp_loss)

                success_count += 1
            else:
                fail_count += 1

    # Compute loss metrics
    if not experiment_losses:
        return None

    ttft_rmse, e2e_rmse, overall_loss = compute_overall_loss(experiment_losses)

    # Prepare per-experiment breakdown if requested
    per_experiment_data = None
    if evaluate_per_experiment:
        per_experiment_data = [
            {
                "experiment_folder": exp_loss.experiment_folder,
                "model": exp_loss.model,
                "workload": exp_loss.workload,
                "ttft_mean_ape": exp_loss.ttft_mean_ape,
                "e2e_mean_ape": exp_loss.e2e_mean_ape,
                "combined_loss": exp_loss.combined_loss,
                "wall_clock_seconds": exp_loss.wall_clock_seconds,
                "latency_ape": {
                    "e2e": {
                        "mean": exp_loss.e2e_mean_ape,
                        "p90": exp_loss.e2e_p90_ape,
                        "p99": exp_loss.e2e_p99_ape,
                    },
                    "ttft": {
                        "mean": exp_loss.ttft_mean_ape,
                        "p90": exp_loss.ttft_p90_ape,
                        "p99": exp_loss.ttft_p99_ape,
                    },
                    "itl": {
                        "mean": exp_loss.itl_mean_ape,
                    },
                },
                "throughput_ape": {
                    "input_tokens_per_sec": exp_loss.input_tokens_per_sec_ape,
                    "output_tokens_per_sec": exp_loss.output_tokens_per_sec_ape,
                    "requests_per_sec": exp_loss.requests_per_sec_ape,
                },
            }
            for exp_loss in sorted(experiment_losses, key=lambda x: x.combined_loss, reverse=True)
        ]

    return LossOutput(
        ttft_rmse=ttft_rmse,
        e2e_rmse=e2e_rmse,
        overall_loss=overall_loss,
        num_experiments=len(experiments),
        num_succeeded=success_count,
        num_failed=fail_count,
        per_experiment=per_experiment_data,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run BLIS binary against ground-truth experiments"
    )
    parser.add_argument(
        "--data-dir",
        default="trainval_data",
        help="Directory containing ground-truth experiments (default: trainval_data)",
    )
    parser.add_argument(
        "--blis-binary",
        default="../blis",
        help="Path to BLIS binary (default: ../blis)",
    )
    parser.add_argument(
        "--output-dir",
        default="validation_results",
        help="Output directory for reports (default: validation_results)",
    )
    parser.add_argument(
        "--latency-model",
        required=True,
        choices=["roofline", "blackbox", "crossmodel", "trained-roofline"],
        help="Latency model backend: roofline, blackbox, crossmodel, trained-roofline",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel experiment runs (default: 4)",
    )
    parser.add_argument(
        "--evaluate-per-experiment",
        action="store_true",
        help="Display per-experiment loss breakdown",
    )

    args = parser.parse_args()

    # Validate BLIS binary exists
    if not os.path.exists(args.blis_binary):
        sys.exit(1)

    # Run evaluation
    try:
        loss_output = run_evaluation(
            data_dir=args.data_dir,
            blis_binary=args.blis_binary,
            output_dir=args.output_dir,
            latency_model=args.latency_model,
            max_workers=args.max_workers,
            evaluate_per_experiment=args.evaluate_per_experiment,
        )

        if loss_output is None:
            sys.exit(1)

        # Output JSON to stdout
        output_dict = {
            "ttft_rmse": loss_output.ttft_rmse,
            "e2e_rmse": loss_output.e2e_rmse,
            "overall_loss": loss_output.overall_loss,
            "num_experiments": loss_output.num_experiments,
            "num_succeeded": loss_output.num_succeeded,
            "num_failed": loss_output.num_failed,
        }

        if loss_output.per_experiment is not None:
            output_dict["per_experiment"] = loss_output.per_experiment

        print(json.dumps(output_dict, indent=2))

    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
