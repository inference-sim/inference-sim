"""LLMServingSim adapter for validation against vLLM ground-truth experiments."""

from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiment.data_model import Experiment

from experiment.adapters.base import SimulatorAdapter
from experiment.data_model import (
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)


# Map ground-truth model IDs (with suffixes) to LLMServingSim perf model names
# (without suffixes).  The Experiment.model field contains the full HuggingFace
# model ID from exp-config.yaml (e.g. "meta-llama/Llama-3.1-8B-Instruct").
# LLMServingSim's perf model directories drop the "-Instruct" suffix.
MODEL_MAP: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B",
    "mistralai/Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
}


def _sample_stages_proportionally(
    stages: list[dict], max_requests: int
) -> list[dict]:
    """Sample stages proportionally to ensure coverage of all stages.

    Args:
        stages: Original stage config with rate/duration
        max_requests: Maximum total requests to sample

    Returns:
        New stage config with adjusted durations to yield max_requests total

    Example:
        Original: [{"rate": 8, "duration": 600}, {"rate": 20, "duration": 600}]
                  → 4,800 + 12,000 = 16,800 requests
        With max_requests=100:
        - Stage 1: 4,800/16,800 * 100 = 29 requests → 29/8 = 3.625s duration
        - Stage 2: 12,000/16,800 * 100 = 71 requests → 71/20 = 3.55s duration
        Returns: [{"rate": 8, "duration": 3.625}, {"rate": 20, "duration": 3.55}]
    """
    # Calculate original total requests
    total_original = sum(
        round(stage["rate"] * stage["duration"]) for stage in stages
    )

    # If already under limit, return as-is
    if total_original <= max_requests:
        return stages

    # Calculate proportional request counts per stage
    sampled_stages = []
    remaining_requests = max_requests

    for i, stage in enumerate(stages):
        stage_original_requests = round(stage["rate"] * stage["duration"])

        # Last stage gets all remaining requests (handles rounding)
        if i == len(stages) - 1:
            stage_sampled_requests = remaining_requests
        else:
            # Proportional allocation
            proportion = stage_original_requests / total_original
            stage_sampled_requests = round(proportion * max_requests)
            remaining_requests -= stage_sampled_requests

        # Calculate new duration to yield sampled request count at same rate
        new_duration = stage_sampled_requests / stage["rate"]

        sampled_stages.append({
            "rate": stage["rate"],
            "duration": new_duration,
        })

    return sampled_stages


def _generate_arrivals(stages: list[dict]) -> list[float]:
    """Generate constant-rate arrival times from stage config.

    Args:
        stages: List of {"rate": req/s, "duration": seconds}

    Returns:
        List of arrival times in seconds
    """
    arrivals: list[float] = []
    t = 0.0

    for stage in stages:
        rate = stage["rate"]
        duration = stage["duration"]
        num_requests = round(rate * duration)
        inter_arrival = 1.0 / rate

        for _ in range(num_requests):
            arrivals.append(t)
            t += inter_arrival

    return arrivals


def _split_by_stage(rows: list[dict], stages: list[dict]) -> list[list[dict]]:
    """Split CSV rows by stage based on arrival time.

    Args:
        rows: CSV rows as dicts (arrival times in nanoseconds)
        stages: Stage config with rate/duration

    Returns:
        List of row buckets, one per stage
    """
    # Calculate stage boundaries in nanoseconds
    boundaries: list[float] = []
    cumulative_time = 0.0
    for stage in stages:
        cumulative_time += stage["duration"]
        boundaries.append(cumulative_time * 1e9)

    # Bucket rows by stage
    buckets: list[list[dict]] = [[] for _ in stages]
    for row in rows:
        arrival_ns = float(row["arrival"])
        for i, boundary_ns in enumerate(boundaries):
            if arrival_ns < boundary_ns:  # Strict < to assign boundary to next stage
                buckets[i].append(row)
                break
        else:
            # Fallback to last stage if beyond all boundaries
            buckets[-1].append(row)

    return buckets


class LLMServingSimAdapter(SimulatorAdapter):
    """Adapter for the LLMServingSim discrete-event simulator.

    Supports H100 hardware with Llama-3.1-8B and Mixtral-8x7B models.
    Generates workloads from ground-truth token counts with constant-rate
    arrivals, executes LLMServingSim via subprocess, and parses results
    into standardised metrics for comparison with vLLM ground truth.
    """

    def __init__(
        self,
        llmservingsim_dir: str,
        use_docker: bool = True,
        max_requests_per_experiment: int | None = None,
    ):
        """Initialize adapter.

        Args:
            llmservingsim_dir: Path to LLMServingSim directory containing
                main.py.
            use_docker: Whether to use Docker for execution (default True).
                If False or Docker is unavailable, falls back to native execution.
            max_requests_per_experiment: Optional limit on number of requests
                per experiment. If set, only the first N requests will be simulated.
                Useful for quick testing (e.g., 100 requests instead of 16,800).

        Raises:
            ValueError: If the directory does not contain main.py.
        """
        self.llmservingsim_dir = os.path.abspath(llmservingsim_dir)
        if not os.path.exists(os.path.join(self.llmservingsim_dir, "main.py")):
            raise ValueError(
                f"Invalid LLMServingSim directory: {llmservingsim_dir}. "
                "Must contain main.py"
            )

        # Set container name before checking Docker availability
        # The official LLMServingSim docker.sh creates "servingsim_docker"
        self.container_name = "servingsim_docker"

        # Check Docker availability
        self.use_docker = use_docker and self._is_docker_available()

        # Store request limit
        self.max_requests_per_experiment = max_requests_per_experiment

        # Ensure LLMServingSim is built if using Docker
        if self.use_docker:
            self._ensure_llmservingsim_built()

    @property
    def name(self) -> str:
        """Return adapter name for identification."""
        return "llmservingsim"

    def can_run(self, experiment: Experiment) -> bool:
        """Check if this adapter supports the given experiment.

        Returns True only if:
        - Hardware is H100
        - Model is in MODEL_MAP
        - Performance model exists for the TP configuration
        - Attention predictions exist (required for simulation)
        - Precision is FP16
        """
        # Check hardware
        if experiment.hardware != "H100":
            return False

        # Check model mapping
        model_sim = MODEL_MAP.get(experiment.model)
        if not model_sim:
            return False

        # Check perf model directory exists
        perf_model_path = os.path.join(
            self.llmservingsim_dir,
            f"llm_profile/perf_models/H100/{model_sim}/tp{experiment.tp}",
        )
        if not os.path.exists(perf_model_path):
            return False

        # Check that attention predictions exist
        # LLMServingSim requires these files to run simulations
        predictions_dir = os.path.join(perf_model_path, "predictions")
        attn_prefill_csv = os.path.join(predictions_dir, "attn_prefill_predictions.csv")
        attn_decode_csv = os.path.join(predictions_dir, "attn_decode_predictions.csv")

        if not os.path.exists(attn_prefill_csv) or not os.path.exists(attn_decode_csv):
            return False

        # Check precision
        if experiment.precision != "FP16":
            return False

        return True

    def _is_docker_available(self) -> bool:
        """Check if Docker is available and container is running.

        Returns:
            True if Docker is available and container exists, False otherwise.
        """
        try:
            # Check if docker command exists
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                return False

            # Check if container exists (running or stopped)
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return self.container_name in result.stdout

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _ensure_container_running(self) -> None:
        """Ensure Docker container is running, start it if stopped."""
        try:
            # Check if container is running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if self.container_name not in result.stdout:
                # Container exists but not running, start it
                subprocess.run(
                    ["docker", "start", self.container_name],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to start Docker container '{self.container_name}': {exc.stderr.decode('utf-8', errors='replace')}"
            )

    def _ensure_llmservingsim_built(self) -> None:
        """Ensure LLMServingSim is built inside Docker container.

        Checks if the AnalyticalAstra binary exists, and builds if necessary.
        """
        self._ensure_container_running()

        # Check if already built
        check_cmd = [
            "docker", "exec", self.container_name,
            "test", "-f", "/app/LLMServingSim/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
        ]

        result = subprocess.run(check_cmd, capture_output=True)
        if result.returncode == 0:
            # Already built
            return

        # Need to build - install dependencies and compile
        print(f"Building LLMServingSim in Docker container '{self.container_name}'...")

        # Install Python dependencies
        deps_cmd = [
            "docker", "exec", self.container_name,
            "pip3", "install", "-q",
            "pyyaml", "pyinstrument", "transformers", "datasets",
            "msgspec", "scikit-learn", "xgboost==3.1.2",
            "matplotlib==3.5.3", "pandas==1.5.3", "numpy==1.23.5"
        ]

        try:
            subprocess.run(deps_cmd, capture_output=True, check=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to install Python dependencies in Docker: {exc.stderr.decode('utf-8', errors='replace')}"
            )

        # Run compile.sh
        compile_cmd = [
            "docker", "exec", self.container_name,
            "bash", "-c", "cd /app/LLMServingSim && ./compile.sh"
        ]

        try:
            result = subprocess.run(compile_cmd, capture_output=True, check=True, timeout=1800)
            print("LLMServingSim built successfully in Docker.")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to compile LLMServingSim in Docker: {exc.stderr.decode('utf-8', errors='replace')}"
            )

    def _generate_cluster_config(self, experiment: "Experiment", output_path: str) -> None:
        """Generate cluster config JSON for the experiment.

        Args:
            experiment: Experiment configuration
            output_path: Where to write the cluster config JSON
        """
        # Load H100 template
        template_path = os.path.join(
            self.llmservingsim_dir,
            "cluster_config/single_node_single_instance_H100.json",
        )
        with open(template_path) as f:
            config = json.load(f)

        # Get LLMServingSim model name
        model_sim = MODEL_MAP[experiment.model]

        # Modify instance config
        instance = config["nodes"][0]["instances"][0]
        instance["model_name"] = model_sim
        instance["npu_num"] = experiment.tp
        instance["npu_group"] = 1  # npus_per_group = npu_num / npu_group = TP
        instance["npu_mem"]["mem_size"] = 80.0  # H100 HBM3

        # Handle multi-instance (dp > 1) - use deep copy to avoid shared dicts
        if experiment.dp and experiment.dp > 1:
            config["nodes"][0]["num_instances"] = experiment.dp
            config["nodes"][0]["instances"] = [
                copy.deepcopy(instance) for _ in range(experiment.dp)
            ]

        # Write config
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

    def _generate_workload(self, experiment: "Experiment", output_path: str) -> None:
        """Generate workload .jsonl file from ground-truth token counts.

        Args:
            experiment: Experiment configuration
            output_path: Where to write the workload .jsonl
        """
        from experiment.ground_truth import resolve_perf_dir

        # Read ground-truth metrics
        perf_dir = resolve_perf_dir(experiment.folder)
        metrics_path = os.path.join(perf_dir, "per_request_lifecycle_metrics.json")

        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"Ground-truth metrics not found: {metrics_path}. "
                "Cannot generate workload without token counts."
            )

        with open(metrics_path) as f:
            requests = json.load(f)

        # Extract token counts
        token_pairs = [
            (req["info"]["input_tokens"], req["info"]["output_tokens"])
            for req in requests
        ]

        # Generate constant-rate arrivals
        stages = experiment.profile_config["load"]["stages"]
        arrivals = _generate_arrivals(stages)

        # Check for mismatch
        if len(token_pairs) < len(arrivals):
            raise ValueError(
                f"Not enough ground-truth requests ({len(token_pairs)}) "
                f"for generated arrivals ({len(arrivals)})"
            )

        # Write workload .jsonl
        with open(output_path, "w") as f:
            for (input_toks, output_toks), arrival_sec in zip(token_pairs, arrivals):
                # Generate dummy token IDs (LLMServingSim only needs counts)
                input_tok_ids = list(range(1, input_toks + 1))

                record = {
                    "input_toks": input_toks,
                    "output_toks": output_toks,
                    "arrival_time_ns": int(arrival_sec * 1e9),
                    "input_tok_ids": input_tok_ids,
                }
                f.write(json.dumps(record) + "\n")

    def _build_cli_args(
        self,
        experiment: "Experiment",
        cluster_config: str,
        workload: str,
        output: str,
    ) -> list[str]:
        """Build LLMServingSim CLI arguments.

        Args:
            experiment: Experiment configuration
            cluster_config: Path to cluster config JSON
            workload: Path to workload .jsonl
            output: Path to output CSV

        Returns:
            List of CLI arguments
        """
        # Calculate total requests
        # Note: If max_requests_per_experiment is set, the experiment's stages
        # have already been proportionally sampled by run() method
        total_requests = sum(
            round(stage["rate"] * stage["duration"])
            for stage in experiment.profile_config["load"]["stages"]
        )

        args = [
            "python",
            "main.py",
            "--cluster-config",
            cluster_config,
            "--dataset",
            workload,
            "--output",
            output,
            "--fp",
            "16",
            "--block-size",
            "16",
            "--max-batch",
            str(experiment.max_num_seqs),
            "--max-num-batched-tokens",
            str(experiment.max_num_batched_tokens),
            "--num-req",
            str(total_requests),
            "--log-level",
            "WARNING",
        ]

        # Add routing policy for multi-instance
        if experiment.dp and experiment.dp > 1:
            args.extend(["--request-routing-policy", "RR"])

        return args

    def _parse_results(self, csv_path: str, experiment: "Experiment") -> SimulatorResult:
        """Parse LLMServingSim CSV output into SimulatorResult.

        Args:
            csv_path: Path to LLMServingSim output CSV
            experiment: Experiment configuration (for folder, stages)

        Returns:
            SimulatorResult with per-stage and summary metrics
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"LLMServingSim output CSV not found: {csv_path}")

        # Read CSV
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"LLMServingSim output CSV is empty: {csv_path}")

        # Get stage config
        stages_config = experiment.profile_config["load"]["stages"]

        # Split by stage
        buckets = _split_by_stage(rows, stages_config)

        # Compute per-stage metrics
        stage_metrics: list[StageMetrics] = []
        for i, bucket in enumerate(buckets):
            stage_metrics.append(self._compute_stage(bucket, i, stages_config[i]))

        # Compute summary (all rows together)
        total_duration = sum(s["duration"] for s in stages_config)
        summary = self._compute_stage(rows, -1, {"rate": 0, "duration": total_duration})

        return SimulatorResult(
            adapter_name=self.name,
            experiment_folder=experiment.folder,
            stages=stage_metrics,
            summary=summary,
        )

    @staticmethod
    def _compute_stage(
        bucket: list[dict],
        stage_index: int,
        stage_cfg: dict,
    ) -> StageMetrics:
        """Compute metrics for a stage.

        Args:
            bucket: CSV rows for this stage
            stage_index: Stage number (-1 for summary)
            stage_cfg: Stage config with rate/duration

        Returns:
            StageMetrics with nested LatencyDistribution and ThroughputMetrics
        """
        zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
        dur = max(1.0, stage_cfg.get("duration", 0))

        if not bucket:
            return StageMetrics(
                stage_index=stage_index,
                rate=float(stage_cfg.get("rate", 0)),
                duration=float(stage_cfg.get("duration", 0)),
                num_requests=0,
                e2e=zero_lat,
                ttft=zero_lat,
                itl=zero_lat,
                throughput=ThroughputMetrics(
                    input_tokens_per_sec=0.0,
                    output_tokens_per_sec=0.0,
                    requests_per_sec=0.0,
                ),
            )

        # Convert nanoseconds to milliseconds
        e2e_vals = np.array([float(r["latency"]) / 1e6 for r in bucket])
        ttft_vals = np.array([float(r["TTFT"]) / 1e6 for r in bucket])
        tpot_vals = np.array([float(r["TPOT"]) / 1e6 for r in bucket])

        # Calculate throughput
        input_tokens = sum(int(r["input"]) for r in bucket)
        output_tokens = sum(int(r["output"]) for r in bucket)

        return StageMetrics(
            stage_index=stage_index,
            rate=float(stage_cfg.get("rate", 0)),
            duration=float(stage_cfg.get("duration", 0)),
            num_requests=len(bucket),
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
                mean=float(np.mean(tpot_vals)),
                p90=float(np.percentile(tpot_vals, 90)),
                p99=float(np.percentile(tpot_vals, 99)),
            ),
            throughput=ThroughputMetrics(
                input_tokens_per_sec=input_tokens / dur,
                output_tokens_per_sec=output_tokens / dur,
                requests_per_sec=len(bucket) / dur,
            ),
        )

    def run(self, experiment: Experiment) -> SimulatorResult:
        """Run LLMServingSim simulation for the experiment.

        Args:
            experiment: Experiment configuration

        Returns:
            SimulatorResult with metrics

        Raises:
            RuntimeError: If LLMServingSim times out or exits with a
                non-zero return code.
        """
        # Apply proportional sampling if max_requests is set (and > 0)
        # max_requests_per_experiment of 0 means unlimited
        original_stages = experiment.profile_config["load"]["stages"]
        if self.max_requests_per_experiment and self.max_requests_per_experiment > 0:
            sampled_stages = _sample_stages_proportionally(
                original_stages, self.max_requests_per_experiment
            )
            # Temporarily modify experiment config with sampled stages
            experiment.profile_config["load"]["stages"] = sampled_stages
        else:
            sampled_stages = original_stages

        try:
            # Create temp directory inside LLMServingSim directory
            # This is necessary because LLMServingSim internally changes cwd to astra-sim/
            # and adds ../ prefix to all paths
            temp_dir = os.path.join(self.llmservingsim_dir, "temp_experiments")
            os.makedirs(temp_dir, exist_ok=True)

            # Use experiment ID for unique temp file names
            exp_id = os.path.basename(experiment.folder)
            cluster_config_path = os.path.join(temp_dir, f"{exp_id}_cluster.json")
            workload_path = os.path.join(temp_dir, f"{exp_id}_workload.jsonl")
            output_path = os.path.join(temp_dir, f"{exp_id}_output.csv")

            try:
                # Generate cluster config
                self._generate_cluster_config(experiment, cluster_config_path)

                # Generate workload
                self._generate_workload(experiment, workload_path)

                # Convert to relative paths from llmservingsim_dir
                # LLMServingSim expects paths like "temp_experiments/cluster.json"
                cluster_config_rel = os.path.relpath(cluster_config_path, self.llmservingsim_dir)
                workload_rel = os.path.relpath(workload_path, self.llmservingsim_dir)
                output_rel = os.path.relpath(output_path, self.llmservingsim_dir)

                # Build CLI args
                args = self._build_cli_args(
                    experiment,
                    cluster_config_rel,
                    workload_rel,
                    output_rel,
                )

                # Execute LLMServingSim (Docker or native)
                try:
                    if self.use_docker:
                        # Ensure container is running
                        self._ensure_container_running()

                        # Build docker exec command
                        # Run from /app/LLMServingSim working directory inside container
                        docker_args = [
                            "docker", "exec", self.container_name,
                            "bash", "-c",
                            f"cd /app/LLMServingSim && {' '.join(args)}"
                        ]

                        subprocess.run(
                            docker_args,
                            capture_output=True,
                            check=True,
                            timeout=3600,  # 1 hour timeout
                        )
                    else:
                        # Native execution (original behavior)
                        subprocess.run(
                            args,
                            capture_output=True,
                            check=True,
                            cwd=self.llmservingsim_dir,
                            timeout=3600,  # 1 hour timeout
                        )
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"LLMServingSim timed out after 1 hour for {experiment.folder}"
                    )
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr.decode("utf-8", errors="replace")
                    mode = "Docker" if self.use_docker else "native"
                    raise RuntimeError(
                        f"LLMServingSim ({mode}) failed (rc={exc.returncode}) for "
                        f"{experiment.folder}: {stderr}"
                    )

                # Parse results
                result = self._parse_results(output_path, experiment)

                # Clean up temp files
                for path in [cluster_config_path, workload_path, output_path]:
                    if os.path.exists(path):
                        os.remove(path)

                return result
            finally:
                # Clean up temp files even if parsing fails
                for path in [cluster_config_path, workload_path, output_path]:
                    if os.path.exists(path):
                        os.remove(path)
        finally:
            # Restore original stages
            experiment.profile_config["load"]["stages"] = original_stages
