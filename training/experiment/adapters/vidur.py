"""Vidur discrete-event simulator adapter."""

from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

from experiment.adapters.base import SimulatorAdapter
from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)
from experiment.ground_truth import resolve_perf_dir
from experiment.vidur_trace_converter import convert_to_vidur_trace

_SUPPORTED_MODELS = {
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
}

_HW_TO_VIDUR: dict[str, str] = {"H100": "h100", "A100-80GB": "a100"}
_HW_TO_VIDUR_NETWORK: dict[str, str] = {
    "H100": "h100_pairwise_nvlink",
    "A100-80GB": "a100_pairwise_nvlink",
}

# Vidur CSV columns (all times in seconds).
_COL_E2E = "request_e2e_time"
_COL_TTFT = "prefill_e2e_time"
_COL_ITL = "decode_time_execution_plus_preemption_normalized"
_COL_PREFILL_TOKENS = "request_num_prefill_tokens"
_COL_DECODE_TOKENS = "request_num_decode_tokens"

# Required columns for a valid request row.
_REQUIRED_COLS = {_COL_E2E, _COL_TTFT, _COL_ITL}


class VidurAdapter(SimulatorAdapter):
    """Adapter for the Vidur discrete-event simulator."""

    def __init__(self, vidur_dir: str):
        self.vidur_dir = vidur_dir

    @property
    def name(self) -> str:
        return "vidur"

    def can_run(self, experiment: Experiment) -> bool:
        return (experiment.model in _SUPPORTED_MODELS
                and experiment.hardware in _HW_TO_VIDUR
                and experiment.precision != "FP8"
                and not experiment.cpu_offload)

    def run(self, experiment: Experiment) -> SimulatorResult:
        if experiment.hardware not in _HW_TO_VIDUR:
            raise ValueError(
                f"Unsupported hardware '{experiment.hardware}' for {self.name} "
                f"(supported: {sorted(_HW_TO_VIDUR)})"
            )
        if experiment.precision == "FP8":
            raise ValueError(
                f"Unsupported precision '{experiment.precision}' for {self.name} "
                f"(Vidur has no FP8 device profiles)"
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Convert trace
            perf_dir = resolve_perf_dir(experiment.folder)
            per_req_path = os.path.join(perf_dir, "per_request_lifecycle_metrics.json")
            trace_csv = os.path.join(tmpdir, "trace.csv")
            convert_to_vidur_trace(per_req_path, trace_csv)

            # 2. Build args
            output_dir = os.path.join(tmpdir, "vidur_output")
            args = self._build_args(experiment, trace_csv, output_dir)

            # 3. Run
            try:
                subprocess.run(args, capture_output=True, check=True, cwd=self.vidur_dir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Vidur failed (rc={exc.returncode}) for "
                    f"{experiment.model}: {stderr}"
                ) from exc

            # 4. Parse — Vidur appends a timestamped subdir to output_dir
            csv_path = self._find_request_metrics_csv(output_dir)
            return self._parse_vidur_results(csv_path, experiment)

    def _build_args(
        self,
        experiment: Experiment,
        trace_csv: str,
        output_dir: str,
    ) -> list[str]:
        args = [
            sys.executable, "-m", "vidur.main",
            "--replica_config_model_name", experiment.model,
            "--replica_config_device", _HW_TO_VIDUR[experiment.hardware],
            "--replica_config_network_device", _HW_TO_VIDUR_NETWORK[experiment.hardware],
            "--replica_config_tensor_parallel_size", str(experiment.tp),
            "--replica_config_num_pipeline_stages", "1",
            "--replica_scheduler_config_type", "vllm",
            "--vllm_scheduler_config_batch_size_cap", str(experiment.max_num_seqs),
            "--vllm_scheduler_config_max_tokens_in_batch", str(experiment.max_num_batched_tokens),
            "--request_generator_config_type", "trace_replay",
            "--trace_request_generator_config_trace_file", trace_csv,
            "--metrics_config_output_dir", output_dir,
        ]
        num_replicas = str(int(experiment.dp)) if experiment.dp and experiment.dp > 1 else "1"
        args += ["--cluster_config_num_replicas", num_replicas]
        if experiment.dp and experiment.dp > 1:
            args += ["--global_scheduler_config_type", "round_robin"]
        return args

    @staticmethod
    def _find_request_metrics_csv(output_dir: str) -> str:
        """Find the request_metrics.csv inside the timestamped subdir."""
        pattern = os.path.join(output_dir, "*", "request_metrics.csv")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(
                f"No request_metrics.csv found under {output_dir}"
            )
        return matches[0]

    def _parse_vidur_results(
        self,
        csv_path: str,
        experiment: Experiment,
    ) -> SimulatorResult:
        """Parse Vidur request_metrics.csv into a SimulatorResult."""
        rows = self._read_csv(csv_path)

        try:
            stages_config = experiment.profile_config["load"]["stages"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Invalid profile_config for {experiment.folder}: "
                f"missing 'load.stages' key"
            ) from exc
        stage_buckets = self._split_rows_by_stage(rows, stages_config)

        stages: list[StageMetrics] = []
        for i, bucket in enumerate(stage_buckets):
            stages.append(self._compute_stage(bucket, i, stages_config[i]))

        summary = self._compute_summary(rows, stages_config)

        return SimulatorResult(
            adapter_name=self.name,
            experiment_folder=experiment.folder,
            stages=stages,
            summary=summary,
        )

    @staticmethod
    def _read_csv(csv_path: str) -> list[dict]:
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = []
            for row in reader:
                if not _REQUIRED_COLS.issubset(row):
                    continue
                try:
                    # Validate that required columns are parseable as float
                    for col in _REQUIRED_COLS:
                        float(row[col])
                except (ValueError, TypeError):
                    continue
                rows.append(row)
            return rows

    @staticmethod
    def _split_rows_by_stage(
        rows: list[dict],
        stages_config: list[dict],
    ) -> list[list[dict]]:
        """Split rows into per-stage buckets using arrival order.

        Vidur's request_metrics.csv rows are in request-ID order (which matches
        arrival order from the trace).  We split by cumulative request counts
        derived from rate × duration for each stage.
        """
        boundaries: list[int] = []
        cumulative = 0
        for s in stages_config:
            cumulative += round(s.get("rate", 0) * s.get("duration", 0))
            boundaries.append(cumulative)

        buckets: list[list[dict]] = [[] for _ in stages_config]
        for idx, row in enumerate(rows):
            assigned = False
            for i, boundary in enumerate(boundaries):
                if idx < boundary:
                    buckets[i].append(row)
                    assigned = True
                    break
            if not assigned and buckets:
                buckets[-1].append(row)
        return buckets

    @staticmethod
    def _compute_stage(
        bucket: list[dict],
        stage_index: int,
        stage_cfg: dict,
    ) -> StageMetrics:
        zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
        dur = max(1.0, stage_cfg.get("duration", 0))

        if not bucket:
            return StageMetrics(
                stage_index=stage_index,
                rate=float(stage_cfg.get("rate", 0)),
                duration=float(stage_cfg.get("duration", 0)),
                num_requests=0,
                e2e=zero_lat, ttft=zero_lat, itl=zero_lat,
                throughput=ThroughputMetrics(0, 0, 0),
            )

        # Vidur times are in seconds — convert to milliseconds.
        e2e_vals = np.array([float(r[_COL_E2E]) * 1000 for r in bucket])
        ttft_vals = np.array([float(r[_COL_TTFT]) * 1000 for r in bucket])
        itl_vals = np.array([float(r[_COL_ITL]) * 1000 for r in bucket])

        prefill_tokens = sum(int(float(r.get(_COL_PREFILL_TOKENS, 0))) for r in bucket)
        decode_tokens = sum(int(float(r.get(_COL_DECODE_TOKENS, 0))) for r in bucket)

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
                mean=float(np.mean(itl_vals)),
                p90=float(np.percentile(itl_vals, 90)),
                p99=float(np.percentile(itl_vals, 99)),
            ),
            throughput=ThroughputMetrics(
                input_tokens_per_sec=prefill_tokens / dur,
                output_tokens_per_sec=decode_tokens / dur,
                requests_per_sec=len(bucket) / dur,
            ),
        )

    def _compute_summary(
        self,
        rows: list[dict],
        stages_config: list[dict],
    ) -> StageMetrics:
        total_duration = sum(s.get("duration", 0) for s in stages_config)
        return self._compute_stage(
            rows,
            stage_index=-1,
            stage_cfg={"rate": 0, "duration": total_duration},
        )
