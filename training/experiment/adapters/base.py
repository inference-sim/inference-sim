"""Base classes for simulator adapters.

SimulatorAdapter
    Abstract base class that all simulator adapters implement.
BaseBLISAdapter
    Shared logic for the three BLIS adapter variants (blackbox, roofline,
    crossmodel): CLI argument construction, result parsing, and per-stage
    request splitting.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import numpy as np
import yaml

from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)


class SimulatorAdapter(ABC):
    """Interface that every simulator adapter must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this adapter (used in reports)."""
        ...

    def can_run(self, experiment: Experiment) -> bool:
        """Return True if this adapter supports the given experiment."""
        return True

    @abstractmethod
    def run(self, experiment: Experiment) -> SimulatorResult:
        """Execute the simulator and return predicted metrics."""
        ...


class BaseBLISAdapter(SimulatorAdapter):
    """Shared logic for all BLIS simulator modes.

    Subclasses override ``run()`` to add mode-specific CLI flags
    (e.g. ``--latency-model roofline``).
    """

    def __init__(self, blis_binary: str):
        self.blis_binary = os.path.abspath(blis_binary)
        self._blis_dir = os.path.dirname(self.blis_binary)

    @staticmethod
    def _normalize_hardware(hardware: str) -> str:
        """Normalize manifest hardware names to BLIS-compatible names.

        The manifest uses "A100-80GB" but BLIS expects "A100-80" in its
        hardware_config.json and defaults.yaml. Other hardware types
        (H100, L40S) pass through unchanged.
        """
        if hardware == "A100-80GB":
            return "A100-80"
        return hardware

    def _build_common_args(
        self,
        experiment: Experiment,
        trace_spec: str,
        results_path: str,
    ) -> list[str]:
        """Build CLI arguments shared by all BLIS modes."""
        return [
            self.blis_binary, "run",
            "--model", experiment.model,
            "--tp", str(experiment.tp),
            "--hardware", self._normalize_hardware(experiment.hardware),
            "--max-num-running-reqs", str(experiment.max_num_seqs),
            "--max-num-scheduled-tokens", str(experiment.max_num_batched_tokens),
            "--total-kv-blocks", str(experiment.total_kv_blocks),
            "--kv-cpu-blocks", str(experiment.cpu_kv_blocks),
            "--kv-offload-threshold", "0.9",
            "--kv-transfer-bandwidth", "0.2",
            "--seed", "42",
            "--workload-spec", trace_spec,
            "--results-path", results_path,
        ]

    @staticmethod
    def _write_workload_spec(experiment: Experiment, output_path: str) -> str:
        """Generate a BLIS WorkloadSpec YAML from the experiment's profile_config.

        Uses the ``inference_perf`` format which BLIS expands into synthetic
        clients matching the experiment's rate/duration stages and token
        distributions.
        """
        stages_config = experiment.profile_config["load"]["stages"]
        data_config = experiment.profile_config.get("data", {})
        sp = data_config.get("shared_prefix", data_config)

        total_requests = sum(
            round(s["rate"] * s["duration"]) for s in stages_config
        )

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

    def _split_requests_by_stage(
        self,
        requests: list[dict],
        stages_config: list[dict],
    ) -> list[list[dict]]:
        """Split a flat request list into per-stage buckets by arrival time.

        Stage boundaries are cumulative durations from ``stages_config``.
        A request at exactly a boundary is assigned to the earlier stage.
        Requests past all boundaries fall into the last stage.
        """
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

    def _parse_blis_results(
        self,
        results_path: str,
        experiment: Experiment,
    ) -> SimulatorResult:
        """Parse BLIS JSON output into a SimulatorResult with per-stage breakdown."""
        with open(results_path) as fh:
            data = json.load(fh)

        # --- Aggregate summary from top-level keys ---
        stages_config = experiment.profile_config["load"]["stages"]
        total_duration = sum(s["duration"] for s in stages_config)

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

        # --- Per-stage metrics from request-level data ---
        raw_requests = [
            r for r in data.get("requests", [])
            if isinstance(r, dict) and self._REQUIRED_REQUEST_KEYS.issubset(r)
        ]
        stage_buckets = self._split_requests_by_stage(raw_requests, stages_config)

        stages: list[StageMetrics] = []
        for i, bucket in enumerate(stage_buckets):
            stages.append(self._compute_stage_from_bucket(bucket, i, stages_config[i]))

        return SimulatorResult(
            adapter_name=self.name,
            experiment_folder=experiment.folder,
            stages=stages,
            summary=summary,
        )

    _REQUIRED_REQUEST_KEYS = {"e2e_ms", "ttft_ms", "itl_ms", "num_prefill_tokens", "num_decode_tokens", "arrived_at"}

    def _compute_stage_from_bucket(
        self,
        bucket: list[dict],
        stage_index: int,
        stage_cfg: dict,
    ) -> StageMetrics:
        """Compute percentile metrics for a single stage bucket."""
        zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)

        # Filter out malformed request records that lack required keys.
        valid = [r for r in bucket if self._REQUIRED_REQUEST_KEYS.issubset(r)]

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
