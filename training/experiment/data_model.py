from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatencyDistribution:
    mean: float | None = None  # milliseconds; None if simulator doesn't provide this metric
    p90: float | None = None
    p99: float | None = None


@dataclass
class ThroughputMetrics:
    input_tokens_per_sec: float
    output_tokens_per_sec: float
    requests_per_sec: float


@dataclass
class StageMetrics:
    stage_index: int
    rate: float  # req/s
    duration: float  # seconds
    num_requests: int
    e2e: LatencyDistribution
    ttft: LatencyDistribution
    itl: LatencyDistribution
    throughput: ThroughputMetrics


@dataclass
class Experiment:
    folder: str
    model: str  # HuggingFace ID, e.g. "meta-llama/Llama-2-7b-hf"
    tp: int
    workload: str  # "general", "codegen", "roleplay", "reasoning"
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    total_kv_blocks: int  # GPU blocks: "GPU KV cache size" tokens / 16
    cpu_kv_blocks: int  # CPU blocks: capacity from cpu_bytes_to_use in vllm.log
    stages: list[StageMetrics]
    summary: StageMetrics
    profile_config: dict  # Raw parsed profile.yaml

    # Manifest metadata (from experiments.json).  Defaults match the original
    # 7-experiment H100/FP16 dataset so that legacy callers without a manifest
    # still produce valid Experiment objects.
    exp_id: int = 0             # Experiment number (1-59)
    hardware: str = "H100"      # "H100", "A100-80GB", "L40S"
    dp: int | None = None       # Data parallelism degree; None = not set
    cpu_offload: bool = False
    gpu_mem_util: float = 0.9
    precision: str = "FP16"     # "FP16", "FP8"
    safe: str = "safe"          # "safe", "unsafe", "uncalibrated"


@dataclass
class SimulatorResult:
    adapter_name: str
    experiment_folder: str
    stages: list[StageMetrics]
    summary: StageMetrics
    wall_clock_seconds: float = 0.0
