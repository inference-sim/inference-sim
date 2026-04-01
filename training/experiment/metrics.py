"""Error metric computations for sim-to-real accuracy comparison.

Functions
---------
compute_mape(predicted, actual)
    Mean Absolute Percentage Error.
compute_mpe(predicted, actual)
    Mean (signed) Percentage Error.
compute_absolute_error(predicted, actual)
    Absolute Error.
compute_errors(experiment, result)
    Compare all metrics between ground truth and simulator, returning
    a list of ``ErrorRecord`` dataclasses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from experiment.data_model import Experiment, SimulatorResult, StageMetrics

logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """One row in the error table: a single metric comparison."""

    simulator: str
    experiment_folder: str
    model: str
    workload: str
    stage_index: int
    metric_name: str
    predicted: float
    actual: float
    mape: float
    mpe: float
    absolute_error: float

    # Metadata from experiments.json manifest
    exp_id: int = 0
    hardware: str = "H100"
    dp: int | None = None
    cpu_offload: bool = False
    gpu_mem_util: float = 0.9
    precision: str = "FP16"
    tp: int = 1
    max_num_batched_tokens: int = 2048


@dataclass
class RuntimeRecord:
    """One row in the runtime table: wall-clock time for a single (adapter, experiment) run."""

    simulator: str
    experiment_folder: str
    model: str
    workload: str
    wall_clock_seconds: float

    # Metadata from experiments.json manifest
    exp_id: int = 0
    hardware: str = "H100"
    dp: int | None = None
    cpu_offload: bool = False
    gpu_mem_util: float = 0.9
    precision: str = "FP16"
    tp: int = 1
    max_num_batched_tokens: int = 2048


def compute_mape(predicted: float, actual: float) -> float:
    """Mean Absolute Percentage Error (single pair).

    Returns 0.0 when both values are zero (perfect match).
    Returns ``float('inf')`` when ``actual == 0`` but ``predicted != 0``
    to avoid masking real errors.
    Returns ``float('inf')`` when ``predicted == 0`` but ``actual != 0``
    (zero prediction is invalid and should be heavily penalized).
    """
    if actual == 0:
        return 0.0 if predicted == 0 else float("inf")
    if predicted == 0:
        return float("inf")  # Zero prediction is nonsensical → infinite penalty
    return abs(predicted - actual) / abs(actual) * 100


def compute_mpe(predicted: float, actual: float) -> float:
    """Mean (signed) Percentage Error (single pair).

    Returns 0.0 when both values are zero (perfect match).
    Returns ``float('inf')`` (or ``float('-inf')``) when ``actual == 0``
    but ``predicted != 0`` to avoid masking real errors.
    Returns ``float('-inf')`` when ``predicted == 0`` but ``actual != 0``
    (zero prediction is invalid and should be heavily penalized).
    """
    if actual == 0:
        if predicted == 0:
            return 0.0
        return float("inf") if predicted > 0 else float("-inf")
    if predicted == 0:
        return float("-inf")  # Zero prediction is nonsensical → infinite penalty (negative since underpredicting)
    return (predicted - actual) / abs(actual) * 100


def compute_absolute_error(predicted: float, actual: float) -> float:
    """Absolute Error."""
    return abs(predicted - actual)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

_LATENCY_METRICS = [
    ("e2e_mean", lambda s: s.e2e.mean),
    ("e2e_p90", lambda s: s.e2e.p90),
    ("e2e_p99", lambda s: s.e2e.p99),
    ("ttft_mean", lambda s: s.ttft.mean),
    ("ttft_p90", lambda s: s.ttft.p90),
    ("ttft_p99", lambda s: s.ttft.p99),
    ("itl_mean", lambda s: s.itl.mean),
    ("itl_p90", lambda s: s.itl.p90),
    ("itl_p99", lambda s: s.itl.p99),
]


def _compare_stages(
    pred: StageMetrics,
    actual: StageMetrics,
    simulator: str,
    experiment: Experiment,
) -> list[ErrorRecord]:
    """Compare all latency metrics between two StageMetrics."""
    records = []
    for metric_name, getter in _LATENCY_METRICS:
        p = getter(pred)
        a = getter(actual)
        if p is None or a is None:
            continue
        records.append(ErrorRecord(
            simulator=simulator,
            experiment_folder=experiment.folder,
            model=experiment.model,
            workload=experiment.workload,
            stage_index=pred.stage_index,
            metric_name=metric_name,
            predicted=p,
            actual=a,
            mape=compute_mape(p, a),
            mpe=compute_mpe(p, a),
            absolute_error=compute_absolute_error(p, a),
            exp_id=experiment.exp_id,
            hardware=experiment.hardware,
            dp=experiment.dp,
            cpu_offload=experiment.cpu_offload,
            gpu_mem_util=experiment.gpu_mem_util,
            precision=experiment.precision,
            tp=experiment.tp,
            max_num_batched_tokens=experiment.max_num_batched_tokens,
        ))
    return records


def compute_errors(
    experiment: Experiment,
    result: SimulatorResult,
) -> list[ErrorRecord]:
    """Compare all stages + summary between ground truth and simulator output.

    Returns one ``ErrorRecord`` per (stage, metric) combination.
    """
    records: list[ErrorRecord] = []

    if len(result.stages) != len(experiment.stages):
        logger.warning(
            "Stage count mismatch for %s (%s): simulator=%d, ground-truth=%d",
            result.adapter_name, experiment.folder,
            len(result.stages), len(experiment.stages),
        )

    # Per-stage
    for pred_stage in result.stages:
        idx = pred_stage.stage_index
        if idx < 0 or idx >= len(experiment.stages):
            continue
        actual_stage = experiment.stages[idx]
        records.extend(_compare_stages(pred_stage, actual_stage, result.adapter_name, experiment))

    # Summary
    records.extend(_compare_stages(result.summary, experiment.summary, result.adapter_name, experiment))

    return records
