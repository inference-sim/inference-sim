"""LLM-Optimizer analytical estimate adapter.

Calls ``estimate_llm_performance`` from the ``llm-optimizer`` package using
**throughput-matching** to derive concurrency without oracle data. The adapter
sweeps concurrency levels [1, 2, 4, 8, ...] once per experiment, then matches
each stage's arrival rate to the predicted throughput to find the equilibrium
concurrency level.

This approach avoids data leakage — no ground-truth latency is used as input.
Instead, concurrency is determined by finding the level where the roofline
model's predicted ``requests_per_sec`` matches the stage's arrival rate,
creating a self-consistent solution via Little's Law.

Since llm-optimizer produces only **mean** latency estimates, P90 and P99
are left as ``None`` (the metrics layer skips comparisons where the
simulator does not provide a value).
"""

from __future__ import annotations

import logging

from experiment.adapters.base import SimulatorAdapter
from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    SimulatorResult,
    StageMetrics,
    ThroughputMetrics,
)

logger = logging.getLogger(__name__)

_HW_TO_LLM_OPT: dict[str, str] = {"H100": "H100", "A100-80GB": "A100"}

def get_model_config_from_hf(model_id: str):
    """Lazy import wrapper — allows mocking without requiring llm_optimizer installed."""
    from llm_optimizer.common import get_model_config_from_hf as _fn
    return _fn(model_id)


def estimate_llm_performance(**kwargs):
    """Lazy import wrapper — allows mocking without requiring llm_optimizer installed."""
    from llm_optimizer.performance import estimate_llm_performance as _fn
    return _fn(**kwargs)


class LLMOptimizerEstimateAdapter(SimulatorAdapter):
    """Adapter wrapping the llm-optimizer roofline performance estimator."""

    @property
    def name(self) -> str:
        return "llm-optimizer-estimate"

    def can_run(self, experiment: Experiment) -> bool:
        """True only when hardware is supported and profile config uses ``shared_prefix``."""
        if experiment.hardware not in _HW_TO_LLM_OPT:
            return False
        if experiment.precision == "FP8" and experiment.hardware == "A100-80GB":
            return False
        try:
            data = experiment.profile_config["data"]
            if data["type"] != "shared_prefix":
                return False
            sp = data["shared_prefix"]
            return all(k in sp for k in ("question_len", "system_prompt_len", "output_len"))
        except (KeyError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sweep_concurrency_levels(
        num_gpus: int,
        gpu_name: str,
        model_config,
        precision: str,
        input_length: int,
        output_length: int,
        max_num_seqs: int,
    ) -> list[tuple[int, object]]:
        """Sweep concurrency levels and collect performance predictions.

        Returns a list of (concurrency, PerformanceResult) pairs, stopping
        when VRAM is exhausted (indicated by ``ttft_ms == inf``).
        """
        concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        concurrency_levels = [c for c in concurrency_levels if c <= max_num_seqs]

        sweep_results = []
        for concurrency in concurrency_levels:
            perf = estimate_llm_performance(
                num_gpus=num_gpus,
                gpu_name=gpu_name,
                model_config=model_config,
                precision=precision,
                concurrency=concurrency,
                input_length=input_length,
                output_length=output_length,
            )
            # Stop if VRAM limit exceeded
            if perf.ttft_ms == float("inf"):
                break
            sweep_results.append((concurrency, perf))

        return sweep_results

    @staticmethod
    def _match_throughput(
        stage_rate: float,
        sweep_results: list[tuple[int, object]],
        tolerance: float = 0.15,
    ) -> tuple[int, object]:
        """Find the concurrency level where predicted throughput ≈ stage rate.

        Uses relative error matching: selects the concurrency whose predicted
        ``requests_per_sec`` is closest to the stage's arrival rate.

        Args:
            stage_rate: Arrival rate in requests/sec from the stage config.
            sweep_results: List of (concurrency, PerformanceResult) pairs.
            tolerance: Relative tolerance for early exit (default 15%).

        Returns:
            (matched_concurrency, matched_PerformanceResult)

        Raises:
            ValueError: If stage_rate is zero or negative.
            RuntimeError: If no valid concurrency level found (all inf or zero).
        """
        if stage_rate <= 0:
            raise ValueError(
                f"Invalid stage rate: {stage_rate}. "
                f"Stages must have positive arrival rate."
            )

        best_match = None
        best_error = float("inf")

        for concurrency, result in sweep_results:
            predicted_rate = result.requests_per_sec

            # Skip invalid results
            if predicted_rate == float("inf") or predicted_rate <= 0:
                continue

            # Calculate relative error
            error = abs(predicted_rate - stage_rate) / stage_rate

            if error < best_error:
                best_error = error
                best_match = (concurrency, result)

            # Early exit if within tolerance
            if error <= tolerance:
                break

        if best_match is None:
            raise RuntimeError(
                f"No valid concurrency level found for rate={stage_rate} req/s"
            )

        return best_match

    @staticmethod
    def _extract_lengths(experiment: Experiment) -> tuple[int, int]:
        """Extract input/output token lengths from profile config."""
        data_cfg = experiment.profile_config["data"]["shared_prefix"]
        input_length = data_cfg["question_len"] + data_cfg["system_prompt_len"]
        output_length = data_cfg["output_len"]
        return input_length, output_length

    @staticmethod
    def _make_latency_dist(mean: float) -> LatencyDistribution:
        return LatencyDistribution(mean=mean)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, experiment: Experiment) -> SimulatorResult:
        if experiment.hardware not in _HW_TO_LLM_OPT:
            raise ValueError(
                f"Unsupported hardware '{experiment.hardware}' for {self.name} "
                f"(supported: {sorted(_HW_TO_LLM_OPT)})"
            )
        if experiment.precision == "FP8" and experiment.hardware == "A100-80GB":
            raise ValueError(
                f"Unsupported precision '{experiment.precision}' on "
                f"'{experiment.hardware}' for {self.name} (A100 has no FP8 TFLOPS)"
            )
        model_config = get_model_config_from_hf(experiment.model)
        precision = experiment.precision.lower()  # "fp16" or "fp8"
        input_length, output_length = self._extract_lengths(experiment)

        # Sweep concurrency levels ONCE per experiment
        sweep_results = self._sweep_concurrency_levels(
            num_gpus=experiment.tp,
            gpu_name=_HW_TO_LLM_OPT[experiment.hardware],
            model_config=model_config,
            precision=precision,
            input_length=input_length,
            output_length=output_length,
            max_num_seqs=experiment.max_num_seqs,
        )

        # Match each stage to its throughput-appropriate concurrency
        stages: list[StageMetrics] = []
        for gt_stage in experiment.stages:
            # Find concurrency where predicted_throughput ≈ stage_rate
            concurrency, perf = self._match_throughput(
                stage_rate=gt_stage.rate,
                sweep_results=sweep_results,
            )
            logger.info(
                f"llm-optimizer stage {gt_stage.stage_index}: "
                f"rate={gt_stage.rate:.1f} req/s → concurrency={concurrency} "
                f"(predicted_rate={perf.requests_per_sec:.2f} req/s)"
            )

            e2e_mean_ms = perf.e2e_latency_s * 1000
            stages.append(StageMetrics(
                stage_index=gt_stage.stage_index,
                rate=gt_stage.rate,
                duration=gt_stage.duration,
                num_requests=gt_stage.num_requests,
                e2e=self._make_latency_dist(e2e_mean_ms),
                ttft=self._make_latency_dist(perf.ttft_ms),
                itl=self._make_latency_dist(perf.itl_ms),
                throughput=ThroughputMetrics(
                    input_tokens_per_sec=perf.input_throughput_tps,
                    output_tokens_per_sec=perf.output_throughput_tps,
                    requests_per_sec=perf.requests_per_sec,
                ),
            ))

        summary = self._weighted_summary(stages)
        return SimulatorResult(
            adapter_name=self.name,
            experiment_folder=experiment.folder,
            stages=stages,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_summary(stages: list[StageMetrics]) -> StageMetrics:
        """Compute a weighted-average summary across stages.

        Latency metrics use request-weighted averaging (correct for per-request
        quantities).  Throughput metrics use duration-weighted averaging
        (correct for rates — tokens/sec, requests/sec).
        """
        total_reqs = sum(s.num_requests for s in stages)
        total_duration = sum(s.duration for s in stages)
        if total_reqs == 0:
            zero = LatencyDistribution(0)
            return StageMetrics(
                stage_index=-1, rate=0, duration=0, num_requests=0,
                e2e=zero, ttft=zero, itl=zero,
                throughput=ThroughputMetrics(0, 0, 0),
            )

        def _req_wavg(getter):
            return sum(getter(s) * s.num_requests for s in stages) / total_reqs

        def _dur_wavg(getter):
            if total_duration <= 0:
                return 0.0
            return sum(getter(s) * s.duration for s in stages) / total_duration

        e2e_mean = _req_wavg(lambda s: s.e2e.mean)
        ttft_mean = _req_wavg(lambda s: s.ttft.mean)
        itl_mean = _req_wavg(lambda s: s.itl.mean)

        return StageMetrics(
            stage_index=-1,
            rate=0.0,
            duration=0.0,
            num_requests=total_reqs,
            e2e=LatencyDistribution(e2e_mean),
            ttft=LatencyDistribution(ttft_mean),
            itl=LatencyDistribution(itl_mean),
            throughput=ThroughputMetrics(
                input_tokens_per_sec=_dur_wavg(lambda s: s.throughput.input_tokens_per_sec),
                output_tokens_per_sec=_dur_wavg(lambda s: s.throughput.output_tokens_per_sec),
                requests_per_sec=_dur_wavg(lambda s: s.throughput.requests_per_sec),
            ),
        )
