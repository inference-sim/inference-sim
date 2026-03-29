"""AIConfigurator analytical estimate adapter.

Calls ``TaskConfig`` + ``TaskRunner`` from the ``aiconfigurator`` SDK once per
experiment, which returns a Pareto DataFrame with performance predictions across
multiple concurrency levels. For each stage, the adapter matches the stage's
arrival rate to the predicted throughput (``seq/s`` column) to find the
equilibrium concurrency level.

This approach avoids data leakage — no ground-truth latency is used as input.
Instead, concurrency is determined by finding the level where AIConfigurator's
predicted ``seq/s`` matches the stage's arrival rate, creating a self-consistent
solution via Little's Law.

Since AIConfigurator produces only **mean** latency estimates (one row per
concurrency level), P90 and P99 are left as ``None`` (the metrics layer
skips comparisons where the simulator does not provide a value).

E2E latency is derived using the standard formula:
``E2E = TTFT + TPOT × (output_length - 1)``
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

# ---------------------------------------------------------------------------
# Model-name mapping: HuggingFace ID → AIConfigurator SupportedModels key
# ---------------------------------------------------------------------------
_MODEL_MAP: dict[str, str] = {
    "meta-llama/Llama-2-7b-hf": "LLAMA2_7B",
    "meta-llama/Llama-2-70b-hf": "LLAMA2_70B",
}

# ---------------------------------------------------------------------------
# Hardware mapping: experiment hardware tag → AIConfigurator system name
# Only H100 has vLLM perf data in AIConfigurator; A100/L40S excluded via can_run.
# ---------------------------------------------------------------------------
_HW_TO_AICONFIG: dict[str, str] = {"H100": "h100_sxm"}

# MoE models that cannot use the vllm backend in AIConfigurator.
_MOE_MODELS: frozenset[str] = frozenset({
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
})


# ---------------------------------------------------------------------------
# Lazy-import wrappers (same pattern as llm_optimizer_est.py)
# ---------------------------------------------------------------------------

def _create_task_config(**kwargs):
    """Lazy import wrapper — allows mocking without requiring aiconfigurator installed."""
    from aiconfigurator.sdk.task import TaskConfig
    return TaskConfig(**kwargs)


def _run_task(task_config):
    """Lazy import wrapper — allows mocking without requiring aiconfigurator installed."""
    from aiconfigurator.sdk.task import TaskRunner
    return TaskRunner().run(task_config)


class AIConfiguratorEstimateAdapter(SimulatorAdapter):
    """Adapter wrapping the AIConfigurator analytical performance estimator."""

    @property
    def name(self) -> str:
        return "aiconfigurator-estimate"

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------

    def can_run(self, experiment: Experiment) -> bool:
        """True when hardware is supported, precision is FP16/FP8, model is not MoE, and profile uses shared_prefix."""
        if experiment.hardware not in _HW_TO_AICONFIG:
            return False
        if experiment.precision not in ("FP16", "FP8"):
            return False
        if experiment.model in _MOE_MODELS:
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
    def _resolve_model_name(hf_model_id: str) -> str:
        """Map a HuggingFace model ID to AIConfigurator's model key.

        Falls back to the raw HF ID for models not in the map (AIConfigurator
        resolves them via ``get_model_config_from_hf_id``).
        """
        return _MODEL_MAP.get(hf_model_id, hf_model_id)

    @staticmethod
    def _match_throughput(stage_rate: float, pareto_df, tolerance: float = 0.15):
        """Find the Pareto DataFrame row where predicted throughput ≈ stage rate.

        Uses relative error matching: selects the row whose predicted ``seq/s``
        is closest to the stage's arrival rate.

        Args:
            stage_rate: Arrival rate in requests/sec from the stage config.
            pareto_df: AIConfigurator Pareto DataFrame with ``seq/s`` column.
            tolerance: Relative tolerance for early exit (default 15%).

        Returns:
            DataFrame row (pandas Series) with the matched configuration.

        Raises:
            ValueError: If stage_rate is zero or negative.
            RuntimeError: If no valid row found (all inf, zero, or negative).
        """
        if stage_rate <= 0:
            raise ValueError(
                f"Invalid stage rate: {stage_rate}. "
                f"Stages must have positive arrival rate."
            )

        best_row = None
        best_error = float("inf")

        for idx, row in pareto_df.iterrows():
            predicted_rate = float(row["seq/s"])

            # Skip invalid results
            if predicted_rate <= 0 or predicted_rate == float("inf"):
                continue

            # Calculate relative error
            error = abs(predicted_rate - stage_rate) / stage_rate

            if error < best_error:
                best_error = error
                best_row = row

            # Early exit if within tolerance
            if error <= tolerance:
                break

        if best_row is None:
            raise RuntimeError(
                f"No valid row found for rate={stage_rate} req/s in Pareto DataFrame"
            )

        return best_row

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
        if experiment.hardware not in _HW_TO_AICONFIG:
            raise ValueError(
                f"Unsupported hardware '{experiment.hardware}' for {self.name} "
                f"(supported: {sorted(_HW_TO_AICONFIG)})"
            )
        model_name = self._resolve_model_name(experiment.model)
        input_length, output_length = self._extract_lengths(experiment)

        # Run AIConfigurator once — it sweeps all concurrency levels.
        system_name = _HW_TO_AICONFIG[experiment.hardware]
        # FP16: explicit float16 profile; FP8: empty list lets AIConfigurator
        # use its native FP8 path.
        if experiment.precision == "FP16":
            profiles = ["float16_default"]
        elif experiment.precision == "FP8":
            profiles = []
        else:
            raise ValueError(
                f"Unsupported precision '{experiment.precision}' for {self.name} "
                f"(supported: FP16, FP8)"
            )

        task_config = _create_task_config(
            serving_mode="agg",
            model_name=model_name,
            system_name=system_name,
            backend_name="vllm",
            total_gpus=experiment.tp,
            isl=input_length,
            osl=output_length,
            # TTFT constraint set to 150s to cover all experiments in dataset
            # (P99=103s). Expands pareto frontier for better concurrency matching.
            # TPOT=200ms is sufficient (actual P99=55ms).
            ttft=150000.0,
            tpot=200.0,
            profiles=profiles,
        )
        result = _run_task(task_config)
        if result is None:
            raise RuntimeError(
                f"AIConfigurator returned None for {model_name} (tp={experiment.tp})"
            )
        pareto_df = result["pareto_df"]
        if pareto_df is None or pareto_df.empty:
            raise RuntimeError(
                f"AIConfigurator returned empty pareto_df for {model_name} (tp={experiment.tp})"
            )

        # Filter to rows matching the experiment's tensor parallelism.
        tp_df = pareto_df[pareto_df["tp"] == experiment.tp].reset_index(drop=True)
        if tp_df.empty:
            raise RuntimeError(
                f"No AIConfigurator results for tp={experiment.tp} "
                f"(available tp values: {sorted(pareto_df['tp'].unique())})"
            )

        # Match each stage to its throughput-appropriate row in Pareto DataFrame
        stages: list[StageMetrics] = []
        for gt_stage in experiment.stages:
            # Find row where predicted_throughput ≈ stage_rate
            row = self._match_throughput(gt_stage.rate, tp_df)

            predicted_rate = float(row["seq/s"])
            concurrency = int(row.get("concurrency", -1))  # -1 if not present
            logger.info(
                f"aiconfigurator stage {gt_stage.stage_index}: "
                f"rate={gt_stage.rate:.1f} req/s → concurrency={concurrency} "
                f"(predicted_rate={predicted_rate:.2f} req/s)"
            )

            ttft_ms = float(row["ttft"])
            tpot_ms = float(row["tpot"])
            # Standard TPOT formula: E2E = TTFT + TPOT × (N - 1)
            e2e_ms = ttft_ms + tpot_ms * (output_length - 1)

            stages.append(StageMetrics(
                stage_index=gt_stage.stage_index,
                rate=gt_stage.rate,
                duration=gt_stage.duration,
                num_requests=gt_stage.num_requests,
                e2e=self._make_latency_dist(e2e_ms),
                ttft=self._make_latency_dist(ttft_ms),
                itl=self._make_latency_dist(tpot_ms),
                throughput=ThroughputMetrics(
                    input_tokens_per_sec=float(row["seq/s"]) * input_length,
                    output_tokens_per_sec=float(row["tokens/s"]),
                    requests_per_sec=float(row["seq/s"]),
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

        E2E is computed from the stage predictions using request-weighted averaging.
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

        # Request-weighted averages for all latency metrics
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
