"""Tests for E2E decomposition error attribution -- synthetic data only.

Verifies each component of the decomposition with known inputs so that
results can be checked analytically.
"""

import numpy as np
import pandas as pd
import pytest

from e2e_decomposition import (
    component_error_attribution,
    compute_blackbox_step_predictions,
    compute_step_time_error,
    estimate_output_processing_contribution,
    estimate_queueing_contribution,
    estimate_step_time_contribution,
)


# ---------------------------------------------------------------------------
# Helpers -- synthetic data construction
# ---------------------------------------------------------------------------


def _make_steps_df(
    n: int = 50,
    prefill_base: float = 200.0,
    decode_base: float = 50.0,
    beta_true: list[float] | None = None,
    noise_std: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a step-level DataFrame with known linear relationship.

    step.duration_us = beta0 + beta1*prefill + beta2*decode + noise
    """
    if beta_true is None:
        beta_true = [100.0, 0.5, 0.3]

    rng = np.random.default_rng(seed)
    prefill = rng.uniform(prefill_base * 0.5, prefill_base * 1.5, size=n)
    decode = rng.uniform(decode_base * 0.5, decode_base * 1.5, size=n)
    noise = rng.normal(0, noise_std, size=n)
    duration = beta_true[0] + beta_true[1] * prefill + beta_true[2] * decode + noise

    return pd.DataFrame({
        "batch.prefill_tokens": prefill,
        "batch.decode_tokens": decode,
        "step.duration_us": duration,
    })


def _make_lifecycle_df(
    n_requests: int = 20,
    input_range: tuple[int, int] = (100, 500),
    output_range: tuple[int, int] = (10, 100),
    e2e_scale: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a lifecycle DataFrame with known structure.

    E2E time = e2e_scale * (input_tokens + output_tokens).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_requests):
        input_tokens = int(rng.integers(input_range[0], input_range[1]))
        output_tokens = int(rng.integers(output_range[0], output_range[1]))
        e2e = e2e_scale * (input_tokens + output_tokens)
        start = float(i * 100.0)
        end = start + e2e
        rows.append({
            "request_id": i,
            "start_time": start,
            "end_time": end,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })
    df = pd.DataFrame(rows).set_index("request_id")
    return df


# ---------------------------------------------------------------------------
# compute_blackbox_step_predictions tests
# ---------------------------------------------------------------------------


class TestComputeBlackboxStepPredictions:
    """Tests for compute_blackbox_step_predictions."""

    def test_known_coefficients(self):
        """With known beta=[100, 0.5, 0.3], predictions should match formula."""
        df = pd.DataFrame({
            "batch.prefill_tokens": [200.0, 400.0, 0.0],
            "batch.decode_tokens": [100.0, 0.0, 300.0],
        })
        beta = [100.0, 0.5, 0.3]
        result = compute_blackbox_step_predictions(df, beta)

        # Row 0: 100 + 0.5*200 + 0.3*100 = 100 + 100 + 30 = 230
        assert result.iloc[0] == pytest.approx(230.0)
        # Row 1: 100 + 0.5*400 + 0.3*0 = 100 + 200 + 0 = 300
        assert result.iloc[1] == pytest.approx(300.0)
        # Row 2: 100 + 0.5*0 + 0.3*300 = 100 + 0 + 90 = 190
        assert result.iloc[2] == pytest.approx(190.0)

    def test_zero_coefficients(self):
        """With all-zero coefficients, predictions should be all zero."""
        df = pd.DataFrame({
            "batch.prefill_tokens": [100.0, 200.0],
            "batch.decode_tokens": [50.0, 60.0],
        })
        result = compute_blackbox_step_predictions(df, [0.0, 0.0, 0.0])
        assert np.all(result.values == pytest.approx(0.0))

    def test_intercept_only(self):
        """With beta=[500, 0, 0], all predictions should be 500."""
        df = pd.DataFrame({
            "batch.prefill_tokens": [100.0, 200.0, 300.0],
            "batch.decode_tokens": [10.0, 20.0, 30.0],
        })
        result = compute_blackbox_step_predictions(df, [500.0, 0.0, 0.0])
        assert np.all(result.values == pytest.approx(500.0))

    def test_returns_series_of_correct_length(self):
        """Result should be a pandas Series with the same length as input."""
        df = _make_steps_df(n=37)
        result = compute_blackbox_step_predictions(df, [1.0, 2.0, 3.0])
        assert isinstance(result, pd.Series)
        assert len(result) == 37

    def test_wrong_number_of_coefficients_raises(self):
        """Passing wrong number of coefficients should raise ValueError."""
        df = pd.DataFrame({
            "batch.prefill_tokens": [100.0],
            "batch.decode_tokens": [50.0],
        })
        with pytest.raises(ValueError, match="Expected 3 beta coefficients"):
            compute_blackbox_step_predictions(df, [1.0, 2.0])


# ---------------------------------------------------------------------------
# compute_step_time_error tests
# ---------------------------------------------------------------------------


class TestComputeStepTimeError:
    """Tests for compute_step_time_error."""

    def test_perfect_prediction_zero_mape(self):
        """When beta coefficients exactly match generating process, MAPE ~ 0."""
        beta_true = [100.0, 0.5, 0.3]
        df = _make_steps_df(n=100, beta_true=beta_true, noise_std=0.0)

        result = compute_step_time_error(df, beta_true)

        assert result["mape"] == pytest.approx(0.0, abs=1e-10)
        assert result["mspe"] == pytest.approx(0.0, abs=1e-10)
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_expected_keys(self):
        """Result should contain all expected metric keys."""
        df = _make_steps_df(n=20)
        result = compute_step_time_error(df, [100.0, 0.5, 0.3])

        expected_keys = {
            "mape", "mspe", "pearson_r", "p99_error",
            "mean_predicted", "mean_actual", "total_predicted", "total_actual",
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_are_floats(self):
        """All returned metric values should be Python floats."""
        df = _make_steps_df(n=20)
        result = compute_step_time_error(df, [100.0, 0.5, 0.3])

        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_overestimation_positive_mspe(self):
        """When predictions are systematically high, MSPE should be positive."""
        beta_true = [100.0, 0.5, 0.3]
        df = _make_steps_df(n=50, beta_true=beta_true, noise_std=0.0)

        # Overestimate by using a larger intercept
        beta_high = [200.0, 0.5, 0.3]
        result = compute_step_time_error(df, beta_high)

        assert result["mspe"] > 0, "Overestimation should produce positive MSPE"
        assert result["mape"] > 0, "Overestimation should produce positive MAPE"

    def test_underestimation_negative_mspe(self):
        """When predictions are systematically low, MSPE should be negative."""
        beta_true = [200.0, 0.5, 0.3]
        df = _make_steps_df(n=50, beta_true=beta_true, noise_std=0.0)

        # Underestimate by using a smaller intercept
        beta_low = [100.0, 0.5, 0.3]
        result = compute_step_time_error(df, beta_low)

        assert result["mspe"] < 0, "Underestimation should produce negative MSPE"

    def test_totals_consistent_with_means(self):
        """total = mean * n for both predicted and actual."""
        n = 40
        df = _make_steps_df(n=n)
        result = compute_step_time_error(df, [100.0, 0.5, 0.3])

        assert result["total_predicted"] == pytest.approx(
            result["mean_predicted"] * n, rel=1e-10
        )
        assert result["total_actual"] == pytest.approx(
            result["mean_actual"] * n, rel=1e-10
        )


# ---------------------------------------------------------------------------
# estimate_queueing_contribution tests
# ---------------------------------------------------------------------------


class TestEstimateQueueingContribution:
    """Tests for estimate_queueing_contribution."""

    def test_known_alpha_coefficients(self):
        """With known alpha and known data, totals should match hand calculation."""
        lifecycle_df = pd.DataFrame({
            "request_id": [0, 1],
            "start_time": [0.0, 0.0],
            "end_time": [1000.0, 2000.0],
            "input_tokens": [100, 200],
            "output_tokens": [50, 60],
        }).set_index("request_id")

        alpha = [10.0, 0.5, 0.0]  # queueing = 10 + 0.5 * input_tokens

        result = estimate_queueing_contribution(lifecycle_df, alpha)

        # Request 0: 10 + 0.5*100 = 60
        # Request 1: 10 + 0.5*200 = 110
        # Mean: (60 + 110) / 2 = 85
        assert result["mean_queueing_us"] == pytest.approx(85.0)
        # Total: 60 + 110 = 170
        assert result["total_queueing_us"] == pytest.approx(170.0)
        # Total E2E: 1000 + 2000 = 3000
        # Fraction: 170 / 3000
        assert result["fraction_of_e2e"] == pytest.approx(170.0 / 3000.0)

    def test_zero_alpha_zero_contribution(self):
        """With alpha=[0,0,0], queueing contribution should be zero."""
        lifecycle_df = _make_lifecycle_df(n_requests=10)
        result = estimate_queueing_contribution(lifecycle_df, [0.0, 0.0, 0.0])

        assert result["mean_queueing_us"] == pytest.approx(0.0)
        assert result["total_queueing_us"] == pytest.approx(0.0)
        assert result["fraction_of_e2e"] == pytest.approx(0.0)

    def test_returns_expected_keys(self):
        """Result should contain all expected keys."""
        lifecycle_df = _make_lifecycle_df(n_requests=5)
        result = estimate_queueing_contribution(lifecycle_df, [1.0, 0.1, 0.0])

        expected_keys = {"mean_queueing_us", "total_queueing_us", "fraction_of_e2e"}
        assert set(result.keys()) == expected_keys

    def test_fraction_between_zero_and_one_for_small_alpha(self):
        """With small alpha coefficients, fraction should be in [0, 1)."""
        lifecycle_df = _make_lifecycle_df(n_requests=20, e2e_scale=1000.0)
        result = estimate_queueing_contribution(lifecycle_df, [1.0, 0.001, 0.0])

        assert 0.0 <= result["fraction_of_e2e"] < 1.0


# ---------------------------------------------------------------------------
# estimate_output_processing_contribution tests
# ---------------------------------------------------------------------------


class TestEstimateOutputProcessingContribution:
    """Tests for estimate_output_processing_contribution."""

    def test_known_alpha2(self):
        """With known alpha2 and known data, results should match hand calculation."""
        lifecycle_df = pd.DataFrame({
            "request_id": [0, 1],
            "start_time": [0.0, 0.0],
            "end_time": [5000.0, 10000.0],
            "input_tokens": [100, 200],
            "output_tokens": [50, 100],
        }).set_index("request_id")

        alpha2 = 2.0  # 2 us per output token

        result = estimate_output_processing_contribution(lifecycle_df, alpha2)

        # Request 0: 2.0 * 50 = 100
        # Request 1: 2.0 * 100 = 200
        # Mean: (100 + 200) / 2 = 150
        assert result["mean_output_us"] == pytest.approx(150.0)
        # Total: 100 + 200 = 300
        assert result["total_output_us"] == pytest.approx(300.0)
        # Total E2E: 5000 + 10000 = 15000
        # Fraction: 300 / 15000 = 0.02
        assert result["fraction_of_e2e"] == pytest.approx(0.02)

    def test_zero_alpha2_zero_contribution(self):
        """With alpha2=0, output processing contribution should be zero."""
        lifecycle_df = _make_lifecycle_df(n_requests=10)
        result = estimate_output_processing_contribution(lifecycle_df, 0.0)

        assert result["mean_output_us"] == pytest.approx(0.0)
        assert result["total_output_us"] == pytest.approx(0.0)
        assert result["fraction_of_e2e"] == pytest.approx(0.0)

    def test_returns_expected_keys(self):
        """Result should contain all expected keys."""
        lifecycle_df = _make_lifecycle_df(n_requests=5)
        result = estimate_output_processing_contribution(lifecycle_df, 1.0)

        expected_keys = {"mean_output_us", "total_output_us", "fraction_of_e2e"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# estimate_step_time_contribution tests
# ---------------------------------------------------------------------------


class TestEstimateStepTimeContribution:
    """Tests for estimate_step_time_contribution."""

    def test_known_values(self):
        """With known step durations and E2E, fraction should be calculable."""
        steps_df = pd.DataFrame({
            "batch.prefill_tokens": [100, 200],
            "batch.decode_tokens": [50, 60],
            "step.duration_us": [1000.0, 2000.0],
        })
        lifecycle_df = pd.DataFrame({
            "request_id": [0],
            "start_time": [0.0],
            "end_time": [5000.0],
            "input_tokens": [300],
            "output_tokens": [100],
        }).set_index("request_id")

        result = estimate_step_time_contribution(steps_df, lifecycle_df)

        assert result["total_step_us"] == pytest.approx(3000.0)
        assert result["total_e2e_us"] == pytest.approx(5000.0)
        assert result["fraction_of_e2e"] == pytest.approx(0.6)
        # 2 steps, 1 request -> 2.0 steps per request
        assert result["mean_steps_per_request"] == pytest.approx(2.0)

    def test_returns_expected_keys(self):
        """Result should contain all expected keys."""
        steps_df = _make_steps_df(n=10)
        lifecycle_df = _make_lifecycle_df(n_requests=5)
        result = estimate_step_time_contribution(steps_df, lifecycle_df)

        expected_keys = {
            "total_step_us", "total_e2e_us", "fraction_of_e2e",
            "mean_steps_per_request",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# component_error_attribution tests
# ---------------------------------------------------------------------------


class TestComponentErrorAttribution:
    """Tests for the full component_error_attribution report."""

    def test_returns_all_top_level_keys(self):
        """Report should have all 7 top-level keys."""
        steps_df = _make_steps_df(n=30)
        lifecycle_df = _make_lifecycle_df(n_requests=10)
        alpha = [0.0, 0.0, 0.0]
        beta = [100.0, 0.5, 0.3]

        result = component_error_attribution(steps_df, lifecycle_df, alpha, beta)

        expected_keys = {
            "step_time", "queueing_time", "output_processing",
            "scheduling", "preemption", "summary", "recommendation",
        }
        assert set(result.keys()) == expected_keys

    def test_step_time_has_error_metrics(self):
        """step_time entry should contain error metrics and fraction."""
        steps_df = _make_steps_df(n=30)
        lifecycle_df = _make_lifecycle_df(n_requests=10)

        result = component_error_attribution(
            steps_df, lifecycle_df, [0.0, 0.0, 0.0], [100.0, 0.5, 0.3]
        )

        step = result["step_time"]
        assert "mape" in step
        assert "mspe" in step
        assert "pearson_r" in step
        assert "p99_error" in step
        assert "fraction_of_e2e" in step
        assert "mean_steps_per_request" in step

    def test_scheduling_and_preemption_are_zero(self):
        """Scheduling and preemption should report current_value = 0."""
        steps_df = _make_steps_df(n=10)
        lifecycle_df = _make_lifecycle_df(n_requests=5)

        result = component_error_attribution(
            steps_df, lifecycle_df, [0.0, 0.0, 0.0], [100.0, 0.5, 0.3]
        )

        assert result["scheduling"]["current_value"] == 0.0
        assert result["preemption"]["current_value"] == 0.0
        assert "note" in result["scheduling"]
        assert "note" in result["preemption"]

    def test_summary_is_string(self):
        """summary and recommendation should be human-readable strings."""
        steps_df = _make_steps_df(n=20)
        lifecycle_df = _make_lifecycle_df(n_requests=10)

        result = component_error_attribution(
            steps_df, lifecycle_df, [0.0, 0.0, 0.0], [100.0, 0.5, 0.3]
        )

        assert isinstance(result["summary"], str)
        assert isinstance(result["recommendation"], str)
        assert len(result["summary"]) > 10
        assert len(result["recommendation"]) > 10

    def test_perfect_beta_low_step_error(self):
        """When beta exactly matches generating process, step MAPE should be ~0."""
        beta_true = [100.0, 0.5, 0.3]
        steps_df = _make_steps_df(n=50, beta_true=beta_true, noise_std=0.0)
        lifecycle_df = _make_lifecycle_df(n_requests=10)

        result = component_error_attribution(
            steps_df, lifecycle_df, [0.0, 0.0, 0.0], beta_true
        )

        assert result["step_time"]["mape"] == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_alpha_produces_nonzero_queueing(self):
        """With nonzero alpha, queueing contribution should be nonzero."""
        steps_df = _make_steps_df(n=20)
        lifecycle_df = _make_lifecycle_df(n_requests=10)
        alpha = [10.0, 0.1, 0.5]

        result = component_error_attribution(
            steps_df, lifecycle_df, alpha, [100.0, 0.5, 0.3]
        )

        assert result["queueing_time"]["total_queueing_us"] > 0
        assert result["output_processing"]["total_output_us"] > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_single_step_experiment(self):
        """Should work with a single step."""
        steps_df = pd.DataFrame({
            "batch.prefill_tokens": [200.0],
            "batch.decode_tokens": [50.0],
            "step.duration_us": [500.0],
        })
        lifecycle_df = pd.DataFrame({
            "request_id": [0],
            "start_time": [0.0],
            "end_time": [600.0],
            "input_tokens": [200],
            "output_tokens": [50],
        }).set_index("request_id")

        result = component_error_attribution(
            steps_df, lifecycle_df, [0.0, 0.0, 0.0], [100.0, 0.5, 0.3]
        )

        assert "step_time" in result
        assert result["step_time"]["mean_steps_per_request"] == pytest.approx(1.0)

    def test_empty_steps_df(self):
        """With empty steps DataFrame, step predictions should still work
        (returning empty Series) and error metrics should be zero."""
        steps_df = pd.DataFrame({
            "batch.prefill_tokens": pd.Series([], dtype=float),
            "batch.decode_tokens": pd.Series([], dtype=float),
            "step.duration_us": pd.Series([], dtype=float),
        })

        result = compute_blackbox_step_predictions(steps_df, [100.0, 0.5, 0.3])
        assert len(result) == 0

    def test_single_request_lifecycle(self):
        """Should work with a single request in lifecycle data."""
        lifecycle_df = pd.DataFrame({
            "request_id": [0],
            "start_time": [0.0],
            "end_time": [1000.0],
            "input_tokens": [300],
            "output_tokens": [80],
        }).set_index("request_id")

        result = estimate_queueing_contribution(lifecycle_df, [5.0, 0.1, 0.0])

        # queueing = 5 + 0.1 * 300 = 35
        assert result["mean_queueing_us"] == pytest.approx(35.0)
        assert result["total_queueing_us"] == pytest.approx(35.0)
        assert result["fraction_of_e2e"] == pytest.approx(35.0 / 1000.0)

    def test_zero_e2e_time_no_division_error(self):
        """When E2E time is zero (start == end), fraction should be 0, not error."""
        lifecycle_df = pd.DataFrame({
            "request_id": [0],
            "start_time": [100.0],
            "end_time": [100.0],
            "input_tokens": [200],
            "output_tokens": [50],
        }).set_index("request_id")

        result = estimate_queueing_contribution(lifecycle_df, [10.0, 0.5, 0.0])
        assert result["fraction_of_e2e"] == 0.0

        result2 = estimate_output_processing_contribution(lifecycle_df, 1.0)
        assert result2["fraction_of_e2e"] == 0.0

    def test_large_dataset(self):
        """Should handle a moderately large dataset without error."""
        steps_df = _make_steps_df(n=5000, seed=123)
        lifecycle_df = _make_lifecycle_df(n_requests=500, seed=123)

        result = component_error_attribution(
            steps_df, lifecycle_df, [1.0, 0.01, 0.5], [100.0, 0.5, 0.3]
        )

        assert isinstance(result["step_time"]["mape"], float)
        assert np.isfinite(result["step_time"]["mape"])
