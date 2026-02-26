"""Tests for validation module — temporal split effectiveness and ProgressIndex proxy.

Uses synthetic data only; no real experiment data required.
"""

import numpy as np
import pandas as pd
import pytest

from validation import temporal_vs_random_split_comparison, validate_progress_index


# ---------------------------------------------------------------------------
# TestTemporalVsRandomSplit
# ---------------------------------------------------------------------------


class TestTemporalVsRandomSplit:
    """Test temporal_vs_random_split_comparison function."""

    @staticmethod
    def _make_autocorrelated_df(n_experiments=3, steps_per_exp=200, seed=42):
        """Create synthetic data with strong temporal autocorrelation.

        Adjacent steps within an experiment share a slowly drifting trend
        (random walk) plus small noise.  This mimics real vLLM step data
        where batch composition changes slowly over time.

        Returns a DataFrame with columns:
            experiment_id, step.id, batch.prefill_tokens, batch.decode_tokens,
            step.duration_us
        """
        rng = np.random.default_rng(seed)
        rows = []
        for exp_idx in range(n_experiments):
            exp_id = f"exp_{exp_idx}"
            # Slow random walk for the trend
            prefill_trend = np.cumsum(rng.normal(0, 5, size=steps_per_exp)) + 500
            decode_trend = np.cumsum(rng.normal(0, 3, size=steps_per_exp)) + 300

            for step_idx in range(steps_per_exp):
                prefill = max(1, prefill_trend[step_idx] + rng.normal(0, 2))
                decode = max(1, decode_trend[step_idx] + rng.normal(0, 1))
                # Duration strongly depends on features + trend (autocorrelated)
                duration = 100 + 0.5 * prefill + 0.3 * decode + rng.normal(0, 5)
                rows.append(
                    {
                        "experiment_id": exp_id,
                        "step.id": step_idx,
                        "batch.prefill_tokens": prefill,
                        "batch.decode_tokens": decode,
                        "step.duration_us": max(1, duration),
                    }
                )
        return pd.DataFrame(rows)

    def test_returns_dict_with_expected_keys(self):
        """Result should have temporal_mape, random_mape, gap, conclusion."""
        df = self._make_autocorrelated_df(n_experiments=2, steps_per_exp=50)
        result = temporal_vs_random_split_comparison(df, seed=42)

        assert isinstance(result, dict)
        expected_keys = {"temporal_mape", "random_mape", "gap", "conclusion"}
        assert set(result.keys()) == expected_keys

    def test_random_mape_lower_on_autocorrelated_data(self):
        """On data with strong temporal autocorrelation, random split leaks.

        Random splitting mixes adjacent (correlated) steps into train and
        test, causing the model to exploit autocorrelation.  This should
        give a lower MAPE than the temporal split, which correctly prevents
        this leakage.
        """
        df = self._make_autocorrelated_df(n_experiments=3, steps_per_exp=200)
        result = temporal_vs_random_split_comparison(df, seed=42)

        # Random split should achieve lower MAPE (it's "cheating" via leakage)
        assert result["random_mape"] < result["temporal_mape"], (
            f"Expected random MAPE ({result['random_mape']:.2f}) < "
            f"temporal MAPE ({result['temporal_mape']:.2f}) due to leakage"
        )

    def test_both_mapes_are_positive(self):
        """Both MAPE values should be positive."""
        df = self._make_autocorrelated_df(n_experiments=2, steps_per_exp=100)
        result = temporal_vs_random_split_comparison(df, seed=42)

        assert result["temporal_mape"] > 0, "Temporal MAPE should be positive"
        assert result["random_mape"] > 0, "Random MAPE should be positive"

    def test_gap_is_temporal_minus_random(self):
        """Gap should equal temporal_mape - random_mape."""
        df = self._make_autocorrelated_df(n_experiments=2, steps_per_exp=100)
        result = temporal_vs_random_split_comparison(df, seed=42)

        expected_gap = result["temporal_mape"] - result["random_mape"]
        assert result["gap"] == pytest.approx(expected_gap, abs=1e-10)

    def test_conclusion_contains_leakage_when_gap_large(self):
        """When gap > 5%, conclusion should mention leakage prevention."""
        df = self._make_autocorrelated_df(n_experiments=3, steps_per_exp=200)
        result = temporal_vs_random_split_comparison(df, seed=42)

        # With strongly autocorrelated data, gap should be > 5%
        if result["gap"] > 5.0:
            assert "leakage" in result["conclusion"].lower()

    def test_works_with_single_experiment(self):
        """Should work even with a single experiment."""
        df = self._make_autocorrelated_df(n_experiments=1, steps_per_exp=100)
        result = temporal_vs_random_split_comparison(df, seed=42)

        assert isinstance(result["temporal_mape"], float)
        assert isinstance(result["random_mape"], float)


# ---------------------------------------------------------------------------
# TestProgressIndexValidation
# ---------------------------------------------------------------------------


class TestProgressIndexValidation:
    """Test validate_progress_index function."""

    @staticmethod
    def _make_lifecycle_df(n_requests=100, seed=42):
        """Create synthetic lifecycle data where E2E time is proportional to total tokens.

        For each request:
            - input_tokens ~ Uniform(50, 500)
            - output_tokens ~ Uniform(10, 200)
            - output_token_times: list of length output_tokens
            - start_time, end_time where E2E = (input_tokens + output_tokens) * rate + noise
        """
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_requests):
            input_tokens = int(rng.integers(50, 500))
            output_tokens = int(rng.integers(10, 200))
            total = input_tokens + output_tokens
            # E2E time is roughly proportional to total tokens
            e2e = total * 0.1 + rng.normal(0, 1.0)
            e2e = max(0.1, e2e)
            start = float(i * 0.5)
            end = start + e2e
            # output_token_times: list of output_tokens timestamps
            token_times = sorted(
                [start + rng.uniform(0, e2e) for _ in range(output_tokens)]
            )
            rows.append(
                {
                    "request_id": i,
                    "start_time": start,
                    "end_time": end,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "output_token_times": token_times,
                }
            )
        df = pd.DataFrame(rows).set_index("request_id")
        return df

    @staticmethod
    def _make_uncorrelated_lifecycle_df(n_requests=100, seed=42):
        """Create lifecycle data where E2E time is NOT correlated with total tokens."""
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_requests):
            input_tokens = int(rng.integers(50, 500))
            output_tokens = int(rng.integers(10, 200))
            # E2E is random, NOT proportional to tokens
            e2e = rng.uniform(1.0, 100.0)
            start = float(i * 0.5)
            end = start + e2e
            token_times = sorted(
                [start + rng.uniform(0, e2e) for _ in range(output_tokens)]
            )
            rows.append(
                {
                    "request_id": i,
                    "start_time": start,
                    "end_time": end,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "output_token_times": token_times,
                }
            )
        df = pd.DataFrame(rows).set_index("request_id")
        return df

    def test_returns_dict_with_expected_keys(self):
        """Result should have correlation, n_requests, passed, conclusion."""
        df = self._make_lifecycle_df(n_requests=50)
        result = validate_progress_index(df)

        assert isinstance(result, dict)
        expected_keys = {"correlation", "n_requests", "passed", "conclusion"}
        assert set(result.keys()) == expected_keys

    def test_high_correlation_on_proportional_data(self):
        """When E2E time is proportional to total_tokens, correlation should be high."""
        df = self._make_lifecycle_df(n_requests=200, seed=42)
        result = validate_progress_index(df)

        assert result["correlation"] > 0.9, (
            f"Expected high correlation on proportional data, got {result['correlation']:.3f}"
        )
        assert result["passed"] is True

    def test_flags_low_correlation(self):
        """When correlation < 0.9, should flag as informational warning."""
        df = self._make_uncorrelated_lifecycle_df(n_requests=200, seed=42)
        result = validate_progress_index(df)

        # Uncorrelated data should have low correlation
        assert result["correlation"] < 0.9
        assert result["passed"] is False
        assert "r1" in result["conclusion"].lower() or "flag" in result["conclusion"].lower()

    def test_n_requests_matches_input(self):
        """n_requests should match the number of rows in the input DataFrame."""
        df = self._make_lifecycle_df(n_requests=75)
        result = validate_progress_index(df)
        assert result["n_requests"] == 75

    def test_correlation_is_float(self):
        """Correlation should be a float."""
        df = self._make_lifecycle_df(n_requests=50)
        result = validate_progress_index(df)
        assert isinstance(result["correlation"], float)

    def test_output_token_times_length_check(self):
        """When output_token_times length matches output_tokens, data is consistent.

        The function should still work correctly and not error even when
        output_token_times has the right length.
        """
        df = self._make_lifecycle_df(n_requests=30)
        # Verify our synthetic data has correct output_token_times lengths
        for _, row in df.iterrows():
            assert len(row["output_token_times"]) == row["output_tokens"]
        # Function should work without errors
        result = validate_progress_index(df)
        assert result["n_requests"] == 30
