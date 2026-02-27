"""Tests for the per-request KV cache length extractor.

Uses synthetic data only -- no dependency on real experiment directories.
"""

import numpy as np
import pandas as pd
import pytest

from lifecycle_kv_extractor import _estimate_kv_length, extract_kv_features


# ---------------------------------------------------------------------------
# Helpers: synthetic data builders
# ---------------------------------------------------------------------------

def _make_lifecycle_df(requests: list[dict]) -> pd.DataFrame:
    """Build a lifecycle DataFrame matching data_loader.load_lifecycle_data() shape.

    Each dict in `requests` must have: start_time (epoch seconds),
    end_time (epoch seconds), input_tokens (int), output_tokens (int),
    output_token_times (list of epoch-second floats).
    """
    df = pd.DataFrame(requests)
    df.index.name = "request_id"
    return df


def _make_steps_df(steps: list[dict]) -> pd.DataFrame:
    """Build a steps DataFrame matching data_loader.load_experiment_steps() shape.

    Each dict in `steps` must have: step.id, step.ts_start_ns, step.ts_end_ns.
    experiment_id is auto-filled.
    """
    df = pd.DataFrame(steps)
    df["experiment_id"] = "synthetic-experiment"
    # Ensure integer types to match real data
    for col in ["step.id", "step.ts_start_ns", "step.ts_end_ns"]:
        df[col] = df[col].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# _estimate_kv_length tests
# ---------------------------------------------------------------------------

class TestEstimateKVLength:
    """Verify KV length estimation for individual requests."""

    def test_all_output_tokens_before_step(self):
        """All output tokens were generated before step start -> full KV."""
        row = pd.Series({
            "input_tokens": 100,
            "output_token_times": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        # step starts at 6s (= 6_000_000_000 ns), all 5 tokens are before
        step_start_ns = 6_000_000_000
        assert _estimate_kv_length(row, step_start_ns) == 100 + 5

    def test_some_output_tokens_before_step(self):
        """Only tokens with timestamp < step_start count."""
        row = pd.Series({
            "input_tokens": 50,
            "output_token_times": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        # step starts at 3.5s -> tokens at 1.0, 2.0, 3.0 are before
        step_start_ns = 3_500_000_000
        assert _estimate_kv_length(row, step_start_ns) == 50 + 3

    def test_no_output_tokens_before_step(self):
        """Step starts before any output tokens -> KV = input_tokens only."""
        row = pd.Series({
            "input_tokens": 200,
            "output_token_times": [10.0, 11.0, 12.0],
        })
        step_start_ns = 5_000_000_000  # 5s, before first token at 10s
        assert _estimate_kv_length(row, step_start_ns) == 200

    def test_empty_output_token_times(self):
        """Prefill-only request (no output tokens generated yet)."""
        row = pd.Series({
            "input_tokens": 128,
            "output_token_times": [],
        })
        step_start_ns = 10_000_000_000
        assert _estimate_kv_length(row, step_start_ns) == 128

    def test_boundary_token_at_exact_step_start(self):
        """Token at exactly step_start_ns is NOT counted (strict <)."""
        row = pd.Series({
            "input_tokens": 100,
            "output_token_times": [1.0, 2.0, 3.0],
        })
        # step starts at exactly 2.0s -> only 1.0s token qualifies
        step_start_ns = 2_000_000_000
        assert _estimate_kv_length(row, step_start_ns) == 100 + 1


# ---------------------------------------------------------------------------
# extract_kv_features tests
# ---------------------------------------------------------------------------

class TestExtractKVFeatures:
    """Verify KV feature extraction on a small synthetic dataset.

    Scenario: 3 requests, 5 steps spanning 0-50 seconds.

    Request 0: input=100, active 5s-25s, output tokens at 10s, 15s, 20s
    Request 1: input=200, active 0s-40s, output tokens at 5s, 10s, 15s, 20s, 25s, 30s, 35s
    Request 2: input=50,  active 20s-45s, output tokens at 25s, 30s
    """

    @pytest.fixture
    def lifecycle_df(self):
        return _make_lifecycle_df([
            {
                "start_time": 5.0,
                "end_time": 25.0,
                "input_tokens": 100,
                "output_tokens": 3,
                "output_token_times": [10.0, 15.0, 20.0],
            },
            {
                "start_time": 0.0,
                "end_time": 40.0,
                "input_tokens": 200,
                "output_tokens": 7,
                "output_token_times": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
            },
            {
                "start_time": 20.0,
                "end_time": 45.0,
                "input_tokens": 50,
                "output_tokens": 2,
                "output_token_times": [25.0, 30.0],
            },
        ])

    @pytest.fixture
    def steps_df(self):
        return _make_steps_df([
            # Step 0: 0-10s
            {"step.id": 0, "step.ts_start_ns": 0_000_000_000, "step.ts_end_ns": 10_000_000_000},
            # Step 1: 10-20s
            {"step.id": 1, "step.ts_start_ns": 10_000_000_000, "step.ts_end_ns": 20_000_000_000},
            # Step 2: 20-30s
            {"step.id": 2, "step.ts_start_ns": 20_000_000_000, "step.ts_end_ns": 30_000_000_000},
            # Step 3: 30-40s
            {"step.id": 3, "step.ts_start_ns": 30_000_000_000, "step.ts_end_ns": 40_000_000_000},
            # Step 4: 40-50s
            {"step.id": 4, "step.ts_start_ns": 40_000_000_000, "step.ts_end_ns": 50_000_000_000},
        ])

    @pytest.fixture
    def result(self, steps_df, lifecycle_df):
        return extract_kv_features(steps_df, lifecycle_df)

    def test_output_has_kv_columns(self, result):
        """Result DataFrame must contain all four KV feature columns."""
        for col in ["kv_mean", "kv_max", "kv_sum", "kv_count"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_preserves_original_columns(self, result, steps_df):
        """All original step columns must be preserved."""
        for col in steps_df.columns:
            assert col in result.columns, f"Original column lost: {col}"

    def test_output_row_count_matches(self, result, steps_df):
        """Output must have the same number of rows as input."""
        assert len(result) == len(steps_df)

    def test_step0_active_requests(self, result):
        """Step 0 (0-10s): Req0 active (5<10, 25>0), Req1 active (0<10, 40>0).

        Req0 at t=0s: input=100, tokens before 0s = 0, kv=100
        Req1 at t=0s: input=200, tokens before 0s = 0, kv=200
        """
        row = result.iloc[0]
        assert row["kv_count"] == 2
        # Req0 kv=100, Req1 kv=200 -> mean=150, max=200, sum=300
        assert row["kv_mean"] == pytest.approx(150.0)
        assert row["kv_max"] == pytest.approx(200.0)
        assert row["kv_sum"] == pytest.approx(300.0)

    def test_step1_active_requests(self, result):
        """Step 1 (10-20s): Req0 active (5<20, 25>10), Req1 active (0<20, 40>10).

        Req0 at t=10s: input=100, tokens before 10s = 0 (token at 10.0 is NOT < 10.0), kv=100
        Req1 at t=10s: input=200, tokens before 10s = 1 (token at 5.0), kv=201
        """
        row = result.iloc[1]
        assert row["kv_count"] == 2
        assert row["kv_mean"] == pytest.approx(150.5)
        assert row["kv_max"] == pytest.approx(201.0)
        assert row["kv_sum"] == pytest.approx(301.0)

    def test_step2_active_requests(self, result):
        """Step 2 (20-30s): All 3 requests active.

        Req0 at t=20s: input=100, tokens before 20s = 2 (10.0, 15.0), kv=102
        Req1 at t=20s: input=200, tokens before 20s = 3 (5.0, 10.0, 15.0), kv=203
        Req2 at t=20s: input=50,  tokens before 20s = 0, kv=50
        """
        row = result.iloc[2]
        assert row["kv_count"] == 3
        expected_sum = 102.0 + 203.0 + 50.0
        assert row["kv_sum"] == pytest.approx(expected_sum)
        assert row["kv_max"] == pytest.approx(203.0)
        assert row["kv_mean"] == pytest.approx(expected_sum / 3.0)

    def test_step3_active_requests(self, result):
        """Step 3 (30-40s): Req1 active (0<40, 40>30), Req2 active (20<40, 45>30).

        Req1 at t=30s: input=200, tokens before 30s = 5 (5,10,15,20,25), kv=205
        Req2 at t=30s: input=50,  tokens before 30s = 1 (25.0), kv=51
        """
        row = result.iloc[3]
        assert row["kv_count"] == 2
        assert row["kv_sum"] == pytest.approx(256.0)
        assert row["kv_max"] == pytest.approx(205.0)
        assert row["kv_mean"] == pytest.approx(128.0)

    def test_step4_only_req2(self, result):
        """Step 4 (40-50s): Only Req2 active (20<50, 45>40).

        Req2 at t=40s: input=50, tokens before 40s = 2 (25.0, 30.0), kv=52
        """
        row = result.iloc[4]
        assert row["kv_count"] == 1
        assert row["kv_sum"] == pytest.approx(52.0)
        assert row["kv_max"] == pytest.approx(52.0)
        assert row["kv_mean"] == pytest.approx(52.0)


class TestExtractKVFeaturesEdgeCases:
    """Edge cases for extract_kv_features."""

    def test_step_with_no_active_requests(self):
        """A step outside all request windows -> zeros for all KV columns."""
        lifecycle_df = _make_lifecycle_df([
            {
                "start_time": 10.0,
                "end_time": 20.0,
                "input_tokens": 100,
                "output_tokens": 3,
                "output_token_times": [12.0, 14.0, 16.0],
            },
        ])
        # Step is entirely before the request
        steps_df = _make_steps_df([
            {"step.id": 0, "step.ts_start_ns": 0, "step.ts_end_ns": 5_000_000_000},
        ])

        result = extract_kv_features(steps_df, lifecycle_df)
        row = result.iloc[0]
        assert row["kv_mean"] == pytest.approx(0.0)
        assert row["kv_max"] == pytest.approx(0.0)
        assert row["kv_sum"] == pytest.approx(0.0)
        assert row["kv_count"] == 0

    def test_step_after_all_requests(self):
        """Step entirely after all requests -> zeros."""
        lifecycle_df = _make_lifecycle_df([
            {
                "start_time": 1.0,
                "end_time": 5.0,
                "input_tokens": 50,
                "output_tokens": 2,
                "output_token_times": [2.0, 3.0],
            },
        ])
        steps_df = _make_steps_df([
            {"step.id": 0, "step.ts_start_ns": 100_000_000_000, "step.ts_end_ns": 110_000_000_000},
        ])

        result = extract_kv_features(steps_df, lifecycle_df)
        row = result.iloc[0]
        assert row["kv_count"] == 0
        assert row["kv_sum"] == pytest.approx(0.0)

    def test_prefill_only_request(self):
        """Request with empty output_token_times -> KV = input_tokens."""
        lifecycle_df = _make_lifecycle_df([
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "input_tokens": 256,
                "output_tokens": 0,
                "output_token_times": [],
            },
        ])
        steps_df = _make_steps_df([
            {"step.id": 0, "step.ts_start_ns": 2_000_000_000, "step.ts_end_ns": 8_000_000_000},
        ])

        result = extract_kv_features(steps_df, lifecycle_df)
        row = result.iloc[0]
        assert row["kv_count"] == 1
        assert row["kv_mean"] == pytest.approx(256.0)
        assert row["kv_max"] == pytest.approx(256.0)
        assert row["kv_sum"] == pytest.approx(256.0)

    def test_empty_lifecycle_df(self):
        """No requests at all -> all steps get zeros."""
        lifecycle_df = _make_lifecycle_df([])
        # Ensure the required columns exist even with no rows
        for col in ["start_time", "end_time", "input_tokens", "output_tokens", "output_token_times"]:
            if col not in lifecycle_df.columns:
                lifecycle_df[col] = pd.Series(dtype="float64")

        steps_df = _make_steps_df([
            {"step.id": 0, "step.ts_start_ns": 0, "step.ts_end_ns": 10_000_000_000},
        ])

        result = extract_kv_features(steps_df, lifecycle_df)
        row = result.iloc[0]
        assert row["kv_count"] == 0
        assert row["kv_sum"] == pytest.approx(0.0)

    def test_multiple_steps_same_request(self):
        """A long-lived request spans multiple steps; KV grows over time."""
        lifecycle_df = _make_lifecycle_df([
            {
                "start_time": 0.0,
                "end_time": 30.0,
                "input_tokens": 100,
                "output_tokens": 6,
                "output_token_times": [5.0, 10.0, 15.0, 20.0, 25.0, 28.0],
            },
        ])
        steps_df = _make_steps_df([
            {"step.id": 0, "step.ts_start_ns": 0_000_000_000, "step.ts_end_ns": 8_000_000_000},
            {"step.id": 1, "step.ts_start_ns": 8_000_000_000, "step.ts_end_ns": 16_000_000_000},
            {"step.id": 2, "step.ts_start_ns": 16_000_000_000, "step.ts_end_ns": 24_000_000_000},
        ])

        result = extract_kv_features(steps_df, lifecycle_df)

        # Step 0 (start=0s): 0 output tokens before 0s -> kv=100
        assert result.iloc[0]["kv_mean"] == pytest.approx(100.0)
        # Step 1 (start=8s): 1 token before 8s (at 5.0) -> kv=101
        assert result.iloc[1]["kv_mean"] == pytest.approx(101.0)
        # Step 2 (start=16s): 3 tokens before 16s (5.0, 10.0, 15.0) -> kv=103
        assert result.iloc[2]["kv_mean"] == pytest.approx(103.0)

        # Verify monotonicity: KV length should be non-decreasing for a single request
        kv_values = result["kv_mean"].values
        assert all(kv_values[i] <= kv_values[i + 1] for i in range(len(kv_values) - 1))
