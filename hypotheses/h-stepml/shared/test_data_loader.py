"""Tests for the StepML ground truth data loader.

TDD: These tests are written first, before the implementation in data_loader.py.
"""

import os

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    load_all_experiments,
    load_experiment_steps,
    load_lifecycle_data,
    parse_experiment_metadata,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "eval", "ground_truth"
)

# Pick a known experiment for single-experiment tests
SAMPLE_EXPERIMENT = "20260217-155451-llama-2-7b-tp1-codegen"
SAMPLE_EXPERIMENT_DIR = os.path.join(DATA_ROOT, SAMPLE_EXPERIMENT)

# All expected step-level feature columns from BATCH_SUMMARY events
EXPECTED_STEP_COLUMNS = [
    "step.id",
    "step.ts_start_ns",
    "step.ts_end_ns",
    "step.duration_us",
    "batch.prefill_tokens",
    "batch.decode_tokens",
    "batch.scheduled_tokens",
    "batch.num_prefill_reqs",
    "batch.num_decode_reqs",
    "batch.num_finished",
    "batch.num_preempted",
    "queue.running_depth",
    "queue.waiting_depth",
    "kv.usage_gpu_ratio",
    "kv.blocks_free_gpu",
    "kv.blocks_total_gpu",
]


class TestParseExperimentMetadata:
    """Verify directory-name parsing into structured metadata."""

    def test_simple_model_name(self):
        meta = parse_experiment_metadata("20260217-155451-llama-2-7b-tp1-codegen")
        assert meta["model"] == "llama-2-7b"
        assert meta["tp"] == 1
        assert meta["workload"] == "codegen"
        assert meta["timestamp"] == "20260217-155451"

    def test_model_with_hf_suffix(self):
        meta = parse_experiment_metadata("20260217-203421-llama-2-70b-hf-tp4-codegen")
        assert meta["model"] == "llama-2-70b-hf"
        assert meta["tp"] == 4
        assert meta["workload"] == "codegen"

    def test_mixtral_model_name(self):
        meta = parse_experiment_metadata(
            "20260218-120914-mixtral-8x7b-v0-1-tp2-codegen"
        )
        assert meta["model"] == "mixtral-8x7b-v0-1"
        assert meta["tp"] == 2
        assert meta["workload"] == "codegen"

    def test_codellama_model_name(self):
        meta = parse_experiment_metadata("20260218-150304-codellama-34b-tp2-general")
        assert meta["model"] == "codellama-34b"
        assert meta["tp"] == 2
        assert meta["workload"] == "general"

    def test_all_workload_types_present(self):
        """Every directory name in the dataset should parse without error."""
        for dirname in os.listdir(DATA_ROOT):
            if not os.path.isdir(os.path.join(DATA_ROOT, dirname)):
                continue
            meta = parse_experiment_metadata(dirname)
            assert meta["workload"] in {
                "codegen",
                "roleplay",
                "reasoning",
                "general",
            }, f"Unexpected workload '{meta['workload']}' in {dirname}"


class TestLoadSingleExperimentSteps:
    """Verify loading step data from a single experiment."""

    @pytest.fixture(scope="class")
    def steps_df(self):
        return load_experiment_steps(SAMPLE_EXPERIMENT_DIR)

    def test_row_count_positive(self, steps_df):
        assert len(steps_df) > 0, "Expected at least one step row"

    def test_all_expected_columns_present(self, steps_df):
        for col in EXPECTED_STEP_COLUMNS:
            assert col in steps_df.columns, f"Missing column: {col}"

    def test_experiment_id_column(self, steps_df):
        assert "experiment_id" in steps_df.columns
        assert (steps_df["experiment_id"] == SAMPLE_EXPERIMENT).all()

    def test_step_duration_positive(self, steps_df):
        assert (steps_df["step.duration_us"] > 0).all(), (
            "All step durations should be positive"
        )

    def test_integer_columns_are_numeric(self, steps_df):
        int_cols = [
            "step.id",
            "step.ts_start_ns",
            "step.ts_end_ns",
            "step.duration_us",
            "batch.prefill_tokens",
            "batch.decode_tokens",
            "batch.scheduled_tokens",
            "batch.num_prefill_reqs",
            "batch.num_decode_reqs",
            "batch.num_finished",
            "batch.num_preempted",
            "queue.running_depth",
            "queue.waiting_depth",
            "kv.blocks_free_gpu",
            "kv.blocks_total_gpu",
        ]
        for col in int_cols:
            assert pd.api.types.is_numeric_dtype(steps_df[col]), (
                f"Column {col} should be numeric, got {steps_df[col].dtype}"
            )

    def test_kv_ratio_is_float(self, steps_df):
        assert pd.api.types.is_float_dtype(steps_df["kv.usage_gpu_ratio"])


class TestLoadAllExperiments:
    """Verify loading and concatenating all experiments."""

    @pytest.fixture(scope="class")
    def all_df(self):
        return load_all_experiments(DATA_ROOT)

    def test_sixteen_unique_experiments(self, all_df):
        unique_ids = all_df["experiment_id"].nunique()
        assert unique_ids == 16, f"Expected 16 experiments, got {unique_ids}"

    def test_positive_rows_per_experiment(self, all_df):
        for exp_id, group in all_df.groupby("experiment_id"):
            assert len(group) > 0, f"Experiment {exp_id} has zero rows"

    def test_metadata_columns_present(self, all_df):
        for col in ["model", "tp", "workload", "timestamp"]:
            assert col in all_df.columns, f"Missing metadata column: {col}"

    def test_tp_values_are_int(self, all_df):
        assert pd.api.types.is_integer_dtype(all_df["tp"])

    def test_all_step_columns_present(self, all_df):
        for col in EXPECTED_STEP_COLUMNS:
            assert col in all_df.columns, f"Missing column: {col}"


class TestStepDurationConsistency:
    """Verify step.duration_us matches (ts_end_ns - ts_start_ns) / 1000."""

    @pytest.fixture(scope="class")
    def steps_df(self):
        return load_experiment_steps(SAMPLE_EXPERIMENT_DIR)

    def test_duration_matches_timestamps(self, steps_df):
        computed_us = (
            steps_df["step.ts_end_ns"] - steps_df["step.ts_start_ns"]
        ) / 1000.0
        reported_us = steps_df["step.duration_us"].astype(float)

        # Allow 1 microsecond tolerance for integer rounding
        diff = np.abs(computed_us - reported_us)
        assert (diff <= 1.0).all(), (
            f"Duration mismatch: max diff = {diff.max():.2f} us, "
            f"mean diff = {diff.mean():.2f} us"
        )


class TestLoadLifecycleData:
    """Verify loading per-request lifecycle metrics."""

    @pytest.fixture(scope="class")
    def lifecycle_df(self):
        return load_lifecycle_data(SAMPLE_EXPERIMENT_DIR)

    def test_row_count_positive(self, lifecycle_df):
        assert len(lifecycle_df) > 0, "Expected at least one request"

    def test_expected_columns(self, lifecycle_df):
        for col in [
            "start_time",
            "end_time",
            "input_tokens",
            "output_tokens",
            "output_token_times",
        ]:
            assert col in lifecycle_df.columns, f"Missing column: {col}"

    def test_start_before_end(self, lifecycle_df):
        assert (lifecycle_df["end_time"] >= lifecycle_df["start_time"]).all()

    def test_token_counts_non_negative(self, lifecycle_df):
        assert (lifecycle_df["input_tokens"] > 0).all()
        # Some requests may have 0 output tokens (e.g., preempted or errored)
        assert (lifecycle_df["output_tokens"] >= 0).all()
        # But the vast majority should have produced output
        assert (lifecycle_df["output_tokens"] > 0).mean() > 0.95

    def test_output_token_times_are_lists(self, lifecycle_df):
        sample = lifecycle_df["output_token_times"].iloc[0]
        assert isinstance(sample, list), (
            f"Expected list, got {type(sample)}"
        )

    def test_request_id_index(self, lifecycle_df):
        assert lifecycle_df.index.name == "request_id"
