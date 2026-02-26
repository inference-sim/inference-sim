"""Tests for sampling bias characterization.

TDD: These tests are written first, before the implementation in
sampling_analysis.py. They verify that the 10% step_tracing_sample_rate
does not introduce systematic bias into the ground truth dataset.
"""

import pytest

from data_loader import load_all_experiments
from sampling_analysis import characterize_sampling


@pytest.fixture(scope="module")
def all_data():
    """Load all experiment data once for the entire test module."""
    return load_all_experiments()


class TestStepIdNotContiguous:
    """step.id values should have gaps, confirming 10% sampling."""

    def test_step_id_not_contiguous(self, all_data):
        """Step IDs within each experiment should NOT be contiguous (1, 2, 3, ...).

        If sampling is 10%, we expect gaps between consecutive step.id values,
        meaning the mean gap should be substantially greater than 1.
        """
        for exp_id, group in all_data.groupby("experiment_id"):
            step_ids = group["step.id"].sort_values()
            gaps = step_ids.diff().dropna()
            mean_gap = gaps.mean()
            assert mean_gap > 2.0, (
                f"Experiment {exp_id}: mean step.id gap is {mean_gap:.2f}, "
                f"expected > 2.0 for 10% sampling (contiguous would be 1.0)"
            )


class TestPerExperimentRepresentation:
    """All 16 experiments should have similar proportions of steps."""

    def test_per_experiment_representation(self, all_data):
        """No single experiment should dominate the dataset.

        Experiments vary in step count due to different model/workload
        throughput characteristics (e.g., llama-2-7b processes more steps
        than mixtral-8x7b in the same wall-clock time). We verify the
        ratio stays within 5x -- enough to confirm no extreme outlier
        while acknowledging legitimate throughput differences.
        """
        counts = all_data.groupby("experiment_id").size()
        min_count = counts.min()
        max_count = counts.max()
        ratio = max_count / min_count
        assert ratio < 5.0, (
            f"Step count ratio between largest and smallest experiment is "
            f"{ratio:.2f} (max={max_count}, min={min_count}), expected < 5.0"
        )


class TestStepIdsNotPeriodic:
    """Gaps between consecutive step.id values should NOT be constant."""

    def test_step_ids_not_periodic(self, all_data):
        """If sampling were periodic (e.g., every 10th step), the gaps would
        all be identical (std close to 0). We verify this is NOT the case.

        A std < 0.5 would indicate near-constant gaps (periodic sampling).
        We expect std >> 0.5 for random/pseudo-random sampling.
        """
        for exp_id, group in all_data.groupby("experiment_id"):
            step_ids = group["step.id"].sort_values()
            gaps = step_ids.diff().dropna()
            gap_std = gaps.std()
            assert gap_std > 0.5, (
                f"Experiment {exp_id}: step.id gap std is {gap_std:.2f}, "
                f"which suggests periodic sampling (expected > 0.5 for "
                f"random sampling)"
            )


class TestCharacterizeReturnsReport:
    """The characterize function should return a well-structured report."""

    def test_characterize_returns_report(self, all_data):
        """characterize_sampling should return a dict with all expected keys."""
        report = characterize_sampling(all_data)

        expected_keys = {
            "total_steps",
            "per_experiment_counts",
            "step_id_gaps",
            "is_periodic",
            "coverage_uniformity",
            "summary",
        }
        assert isinstance(report, dict)
        assert set(report.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(report.keys())}, "
            f"Extra keys: {set(report.keys()) - expected_keys}"
        )

    def test_total_steps_is_positive_int(self, all_data):
        report = characterize_sampling(all_data)
        assert isinstance(report["total_steps"], int)
        assert report["total_steps"] > 0

    def test_per_experiment_counts_has_all_experiments(self, all_data):
        report = characterize_sampling(all_data)
        n_experiments = all_data["experiment_id"].nunique()
        assert len(report["per_experiment_counts"]) == n_experiments

    def test_step_id_gaps_has_per_experiment_stats(self, all_data):
        report = characterize_sampling(all_data)
        gaps = report["step_id_gaps"]
        for exp_id in all_data["experiment_id"].unique():
            assert exp_id in gaps, f"Missing experiment {exp_id} in step_id_gaps"
            stats = gaps[exp_id]
            for key in ["mean", "std", "min", "max"]:
                assert key in stats, (
                    f"Missing key '{key}' in step_id_gaps[{exp_id}]"
                )

    def test_is_periodic_is_bool(self, all_data):
        report = characterize_sampling(all_data)
        assert isinstance(report["is_periodic"], bool)

    def test_is_not_periodic(self, all_data):
        """Given the data has random sampling, is_periodic should be False."""
        report = characterize_sampling(all_data)
        assert report["is_periodic"] is False

    def test_coverage_uniformity_is_low(self, all_data):
        """Coefficient of variation should be low (experiments are similar size)."""
        report = characterize_sampling(all_data)
        assert isinstance(report["coverage_uniformity"], float)
        # CV < 0.5 means reasonably uniform
        assert report["coverage_uniformity"] < 0.5

    def test_summary_is_nonempty_string(self, all_data):
        report = characterize_sampling(all_data)
        assert isinstance(report["summary"], str)
        assert len(report["summary"]) > 0
