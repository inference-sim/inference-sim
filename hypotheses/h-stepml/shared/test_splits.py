"""Tests for data splitting strategies.

TDD: These tests are written first, before the implementation in splits.py.

Covers:
- Temporal split (train/valid/test by step ordering within experiments)
- Leave-one-model-out cross-validation
- Leave-one-workload-out cross-validation
"""

import numpy as np
import pytest

from data_loader import load_all_experiments
from splits import leave_one_model_out, leave_one_workload_out, temporal_split


@pytest.fixture(scope="module")
def all_data():
    return load_all_experiments()


# ---------------------------------------------------------------------------
# Temporal split tests
# ---------------------------------------------------------------------------


class TestTemporalSplitOrdering:
    """BC-0-2: Within each experiment, all training step_ids < all validation
    step_ids < all test step_ids."""

    def test_temporal_split_ordering(self, all_data):
        splits = temporal_split(all_data)
        train_idx = splits["train"]
        valid_idx = splits["valid"]
        test_idx = splits["test"]

        for exp_id, group in all_data.groupby("experiment_id"):
            group_positions = group.index
            exp_train = np.intersect1d(train_idx, group_positions)
            exp_valid = np.intersect1d(valid_idx, group_positions)
            exp_test = np.intersect1d(test_idx, group_positions)

            if len(exp_train) > 0 and len(exp_valid) > 0:
                train_step_ids = all_data.loc[exp_train, "step.id"].values
                valid_step_ids = all_data.loc[exp_valid, "step.id"].values
                assert train_step_ids.max() < valid_step_ids.min(), (
                    f"Experiment {exp_id}: max train step.id "
                    f"({train_step_ids.max()}) >= min valid step.id "
                    f"({valid_step_ids.min()})"
                )

            if len(exp_valid) > 0 and len(exp_test) > 0:
                valid_step_ids = all_data.loc[exp_valid, "step.id"].values
                test_step_ids = all_data.loc[exp_test, "step.id"].values
                assert valid_step_ids.max() < test_step_ids.min(), (
                    f"Experiment {exp_id}: max valid step.id "
                    f"({valid_step_ids.max()}) >= min test step.id "
                    f"({test_step_ids.min()})"
                )


class TestTemporalSplitProportions:
    """Approx 60/20/20 split within each experiment (+-2%)."""

    def test_temporal_split_proportions(self, all_data):
        splits = temporal_split(all_data)
        train_idx = splits["train"]
        valid_idx = splits["valid"]
        test_idx = splits["test"]

        for exp_id, group in all_data.groupby("experiment_id"):
            group_positions = group.index
            n = len(group_positions)

            n_train = len(np.intersect1d(train_idx, group_positions))
            n_valid = len(np.intersect1d(valid_idx, group_positions))
            n_test = len(np.intersect1d(test_idx, group_positions))

            train_frac = n_train / n
            valid_frac = n_valid / n
            test_frac = n_test / n

            assert abs(train_frac - 0.6) <= 0.02, (
                f"Experiment {exp_id}: train fraction {train_frac:.3f} "
                f"not within +-2% of 0.6"
            )
            assert abs(valid_frac - 0.2) <= 0.02, (
                f"Experiment {exp_id}: valid fraction {valid_frac:.3f} "
                f"not within +-2% of 0.2"
            )
            assert abs(test_frac - 0.2) <= 0.02, (
                f"Experiment {exp_id}: test fraction {test_frac:.3f} "
                f"not within +-2% of 0.2"
            )


class TestTemporalSplitStratification:
    """BC-0-3: All 16 experiments are represented in each split."""

    def test_temporal_split_stratification(self, all_data):
        splits = temporal_split(all_data)

        for split_name in ["train", "valid", "test"]:
            split_idx = splits[split_name]
            experiments_in_split = all_data.loc[split_idx, "experiment_id"].nunique()
            assert experiments_in_split == 16, (
                f"Split '{split_name}' contains {experiments_in_split} "
                f"experiments, expected 16"
            )


class TestTemporalSplitNoOverlap:
    """Train, valid, test indices are disjoint; union equals full dataset."""

    def test_temporal_split_no_overlap(self, all_data):
        splits = temporal_split(all_data)
        train_idx = splits["train"]
        valid_idx = splits["valid"]
        test_idx = splits["test"]

        # Disjoint check
        assert len(np.intersect1d(train_idx, valid_idx)) == 0, (
            "Train and valid indices overlap"
        )
        assert len(np.intersect1d(train_idx, test_idx)) == 0, (
            "Train and test indices overlap"
        )
        assert len(np.intersect1d(valid_idx, test_idx)) == 0, (
            "Valid and test indices overlap"
        )

        # Union equals full dataset
        all_split_idx = np.sort(np.concatenate([train_idx, valid_idx, test_idx]))
        expected_idx = np.sort(all_data.index.values)
        np.testing.assert_array_equal(
            all_split_idx, expected_idx, err_msg="Union of splits != full dataset"
        )


# ---------------------------------------------------------------------------
# Leave-one-model-out tests
# ---------------------------------------------------------------------------


class TestLeaveOneModelOutFolds:
    """4 folds; each fold holds out exactly 1 model's data; holdout model
    appears only in test, not train."""

    def test_leave_one_model_out_folds(self, all_data):
        folds = leave_one_model_out(all_data)

        assert len(folds) == 4, f"Expected 4 folds, got {len(folds)}"

        holdout_models = set()
        for fold in folds:
            assert "train" in fold
            assert "test" in fold
            assert "holdout_model" in fold

            holdout = fold["holdout_model"]
            holdout_models.add(holdout)

            train_idx = fold["train"]
            test_idx = fold["test"]

            # Holdout model should NOT appear in training set
            train_models = all_data.loc[train_idx, "model"].unique()
            assert holdout not in train_models, (
                f"Holdout model '{holdout}' found in training set"
            )

            # All test rows should belong to the holdout model
            test_models = all_data.loc[test_idx, "model"].unique()
            assert len(test_models) == 1, (
                f"Expected exactly 1 model in test set, got {len(test_models)}: "
                f"{test_models}"
            )
            assert test_models[0] == holdout, (
                f"Test model '{test_models[0]}' != holdout '{holdout}'"
            )

            # Train + test should be disjoint
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # Train + test should cover the full dataset
            all_idx = np.sort(np.concatenate([train_idx, test_idx]))
            expected_idx = np.sort(all_data.index.values)
            np.testing.assert_array_equal(all_idx, expected_idx)

        # All 4 models should have been held out across folds
        assert len(holdout_models) == 4, (
            f"Expected 4 holdout models, got {len(holdout_models)}: "
            f"{holdout_models}"
        )


# ---------------------------------------------------------------------------
# Leave-one-workload-out tests
# ---------------------------------------------------------------------------


class TestLeaveOneWorkloadOutFolds:
    """4 folds; each fold holds out exactly 1 workload's data; holdout workload
    appears only in test, not train."""

    def test_leave_one_workload_out_folds(self, all_data):
        folds = leave_one_workload_out(all_data)

        assert len(folds) == 4, f"Expected 4 folds, got {len(folds)}"

        holdout_workloads = set()
        for fold in folds:
            assert "train" in fold
            assert "test" in fold
            assert "holdout_workload" in fold

            holdout = fold["holdout_workload"]
            holdout_workloads.add(holdout)

            train_idx = fold["train"]
            test_idx = fold["test"]

            # Holdout workload should NOT appear in training set
            train_workloads = all_data.loc[train_idx, "workload"].unique()
            assert holdout not in train_workloads, (
                f"Holdout workload '{holdout}' found in training set"
            )

            # All test rows should belong to the holdout workload
            test_workloads = all_data.loc[test_idx, "workload"].unique()
            assert len(test_workloads) == 1, (
                f"Expected exactly 1 workload in test set, got "
                f"{len(test_workloads)}: {test_workloads}"
            )
            assert test_workloads[0] == holdout, (
                f"Test workload '{test_workloads[0]}' != holdout '{holdout}'"
            )

            # Train + test should be disjoint
            assert len(np.intersect1d(train_idx, test_idx)) == 0

            # Train + test should cover the full dataset
            all_idx = np.sort(np.concatenate([train_idx, test_idx]))
            expected_idx = np.sort(all_data.index.values)
            np.testing.assert_array_equal(all_idx, expected_idx)

        # All 4 workloads should have been held out across folds
        assert len(holdout_workloads) == 4, (
            f"Expected 4 holdout workloads, got {len(holdout_workloads)}: "
            f"{holdout_workloads}"
        )


# ---------------------------------------------------------------------------
# Reproducibility test
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same call with same seed produces identical indices."""

    def test_reproducibility(self, all_data):
        splits_a = temporal_split(all_data, seed=42)
        splits_b = temporal_split(all_data, seed=42)

        for key in ["train", "valid", "test"]:
            np.testing.assert_array_equal(
                splits_a[key],
                splits_b[key],
                err_msg=f"Split '{key}' differs between two calls with same seed",
            )
