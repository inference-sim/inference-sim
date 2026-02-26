"""Data splitting strategies for StepML experiments.

Provides three splitting functions for ML training and evaluation:

- **temporal_split**: Within each experiment, sorts steps by step.id and
  assigns the first 60% to train, next 20% to validation, last 20% to test.
  Every experiment is represented in all three splits (stratified).

- **leave_one_model_out**: 4-fold cross-validation where each fold holds out
  all data from one model. Model names are normalized so that variants
  (e.g., "llama-2-70b" and "llama-2-70b-hf") are grouped together.

- **leave_one_workload_out**: 4-fold cross-validation where each fold holds
  out all data from one workload type.

All functions return integer indices (into the DataFrame's index) rather than
boolean masks, making them compatible with iloc/loc indexing.
"""

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model name normalization
# ---------------------------------------------------------------------------
# Some experiments use variant names for the same model, e.g.:
#   "llama-2-70b" and "llama-2-70b-hf"
# We normalize by stripping known suffixes so they group together.
_MODEL_SUFFIX_PATTERN = re.compile(r"-hf$")


def _normalize_model_name(name: str) -> str:
    """Normalize model names by stripping variant suffixes.

    Examples:
        "llama-2-70b-hf" -> "llama-2-70b"
        "llama-2-7b"     -> "llama-2-7b"
        "mixtral-8x7b-v0-1" -> "mixtral-8x7b-v0-1"
    """
    return _MODEL_SUFFIX_PATTERN.sub("", name)


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    valid_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Split data temporally within each experiment.

    Within each experiment (grouped by ``experiment_id``), rows are sorted by
    ``step.id`` and the first ``train_frac`` fraction is assigned to training,
    the next ``valid_frac`` fraction to validation, and the remainder to test.

    This split is deterministic given the data (step ordering is fixed), so
    the ``seed`` parameter is included for API consistency but does not affect
    the result.

    Args:
        df: DataFrame with ``experiment_id`` and ``step.id`` columns.
        train_frac: Fraction of steps for training (default 0.6).
        valid_frac: Fraction of steps for validation (default 0.2).
        seed: Included for API consistency; does not affect the split.

    Returns:
        Dict with keys ``"train"``, ``"valid"``, ``"test"``, each mapping to a
        sorted numpy array of integer row indices from df.index.
    """
    train_indices = []
    valid_indices = []
    test_indices = []

    for _exp_id, group in df.groupby("experiment_id"):
        # Sort by step.id within this experiment
        sorted_group = group.sort_values("step.id")
        indices = sorted_group.index.values
        n = len(indices)

        # Compute split boundaries
        n_train = int(round(n * train_frac))
        n_valid = int(round(n * valid_frac))

        train_indices.append(indices[:n_train])
        valid_indices.append(indices[n_train : n_train + n_valid])
        test_indices.append(indices[n_train + n_valid :])

    return {
        "train": np.sort(np.concatenate(train_indices)),
        "valid": np.sort(np.concatenate(valid_indices)),
        "test": np.sort(np.concatenate(test_indices)),
    }


# ---------------------------------------------------------------------------
# Leave-one-model-out
# ---------------------------------------------------------------------------


def leave_one_model_out(df: pd.DataFrame) -> list:
    """Leave-one-model-out cross-validation folds.

    Produces 4 folds, one per unique model. Model names are normalized so
    that variants like "llama-2-70b" and "llama-2-70b-hf" are treated as
    the same model.

    Within each fold, the test set contains all rows for the held-out model,
    and the training set contains all remaining rows.

    Args:
        df: DataFrame with a ``model`` column.

    Returns:
        List of 4 dicts, each with keys:
        - ``"train"``: numpy array of training row indices
        - ``"test"``: numpy array of test row indices
        - ``"holdout_model"``: the normalized model name held out

    Note:
        As a side effect, the ``model`` column in ``df`` is normalized
        in-place (e.g., "llama-2-70b-hf" becomes "llama-2-70b"). This
        ensures downstream consumers see consistent model names.
    """
    normalized = df["model"].map(_normalize_model_name)
    unique_models = sorted(normalized.unique())

    folds = []
    for holdout_model in unique_models:
        is_holdout = normalized == holdout_model
        test_idx = df.index[is_holdout].values
        train_idx = df.index[~is_holdout].values

        folds.append(
            {
                "train": np.sort(train_idx),
                "test": np.sort(test_idx),
                "holdout_model": holdout_model,
            }
        )

    # Normalize the model column in the dataframe so downstream consumers
    # see consistent names. This is intentional: variants of the same model
    # should be unified.
    df["model"] = normalized.values

    return folds


# ---------------------------------------------------------------------------
# Leave-one-workload-out
# ---------------------------------------------------------------------------


def leave_one_workload_out(df: pd.DataFrame) -> list:
    """Leave-one-workload-out cross-validation folds.

    Produces 4 folds, one per unique workload. Within each fold, the test set
    contains all rows for the held-out workload, and the training set contains
    all remaining rows.

    Args:
        df: DataFrame with a ``workload`` column.

    Returns:
        List of 4 dicts, each with keys:
        - ``"train"``: numpy array of training row indices
        - ``"test"``: numpy array of test row indices
        - ``"holdout_workload"``: the workload name held out
    """
    unique_workloads = sorted(df["workload"].unique())

    folds = []
    for holdout_workload in unique_workloads:
        is_holdout = df["workload"] == holdout_workload
        test_idx = df.index[is_holdout].values
        train_idx = df.index[~is_holdout].values

        folds.append(
            {
                "train": np.sort(train_idx),
                "test": np.sort(test_idx),
                "holdout_workload": holdout_workload,
            }
        )

    return folds
