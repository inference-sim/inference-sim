"""Baseline models for step-time prediction.

Provides two baselines:
- BlackboxBaseline: re-trained 3-coefficient linear regression matching BLIS blackbox model
- NaiveMeanBaseline: always predicts the training set mean step duration

Plus calibration and gating utilities for the StepML research workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from evaluation import (
    compute_mape,
    compute_mspe,
    compute_p99_error,
    compute_pearson_r,
)

# Feature columns used by the blackbox model
_FEATURE_COLS = ["batch.prefill_tokens", "batch.decode_tokens"]
# Target column
_TARGET_COL = "step.duration_us"


class BlackboxBaseline:
    """Re-trained 3-coefficient linear regression matching BLIS blackbox model.

    StepTime = beta0 + beta1 * batch.prefill_tokens + beta2 * batch.decode_tokens

    This mirrors sim/latency/latency.go:23-38 (BlackboxLatencyModel.StepTime),
    where batch.prefill_tokens corresponds to cacheMissTokens and
    batch.decode_tokens corresponds to decodeTokens.
    """

    def __init__(self) -> None:
        self._model: LinearRegression | None = None

    def fit(self, train_df: pd.DataFrame) -> BlackboxBaseline:
        """Train on step-level data. Uses sklearn LinearRegression."""
        X = train_df[_FEATURE_COLS].values
        y = train_df[_TARGET_COL].values
        self._model = LinearRegression()
        self._model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict step.duration_us from batch features."""
        if self._model is None:
            raise RuntimeError("BlackboxBaseline.predict() called before fit()")
        X = df[_FEATURE_COLS].values
        return self._model.predict(X)

    @property
    def coefficients(self) -> dict:
        """Return {"beta0": intercept, "beta1": prefill_coeff, "beta2": decode_coeff}."""
        if self._model is None:
            raise RuntimeError("BlackboxBaseline.coefficients accessed before fit()")
        return {
            "beta0": float(self._model.intercept_),
            "beta1": float(self._model.coef_[0]),
            "beta2": float(self._model.coef_[1]),
        }


class NaiveMeanBaseline:
    """Always predicts the training set mean step duration."""

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, train_df: pd.DataFrame) -> NaiveMeanBaseline:
        """Compute and store the training set mean of step.duration_us."""
        self._mean = float(train_df[_TARGET_COL].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return an array of the training mean, one per input row."""
        if self._mean is None:
            raise RuntimeError("NaiveMeanBaseline.predict() called before fit()")
        return np.full(len(df), self._mean)


def compute_baseline_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baselines: dict | None = None,
) -> dict:
    """Compute evaluation metrics for all baselines.

    Args:
        train_df: Training split DataFrame with feature and target columns.
        test_df: Test split DataFrame with feature and target columns.
        baselines: Optional dict of {name: baseline_instance}. If None, uses
            default baselines (blackbox + naive_mean).

    Returns:
        Dict like:
        {
            "blackbox": {"mape": 15.2, "mspe": -3.1, "pearson_r": 0.89, "p99_error": 45.3},
            "naive_mean": {"mape": 42.1, ...},
        }
        Uses evaluation.py functions (compute_mape, compute_mspe, compute_pearson_r, compute_p99_error).
    """
    if baselines is None:
        baselines = {
            "blackbox": BlackboxBaseline(),
            "naive_mean": NaiveMeanBaseline(),
        }

    actual = test_df[_TARGET_COL].values
    report: dict = {}

    for name, model in baselines.items():
        model.fit(train_df)
        predicted = model.predict(test_df)

        pearson_r = compute_pearson_r(predicted, actual)
        # pearsonr returns NaN for constant inputs (e.g., NaiveMeanBaseline).
        # Replace with 0.0 — constant predictions have zero useful correlation.
        if np.isnan(pearson_r):
            pearson_r = 0.0

        report[name] = {
            "mape": float(compute_mape(predicted, actual)),
            "mspe": float(compute_mspe(predicted, actual)),
            "pearson_r": float(pearson_r),
            "p99_error": float(compute_p99_error(predicted, actual)),
        }

    return report


def calibrate_short_circuit_threshold(blackbox_mape: float) -> float:
    """Determine the short-circuit threshold for StepML improvement.

    If blackbox MAPE > 25%, threshold = blackbox_MAPE + 10%.
    Otherwise threshold = 35% (25% + 10% buffer).

    Args:
        blackbox_mape: The blackbox baseline's MAPE as a percentage.

    Returns:
        The threshold as a percentage.
    """
    if blackbox_mape > 25.0:
        return blackbox_mape + 10.0
    return 35.0


def check_r4_gate(blackbox_e2e_mean_error: float) -> dict:
    """Check if blackbox is already good enough (R4 risk).

    If abs(blackbox_e2e_mean_error) < 12%, flag for research justification review.
    This means the blackbox model is already performing well at the E2E level,
    so additional ML complexity may not be justified.

    Args:
        blackbox_e2e_mean_error: The blackbox baseline's E2E mean error as a percentage.

    Returns:
        {"passed": bool, "e2e_error": float, "message": str}
        passed=True means no flag (research is justified).
        passed=False means flagged (blackbox already good enough, needs justification).
    """
    e2e_error = float(blackbox_e2e_mean_error)
    flagged = abs(e2e_error) < 12.0

    if flagged:
        message = (
            f"R4 gate FLAGGED: blackbox E2E mean error = {e2e_error:.1f}% "
            f"(|error| < 12%). Blackbox may already be sufficient. "
            f"Review research justification before proceeding."
        )
    else:
        message = (
            f"R4 gate passed: blackbox E2E mean error = {e2e_error:.1f}% "
            f"(|error| >= 12%). Step-level ML improvement is justified."
        )

    return {
        "passed": not flagged,
        "e2e_error": e2e_error,
        "message": message,
    }
