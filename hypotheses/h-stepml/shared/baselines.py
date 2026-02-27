"""Baseline models for step-time prediction.

Provides two baselines:
- BlackboxBaseline: re-trained 3-coefficient linear regression matching BLIS blackbox model
- NaiveMeanBaseline: always predicts the training set mean step duration

Plus calibration and gating utilities for the StepML research workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

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
    """Re-trained 3-coefficient non-negative linear model matching BLIS blackbox.

    StepTime = beta0 + beta1 * batch.prefill_tokens + beta2 * batch.decode_tokens

    All three coefficients are constrained to be >= 0 via NNLS (non-negative
    least squares). The intercept is modelled as a third feature (ones column)
    so that the non-negativity constraint applies uniformly to all betas.

    Column normalization is applied before NNLS to avoid scale mismatch between
    the ones column (~1) and token columns (~0-2048), then coefficients are
    rescaled to original units.
    """

    def __init__(self) -> None:
        self._coeffs: np.ndarray | None = None  # [beta0, beta1, beta2]

    def fit(self, train_df: pd.DataFrame) -> BlackboxBaseline:
        """Train on step-level data using column-normalized NNLS.

        Constructs A = [ones, prefill_tokens, decode_tokens], normalizes each
        column by its L2 norm, solves NNLS, then rescales coefficients back.
        """
        X = train_df[_FEATURE_COLS].to_numpy(dtype=np.float64)
        y = train_df[_TARGET_COL].to_numpy(dtype=np.float64)

        # Build design matrix: [ones, prefill, decode]
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        A = np.hstack([ones, X])

        # Column-normalize to fix conditioning
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms == 0] = 1.0  # guard against zero columns
        A_normed = A / col_norms

        # Solve NNLS on normalized problem
        x_normed, _ = nnls(A_normed, y)

        # Rescale coefficients back to original units
        self._coeffs = x_normed / col_norms
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict step.duration_us from batch features."""
        if self._coeffs is None:
            raise RuntimeError("BlackboxBaseline.predict() called before fit()")
        X = df[_FEATURE_COLS].to_numpy(dtype=np.float64)
        return self._coeffs[0] + X @ self._coeffs[1:]

    @property
    def coefficients(self) -> dict:
        """Return {"beta0": intercept, "beta1": prefill_coeff, "beta2": decode_coeff}."""
        if self._coeffs is None:
            raise RuntimeError("BlackboxBaseline.coefficients accessed before fit()")
        return {
            "beta0": float(self._coeffs[0]),
            "beta1": float(self._coeffs[1]),
            "beta2": float(self._coeffs[2]),
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
