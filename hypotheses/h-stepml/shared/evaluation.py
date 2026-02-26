"""Evaluation harness for step-time prediction models.

Pure computation on arrays — no dependency on real data or external services.
All percentage values are returned as floats where 10.0 means 10%.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def compute_mape(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Excludes rows where actual == 0 to avoid division by zero.
    Returns percentage as a float (e.g., 10.0 for 10%).
    """
    mask = actual != 0
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    if len(actual_filtered) == 0:
        return 0.0
    abs_pct_errors = np.abs((predicted_filtered - actual_filtered) / actual_filtered)
    return float(np.mean(abs_pct_errors) * 100.0)


def compute_mspe(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean Signed Percentage Error.

    Positive values indicate overestimation; negative values indicate
    underestimation. Excludes rows where actual == 0.
    Returns percentage as a float (e.g., 15.0 for +15%).
    """
    mask = actual != 0
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    if len(actual_filtered) == 0:
        return 0.0
    signed_pct_errors = (predicted_filtered - actual_filtered) / actual_filtered
    return float(np.mean(signed_pct_errors) * 100.0)


def compute_pearson_r(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Pearson correlation coefficient between predicted and actual."""
    r, _ = pearsonr(predicted, actual)
    return float(r)


def compute_p99_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    """99th percentile of absolute percentage errors.

    Excludes rows where actual == 0.
    Returns percentage as a float.
    """
    mask = actual != 0
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    if len(actual_filtered) == 0:
        return 0.0
    abs_pct_errors = np.abs((predicted_filtered - actual_filtered) / actual_filtered) * 100.0
    return float(np.percentile(abs_pct_errors, 99))


def bootstrap_ci(
    metric_fn,
    predicted: np.ndarray,
    actual: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for any metric function.

    Args:
        metric_fn: A function with signature (predicted, actual) -> float.
        predicted: Predicted values array.
        actual: Actual values array.
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    rng = np.random.default_rng(seed)
    n = len(predicted)
    bootstrap_estimates = np.empty(n_resamples)

    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        bootstrap_estimates[i] = metric_fn(predicted[indices], actual[indices])

    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_e2e_mean_error(
    predicted_step_times: dict,
    actual_e2e_times: dict,
) -> float:
    """Workload-level end-to-end mean error.

    For each request, sums predicted step times to get predicted E2E.
    Computes mean predicted E2E across requests and compares to the mean
    of actual E2E times.

    Args:
        predicted_step_times: {request_id: [step_time1, step_time2, ...]}
        actual_e2e_times: {request_id: float}

    Returns:
        Percentage error as a float (e.g. 10.0 for 10%).
    """
    request_ids = sorted(set(predicted_step_times.keys()) & set(actual_e2e_times.keys()))
    if not request_ids:
        return 0.0

    predicted_e2e = np.array([sum(predicted_step_times[rid]) for rid in request_ids])
    actual_e2e = np.array([actual_e2e_times[rid] for rid in request_ids])

    mean_predicted = np.mean(predicted_e2e)
    mean_actual = np.mean(actual_e2e)

    if mean_actual == 0:
        return 0.0

    return float((mean_predicted - mean_actual) / mean_actual * 100.0)
