"""Tests for the evaluation harness — pure computation on arrays, no real data needed."""

import numpy as np
import pytest
from scipy.stats import pearsonr

from evaluation import (
    bootstrap_ci,
    compute_e2e_mean_error,
    compute_mape,
    compute_mspe,
    compute_p99_error,
    compute_pearson_r,
)


# ---------------------------------------------------------------------------
# MAPE tests
# ---------------------------------------------------------------------------

def test_mape_perfect_prediction():
    """MAPE of identical arrays is 0."""
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([100.0, 200.0, 300.0])
    assert compute_mape(predicted, actual) == pytest.approx(0.0)


def test_mape_known_value():
    """predicted=[110, 90], actual=[100, 100] -> MAPE = 10%."""
    predicted = np.array([110.0, 90.0])
    actual = np.array([100.0, 100.0])
    assert compute_mape(predicted, actual) == pytest.approx(10.0)


def test_mape_ignores_zero_actual():
    """Rows where actual=0 are excluded to avoid division by zero."""
    predicted = np.array([110.0, 50.0, 90.0])
    actual = np.array([100.0, 0.0, 100.0])
    # Only rows 0 and 2 count: |10/100| + |10/100| / 2 * 100 = 10%
    assert compute_mape(predicted, actual) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# MSPE tests
# ---------------------------------------------------------------------------

def test_mspe_positive_bias():
    """predicted=[120, 110], actual=[100, 100] -> MSPE = +15% (overestimation)."""
    predicted = np.array([120.0, 110.0])
    actual = np.array([100.0, 100.0])
    # (20/100 + 10/100) / 2 * 100 = 15%
    assert compute_mspe(predicted, actual) == pytest.approx(15.0)


def test_mspe_negative_bias():
    """predicted=[80, 90], actual=[100, 100] -> MSPE = -15% (underestimation)."""
    predicted = np.array([80.0, 90.0])
    actual = np.array([100.0, 100.0])
    # (-20/100 + -10/100) / 2 * 100 = -15%
    assert compute_mspe(predicted, actual) == pytest.approx(-15.0)


def test_mspe_unbiased():
    """predicted=[110, 90], actual=[100, 100] -> MSPE = 0% (errors cancel)."""
    predicted = np.array([110.0, 90.0])
    actual = np.array([100.0, 100.0])
    # (10/100 + -10/100) / 2 * 100 = 0%
    assert compute_mspe(predicted, actual) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pearson r tests
# ---------------------------------------------------------------------------

def test_pearson_r_perfect():
    """Identical arrays -> r = 1.0."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_pearson_r(arr, arr) == pytest.approx(1.0)


def test_pearson_r_known():
    """Verify against scipy.stats.pearsonr for a known pair."""
    predicted = np.array([1.0, 2.5, 3.0, 4.5, 5.0])
    actual = np.array([1.1, 2.3, 2.9, 4.6, 5.2])
    expected_r, _ = pearsonr(predicted, actual)
    assert compute_pearson_r(predicted, actual) == pytest.approx(expected_r, abs=1e-10)


# ---------------------------------------------------------------------------
# P99 error test
# ---------------------------------------------------------------------------

def test_p99_error():
    """Verify 99th percentile of absolute percentage errors."""
    # 100 points: 99 with 5% error, 1 with 50% error
    actual = np.full(100, 100.0)
    predicted = np.full(100, 105.0)  # 5% error everywhere
    predicted[0] = 150.0  # one outlier at 50% error

    p99 = compute_p99_error(predicted, actual)
    # 99th percentile should be near 50% (the outlier)
    # With 100 points, np.percentile(sorted_errors, 99) interpolates between
    # the 99th and 100th values. 99 values at 5%, 1 value at 50%.
    # The 99th percentile (0-indexed position 99*0.99 = 98.01) should be
    # between 5% and 50%.
    assert p99 > 5.0
    assert p99 <= 50.0


# ---------------------------------------------------------------------------
# Bootstrap CI test
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_true():
    """CI from bootstrap_ci should contain the point estimate.

    This is a statistical test — it may rarely fail due to randomness, but
    with a fixed seed and reasonable data it should be stable.
    """
    predicted = np.array([110.0, 90.0, 105.0, 95.0, 100.0])
    actual = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

    point_estimate = compute_mape(predicted, actual)
    lo, hi = bootstrap_ci(compute_mape, predicted, actual, n_resamples=2000, seed=42)

    assert lo <= point_estimate <= hi
    # CI should have non-zero width
    assert hi > lo


# ---------------------------------------------------------------------------
# E2E mean error tests
# ---------------------------------------------------------------------------

def test_e2e_mean_error_perfect():
    """If all predicted step times match actual E2E, error = 0."""
    # Request "a": 3 steps of 10ms each = 30ms total, actual E2E = 30ms
    # Request "b": 2 steps of 20ms each = 40ms total, actual E2E = 40ms
    predicted_step_times = {
        "a": [10.0, 10.0, 10.0],
        "b": [20.0, 20.0],
    }
    actual_e2e_times = {
        "a": 30.0,
        "b": 40.0,
    }
    assert compute_e2e_mean_error(predicted_step_times, actual_e2e_times) == pytest.approx(0.0)


def test_e2e_mean_error_known():
    """Manually constructed example with known E2E mean error.

    Request "a": predicted steps sum to 33ms, actual E2E = 30ms
    Request "b": predicted steps sum to 44ms, actual E2E = 40ms

    Mean predicted E2E = (33 + 44) / 2 = 38.5
    Mean actual E2E    = (30 + 40) / 2 = 35.0
    Error = (38.5 - 35.0) / 35.0 * 100 = 10.0%
    """
    predicted_step_times = {
        "a": [11.0, 11.0, 11.0],
        "b": [22.0, 22.0],
    }
    actual_e2e_times = {
        "a": 30.0,
        "b": 40.0,
    }
    assert compute_e2e_mean_error(predicted_step_times, actual_e2e_times) == pytest.approx(10.0)
