"""Tests for baseline models — uses synthetic data only, no real data files needed.

Tests follow TDD: written before baselines.py implementation.
"""

import numpy as np
import pandas as pd
import pytest

from baselines import (
    BlackboxBaseline,
    NaiveMeanBaseline,
    calibrate_short_circuit_threshold,
    check_r4_gate,
    compute_baseline_report,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data construction
# ---------------------------------------------------------------------------

def _make_linear_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with a known linear relationship.

    step.duration_us = 500 + 2.0 * batch.prefill_tokens + 3.0 * batch.decode_tokens + noise
    """
    rng = np.random.default_rng(seed)
    prefill = rng.integers(10, 500, size=n).astype(float)
    decode = rng.integers(1, 200, size=n).astype(float)
    noise = rng.normal(0, 5, size=n)
    duration = 500.0 + 2.0 * prefill + 3.0 * decode + noise
    return pd.DataFrame({
        "batch.prefill_tokens": prefill,
        "batch.decode_tokens": decode,
        "step.duration_us": duration,
    })


def _make_constant_df(n: int = 50, duration: float = 1000.0) -> pd.DataFrame:
    """Create a DataFrame where all durations are the same constant."""
    return pd.DataFrame({
        "batch.prefill_tokens": np.full(n, 100.0),
        "batch.decode_tokens": np.full(n, 50.0),
        "step.duration_us": np.full(n, duration),
    })


def _make_different_linear_df(n: int = 100, seed: int = 99) -> pd.DataFrame:
    """Create a DataFrame with a different linear relationship.

    step.duration_us = 200 + 10.0 * batch.prefill_tokens + 1.0 * batch.decode_tokens + noise
    """
    rng = np.random.default_rng(seed)
    prefill = rng.integers(10, 500, size=n).astype(float)
    decode = rng.integers(1, 200, size=n).astype(float)
    noise = rng.normal(0, 5, size=n)
    duration = 200.0 + 10.0 * prefill + 1.0 * decode + noise
    return pd.DataFrame({
        "batch.prefill_tokens": prefill,
        "batch.decode_tokens": decode,
        "step.duration_us": duration,
    })


# ---------------------------------------------------------------------------
# BlackboxBaseline tests
# ---------------------------------------------------------------------------

class TestBlackboxBaseline:
    """Tests for the BlackboxBaseline (3-coefficient linear regression)."""

    def test_produces_three_coefficients(self):
        """BC-0-1: BlackboxBaseline trains on 2 features and produces 3 coefficients
        (intercept + 2 slopes)."""
        df = _make_linear_df()
        model = BlackboxBaseline().fit(df)
        coeffs = model.coefficients

        assert "beta0" in coeffs, "Missing intercept coefficient beta0"
        assert "beta1" in coeffs, "Missing prefill coefficient beta1"
        assert "beta2" in coeffs, "Missing decode coefficient beta2"
        assert len(coeffs) == 3, f"Expected 3 coefficients, got {len(coeffs)}"

    def test_coefficients_approximate_known_relationship(self):
        """Coefficients should approximate the generating relationship.

        Data generated with: 500 + 2.0*prefill + 3.0*decode + noise(0,5)
        """
        df = _make_linear_df(n=1000, seed=42)
        model = BlackboxBaseline().fit(df)
        coeffs = model.coefficients

        assert coeffs["beta0"] == pytest.approx(500.0, abs=20.0)
        assert coeffs["beta1"] == pytest.approx(2.0, abs=0.2)
        assert coeffs["beta2"] == pytest.approx(3.0, abs=0.2)

    def test_coefficients_change_with_different_data(self):
        """BC-0-4: Coefficients are re-trained on training data, not hardcoded.

        Training on different data must produce different coefficients.
        """
        df_a = _make_linear_df(n=200, seed=42)
        df_b = _make_different_linear_df(n=200, seed=99)

        model_a = BlackboxBaseline().fit(df_a)
        model_b = BlackboxBaseline().fit(df_b)

        # At least the slopes should be clearly different
        assert model_a.coefficients["beta1"] != pytest.approx(
            model_b.coefficients["beta1"], abs=1.0
        ), "beta1 should differ between different training data"
        assert model_a.coefficients["beta0"] != pytest.approx(
            model_b.coefficients["beta0"], abs=50.0
        ), "beta0 should differ between different training data"

    def test_predict_returns_correct_length(self):
        """predict() returns an array with the same length as input."""
        train_df = _make_linear_df(n=100)
        test_df = _make_linear_df(n=37, seed=99)
        model = BlackboxBaseline().fit(train_df)

        preds = model.predict(test_df)
        assert len(preds) == len(test_df), (
            f"Expected {len(test_df)} predictions, got {len(preds)}"
        )

    def test_predict_positive_on_realistic_data(self):
        """Predictions should be positive on realistic (positive feature) data."""
        train_df = _make_linear_df(n=200)
        test_df = _make_linear_df(n=50, seed=99)
        model = BlackboxBaseline().fit(train_df)

        preds = model.predict(test_df)
        assert np.all(preds > 0), "All predictions should be positive on realistic data"

    def test_predict_returns_ndarray(self):
        """predict() should return a numpy ndarray."""
        df = _make_linear_df(n=50)
        model = BlackboxBaseline().fit(df)
        preds = model.predict(df)
        assert isinstance(preds, np.ndarray)

    def test_fit_returns_self(self):
        """fit() should return self for method chaining."""
        df = _make_linear_df(n=50)
        model = BlackboxBaseline()
        result = model.fit(df)
        assert result is model


# ---------------------------------------------------------------------------
# NaiveMeanBaseline tests
# ---------------------------------------------------------------------------

class TestNaiveMeanBaseline:
    """Tests for the NaiveMeanBaseline (always predicts training mean)."""

    def test_predicts_training_mean(self):
        """NaiveMeanBaseline returns the training set mean for all predictions."""
        df = _make_constant_df(n=50, duration=1000.0)
        model = NaiveMeanBaseline().fit(df)

        preds = model.predict(df)
        assert np.all(preds == pytest.approx(1000.0))

    def test_predicts_training_mean_varied_data(self):
        """Mean of step.duration_us in training data should be the prediction."""
        train_df = pd.DataFrame({
            "batch.prefill_tokens": [100.0, 200.0, 300.0],
            "batch.decode_tokens": [10.0, 20.0, 30.0],
            "step.duration_us": [100.0, 200.0, 300.0],
        })
        model = NaiveMeanBaseline().fit(train_df)
        expected_mean = 200.0  # (100 + 200 + 300) / 3

        test_df = pd.DataFrame({
            "batch.prefill_tokens": [50.0, 999.0],
            "batch.decode_tokens": [5.0, 99.0],
            "step.duration_us": [50.0, 999.0],
        })
        preds = model.predict(test_df)
        assert np.all(preds == pytest.approx(expected_mean))

    def test_predict_returns_correct_length(self):
        """predict() returns an array with the same length as input."""
        train_df = _make_linear_df(n=100)
        test_df = _make_linear_df(n=23, seed=77)
        model = NaiveMeanBaseline().fit(train_df)

        preds = model.predict(test_df)
        assert len(preds) == len(test_df)

    def test_predict_positive_on_realistic_data(self):
        """Predictions should be positive when training data has positive durations."""
        train_df = _make_linear_df(n=100)
        test_df = _make_linear_df(n=50, seed=99)
        model = NaiveMeanBaseline().fit(train_df)

        preds = model.predict(test_df)
        assert np.all(preds > 0)

    def test_predict_returns_ndarray(self):
        """predict() should return a numpy ndarray."""
        df = _make_linear_df(n=50)
        model = NaiveMeanBaseline().fit(df)
        preds = model.predict(df)
        assert isinstance(preds, np.ndarray)

    def test_fit_returns_self(self):
        """fit() should return self for method chaining."""
        df = _make_linear_df(n=50)
        model = NaiveMeanBaseline()
        result = model.fit(df)
        assert result is model


# ---------------------------------------------------------------------------
# compute_baseline_report tests
# ---------------------------------------------------------------------------

class TestComputeBaselineReport:
    """Tests for compute_baseline_report()."""

    def test_returns_dict_with_expected_keys(self):
        """Report should have entries for each baseline with standard metrics."""
        train_df = _make_linear_df(n=100, seed=42)
        test_df = _make_linear_df(n=50, seed=99)

        report = compute_baseline_report(train_df, test_df)

        assert "blackbox" in report
        assert "naive_mean" in report

        for name in ["blackbox", "naive_mean"]:
            entry = report[name]
            assert "mape" in entry, f"Missing 'mape' in {name}"
            assert "mspe" in entry, f"Missing 'mspe' in {name}"
            assert "pearson_r" in entry, f"Missing 'pearson_r' in {name}"
            assert "p99_error" in entry, f"Missing 'p99_error' in {name}"

    def test_blackbox_better_than_naive_on_linear_data(self):
        """On data with a clear linear relationship, blackbox should have lower MAPE."""
        train_df = _make_linear_df(n=500, seed=42)
        test_df = _make_linear_df(n=100, seed=99)

        report = compute_baseline_report(train_df, test_df)

        assert report["blackbox"]["mape"] < report["naive_mean"]["mape"], (
            "BlackboxBaseline should have lower MAPE than NaiveMeanBaseline on linear data"
        )

    def test_metrics_are_numeric(self):
        """All metric values should be finite floats."""
        train_df = _make_linear_df(n=100, seed=42)
        test_df = _make_linear_df(n=50, seed=99)

        report = compute_baseline_report(train_df, test_df)

        for name, entry in report.items():
            for metric, value in entry.items():
                assert isinstance(value, float), (
                    f"{name}.{metric} should be float, got {type(value)}"
                )
                assert np.isfinite(value), (
                    f"{name}.{metric} should be finite, got {value}"
                )

    def test_custom_baselines_parameter(self):
        """When baselines dict is provided, only those baselines appear in report."""
        train_df = _make_linear_df(n=100, seed=42)
        test_df = _make_linear_df(n=50, seed=99)

        custom_baselines = {"my_blackbox": BlackboxBaseline()}
        report = compute_baseline_report(train_df, test_df, baselines=custom_baselines)

        assert "my_blackbox" in report
        assert "naive_mean" not in report

    def test_pearson_r_high_for_blackbox_on_linear_data(self):
        """Blackbox should achieve high Pearson r on cleanly linear data."""
        train_df = _make_linear_df(n=500, seed=42)
        test_df = _make_linear_df(n=100, seed=99)

        report = compute_baseline_report(train_df, test_df)

        assert report["blackbox"]["pearson_r"] > 0.95, (
            "BlackboxBaseline should have high Pearson r on linear data"
        )


# ---------------------------------------------------------------------------
# calibrate_short_circuit_threshold tests
# ---------------------------------------------------------------------------

class TestCalibrateShortCircuitThreshold:
    """Tests for calibrate_short_circuit_threshold()."""

    def test_high_mape_returns_mape_plus_ten(self):
        """If blackbox MAPE > 25%, threshold = blackbox_MAPE + 10%."""
        threshold = calibrate_short_circuit_threshold(30.0)
        assert threshold == pytest.approx(40.0)

    def test_high_mape_boundary(self):
        """At exactly 25.1%, threshold = 35.1%."""
        threshold = calibrate_short_circuit_threshold(25.1)
        assert threshold == pytest.approx(35.1)

    def test_low_mape_returns_default_35(self):
        """If blackbox MAPE <= 25%, threshold = 35% (25% + 10% buffer)."""
        threshold = calibrate_short_circuit_threshold(15.0)
        assert threshold == pytest.approx(35.0)

    def test_boundary_at_25(self):
        """At exactly MAPE = 25%, threshold = 35% (not exceeded)."""
        threshold = calibrate_short_circuit_threshold(25.0)
        assert threshold == pytest.approx(35.0)

    def test_very_high_mape(self):
        """Very high MAPE still returns MAPE + 10."""
        threshold = calibrate_short_circuit_threshold(80.0)
        assert threshold == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# check_r4_gate tests
# ---------------------------------------------------------------------------

class TestCheckR4Gate:
    """Tests for check_r4_gate()."""

    def test_good_blackbox_flags_review(self):
        """If |e2e_error| < 12%, the gate flags for review (passed=False means flagged)."""
        result = check_r4_gate(5.0)
        assert result["passed"] is False, "Should flag when blackbox is already good"
        assert "e2e_error" in result
        assert "message" in result
        assert result["e2e_error"] == pytest.approx(5.0)

    def test_negative_error_within_threshold(self):
        """Negative error with |error| < 12% should also flag."""
        result = check_r4_gate(-8.0)
        assert result["passed"] is False

    def test_poor_blackbox_passes(self):
        """If |e2e_error| >= 12%, the gate passes (no flag needed)."""
        result = check_r4_gate(15.0)
        assert result["passed"] is True

    def test_boundary_at_12(self):
        """At exactly 12%, gate passes (not flagged)."""
        result = check_r4_gate(12.0)
        assert result["passed"] is True

    def test_negative_boundary(self):
        """At exactly -12%, gate passes (not flagged)."""
        result = check_r4_gate(-12.0)
        assert result["passed"] is True

    def test_returns_dict_with_required_keys(self):
        """Result always contains passed, e2e_error, and message keys."""
        result = check_r4_gate(20.0)
        assert "passed" in result
        assert "e2e_error" in result
        assert "message" in result
        assert isinstance(result["passed"], bool)
        assert isinstance(result["e2e_error"], float)
        assert isinstance(result["message"], str)
