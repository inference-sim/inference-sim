"""Component-level error attribution for E2E latency prediction.

Analyzes which of the 5 LatencyModel methods contributes most to E2E
prediction error.  This informs research ideation by identifying where
to focus improvement efforts.

E2E(request) = QueueingTime(req)              [1] arrival-to-queue delay
             + (time waiting in WaitQ)         [2] emergent from simulation dynamics
             + SchedulingProcessingTime()      [3] per-request scheduling overhead
             + sum StepTime(batch)             [4] GPU execution per batch step
             + sum OutputTokenProcessingTime() [5] per-token post-processing
             + KV transfer latency             [6] CPU<->GPU offload/reload
             + PreemptionProcessingTime()      [7] if preempted and re-queued

The 5 LatencyModel methods and their blackbox implementations:
  1. StepTime:                     beta0 + beta1*prefill + beta2*decode
  2. QueueingTime:                 alpha0 + alpha1*input_len
  3. OutputTokenProcessingTime:    constant alpha2 (per output token)
  4. SchedulingProcessingTime:     returns 0
  5. PreemptionProcessingTime:     returns 0

All percentage values are returned as floats where 10.0 means 10%.
All time values are in microseconds unless otherwise noted.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from evaluation import compute_mape, compute_mspe, compute_p99_error, compute_pearson_r
from data_loader import DEFAULT_DATA_ROOT, load_all_experiments, load_lifecycle_data


def compute_blackbox_step_predictions(
    steps_df: pd.DataFrame,
    beta_coeffs: list[float],
) -> pd.Series:
    """Predict step time using blackbox model: beta0 + beta1*prefill + beta2*decode.

    Args:
        steps_df: DataFrame with batch.prefill_tokens and batch.decode_tokens columns.
        beta_coeffs: [beta0, beta1, beta2].

    Returns:
        Series of predicted step times (microseconds).
    """
    if len(beta_coeffs) != 3:
        raise ValueError(f"Expected 3 beta coefficients, got {len(beta_coeffs)}")

    beta0, beta1, beta2 = beta_coeffs
    prefill = steps_df["batch.prefill_tokens"].astype(np.float64)
    decode = steps_df["batch.decode_tokens"].astype(np.float64)
    predicted = beta0 + beta1 * prefill + beta2 * decode
    return predicted


def compute_step_time_error(
    steps_df: pd.DataFrame,
    beta_coeffs: list[float],
) -> dict:
    """Compute step time prediction error metrics.

    Args:
        steps_df: DataFrame with batch.prefill_tokens, batch.decode_tokens,
            and step.duration_us columns.
        beta_coeffs: [beta0, beta1, beta2].

    Returns:
        Dict with keys: mape, mspe (signed), pearson_r, p99_error,
        mean_predicted, mean_actual, total_predicted, total_actual.
    """
    predicted = compute_blackbox_step_predictions(steps_df, beta_coeffs).values
    actual = steps_df["step.duration_us"].astype(np.float64).values

    # pearsonr requires at least 2 data points
    if len(predicted) < 2:
        pearson_r = 0.0
    else:
        pearson_r = compute_pearson_r(predicted, actual)
        if np.isnan(pearson_r):
            pearson_r = 0.0

    return {
        "mape": float(compute_mape(predicted, actual)),
        "mspe": float(compute_mspe(predicted, actual)),
        "pearson_r": float(pearson_r),
        "p99_error": float(compute_p99_error(predicted, actual)),
        "mean_predicted": float(np.mean(predicted)),
        "mean_actual": float(np.mean(actual)),
        "total_predicted": float(np.sum(predicted)),
        "total_actual": float(np.sum(actual)),
    }


def estimate_queueing_contribution(
    lifecycle_df: pd.DataFrame,
    alpha_coeffs: list[float],
) -> dict:
    """Estimate queueing time contribution.

    For each request: queueing = alpha0 + alpha1 * input_tokens.
    Compares total predicted queueing time against total E2E time.

    Args:
        lifecycle_df: DataFrame indexed by request_id with columns:
            start_time, end_time, input_tokens.
        alpha_coeffs: [alpha0, alpha1, alpha2] (alpha2 unused here).

    Returns:
        Dict with keys: mean_queueing_us, total_queueing_us, fraction_of_e2e.
        fraction_of_e2e = total_queueing / total_e2e where
        e2e = end_time - start_time.
    """
    alpha0 = alpha_coeffs[0]
    alpha1 = alpha_coeffs[1]

    input_tokens = lifecycle_df["input_tokens"].astype(np.float64).values
    queueing = alpha0 + alpha1 * input_tokens

    e2e = (
        lifecycle_df["end_time"].astype(np.float64).values
        - lifecycle_df["start_time"].astype(np.float64).values
    )
    total_e2e = float(np.sum(e2e))
    total_queueing = float(np.sum(queueing))

    if total_e2e == 0:
        fraction = 0.0
    else:
        fraction = total_queueing / total_e2e

    return {
        "mean_queueing_us": float(np.mean(queueing)),
        "total_queueing_us": total_queueing,
        "fraction_of_e2e": fraction,
    }


def estimate_output_processing_contribution(
    lifecycle_df: pd.DataFrame,
    alpha2: float,
) -> dict:
    """Estimate output token processing contribution.

    For each request: output_processing = alpha2 * output_tokens.

    Args:
        lifecycle_df: DataFrame indexed by request_id with columns:
            start_time, end_time, output_tokens.
        alpha2: Per-token processing time (microseconds).

    Returns:
        Dict with keys: mean_output_us, total_output_us, fraction_of_e2e.
    """
    output_tokens = lifecycle_df["output_tokens"].astype(np.float64).values
    output_processing = alpha2 * output_tokens

    e2e = (
        lifecycle_df["end_time"].astype(np.float64).values
        - lifecycle_df["start_time"].astype(np.float64).values
    )
    total_e2e = float(np.sum(e2e))
    total_output = float(np.sum(output_processing))

    if total_e2e == 0:
        fraction = 0.0
    else:
        fraction = total_output / total_e2e

    return {
        "mean_output_us": float(np.mean(output_processing)),
        "total_output_us": total_output,
        "fraction_of_e2e": fraction,
    }


def estimate_step_time_contribution(
    steps_df: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
) -> dict:
    """Estimate step time's share of total E2E time from ground truth.

    Sums all step.duration_us and compares against total E2E from lifecycle.

    Args:
        steps_df: DataFrame with step.duration_us column.
        lifecycle_df: DataFrame indexed by request_id with start_time, end_time.

    Returns:
        Dict with keys: total_step_us, total_e2e_us, fraction_of_e2e,
        mean_steps_per_request (total steps / number of requests).
    """
    total_step = float(steps_df["step.duration_us"].astype(np.float64).sum())

    e2e = (
        lifecycle_df["end_time"].astype(np.float64).values
        - lifecycle_df["start_time"].astype(np.float64).values
    )
    total_e2e = float(np.sum(e2e))
    n_requests = len(lifecycle_df)
    n_steps = len(steps_df)

    if total_e2e == 0:
        fraction = 0.0
    else:
        fraction = total_step / total_e2e

    if n_requests == 0:
        mean_steps_per_request = 0.0
    else:
        mean_steps_per_request = n_steps / n_requests

    return {
        "total_step_us": total_step,
        "total_e2e_us": total_e2e,
        "fraction_of_e2e": fraction,
        "mean_steps_per_request": mean_steps_per_request,
    }


def component_error_attribution(
    steps_df: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    alpha_coeffs: list[float],
    beta_coeffs: list[float],
) -> dict:
    """Full component-level error attribution report.

    Combines step-time error metrics, queueing contribution, output
    processing contribution, and step-time contribution fractions into
    a unified report that identifies where E2E prediction error originates.

    Args:
        steps_df: DataFrame with batch.prefill_tokens, batch.decode_tokens,
            step.duration_us.
        lifecycle_df: DataFrame indexed by request_id with start_time,
            end_time, input_tokens, output_tokens.
        alpha_coeffs: [alpha0, alpha1, alpha2].
        beta_coeffs: [beta0, beta1, beta2].

    Returns:
        Dict with keys:
        - step_time: step-level error metrics + contribution fraction
        - queueing_time: predicted contribution + fraction of E2E
        - output_processing: predicted contribution + fraction of E2E
        - scheduling: current status (returns 0)
        - preemption: current status (returns 0)
        - summary: which component dominates E2E error
        - recommendation: where to focus improvement
    """
    # Step time error metrics
    step_error = compute_step_time_error(steps_df, beta_coeffs)

    # Step time contribution to E2E
    step_contrib = estimate_step_time_contribution(steps_df, lifecycle_df)

    # Queueing time contribution
    queueing = estimate_queueing_contribution(lifecycle_df, alpha_coeffs)

    # Output processing contribution
    alpha2 = alpha_coeffs[2] if len(alpha_coeffs) >= 3 else 0.0
    output = estimate_output_processing_contribution(lifecycle_df, alpha2)

    # Identify dominant error source
    step_frac = step_contrib["fraction_of_e2e"]
    queue_frac = queueing["fraction_of_e2e"]
    output_frac = output["fraction_of_e2e"]

    fractions = {
        "step_time": step_frac,
        "queueing_time": queue_frac,
        "output_processing": output_frac,
    }
    dominant_component = max(fractions, key=fractions.get)

    summary = (
        f"Step time accounts for {step_frac:.1%} of total E2E, "
        f"with MAPE={step_error['mape']:.1f}% and MSPE={step_error['mspe']:.1f}%. "
        f"Queueing accounts for {queue_frac:.1%}, "
        f"output processing for {output_frac:.1%}. "
        f"Dominant component: {dominant_component}."
    )

    if dominant_component == "step_time" and step_error["mape"] > 10.0:
        recommendation = (
            "Step time dominates E2E and has significant prediction error "
            f"(MAPE={step_error['mape']:.1f}%). Focus on improving StepTime "
            "prediction (beta coefficients or ML replacement)."
        )
    elif dominant_component == "queueing_time":
        recommendation = (
            "Queueing time dominates E2E. Focus on improving QueueingTime "
            "prediction (alpha0/alpha1 coefficients) or reducing actual queueing."
        )
    elif dominant_component == "output_processing":
        recommendation = (
            "Output processing dominates E2E. Focus on improving "
            "OutputTokenProcessingTime (alpha2 coefficient)."
        )
    else:
        recommendation = (
            f"Step time dominates E2E with low error (MAPE={step_error['mape']:.1f}%). "
            "Consider whether other components (scheduling, preemption) are "
            "under-modeled."
        )

    return {
        "step_time": {
            **step_error,
            "fraction_of_e2e": step_frac,
            "mean_steps_per_request": step_contrib["mean_steps_per_request"],
        },
        "queueing_time": queueing,
        "output_processing": output,
        "scheduling": {
            "current_value": 0.0,
            "note": (
                "SchedulingProcessingTime currently returns 0. "
                "If real scheduling overhead is non-trivial, this is an "
                "unmodeled error source."
            ),
        },
        "preemption": {
            "current_value": 0.0,
            "note": (
                "PreemptionProcessingTime currently returns 0. "
                "If preemptions are frequent, this is an unmodeled error source."
            ),
        },
        "summary": summary,
        "recommendation": recommendation,
    }


def component_error_attribution_all_experiments(
    data_root: str = None,
    coefficients: dict = None,
) -> pd.DataFrame:
    """Run error attribution across all experiments.

    Iterates over experiment directories under data_root, loads step-level
    and lifecycle data for each, and computes per-experiment error
    attribution using the provided (or default zero) coefficients.

    Args:
        data_root: Path to ground truth data directory. Defaults to
            eval/ground_truth/ relative to this file.
        coefficients: Optional dict mapping experiment_id to
            {"alpha": [a0, a1, a2], "beta": [b0, b1, b2]}.
            If None or if an experiment_id is missing, uses zeros.

    Returns:
        DataFrame with one row per experiment, columns for each component's
        error contribution and fraction.
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    if coefficients is None:
        coefficients = {}

    default_alpha = [0.0, 0.0, 0.0]
    default_beta = [0.0, 0.0, 0.0]

    rows = []
    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        traces_path = os.path.join(dirpath, "traces.json")
        lifecycle_path = os.path.join(
            dirpath, "results", "per_request_lifecycle_metrics.json"
        )
        if not os.path.isfile(traces_path) or not os.path.isfile(lifecycle_path):
            continue

        from data_loader import load_experiment_steps, load_lifecycle_data as _load_lc

        steps_df = load_experiment_steps(dirpath)
        lifecycle_df = _load_lc(dirpath)

        if steps_df.empty or lifecycle_df.empty:
            continue

        exp_coeffs = coefficients.get(dirname, {})
        alpha = exp_coeffs.get("alpha", default_alpha)
        beta = exp_coeffs.get("beta", default_beta)

        report = component_error_attribution(steps_df, lifecycle_df, alpha, beta)

        rows.append({
            "experiment_id": dirname,
            "step_mape": report["step_time"]["mape"],
            "step_mspe": report["step_time"]["mspe"],
            "step_pearson_r": report["step_time"]["pearson_r"],
            "step_p99_error": report["step_time"]["p99_error"],
            "step_fraction_of_e2e": report["step_time"]["fraction_of_e2e"],
            "queueing_fraction_of_e2e": report["queueing_time"]["fraction_of_e2e"],
            "output_fraction_of_e2e": report["output_processing"]["fraction_of_e2e"],
            "mean_steps_per_request": report["step_time"]["mean_steps_per_request"],
            "recommendation": report["recommendation"],
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
