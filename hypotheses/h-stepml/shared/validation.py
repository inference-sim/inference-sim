"""Validation functions for StepML WP0 infrastructure.

Part A: Temporal split effectiveness — confirms that temporal splitting
prevents autocorrelation leakage by comparing random vs temporal split MAPE.

Part B: ProgressIndex as KV cache length proxy (R1 gate) — validates that
total tokens (input + output) correlate with end-to-end request time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from evaluation import compute_mape, compute_pearson_r
from splits import temporal_split


def temporal_vs_random_split_comparison(
    all_df: pd.DataFrame,
    seed: int = 42,
) -> dict:
    """Compare random vs temporal split to validate temporal split prevents leakage.

    Trains a Ridge regression on batch features (batch.prefill_tokens,
    batch.decode_tokens) to predict step.duration_us.  Evaluates on
    test splits from both a random 60/20/20 split and the canonical
    temporal split.

    If random split MAPE is much lower (gap > 5%), it confirms that temporal
    splitting correctly prevents autocorrelation leakage: adjacent steps are
    correlated, so random splitting "cheats" by mixing correlated steps
    across train and test.

    Args:
        all_df: DataFrame with columns experiment_id, step.id,
            batch.prefill_tokens, batch.decode_tokens, step.duration_us.
        seed: Random seed for the random split and Ridge model.

    Returns:
        Dict with keys:
        - temporal_mape: MAPE (%) on the temporal test split
        - random_mape: MAPE (%) on the random test split
        - gap: temporal_mape - random_mape (positive means temporal is harder)
        - conclusion: Human-readable interpretation of the result
    """
    feature_cols = ["batch.prefill_tokens", "batch.decode_tokens"]
    target_col = "step.duration_us"

    X = all_df[feature_cols].values.astype(np.float64)
    y = all_df[target_col].values.astype(np.float64)

    # --- Temporal split ---
    temporal_splits = temporal_split(all_df, seed=seed)
    train_idx_t = temporal_splits["train"]
    test_idx_t = temporal_splits["test"]

    model_t = Ridge(alpha=1.0)
    model_t.fit(X[train_idx_t], y[train_idx_t])
    pred_t = model_t.predict(X[test_idx_t])
    temporal_mape = compute_mape(pred_t, y[test_idx_t])

    # --- Random split (60/20/20 shuffled) ---
    rng = np.random.default_rng(seed)
    n = len(all_df)
    perm = rng.permutation(n)
    n_train = int(round(n * 0.6))
    n_valid = int(round(n * 0.2))
    train_idx_r = perm[:n_train]
    # valid_idx_r = perm[n_train : n_train + n_valid]  # not used
    test_idx_r = perm[n_train + n_valid :]

    model_r = Ridge(alpha=1.0)
    model_r.fit(X[train_idx_r], y[train_idx_r])
    pred_r = model_r.predict(X[test_idx_r])
    random_mape = compute_mape(pred_r, y[test_idx_r])

    gap = temporal_mape - random_mape

    if gap > 5.0:
        conclusion = (
            f"Temporal split MAPE ({temporal_mape:.1f}%) is {gap:.1f}pp higher "
            f"than random split MAPE ({random_mape:.1f}%). "
            f"This confirms temporal splitting prevents autocorrelation leakage. "
            f"Random splitting allows correlated adjacent steps to leak between "
            f"train and test sets, artificially lowering error."
        )
    elif gap > 0:
        conclusion = (
            f"Temporal split MAPE ({temporal_mape:.1f}%) is {gap:.1f}pp higher "
            f"than random split MAPE ({random_mape:.1f}%). "
            f"Small gap suggests mild autocorrelation in the data."
        )
    else:
        conclusion = (
            f"Temporal split MAPE ({temporal_mape:.1f}%) is not higher "
            f"than random split MAPE ({random_mape:.1f}%). "
            f"No evidence of autocorrelation leakage in this dataset."
        )

    return {
        "temporal_mape": float(temporal_mape),
        "random_mape": float(random_mape),
        "gap": float(gap),
        "conclusion": conclusion,
    }


def validate_progress_index(
    lifecycle_df: pd.DataFrame,
) -> dict:
    """Validate ProgressIndex as KV cache length proxy (R1 gate).

    ProgressIndex is defined as input_tokens + output_tokens (the total
    tokens processed by a request).  This function validates that total
    tokens correlate with end-to-end request time, confirming the proxy
    is informative for KV cache modeling.

    The R1 gate passes if Pearson correlation > 0.9.  If correlation < 0.9,
    an informational flag is raised but the gate does not abort.

    Args:
        lifecycle_df: DataFrame indexed by request_id with columns:
            start_time, end_time, input_tokens, output_tokens,
            output_token_times.

    Returns:
        Dict with keys:
        - correlation: Pearson r between total_tokens and E2E time
        - n_requests: Number of requests analyzed
        - passed: True if correlation >= 0.9
        - conclusion: Human-readable interpretation
    """
    n_requests = len(lifecycle_df)

    total_tokens = (
        lifecycle_df["input_tokens"].values.astype(np.float64)
        + lifecycle_df["output_tokens"].values.astype(np.float64)
    )
    e2e_times = (
        lifecycle_df["end_time"].values.astype(np.float64)
        - lifecycle_df["start_time"].values.astype(np.float64)
    )

    correlation = compute_pearson_r(total_tokens, e2e_times)
    passed = correlation >= 0.9

    if passed:
        conclusion = (
            f"ProgressIndex proxy validated: Pearson r = {correlation:.3f} "
            f"(n={n_requests}). Total tokens (input + output) correlate "
            f"strongly with E2E time, confirming ProgressIndex is an "
            f"informative KV cache length proxy."
        )
    else:
        conclusion = (
            f"R1 flag: ProgressIndex correlation is {correlation:.3f} "
            f"(n={n_requests}), below the 0.9 threshold. "
            f"Total tokens are a weak proxy for E2E time in this dataset. "
            f"This is an informational flag — does not abort."
        )

    return {
        "correlation": float(correlation),
        "n_requests": n_requests,
        "passed": passed,
        "conclusion": conclusion,
    }
