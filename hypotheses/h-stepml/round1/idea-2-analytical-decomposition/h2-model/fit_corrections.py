"""Fit learned correction factors on analytical decomposition.

Three model variants:
  - 9-parameter global: 4 component factors + 1 overhead + 4 MFU discounts
  - 16-parameter per-phase: separate factors for compute/memory-bound regimes
  - 36-parameter per-model: 9 parameters per model family (4 models x 9)

Uses h1's analytical FLOPs components as the backbone.
Fitting via scipy.optimize.least_squares with log-ratio objective.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from splits import temporal_split
from evaluation import compute_mape

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H1_DIR = os.path.join(SCRIPT_DIR, "..", "h1-features")
TARGET_COL = "step.duration_us"

# Model name → integer index for per-model parameters
MODEL_INDEX = {
    "codellama-34b": 0,
    "llama-2-7b": 1,
    "llama-2-70b": 2,
    "llama-2-70b-hf": 2,
    "mixtral-8x7b-v0-1": 3,
}


def predict_9param(params, components):
    """9-parameter global model.

    params: [pf_gemm_factor, pf_attn_factor, dc_gemm_factor, dc_attn_factor,
             overhead_us, mfu_pf_gemm, mfu_pf_attn, mfu_dc_gemm, mfu_dc_attn]

    Since decode_attn is 0 (no per-request KV), the dc_attn parameters are inert.
    The overhead term absorbs decode attention and other unmodeled effects.
    """
    pf_gf, pf_af, dc_gf, dc_af, overhead = params[:5]
    mfu_pg, mfu_pa, mfu_dg, mfu_da = params[5:9]

    # Apply correction factors and MFU discounts
    # MFU discount: divide analytical time by MFU (analytical assumes peak throughput)
    predicted = (
        pf_gf * components["prefill_gemm_us"] / max(mfu_pg, 0.01)
        + pf_af * components["prefill_attn_us"] / max(mfu_pa, 0.01)
        + dc_gf * components["decode_gemm_us"] / max(mfu_dg, 0.01)
        + dc_af * components["decode_attn_us"] / max(mfu_da, 0.01)
        + overhead
    )
    return np.maximum(predicted, 1.0)  # floor at 1 us


def predict_9param_vec(params, pf_gemm, pf_attn, dc_gemm, dc_attn):
    """Vectorized 9-parameter prediction."""
    pf_gf, pf_af, dc_gf, dc_af, overhead = params[:5]
    mfu_pg, mfu_pa, mfu_dg, mfu_da = params[5:9]

    predicted = (
        pf_gf * pf_gemm / max(mfu_pg, 0.01)
        + pf_af * pf_attn / max(mfu_pa, 0.01)
        + dc_gf * dc_gemm / max(mfu_dg, 0.01)
        + dc_af * dc_attn / max(mfu_da, 0.01)
        + overhead
    )
    return np.maximum(predicted, 1.0)


def residual_log_ratio(params, pf_gemm, pf_attn, dc_gemm, dc_attn, actual):
    """Log-ratio residual for symmetric relative errors."""
    predicted = predict_9param_vec(params, pf_gemm, pf_attn, dc_gemm, dc_attn)
    # log(predicted/actual) — symmetric in over/under-prediction
    return np.log(predicted / np.maximum(actual, 1.0))


def fit_9param(train_df):
    """Fit 9-parameter global model on training data."""
    pf_gemm = train_df["prefill_gemm_us"].values
    pf_attn = train_df["prefill_attn_us"].values
    dc_gemm = train_df["decode_gemm_us"].values
    dc_attn = train_df["decode_attn_us"].values
    actual = train_df[TARGET_COL].values.astype(np.float64)

    # Initial guess: factors=1, overhead=50us, MFU=0.5
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 50.0, 0.5, 0.5, 0.5, 0.5])

    # Bounds: factors > 0, overhead > 0, MFU in (0.01, 1.0)
    lower = [0.01, 0.01, 0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01]
    upper = [100.0, 100.0, 100.0, 100.0, 1e6, 1.0, 1.0, 1.0, 1.0]

    result = least_squares(
        residual_log_ratio, x0,
        args=(pf_gemm, pf_attn, dc_gemm, dc_attn, actual),
        bounds=(lower, upper),
        method="trf",
        max_nfev=5000,
    )
    return result.x


def fit_36param_per_model(train_df):
    """Fit 36-parameter model (9 per model family)."""
    params_per_model = {}
    for model_name in sorted(train_df["model"].unique()):
        model_data = train_df[train_df["model"] == model_name]
        if len(model_data) < 20:
            # Fall back to global params
            params_per_model[model_name] = None
            continue
        params_per_model[model_name] = fit_9param(model_data)
    return params_per_model


def predict_with_params(df, params_dict, global_params):
    """Predict using per-model params with global fallback."""
    predicted = np.zeros(len(df))
    for model_name, group in df.groupby("model"):
        p = params_dict.get(model_name)
        if p is None:
            p = global_params
        idx = group.index
        preds = predict_9param_vec(
            p,
            group["prefill_gemm_us"].values,
            group["prefill_attn_us"].values,
            group["decode_gemm_us"].values,
            group["decode_attn_us"].values,
        )
        predicted[df.index.get_indexer(idx)] = preds
    return predicted


def main():
    # Load h1 components
    comp_path = os.path.join(H1_DIR, "output", "step_components.csv")
    df = pd.read_csv(comp_path)
    print(f"Loaded {len(df)} rows from {comp_path}", file=sys.stderr)

    # Temporal split
    split = temporal_split(df)
    train_df = df.loc[split["train"]].copy()
    valid_df = df.loc[split["valid"]].copy()
    test_df = df.loc[split["test"]].copy()
    print(f"  Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}", file=sys.stderr)

    # ========================================
    # Variant 1: 9-parameter global
    # ========================================
    print("\nFitting 9-parameter global model...", file=sys.stderr)
    global_params = fit_9param(train_df)
    print(f"  Params: {dict(zip(['pf_gf','pf_af','dc_gf','dc_af','overhead','mfu_pg','mfu_pa','mfu_dg','mfu_da'], global_params))}", file=sys.stderr)

    test_pred_9 = predict_9param_vec(
        global_params,
        test_df["prefill_gemm_us"].values,
        test_df["prefill_attn_us"].values,
        test_df["decode_gemm_us"].values,
        test_df["decode_attn_us"].values,
    )
    mape_9 = compute_mape(test_pred_9, test_df[TARGET_COL].values)
    print(f"  9-param global test MAPE: {mape_9:.1f}%", file=sys.stderr)

    # ========================================
    # Variant 2: 36-parameter per-model
    # ========================================
    print("\nFitting 36-parameter per-model models...", file=sys.stderr)
    per_model_params = fit_36param_per_model(train_df)
    for mn, p in per_model_params.items():
        if p is not None:
            print(f"  {mn}: overhead={p[4]:.1f}us, mfu_dc_gemm={p[7]:.3f}", file=sys.stderr)

    test_pred_36 = predict_with_params(test_df, per_model_params, global_params)
    mape_36 = compute_mape(test_pred_36, test_df[TARGET_COL].values)
    print(f"  36-param per-model test MAPE: {mape_36:.1f}%", file=sys.stderr)

    # ========================================
    # Per-experiment evaluation (both variants)
    # ========================================
    print("\nPer-experiment evaluation...", file=sys.stderr)
    all_predictions = []

    for exp_id in sorted(df["experiment_id"].unique()):
        exp_test = test_df[test_df["experiment_id"] == exp_id]
        if len(exp_test) == 0:
            continue

        # 9-param predictions
        pred_9 = predict_9param_vec(
            global_params,
            exp_test["prefill_gemm_us"].values,
            exp_test["prefill_attn_us"].values,
            exp_test["decode_gemm_us"].values,
            exp_test["decode_attn_us"].values,
        )

        # 36-param predictions
        model_name = exp_test["model"].iloc[0]
        p36 = per_model_params.get(model_name, global_params)
        if p36 is None:
            p36 = global_params
        pred_36 = predict_9param_vec(
            p36,
            exp_test["prefill_gemm_us"].values,
            exp_test["prefill_attn_us"].values,
            exp_test["decode_gemm_us"].values,
            exp_test["decode_attn_us"].values,
        )

        for variant, preds in [("9param_global", pred_9), ("36param_per_model", pred_36)]:
            pred_df = exp_test[["experiment_id", "model", "workload", "step.id", TARGET_COL]].copy()
            pred_df["predicted_us"] = preds
            pred_df["model_type"] = variant
            all_predictions.append(pred_df)

    preds_df = pd.concat(all_predictions, ignore_index=True)
    pred_path = os.path.join(SCRIPT_DIR, "output", "predictions.csv")
    preds_df.to_csv(pred_path, index=False)
    print(f"\n  Saved {len(preds_df)} predictions to {pred_path}", file=sys.stderr)

    # Save fitted parameters
    params_out = {
        "9param_global": global_params.tolist(),
        "36param_per_model": {k: v.tolist() if v is not None else None for k, v in per_model_params.items()},
    }
    params_path = os.path.join(SCRIPT_DIR, "output", "fitted_params.json")
    with open(params_path, "w") as f:
        json.dump(params_out, f, indent=2)


if __name__ == "__main__":
    main()
