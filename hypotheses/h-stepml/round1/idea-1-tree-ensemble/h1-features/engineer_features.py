"""Engineer 30 physics-informed features for step-time prediction.

6 feature groups (30 total):
  1. Batch tokens (5): raw batch features
  2. KV proxies (5): system-state KV cache proxies
  3. Phase indicators (4): prefill/decode phase structure
  4. Physics features (8): roofline-derived compute/memory features
  5. Architecture (3): model identity, TP, MoE flag
  6. Interaction terms (5): cross-feature products

Saves augmented DataFrame to output/features.csv.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from data_loader import load_all_experiments

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "model_configs")
HW_CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "hardware_config.json")

MODEL_DIR_MAP = {
    "llama-2-7b": "llama-2-7b-hf",
    "llama-2-70b": "llama-2-70b-hf",
    "llama-2-70b-hf": "llama-2-70b-hf",
    "codellama-34b": "codellama-34b-instruct-hf",
    "mixtral-8x7b-v0-1": "mixtral-8x7b-v0.1",
}

TP_MAP = {
    "llama-2-7b": 1,
    "llama-2-70b": 4,
    "llama-2-70b-hf": 4,
    "codellama-34b": 2,
    "mixtral-8x7b-v0-1": 2,
}

# Label encoding for models (stable ordering)
MODEL_LABEL_MAP = {
    "codellama-34b": 0,
    "llama-2-7b": 1,
    "llama-2-70b": 2,
    "llama-2-70b-hf": 2,
    "mixtral-8x7b-v0-1": 3,
}


def load_hw_config() -> dict:
    with open(HW_CONFIG_PATH) as f:
        hw = json.load(f)
    h100 = hw["H100"]
    return {
        "peak_flops": h100["TFlopsPeak"] * 1e12,
        "peak_bw": h100["BwPeakTBs"] * 1e12 * h100.get("bwEfficiencyFactor", 0.82),
    }


def load_model_arch(model_name: str) -> dict:
    config_dir = MODEL_DIR_MAP.get(model_name)
    if config_dir is None:
        raise ValueError(f"Unknown model: {model_name}")
    config_path = os.path.join(MODEL_CONFIG_DIR, config_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    d = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    H_kv = cfg.get("num_key_value_heads", H)
    d_head = d // H

    return {
        "hidden_dim": d,
        "num_layers": cfg["num_hidden_layers"],
        "num_heads": H,
        "num_kv_heads": H_kv,
        "head_dim": d_head,
        "intermediate_dim": cfg["intermediate_size"],
        "num_experts": cfg.get("num_local_experts", 1),
        "active_experts": cfg.get("num_experts_per_tok", 1),
        "moe_factor": cfg.get("num_experts_per_tok", 1) / cfg.get("num_local_experts", 1),
        "tp": TP_MAP[model_name],
    }


# All 30 feature column names (in order)
FEATURE_COLS = [
    # Group 1: Batch tokens (5)
    "f_prefill_tokens",
    "f_decode_tokens",
    "f_scheduled_tokens",
    "f_num_prefill_reqs",
    "f_num_decode_reqs",
    # Group 2: KV proxies (5)
    "f_kv_gpu_usage",
    "f_kv_blocks_used",
    "f_kv_blocks_free",
    "f_running_depth",
    "f_kv_pressure",
    # Group 3: Phase indicators (4)
    "f_prefill_fraction",
    "f_decode_fraction",
    "f_is_mixed_batch",
    "f_is_pure_prefill",
    # Group 4: Physics features (8)
    "f_total_flops_estimate",
    "f_arithmetic_intensity",
    "f_compute_bound_indicator",
    "f_prefill_compute_intensity",
    "f_decode_memory_intensity",
    "f_attention_flops_ratio",
    "f_gemm_flops_ratio",
    "f_active_param_ratio",
    # Group 5: Architecture (3)
    "f_model_id",
    "f_tp_degree",
    "f_is_moe",
    # Group 6: Interaction terms (5)
    "f_prefill_x_decode",
    "f_kv_pressure_x_decode_reqs",
    "f_intensity_x_moe",
    "f_prefill_x_kv_usage",
    "f_batch_size_x_kv_pressure",
]


def engineer_features_for_group(df: pd.DataFrame, arch: dict, hw: dict) -> pd.DataFrame:
    """Compute all 30 features for rows sharing one architecture."""
    d = arch["hidden_dim"]
    L = arch["num_layers"]
    H_kv = arch["num_kv_heads"]
    d_head = arch["head_dim"]
    d_ff = arch["intermediate_dim"]
    tp = arch["tp"]
    moe = arch["moe_factor"]
    peak_flops = hw["peak_flops"]
    peak_bw = hw["peak_bw"]
    d_kv = H_kv * d_head
    bytes_per_param = 2.0  # BF16

    out = pd.DataFrame(index=df.index)

    P = df["batch.prefill_tokens"].fillna(0).astype(np.float64)
    D = df["batch.decode_tokens"].fillna(0).astype(np.float64)
    S = df["batch.scheduled_tokens"].fillna(0).astype(np.float64)
    n_pf = df["batch.num_prefill_reqs"].fillna(0).astype(np.float64)
    n_dc = df["batch.num_decode_reqs"].fillna(0).astype(np.float64)

    # Group 1: Batch tokens
    out["f_prefill_tokens"] = P
    out["f_decode_tokens"] = D
    out["f_scheduled_tokens"] = S
    out["f_num_prefill_reqs"] = n_pf
    out["f_num_decode_reqs"] = n_dc

    # Group 2: KV proxies
    kv_usage = df["kv.usage_gpu_ratio"].fillna(0).astype(np.float64)
    kv_total = df["kv.blocks_total_gpu"].fillna(1).astype(np.float64)
    kv_free = df["kv.blocks_free_gpu"].fillna(0).astype(np.float64)
    kv_used = kv_total - kv_free
    running = df["queue.running_depth"].fillna(0).astype(np.float64)

    out["f_kv_gpu_usage"] = kv_usage
    out["f_kv_blocks_used"] = kv_used
    out["f_kv_blocks_free"] = kv_free
    out["f_running_depth"] = running
    out["f_kv_pressure"] = kv_used / kv_total.clip(lower=1)

    # Group 3: Phase indicators
    S_safe = S.clip(lower=1)
    out["f_prefill_fraction"] = P / S_safe
    out["f_decode_fraction"] = D / S_safe
    out["f_is_mixed_batch"] = ((n_pf > 0) & (n_dc > 0)).astype(np.float64)
    out["f_is_pure_prefill"] = (n_dc == 0).astype(np.float64)

    # Group 4: Physics features
    # GEMM FLOPs for all scheduled tokens
    attn_proj = 2 * S * (d * d + 2 * d * d_kv) + 2 * S * d * d
    mlp = 2 * S * 3 * d * d_ff * moe
    total_flops = (attn_proj + mlp) * L / tp

    # Weight bytes (loaded once per step)
    weights_per_layer = (d * (d + 2 * d_kv) + d * d + 3 * d * d_ff) * bytes_per_param
    total_weight_bytes = weights_per_layer * L / tp

    out["f_total_flops_estimate"] = total_flops
    out["f_arithmetic_intensity"] = total_flops / np.maximum(total_weight_bytes, 1)

    machine_balance = peak_flops / peak_bw
    out["f_compute_bound_indicator"] = (out["f_arithmetic_intensity"] > machine_balance).astype(np.float64)

    # Per-phase intensities
    prefill_flops = np.where(P > 0,
        (2 * P * (d*d + 2*d*d_kv) + 2*P*d*d + 2*P*3*d*d_ff*moe) * L / tp, 0)
    decode_flops = np.where(D > 0,
        (2 * D * (d*d + 2*d*d_kv) + 2*D*d*d + 2*D*3*d*d_ff*moe) * L / tp, 0)

    out["f_prefill_compute_intensity"] = np.where(P > 0, prefill_flops / np.maximum(total_weight_bytes, 1), 0)
    out["f_decode_memory_intensity"] = np.where(D > 0, total_weight_bytes / np.maximum(decode_flops, 1), 0)

    # Attention vs GEMM ratio: without per-request KV, attention FLOPs are unknown
    out["f_attention_flops_ratio"] = 0.0
    out["f_gemm_flops_ratio"] = 1.0
    out["f_active_param_ratio"] = moe

    # Group 5: Architecture
    out["f_model_id"] = df["model"].map(MODEL_LABEL_MAP).fillna(-1).astype(np.float64)
    out["f_tp_degree"] = float(tp)
    out["f_is_moe"] = 1.0 if arch["num_experts"] > 1 else 0.0

    # Group 6: Interaction terms
    out["f_prefill_x_decode"] = P * D
    kv_pressure = out["f_kv_pressure"]
    out["f_kv_pressure_x_decode_reqs"] = kv_pressure * n_dc
    out["f_intensity_x_moe"] = out["f_arithmetic_intensity"] * out["f_is_moe"]
    out["f_prefill_x_kv_usage"] = P * kv_usage
    out["f_batch_size_x_kv_pressure"] = (n_pf + n_dc) * kv_pressure

    return out


def main():
    print("Loading all experiments...", file=sys.stderr)
    df = load_all_experiments()
    print(f"  Loaded {len(df)} steps from {df['experiment_id'].nunique()} experiments", file=sys.stderr)

    hw = load_hw_config()

    # Pre-load architectures
    arch_cache: dict[str, dict] = {}
    for model_name in df["model"].unique():
        try:
            arch_cache[model_name] = load_model_arch(model_name)
            print(f"  Loaded arch: {model_name}", file=sys.stderr)
        except (ValueError, FileNotFoundError) as e:
            print(f"  WARNING: Skipping model {model_name}: {e}", file=sys.stderr)

    # Engineer features per model group
    print("Engineering 30 features...", file=sys.stderr)
    feature_parts = []
    for model_name, group in df.groupby("model"):
        if model_name not in arch_cache:
            continue
        features = engineer_features_for_group(group, arch_cache[model_name], hw)
        feature_parts.append(features)

    features_df = pd.concat(feature_parts)

    # Join features with original data
    keep_cols = [
        "experiment_id", "model", "workload", "tp", "step.id", "step.duration_us",
        "batch.prefill_tokens", "batch.decode_tokens",
    ]
    result = df[keep_cols].join(features_df)
    result = result.dropna(subset=["f_prefill_tokens"])

    output_path = os.path.join(SCRIPT_DIR, "output", "features.csv")
    result.to_csv(output_path, index=False)
    print(f"  Saved {len(result)} rows with {len(FEATURE_COLS)} features to {output_path}", file=sys.stderr)

    # Feature statistics
    print(f"\n  Feature value ranges:", file=sys.stderr)
    for col in FEATURE_COLS[:10]:  # first 10 for brevity
        vals = result[col]
        print(f"    {col:40s}  min={vals.min():.2f}  max={vals.max():.2f}  mean={vals.mean():.2f}", file=sys.stderr)
    print(f"    ... ({len(FEATURE_COLS) - 10} more features)", file=sys.stderr)


if __name__ == "__main__":
    main()
