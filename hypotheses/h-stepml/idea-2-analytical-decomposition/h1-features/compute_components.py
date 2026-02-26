"""Compute 4 analytical FLOPs components for each step.

For each step in the ground-truth data, derives:
  - prefill_gemm_us: estimated prefill GEMM time from architecture FLOPs
  - prefill_attn_us: estimated prefill attention time
  - decode_gemm_us: estimated decode GEMM time
  - decode_attn_us: placeholder (per-request KV lengths unavailable at step level)

Saves augmented DataFrame to output/step_components.csv.
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


def compute_components_vectorized(df: pd.DataFrame, arch: dict, hw: dict) -> pd.DataFrame:
    """Compute analytical FLOPs components for all rows sharing one architecture."""
    d = arch["hidden_dim"]
    L = arch["num_layers"]
    H = arch["num_heads"]
    H_kv = arch["num_kv_heads"]
    d_head = arch["head_dim"]
    d_ff = arch["intermediate_dim"]
    tp = arch["tp"]
    moe = arch["moe_factor"]
    peak = hw["peak_flops"]
    d_kv = H_kv * d_head

    P = df["batch.prefill_tokens"].fillna(0).astype(np.float64).values
    D = df["batch.decode_tokens"].fillna(0).astype(np.float64).values

    # Prefill GEMM: QKV + O projections + MLP (gate/up/down)
    attn_proj_p = 2 * P * (d * d + 2 * d * d_kv) + 2 * P * d * d
    mlp_p = 2 * P * 3 * d * d_ff * moe
    prefill_gemm_flops = (attn_proj_p + mlp_p) * L / tp

    # Prefill Attention: QK^T + AV with causal masking (avg context ≈ P/2)
    avg_ctx = np.maximum(P / 2, 1)
    prefill_attn_flops = 2 * (2 * H * P * avg_ctx * d_head) * L / tp
    # Zero out where P == 0
    prefill_attn_flops = np.where(P > 0, prefill_attn_flops, 0.0)

    # Decode GEMM: same structure, D tokens
    attn_proj_d = 2 * D * (d * d + 2 * d * d_kv) + 2 * D * d * d
    mlp_d = 2 * D * 3 * d * d_ff * moe
    decode_gemm_flops = (attn_proj_d + mlp_d) * L / tp

    # Decode Attention: set to 0 — per-request KV lengths unavailable
    decode_attn_flops = np.zeros(len(df))

    # Convert FLOPs → microseconds (no MFU correction — raw analytical)
    result = pd.DataFrame(index=df.index)
    result["prefill_gemm_flops"] = prefill_gemm_flops
    result["prefill_attn_flops"] = prefill_attn_flops
    result["decode_gemm_flops"] = decode_gemm_flops
    result["decode_attn_flops"] = decode_attn_flops
    result["prefill_gemm_us"] = prefill_gemm_flops / peak * 1e6
    result["prefill_attn_us"] = prefill_attn_flops / peak * 1e6
    result["decode_gemm_us"] = decode_gemm_flops / peak * 1e6
    result["decode_attn_us"] = decode_attn_flops / peak * 1e6
    result["total_analytical_us"] = (
        result["prefill_gemm_us"]
        + result["prefill_attn_us"]
        + result["decode_gemm_us"]
        + result["decode_attn_us"]
    )
    return result


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
            print(f"  Loaded arch: {model_name} (d={arch_cache[model_name]['hidden_dim']}, "
                  f"L={arch_cache[model_name]['num_layers']}, tp={arch_cache[model_name]['tp']})",
                  file=sys.stderr)
        except (ValueError, FileNotFoundError) as e:
            print(f"  WARNING: Skipping model {model_name}: {e}", file=sys.stderr)

    # Compute components per model group (vectorized)
    print("Computing analytical components...", file=sys.stderr)
    comp_parts = []
    for model_name, group in df.groupby("model"):
        if model_name not in arch_cache:
            continue
        comp = compute_components_vectorized(group, arch_cache[model_name], hw)
        comp_parts.append(comp)

    comp_df = pd.concat(comp_parts)
    result = df.join(comp_df)
    result = result.dropna(subset=["prefill_gemm_us"])

    output_path = os.path.join(SCRIPT_DIR, "output", "step_components.csv")
    result.to_csv(output_path, index=False)
    print(f"  Saved {len(result)} rows to {output_path}", file=sys.stderr)

    # Phase breakdown
    pure_prefill = result[result["batch.decode_tokens"] == 0]
    pure_decode = result[result["batch.prefill_tokens"] == 0]
    mixed = result[(result["batch.prefill_tokens"] > 0) & (result["batch.decode_tokens"] > 0)]
    print(f"\n  Phase breakdown:", file=sys.stderr)
    print(f"    Pure prefill: {len(pure_prefill)} steps", file=sys.stderr)
    print(f"    Pure decode:  {len(pure_decode)} steps", file=sys.stderr)
    print(f"    Mixed batch:  {len(mixed)} steps", file=sys.stderr)


if __name__ == "__main__":
    main()
