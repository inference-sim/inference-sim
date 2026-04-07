#!/usr/bin/env python3
"""Generate kernel_profile.yaml for the BLIS kernel-lookup latency backend.

Queries aiconfigurator's GPU operations database to produce per-layer latency
lookup tables (one YAML per training experiment). All tables store per-layer
latencies in microseconds, matching the BLIS Go runtime's KernelLookupModel
conventions.

IMPORTANT: The aiconfigurator package is NOT installed — use sys.path.insert().
Use the metadata patch at the top to avoid PackageNotFoundError.

Usage:
    cd /path/to/inference-sim
    python training/scripts/generate_kernel_profile.py \
        --model meta-llama/Llama-2-7b-hf --tp 1 \
        --output training/kernel_profiles/llama-2-7b-tp1.yaml

    python training/scripts/generate_kernel_profile.py \
        --from-exp-dir training/trainval_data/ \
        --output-dir training/kernel_profiles/
"""

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Patch metadata version lookup — aiconfigurator is not pip-installed
# ---------------------------------------------------------------------------
_orig_metadata_version = importlib.metadata.version


def _patched_version(name):
    if name == "aiconfigurator":
        return "0.8.0"
    return _orig_metadata_version(name)


importlib.metadata.version = _patched_version

# Add aiconfigurator to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "aiconfigurator" / "src"))

from aiconfigurator.sdk.perf_database import PerfDatabase  # noqa: E402
from aiconfigurator.sdk.common import (  # noqa: E402
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)

SYSTEMS_ROOT = str(
    _REPO_ROOT / "aiconfigurator" / "src" / "aiconfigurator" / "systems"
)

# ---------------------------------------------------------------------------
# Lookup table grid points
# ---------------------------------------------------------------------------
# context_gemm / allreduce / moe_compute: total prefill tokens across batch
CONTEXT_TOKEN_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# generation_gemm: decode batch size (1 token per request)
DECODE_TOKEN_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# context_attention: batch_size x ISL (2D)
BATCH_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
ISL_GRID = [64, 128, 256, 512, 1024, 2048, 4096]
# generation_attention: decode_batch x context_length (2D)
CTX_GRID = [64, 128, 256, 512, 1024, 2048, 4096]

# ---------------------------------------------------------------------------
# Hardcoded model architectures (avoids HF downloads for gated models)
# ---------------------------------------------------------------------------
MODEL_ARCHITECTURES = {
    "meta-llama/Llama-2-7b-hf": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 32,  # MHA
        "head_dim": 128,
        "inter_size": 11008,
        "vocab_size": 32000,
        "num_experts": 0,
        "topk": 0,
        "moe_inter_size": 0,
        "dense_inter_size": 11008,
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "num_layers": 80,
        "hidden_size": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,  # GQA
        "head_dim": 128,
        "inter_size": 28672,
        "vocab_size": 128256,
        "num_experts": 0,
        "topk": 0,
        "moe_inter_size": 0,
        "dense_inter_size": 28672,
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "num_layers": 40,
        "hidden_size": 5120,
        "num_heads": 32,
        "num_kv_heads": 8,  # GQA
        "head_dim": 128,  # 5120/40=128 (note: 40 internal heads, 32 reported)
        "inter_size": 14336,
        "vocab_size": 131072,
        "num_experts": 0,
        "topk": 0,
        "moe_inter_size": 0,
        "dense_inter_size": 14336,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "num_layers": 28,
        "hidden_size": 3584,
        "num_heads": 28,
        "num_kv_heads": 4,  # GQA
        "head_dim": 128,
        "inter_size": 18944,
        "vocab_size": 152064,
        "num_experts": 0,
        "topk": 0,
        "moe_inter_size": 0,
        "dense_inter_size": 18944,
    },
    "01-ai/Yi-34B": {
        "num_layers": 60,
        "hidden_size": 7168,
        "num_heads": 56,
        "num_kv_heads": 8,  # GQA
        "head_dim": 128,
        "inter_size": 20480,
        "vocab_size": 64000,
        "num_experts": 0,
        "topk": 0,
        "moe_inter_size": 0,
        "dense_inter_size": 20480,
    },
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic": {
        "num_layers": 48,
        "hidden_size": 5120,
        "num_heads": 40,
        "num_kv_heads": 8,  # GQA
        "head_dim": 128,
        "inter_size": 8192,  # MoE expert FFN dim
        "vocab_size": 202048,
        "num_experts": 16,
        "topk": 1,
        "moe_inter_size": 8192,
        "dense_inter_size": 16384,  # Dense layer FFN dim
    },
}


def get_model_info(model_path: str) -> dict:
    """Get model architecture from hardcoded table."""
    if model_path not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"Unknown model {model_path!r}. "
            f"Known models: {list(MODEL_ARCHITECTURES.keys())}"
        )
    return dict(MODEL_ARCHITECTURES[model_path])


def get_quant_modes(model_path: str):
    """Determine quantization modes from model name.

    For FP8 models: GEMM in FP8, FMHA in float16 (no FP8 FMHA data available),
    KV cache in FP8.
    """
    is_fp8 = "fp8" in model_path.lower()
    return (
        GEMMQuantMode.fp8 if is_fp8 else GEMMQuantMode.float16,
        # FMHA is always float16 — no FP8 FMHA data in the database
        FMHAQuantMode.float16,
        KVCacheQuantMode.fp8 if is_fp8 else KVCacheQuantMode.float16,
    )


def gemm_dims(info: dict, tp: int) -> dict:
    """Compute GEMM shapes for each op type, adjusted for TP.

    For dense models (or dense layers of MoE models), uses inter_size for
    gate_up/down. MoE expert GEMMs are handled separately by query_moe().
    """
    h = info["hidden_size"]
    n_heads = info["num_heads"]
    n_kv = info["num_kv_heads"]
    d_h = info["head_dim"]
    # Use dense_inter_size for the per-layer GEMM (dense FFN layers)
    inter = info["dense_inter_size"]

    heads_per_gpu = max(1, n_heads // tp)
    kv_per_gpu = max(1, n_kv // tp)

    return {
        "qkv": {"n": heads_per_gpu * d_h + 2 * kv_per_gpu * d_h, "k": h},
        "proj": {"n": h, "k": heads_per_gpu * d_h},
        "gate_up": {"n": 2 * (inter // tp), "k": h},
        "down": {"n": h, "k": inter // tp},
    }


def query_1d_gemm(db, dims: dict, token_grid: list, quant_mode) -> list:
    """Query total per-layer GEMM latency (QKV+proj+gate_up+down, per layer, us).

    aiconfigurator db.query_gemm() returns per-invocation ms for one GEMM.
    We sum 4 GEMMs per layer and convert ms → µs.
    Result is per-layer (Go runtime multiplies by numLayers).
    """
    out = []
    for tokens in token_grid:
        total_ms = 0.0
        for name, d in dims.items():
            try:
                r = db.query_gemm(
                    m=int(tokens), n=d["n"], k=d["k"], quant_mode=quant_mode
                )
                total_ms += float(r)
            except Exception:
                # Fallback to float16 if FP8 shape unavailable
                try:
                    r = db.query_gemm(
                        m=int(tokens),
                        n=d["n"],
                        k=d["k"],
                        quant_mode=GEMMQuantMode.float16,
                    )
                    total_ms += float(r)
                except Exception:
                    pass
        # ms -> us (raw query is already per-invocation = per-layer; do NOT divide by num_layers)
        out.append(round(total_ms * 1000.0, 6))
    return out


def query_2d_context_attn(
    db,
    info: dict,
    tp: int,
    batch_grid: list,
    isl_grid: list,
    fmha_mode,
    kv_mode,
) -> list:
    """Query context attention per layer (us).

    Returns rows indexed by ISL (secondary axis), columns by batch_size (primary).
    LatencyUs[isl_idx][batch_idx] matches Lookup2D convention.
    """
    heads_per_gpu = max(1, info["num_heads"] // tp)
    kv_per_gpu = max(1, info["num_kv_heads"] // tp)
    d_h = info["head_dim"]
    # n_kv=0 means MHA (same as n) in aiconfigurator API
    n_kv = 0 if kv_per_gpu == heads_per_gpu else kv_per_gpu

    rows = []
    for isl in isl_grid:
        row = []
        for b in batch_grid:
            try:
                r = db.query_context_attention(
                    b=int(b),
                    s=int(isl),
                    prefix=0,
                    n=heads_per_gpu,
                    n_kv=n_kv,
                    kvcache_quant_mode=kv_mode,
                    fmha_quant_mode=fmha_mode,
                    head_size=d_h,
                )
                row.append(round(float(r) * 1000.0, 6))  # per-invocation = per-layer
            except Exception:
                row.append(0.0)
        rows.append(row)
    return rows


def query_2d_gen_attn(
    db, info: dict, tp: int, token_grid: list, ctx_grid: list, kv_mode
) -> list:
    """Query generation attention per layer (us).

    Returns rows indexed by context length (secondary), columns by tokens (primary).
    LatencyUs[ctx_idx][token_idx] matches Lookup2D convention.
    """
    heads_per_gpu = max(1, info["num_heads"] // tp)
    kv_per_gpu = max(1, info["num_kv_heads"] // tp)
    d_h = info["head_dim"]
    n_kv = 0 if kv_per_gpu == heads_per_gpu else kv_per_gpu

    rows = []
    for ctx in ctx_grid:
        row = []
        for tokens in token_grid:
            try:
                r = db.query_generation_attention(
                    b=int(tokens),
                    s=int(ctx),
                    n=heads_per_gpu,
                    n_kv=n_kv,
                    kvcache_quant_mode=kv_mode,
                    head_size=d_h,
                )
                row.append(round(float(r) * 1000.0, 6))  # per-invocation = per-layer
            except Exception:
                row.append(0.0)
        rows.append(row)
    return rows


def query_allreduce(
    db, info: dict, tp: int, token_grid: list
) -> list:
    """Query AllReduce latency per invocation (us), keyed by token count.

    Stores per-invocation latency — the BLIS runtime multiplies by
    allReduceUnits (= 2*numDenseLayers + numMoELayers), not numLayers.
    For TP=1: all zeros (no AllReduce needed).
    """
    if tp == 1:
        return [0.0] * len(token_grid)

    h = info["hidden_size"]
    out = []
    for tokens in token_grid:
        msg_size = int(tokens) * h  # element count
        try:
            r = db.query_custom_allreduce(
                quant_mode=CommQuantMode.half, tp_size=tp, size=msg_size
            )
            out.append(round(float(r) * 1000.0, 6))  # ms -> us
        except Exception:
            out.append(0.0)
    return out


def query_moe(
    db, info: dict, tp: int, num_moe_layers: int, token_grid: list, quant_mode
) -> list:
    """Query MoE expert computation per MoE layer (us).

    Falls back to GEMM-based estimation if aiconfigurator MoE data is
    unavailable for the exact configuration (e.g., 16-expert Scout).
    """
    h = info["hidden_size"]
    moe_inter = info["moe_inter_size"]
    topk = info["topk"]
    num_experts = info["num_experts"]

    if num_experts == 0 or num_moe_layers == 0:
        return None

    # Map GEMM quant mode to MoE quant mode
    moe_quant = MoEQuantMode.fp8 if quant_mode == GEMMQuantMode.fp8 else MoEQuantMode.float16

    out = []
    for tokens in token_grid:
        try:
            r = db.query_moe(
                num_tokens=int(tokens),
                hidden_size=h,
                inter_size=moe_inter,
                topk=topk,
                num_experts=num_experts,
                moe_tp_size=tp,
                moe_ep_size=1,
                quant_mode=moe_quant,
                workload_distribution="uniform",
                is_context=True,
            )
            out.append(round(float(r) * 1000.0, 6))  # per-invocation = per MoE layer
        except Exception:
            # Fallback: estimate MoE via individual expert GEMMs
            # Each expert: gate_up (2*moe_inter, h) + down (h, moe_inter)
            # Total per token: topk experts activated
            try:
                gate_up = db.query_gemm(
                    m=int(tokens) * topk,
                    n=2 * (moe_inter // tp),
                    k=h,
                    quant_mode=quant_mode,
                )
                down = db.query_gemm(
                    m=int(tokens) * topk,
                    n=h,
                    k=moe_inter // tp,
                    quant_mode=quant_mode,
                )
                total_ms = float(gate_up) + float(down)
                out.append(round(total_ms * 1000.0, 6))  # per-invocation = per MoE layer
            except Exception:
                out.append(0.0)
    return out


def detect_moe_layers(model_path: str, num_layers: int, num_experts: int) -> tuple:
    """Detect number of MoE and dense layers.

    Scout (interleaved, InterleaveMoELayerStep=1): half MoE, half dense.
    Other MoE models (uniform): all layers are MoE.
    Dense models: all layers are dense.
    """
    if num_experts == 0:
        return 0, num_layers
    is_interleaved = any(
        x in model_path.lower() for x in ["scout", "16e"]
    )
    if is_interleaved:
        num_moe = num_layers // 2
        return num_moe, num_layers - num_moe
    return num_layers, 0


def generate_profile(
    model_path: str,
    tp: int,
    gpu: str,
    backend: str,
    version: str,
    output_path: str,
) -> None:
    """Generate and write kernel_profile.yaml for one model/TP combination."""
    print(f"  Loading model info: {model_path}")
    info = get_model_info(model_path)
    num_layers = info["num_layers"]
    num_experts = info["num_experts"]
    num_moe, num_dense = detect_moe_layers(model_path, num_layers, num_experts)

    print(
        f"  layers={num_layers} (MoE={num_moe}, dense={num_dense}), "
        f"h={info['hidden_size']}, heads={info['num_heads']}, "
        f"kv_heads={info['num_kv_heads']}, inter={info['inter_size']}"
    )

    print(f"  Loading PerfDatabase: {gpu}/{backend}/{version}")
    db = PerfDatabase(gpu, backend, version, systems_root=SYSTEMS_ROOT)

    gemm_mode, fmha_mode, kv_mode = get_quant_modes(model_path)

    print("  Querying context GEMM...")
    ctx_gemm = query_1d_gemm(db, gemm_dims(info, tp), CONTEXT_TOKEN_GRID, gemm_mode)

    print("  Querying context attention...")
    ctx_attn = query_2d_context_attn(
        db, info, tp, BATCH_GRID, ISL_GRID, fmha_mode, kv_mode
    )

    print("  Querying generation GEMM...")
    gen_gemm = query_1d_gemm(db, gemm_dims(info, tp), DECODE_TOKEN_GRID, gemm_mode)

    print("  Querying generation attention...")
    gen_attn = query_2d_gen_attn(
        db, info, tp, DECODE_TOKEN_GRID, CTX_GRID, kv_mode
    )

    print(f"  Querying AllReduce (tp={tp})...")
    allreduce = query_allreduce(db, info, tp, CONTEXT_TOKEN_GRID)

    moe_compute = None
    if num_moe > 0:
        print(
            f"  Querying MoE compute (experts={num_experts}, topk={info['topk']})..."
        )
        moe_compute = query_moe(
            db, info, tp, num_moe, CONTEXT_TOKEN_GRID, gemm_mode
        )

    # Build profile dict matching Go KernelProfile struct
    profile = {
        "gpu": gpu,
        "backend": backend,
        "version": version,
        "model": model_path,
        "tp": tp,
        "num_layers": num_layers,
        "num_moe_layers": num_moe,
        "num_dense_layers": num_dense,
        "hidden_dim": info["hidden_size"],
        "context_gemm": {
            "tokens": [float(t) for t in CONTEXT_TOKEN_GRID],
            "latency_us": ctx_gemm,
        },
        "context_attention": {
            "batch_size": [float(b) for b in BATCH_GRID],
            "isl": [float(s) for s in ISL_GRID],
            "latency_us": ctx_attn,
        },
        "generation_gemm": {
            "tokens": [float(t) for t in DECODE_TOKEN_GRID],
            "latency_us": gen_gemm,
        },
        "generation_attention": {
            "tokens": [float(t) for t in DECODE_TOKEN_GRID],
            "context": [float(c) for c in CTX_GRID],
            "latency_us": gen_attn,
        },
        "allreduce": {
            "tokens": [float(t) for t in CONTEXT_TOKEN_GRID],
            "latency_us": allreduce,
        },
    }
    if moe_compute is not None:
        profile["moe_compute"] = {
            "tokens": [float(t) for t in CONTEXT_TOKEN_GRID],
            "latency_us": moe_compute,
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)
    print(f"  Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", help="HuggingFace model path")
    parser.add_argument("--tp", type=int, help="Tensor parallelism degree")
    parser.add_argument("--output", help="Output YAML path")
    parser.add_argument("--gpu", default="h100_sxm")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--version", default="0.14.0")
    parser.add_argument(
        "--from-exp-dir",
        help="Generate profiles for all exp-config.yaml in this dir tree",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory when using --from-exp-dir",
    )
    args = parser.parse_args()

    if args.from_exp_dir:
        import glob
        import shutil

        configs = sorted(
            glob.glob(os.path.join(args.from_exp_dir, "*/exp-config.yaml"))
        )
        if not configs:
            print(f"No exp-config.yaml found under {args.from_exp_dir}")
            sys.exit(1)

        # Deduplicate: same model+tp share a profile (copy instead of re-query)
        seen = {}
        for cfg_path in configs:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            model = cfg["model"]
            tp = cfg.get("tensor_parallelism", 1)
            exp_name = os.path.basename(os.path.dirname(cfg_path))
            out = os.path.join(args.output_dir or ".", f"{exp_name}.yaml")
            key = (model, tp)
            if key in seen and os.path.exists(seen[key]):
                print(f"\n=== {exp_name} (reusing {os.path.basename(seen[key])}) ===")
                os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                shutil.copy2(seen[key], out)
                print(f"  Copied: {out}")
            else:
                print(f"\n=== {exp_name} ===")
                generate_profile(model, tp, args.gpu, args.backend, args.version, out)
                seen[key] = out
    else:
        if not (args.model and args.tp and args.output):
            parser.error("--model, --tp, and --output are required")
        generate_profile(
            args.model, args.tp, args.gpu, args.backend, args.version, args.output
        )


if __name__ == "__main__":
    main()
