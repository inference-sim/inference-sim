#!/usr/bin/env python3
"""Derive ideal warm-start γ coefficients by comparing BLIS kernel-lookup
predictions (at γ=1) against aiconfigurator's own per-operation predictions
for the same batch shapes.

At γ=1 and zero load, BLIS (with correct basis function semantics) should
predict the same step times as aiconfigurator. Any ratio AIC/BLIS IS the
warm-start γ that makes BLIS match aiconfigurator before optimization.

Usage:
    cd /path/to/inference-sim
    PYTHONPATH=aiconfigurator/src:training \
        python3 training/scripts/compute_warmstart_gammas.py
"""

from __future__ import annotations
import importlib.metadata, statistics, sys
from pathlib import Path
import yaml

# Patch metadata lookup — aiconfigurator not pip-installed
_orig_mv = importlib.metadata.version
importlib.metadata.version = lambda n: "0.8.0" if n == "aiconfigurator" else _orig_mv(n)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "aiconfigurator" / "src"))
sys.path.insert(0, str(REPO_ROOT / "training"))

from aiconfigurator.sdk.perf_database import PerfDatabase
from scripts.generate_kernel_profile import (
    get_model_info, gemm_dims,
    query_1d_gemm, query_2d_context_attn, query_2d_gen_attn, query_allreduce,
    SYSTEMS_ROOT, get_quant_modes,
)

PROFILES_DIR = REPO_ROOT / "training" / "kernel_profiles"

MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct",   1, "64-qwen2-5-7b-instruct-tp1-roleplay-1-1.yaml"),
    ("meta-llama/Llama-3.1-70B-Instruct", 4, "60-llama-3-1-70b-tp4-general-lite-4-1.yaml"),
    ("RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic", 2, "17-llama-4-scout-17b-16e-tp2-general-lite-2-1.yaml"),
    ("01-ai/Yi-34B",                2, "65-01-ai-yi-34b-tp2-general-lite-2-1.yaml"),
]

PREFILL_CASES = [(1, 512), (4, 256), (8, 128), (1, 1024)]
DECODE_CASES  = [(1, 512), (8, 512), (32, 1024), (128, 512)]


# ---------------------------------------------------------------------------
# BLIS profile interpolation (γ=1 predictions from YAML lookup tables)
# ---------------------------------------------------------------------------

def interp1d(xs, ys, x):
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= x:
            f = (x - xs[i-1]) / (xs[i] - xs[i-1])
            return ys[i-1] + f * (ys[i] - ys[i-1])
    return ys[-1]

def interp2d(p_ax, s_ax, table, p, s):
    def at_pi(pi):
        return interp1d(s_ax, [table[si][pi] for si in range(len(s_ax))], s)
    if p <= p_ax[0]: return at_pi(0)
    if p >= p_ax[-1]: return at_pi(len(p_ax)-1)
    for i in range(1, len(p_ax)):
        if p_ax[i] >= p:
            f = (p - p_ax[i-1]) / (p_ax[i] - p_ax[i-1])
            return at_pi(i-1) + f*(at_pi(i) - at_pi(i-1))
    return at_pi(len(p_ax)-1)

def blis_prefill(prof, num_seqs, isl):
    L = prof["num_layers"]
    total = num_seqs * isl
    n_dense = prof.get("num_dense_layers", L)
    n_moe   = prof.get("num_moe_layers", 0)
    ar_u = 2 * n_dense + n_moe
    gt = prof["gemm"]
    t_g = interp1d(gt["tokens"], gt["latency_us"], total) * L
    ca = prof["context_attention"]
    t_a = interp2d(ca["batch_size"], ca["isl"], ca["latency_us"], num_seqs, isl) * L
    art = prof["allreduce"]
    ar_oh = art["latency_us"][0]
    t_ar = max(0, interp1d(art["tokens"], art["latency_us"], total) - ar_oh) * ar_u
    mt = prof.get("moe_compute"); t_m = 0
    if mt and n_moe > 0:
        t_m = interp1d(mt["tokens"], mt["latency_us"], total) * n_moe
    return dict(gemm=t_g, attn=t_a, ar=t_ar, moe=t_m)

def blis_decode(prof, batch, ctx):
    L = prof["num_layers"]
    n_dense = prof.get("num_dense_layers", L)
    n_moe   = prof.get("num_moe_layers", 0)
    ar_u = 2 * n_dense + n_moe
    gt = prof["gemm"]
    t_g = interp1d(gt["tokens"], gt["latency_us"], float(batch)) * L
    ga = prof["generation_attention"]
    t_a = interp2d(ga["tokens"], ga["context"], ga["latency_us"], batch, ctx) * L
    art = prof["allreduce"]
    ar_oh = art["latency_us"][0]
    t_ar = max(0, interp1d(art["tokens"], art["latency_us"], float(batch)) - ar_oh) * ar_u
    mt = prof.get("moe_compute"); t_m = 0
    if mt and n_moe > 0:
        t_m = interp1d(mt["tokens"], mt["latency_us"], float(batch)) * n_moe
    return dict(gemm=t_g, attn=t_a, ar=t_ar, moe=t_m)


# ---------------------------------------------------------------------------
# aiconfigurator direct predictions (same query functions as generate_kernel_profile.py)
# ---------------------------------------------------------------------------

def aic_prefill(db, info, tp, gmode, fmode, kmode, num_seqs, isl):
    L = info["num_layers"]
    n_moe = info.get("num_moe_layers", 0)
    n_dense = L - n_moe
    ar_u = 2 * n_dense + n_moe
    total = num_seqs * isl
    t_g = query_1d_gemm(db, gemm_dims(info, tp), [int(total)], gmode)[0] * L
    rows = query_2d_context_attn(db, info, tp, [int(num_seqs)], [int(isl)], fmode, kmode)
    t_a = rows[0][0] * L
    ar_oh = query_allreduce(db, info, tp, [1])[0]
    raw_ar = query_allreduce(db, info, tp, [int(total)])[0]
    t_ar = max(0, raw_ar - ar_oh) * ar_u
    return dict(gemm=t_g, attn=t_a, ar=t_ar)

def aic_decode(db, info, tp, gmode, fmode, kmode, batch, ctx):
    L = info["num_layers"]
    n_moe = info.get("num_moe_layers", 0)
    n_dense = L - n_moe
    ar_u = 2 * n_dense + n_moe
    t_g = query_1d_gemm(db, gemm_dims(info, tp), [int(batch)], gmode)[0] * L
    rows = query_2d_gen_attn(db, info, tp, [int(batch)], [int(ctx)], kmode)
    t_a = rows[0][0] * L
    ar_oh = query_allreduce(db, info, tp, [1])[0]
    raw_ar = query_allreduce(db, info, tp, [int(batch)])[0]
    t_ar = max(0, raw_ar - ar_oh) * ar_u
    return dict(gemm=t_g, attn=t_a, ar=t_ar)


def compare_model(model_name, tp, profile_file):
    print(f"\n{'='*76}")
    print(f"  {model_name}  TP={tp}")
    print(f"{'='*76}")
    prof = yaml.safe_load(open(PROFILES_DIR / profile_file))
    info = get_model_info(model_name)
    info["num_moe_layers"] = prof.get("num_moe_layers", 0)
    db = PerfDatabase("h100_sxm", "vllm", "0.14.0", systems_root=SYSTEMS_ROOT)
    gmode, fmode, kmode = get_quant_modes(model_name)

    g_gemm, g_attn = [], []
    hdr = f"  {'case':>10}  {'B_gemm':>8} {'A_gemm':>8} {'γ₁':>6}  {'B_attn':>8} {'A_attn':>8} {'γ_a':>6}  {'B_tot':>8} {'A_tot':>8}"

    for phase, cases in [("PREFILL", PREFILL_CASES), ("DECODE", DECODE_CASES)]:
        print(f"\n  {phase}:"); print(hdr); print("  " + "-"*78)
        for (a, b) in cases:
            try:
                if phase == "PREFILL":
                    bv = blis_prefill(prof, a, b)
                    av = aic_prefill(db, info, tp, gmode, fmode, kmode, a, b)
                else:
                    bv = blis_decode(prof, a, b)
                    av = aic_decode(db, info, tp, gmode, fmode, kmode, a, b)
                g1 = av["gemm"]/bv["gemm"] if bv["gemm"]>0 else float("nan")
                ga = av["attn"]/bv["attn"] if bv["attn"]>0 else float("nan")
                g_gemm.append(g1); g_attn.append(ga)
                bt = sum(bv.values()); at = sum(av.values())
                print(f"  ({a:3d},{b:4d})  {bv['gemm']:8.1f} {av['gemm']:8.1f} {g1:6.3f}  {bv['attn']:8.1f} {av['attn']:8.1f} {ga:6.3f}  {bt:8.1f} {at:8.1f}")
            except Exception as e:
                print(f"  ({a:3d},{b:4d})  ERROR: {e}")

    if g_gemm:
        g1m = statistics.median(g_gemm)
        gam = statistics.median(g_attn)
        print(f"\n  Warm-start: γ₁={g1m:.4f}  γ₂=γ₃={gam:.4f}")
        return g1m, gam
    return None, None


def main():
    results = []
    for args in MODELS:
        g1, ga = compare_model(*args)
        if g1 is not None:
            results.append((g1, ga))

    if not results:
        print("\nNo results — check aiconfigurator availability.")
        return

    g1f = statistics.median([r[0] for r in results])
    gaf = statistics.median([r[1] for r in results])
    print(f"\n{'='*76}")
    print(f"  RECOMMENDED WARM-START  (median across all models)")
    print(f"{'='*76}")
    print(f"  γ₁ (GEMM)       = {g1f:.6f}")
    print(f"  γ₂ (ctx-attn)   = {gaf:.6f}")
    print(f"  γ₃ (gen-attn)   = {gaf:.6f}  (same basis as γ₂)")
    print(f"  γ₄              = 0.0  (unused)")
    print(f"  γ₅ (AllReduce)  = 0.0  (CUDA-graph overlap — optimizer tunes)")
    print(f"  γ₆ (MoE)        = 1.0")
    print(f"  γ₇ (µs/layer)   = 0.0")
    print(f"  γ₈ (µs/req)     = 40.0")
    print(f"  γ₉ (µs/step)    = 3.0")
    print(f"  γ₁₀ (reserved)  = 100.0")
    print(f"\n  beta_coeffs = [{g1f:.6f},{gaf:.6f},{gaf:.6f},0.0,0.0,1.0,0.0,40.0,3.0,100.0]")


if __name__ == "__main__":
    main()
