#!/usr/bin/env python3
"""Match BLIS kernel-lookup warm-start to aiconfigurator at batch=1 (zero queueing).

Runs both tools at batch_size=1 for apples-to-apples comparison:
  - aiconfigurator run_agg(batch=1): predicts TTFT and E2E including scheduling overhead
  - BLIS kernel-lookup at γ=1, α=0: predicts from pure kernel sums

The TTFT gap (AIC - BLIS) at batch=1, zero queueing = α₀ warm-start value.
The per-token gap determines α₂.

Usage:
    cd /path/to/inference-sim
    PYTHONPATH=aiconfigurator/src:training \
        python3 training/scripts/match_aiconfigurator_warmstart.py
"""

from __future__ import annotations
import importlib.metadata, statistics, sys
from pathlib import Path
import yaml

_orig_mv = importlib.metadata.version
importlib.metadata.version = lambda n: "0.8.0" if n == "aiconfigurator" else _orig_mv(n)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "aiconfigurator" / "src"))
sys.path.insert(0, str(REPO_ROOT / "training"))

from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.models import LLAMAModel
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.common import MoEQuantMode
from scripts.generate_kernel_profile import (
    get_model_info, SYSTEMS_ROOT, get_quant_modes,
)

PROFILES_DIR = REPO_ROOT / "training" / "kernel_profiles"

# Dense models only (MoE needs additional moe_tp/ep config)
EXPERIMENTS = [
    ("Qwen/Qwen2.5-7B-Instruct",              1, "64-qwen2-5-7b-instruct-tp1-roleplay-1-1.yaml",     [256,512,1024], 200),
    ("mistralai/Mistral-Nemo-Instruct-2407",   1, "63-mistral-nemo-12b-tp1-codegen-1-1.yaml",         [256,512,1024], 200),
    ("meta-llama/Llama-2-7b-hf",               1, "20260217-155451-llama-2-7b-tp1-codegen.yaml",      [256,512,1024], 200),
    ("meta-llama/Llama-3.1-70B-Instruct",      4, "60-llama-3-1-70b-tp4-general-lite-4-1.yaml",       [256,512],      200),
    ("01-ai/Yi-34B",                           2, "65-01-ai-yi-34b-tp2-general-lite-2-1.yaml",         [256,512],      200),
]

VOCAB = {
    "Qwen/Qwen2.5-7B-Instruct": 152064,
    "mistralai/Mistral-Nemo-Instruct-2407": 131072,
    "meta-llama/Llama-2-7b-hf": 32000,
    "meta-llama/Llama-3.1-70B-Instruct": 128256,
    "01-ai/Yi-34B": 64000,
}

# ───────────────────────────────────────────────────────────────────────────
# BLIS profile interpolation (γ=1, α=0)
# ───────────────────────────────────────────────────────────────────────────

def interp1d(xs, ys, x):
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= x:
            f=(x-xs[i-1])/(xs[i]-xs[i-1]); return ys[i-1]+f*(ys[i]-ys[i-1])
    return ys[-1]

def interp2d(px,sx,tbl,p,s):
    def at(pi): return interp1d(sx,[tbl[si][pi] for si in range(len(sx))],s)
    if p<=px[0]: return at(0)
    if p>=px[-1]: return at(len(px)-1)
    for i in range(1,len(px)):
        if px[i]>=p:
            f=(p-px[i-1])/(px[i]-px[i-1]); return at(i-1)+f*(at(i)-at(i-1))
    return at(len(px)-1)

def blis_step_ms(prof, total_tokens, num_seqs=1, isl=None, mode="prefill", ctx=None):
    """BLIS step time (ms) at γ=1, including logits GEMM. mode='prefill'|'decode'."""
    L=prof["num_layers"]; n_dense=prof.get("num_dense_layers",L); n_moe=prof.get("num_moe_layers",0)
    ar_u=2*n_dense+n_moe; art=prof["allreduce"]; ar_oh=art["latency_us"][0]
    gt=prof["gemm"]
    t_g=interp1d(gt["tokens"],gt["latency_us"],float(total_tokens))*L
    t_ar=max(0,interp1d(art["tokens"],art["latency_us"],float(total_tokens))-ar_oh)*ar_u
    mt=prof.get("moe_compute"); t_m=0
    if mt and n_moe>0: t_m=interp1d(mt["tokens"],mt["latency_us"],float(total_tokens))*n_moe
    # Logits GEMM: once per step (scale=1, not per-layer), scaled by γ₁ (same correction as GEMM)
    lg=prof.get("logits_gemm"); t_logits=0
    if lg: t_logits=interp1d(lg["tokens"],lg["latency_us"],float(total_tokens))
    if mode=="prefill":
        ca=prof["context_attention"]
        t_a=interp2d(ca["batch_size"],ca["isl"],ca["latency_us"],float(num_seqs),float(isl))*L
    else:
        ga=prof["generation_attention"]
        t_a=interp2d(ga["tokens"],ga["context"],ga["latency_us"],float(total_tokens),float(ctx))*L
    return (t_g+t_logits+t_a+t_ar+t_m)/1000.0

# ───────────────────────────────────────────────────────────────────────────
# aiconfigurator predictions
# ───────────────────────────────────────────────────────────────────────────

def make_model(model_name, tp, gmode, kmode, fmode, info):
    moe_q = MoEQuantMode.fp8 if gmode and "fp8" in str(gmode).lower() else MoEQuantMode.float16
    mc = ModelConfig(tp_size=tp, gemm_quant_mode=gmode, kvcache_quant_mode=kmode,
                     fmha_quant_mode=fmode, moe_quant_mode=moe_q)
    vocab = VOCAB.get(model_name, 32000)
    return LLAMAModel(
        model_name,"llama","LlamaForCausalLM",
        info["num_layers"],info["num_heads"],info["num_kv_heads"],
        info["head_dim"],info["hidden_size"],info["inter_size"],
        vocab, info["num_layers"]*2048, mc,
    )

def aic_run(backend, model, db, isl, osl):
    """aiconfigurator TTFT and E2E (ms) at batch=1."""
    rc = RuntimeConfig(batch_size=1, isl=isl, osl=osl)
    s = backend.run_agg(model, db, rc, ctx_tokens=isl)
    d = s.get_result_dict()
    ttft = float(d["ttft"])
    e2e  = float(d["request_latency"])
    # Sanity check: TTFT should be ~5-500ms; if way off, adjust units
    if ttft < 0.01:    ttft*=1000; e2e*=1000   # was in seconds
    if ttft > 10000:   ttft/=1000; e2e/=1000   # was in µs
    return ttft, e2e

# ───────────────────────────────────────────────────────────────────────────
# Main comparison
# ───────────────────────────────────────────────────────────────────────────

def main():
    backend = VLLMBackend()
    all_alpha0, all_alpha2 = [], []

    for (model_name, tp, pfile, isl_list, osl) in EXPERIMENTS:
        print(f"\n{'='*76}")
        short = model_name.split("/")[-1]
        print(f"  {short:<36}  TP={tp}  OSL={osl}")
        print(f"{'='*76}")

        prof  = yaml.safe_load(open(PROFILES_DIR / pfile))
        info  = get_model_info(model_name)
        gmode,fmode,kmode = get_quant_modes(model_name)
        db    = PerfDatabase("h100_sxm","vllm","0.14.0",systems_root=SYSTEMS_ROOT)
        model = make_model(model_name, tp, gmode, kmode, fmode, info)

        print(f"\n  {'ISL':>5} | {'BLIS_pf':>8} {'AIC_TTFT':>9} | {'gap→α₀':>9} | "
              f"{'BLIS_dc':>8} {'AIC_TPOT':>9} | {'gap→α₂/tok':>10}")
        print(f"  {'-'*80}")

        for isl in isl_list:
            try:
                blis_pf = blis_step_ms(prof, isl, num_seqs=1, isl=isl, mode="prefill")
                avg_ctx = isl + osl//2
                blis_dc = blis_step_ms(prof, 1, mode="decode", ctx=avg_ctx)
                blis_e2e = blis_pf + (osl-1)*blis_dc

                aic_ttft, aic_e2e = aic_run(backend, model, db, isl, osl)
                aic_tpot = (aic_e2e - aic_ttft) / max(osl-1,1)

                alpha0_us = (aic_ttft - blis_pf)*1000   # ms→µs
                alpha2_us = (aic_tpot - blis_dc)*1000   # per-token gap µs

                all_alpha0.append(alpha0_us)
                all_alpha2.append(alpha2_us)

                print(f"  {isl:>5} | {blis_pf:>8.2f} {aic_ttft:>9.2f} | {alpha0_us:>+9.1f}µs | "
                      f"{blis_dc:>8.2f} {aic_tpot:>9.2f} | {alpha2_us:>+10.2f}µs/tok")
            except Exception as e:
                print(f"  {isl:>5} | ERROR: {e}")

    # Summary
    print(f"\n{'='*76}")
    print(f"  WARM-START ALPHA COEFFICIENTS  (median across all models × ISLs)")
    print(f"{'='*76}")
    a0_med = statistics.median(all_alpha0) if all_alpha0 else 0
    a2_med = statistics.median(all_alpha2) if all_alpha2 else 0
    a0_p25 = sorted(all_alpha0)[len(all_alpha0)//4] if all_alpha0 else 0
    a0_p75 = sorted(all_alpha0)[3*len(all_alpha0)//4] if all_alpha0 else 0
    print(f"  α₀ (QueueingTime per request):   median={a0_med:+.1f} µs  "
          f"[p25={a0_p25:+.0f}, p75={a0_p75:+.0f}]")
    print(f"  α₂ (OutputToken processing):     median={a2_med:+.2f} µs/tok")
    print(f"  α₁ (PostDecodeFixedOverhead):    0.0 µs  (not needed)")
    print(f"\n  Final warm-start (matching aiconfigurator at batch=1):")
    print(f"    alpha_coeffs = [{max(0,a0_med):.1f}, 0.0, {max(0,a2_med):.2f}]")
    print(f"    beta_coeffs  = [1.0,1.0,1.0,1.0,0.0,1.0,0.0,40.0,3.0,100.0]")

if __name__ == "__main__":
    main()
