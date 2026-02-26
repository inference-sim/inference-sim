# Idea 2, H1: Analytical Component Decomposition — FINDINGS

**Status:** Not Supported (formal criteria), Informative (per-experiment signal)
**Date:** 2026-02-26

## Summary

The 4-component analytical decomposition (prefill GEMM, prefill attention, decode GEMM, decode attention) was computed for all 122,752 steps and correlated with observed step duration. The formal hypothesis criteria (≥2 components with Pearson r > 0.8 on pure-phase subsets) is not met — primarily because vLLM's chunked prefill produces only **6 pure-prefill steps** across the entire dataset. Per-experiment correlations reveal strong signal for general workloads but weak signal for reasoning/roleplay.

## Experimental Setup

- **Method:** Correlation analysis (no train/test split — this is a structural validation)
- **Components computed:**
  - Prefill GEMM FLOPs: QKV + O projections + MLP (gate/up/down), scaled by architecture and TP
  - Prefill Attention FLOPs: QK^T + AV with causal masking (avg context ≈ P/2)
  - Decode GEMM FLOPs: same structure, decode tokens
  - Decode Attention FLOPs: **set to 0** — per-request KV cache lengths unavailable in step-level data
- **Architecture parameters:** Loaded from HuggingFace config.json for each of 4 models
- **Hardware:** H100 peak_flops = 989.5 TFLOP/s BF16 (no MFU correction — raw analytical)
- **Data:** 122,752 steps from 16 experiments

## Results

### Phase Breakdown

| Phase | Steps | % of Total |
|-------|-------|-----------|
| Pure prefill (decode_tokens = 0) | 6 | 0.005% |
| Pure decode (prefill_tokens = 0) | 98,961 | 80.6% |
| Mixed batch | 23,789 | 19.4% |

**Critical finding:** vLLM's chunked prefill scheduling produces almost no pure-prefill batches. The hypothesis assumed pure-phase subsets would be large enough for meaningful correlation analysis — this assumption is wrong.

### Correlation on Pure-Phase Subsets

| Subset | Component | Pearson r | n_steps |
|--------|-----------|-----------|---------|
| pure_prefill | prefill_gemm_us | N/A | 6 |
| pure_prefill | prefill_attn_us | N/A | 6 |
| pure_prefill | total_analytical_us | N/A | 6 |
| pure_prefill | batch.prefill_tokens (baseline) | N/A | 6 |
| pure_decode | decode_gemm_us | 0.4367 | 98,961 |
| pure_decode | decode_attn_us | N/A | 98,961 |
| pure_decode | total_analytical_us | 0.4367 | 98,961 |
| pure_decode | batch.decode_tokens (baseline) | 0.4202 | 98,961 |
| all | total_analytical_us | 0.3350 | 122,752 |

**Hypothesis verdict: 0/4 components achieve r > 0.8. FAIL.**

### Per-Experiment Correlations (total_analytical_us)

| Model | Workload | Pearson r |
|-------|----------|-----------|
| mixtral-8x7b-v0-1 | general | 0.9420 |
| llama-2-7b | general | 0.8593 |
| codellama-34b | general | 0.6436 |
| llama-2-70b | general | 0.5954 |
| codellama-34b | codegen | 0.4343 |
| llama-2-70b-hf | codegen | 0.3600 |
| llama-2-70b | roleplay | 0.3380 |
| mixtral-8x7b-v0-1 | codegen | 0.2528 |
| llama-2-7b | codegen | 0.2490 |
| llama-2-70b-hf | reasoning | 0.2355 |
| codellama-34b | roleplay | 0.2024 |
| llama-2-7b | roleplay | 0.1674 |
| mixtral-8x7b-v0-1 | roleplay | 0.1207 |
| codellama-34b | reasoning | 0.1171 |
| llama-2-7b | reasoning | 0.1160 |
| mixtral-8x7b-v0-1 | reasoning | 0.0216 |

## Analysis

### Why Pure-Phase Correlation Fails

1. **Only 6 pure-prefill steps exist.** vLLM's continuous batching + chunked prefill means nearly every batch contains at least one decode request. The experimental design assumed pure-phase batches would be common — they are not under continuous batching.

2. **Decode attention is zero.** Without per-request KV cache lengths (ProgressIndex), decode attention FLOPs cannot be computed. Since 80.6% of steps are pure-decode, the analytical decomposition is missing the dominant time component for the majority of steps.

3. **Decode GEMM correlates at r=0.437** — meaningful but not strong. Decode GEMM FLOPs are proportional to `decode_tokens * model_size`, which captures the linear scaling but misses the nonlinear effects of varying KV cache lengths and batch composition.

### Where the Decomposition Has Signal

The per-experiment correlations reveal a clear pattern:
- **"General" workloads show strong correlation** (Mixtral r=0.94, Llama-7B r=0.86) — these have moderate batch sizes and relatively uniform batch compositions where GEMM dominates
- **Reasoning/roleplay workloads show weak correlation** (r < 0.25) — these have long contexts where attention (KV-dependent) dominates over GEMM, and the missing decode_attn makes the analytical estimate incorrect

### Structural Limitation

The decomposition's backbone is correct in principle — GEMM and attention are the dominant compute components of transformer inference. However, **the step-level data lacks the per-request granularity needed to compute attention FLOPs**. The feature gap identified in `problem.md` (no per-request KV cache lengths) is the binding constraint, not the decomposition methodology.

## Verdict

**The formal hypothesis is not supported** (0/4 components > 0.8), but the root cause is data availability, not decomposition structure:

1. Pure-phase subsets are too small (6 prefill steps) for meaningful correlation — chunked prefill invalidates the experimental design
2. Decode attention is uncomputable without per-request KV lengths — the most important component for 80.6% of steps is missing
3. Per-experiment correlations show the decomposition has signal where GEMM dominates (general workloads), confirming the physics is sound

**This is an informational finding, not a kill signal for the decomposition approach.** If per-request ProgressIndex were available in step-level data (via LatencyModel interface extension), the decomposition would be fully computable and likely correlate much more strongly.
