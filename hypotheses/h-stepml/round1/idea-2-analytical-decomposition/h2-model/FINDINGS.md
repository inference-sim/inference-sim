# Idea 2, H2: Learned Correction Factors — FINDINGS

**Status:** Not Supported
**Date:** 2026-02-26

## Summary

Learned multiplicative correction factors on the analytical decomposition fail to achieve competitive per-step MAPE. The 9-parameter global model achieves 96.0% average MAPE; the 36-parameter per-model variant achieves 78.7% — both worse than the blackbox baseline (70.4%). The analytical backbone is structurally incomplete without per-request KV cache information, and no amount of correction factor tuning can compensate.

## Experimental Setup

- **Model variants:**
  - **9-parameter global:** 4 component factors + 1 overhead + 4 MFU discounts, shared across all models
  - **36-parameter per-model:** 9 parameters × 4 model families
- **Fitting:** scipy.optimize.least_squares with 'trf' method, log-ratio objective: `min Σ (log(predicted/actual))²` for symmetric relative errors
- **Bounds:** Component factors ∈ [0.01, 100], overhead ∈ [0, 1e6], MFU ∈ [0.01, 1.0]
- **Split:** Temporal 60/20/20 (global split across all experiments)
- **Baseline:** Per-experiment blackbox (2-feature linear regression)

## Results

### Aggregate Metrics

| Variant | Avg MAPE | Median MAPE | Avg MSPE | Exps < 15% | Exps < 25% |
|---------|----------|-------------|----------|------------|------------|
| 9-param global | 96.0% | 83.3% | +4.2% | 0/16 | 0/16 |
| 36-param per-model | 78.7% | 79.4% | +3.7% | 0/16 | 0/16 |
| Blackbox baseline | 70.4% | — | — | 1/16 | 1/16 |

### Per-Experiment Results (36-parameter variant)

| Model | Workload | 36-param MAPE | Blackbox MAPE | 36-param MSPE | 36-param r |
|-------|----------|--------------|---------------|---------------|-----------|
| llama-2-70b | roleplay | 41.9% | 128.6% | +14.6% | 0.523 |
| llama-2-7b | roleplay | 42.9% | 40.3% | +27.9% | 0.560 |
| llama-2-7b | codegen | 58.6% | 69.7% | +29.9% | 0.422 |
| codellama-34b | roleplay | 70.7% | 30.8% | +66.9% | 0.438 |
| codellama-34b | general | 74.7% | 151.1% | -34.2% | 0.420 |
| llama-2-70b-hf | reasoning | 75.3% | 14.2% | -72.8% | 0.962 |
| llama-2-70b | general | 76.1% | 61.2% | -70.3% | 0.268 |
| llama-2-7b | general | 78.9% | 72.9% | -78.7% | 0.669 |
| llama-2-70b-hf | codegen | 79.9% | 90.8% | +20.4% | 0.687 |
| llama-2-7b | reasoning | 83.5% | 123.5% | -61.6% | 0.163 |
| mixtral-8x7b-v0-1 | codegen | 84.4% | 19.0% | +83.6% | 0.184 |
| mixtral-8x7b-v0-1 | roleplay | 86.0% | 33.6% | +82.8% | 0.346 |
| codellama-34b | reasoning | 91.4% | 37.3% | -90.5% | 0.925 |
| mixtral-8x7b-v0-1 | reasoning | 92.4% | 222.8% | -82.2% | 0.852 |
| mixtral-8x7b-v0-1 | general | 99.7% | 9.2% | +99.7% | 0.657 |
| codellama-34b | codegen | 123.4% | 21.6% | +123.0% | 0.228 |

### Fitted Parameters (9-param global)

| Parameter | Value | Interpretation |
|-----------|-------|---------------|
| prefill_gemm_factor | 0.014 | Prefill GEMM heavily discounted (analytical overestimates) |
| prefill_attn_factor | 4.17 | Prefill attention amplified (compensating for causal masking approximation) |
| decode_gemm_factor | 0.49 | Decode GEMM at ~50% of analytical (MFU effect) |
| decode_attn_factor | 39.5 | **Extreme amplification** — trying to compensate for zero decode_attn |
| overhead_us | 85.7 | Base overhead per step |
| mfu_prefill_gemm | 0.998 | Near peak (unrealistic) |
| mfu_prefill_attn | 0.881 | Reasonable |
| mfu_decode_gemm | 0.744 | Reasonable |
| mfu_decode_attn | 0.500 | At lower bound (inert — decode_attn is zero) |

### Fitted Parameters (36-param, per-model overhead)

All 4 model families converged to **overhead = 0**, meaning the optimizer pushes all explanatory power into the component factors. MFU values vary from 0.49 (Mixtral) to 0.87 (Llama-70B), which are physically plausible.

## Analysis

### Why Correction Factors Fail

1. **Missing backbone component:** `decode_attn_us = 0` for all steps because per-request KV cache lengths are unavailable. For pure-decode batches (80.6% of data), the analytical prediction is `decode_gemm_factor × decode_gemm_us + overhead`. This misses the attention component entirely, which is the dominant cost for long-context decode batches.

2. **The optimizer compensates pathologically:** `decode_attn_factor = 39.5` is the optimizer trying to amplify a zero signal. Since `decode_attn_us = 0`, this parameter is completely inert. The `overhead` term (85.7μs) and `decode_gemm_factor` (0.49) absorb what they can, but cannot model KV-length-dependent variation.

3. **MSPE is low but meaningless:** Average |MSPE| = 3.7% appears good, but this is because the errors are enormous in both directions (some steps are overestimated by 100%+, others underestimated by 90%+), and they happen to roughly cancel in the mean.

### Comparison with Blackbox

The 36-parameter model is **worse** than the 2-parameter blackbox on average (78.7% vs 70.4%). This is remarkable: 36 free parameters with physics-based features perform worse than 2 coefficients on raw token counts. The explanation is that the blackbox model's `decode_tokens` feature is a *better* proxy for decode step time than the analytically-derived `decode_gemm_us` — because `decode_tokens` implicitly correlates with batch size, which correlates with attention cost, while `decode_gemm_us` explicitly ignores attention.

### Where Correction Factors Help

The model does improve a few experiments relative to blackbox:
- Llama-70B-roleplay: 128.6% → 41.9% (blackbox fails badly here; corrections help)
- CodeLlama-general: 151.1% → 74.7%
- Llama-7B-reasoning: 123.5% → 83.5%

These are experiments where the blackbox is particularly bad, and any structured prediction helps.

### Parameter Efficiency

Moving from 9 to 36 parameters reduces average MAPE by 17.3 pp (96.0% → 78.7%). This is a meaningful but insufficient improvement — 4× more parameters for 18% relative improvement suggests diminishing returns. The bottleneck is the backbone, not the correction factor count.

## Verdict

**The hypothesis is not supported.** Learned correction factors on an incomplete analytical backbone cannot compete with even the simple blackbox model. The root cause is identical to h1: **decode attention FLOPs are uncomputable without per-request KV cache lengths**, making the decomposition structurally incomplete for the 80.6% of steps that are pure-decode.

### Comparison with Idea 1

| Metric | Idea 2 (36-param) | Idea 1 (XGBoost) | Blackbox |
|--------|-------------------|-------------------|----------|
| Avg MAPE | 78.7% | 34.0% | 70.4% |
| Exps < 15% | 0/16 | 5/16 | 1/16 |
| Exps < 25% | 0/16 | 6/16 | 1/16 |
| Parameters | 36 | ~30K per model | 3 |

Idea 1's XGBoost dramatically outperforms Idea 2's correction factors because:
1. XGBoost uses **system-state KV features** (kv_blocks_used, running_depth) as proxies for per-request KV information — imperfect but informative
2. XGBoost can learn **nonlinear relationships** between features and step time
3. Per-experiment training gives XGBoost implicit architecture-specific calibration

### Recommendation

**Abandon Idea 2** in its current form. The analytical decomposition is sound in principle but cannot be validated or deployed without per-request KV cache information. If the `LatencyModel` interface is extended to provide per-request `ProgressIndex`, Idea 2 could be revisited — the decomposition + correction factors approach would have the advantage of human interpretability (9 auditable constants vs 30K tree parameters) and smaller binary size.
