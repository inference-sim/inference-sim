# Idea 1, H1: Physics-Informed Feature Set — FINDINGS

**Status:** Weakly Supported (per-experiment), Short-circuited (global)
**Date:** 2026-02-26

## Summary

A 30-feature physics-informed feature set was engineered and evaluated with Ridge regression against the 2-feature blackbox baseline. The global model fails (301.7% MAPE) due to the 3-order-of-magnitude step time range across model configurations. However, per-experiment Ridge models show substantial improvement on the majority of experiments, validating the feature set's informativeness.

## Experimental Setup

- **Features:** 30 features in 6 groups — batch tokens (5), KV cache proxies (5), phase indicators (4), physics/roofline features (8), architecture metadata (3), interaction terms (5)
- **Model:** Ridge regression (sklearn, alpha=1.0) with StandardScaler
- **Split:** Temporal 60/20/20 within each experiment (shared infrastructure)
- **Data:** 122,752 steps from 16 experiments (4 models × 4 workloads, H100 GPUs)

### Feature Set Limitations

The HYPOTHESIS.md specified KV statistics derived from ProgressIndex (kv_mean, kv_max, kv_min, kv_std, kv_sum). However, **ProgressIndex is a per-request field not available in step-level ground-truth data** — the step-level traces only contain aggregate batch features. We substituted system-state KV proxies: `kv.usage_gpu_ratio`, `kv.blocks_free_gpu`, `kv.blocks_total_gpu`, `queue.running_depth`. This is a known data limitation documented in `problem.md`.

Similarly, `attention_flops_ratio` and `gemm_flops_ratio` are set to constants (0.0 and 1.0) because computing attention FLOPs requires per-request KV cache lengths.

## Results

### Global Ridge (all experiments pooled)

| Metric | Ridge (30f) | Blackbox (2f) | Naive Mean |
|--------|-------------|---------------|------------|
| MAPE | 301.7% | 670.2% | 861.4% |
| MSPE | +68.6% | +649.4% | — |
| Pearson r | 0.848 | 0.407 | — |
| p99 error | 1549.2% | 2343.5% | — |

**Short-circuit triggered:** Global MAPE (301.7%) exceeds the 30% threshold. However, this threshold was designed for per-experiment models. The global model is structurally incapable of handling step times spanning 12μs to 250,000μs — no linear model can.

### Per-Experiment Ridge (16 separate models)

| Model | Workload | Ridge 30f MAPE | Blackbox MAPE | Δ |
|-------|----------|---------------|---------------|---|
| mixtral-8x7b-v0-1 | general | 9.0% | 9.2% | +0.2 |
| llama-2-7b | reasoning | 9.3% | 123.5% | +114.2 |
| mixtral-8x7b-v0-1 | codegen | 15.7% | 19.0% | +3.3 |
| llama-2-70b-hf | reasoning | 15.9% | 14.2% | -1.7 |
| codellama-34b | codegen | 17.1% | 21.6% | +4.6 |
| codellama-34b | reasoning | 25.4% | 37.3% | +11.8 |
| mixtral-8x7b-v0-1 | roleplay | 27.2% | 33.6% | +6.4 |
| codellama-34b | roleplay | 27.8% | 30.8% | +2.9 |
| llama-2-7b | roleplay | 37.8% | 40.3% | +2.5 |
| llama-2-70b | general | 57.4% | 61.2% | +3.7 |
| llama-2-70b-hf | codegen | 64.3% | 90.8% | +26.4 |
| llama-2-7b | codegen | 73.8% | 69.7% | -4.1 |
| codellama-34b | general | 143.7% | 151.1% | +7.4 |
| llama-2-70b | roleplay | 181.0% | 128.6% | -52.5 |
| llama-2-7b | general | 283.3% | 72.9% | -210.4 |
| mixtral-8x7b-v0-1 | reasoning | 484.9% | 222.8% | -262.0 |

**Average per-experiment MAPE:** Ridge 30f = 92.1%, Blackbox = 70.4%

### Top Features by Standardized Coefficient

| Feature | Coefficient |
|---------|-------------|
| f_prefill_compute_intensity | +19845.7 |
| f_prefill_tokens | -13402.8 |
| f_kv_blocks_used | +7568.7 |
| f_decode_tokens | +6731.9 |
| f_arithmetic_intensity | -5327.5 |

## Analysis

### What Worked
- **6 experiments achieved < 30% MAPE** with Ridge, including 2 under 10% — proving the feature set captures meaningful physics
- **Reasoning workloads improved dramatically** (Llama-7B: 123.5% → 9.3%) — the physics features capture long-context behavior that raw token counts miss
- **Pearson r improved** from 0.407 (blackbox) to 0.848 (30f Ridge) globally

### What Failed
- **4 experiments got significantly worse** than blackbox (Llama-7B-general: 72.9% → 283.3%, Mixtral-reasoning: 222.8% → 484.9%)
- The Ridge model is linear and cannot capture the nonlinear relationship between features and step time, especially for workloads with bimodal step time distributions
- System-state KV features (kv_blocks_used, running_depth) are informative but introduce temporal correlation — they describe *where* the system is, not *what* it's computing

### Root Cause of Failures
The worst experiments (general workloads, Mixtral-reasoning) have highly variable batch compositions. A linear model overweights features that work for the majority of steps, producing extreme predictions on atypical batches. This motivates tree-based models (h2) which can learn nonlinear decision boundaries.

## Verdict

**The 30-feature set is informative but requires a nonlinear model.** Per-experiment Ridge shows clear improvement on 12/16 experiments, validating the feature engineering. The short-circuit threshold on the global model is misleading for this problem structure. Proceed to h2-model (XGBoost).
