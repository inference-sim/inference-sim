# Idea 2, H1: Analytical Component Decomposition Captures Physical Structure

**Status:** Not Supported (0/4 components > 0.8; only 6 pure-prefill steps)
**Family:** Performance-regime
**Type:** Type 2 (Statistical -- Dominance)
**Date:** 2026-02-26

## Hypothesis

> Decomposing step time into GEMM, attention, memory access, and overhead components -- each derived from batch Request objects and model architecture parameters -- captures the physically relevant structure. Per-component analytical estimates correlate >0.8 with actual step time contributions.

## Refuted-If

None of the 4 analytical components (prefill GEMM, prefill attention, decode GEMM, decode attention) achieves Pearson r > 0.6 with observed step duration on their respective pure-phase subsets. If the analytical decomposition does not even correlate with step time in pure-phase batches (where only one component dominates), the decomposition is structurally wrong and cannot be corrected by learned factors.

## Experiment Design

**Classification:** Statistical/Dominance

**Method:**
1. **Component derivation:** For each step, compute 4 analytical time estimates using the roofline methodology:
   - Prefill GEMM: `2 * num_layers * prefill_tokens * hidden_dim^2 * (1 + 2*intermediate_ratio) / (peak_flops * tp_degree)` (adjusted for MoE: multiply intermediate term by `active_experts/num_experts`)
   - Prefill attention: `2 * num_layers * num_heads * prefill_tokens * head_dim * kv_sum_prefill / (peak_flops * tp_degree)`
   - Decode GEMM: `2 * num_layers * decode_tokens * hidden_dim^2 * (1 + 2*intermediate_ratio) / (peak_flops * tp_degree)`
   - Decode attention: `2 * num_layers * num_heads * decode_tokens * head_dim * kv_sum_decode / (peak_flops * tp_degree)` (using per-request KV lengths from ProgressIndex)
2. **Pure-phase correlation:** On pure-prefill steps (decode_tokens=0) and pure-decode steps (prefill_tokens=0), compute Pearson r between each component's analytical estimate and observed step.duration_us. Pure-phase steps isolate individual components.
3. **Mixed-batch residual:** On mixed steps, compute the residual after subtracting the sum of all 4 components. This residual represents the non-additive interaction term (documented in BLIS H5).

**Data:** ~122,752 steps from 16 experiments (4 models x 4 workloads, H100 GPUs)
**Split:** No train/test split needed -- this is a correlation analysis, not a predictive model
**Baselines:** Pearson r of raw prefill_tokens and decode_tokens with step duration (the blackbox feature set)
**Success metric:** At least 2 of 4 components achieve Pearson r > 0.8 on their respective pure-phase subsets

## Feature Set

Per-request fields used to derive components:
- `ProgressIndex` -- per-request KV cache length (number of tokens already generated)
- `NumNewTokens` -- tokens generated in this step for this request
- `InputTokens` length -- original input length (for prefill token count)
- `OutputTokens` length -- tokens generated so far (for progress tracking)

Architecture parameters (from HuggingFace config.json):
- `hidden_dim` (hidden_size)
- `num_layers` (num_hidden_layers)
- `num_heads` (num_attention_heads)
- `head_dim` (hidden_size / num_attention_heads)
- `intermediate_dim` (intermediate_size)
- `num_experts` (num_local_experts, default 1 for dense)
- `active_experts` (num_experts_per_tok, default 1 for dense)
- `tp_degree` (from experiment configuration)

Hardware parameters (from hardware_config.json):
- `peak_flops_bf16` -- BF16 TFLOP/s for the GPU
- `peak_bandwidth` -- HBM bandwidth in GB/s

## Related Work

- **Williams et al.** (CACM 2009): The original roofline model -- performance is bounded by min(peak_compute, data * bandwidth). This hypothesis extends the roofline from a bound to a decomposed estimator.
- **Sarathi-Serve** (Agrawal et al., OSDI 2024): Chunked prefill serving system that explicitly models prefill-decode interaction. Their analysis of stall-free scheduling motivates treating mixed batches as non-additive.
- **Physics-informed ML** (Willard & Jia, 2020): Survey of techniques combining physical constraints with learned models. The analytical decomposition provides the physical backbone that learned corrections refine.

## Go Integration Path

The 4-component decomposition would be implemented as a `DecomposeStepTime(batch BatchContext, model ModelConfig, hw HardwareCalib) StepComponents` function in `sim/latency/`. This is a pure analytical computation that produces a `StepComponents` struct with `PrefillGEMM`, `PrefillAttention`, `DecodeGEMM`, `DecodeAttention` fields (all `time.Duration`). The decomposition reuses computation logic already present in `sim/latency/roofline.go` (`calculateTransformerFlops`, `calculateMemoryAccessBytes`) but restructured to return per-component times rather than a single total.
