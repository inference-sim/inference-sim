# Iteration 15: Three-Axis Correction — Decode Amplification + MoE Non-Compute + Dense Batching

## Executive Summary

Iterations 10-14 have failed catastrophically (loss 2000-4000%, 13-27× worse than iter7's 155%). Iter14's β₅ fix was necessary but insufficient (loss barely improved: 2387% → 2319%). Analysis of **baseline simulator errors** reveals the roofline model has THREE INDEPENDENT systematic errors, each requiring separate correction:

1. **Decode underestimation** (-90% to -95% ITL MPE across ALL models)
2. **MoE underestimation** (-99% to -50% TTFT MPE for Scout experiments)
3. **Dense overestimation** (+330% to +3031% TTFT MPE for dense models)

**Iter15 strategy**: Address all three errors simultaneously with cold-start optimization:
- **β₁, β₄ amplification** (5-15×, 3-8×): Decode phase 10-20× slower than roofline predicts
- **β₈ MoE non-compute** (NEW): Routing latency + load imbalance beyond FLOPs
- **β₉ prefill batching penalty** (NEW): Mixed prefill/decode batch MFU degradation
- **Cold-start initialization**: Random uniform (not warm-start from iter7/iter14)

---

## H-main: Three-Axis Correction Recovers from Catastrophic Failure

**Prediction**: Overall loss will decrease from 2319% (iter14) to **<300%** (≥87% improvement), with:
- TTFT RMSE: 1314% → <150% (≥89% improvement)
- E2E RMSE: 1006% → <150% (≥85% improvement)

**Quantitative per-model predictions**:

| Model Type | Experiment | Iter14 TTFT APE | Predicted Iter15 TTFT APE | Improvement |
|-----------|------------|-----------------|---------------------------|-------------|
| **Scout MoE** | exp_17 (general) | 847% | <150% | ≥82% |
| **Scout MoE** | exp_20 (codegen) | 544% | <120% | ≥78% |
| **Dense** | exp_63 (Mistral codegen) | 1193% | <250% | ≥79% |
| **Dense** | exp_60 (Llama-3.1-70B) | 1033% | <200% | ≥81% |

**Causal Mechanism**:

The roofline model's theoretical time estimates are systematically wrong in THREE orthogonal dimensions:

**1. Decode Underestimation (All Models)**:
- **Evidence**: baseline_errors.json shows -90% to -95% ITL MPE across ALL 13 experiments (Scout, Llama, Mistral, Yi, Qwen)
- **Root cause**: Roofline assumes decode achieves theoretical MFU (0.10-0.20 for memory-bound, 0.60-0.80 for compute-bound). Reality: vLLM decode is 10-20× slower due to:
  - Small per-request GEMMs with poor tensor core utilization
  - PagedAttention kernel launch overhead per layer
  - KV cache pointer chasing and non-contiguous memory access
  - Per-layer synchronization barriers in TP mode
- **Fix**: Increase β₁ (decode memory MFU) from 1.0-1.15 to **5.0-15.0** and β₄ (decode compute MFU) from 0.7-0.85 to **3.0-8.0**
- **Physics**: β₁ × roofline_decode_memory_time and β₄ × roofline_decode_compute_time now amplify by 5-15× to match 10-20× slower reality

**2. MoE Underestimation (Scout Experiments)**:
- **Evidence**: baseline_errors.json shows Scout TTFT MPE ranging from -99.88% (exp_17) to -50.07% (exp_20), averaging -69% underestimation
- **Root cause**: Roofline computes expert FLOPs correctly, but MoE has NON-COMPUTE overhead beyond expert execution:
  - **Expert routing**: Token-to-expert dispatch (scatter/gather) — not captured by FLOPs
  - **Load imbalance**: When some experts get >2× avg load, stragglers dominate latency
  - **Expert communication**: All-to-all tensor routing in distributed MoE (TP > 1)
- **Fix**: Add **β₈ (MoE non-compute latency)** — per-token non-compute overhead in MoE layers
  - Basis function: `num_moe_layers × (prefill_tokens + decode_tokens) × moe_routing_latency_per_token`
  - Expected range: β₈ = 20-80 μs/token (captures routing + load imbalance + communication beyond FLOPs)
- **Physics**: β₅ (MoE gating) captures the gating network FLOPs (routing probability computation). β₈ captures the LATENCY of actually routing tokens and handling load imbalance, which is not compute-bound but communication/synchronization-bound.

**3. Dense Overestimation (Dense Models)**:
- **Evidence**: baseline_errors.json shows dense TTFT MPE ranging from +330% (exp_60, Llama-3.1-70B) to +3031% (exp_63, Mistral codegen), averaging +820% overestimation
- **Root cause**: Roofline assumes prefill batches achieve near-peak MFU (0.50-0.60 on H100) and overestimates by 3-30×. Reality: vLLM prefill is slower due to:
  - **Batch heterogeneity**: Mixing 1-token decode requests with 512-token prefill requests causes kernel inefficiency
  - **Memory layout mismatch**: Prefill expects contiguous token sequences, decode expects scattered KV blocks
  - **Synchronization overhead**: vLLM must wait for ALL requests in batch, not just prefill subset
- **Fix**: **β₀ (prefill MFU scaling)** — SCALE DOWN roofline estimate (not up!)
  - Expected range: β₀ = 0.05-0.25 (was 0.16-0.22 in iter7)
  - Physics: If roofline predicts 10 seconds and reality is 1 second, then β₀ = 0.1 (divide by 10×)
  - Roofline overestimates by 3-30× → β₀ must be 0.03-0.3 to correct
- **Additional fix**: Add **β₉ (prefill batching penalty)** — additive heterogeneity overhead
  - Basis function: `num_prefill_tokens × (1.0 + num_decode_requests / max(1, num_prefill_tokens))`
  - Pure prefill batch → factor = 1.0 (base penalty), high heterogeneity → factor = 1.0 + large_ratio
  - Expected range: β₉ = 0.5-2.0 μs/token (additive overhead on top of β₀ × roofline_prefill_time)
- **Physics**: β₀ scales DOWN the roofline overestimate. β₉ adds a small heterogeneity-dependent penalty on top.

**Why all three fixes are needed simultaneously**:
- Iter14 fixed β₅ (MoE gating layer count) but ignored the other two axes → loss barely improved (2387% → 2319%, only 2.8%)
- Fixing decode alone would help ITL but not TTFT
- Fixing MoE alone would help Scout but not dense models
- Fixing dense alone would help TTFT but not ITL
- **All three errors are orthogonal** — they must be addressed together for the optimizer to converge

**Code Citations**:
- **Decode underestimation**: `sim/latency/evolved_model.go:172-192` (β₁, β₄ decode basis functions)
- **MoE underestimation**: `sim/latency/evolved_model.go:238-271` (β₅ MoE gating, will add β₈ MoE non-compute)
- **Dense overestimation**: `sim/latency/evolved_model.go:149-163` (β₀ prefill compute, will add β₉ prefill batching penalty)
- **vLLM decode performance**: `training/references/vidur/` (vidur achieves -14% to -32% TTFT MPE by using conservative decode estimates)

**Diagnostic Clause**: *If this fails to achieve <300% loss, it indicates one of three failure modes:*
1. **Coefficient magnitude wrong** (β₁/β₄/β₈/β₉ ranges incorrect) → Expand bounds and re-optimize with 3000 trials
2. **Basis functions still structurally wrong** (missing another overhead term) → Profile real vLLM to identify what's missing
3. **Cold-start optimization failed to escape bad basin** → Try different optimizer (CMA-ES instead of TPE) or increase budget to 5000 trials

---

## H-ablation-decode: Decode Amplification is Primary Driver

**Prediction**: Removing decode amplification (reverting β₁, β₄ to iter14 ranges 1.0-1.15, 0.7-0.85) will increase loss by ≥1000pp (to >1300%), because decode underestimation affects ALL experiments uniformly (-90% to -95% ITL MPE baseline).

**Causal Mechanism**: ITL (inter-token latency) measures decode phase latency. Baseline shows -90% to -95% ITL MPE across all 13 experiments (Scout, dense, all workloads). This universal underestimation suggests a systematic error in decode time calculation that dominates E2E latency (most tokens are decode, not prefill).

**Diagnostic Clause**: *If removing decode amplification only increases loss by <500pp, decode is not the dominant error term — investigate whether MoE or dense batching terms absorbed decode error signal during optimization.*

---

## H-ablation-moe: MoE Non-Compute Fixes Scout Underestimation

**Prediction**: Removing β₈ (MoE non-compute) will increase Scout experiment APE by ≥200pp (exp_17: <150% → >350%), because Scout baseline shows -69% avg TTFT MPE (needs 2-3× correction beyond compute).

**Causal Mechanism**: β₅ (MoE gating) captures gating network FLOPs (routing probability computation). But MoE expert execution has non-compute latency from token routing (scatter/gather), load imbalance (stragglers), and expert communication (all-to-all). β₈ captures this non-FLOPs overhead that roofline misses entirely.

**Diagnostic Clause**: *If removing β₈ only increases Scout APE by <50pp, MoE non-compute overhead is negligible — MoE underestimation comes from incorrect expert FLOPs calculation (investigate roofline MoE FLOPs formula).*

---

## H-ablation-batching: Dense Batching Penalty Fixes Overestimation

**Prediction**: Removing β₉ (prefill batching penalty) will increase dense codegen experiment APE by ≥300pp (exp_63: <250% → >550%), because dense codegen baseline shows +1031% TTFT MPE (needs MFU degradation model).

**Causal Mechanism**: Codegen workloads have high heterogeneity (long prefills + short outputs → many 1-token decode requests mixed with large prefills). β₉'s `batch_heterogeneity_factor = num_decode_tokens / num_prefill_tokens` will be large for codegen, adding significant overhead. Roleplay/reasoning workloads have lower heterogeneity (fewer decode requests per batch) → smaller β₉ contribution.

**Diagnostic Clause**: *If removing β₉ only increases dense APE by <100pp, batching inefficiency is not the cause of dense overestimation — investigate whether β₀ (prefill MFU) is too optimistic (should be 0.8-1.2, not 0.16-0.22).*

---

## H-boundary: Cold-Start vs Warm-Start Initialization

**Prediction**: Cold-start (random uniform initialization) will find a lower loss minimum than warm-start from iter7 by ≥100pp, because:
1. Dataset shifted between iter7 and iter13 (reasoning → reasoning-lite)
2. Iter14 warm-started from iter7 and failed (loss 2319%)
3. New basis functions (β₈, β₉) have no iter7 equivalents to initialize from

**Causal Mechanism**: Warm-starting from iter7 biases the optimizer to stay near iter7's basin of attraction (β₀=0.191, β₁=1.108, etc.). But the dataset changed and new basis functions were added, so iter7's optimal point is no longer valid. Cold-start allows the optimizer to explore the full parameter space without bias.

**Diagnostic Clause**: *If cold-start performs WORSE than warm-start by >50pp, the search space is too large (10 coefficients × [lower, upper] bounds) — reduce basis functions or add physics-based priors.*

---

## H-error-pattern: Workload-Specific Predictions

**Prediction**: After iter15, error distribution will show:
- **Scout experiments**: <150% TTFT APE (down from 342-847% in iter14)
- **Dense codegen** (high heterogeneity): <250% TTFT APE (down from 1193-1417% in iter14)
- **Dense general-lite**: <200% TTFT APE (down from 1033-3774% in iter14)
- **Reasoning-lite** (numerical failures in iter14): Return valid results (no 100% timeouts)

**Causal Mechanism**:
- Scout improves most because β₈ directly addresses -69% avg TTFT baseline underestimation
- Dense codegen improves because β₉ captures high batch heterogeneity
- General-lite improves because β₁/β₄ decode amplification fixes -90% ITL baseline
- Reasoning-lite recovers because massive decode amplification prevents negative/zero latency predictions that caused timeouts

**Diagnostic Clause**: *If error pattern does NOT match predictions:*
- Scout still >300% → β₈ basis function wrong (investigate MoE routing implementation in vLLM)
- Dense codegen still >500% → β₉ heterogeneity model wrong (investigate batch composition in training data)
- Reasoning-lite still 100% → numerical stability issue not fixed (investigate clamping logic in StepTime)

---

## H-robustness: Coefficient Physical Plausibility

**Prediction**: After optimization, coefficients will satisfy physical bounds:
- β₀ (prefill MFU scaling): 0.05-0.25 (SCALE DOWN from iter7's 0.16-0.22 to fix 3-30× roofline overestimation)
- β₁ (decode memory MFU): 5.0-15.0 (10-20× amplification vs iter7's 1.1)
- β₄ (decode compute MFU): 3.0-8.0 (5-10× amplification vs iter7's 0.71)
- β₅ (MoE gating): 20-50 (same range as iter14 prediction, now that layer multiplier is fixed)
- β₈ (MoE non-compute latency): 10-40 μs/token (routing + load imbalance overhead, reduced from initial 20-80)
- β₉ (prefill batching penalty): 0.5-2.0 μs/token (heterogeneity-induced overhead)

**Causal Mechanism**: Coefficients outside these ranges indicate unphysical behavior (e.g., β₁ > 20 would suggest decode is 20× slower than memory bandwidth limit, which is impossible).

**Diagnostic Clause**: *If any coefficient is out of range after optimization:*
- β₀ > 0.3 → Dense overestimation not fully fixed (roofline still predicting too high, need deeper investigation)
- β₁ or β₄ > 20 → Basis function structurally wrong (capturing non-decode overhead)
- β₈ > 50 μs/token → MoE non-compute model double-counts FLOPs (collinearity with β₅)
- β₉ > 3.0 μs/token → Batching penalty model too aggressive (should be additive, not multiplicative)

---

## Coefficient Architecture (10 beta terms total)

**Iter15 introduces 2 new beta coefficients** (β₈, β₉), increasing from 8 to **10 total**:

| Index | Name | Role | Expected Range | Change from Iter14 |
|-------|------|------|----------------|-------------------|
| β₀ | Prefill compute MFU | Scales roofline prefill estimate | **0.05-0.25** | **SCALE DOWN** (was 0.16-0.22) |
| β₁ | Decode memory MFU | Scales roofline decode memory | **5.0-15.0** | **10× amplification** (was 1.0-1.15) |
| β₂ | TP communication | Scales TP all-reduce time | 0.15-0.25 | Unchanged |
| β₃ | KV management base | Per-request KV overhead | 0.4-1.5 ms | Unchanged |
| β₄ | Decode compute MFU | Scales roofline decode compute | **3.0-8.0** | **5× amplification** (was 0.7-0.85) |
| β₅ | MoE gating efficiency | Scales MoE gating FLOPs | 20-50 | Unchanged (fixed in iter14) |
| β₆ | Scheduler overhead | Per-request scheduler cost | 40-100 ms | Unchanged |
| β₇ | Decode per-request | Per-decode-request overhead | 15-30 ms | Unchanged |
| **β₈** | **MoE non-compute** | **Per-token routing latency** | **10-40 μs/token** | **NEW** (reduced from initial 20-80) |
| **β₉** | **Prefill batching** | **Heterogeneity penalty** | **0.5-2.0 μs/token** | **NEW** |

**Alpha coefficients unchanged** (3 total: α₀, α₁, α₂).

---

## Experimental Design

**Optimization configuration**:
- **Initialization**: Hybrid (trial 0 = physics midpoints, trials 1-1999 = TPE exploration)
- **Algorithm**: Optuna TPE (Tree-structured Parzen Estimator) — handles 10-dimensional search better than GP
- **Budget**: 2000 trials (default as of 2026-03-31, increased from 1000 to handle 10D search space)
  - Command: `python3 inner_loop_optimize.py --iteration 15`
- **Objective**: Minimize `overall_loss = RMSE[APE_TTFT] + RMSE[APE_E2E]`

**Validation approach**:
1. Run optimization (Agent 2)
2. Compare iter15 vs iter14 per-experiment APE (Agent 3)
3. Verify coefficients in expected ranges (H-robustness)
4. Verify cold-start found better minimum than iter7 warm-start (H-boundary)
5. Run ablations (remove β₁/β₄, remove β₈, remove β₉) to validate each component

---

## Success Criteria

| Metric | Iter14 (Baseline) | Iter15 Target | Threshold |
|--------|------------------|---------------|-----------|
| **Overall Loss** | 2319% | <300% | ✅ if <300% (≥87% improvement) |
| **TTFT RMSE** | 1314% | <150% | ✅ if <150% (≥89% improvement) |
| **E2E RMSE** | 1006% | <150% | ✅ if <150% (≥85% improvement) |
| **Scout avg TTFT APE** | 527% | <150% | ✅ if <150% (≥71% improvement) |
| **Dense avg TTFT APE** | 1698% | <200% | ✅ if <200% (≥88% improvement) |
| **Reasoning-lite failures** | 3/3 (100% error) | 0/3 | ✅ if all return valid results |

**Minimum viable success**: Overall loss <500% (from 2319%) with all reasoning-lite experiments returning valid results.

**Stretch goal**: Overall loss <200% (approaching iter7's 155% baseline).
