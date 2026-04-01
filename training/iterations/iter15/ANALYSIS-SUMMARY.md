# Iteration 15: Analysis Summary

## Executive Summary

**Iteration 15 FAILED catastrophically** - loss INCREASED from 2319% (iter14) to **6538%** (182% worse).

The iteration attempted a "three-axis correction" strategy:
1. Decode amplification (β₁, β₄ at 5-15×, 3-8×)
2. MoE non-compute term (β₈, NEW)
3. Prefill batching penalty (β₉, NEW)

**What actually happened**:
- ✅ Decode amplification was USED (β₁=6.4, β₄=6.5) - helped reasoning-lite (E2E 75-180%)
- ❌ MoE non-compute was REJECTED (β₈≈0) - Scout errors INCREASED (527% → 1068%)
- ❌ Prefill batching was REJECTED (β₉≈0) - Dense errors remain catastrophic (1300-4000%)

## Hypothesis Validation Results

| Hypothesis | Verdict | Key Finding |
|------------|---------|-------------|
| **H-main** (Three-axis correction) | ❌ REJECTED | Loss increased 182%, all metrics worse |
| **H-ablation-decode** | ⚠️ PARTIAL | Decode amplification helps decode-heavy, but can't fix prefill |
| **H-ablation-moe** | ❌ REJECTED | β₈ collapsed to 0, MoE FLOPs likely wrong |
| **H-ablation-batching** | ❌ REJECTED | β₉ collapsed to 0, batch heterogeneity not the issue |
| **H-boundary** (cold-start) | ❌ REJECTED | Cold-start 4219pp worse than warm-start |
| **H-error-pattern** | ⚠️ PARTIAL | Only reasoning-lite improved (30-668%), Scout/dense got worse |
| **H-robustness** | ⚠️ PARTIAL | 6/10 coefficients in range, but 4/10 collapsed (β₃,β₆,β₇,β₈,β₉) |

**Overall**: 0/7 confirmed, 3/7 partial, 4/7 rejected.

## Root Cause: Scaling Broken Formulas Doesn't Fix Them

The fundamental error in iter15 was attempting to **FIX roofline estimates by scaling them** (β × roofline_term), rather than **REPLACING them with vLLM-accurate formulas**.

**Evidence**:
1. **Prefill**: β₀=0.092 scales down roofline by 11×, but dense TTFT is still 13-40× wrong → base roofline is 140-440× off
2. **MoE**: β₈ (non-compute correction) rejected by optimizer → MoE FLOPs calculation itself is wrong
3. **Batching**: β₉ (heterogeneity penalty) rejected by optimizer → dense overestimation is NOT about batching

**You cannot fix a broken formula by multiplying it by a constant.**

## Critical Learnings

### ✅ What Worked

**Decode amplification helps decode-heavy workloads**:
- Reasoning-lite E2E APE: 75-180% (vs 100% timeout in iter14)
- β₁=6.4, β₄=6.5 amplify decode time → prevents underestimation for long outputs
- **Limitation**: Only helps when decode DOMINATES total latency (256-512 output tokens)

### ❌ What Failed

**1. Roofline-based prefill model is fundamentally broken**:
- Dense TTFT APE: 1300-4000% (13-40× too fast)
- β₀=0.092 provides 11× scale-down, still insufficient
- Pattern: Shorter prompts (64 tokens) have WORSE errors than longer prompts (512 tokens)
- Root cause: Base roofline prefill calculation (FLOPs or memory formula) is orders of magnitude wrong

**2. MoE non-compute hypothesis was wrong**:
- β₈ collapsed to 0 (optimizer rejected it)
- Scout errors INCREASED from 527% (iter14) to 1068% (iter15)
- Alternative hypothesis: MoE FLOPs calculation is wrong (active vs total experts? load imbalance in FLOPs?)

**3. Prefill batching hypothesis was wrong**:
- β₉ collapsed to 0 (optimizer rejected it)
- Dense roleplay (low heterogeneity) has WORSE errors than dense codegen (high heterogeneity)
- Alternative hypothesis: Dense overestimation is due to wrong BASE prefill FLOPs, not batching

**4. Cold-start in 10D space is inefficient**:
- Loss 6538% (cold-start) vs 2319% (iter14 warm-start) - 2.8× worse
- 2000 trials in 10D → only 200 per dimension (sparse coverage)
- Optimizer rejected 5/10 new coefficients (β₃,β₆,β₇,β₈,β₉)

## Recommendations for iter16

### Priority 1: Fix Basis Functions (MUST DO)

**Stop scaling roofline. Start profiling vLLM.**

1. **Profile real vLLM prefill latency**:
   - Vary: batch size, sequence length, TP
   - Extract: Empirical formula `prefill_time = f(batch, seq_len, TP, params)`
   - **Replace** `β₀ × roofline_prefill` with empirical model

2. **Profile real vLLM decode latency**:
   - Vary: batch size, KV cache size, TP
   - Check: Is decode time linear in roofline estimate? Batch size dependent?
   - **Validate** β₁, β₄ functional forms

3. **Investigate MoE FLOPs calculation**:
   - Check: Are active vs total experts counted correctly?
   - Profile: Scout vs equivalent dense model (TTFT ratio)
   - **Fix** MoE expert FLOPs formula in roofline

### Priority 2: Reduce Dimensionality

**Remove optimizer-rejected terms**:
- β₃ (KV mgmt): 0.001 (expected 0.4-1.5 ms)
- β₆ (scheduler): 0.042 (expected 40-100 ms)
- β₇ (decode per-req): 0.016 (expected 15-30 ms)
- β₈ (MoE non-compute): 3.7e-5 (expected 10-40 μs)
- β₉ (prefill batching): 7.2e-7 (expected 0.5-2.0 μs)

**Result**: 10 → 5 beta coefficients (β₀, β₁, β₂, β₄, β₅)

**Test collinearity**:
- β₁ (decode mem) and β₄ (decode comp) both ≈6.5
- If collinear, combine into β_decode → 4 coefficients

### Priority 3: Use Warm-Start (Once Basis Functions Fixed)

- Start from iter7 coefficients (β₀=0.191, β₁=1.108, ...)
- Use 1000 trials (not 2000) - more sample-efficient
- Add physics priors from vLLM profiling (not physics midpoints)

## Error Patterns

### Catastrophic Failures (TTFT > 4000%)
- **Dense roleplay**: 4124-4151% TTFT, 9711-11357% E2E
- Characteristic: Short prompts (64 tokens), long outputs (128 tokens)
- Decode-heavy, but TTFT measures PREFILL → prefill basis functions catastrophically wrong

### High Errors (TTFT 1000-2000%)
- **Dense codegen**: 1437-1847% TTFT
- **Scout MoE**: 708-1634% TTFT (avg 1068%)
- Pattern: Scout errors INCREASED from iter14 (527%) despite β₈ term

### Success (TTFT < 100%)
- **Scout reasoning-lite** (exp_48): 30% TTFT, 27% E2E
- Only experiment with <100% APE across all metrics
- Characteristic: Balanced workload (512 input, 256 output, MoE, TP=2)

### Partial Success (E2E < 200%)
- **Reasoning-lite** (dense): 75-180% E2E, but 238-668% TTFT
- Decode amplification helps E2E (long outputs), but TTFT (prefill) still wrong

## Coefficient Analysis Summary

| Coefficient | Value | Expected | Status | Interpretation |
|-------------|-------|----------|--------|----------------|
| β₀ | 0.092 | 0.05-0.25 | ✅ IN RANGE | Prefill MFU: 11× scale-down, insufficient |
| β₁ | 6.398 | 5.0-15.0 | ✅ IN RANGE | Decode mem MFU: 6.4× amplification, helps reasoning-lite |
| β₂ | 0.207 | 0.15-0.25 | ✅ IN RANGE | TP comm: 20.7% of theoretical |
| β₃ | 0.001 | 0.4-1.5 ms | ❌ COLLAPSED | KV mgmt: REMOVE for iter16 |
| β₄ | 6.471 | 3.0-8.0 | ✅ IN RANGE | Decode comp MFU: 6.5× amplification, test collinearity with β₁ |
| β₅ | 33.569 | 20-50 | ✅ IN RANGE | MoE gating: 33× overhead, but Scout still fails |
| β₆ | 0.042 | 40-100 ms | ❌ COLLAPSED | Scheduler: REMOVE for iter16 |
| β₇ | 0.016 | 15-30 ms | ❌ COLLAPSED | Decode per-req: REMOVE for iter16 |
| β₈ | 3.7e-5 | 10-40 μs | ❌ REJECTED | MoE non-compute: REMOVE for iter16 |
| β₉ | 7.2e-7 | 0.5-2.0 μs | ❌ REJECTED | Prefill batching: REMOVE for iter16 |

**6/10 in range, but 4/10 collapsed. Remove collapsed terms → 5 coefficients.**

## Key Insights for Future Iterations

1. **Roofline is a guide, not a model**: Use roofline to understand bottlenecks, but don't use `roofline_time` as a basis function directly
2. **Profile before modeling**: Iter15 added β₈, β₉ based on physics intuition, but vLLM rejected both → profile first
3. **Functional form matters more than magnitude**: β₀=0.092 is "physically plausible" but model still fails → wrong formula, not wrong scaling
4. **One success is valuable**: exp_48 (Scout reasoning-lite, 30% TTFT) is the ONLY <100% result → analyze what makes it work
5. **Optimizer feedback is reliable**: When optimizer pushes coefficient to 0 (β₈, β₉), trust it → hypothesis was wrong

## Cross-Validation Status

**CV tests NOT run** - iteration failed catastrophically (loss 6538%, target <300%).

**Criteria not met**:
- ❌ All hypotheses confirmed: 0/7 confirmed
- ❌ Overall loss < 80%: 6538% (82× above threshold)
- ❌ No experiment APE > 100%: 14/15 experiments have TTFT > 100%
- ⚠️ Coefficients plausible: 6/10 in range, 4/10 collapsed

**Recommendation**: Fix basis functions (Priority 1) before running CV.

## Files Generated

1. **iter15-HYPOTHESIS-validation.md**: Detailed validation of all 7 hypotheses with evidence, verdicts, causal analysis
2. **iter15-FINDINGS.md**: Principles extracted, coefficient analysis, recommendations for iter16
3. **ANALYSIS-SUMMARY.md**: This executive summary

## Next Steps

1. **Agent 1 (Design)**: Read FINDINGS.md → design iter16 with:
   - vLLM prefill profiling → empirical prefill formula
   - MoE FLOPs investigation → fix expert calculation
   - Remove collapsed terms (β₃,β₆,β₇,β₈,β₉) → 5 coefficients
2. **Agent 2 (Optimization)**: Run iter16 with warm-start from iter7, 1000 trials
3. **Agent 3 (Analysis)**: Validate iter16 hypotheses, check if prefill profiling helps

**Expected iter16 outcome**: If prefill and MoE FLOPs are fixed, loss should decrease from 6538% to <500% (10× improvement).
