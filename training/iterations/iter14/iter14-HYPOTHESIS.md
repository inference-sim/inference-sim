# Iteration 14: Fix β₅ MoE Gating Layer Multiplier Bug

## Overview

Iteration 14 addresses the **single critical bug** identified in iter13: β₅ (MoE gating) basis function is missing a `× numMoELayers` multiplier. This caused β₅ to explode 46,800× (from 0.0411 in iter7 to 1924.4 in iter13), producing the worst iteration in training history (loss 2387%, 15.4× worse than iter7's 155%).

**Key Evidence**: β₅ = 1924.4 / 56 layers = **34.4** (within expected 1-50 range), strongly suggesting the coefficient is compensating for a missing layer multiplier.

**Strategy**: **One bug, one fix, one iteration.** No other architectural changes. Return to iter7's stable 8-beta architecture (β₀-β₅, β₆-β₇, indices 0-7), fix the β₅ bug only, and validate the fix works before adding β₈ or β₁₀ back.

---

## H-main: β₅ Layer Multiplier Fix Recovers Model Stability

**Prediction**: After adding the missing `× numMoELayers` multiplier to β₅ basis function and returning to iter7's 8-beta architecture:

**Overall Performance**:
- Overall loss: 2387% (iter13) → **<180%** (≥92% improvement, targeting iter7 baseline 155% + margin)
- TTFT RMSE: 1370% (iter13) → **<80%** (≥94% improvement)
- E2E RMSE: 1017% (iter13) → **<100%** (≥90% improvement)

**Coefficient Stability** (all within expected ranges):
- β₅: 1924.4 (iter13) → **1-50 dimensionless** (38-1924× decrease, within physical range)
  - Expected: ~20-40 (similar to other MFU scaling factors β₀, β₁, β₄)
  - Basis function now correctly computes total gating time across all MoE layers
  - Coefficient should approach 1.0 if gating efficiency (30%) is accurate

**MoE Experiments Recovery**:
- Scout general-lite: 847% TTFT (iter13) → **<100%** (≥747pp improvement)
- Scout reasoning-lite: 100% TTFT (iter13, complete failure) → **<100%** (recover from catastrophic timeout)
- Scout codegen: 559% TTFT (iter13) → **<100%** (≥459pp improvement)
- Scout roleplay: 384% TTFT (iter13) → **<90%** (≥294pp improvement)

**Non-MoE Experiments** (no β₅ contribution, should recover to iter7 baseline):
- Llama-2 codegen: 1417% TTFT (iter13) → **<15%** (recover to iter7's 9.3%)
- Mistral codegen: 1193% TTFT (iter13) → **<25%** (recover to iter7's 20%)
- Llama-3.1 codegen: 742% TTFT (iter13) → **<35%** (recover to iter7's 29%)

**Reasoning-Lite Experiments** (complete simulator failures in iter13):
- All three reasoning-lite experiments: 100% error (iter13) → **<70%** (eliminate timeout catastrophe)
  - Root cause: β₅=1924.4 caused 251× latency overestimation → all requests timed out → 0 completed requests → 100% APE
  - Expected: After fix, β₅ contribution reduces 56× → latencies become realistic → requests complete → APE recovers

**Causal Mechanism**:

**Why iter13 failed (β₅ = 1924.4 WITHOUT layer multiplier)**:

The MoE gating basis function in iter13 computed:
```
gatingFlops = 2 × totalTokens × HiddenDim × NumLocalExperts
gatingTimeSeconds = gatingFlops / (peakFlops × 0.30)
```

This computes gating time for a **single MoE layer**, but Scout has **56 MoE layers**. The optimizer inflated β₅ to 1924.4 to compensate:
- Single-layer gating time (basis function): ~0.04μs per token
- Scout needs: 56 layers × 0.04μs = **2.24μs** total gating time
- But basis function only gave 0.04μs → β₅ inflated to **1924.4** to reach 1924.4 × 0.04μs = **77μs**
- Result: **34× over-correction** (77μs vs 2.24μs needed)

**Why iter14 will succeed (β₅ ≈ 20-40 WITH layer multiplier)**:

After adding `× numMoELayers` multiplier:
```
gatingTimeSeconds = (gatingFlops × numMoELayers) / (peakFlops × 0.30)
```

This computes gating time for **all 56 Scout MoE layers** correctly:
- Multi-layer gating time (fixed basis function): 56 × 0.04μs = **2.24μs** per token
- Optimizer only needs β₅ ≈ **20-40** to match reality (not 1924.4)
- Physically plausible: β₅ ≈ 30 is similar to other efficiency factors (β₀=0.19, β₁=1.11, β₄=0.71)

**Code Citations**:

**Current (BUGGY) iter13 implementation** (`sim/latency/evolved_model.go:230-241`):
```go
// MISSING numMoELayers multiplier!
gatingFlops := 2.0 * float64(totalTokens) * float64(m.modelConfig.HiddenDim) * float64(m.modelConfig.NumLocalExperts)
moeGatingTimeSeconds = gatingFlops / tpFactor / (peakFlops * gatingEfficiency)
```

**Correct pattern from β₈ MoE routing** (`sim/latency/evolved_model.go:269-290`):
```go
// β₈ CORRECTLY multiplies by numMoELayers (line 290)
var numMoELayers float64
if m.modelConfig.InterleaveMoELayerStep > 0 {
    numMoELayers = ... // handle interleaved MoE architectures
} else {
    numMoELayers = float64(m.modelConfig.NumLayers)
}
routedTokens := numMoELayers * totalTokens * numExpertsPerTok / tpFactor
```

**Iter14 fix** (add same pattern to β₅):
```go
// Calculate numMoELayers (same as β₈, lines 269-277)
var numMoELayers float64
if m.modelConfig.InterleaveMoELayerStep > 0 {
    // Handle Scout-style interleaved MoE+dense architectures
    numMoELayers = float64(m.modelConfig.NumLayers) / (float64(m.modelConfig.InterleaveMoELayerStep) / (float64(m.modelConfig.InterleaveMoELayerStep-1) + 1e-6))
    if float64(m.modelConfig.InterleaveMoELayerStep) < numMoELayers {
        numMoELayers = float64(m.modelConfig.InterleaveMoELayerStep)
    }
} else {
    numMoELayers = float64(m.modelConfig.NumLayers)
}

// Now multiply gating time by numMoELayers
gatingFlopsPerLayer := 2.0 * float64(totalTokens) * float64(m.modelConfig.HiddenDim) * float64(m.modelConfig.NumLocalExperts)
gatingTimePerLayerSeconds := gatingFlopsPerLayer / tpFactor / (peakFlops * gatingEfficiency)
moeGatingTimeSeconds = gatingTimePerLayerSeconds * numMoELayers  // ← FIX: multiply by layers
```

**Diagnostic Clause**: *If this fails (loss >250%), it indicates one of three scenarios:*

**Scenario 1** (β₅ still explodes >200):
- **Indicates**: Gating efficiency assumption (30%) is too optimistic
- **Investigate**: Real MoE gating networks are memory-bound (small GEMMs, poor tensor core utilization)
- **Expected MFU**: 1-5% (not 30%)
- **Action**: Reduce gating efficiency 0.30 → 0.05 in iter15

**Scenario 2** (β₅ collapses <0.1):
- **Indicates**: Collinearity between β₅ (gating) and β₈ (routing) OR β₅ basis function over-corrected
- **Investigate**: Correlation analysis between β₅ and other MoE terms across trials
- **Action**: If r(β₅, other) > 0.7, reformulate β₅ to be orthogonal

**Scenario 3** (β₅ converges to 10-50 but loss >250%):
- **Indicates**: Other architectural problems (not β₅) dominate error
- **Investigate**: Per-experiment error breakdown - which experiments still fail?
- **Action**: If Scout still fails, investigate β₁₀ (batching inefficiency) in iter15; if dense models fail, investigate roofline baseline assumptions

---

## H-architecture-stability: Iter7 8-Beta Architecture Without β₈+β₁₀ Prevents Cascading Failures

**Prediction**: By returning to iter7's stable 8-beta architecture (β₀-β₅, β₆-β₇) and NOT adding β₈ (MoE routing) or β₁₀ (batching inefficiency), coefficient explosions will be eliminated:

**Coefficient Ranges** (all within physically plausible ranges):
- β₀ (prefill MFU): **0.16-0.22** (iter7: 0.191 ✓)
- β₁ (decode memory MFU): **1.00-1.15** (iter7: 1.108 ✓)
- β₂ (TP comm scaling): **0.15-0.25** (iter7: 0.185 ✓)
- β₃ (KV base overhead): **0.4-1.5ms** (iter7: 4.4ms, may decrease after β₅ fix)
- β₄ (decode compute MFU): **0.70-0.85** (iter7: 0.713 ✓)
- β₅ (MoE gating): **1-50** (iter7: 0.0411 collapsed, expect 20-40 after fix)
- β₆ (scheduler overhead): **40-100ms** (iter7: 13.2ms, may increase slightly)
- β₇ (decode per-request): **15-30ms** (iter7: 26.3ms ✓)

**No Cascading Explosions**:
- Iter9-13 pattern: Adding β₈+β₁₀ → β₅ explodes → cascading failures in α₁, β₂, β₄, β₆, β₈
- Iter14 avoids this by NOT adding β₈+β₁₀ until β₅ is proven stable

**Causal Mechanism**:

Iter9-13 demonstrated that **adding new coefficients to an architecture with a broken β₅ causes catastrophic instability**:
- Iter9: Added β₉ (FP8) → β₆ +654%, β₂ +343%, loss 161%
- Iter10: Added β₁₀ + β₃' → β₅ exploded, loss 4267%
- Iter11: Same as iter10 (audited basis functions, confirmed correct) → loss 4084%
- Iter12: Widened β₃' bounds → β₃' collapsed, loss 2590%
- Iter13: Returned to iter7 + β₈ + β₁₀ → β₅ exploded 46,800×, loss 2387%

**Pattern**: ANY addition to iter7 baseline triggered β₅ explosion when β₅ basis function was broken.

**Why iter14 will break this pattern**:
1. **Fix β₅ basis function FIRST** → β₅ converges to 20-40 (physically plausible)
2. **Validate stability** → All 8 coefficients in expected ranges
3. **THEN add complexity** → β₈, β₁₀ can be added in iter15+ without triggering explosions

**Evidence from iter7**: With β₅=0.0411 (collapsed but stable), iter7 achieved 155% loss with 6/8 coefficients in expected ranges. After β₅ fix, expect 7-8/8 coefficients stable.

**Diagnostic Clause**: *If this fails (>3 coefficients out of range), it indicates:*
- **Scenario 1**: β₅ fix insufficient (still broken) → revisit efficiency assumption or remove β₅ entirely
- **Scenario 2**: Warm-start from iter7 invalid (dataset changed between iter7/iter13) → cold-start optimization from random initialization
- **Scenario 3**: Other collinearities exist (β₀ vs β₁, β₂ vs β₃) → reformulate basis functions for orthogonality

---

## H-reasoning-lite-recovery: Fixing β₅ Eliminates 100% Timeout Catastrophe

**Prediction**: All three reasoning-lite experiments that failed with exactly 100% error in iter13 will recover to <70% error:

**Experiments**:
- Scout reasoning-lite-2-1: 100% TTFT/E2E (iter13) → **<70%** (recover from timeout)
- Qwen2.5 reasoning-lite-1-1: 100% TTFT/E2E (iter13) → **<65%** (recover from timeout)
- Llama-2 reasoning-lite-1-1: 100% TTFT/E2E (iter13) → **<70%** (recover from timeout)

**Causal Mechanism**:

**Why iter13 produced 100% error** (DEBUGGED via BLIS simulation):

1. **β₅=1924.4 caused catastrophic latency overestimation**:
   - Per-token MoE gating contribution: 1924.4 × 0.04μs = **77μs per token per layer**
   - For 56-layer Scout: 77μs × 56 = **4.3ms per token** (vs ~0.017ms actual)
   - For reasoning-lite (934 prefill + 1448 decode = 2382 tokens): 4.3ms × 2382 = **10,268ms** predicted
   - Ground truth E2E: 40.835ms → **251× underestimate** (predicted 10.3s vs actual 40ms)

2. **All requests timed out before completion**:
   - BLIS simulation: 23 requests injected, **0 completed** (all timed out due to massive latency prediction)
   - Summary metrics calculated from completed requests only
   - Result: `e2e_mean_ms = 0`, `ttft_mean_ms = 0` (no data to compute mean from)

3. **APE calculation produced exactly 100%**:
   - `APE = |predicted - actual| / actual × 100`
   - `E2E APE = |0 - 40.835| / 40.835 × 100 = 100%`
   - `TTFT APE = |0 - 0.138| / 0.138 × 100 = 100%`

**Why iter14 will fix this** (β₅ ≈ 30, WITH layer multiplier):

1. **β₅≈30 produces realistic latency estimates**:
   - Per-token MoE gating contribution: 30 × (0.04μs × 56 layers) = 30 × 2.24μs = **67.2μs per token** (vs 4300μs in iter13)
   - For reasoning-lite (2382 tokens): 67.2μs × 2382 = **160ms** predicted (vs 10,268ms in iter13)
   - Ground truth E2E: 40.835ms → **3.9× overestimate** (reasonable, not 251×)

2. **Requests will complete successfully**:
   - Predicted latencies 64× lower (160ms vs 10,268ms)
   - Requests will complete within timeout window
   - Summary metrics will be computed from actual data (not zeros)

3. **APE will reflect real prediction error** (not catastrophic timeout):
   - `E2E APE = |160 - 40.835| / 40.835 × 100 ≈ 292%` (overestimate, but not 100% timeout)
   - TTFT APE likely similar (50-70% range, consistent with other reasoning-lite experiments)

**Code Citations**: See H-main for β₅ fix implementation details.

**Diagnostic Clause**: *If this fails (any reasoning-lite still returns 100% error), it indicates:*

**Scenario 1** (all three still 100%):
- **Indicates**: β₅ fix insufficient OR other numerical overflow in evolved model
- **Investigate**: Add defensive guards to catch negative/overflow StepTime values before they cause timeout catastrophe
- **Action**: Add logging: `if stepTime < 0 || stepTime > 1e12 { log error, clamp value }`

**Scenario 2** (1-2 still 100%, others recover):
- **Indicates**: Data quality issue in specific reasoning-lite experiments
- **Investigate**: Validate ground truth `per_request_lifecycle_metrics.json` for failing experiments
- **Action**: If data corrupted, exclude experiment or re-collect ground truth

**Scenario 3** (all recover to 30-70%):
- **Indicates**: Success! β₅ fix eliminated timeout catastrophe
- **Action**: Proceed to iter15 with confidence, consider adding β₈ or β₁₀ back

---

## H-non-moe-recovery: Dense Model Experiments Recover to Iter7 Baseline

**Prediction**: Non-MoE experiments (no β₅ contribution) will recover to iter7 baseline performance, confirming β₅ was the sole cause of iter13 catastrophe:

**Dense Model Experiments** (β₅ contribution = 0 since NumLocalExperts = 1):
- Llama-2 codegen: 1417% TTFT (iter13) → **<15%** (recover to iter7's 9.3%)
- Llama-2 roleplay: 1123% TTFT (iter13) → **<60%** (recover to iter7's 56%)
- Llama-2 general: 1513% TTFT (iter13) → **<10%** (recover to iter7's 4.6%)
- Mistral codegen: 1193% TTFT (iter13) → **<25%** (recover to iter7's 20%)
- Mistral general-lite: 3965% TTFT (iter13) → **<95%** (recover to iter7's 90%)
- Qwen2.5 roleplay: 1137% TTFT (iter13) → **<60%** (recover to iter7's 57%)
- Yi-34B general-lite: 1183% TTFT (iter13) → **<50%** (recover to iter7's 48%)
- Llama-3.1 codegen: 742% TTFT (iter13) → **<35%** (recover to iter7's 29%)
- Llama-3.1 general-lite: 1010% TTFT (iter13) → **<45%** (recover to iter7's 41%)

**Causal Mechanism**:

Dense models (Llama-2, Llama-3.1, Mistral, Qwen2.5, Yi-34B) have `NumLocalExperts = 1`, so the MoE gating code path is **skipped entirely**:

```go
// sim/latency/evolved_model.go:230
if m.modelConfig.NumLocalExperts > 1 {
    // β₅ MoE gating computation
    // Dense models NEVER enter this block!
}
moeGatingContribution := m.Beta[5] * moeGatingTimeUs  // moeGatingTimeUs = 0 for dense models
```

**Why iter13 failed for dense models** (even though β₅ doesn't affect them):

Iter13's catastrophic β₅ explosion caused **cascading coefficient instability** that affected ALL experiments:
- α₁ (tokenization): 118μs (iter7) → **54.3μs** (iter13, -54%, below physical range)
- β₂ (TP comm): 0.185 (iter7) → **0.362** (iter13, +96% explosion)
- β₄ (decode compute): 0.713 (iter7) → **0.562** (iter13, -21%, below range)
- β₆ (scheduler): 13.2ms (iter7) → **10.0ms** (iter13, -24%, below range)

**How this happens**: The optimizer saw massive error from MoE experiments (β₅ explosion) and tried to compensate by adjusting **global coefficients** (α₁, β₂, β₄, β₆) that affect all experiments. Result: Dense models also failed, even though they don't use β₅.

**Why iter14 will fix dense models**:

After β₅ fix:
1. **MoE experiments stabilize** → β₅ converges to 20-40 (not 1924.4)
2. **Optimizer no longer compensates globally** → α₁, β₂, β₄, β₆ return to iter7 values
3. **Dense models recover** → Same architecture as iter7, same coefficients, same performance

**Expected coefficient recovery**:
- α₁: 54.3μs (iter13) → **100-120μs** (recover to iter7's 118μs)
- β₂: 0.362 (iter13) → **0.18-0.20** (recover to iter7's 0.185)
- β₄: 0.562 (iter13) → **0.70-0.75** (recover to iter7's 0.713)
- β₆: 10.0ms (iter13) → **13-15ms** (recover to iter7's 13.2ms)

**Diagnostic Clause**: *If this fails (dense models don't recover to within 10pp of iter7), it indicates:*

**Scenario 1** (all dense models still 5-10× worse than iter7):
- **Indicates**: Other coefficients (β₀, β₁, β₃, β₇) out of range, not just β₅ cascade
- **Investigate**: Compare iter14 vs iter7 coefficients for all 8 betas
- **Action**: If warm-start from iter7 failed, try cold-start optimization from random initialization

**Scenario 2** (some dense models recover, others don't):
- **Indicates**: Dataset change between iter7/iter13 (reasoning → reasoning-lite) affects recovery
- **Investigate**: Which models fail? Compare workload characteristics
- **Action**: If reasoning-lite fails universally, consider reverting to original reasoning dataset

**Scenario 3** (all dense models recover to iter7 ±10pp):
- **Indicates**: **Success!** β₅ was the sole cause of cascading failures
- **Action**: Proceed to iter15 with confidence, β₅ fix validated across dense and MoE models

---

## Summary of Hypothesis Bundle

| Hypothesis | Type | Key Prediction | Success Threshold | Diagnostic Scenario |
|------------|------|----------------|-------------------|---------------------|
| **H-main** | Bug fix validation | β₅: 1924.4 → 1-50 after adding layer multiplier | Overall loss <180%, β₅ in range 1-50 | If β₅ >200: gating efficiency wrong (30% → 5%) |
| **H-architecture-stability** | Architectural validation | All 8 coefficients in expected ranges | ≥7/8 coefficients stable | If <5/8 stable: other collinearities exist |
| **H-reasoning-lite-recovery** | Catastrophic failure fix | 100% timeout errors → <70% real errors | 0/3 experiments return 100% | If still 100%: add defensive guards |
| **H-non-moe-recovery** | Cascade isolation | Dense models recover to iter7 baseline | ≥8/9 dense experiments within 10pp of iter7 | If don't recover: warm-start failed |

**Overall Success Criteria**:
- Loss <180% (iter7 baseline 155% + 25pp margin for dataset shift)
- β₅ converges to 1-50 (physically plausible)
- Zero 100% timeout errors (reasoning-lite recovers)
- Dense models recover to iter7 ±10pp (proves β₅ was sole cascade cause)

If ALL four hypotheses confirmed → Iter14 **validates β₅ fix**, proceed to iter15 to add β₈+β₁₀ back.

If ANY hypothesis refuted → Investigate diagnostic scenarios before proceeding.
