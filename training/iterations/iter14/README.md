# Iteration 14: Fix β₅ MoE Gating Layer Multiplier Bug

## Quick Summary

**Strategy**: One bug, one fix, one iteration.

**Critical Bug Fixed**: Added missing `× numMoELayers` multiplier to β₅ (MoE gating) basis function.

**Architecture**: Returned to iter7's stable 8-beta baseline (β₀-β₇), removed β₈ and β₁₀ until β₅ proven stable.

**Expected Outcome**: Loss 2387% (iter13) → <180% (targeting iter7 baseline 155% + margin).

---

## The Bug

**Iter13 Catastrophic Failure**: Loss 2387%, 15.4× worse than iter7's 155%, worst iteration in training history.

**Root Cause**: β₅ (MoE gating) basis function was computing gating time for a **single MoE layer** but Scout has **56 MoE layers**.

**Evidence**:
- β₅ exploded from 0.0411 (iter7) to 1924.4 (iter13) — 46,800× increase
- 1924.4 / 56 layers = **34.4** (within expected 1-50 range for a per-layer coefficient!)
- The β₈ (MoE routing) code already correctly multiplied by `numMoELayers` (line 290)
- But β₅ (MoE gating) was missing this critical multiplier (lines 230-238)

**Impact**:
- Three reasoning-lite experiments returned 100% error (request timeouts due to 251× latency overestimation)
- All experiments catastrophically failed (0/15 success rate)
- Dense models also failed due to cascading coefficient instability

---

## The Fix

**Location**: `sim/latency/evolved_model.go`, lines 238-271

**Before (iter13, BUGGY)**:
```go
// MISSING numMoELayers multiplier!
gatingFlops := 2.0 * float64(totalTokens) * float64(m.modelConfig.HiddenDim) * float64(m.modelConfig.NumLocalExperts)
moeGatingTimeSeconds = gatingFlops / tpFactor / (peakFlops * gatingEfficiency)
```

**After (iter14, FIXED)**:
```go
// Calculate numMoELayers (same pattern as β₈ MoE routing)
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

// Compute gating FLOPs per layer
gatingFlopsPerLayer := 2.0 * float64(totalTokens) * float64(m.modelConfig.HiddenDim) * float64(m.modelConfig.NumLocalExperts)
gatingTimePerLayerSeconds := gatingFlopsPerLayer / tpFactor / (peakFlops * gatingEfficiency)

// **CRITICAL FIX**: Multiply by numMoELayers to account for ALL MoE layers
moeGatingTimeSeconds = gatingTimePerLayerSeconds * numMoELayers
```

**Key Change**: Added `× numMoELayers` multiplication on final line, copying the exact pattern from β₈ (MoE routing).

---

## Architecture Changes

**Returned to Iter7 Baseline** (8 beta coefficients):

| Coefficient | Description | Expected Value (Iter14) | Iter7 Value | Status |
|-------------|-------------|-------------------------|-------------|--------|
| β₀ | Prefill compute MFU | 0.16-0.22 | 0.191 | ✓ |
| β₁ | Decode memory MFU | 1.00-1.15 | 1.108 | ✓ |
| β₂ | TP comm scaling | 0.15-0.25 | 0.185 | ✓ |
| β₃ | KV base overhead | 0.4-1.5ms | 4.4ms | May decrease |
| β₄ | Decode compute MFU | 0.70-0.85 | 0.713 | ✓ |
| β₅ | MoE gating | **1-50** | 0.0411 | **FIX TARGET** |
| β₆ | Scheduler overhead | 40-100ms | 13.2ms | May increase |
| β₇ | Decode per-request | 15-30ms | 26.3ms | ✓ |

**Removed (temporarily)**:
- ❌ β₈ (MoE routing, from iter8/iter13) — will add back in iter15 after β₅ proven stable
- ❌ β₁₀ (batching inefficiency, from iter9/iter13) — will add back in iter15 after β₅ proven stable

**Rationale**: ANY addition to iter7 baseline triggered β₅ explosion when basis function broken. Must fix β₅ FIRST, validate stability, THEN add complexity.

---

## Hypothesis Bundle

### H-main: β₅ Layer Multiplier Fix Recovers Model Stability

**Prediction**: After adding the missing `× numMoELayers` multiplier:
- Overall loss: 2387% → **<180%** (≥92% improvement)
- β₅: 1924.4 → **1-50 dimensionless** (38-1924× decrease)
- MoE experiments: Scout general-lite 847% → <100% TTFT
- Reasoning-lite: 100% timeout errors → <70% real errors
- Dense models: Recover to iter7 ±10pp

**Causal Mechanism**: Without layer multiplier, basis function computed ~0.04μs per token (single layer). Scout has 56 MoE layers, so optimizer inflated β₅ to 1924.4 to compensate. After fix, basis function correctly computes 56 × 0.04μs = 2.24μs, allowing β₅ to converge to 20-40 (physically plausible).

### H-architecture-stability: Iter7 8-Beta Architecture Prevents Cascading Failures

**Prediction**: All 8 coefficients in expected ranges (≥7/8 stable), no cascading explosions like iter9-13.

**Causal Mechanism**: Iter9-13 demonstrated that adding new coefficients to architecture with broken β₅ causes catastrophic instability. Iter14 avoids this by NOT adding β₈+β₁₀ until β₅ proven stable.

### H-reasoning-lite-recovery: Fixing β₅ Eliminates 100% Timeout Catastrophe

**Prediction**: All three reasoning-lite experiments that failed with 100% error in iter13 will recover to <70% error.

**Causal Mechanism**: Iter13's β₅=1924.4 caused 251× latency overestimation → all requests timed out → 0 completed requests → 100% APE. After fix, β₅≈30 produces realistic latencies → requests complete → APE reflects real prediction error.

### H-non-moe-recovery: Dense Model Experiments Recover to Iter7 Baseline

**Prediction**: Dense models (no β₅ contribution) recover to iter7 ±10pp, proving β₅ was sole cascade cause.

**Causal Mechanism**: Iter13's β₅ explosion caused cascading coefficient instability (α₁ -54%, β₂ +96%, β₄ -21%, β₆ -24%) that affected ALL experiments. After β₅ fix, global coefficients return to iter7 values, dense models recover.

---

## Validation Checklist

Before accepting iter14 results:

**β₅ Convergence**:
- [ ] β₅ converges to 1-50 (physically plausible)
- [ ] β₅ does NOT explode >100 (bounds prevent this)
- [ ] β₅ does NOT collapse <0.1 (indicates collinearity or over-correction)

**Overall Performance**:
- [ ] Overall loss <180% (iter7 baseline 155% + margin)
- [ ] TTFT RMSE <80% (iter13: 1370%)
- [ ] E2E RMSE <100% (iter13: 1017%)

**Experiment Recovery**:
- [ ] Zero 100% timeout errors (reasoning-lite recovers)
- [ ] MoE experiments: Scout general-lite <100% TTFT (iter13: 847%)
- [ ] Dense models: ≥8/9 within 10pp of iter7

**Coefficient Stability**:
- [ ] ≥7/8 coefficients in expected ranges
- [ ] No cascading explosions (β₂, β₄, β₆ stable)
- [ ] α₁ recovers to 100-120μs (iter13: 54.3μs)

---

## Expected Outcomes

### Success Scenario (β₅ = 20-40, loss <180%)

**What this validates**:
- ✅ β₅ basis function bug was the sole cause of iter9-13 catastrophic failures
- ✅ Iter7 architecture is stable and can serve as foundation for future improvements
- ✅ Ready to add β₈ (MoE routing) and β₁₀ (batching inefficiency) in iter15

**Next steps**:
1. Proceed to iter15: Add β₈ back (proven real mechanism from iter8)
2. If iter15 stable: Add β₁₀ back in iter16 (addressing sequence-length bottleneck)
3. Target: Loss <140% by iter16 (iter7 baseline - 15pp)

### Failure Scenarios

**Scenario 1: β₅ still explodes >200**
- **Indicates**: Gating efficiency assumption (30%) too optimistic
- **Action**: Reduce gating efficiency 0.30 → 0.05 in iter15 (MoE gating is memory-bound, small GEMMs with poor tensor core utilization)

**Scenario 2: β₅ collapses <0.1**
- **Indicates**: Collinearity OR basis function over-corrected
- **Action**: Check correlation between β₅ and other coefficients; if r > 0.7, reformulate β₅ for orthogonality

**Scenario 3: β₅ converges to 10-50 but loss >250%**
- **Indicates**: Other architectural problems (not β₅) dominate error
- **Action**: Investigate per-experiment error breakdown; if Scout still fails, re-examine β₁₀; if dense models fail, re-examine roofline baseline

**Scenario 4: Dense models don't recover to iter7**
- **Indicates**: Dataset change (reasoning → reasoning-lite) affects recovery OR warm-start failed
- **Action**: Try cold-start optimization from random initialization

---

## Files Modified

1. **`sim/latency/evolved_model.go`**:
   - Fixed β₅ basis function: added `× numMoELayers` multiplier (lines 238-271)
   - Reverted to iter7 8-beta architecture (removed β₈, β₁₀)
   - Updated file header comment with iter14 strategy
   - Updated NewEvolvedModel validation to expect 8 beta coefficients

2. **`iterations/iter14/iter14-HYPOTHESIS.md`**:
   - H-main: β₅ layer multiplier fix recovers model stability
   - H-architecture-stability: Iter7 8-beta prevents cascading failures
   - H-reasoning-lite-recovery: Fixing β₅ eliminates 100% timeout catastrophe
   - H-non-moe-recovery: Dense models recover to iter7 baseline

3. **`iterations/iter14/iteration_manifest.yaml`**:
   - Declares backend="evolved" (constant across iterations)
   - Lists modified files
   - Documents reasoning for iter14 changes

4. **`iterations/iter14/coefficient_bounds.yaml`**:
   - Warm-start from iter7 optimal coefficients
   - Tightened β₅ bounds: [0, 2000] → [0, 100] to prevent future explosions
   - All 8 beta coefficients with initial values and bounds

---

## Key Insights

1. **The Power of Evidence-Based Debugging**: β₅ = 1924.4 / 56 = 34.4 (within expected 1-50 range) was the smoking gun that led directly to the fix.

2. **Pattern Matching Across Codebase**: β₈ (MoE routing) already had the correct `× numMoELayers` pattern. Copying this pattern to β₅ was the one-line fix.

3. **One Bug, One Fix**: Resisting the temptation to add β₈+β₁₀ back immediately. Must validate β₅ fix works before adding complexity.

4. **Warm-Start Strategy**: Using iter7 optimal coefficients (not iter13's inflated values) provides stable starting point for optimization.

5. **Defensive Bounds**: Tightening β₅ bounds to [0, 100] prevents future explosions even if basis function has other issues.

---

## What Success Looks Like

**Iteration 14 succeeds if**:
- Loss drops from 2387% to <180% (10-20× improvement)
- β₅ converges to 1-50 (not 1924.4)
- Zero 100% timeout errors
- Dense models recover to iter7 ±10pp
- All 8 coefficients in expected ranges

**If successful**, iteration 14 will have:
- Validated the β₅ bug diagnosis
- Restored model stability to iter7 baseline
- Unlocked path forward to add β₈+β₁₀ in iter15-16
- Proved that "one bug, one fix" strategy works for catastrophic failures

**Bottom Line**: Iter14 is not about achieving best-ever performance. It's about **fixing the foundation** so that future iterations can build on solid ground.
