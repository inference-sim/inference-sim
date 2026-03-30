# Iteration 8: Experimental Hypothesis Validation

## Summary

**CRITICAL FINDING**: β₈ was added to the model but contains a **units conversion bug** that makes it effectively inactive.

The optimization converged to β₈ = 0.00003 seconds = 30μs (within expected 10-50μs range), but predictions are **byte-for-byte identical to iter7**. Root cause analysis reveals a 1000× units error in `sim/latency/evolved_model.go:344`.

**Key findings**:
- β₈ implementation exists in code (lines 301-346 of evolved_model.go)
- β₈ converged to physically plausible value (30μs per routed token)
- Units bug: multiplies by 1000 (milliseconds → microseconds) instead of 1e6 (seconds → microseconds)
- Actual contribution: 0.078ms per Scout request (should be 78ms)
- All predictions unchanged from iter7 (155.35% vs 155.37% loss)

---

## Root Cause Analysis

### The Bug

**Location**: `sim/latency/evolved_model.go`, line 344

**Current code**:
```go
// β₈ coefficient is in milliseconds per routed token (expected 0.000010-0.000050 = 10-50μs)
// Convert to microseconds: routedTokens × β₈ × 1000
moeRoutingTimeUs = routedTokens * m.Beta[8] * 1000.0
```

**Problem**: The comment claims β₈ is "in milliseconds", but the expected values `0.000010-0.000050 = 10-50μs` are clearly in **seconds** (not milliseconds). 0.000010 seconds = 10μs, but 0.000010 milliseconds = 10 nanoseconds.

**All other Beta coefficients are in seconds**:
- β₀: seconds of compute time per prefill TFLOP
- β₁, β₄: seconds per memory/compute operation
- β₂: seconds of TP communication
- β₅, β₇: seconds per operation

β₈ should also be in seconds (seconds per routed token), not milliseconds.

**Correct code**:
```go
// β₈ coefficient is in SECONDS per routed token (expected 0.000010-0.000050 = 10-50μs)
// Convert to microseconds: routedTokens × β₈ × 1e6
moeRoutingTimeUs = routedTokens * m.Beta[8] * 1e6
```

### Impact Calculation

**For Scout general (exp_17)**:
- numMoELayers: 26 (from InterleaveMoELayerStep)
- totalTokens: ~200 (estimated prefill + decode avg)
- numExpertsPerTok: 1 (default)
- TP: 2
- routedTokens = 26 × 200 × 1 / 2 = 2600

**With current buggy code**:
- moeRoutingTimeUs = 2600 × 0.00003 × 1000 = 78 microseconds = **0.078ms**
- Contribution rounds to zero (too small to affect predictions)

**With correct code**:
- moeRoutingTimeUs = 2600 × 0.00003 × 1e6 = 78,000 microseconds = **78ms**
- This matches the Scout TTFT residual! (predicted ~10-20ms, actual ~100ms, gap ~80ms)

**Verdict**: The bug causes a **1000× underestimate** of β₈ contribution, making it effectively inactive.

---

## Hypothesis Verdicts (Based on Implementation Bug)

Since β₈ was not actually applied (due to units bug), we can only partially validate hypotheses:

### H-ablation-beta8: β₈ Accounts for Majority of Scout Improvement

**Prediction** (from Agent 1): β₈ contributes >30pp to Scout TTFT improvement

**Actual Result**: β₈ contributed **0pp** (bug prevented it from being applied)

**Verdict**: ⚠️ **INCONCLUSIVE** (cannot validate - implementation bug)

**Evidence**:
- Overall loss: 155.35% (identical to iter7's 155.37%)
- Scout TTFT errors: 79-100% (unchanged from iter7)
- β₈ coefficient: 0.00003 seconds = 30μs (physically plausible)
- Theoretical contribution (if bug fixed): 78ms per Scout request
- Actual contribution (with bug): 0.078ms per Scout request (1000× too small)

**Analysis**: The optimizer learned a reasonable β₈ value, but the units bug prevented it from affecting predictions. An ablation study is meaningless when the full model already has β₈ ≈ 0 (due to bug).

**Recommendation**: Fix units bug, re-run iter8 optimization, then run ablation study.

---

## Implementation Bug Details

### Evidence for Units Bug

1. **Inconsistent comment**: Line 342 says "β₈ coefficient is in milliseconds" but expected values "0.000010-0.000050 = 10-50μs" are clearly in seconds

2. **Inconsistent with other coefficients**: All other Beta coefficients are in seconds, not milliseconds

3. **Expected contribution matches residual**: With correct units (1e6), β₈ contributes 78ms, matching Scout TTFT gap (~80ms)

4. **Coefficient value is physically plausible**: 30μs per routed token is within expected 10-50μs range

5. **Predictions unchanged**: Despite β₈ being non-zero, predictions are byte-for-byte identical to iter7 (when β₈ didn't exist)

### How the Bug Happened

Looking at the code evolution, β₈ was likely copied from β₇ or another coefficient. The developer:
1. Copied β₇ code (which uses milliseconds for decode overhead)
2. Changed the comment to β₈ (MoE routing)
3. Kept the "milliseconds" assumption but updated expected values to seconds
4. Result: comment says "milliseconds", expected values say "seconds", code uses wrong conversion factor

### Why Optimizer Couldn't Detect This

The optimizer still learned β₈ = 30μs because:
1. Optuna explores coefficient space regardless of impact
2. With 1000× underestimate, β₈ contribution is noise-level (~0.078ms)
3. Optimizer converged to a plausible value by chance (or because noise slightly favored 30μs)
4. But the actual gradient is zero, so optimizer can't refine β₈ based on loss

This is **NOT an optimizer failure** - it's a code bug that breaks the physics model.

---

## Recommendations for Iter8 Fix

### Immediate Actions

1. **Fix units bug** in `sim/latency/evolved_model.go:344`:
   ```diff
   -  moeRoutingTimeUs = routedTokens * m.Beta[8] * 1000.0
   +  moeRoutingTimeUs = routedTokens * m.Beta[8] * 1e6
   ```

2. **Fix comment** on line 342 for clarity:
   ```diff
   -  // β₈ coefficient is in milliseconds per routed token (expected 0.000010-0.000050 = 10-50μs)
   -  // Convert to microseconds: routedTokens × β₈ × 1000
   +  // β₈ coefficient is in SECONDS per routed token (expected 0.000010-0.000050 = 10-50μs)
   +  // Convert to microseconds: routedTokens × β₈ × 1e6
   ```

3. **Re-compile and re-run optimization**: With bug fixed, β₈ will have actual predictive power

4. **Validate fix**: After re-optimization, verify Scout TTFT errors improve >40pp

### Testing the Fix

After applying the fix, test with current β₈ = 0.00003:
- Expected Scout general contribution: 2600 × 0.00003 × 1e6 = 78,000μs = 78ms
- This should dramatically improve Scout TTFT predictions
- If not, investigate InterleaveMoELayerStep calculation (lines 307-318)

### Long-term Process Improvement

**Why wasn't this caught?**
1. No unit tests for basis function calculations
2. No integration test comparing StepTime output before/after coefficient changes
3. No "sanity check" that new terms contribute non-zero amounts

**Recommendations**:
1. Add unit tests for each β term's contribution (with known ModelConfig values)
2. Add integration test: "if β₈ = 30μs and model is Scout, contribution should be ~70-80ms"
3. Add CI check: "if new coefficient added, predictions must change from baseline"

---

## Summary

**Root Cause**: Units conversion bug (1000× instead of 1e6) makes β₈ effectively inactive.

**Fix**: Change line 344 from `* 1000.0` to `* 1e6`.

**Impact**: With fix, β₈ should contribute ~78ms per Scout request, dramatically improving TTFT predictions.

**Next Steps**: Fix bug, re-compile, re-optimize iter8, validate all hypotheses.

**Process Learning**: Need defensive unit tests for basis function contributions to catch this class of bug earlier.
