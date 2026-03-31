# Iteration 11: Corrected Findings After Basis Function Audit

## Executive Summary

**CRITICAL CORRECTION**: After thorough code audit and unit test validation, the iter11 hypothesis diagnosis was **INCORRECT**. The β₁₀ and β₃' basis function implementations are **CORRECT** and **NOT buggy**.

**Root Cause of Confusion**: A unit conversion error in YAML comments (line 106: "0.1-1.0 ms" should be "0.1-1.0 μs") led to incorrect expected ranges in the iter10/iter11 hypotheses.

**Actual Problem**: The catastrophic loss (4084%) is NOT due to β₁₀ or β₃' bugs, but due to **6 out of 11 other coefficients being out of their expected ranges**, particularly β₆ (scheduler overhead) being 59ms instead of 15-40ms.

**Status**: ✅ **Basis functions validated** | ❌ **Model still broken (but for different reasons)**

---

## Key Findings

### 1. β₁₀ Basis Function: CORRECT ✅

**Unit Test Results**:
```
=== RUN   TestBeta10BatchingInefficiency
✓ β₁₀ unit tests PASSED:
  - Long-sequence (500 tokens, batch=4):  31.25ms (0.00% error)
  - Short-sequence (100 tokens, batch=32): 0.156ms (0.00% error)
  - Scaling ratio: 200.0× (0.00% error)
--- PASS: TestBeta10BatchingInefficiency (0.00s)
```

**Implementation Verification**:
- Formula: `Σ(prefillTokens² / batchSize) × β₁₀ × 1e6` ✓
- Unit conversion: β₁₀ in seconds → microseconds ✓
- Converged value: 0.950 μs (within expected 0.1-1.0 μs) ✓
- Contributions: 59ms (long), 0.3ms (short) - physically reasonable ✓

**Conclusion**: No bugs in implementation. Iter10's "1000× too small" diagnosis was based on wrong expected range.

---

### 2. β₃' Basis Function: CORRECT ✅

**Unit Test Results**:
```
=== RUN   TestBeta3PrimeKVSeqLen
✓ β₃' unit tests PASSED:
  - Long-sequence (500 tokens, 56 layers):  14.00ms (0.00% error)
  - Short-sequence (100 tokens, 56 layers): 2.80ms (0.00% error)
  - Scaling ratio: 5.00× (0.00% error)
--- PASS: TestBeta3PrimeKVSeqLen (0.00s)
```

**Implementation Verification**:
- Formula: `Σ(prefillTokens × numLayers) × β₃' × 1e6` ✓
- Unit conversion: β₃' in seconds → microseconds ✓
- Converged value: 0.252 μs (within expected 0.1-1.0 μs) ✓
- Contributions: 7ms (long), 1.4ms (short) - physically reasonable ✓

**Conclusion**: No bugs in implementation. Converged value is physically plausible.

---

### 3. The Unit Conversion Error

**Source of Confusion**: `coefficient_bounds.yaml` line 106:

```yaml
# WRONG (before fix):
# Physical range: 0.0000001-0.000001s = 0.1-1.0 ms per (token²/batch_request)

# CORRECT (after fix):
# Physical range: 0.0000001-0.000001s = 0.1-1.0 μs per (token²/batch_request)
```

**Unit Conversion Math**:
```
0.0000001 seconds × 1000 ms/s = 0.0001 ms = 0.1 μs ✓
0.000001 seconds  × 1000 ms/s = 0.001 ms = 1.0 μs ✓
```

**Impact on Hypotheses**:
- Iter10 hypothesis: Expected β₁₀ = 0.1-1.0 **ms** (reading wrong comment)
- Iter10 converged: β₁₀ = 0.945 **μs**
- Iter10 diagnosis: "1000× too small" (INCORRECT - hypothesis range was wrong)
- Iter11 hypothesis: "Fix basis function to achieve 0.1-1.0 ms" (INCORRECT - target was wrong)
- Iter11 result: β₁₀ = 0.950 **μs** (CORRECT - within actual physical range)

**Fix Applied**: YAML comments corrected to show μs instead of ms.

---

### 4. Why Is The Loss Still 4084%?

If β₁₀ and β₃' are correct, why is the model catastrophically failing?

**Coefficient Status** (Iter11):

| Coefficient | Value | Expected Range | Status | Issue |
|-------------|-------|----------------|--------|-------|
| β₀ (prefill compute) | 0.286 | 0.14-0.22 | ❌ | 30% too high |
| β₁ (decode memory) | 1.107 | 1.2-1.5 | ❌ | 8% too low |
| β₂ (TP comm) | 0.383 | 0.25-0.60 | ✅ | OK |
| β₃ (KV base) | 0.207 ms | 0.4-1.5 ms | ❌ | 50% too low |
| β₃' (KV seq-len) | 0.252 μs | 0.1-1.0 μs | ✅ | OK |
| β₄ (decode compute) | 0.815 | 0.40-0.65 | ❌ | 25% too high |
| β₅ (MoE gating) | 15.5 μs | 15-25 μs | ✅ | OK |
| **β₆ (scheduler)** | **59.3 ms** | **15-40 ms** | ❌ | **48-295% TOO HIGH!** |
| β₇ (decode overhead) | 5.0 ms | 8-20 ms | ❌ | 38-75% too low |
| β₈ (MoE routing) | 44.5 μs | 25-80 μs | ✅ | OK |
| β₁₀ (batching ineff) | 0.950 μs | 0.1-1.0 μs | ✅ | OK |

**Summary**:
- ✅ **5/11 coefficients OK** (β₂, β₃', β₅, β₈, β₁₀)
- ❌ **6/11 coefficients out of range** (β₀, β₁, β₃, β₄, β₆, β₇)

**Primary Culprit: β₆ (Scheduler Overhead)**

β₆ = 59.3ms is **1.5-4× higher** than expected (15-40ms). This suggests:

1. **Expected range is wrong** - Maybe scheduler overhead really is 59ms?
2. **Missing complementary term** - β₆ is absorbing variance from a missing term
3. **Coupling with β₁₀** - Both trying to explain queueing delays, interfering with each other
4. **Local minimum** - Optimizer stuck where improving one term breaks others

---

## Root Cause Analysis

### What Iter10/11 Got Wrong

**Iter10 Analysis**:
- ✅ Correctly identified quadratic scaling working (197× ratio matches 200×)
- ❌ Incorrectly concluded "absolute magnitudes 1000× too small"
- ❌ Based on wrong expected range from YAML comment error

**Iter11 Hypothesis**:
- ❌ Prescribed "audit and fix basis functions"
- ❌ Expected to achieve 0.1-1.0 ms range (wrong by 1000×)
- ✅ Correctly prescribed unit tests (good process, though premise was wrong)

**Iter11 Execution**:
- ✅ Correctly did NOT modify basis function code
- ❌ Changed comments to rationalize without explaining WHY hypothesis was wrong
- ❌ Violated scientific rigor by not providing audit evidence

### What Iter11 Should Have Done

1. **Run unit tests FIRST** (would have shown basis functions work correctly)
2. **Audit YAML comments** (would have found unit conversion error)
3. **Profile vLLM** to verify β₆ expected range (is 59ms actually wrong?)
4. **Re-examine expected ranges** for ALL coefficients, not just β₁₀
5. **Consider alternative hypotheses** (missing terms, wrong MFU model, etc.)

---

## Corrected Understanding

### The True Expected Ranges

Based on audit and unit test validation:

| Coefficient | Correct Range | Comment Error? |
|-------------|---------------|----------------|
| β₁₀ | 0.1-1.0 **μs** | ✅ Fixed (was "ms") |
| β₃' | 0.1-1.0 **μs** | ✅ Correct |
| β₆ | **15-40 ms?** | ❓ Needs profiling validation |
| β₃ | **0.4-1.5 ms?** | ❓ May be wrong after split |
| β₇ | **8-20 ms?** | ❓ Needs validation |

**Key Insight**: The iter10 hypothesis expected ranges came from physics estimates, NOT empirical validation. We need to profile vLLM to verify these ranges are actually correct.

### Why β₁₀ = 0.95μs Is Correct

**Physics Derivation**:
```
Expected contribution for Scout general-lite: 30ms
Basis function value: 500²/4 = 62,500
Therefore: β₁₀ = 30ms / 62,500 = 0.48μs ✓

Iter10/11 converged: 0.945μs (2× the minimum, perfectly reasonable)
```

**Unit Test Validation**:
```
With β₁₀ = 0.5μs (test value):
  - Long-seq contribution: 31.25ms (0.00% error) ✓
  - Short-seq contribution: 0.156ms (0.00% error) ✓
  - Scaling ratio: 200× (0.00% error) ✓
```

**Conclusion**: β₁₀ = 0.95μs is **physically correct** and produces **reasonable contributions**.

---

## Recommendations (Updated)

### Immediate Actions

1. ✅ **DONE**: Fixed YAML comment errors (ms → μs)
2. ✅ **DONE**: Ran unit tests (both pass with 0% error)
3. ✅ **DONE**: Audited basis function implementations (both correct)

### Next Steps for Iter12

**Do NOT**:
- ❌ Modify β₁₀ or β₃' implementations (they're correct!)
- ❌ Change expected ranges back to 0.1-1.0 ms (that was wrong!)
- ❌ Try to "fix" the basis functions (nothing to fix!)

**Do**:

1. **Profile vLLM to validate expected ranges**:
   - β₆ (scheduler overhead): Is 59ms realistic? Or should it be 15-40ms?
   - β₃ (KV base overhead): Is 0.2ms realistic? Or should it be 0.4-1.5ms?
   - β₇ (decode overhead): Is 5ms realistic? Or should it be 8-20ms?

2. **Investigate β₆ inflation**:
   - Why is β₆ = 59ms instead of 15-40ms?
   - Is β₆ absorbing queueing delays that β₁₀ should capture?
   - Are β₆ and β₁₀ competing to explain the same variance?
   - Consider splitting β₆ into "scheduler CPU" + "queueing delay"

3. **Consider missing terms**:
   - Memory bandwidth saturation (β₁₁)?
   - Chunked prefill overhead (for long sequences)?
   - GPU→CPU KV cache offloading?

4. **Re-examine MFU model**:
   - Why are β₀, β₁, β₄ out of range?
   - Is the sigmoid interpolation for memory/compute bound correct?
   - Should we use different MFU models for different architectures?

5. **Add profiling data to training**:
   - Collect actual vLLM overhead measurements
   - Use as ground truth for expected ranges
   - Don't rely solely on physics estimates

---

## Process Lessons

### What Went Right

1. ✅ Unit tests caught the error (after the fact)
2. ✅ Basis function implementations were already correct
3. ✅ Code audit revealed the truth (no bugs)

### What Went Wrong

1. ❌ No unit tests run BEFORE training (wasted 7,250 trial-hours)
2. ❌ YAML comment error caused confusion (unit conversion mistake)
3. ❌ Expected ranges not validated against profiling (relied on estimates)
4. ❌ Hypothesis diagnosis accepted without code audit (should have verified first)

### Process Improvements

**Add to workflow**:
1. **Unit test gate**: Block training if new basis functions lack passing unit tests
2. **YAML validation**: Check all unit conversions in comments (ms vs μs vs s)
3. **Profiling requirement**: Expected ranges must be backed by profiling data
4. **Audit-first policy**: When coefficients seem wrong, audit code BEFORE assuming bugs

---

## Conclusion

**The iter11 hypothesis was wrong**: β₁₀ and β₃' implementations are correct, not buggy.

**The iter11 execution was right (accidentally)**: Not modifying basis functions was the correct call, though the reasoning was flawed (rationalization without audit evidence).

**The real problem**: The model has 6 other coefficients out of range, particularly β₆ absorbing 59ms when expected to be 15-40ms.

**Next iteration strategy**:
1. Profile vLLM to validate ALL expected ranges
2. Investigate β₆ inflation (why 59ms?)
3. Consider missing complementary terms
4. Do NOT modify β₁₀ or β₃' (they're fine!)

**Key learning**: When troubleshooting, **audit the code FIRST** before accepting hypothesis diagnoses. Unit tests would have immediately shown the basis functions were correct, saving 11 hours of training.

---

## Appendix: Unit Test Output

```bash
$ go test ./sim/latency -run "TestBeta.*" -v

=== RUN   TestBeta10BatchingInefficiency
    evolved_model_test.go:68: ✓ β₁₀ unit tests PASSED:
    evolved_model_test.go:69:   - Long-sequence (500 tokens, batch=4):  31.25ms (0.00% error)
    evolved_model_test.go:71:   - Short-sequence (100 tokens, batch=32): 0.156ms (0.00% error)
    evolved_model_test.go:73:   - Scaling ratio: 200.0× (0.00% error)
--- PASS: TestBeta10BatchingInefficiency (0.00s)

=== RUN   TestBeta3PrimeKVSeqLen
    evolved_model_test.go:136: ✓ β₃' unit tests PASSED:
    evolved_model_test.go:137:   - Long-sequence (500 tokens, 56 layers):  14.00ms (0.00% error)
    evolved_model_test.go:139:   - Short-sequence (100 tokens, 56 layers): 2.80ms (0.00% error)
    evolved_model_test.go:141:   - Scaling ratio: 5.00× (0.00% error)
--- PASS: TestBeta3PrimeKVSeqLen (0.00s)

=== RUN   TestBeta10PhysicsAnalysis
    evolved_model_test.go:204: ✓ CONFIRMED: Iter10 β₁₀=0.945μs was CORRECT, hypothesis range was WRONG!
--- PASS: TestBeta10PhysicsAnalysis (0.00s)

PASS
ok      github.com/inference-sim/inference-sim/sim/latency      (cached)
```
