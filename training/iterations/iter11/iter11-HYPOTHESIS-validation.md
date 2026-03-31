# Iteration 11: Hypothesis Validation

## Summary

**Overall Verdict**: ❌ **REJECTED (Premise Was Incorrect)**

All 7 hypotheses were based on the incorrect premise that β₁₀ and β₃' basis functions had "formulation bugs." After code audit and unit test validation, the basis functions are **CORRECT** with **zero bugs**. The hypotheses cannot be properly evaluated because the foundational assumption was wrong.

**Root Cause**: YAML comment error ("0.1-1.0 ms" should be "0.1-1.0 μs") led to wrong expected ranges, causing the incorrect diagnosis.

**Actual Problem**: 6 out of 11 other coefficients are out of range, particularly β₆ = 59ms (expected 15-40ms).

---

## H-main: Fixed Basis Functions Enable Sequence-Length Overhead Capture

**Status**: ❌ **REJECTED (Not Testable - Premise Incorrect)**

### Prediction
After fixing β₁₀ and β₃' formulation bugs:
- Overall loss: 160.6% → **<90%**
- TTFT RMSE: 64.8% → **<40%**
- E2E RMSE: 95.8% → **<55%**
- β₁₀ coefficient: **0.1-1.0 ms** per (token²/batch_request)
- β₃' coefficient: **0.1-1.0 μs** per (token×layer)
- β₆ reversion: **15-40ms**

### Result
- Overall loss: **4084.44%** (25× worse than iter9, 45× worse than target)
- TTFT RMSE: **1423.25%** (22× worse than iter9, 36× worse than target)
- E2E RMSE: **2661.18%** (28× worse than iter9, 48× worse than target)
- β₁₀ coefficient: **0.950 μs** (NOT 0.1-1.0 ms!)
- β₃' coefficient: **0.252 μs** (within 0.1-1.0 μs range ✓)
- β₆: **59.3 ms** (still inflated, not reverted)

### Verdict: ❌ REJECTED

**Why**: The hypothesis predicted β₁₀ = 0.1-1.0 **ms**, but:
1. The expected range was wrong (should be 0.1-1.0 **μs**)
2. β₁₀ = 0.950 μs is actually **CORRECT** and within the proper range
3. Basis functions had **no bugs** to fix
4. Loss didn't improve because the premise was wrong - we weren't fixing actual bugs

**Unit Test Evidence**:
```
TestBeta10BatchingInefficiency: PASS (0.00% error)
TestBeta3PrimeKVSeqLen: PASS (0.00% error)
```

The basis function implementations are correct. Cannot evaluate "after fixing bugs" when there were no bugs.

---

## H-unit-tests: Unit Tests Catch Formulation Bugs Before Training

**Status**: ⚠️ **PARTIAL (Tests Exist But Weren't Used to Prevent Waste)**

### Prediction
Unit tests for β₁₀ and β₃' basis functions will validate expected contributions within 10% tolerance and catch formulation bugs before training.

### Result
Unit tests **DO exist** in `sim/latency/evolved_model_test.go`:
- `TestBeta10BatchingInefficiency` ✅ PASSES (0% error)
- `TestBeta3PrimeKVSeqLen` ✅ PASSES (0% error)
- `TestBeta10PhysicsAnalysis` ✅ PASSES (validates expected ranges)

**Problem**: Tests were created but **NOT run before the 11-hour training run**.

### Verdict: ⚠️ PARTIAL

**What went right**: Tests exist and prove implementations are correct
**What went wrong**: Tests weren't run before training, failing to prevent 5,500 trial-hours of waste

**Evidence**:
```bash
$ go test ./sim/latency -run "TestBeta.*" -v
=== RUN   TestBeta10BatchingInefficiency
    ✓ Long-sequence: 31.25ms (0.00% error)
    ✓ Short-sequence: 0.156ms (0.00% error)
    ✓ Scaling ratio: 200.0× (0.00% error)
--- PASS: TestBeta10BatchingInefficiency (0.00s)

=== RUN   TestBeta3PrimeKVSeqLen
    ✓ Long-sequence: 14.00ms (0.00% error)
    ✓ Short-sequence: 2.80ms (0.00% error)
    ✓ Scaling ratio: 5.00× (0.00% error)
--- PASS: TestBeta3PrimeKVSeqLen (0.00s)
```

If these tests had been run BEFORE training, we would have immediately known the basis functions were correct, saving 11 hours.

---

## H-scheduler-reversion: β₆ Reverts After β₁₀ Fix

**Status**: ❌ **REFUTED**

### Prediction
After fixing β₁₀ basis function: β₆ = **15-40ms** (60-85% decrease from iter9's 99ms)

### Result
β₆ = **59.3 ms** (still 48-295% above expected 15-40ms range)

### Verdict: ❌ REFUTED

**Why it failed**:
1. β₁₀ had no bugs to fix, so no "fix" occurred
2. β₆ decreased from iter9's 99ms to 59ms, but not to the expected 15-40ms
3. β₆ may be absorbing overhead that β₁₀ cannot capture (memory bandwidth? queueing delays?)

**Interpretation**: Even with β₁₀ correctly implemented and producing reasonable contributions (59ms for long-seq), β₆ remains inflated. This suggests:
- Either the β₆ expected range is wrong (should be 50-80ms)
- Or β₆ is absorbing variance from a missing complementary term
- Or β₁₀ and β₆ are competing to explain the same variance

**Next step**: Profile vLLM scheduler to measure actual overhead.

---

## H-kv-scaling: β₃ and β₃' Capture Base + Sequence-Length KV Overhead

**Status**: ⚠️ **PARTIALLY REFUTED**

### Prediction
- β₃ (base KV overhead): **0.4-1.5ms** per request
- β₃' (sequence-length KV overhead): **0.1-1.0 μs** per (token×layer)

### Result
- β₃: **0.207 ms** (50% below expected 0.4-1.5ms range)
- β₃' **0.252 μs** (within expected 0.1-1.0 μs range ✓)

### Verdict: ⚠️ PARTIALLY REFUTED

**What worked**: β₃' is within range and produces reasonable contributions (7ms long-seq, 1.4ms short-seq)

**What didn't work**: β₃ is too low (0.2ms vs 0.4-1.5ms expected)

**Interpretation**:
- β₃' basis function is correct and converges to physically plausible value
- β₃ being too low suggests either:
  - The split concept is capturing too much overhead in β₃'
  - Expected range for β₃ is wrong (maybe PagedAttention base overhead is <0.4ms)
  - β₃ is being suppressed by optimizer to compensate for other issues

**Next step**: Profile PagedAttention to measure actual base overhead.

---

## H-boundary-seq-length: β₁₀ Effect Scales Quadratically with Sequence Length

**Status**: ✅ **CONFIRMED**

### Prediction
β₁₀ contributions scale quadratically with sequence length (long/short ratio 10-40×).

### Result
**Unit test validation**:
- Long-seq (500 tokens, batch=4): 31.25ms
- Short-seq (100 tokens, batch=32): 0.156ms
- Ratio: **200×** (matches expected (500/100)² × (32/4) = 200×)

**Actual iter11 contributions** (β₁₀ = 0.950 μs):
- Long-seq: 59.4ms
- Short-seq: 0.297ms
- Ratio: **200×** ✓

### Verdict: ✅ CONFIRMED

**Evidence**: Both unit tests and actual training results show perfect quadratic scaling. The functional form `prefillTokens²/batchSize` is correct.

**This was also confirmed in iter10**: Long/short ratio = 197× matched expected 200×.

**Conclusion**: The basis function formula is correct. The issue was never the implementation - it was the expected coefficient range (ms vs μs).

---

## H-alpha-stability: Constrained Alpha Bounds Prevent Spurious Reduction

**Status**: ✅ **CONFIRMED**

### Prediction
Alpha coefficients remain within bounds, no lower-bound saturation:
- α₀ ≥ 0.5ms
- α₁ ≥ 50μs
- α₂ ≥ 40μs

### Result
- α₀ = **1.25ms** (within [0.5ms, 5.0ms], not saturated ✓)
- α₁ = **62.7 μs** (within [50μs, 300μs], not saturated ✓)
- α₂ = **75.7 μs** (within [40μs, 250μs], not saturated ✓)

### Verdict: ✅ CONFIRMED

Alpha constraints successfully prevented spurious reduction. All three coefficients are within physically plausible ranges and none hit lower bounds.

**Comparison to iter9**:
- α₀: 2.48ms → 1.25ms (50% decrease, but still above lower bound)
- α₁: 127.6μs → 62.7μs (51% decrease, but still above lower bound)
- α₂: 135.0μs → 75.7μs (44% decrease, but still above lower bound)

The decreases are significant but not compensatory (not trying to hit zero). Alpha bounds are working as designed.

---

## H-error-pattern-dense: Dense Long-Sequence Experiments Should Also Improve

**Status**: ❌ **REFUTED**

### Prediction
Dense long-sequence experiments improve >20pp TTFT after fixing β₁₀ and β₃':
- Mistral Nemo general-lite: 91% → <70% TTFT
- Llama-2-7b reasoning-lite: 84% → <60% TTFT
- Qwen2.5-7b reasoning-lite: 79% → <55% TTFT
- 01-ai Yi-34B general-lite: 78% → <55% TTFT
- Llama-3.1-70B general-lite: 77% → <55% TTFT

### Result
**All experiments failed catastrophically**:
- Mistral Nemo general-lite: **2538% TTFT** (worse than prediction by 36×)
- Llama-2-7b reasoning-lite: **525% TTFT** (worse than prediction by 9×)
- Qwen2.5-7b reasoning-lite: **1007% TTFT** (worse than prediction by 18×)
- 01-ai Yi-34B general-lite: **1140% TTFT** (worse than prediction by 21×)
- Llama-3.1-70B general-lite: **1131% TTFT** (worse than prediction by 21×)

### Verdict: ❌ REFUTED

**Why it failed**:
1. Basis functions had no bugs to fix
2. Universal catastrophic failure across ALL experiments (Scout AND dense)
3. No differential improvement pattern

**Interpretation**: The catastrophic failure is independent of architecture (MoE vs dense) and sequence length, confirming the problem is NOT β₁₀ or β₃' but rather:
- β₆ being massively out of range (59ms vs 15-40ms)
- 5 other coefficients also out of range
- Model misspecification (missing terms or wrong expected ranges)

---

## Overall Assessment

### What Iter11 Got Wrong

**Hypothesis Premise**: "β₁₀ and β₃' have formulation bugs that need fixing"
- ❌ **INCORRECT**: Both implementations are correct (0% error in unit tests)

**Expected Ranges**: "β₁₀ = 0.1-1.0 ms per (token²/batch_request)"
- ❌ **INCORRECT**: Should be 0.1-1.0 **μs** (1000× wrong due to YAML typo)

**Diagnosis**: "1000× too small" (iter10 analysis)
- ❌ **INCORRECT**: β₁₀ = 0.945 μs is within correct 0.1-1.0 μs range

### What Iter11 Got Right

**Unit tests exist**: ✅ Tests prove implementations are correct
**Quadratic scaling**: ✅ Confirmed β₁₀ scales correctly (200× ratio)
**Alpha constraints**: ✅ Prevented spurious reduction

### Cost of Wrong Premise

**Time wasted**: 7,250 trial-hours (iter10 + iter11)
**Prevention cost**: 5 minutes to run unit tests
**ROI**: 87,000× if tests had been run first

### The Real Problem

**6 out of 11 coefficients are out of range**:
- **β₆ = 59ms** (expected 15-40ms) → **PRIMARY CULPRIT**
- β₃ = 0.2ms (expected 0.4-1.5ms)
- β₇ = 5.0ms (expected 8-20ms)
- β₀, β₁, β₄ also out of range

**Root causes**:
1. Expected ranges may be wrong (need profiling validation)
2. Missing complementary terms (memory bandwidth saturation?)
3. Model over-parameterization (too many competing terms?)

---

## Recommendations for Iter12

### REQUIRED Before Training

1. **Profile vLLM** to validate expected ranges (2-3 days):
   - Measure actual β₆ scheduler overhead (15-40ms or 50-80ms?)
   - Measure actual β₃ base KV overhead (0.2ms or 0.4-1.5ms?)
   - Measure actual β₇ decode overhead (5ms or 8-20ms?)
   - Update expected ranges based on measurements, not estimates

2. **Run unit tests** (5 minutes):
   - Validate any new/modified basis functions
   - Catch bugs BEFORE 11-hour training runs

3. **Manual validation** (30 minutes):
   - Calculate expected contributions by hand
   - Verify they match unit test expectations

### Do NOT Do

1. ❌ Modify β₁₀ or β₃' implementations (they're correct!)
2. ❌ Try to "fix" basis functions that aren't broken
3. ❌ Train without profiling validation
4. ❌ Accept "1000× wrong" without checking for unit errors

### Expected Outcome

**If expected ranges are corrected** (based on profiling):
- Loss should improve significantly (target <110%)
- Most coefficients should fall within validated ranges
- Model should converge to physically plausible solution

**If expected ranges were already correct**:
- Need to add missing complementary terms
- Or simplify model (remove competing terms)
- Or investigate optimizer getting stuck in local minimum

---

## Conclusion

**All 7 hypotheses were based on incorrect premise**. The basis functions are correct (0% error in unit tests). The iter10/11 diagnosis of "formulation bugs" was wrong, caused by a YAML comment typo that created 1000× wrong expected ranges.

**The real problem**: β₆ = 59ms (vs 15-40ms expected) plus 5 other coefficients out of range. These issues have nothing to do with β₁₀ or β₃'.

**Key learning**: Always run unit tests BEFORE training and audit code BEFORE accepting hypothesis diagnoses. A 5-minute unit test would have saved 7,250 trial-hours.

**Next iteration**: Profile vLLM to validate ALL expected ranges, then redesign iter12 based on measurements, not estimates.
