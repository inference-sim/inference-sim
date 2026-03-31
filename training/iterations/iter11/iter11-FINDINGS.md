# Iteration 11: Findings and Principles

## Summary

**Catastrophic model divergence (again)**: Iter11 overall loss remains at **4084.44%** (identical to iter10's 4267.22%, still 25× worse than iter9's 160.6%). Root cause: **Basis function bugs were NOT fixed** — instead, someone changed the hypothesis expected ranges to match the buggy code output, violating the scientific method.

**Critical Failure**: The iter11 hypothesis explicitly required "Audit and fix β₁₀ basis function (lines 369-406 in evolved_model.go)" and "Audit and fix β₃' basis function (lines 240-262 in evolved_model.go)". Instead of fixing the code bugs, the commit `bf67df72` changed only the **comments** in `evolved_model.go` to rationalize β₁₀=0.945μs as "correct" by claiming the hypothesis expected ranges were wrong by 1000×.

**Key Learning**: **Never rationalize buggy output by changing the hypothesis**. When basis functions produce coefficients 1000× off from physical estimates, this indicates a code bug, not a wrong hypothesis. The iter10 analysis correctly identified factor-of-1000 and factor-of-65 errors requiring code fixes. Iter11 failed because it changed documentation instead of code.

**Recommendation**: **Revert to iter9**, actually audit and fix β₁₀ and β₃' basis function implementations with unit tests (as originally specified), then retry iter12 with validated basis functions.

---

## Hypothesis Evaluation

### H-main: Fixed Basis Functions Enable Sequence-Length Overhead Capture

**Status**: **REJECTED (Not Tested)**

**Prediction**: After fixing β₁₀ and β₃' formulation bugs:
- Overall loss: 160.6% → **<90%**
- TTFT RMSE: 64.8% → **<40%**
- E2E RMSE: 95.8% → **<55%**
- β₁₀ coefficient: **0.1-1.0 ms** per (token²/batch_request)
- β₃' coefficient: **0.1-1.0 μs** per (token×layer)

**Result**:
- Overall loss: **4084.44%** (26× worse than target, 25× worse than iter9)
- TTFT RMSE: **1423.25%** (36× worse than target, 22× worse than iter9)
- E2E RMSE: **2661.18%** (48× worse than target, 28× worse than iter9)
- β₁₀ coefficient: **0.950 μs** (1000× smaller than predicted 0.1-1.0 ms)
- β₃' (β₄) coefficient: **0.252 μs** (within predicted range but down 250× from iter10)

**Verdict**: **REJECTED** — H-main cannot be evaluated because the prerequisite (fixing basis function code) was not completed. The basis functions were NOT fixed; only comments were changed to rationalize buggy output. β₁₀ converged to 0.950μs (identical to iter10's 0.945μs), proving the code bug persists.

---

### H-unit-tests: Unit Tests Catch Formulation Bugs Before Training

**Status**: **REJECTED (Not Implemented)**

**Prediction**: Unit tests for β₁₀ and β₃' basis functions validate expected contributions within 10% tolerance.

**Result**: **No unit tests were written**. No file `sim/latency/evolved_model_test.go` exists with the specified `TestBeta10BatchingInefficiency` or `TestBeta3PrimeKVSeqLen` functions.

**Verdict**: **REJECTED** — The iter11 hypothesis explicitly specified: "Write unit tests BEFORE training to validate expected contributions" (lines 18, 102-189 in iter11-HYPOTHESIS.md). This step was skipped entirely, allowing the catastrophic training run to proceed without validation.

---

### H-scheduler-reversion: β₆ Reverts After β₁₀ Fix

**Status**: **REFUTED**

**Prediction**: β₆ = **15-40ms** per request (60-85% decrease from iter9's 99ms)

**Result**: β₆ = **15.5 μs** (99.98% decrease from iter9's 99ms, 1000× too small)

**Verdict**: **REFUTED** — β₆ collapsed catastrophically to 15.5μs (expected 15-40**ms**), indicating the optimizer tried to compensate for broken β₁₀ by zeroing scheduler overhead. This is physically implausible (vLLM scheduler CPU overhead cannot be 15μs).

---

### H-kv-scaling: β₃ and β₃' Capture Base + Sequence-Length KV Overhead

**Status**: **PARTIALLY REFUTED**

**Prediction**:
- β₃ (base KV overhead): **0.4-1.5ms** per request
- β₃' (sequence-length KV overhead): **0.1-1.0 μs** per (token×layer)

**Result**:
- β₃: **0.207ms** (within predicted range but 46× smaller than iter9's 9.6ms)
- β₃' (β₄): **0.252 μs** (within predicted 0.1-1.0μs range)

**Verdict**: **PARTIALLY REFUTED** — β₃' is within predicted range, BUT this is likely spurious because overall loss is catastrophic. β₃ decreased 46× from iter9, suggesting the split concept may be fundamentally flawed or the basis functions remain buggy.

---

### H-boundary-seq-length: β₁₀ Effect Scales Quadratically with Sequence Length

**Status**: **NOT TESTABLE**

**Prediction**: β₁₀ contributions scale quadratically (long/short 10-40× ratio).

**Result**: Cannot evaluate because β₁₀=0.950μs is 1000× too small, making all contributions negligible (contributions would be 0.0003-0.059ms vs expected 0.5-80ms).

**Verdict**: **NOT TESTABLE** — The basis function bug makes it impossible to assess quadratic scaling. Iter10 showed the functional form was correct (197× ratio matches expected 200×), but absolute magnitudes remain wrong.

---

### H-alpha-stability: Constrained Alpha Bounds Prevent Spurious Reduction

**Status**: **CONFIRMED**

**Prediction**: Alpha coefficients remain within physically plausible ranges, no lower-bound saturation.

**Result**:
- α₀ = **1.25ms** (within bounds [0.5ms, 5.0ms], not saturated ✓)
- α₁ = **62.7 μs/tok** (within bounds [50μs, 300μs], not saturated ✓)
- α₂ = **75.7 μs/tok** (within bounds [40μs, 250μs], not saturated ✓)

**Verdict**: **CONFIRMED** — Alpha constraints successfully prevented spurious reduction. However, this is insufficient to fix the catastrophic model divergence caused by broken beta basis functions.

---

### H-error-pattern-dense: Dense Long-Sequence Experiments Should Also Improve

**Status**: **REFUTED**

**Prediction**: Dense model long-sequence experiments improve >20pp TTFT.

**Result**: All experiments failed catastrophically with 150-2500% TTFT errors. No improvement over iter10; universal failure across all architectures and workloads.

**Verdict**: **REFUTED** — Cannot evaluate improvement because all experiments failed. The universal failure pattern confirms the catastrophic model divergence is due to basis function bugs, not missing physics terms.

---

## Error Analysis

### Catastrophic Loss Explosion (Unchanged from Iter10)

**Pattern**: All 15 experiments failed with errors in 150-5200% range (median ~1800%), nearly identical to iter10.

**High-error experiments** (APE > 1000%):
- **Mistral Nemo general-lite** (exp 62): TTFT=2538%, E2E=5223% — Dense long-sequence
- **Llama-2-7b codegen**: TTFT=2297%, E2E=4544% — Dense moderate-sequence
- **Mistral Nemo codegen** (exp 63): TTFT=1953%, E2E=3690% — Dense moderate-sequence
- **Qwen2.5-7b roleplay** (exp 64): TTFT=1641%, E2E=3172% — Dense short-sequence
- **Llama-2-7b general**: TTFT=1590%, E2E=2881% — Dense moderate-sequence
- **Llama-2-7b roleplay**: TTFT=1660%, E2E=2716% — Dense short-sequence
- **01-ai Yi-34B general-lite** (exp 65): TTFT=1140%, E2E=2080% — Dense long-sequence
- **Llama-3.1-70B codegen** (exp 61): TTFT=1160%, E2E=1932% — Dense moderate-sequence
- **Llama-3.1-70B general-lite** (exp 60): TTFT=1131%, E2E=1855% — Dense long-sequence
- **Qwen2.5-7b reasoning-lite** (exp 66): TTFT=1007%, E2E=1326% — Dense long-sequence
- **Scout codegen** (exp 20): TTFT=814%, E2E=1444% — MoE moderate-sequence
- **Scout general-lite** (exp 17): TTFT=765%, E2E=1447% — MoE long-sequence
- **Scout roleplay** (exp 21): TTFT=631%, E2E=905% — MoE short-sequence
- **Llama-2-7b reasoning-lite** (exp 67): TTFT=525%, E2E=792% — Dense long-sequence

**Lowest-error experiment** (still catastrophic):
- **Scout reasoning-lite** (exp 48): TTFT=150%, E2E=272% — MoE long-sequence

**Comparison to Iter10**: Errors are within ±10% of iter10 values for all experiments, confirming that **no meaningful change occurred** between iter10 and iter11.

---

## Root Cause: Documentation Changes Instead of Code Fixes

### Principle 1: Never Rationalize Buggy Output by Changing the Hypothesis

**Evidence**:

Commit `bf67df72` ("feat(training): add iter9 hypothesis and update evolved model to iter10") changed `sim/latency/evolved_model.go` comments to say:

> "**CRITICAL CORRECTION FROM ITER10**: Iter10 suffered catastrophic failure... Root cause analysis revealed that the BASIS FUNCTION IMPLEMENTATIONS WERE CORRECT, but the HYPOTHESIS EXPECTED RANGES WERE WRONG by factor of 1000×."
>
> "Iter10 β₁₀ converged to 0.945 μs, which is actually IN THE CORRECT PHYSICAL RANGE! The hypothesis incorrectly expected 0.1-1.0 **ms**, but physics analysis shows it should be 0.1-1.0 **μs** per (token²/batch_request) — a factor of 1000× difference."

This directly contradicts the iter11 hypothesis (lines 13-17):

> "**Iter11 Strategy**:
> 1. **Audit and fix β₁₀ basis function** (lines 369-406 in evolved_model.go) — likely seconds-to-microseconds conversion error or missing multiplication factor
> 2. **Audit and fix β₃' basis function** (lines 240-262 in evolved_model.go) — likely microseconds-to-seconds conversion error or incorrect scaling
> 3. **Write unit tests** BEFORE training to validate expected contributions"

**Mechanism**:

The iter11 hypothesis correctly identified that β₁₀=0.945μs (expected 0.1-1.0ms) represents a **factor-of-1000 error** in the basis function implementation. The hypothesis prescribed:

1. Audit basis function code
2. Fix unit conversion bugs
3. Write unit tests to validate
4. Then train

Instead, what happened:

1. ❌ Basis function code was NOT audited or fixed
2. ❌ Unit tests were NOT written
3. ✅ Training proceeded anyway (500 trials × 11 hours = wasted compute)
4. ❌ Comments were changed to rationalize 0.945μs as "correct"

**Why this violates the scientific method**:

- **Hypothesis → Experiment → Analysis**: If experiment results contradict predictions, you investigate whether (1) hypothesis was wrong, (2) implementation was wrong, or (3) measurement was wrong.
- **NOT Hypothesis → Experiment → Change Hypothesis**: You cannot retroactively change the hypothesis to match buggy results without investigating the implementation.

**Evidence the hypothesis was correct**:

The iter10 analysis (lines 60-77 in iter10-FINDINGS.md) provided strong evidence for the factor-of-1000 bug:

> "**Evidence for correct functional form**: Long/short sequence ratio = 197× (matches expected quadratic scaling), but absolute magnitudes are 1000× too small"

This proves:
1. The quadratic functional form `prefillTokens²/batchSize` is CORRECT (scaling matches theory)
2. The absolute magnitude is 1000× too small (unit bug)

Changing the expected range from 0.1-1.0ms to 0.1-1.0μs ignores this evidence and accepts physically implausible coefficients.

**Action**:

1. **Revert commit bf67df72** (at least the comment changes that rationalize buggy output)
2. **Actually audit basis function code**:
   - `sim/latency/evolved_model.go`: Lines 369-406 (β₁₀ implementation)
   - `sim/latency/evolved_model.go`: Lines 240-262 (β₃' implementation)
3. **Look for unit conversion bugs**:
   - Check if `m.Beta[10]` is multiplied or divided incorrectly
   - Verify time units (seconds vs milliseconds vs microseconds)
   - Verify the aggregation logic matches the documented formula
4. **Write unit tests** (as specified in iter11-HYPOTHESIS.md lines 102-189)
5. **DO NOT proceed to iter12** until unit tests pass

---

### Principle 2: Unit Tests Are Mandatory for New Basis Functions

**Evidence**:

Iter10 and iter11 both ran 500 trials (11+ hours each) without validating basis functions first. A single unit test would have caught the factor-of-1000 error in seconds.

**Cost**:
- Iter10: 250 trials × 7 hours = 1750 trial-hours of wasted compute
- Iter11: 500 trials × 11 hours = 5500 trial-hours of wasted compute
- Total: **7250 trial-hours** wasted due to missing unit tests

**Benefit of unit testing**:
- 5 minutes to write unit test
- Instant feedback on factor-of-1000 errors
- Prevents catastrophic training runs

**Action**:

Add "Write Unit Tests for New Basis Functions" as a **mandatory gate** in the design agent workflow. No iteration should proceed to training without passing unit tests for all new/modified basis functions.

---

## Coefficient Analysis

### Beta Coefficients (Iter11 vs Iter10 vs Iter9)

| Coefficient | Iter11 | Iter10 | Iter9 | Physical Range | Status |
|-------------|--------|--------|-------|----------------|--------|
| β₀ (prefill MFU) | 0.286 | 0.286 | 0.142 | 0.14-0.22 | ❌ Too high (30% above range) |
| β₁ (decode mem MFU) | 1.107 | 1.054 | 1.078 | 1.2-1.5 | ⚠️ Slightly low (8% below range) |
| β₂ (TP comm) | 0.382 | 0.368 | 0.815 | 0.25-0.60 | ⚠️ Near upper bound |
| β₃ (KV base) | 0.207ms | 0.402ms | 9.6ms | 0.4-1.5ms | ❌ Too low (50% below range) |
| β₃' (KV seq-len) | 0.252μs | 65.8μs | N/A | 0.1-1.0μs | ✅ Within range (but spurious) |
| β₄ (decode comp MFU) | 0.815 | 0 | 0.537 | 0.40-0.65 | ❌ Too high (25% above range) |
| β₅ (MoE gating) | 15.5μs | 13.3μs | 19.8μs | 15-25μs | ✅ Within range |
| β₆ (scheduler) | 15.5μs | 29.4μs | 99.3ms | 15-40ms | ❌ 1000× too small |
| β₇ (decode per-req) | 59.3ms | 53.5ms | 11.0ms | 8-20ms | ❌ 3× too high |
| β₈ (MoE routing) | 5.01ms | 5.00ms | 72.7μs | 25-80μs | ❌ 60× too high |
| β₁₀ (batching ineff) | 0.950μs | 0.945μs | N/A | 0.1-1.0ms | ❌ 1000× too small |

**Observations**:

1. **β₁₀ unchanged**: 0.950μs vs 0.945μs (0.5% difference) proves basis function was NOT fixed
2. **β₆ collapsed**: 15.5μs is 1000× smaller than physical scheduler overhead (15-40ms)
3. **β₇ exploded**: 59.3ms is 3× higher than physical decode overhead (8-20ms)
4. **β₈ exploded**: 5.01ms is 60× higher than physical MoE routing (25-80μs)
5. **β₃' "correct"**: 0.252μs is within predicted 0.1-1.0μs, but this is likely spurious given catastrophic overall loss

**Interpretation**:

The optimizer tried to compensate for broken β₁₀ (1000× too small) by:
- Collapsing β₆ to zero (abandoning scheduler overhead)
- Inflating β₇ (absorbing some queueing delay into decode overhead)
- Inflating β₈ (absorbing some batching inefficiency into MoE routing)

These compensations are physically implausible and caused the catastrophic loss explosion.

---

### Alpha Coefficients (Stable)

| Coefficient | Iter11 | Iter9 | Physical Range | Status |
|-------------|--------|-------|----------------|--------|
| α₀ (API overhead) | 1.25ms | 2.48ms | 0.8-2.5ms | ✅ Within bounds |
| α₁ (tokenization) | 62.7μs | 127.6μs | 60-150μs | ✅ Within bounds |
| α₂ (detokenization) | 75.7μs | 135.0μs | 50-120μs | ⚠️ Slightly high |

**Observations**:

1. Alpha constraints successfully prevented spurious reduction (no lower-bound saturation)
2. Alpha coefficients are physically plausible
3. Decreases from iter9 are within reasonable ranges (not compensatory)

---

## Recommendations

### Immediate Actions

1. **STOP**: Do not proceed to iter12 without fixing iter11's failures
2. **Revert**: Roll back commit `bf67df72` comment changes that rationalize buggy output
3. **Audit**: Actually inspect β₁₀ and β₃' basis function code (lines 369-406, 240-262 in evolved_model.go)
4. **Unit Test**: Write and pass all unit tests specified in iter11-HYPOTHESIS.md (lines 102-189)
5. **Validate**: Manually compute expected contributions for test cases and compare to code output

### Process Improvements

1. **Mandatory Unit Testing Gate**: Add to design agent workflow:
   - ❌ BLOCK training if new/modified basis functions lack unit tests
   - ❌ BLOCK training if any unit test fails
   - ✅ ALLOW training only after all unit tests pass

2. **Scientific Rigor**: Never change hypothesis expected ranges to match buggy output without:
   - Auditing implementation code
   - Providing evidence the implementation is correct
   - Explaining why the hypothesis was wrong

3. **Cost/Benefit Analysis**: Before each training run, ask:
   - What is the probability this run will succeed given current evidence?
   - If basis functions are untested, probability of success ≈ 0%
   - 11 hours of compute × $X per hour × 0% success = wasted money

### Iter12 Strategy

**Prerequisites (must complete before iter12)**:
1. ✅ Audit β₁₀ and β₃' basis function implementations
2. ✅ Write and pass unit tests (TestBeta10BatchingInefficiency, TestBeta3PrimeKVSeqLen)
3. ✅ Manually validate at least 3 test cases per basis function

**Then**:
1. Warm-start from iter9 (NOT iter10 or iter11)
2. Train with validated basis functions
3. Expected: Overall loss 160.6% → <90% (if basis functions are truly fixed)

---

## Success Criteria Evaluation

### Tier 1 (Full Success): **FAILED**
- Overall loss <90%: ❌ Got 4084.44%
- TTFT RMSE <40%: ❌ Got 1423.25%
- E2E RMSE <55%: ❌ Got 2661.18%
- Scout long-sequence <60% TTFT: ❌ Got 150-765%
- β₁₀ = 0.1-1.0 ms: ❌ Got 0.950 μs (1000× too small)
- β₃' = 0.1-1.0 μs: ✅ Got 0.252 μs (within range, but spurious)
- β₆ = 15-40ms: ❌ Got 15.5 μs (1000× too small)
- All unit tests PASS: ❌ No unit tests written

### Tier 2 (Partial Success): **FAILED**
- Overall loss <110%: ❌ Got 4084.44%
- Scout long-sequence <70% TTFT: ❌ Got 150-765%
- β₁₀ and β₃' physically plausible: ❌ β₁₀ is 1000× too small
- 2/3 coefficient explosions decrease >30%: ❌ β₆ collapsed, β₇/β₈ inflated

### Tier 3 (Failure): **CONFIRMED**
- Overall loss >130%: ✅ Got 4084.44% (catastrophic)
- Scout long-sequence >80% TTFT: ✅ Got 150-765%
- β₁₀ converged to zero OR >5ms: ✅ Got 0.950μs (implausible)
- β₆ remains >80ms: ✅ Got 15.5μs (collapsed, not remained high)
- Unit tests FAIL: ✅ No unit tests written (automatic fail)

**Overall Verdict**: **Tier 3 Failure (Catastrophic)**

---

## Lessons Learned

### What Went Wrong

1. **Ignored hypothesis prescriptions**: Iter11 hypothesis explicitly required code fixes and unit tests, but only comments were changed
2. **Rationalized buggy output**: Changed expected ranges to match buggy coefficients instead of fixing code
3. **Skipped validation**: No unit tests written before 11-hour training run
4. **Wasted 7250 trial-hours**: Iter10 + iter11 both failed due to same basis function bugs

### What Would Have Prevented This

1. **5 minutes of unit testing**: A single test validating β₁₀ contributions would have caught factor-of-1000 error
2. **Code review of basis functions**: Auditing lines 369-406 would have revealed unit conversion bug
3. **Mandatory process gate**: "No training without passing unit tests" rule

### Key Principles

1. **Unit test new basis functions BEFORE training** (saves 1000× compute time)
2. **Never rationalize buggy output by changing hypothesis** (violates scientific method)
3. **When coefficients are 1000× off, it's a code bug** (not a wrong hypothesis)
4. **Process discipline prevents catastrophic failures** (mandatory gates matter)

---

## Conclusion

Iter11 catastrophically failed (loss 4084.44%, 25× worse than iter9) because basis function bugs were NOT fixed as prescribed by the iter11 hypothesis. Instead, comments were changed to rationalize buggy output, violating basic scientific rigor. The optimizer cannot converge with broken basis functions that produce contributions 1000× off from physical estimates.

**DO NOT proceed to iter12** until:
1. ✅ Basis function code is audited and fixed
2. ✅ Unit tests are written and passing
3. ✅ Manual validation confirms expected contributions match test cases

The iter11 hypothesis was CORRECT about what needed to be done. The failure was in EXECUTION — changing documentation instead of code.
