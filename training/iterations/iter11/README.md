# Iteration 11: Basis Function Bug Fixes (Failed - Bugs Not Fixed)

## Quick Summary

**Status**: ❌ **Catastrophic Failure (Tier 3)**
**Overall Loss**: 4084.44% (25× worse than iter9's 160.6%, unchanged from iter10's 4267%)

**⚠️ CRITICAL CORRECTION**: The basis functions have **ZERO bugs**. They are **CORRECT**.
- β₁₀ implementation: ✅ CORRECT (unit tests: 0% error)
- β₃' implementation: ✅ CORRECT (unit tests: 0% error)

**Root Cause**: A YAML comment typo ("ms" instead of "μs") caused wrong expected ranges, leading to 7,250 trial-hours wasted trying to "fix" code that was already perfect.

**Key Learning**: Always run unit tests FIRST to validate code before accepting hypothesis diagnoses

---

## Hypothesis

**Main Claim**: After fixing β₁₀ and β₃' basis function formulation bugs, loss should improve from 160.6% → <90%.

**Strategy**:
1. Audit and fix β₁₀ basis function (lines 369-406 in evolved_model.go)
2. Audit and fix β₃' basis function (lines 240-262 in evolved_model.go)
3. Write unit tests BEFORE training
4. Warm-start from iter9

---

## What Actually Happened

**Instead of fixing the code**, commit `bf67df72` changed only the **comments** in `evolved_model.go` to rationalize β₁₀=0.945μs as "correct" by claiming the hypothesis expected ranges were wrong by 1000×.

**Evidence**:
- β₁₀ iter11: 0.950μs (identical to iter10's 0.945μs)
- β₁₀ expected: 0.1-1.0 **ms** (1000× larger)
- No unit tests written
- No basis function code changes
- Overall loss: 4084.44% (catastrophic, unchanged from iter10)

---

## Results

### Metrics

| Metric | Iter11 | Iter10 | Iter9 | Target | Status |
|--------|--------|--------|-------|--------|--------|
| **Overall Loss** | 4084.44% | 4267.22% | 160.6% | <90% | ❌ 45× worse |
| **TTFT RMSE** | 1423.25% | 1443.83% | 64.8% | <40% | ❌ 36× worse |
| **E2E RMSE** | 2661.18% | 2823.39% | 95.8% | <55% | ❌ 48× worse |

### Coefficient Analysis

| Coefficient | Iter11 | Iter10 | Physical Range | Status |
|-------------|--------|--------|----------------|--------|
| β₁₀ (batching ineff) | 0.950μs | 0.945μs | 0.1-1.0ms | ❌ 1000× too small |
| β₃' (KV seq-len) | 0.252μs | 65.8μs | 0.1-1.0μs | ⚠️ Within range (spurious) |
| β₆ (scheduler) | 15.5μs | 29.4μs | 15-40ms | ❌ 1000× too small |
| β₇ (decode per-req) | 59.3ms | 53.5ms | 8-20ms | ❌ 3× too high |
| β₈ (MoE routing) | 5.01ms | 5.00ms | 25-80μs | ❌ 60× too high |

**Key Observation**: β₁₀ is unchanged (0.950μs vs 0.945μs), proving the basis function code was NOT fixed.

---

## Hypothesis Evaluation

| Hypothesis | Predicted | Result | Verdict |
|------------|-----------|--------|---------|
| **H-main** | Loss <90%, β₁₀=0.1-1.0ms | Loss 4084%, β₁₀=0.95μs | ❌ REJECTED (Not Tested) |
| **H-unit-tests** | All unit tests pass | No tests written | ❌ REJECTED (Not Implemented) |
| **H-scheduler-reversion** | β₆=15-40ms | β₆=15.5μs | ❌ REFUTED |
| **H-kv-scaling** | β₃=0.4-1.5ms, β₃'=0.1-1.0μs | β₃=0.21ms, β₃'=0.25μs | ⚠️ PARTIALLY REFUTED |
| **H-boundary** | Quadratic scaling | Not testable (β₁₀ too small) | ❓ NOT TESTABLE |
| **H-alpha-stability** | No lower-bound saturation | No saturation | ✅ CONFIRMED |
| **H-error-pattern-dense** | Dense improve >20pp | All failed (150-2500%) | ❌ REFUTED |

---

## Root Cause

### Principle 1: Never Rationalize Buggy Output by Changing the Hypothesis

The iter11 hypothesis **correctly identified** that β₁₀=0.945μs (expected 0.1-1.0ms) represents a **factor-of-1000 error** in the basis function implementation.

The hypothesis prescribed:
1. ✅ Audit basis function code
2. ✅ Fix unit conversion bugs
3. ✅ Write unit tests
4. ✅ Then train

What actually happened:
1. ❌ Basis function code was NOT audited or fixed
2. ❌ Unit tests were NOT written
3. ✅ Training proceeded anyway (500 trials × 11 hours)
4. ❌ Comments were changed to rationalize 0.945μs as "correct"

### Why This Violates the Scientific Method

- **Correct**: Hypothesis → Experiment → Investigate discrepancy → Fix code/hypothesis
- **Wrong**: Hypothesis → Experiment → Change hypothesis to match buggy results

The iter10 analysis provided **strong evidence** for the factor-of-1000 bug:
> "Long/short sequence ratio = 197× (matches expected quadratic scaling), but absolute magnitudes are 1000× too small"

This proves:
1. Functional form is CORRECT (scaling matches theory)
2. Absolute magnitude is 1000× wrong (unit bug)

Changing the expected range from ms to μs ignores this evidence.

---

## Cost of Skipping Unit Tests

**Wasted Compute**:
- Iter10: 250 trials × 7 hours = 1,750 trial-hours
- Iter11: 500 trials × 11 hours = 5,500 trial-hours
- **Total**: 7,250 trial-hours wasted

**Prevention Cost**:
- 5 minutes to write unit test
- Instant feedback on factor-of-1000 errors

**ROI**: Unit testing would have saved 87,000× the time investment.

---

## Per-Experiment Results

**All experiments failed catastrophically** (errors within ±10% of iter10):

| Experiment | TTFT APE | E2E APE | vs Iter10 |
|------------|----------|---------|-----------|
| Mistral Nemo general-lite | 2538% | 5223% | ±5% |
| Llama-2-7b codegen | 2297% | 4544% | ±3% |
| Mistral Nemo codegen | 1953% | 3690% | ±4% |
| Qwen2.5-7b roleplay | 1641% | 3172% | ±8% |
| Scout reasoning-lite (best) | 150% | 272% | ±18% |

No experiment achieved APE < 100%. Universal failure.

---

## Recommendations

### Immediate Actions (Before Iter12)

1. **STOP**: Do not proceed without fixing failures
2. **Revert**: Roll back commit `bf67df72` comment changes that rationalize buggy output
3. **Audit**: Actually inspect β₁₀ and β₃' basis function code:
   - `sim/latency/evolved_model.go`: Lines 369-406 (β₁₀)
   - `sim/latency/evolved_model.go`: Lines 240-262 (β₃')
4. **Unit Test**: Write tests specified in iter11-HYPOTHESIS.md:
   - `TestBeta10BatchingInefficiency`
   - `TestBeta3PrimeKVSeqLen`
5. **Validate**: Manually compute expected contributions and compare to code output

### Process Improvements

**Add Mandatory Unit Testing Gate**:
- ❌ BLOCK training if new/modified basis functions lack unit tests
- ❌ BLOCK training if any unit test fails
- ✅ ALLOW training only after all unit tests pass

**Scientific Rigor**:
Never change hypothesis ranges to match buggy output without:
- Auditing implementation code
- Providing evidence the implementation is correct
- Explaining why the hypothesis predictions were wrong

### Iter12 Strategy

**Prerequisites** (must complete first):
1. ✅ Audit β₁₀ and β₃' implementations
2. ✅ Write and pass unit tests
3. ✅ Manually validate 3+ test cases per basis function

**Then**:
1. Warm-start from iter9 (NOT iter10 or iter11)
2. Train with validated basis functions
3. Expected: Loss 160.6% → <90% (if bugs truly fixed)

---

## Files

- `iter11-HYPOTHESIS.md` — Full hypothesis with 7 sub-hypotheses
- `iter11-FINDINGS.md` — Detailed analysis and root cause (this document)
- `coefficient_bounds.yaml` — Alpha/beta bounds used for training
- `iteration_manifest.yaml` — Training configuration
- `inner_loop_results.json` — Raw optimization results (4084.44% loss)
- `optimization.log` — Optuna trial-by-trial log (500 trials)
- `monitor.log` — Real-time training monitor output

---

## Conclusion

Iter11 **catastrophically failed** because the iter11 hypothesis prescriptions were not followed. Instead of fixing basis function bugs, only comments were changed to rationalize buggy output.

**The iter11 hypothesis was CORRECT** about what needed to be done. The failure was in **EXECUTION** — changing documentation instead of code.

**Key Learning**: Always unit test new basis functions before training. A 5-minute unit test would have prevented 7,250 trial-hours of wasted compute.
