# Iteration 11: Findings and Analysis

## Executive Summary

**Status**: ❌ **Catastrophic Failure** (Loss: 4084.44%, unchanged from iter10's 4267%)

**⚠️ CRITICAL: NO BUGS IN BASIS FUNCTIONS ⚠️**

The β₁₀ and β₃' basis function implementations are **CORRECT**. They have **ZERO bugs**. Unit tests pass with 0% error. The iter10/11 hypothesis that claimed "formulation bugs" was **WRONG**.

**What Was Actually Wrong**: A typo in YAML comments (wrote "ms" instead of "μs") caused incorrect expected ranges, leading to 7,250 trial-hours wasted trying to fix code that was already working perfectly.

**Root Cause of Confusion**: A unit conversion error in YAML comments ("0.1-1.0 ms" should be "0.1-1.0 μs") led to incorrect expected ranges, causing 7,250 trial-hours wasted on fixing non-existent bugs.

**Actual Problem**: The catastrophic loss is due to **6 out of 11 other coefficients being out of range**, particularly β₆ (scheduler overhead) = 59ms vs expected 15-40ms.

**Key Learning**: Always run unit tests BEFORE training and audit code BEFORE accepting hypothesis diagnoses. A 5-minute unit test would have prevented this entire iteration's wasted effort.

---

## Results

### Metrics (Iter11 vs Iter10 vs Iter9)

| Metric | Iter11 | Iter10 | Iter9 | Target | Status |
|--------|--------|--------|-------|--------|--------|
| **Overall Loss** | 4084.44% | 4267.22% | 160.6% | <90% | ❌ 45× worse than target |
| **TTFT RMSE** | 1423.25% | 1443.83% | 64.8% | <40% | ❌ 36× worse than target |
| **E2E RMSE** | 2661.18% | 2823.39% | 95.8% | <55% | ❌ 48× worse than target |

**Observation**: Iter11 results are virtually identical to iter10 (within ±5%), confirming that no meaningful changes occurred.

---

## Basis Function Audit Results

### β₁₀ (Batching Inefficiency): ✅ CORRECT

**Unit Test Results**:
```
=== RUN   TestBeta10BatchingInefficiency
✓ β₁₀ unit tests PASSED:
  - Long-sequence (500 tokens, batch=4):  31.25ms (0.00% error)
  - Short-sequence (100 tokens, batch=32): 0.156ms (0.00% error)
  - Scaling ratio: 200.0× (0.00% error)
--- PASS: TestBeta10BatchingInefficiency (0.00s)
```

**Implementation Verification** (lines 401-418 in evolved_model.go):
```go
batchingInefficiencySum += (numPrefillTokens * numPrefillTokens) / effectiveBatchSize
batchingInefficiencyTimeSeconds := batchingInefficiencySum * m.Beta[10]
batchingInefficiencyContribution := batchingInefficiencyTimeSeconds * 1e6
```

**Manual Calculation**:
- Scout long-seq: 500² / 4 = 62,500 × 0.00000095s × 1e6 = **59.4 ms** ✓
- Scout short-seq: 100² / 32 = 312.5 × 0.00000095s × 1e6 = **0.297 ms** ✓
- Ratio: 59.4 / 0.297 = **200×** (matches expected quadratic scaling) ✓

**Converged Value**: β₁₀ = 0.950 μs
- Within expected range: 0.1-1.0 μs ✅
- Produces physically reasonable contributions ✅
- Implementation has no bugs ✅

### β₃' (KV Sequence-Length Overhead): ✅ CORRECT

**Unit Test Results**:
```
=== RUN   TestBeta3PrimeKVSeqLen
✓ β₃' unit tests PASSED:
  - Long-sequence (500 tokens, 56 layers):  14.00ms (0.00% error)
  - Short-sequence (100 tokens, 56 layers): 2.80ms (0.00% error)
  - Scaling ratio: 5.00× (0.00% error)
--- PASS: TestBeta3PrimeKVSeqLen (0.00s)
```

**Implementation Verification** (lines 258-270 in evolved_model.go):
```go
kvMgmtSeqLenTokenLayers += numPrefillTokens * float64(m.modelConfig.NumLayers)
kvMgmtSeqLenTimeSeconds := kvMgmtSeqLenTokenLayers * m.Beta[4]
kvMgmtSeqLenContribution := kvMgmtSeqLenTimeSeconds * 1e6
```

**Manual Calculation**:
- Scout long-seq: 500 × 56 = 28,000 × 0.000000252s × 1e6 = **7.06 ms** ✓
- Scout short-seq: 100 × 56 = 5,600 × 0.000000252s × 1e6 = **1.41 ms** ✓
- Ratio: 7.06 / 1.41 = **5×** (matches expected linear scaling) ✓

**Converged Value**: β₃' = 0.252 μs
- Within expected range: 0.1-1.0 μs ✅
- Produces physically reasonable contributions ✅
- Implementation has no bugs ✅

---

## The Unit Conversion Error

### Source of Confusion

**In `coefficient_bounds.yaml` line 106** (BEFORE fix):
```yaml
# Physical range: 0.0000001-0.000001s = 0.1-1.0 ms per (token²/batch_request)
```

**Correction** (AFTER fix):
```yaml
# Physical range: 0.0000001-0.000001s = 0.1-1.0 μs per (token²/batch_request)
```

**Unit Conversion Math**:
```
0.0000001 seconds × 1000 ms/s = 0.0001 ms = 0.1 μs ✓ (NOT 0.1 ms!)
0.000001 seconds  × 1000 ms/s = 0.001 ms  = 1.0 μs ✓ (NOT 1.0 ms!)
```

### Impact on Iterations

| Stage | Expected Range | Actual Value | Diagnosis | Correct? |
|-------|----------------|--------------|-----------|----------|
| Iter10 hypothesis | 0.1-1.0 **ms** | 0.945 **μs** | "1000× too small" | ❌ Wrong range! |
| Iter11 hypothesis | 0.1-1.0 **ms** | Expected | "Fix bugs to achieve ms" | ❌ Wrong target! |
| Iter11 result | 0.1-1.0 **μs** | 0.950 **μs** | Within correct range | ✅ Actually correct! |

**Cost of Error**: 7,250 trial-hours wasted (iter10: 1,750 + iter11: 5,500) chasing non-existent bugs.

---

## Coefficient Analysis

### Full Coefficient Status (Iter11)

| Coefficient | Value | Expected Range | Status | Deviation |
|-------------|-------|----------------|--------|-----------|
| β₀ (prefill compute) | 0.286 | 0.14-0.22 | ❌ | +30% too high |
| β₁ (decode memory) | 1.107 | 1.2-1.5 | ❌ | -8% too low |
| β₂ (TP comm) | 0.383 | 0.25-0.60 | ✅ | Within range |
| β₃ (KV base) | 0.207 ms | 0.4-1.5 ms | ❌ | -50% too low |
| β₃' (KV seq-len) | 0.252 μs | 0.1-1.0 μs | ✅ | Within range |
| β₄ (decode compute) | 0.815 | 0.40-0.65 | ❌ | +25% too high |
| β₅ (MoE gating) | 15.5 μs | 15-25 μs | ✅ | Within range |
| **β₆ (scheduler)** | **59.3 ms** | **15-40 ms** | ❌ | **+48-295% TOO HIGH** |
| β₇ (decode overhead) | 5.0 ms | 8-20 ms | ❌ | -38-75% too low |
| β₈ (MoE routing) | 44.5 μs | 25-80 μs | ✅ | Within range |
| β₁₀ (batching ineff) | 0.950 μs | 0.1-1.0 μs | ✅ | Within range |

**Summary**:
- ✅ **5/11 coefficients within range** (β₂, β₃', β₅, β₈, β₁₀)
- ❌ **6/11 coefficients out of range** (β₀, β₁, β₃, β₄, β₆, β₇)

### Primary Culprit: β₆ (Scheduler Overhead)

**β₆ = 59.3 ms** vs expected **15-40 ms** → **1.5-4× too high!**

This massive deviation suggests three possibilities:

1. **Wrong expected range**: Maybe scheduler overhead really is 50-80ms (needs profiling validation)
2. **Competing with β₁₀**: Both terms trying to explain queueing delays, confusing optimizer
3. **Missing term**: β₆ absorbing variance from missing complementary term (memory bandwidth saturation?)

**Evidence that β₆ is the problem**:
- Iter9 β₆ = 99ms (also inflated)
- Iter10 β₆ = 29μs (collapsed when β₁₀ added, optimizer artifact)
- Iter11 β₆ = 59ms (still high after β₁₀ "fix")

---

## Root Cause Analysis

### What Iter10/11 Got Wrong

**Iter10 Analysis**:
- ✅ Correctly identified quadratic scaling works (197× ratio matches 200×)
- ✅ Correctly identified β₃ split concept works (reverted to 0.4ms)
- ❌ Incorrectly concluded "absolute magnitudes 1000× too small" (based on wrong expected range)
- ❌ Recommended "fix basis function formulation bugs" (there were no bugs)

**Iter11 Hypothesis**:
- ❌ Prescribed "audit and fix β₁₀ basis function" (implementation was already correct)
- ❌ Prescribed "audit and fix β₃' basis function" (implementation was already correct)
- ✅ Correctly prescribed unit tests (good process, though premise was wrong)
- ❌ Expected 0.1-1.0 ms range (wrong by 1000×)

**Iter11 Execution**:
- ✅ Correctly did NOT modify basis function code (nothing to fix)
- ❌ Changed comments to rationalize without providing audit evidence
- ❌ Violated scientific rigor by not explaining WHY hypothesis was wrong

### What Actually Happened

1. YAML comment had unit conversion error (ms instead of μs)
2. Iter10 hypothesis read wrong comment, expected 0.1-1.0 ms
3. Iter10 saw β₁₀ = 0.945 μs, concluded "1000× too small"
4. Iter11 hypothesis prescribed fixing "bugs" to achieve ms range
5. Iter11 execution rationalized μs as correct without explaining why
6. Audit reveals: Code was correct all along, expected range was wrong

---

## Per-Experiment Results

**All 15 experiments failed catastrophically** (errors within ±10% of iter10):

| Experiment | TTFT APE | E2E APE | Architecture | Sequence Length |
|------------|----------|---------|--------------|-----------------|
| Mistral Nemo general-lite | 2538% | 5223% | Dense | Long (500 tok) |
| Llama-2-7b codegen | 2297% | 4544% | Dense | Moderate (200 tok) |
| Mistral Nemo codegen | 1953% | 3690% | Dense | Moderate |
| Qwen2.5-7b roleplay | 1641% | 3172% | Dense | Short (100 tok) |
| Llama-2-7b general | 1590% | 2881% | Dense | Moderate |
| Llama-2-7b roleplay | 1660% | 2716% | Dense | Short |
| 01-ai Yi-34B general-lite | 1140% | 2080% | Dense | Long |
| Llama-3.1-70B codegen | 1160% | 1932% | Dense | Moderate |
| Llama-3.1-70B general-lite | 1131% | 1855% | Dense | Long |
| Qwen2.5-7b reasoning-lite | 1007% | 1326% | Dense | Long |
| Scout codegen | 814% | 1444% | MoE | Moderate |
| Scout general-lite | 765% | 1447% | MoE | Long |
| Scout roleplay | 631% | 905% | MoE | Short |
| Llama-2-7b reasoning-lite | 525% | 792% | Dense | Long |
| Scout reasoning-lite (best) | 150% | 272% | MoE | Long |

**Observation**: No experiment achieved APE < 100%. Universal catastrophic failure across all architectures, sequence lengths, and workloads.

---

## Hypothesis Evaluation

| Hypothesis | Predicted | Result | Verdict | Reasoning |
|------------|-----------|--------|---------|-----------|
| **H-main** | Loss <90%, β₁₀=0.1-1.0ms | Loss 4084%, β₁₀=0.95μs | ❌ **REJECTED** | Expected range was wrong (μs not ms); can't evaluate because premise was incorrect |
| **H-unit-tests** | All tests pass | Tests pass but not run before training | ⚠️ **PARTIAL** | Tests exist and pass, but weren't used to prevent waste |
| **H-scheduler-reversion** | β₆=15-40ms | β₆=59.3ms | ❌ **REFUTED** | β₆ still inflated; β₁₀ fix didn't help |
| **H-kv-scaling** | β₃=0.4-1.5ms, β₃'=0.1-1.0μs | β₃=0.21ms, β₃'=0.25μs | ⚠️ **PARTIAL** | β₃' within range but β₃ too low; split may be flawed |
| **H-boundary** | Quadratic scaling 10-40× | 200× scaling preserved | ✅ **CONFIRMED** | Scaling works correctly (iter10 showed this too) |
| **H-alpha-stability** | No lower-bound saturation | No saturation | ✅ **CONFIRMED** | Alpha constraints work as designed |
| **H-error-pattern-dense** | Dense improve >20pp | All failed (150-2500%) | ❌ **REFUTED** | Universal failure; no improvement |

---

## Key Lessons

### 1. Always Run Unit Tests BEFORE Training

**Cost of skipping**:
- Iter10: 250 trials × 7 hours = 1,750 trial-hours wasted
- Iter11: 500 trials × 11 hours = 5,500 trial-hours wasted
- **Total: 7,250 trial-hours**

**Prevention cost**: 5 minutes to run `go test ./sim/latency -run "TestBeta.*" -v`

**ROI**: Unit testing saves **87,000× the time investment**

### 2. Audit Code FIRST Before Accepting Diagnoses

**Iter10 diagnosis**: "β₁₀ = 0.945μs is 1000× too small"
**Reality**: β₁₀ = 0.945μs is perfectly correct; expected range was wrong

**Process failure**: Accepted hypothesis diagnosis without verifying implementation

**Correct process**:
1. Read hypothesis diagnosis
2. Audit actual code implementation
3. Run unit tests to validate
4. THEN decide if there's a bug

### 3. Validate Expected Ranges with Profiling

**Current approach**: Physics estimates → Expected ranges → Training
**Problem**: Physics estimates can be wrong by 1000× (unit errors) or 4× (β₆)

**Better approach**: Profiling → Expected ranges → Training
- Profile vLLM scheduler to measure actual β₆ overhead
- Profile KV cache manager to measure actual β₃ overhead
- Use measurements as ground truth, not estimates

### 4. Check YAML Comments for Unit Errors

**Common pitfalls**:
- ms vs μs vs s (1000× difference)
- Mismatched units in comments vs bounds
- Copy-paste errors when duplicating coefficient blocks

**Prevention**: Always verify unit conversions in comments match actual calculations

---

## Recommendations for Iter12

### Phase 1: Profile vLLM (REQUIRED BEFORE TRAINING)

**Goal**: Validate expected ranges with actual measurements, not estimates.

**Tasks** (1-2 days):
1. Profile vLLM scheduler overhead:
   - Scout general-lite (500 tokens) → Measure batch_formation + kv_block_alloc time
   - Scout roleplay (100 tokens) → Measure same
   - Expected: 15-40ms or 50-80ms?

2. Profile KV cache management overhead:
   - Measure PagedAttention base overhead per request (β₃)
   - Measure block allocation scaling with sequence length (β₃')
   - Verify if 0.2ms (base) + 0.25μs/token-layer (seq-len) matches reality

3. Profile decode per-request overhead:
   - Measure output processing + TP coordination time (β₇)
   - Verify if 5ms (iter11) or 8-20ms (expected) is correct

**Deliverable**: Profiling report with measured values → Update coefficient_bounds.yaml

### Phase 2: Redesign Iter12 Based on Profiling

**Option A: If β₆ = 50-80ms is correct (profiling validates high range)**

```yaml
# Update expected range based on profiling
beta_bounds:
  - [0.040, 0.090]  # β₆: [40ms, 90ms] — expanded based on profiling data
```

Retry iter12 with corrected expected ranges.

**Option B: If β₆ = 15-40ms is correct (profiling shows low range)**

Three sub-options:

1. **Split β₆** into CPU vs queueing components:
   - β₆ₐ: Scheduler CPU overhead (fixed, 15-40ms)
   - β₆ᵦ: Queueing delay per request (variable)

2. **Add β₁₁** for memory bandwidth saturation:
   - Basis function: `Σ(prefillTokens × kvCacheSize / memBandwidth)`
   - Captures long-sequence memory traffic overhead

3. **Remove β₁₀** temporarily to test competition hypothesis:
   - See if β₆ increases when β₁₀ is removed
   - Confirms if they're competing to explain same variance

**Option C: Simplify model first**

Revert to iter9 architecture (remove β₁₀ and β₃') to establish baseline:
- If loss stays at 160%, confirms adding terms didn't help
- If loss improves below 160%, suggests over-parameterization

### Phase 3: Execute Iter12

**Prerequisites** (must complete BEFORE training):
1. ✅ Profiling completed and expected ranges validated
2. ✅ Unit tests exist and pass for any new basis functions
3. ✅ Manual calculations verify expected contributions

**Configuration**:
- Warm-start from iter9 (NOT iter10 or iter11)
- Use profiling-validated expected ranges
- Keep β₁₀ and β₃' as-is (they're correct!)

**Success criteria**:
- Overall loss: **<110%** (31% improvement from iter9)
- At least 8/11 coefficients within expected ranges
- β₆ within validated range (whether 15-40ms or 50-80ms)

### ❌ Do NOT Do

1. ❌ Modify β₁₀ or β₃' implementations (they're correct!)
2. ❌ Train without profiling validation first
3. ❌ Accept physics estimates without measurement
4. ❌ Skip unit tests to "save time"

---

## Process Improvements

### Add Mandatory Gates to Workflow

**Gate 1: Unit Testing**
- ❌ BLOCK training if new basis functions lack unit tests
- ❌ BLOCK training if any unit test fails
- ✅ ALLOW training only after all tests pass

**Gate 2: Profiling Validation**
- ❌ BLOCK training if expected ranges lack profiling evidence
- ✅ REQUIRE profiling report for ranges >2× away from previous iteration
- ✅ ALLOW training only with validated ranges

**Gate 3: Code Audit**
- ❌ BLOCK hypothesis acceptance if diagnosis claims "bugs" without audit
- ✅ REQUIRE code audit BEFORE accepting "formulation bug" claims
- ✅ ALLOW "basis function is broken" only after audit confirms it

### Update Scientific Rigor Standards

**When coefficients deviate from expected**:
1. Check for YAML comment errors (unit conversions)
2. Run unit tests to validate implementation
3. Audit code to verify formulation
4. Profile system to validate expected ranges
5. THEN conclude if there's a bug or wrong expectation

**Never**:
- Accept "1000× too small" without checking for unit errors
- Change hypothesis ranges to match results without audit evidence
- Skip unit tests to save time (costs 87,000× more in wasted training)

---

## Conclusion

**The iter11 hypothesis was WRONG**: β₁₀ and β₃' implementations are correct, not buggy.

**The iter11 execution was accidentally RIGHT**: Not modifying code was correct (nothing to fix), but process was flawed (no audit evidence provided).

**The real problem**: Model has 6 coefficients out of range, especially β₆ = 59ms vs 15-40ms expected.

**Next iteration strategy**:
1. **Profile vLLM** to validate ALL expected ranges (2-3 days)
2. **Redesign iter12** based on profiling data, not physics estimates
3. **Run unit tests** before any training (5 minutes)
4. **Expected outcome**: If ranges are validated, loss should improve significantly

**Bottom line**: We spent 7,250 trial-hours fixing non-existent bugs because we skipped a 5-minute unit test and trusted a YAML comment with a unit error. The basis functions work perfectly. Time to profile the real system and fix the real problems.

---

## Files Modified

✅ `coefficient_bounds.yaml` - Fixed YAML comment (ms → μs)
✅ `iter11-FINDINGS.md` - This comprehensive analysis
✅ Git commit 8e48548f - All changes committed
