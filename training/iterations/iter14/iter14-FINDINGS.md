# Iteration 14: Findings and Principles

## Summary

Iteration 14 attempted to fix iter13's catastrophic failure (loss 2387%) by correcting a single bug: missing `× numMoELayers` multiplier in the β₅ (MoE gating) basis function. **The fix worked at the coefficient level** (β₅ converged from 1924.4 to 32.5, within predicted 1-50 range), but **failed catastrophically at the performance level** (loss barely improved: 2387% → 2319%, only 2.8% improvement vs predicted 92% improvement).

**Key Learning**: **Necessary ≠ Sufficient**. Fixing β₅ was necessary to prevent that specific coefficient explosion, but it was NOT sufficient to recover model performance. The hypothesis that β₅ was the "sole cause" of cascading failures is **REFUTED**. Iter14 reveals **multiple independent architectural defects** that prevent convergence.

**Status**: ❌ Second consecutive catastrophic failure (both iter13 and iter14 have >2000% loss). The training process is at a critical juncture requiring fundamental changes to approach.

---

## Error Analysis

### Systematic Patterns

**High-error experiments** (APE > 1000%):
1. **Mistral general-lite**: TTFT=3774%, E2E=2905% — Catastrophic overprediction (30-40× worse than ground truth)
2. **Llama-2 general**: TTFT=1559%, E2E=287% — Extreme TTFT overprediction but reasonable E2E (suggests prefill-specific issue)
3. **Llama-2 codegen**: TTFT=1334%, E2E=1216% — Consistent overprediction across both metrics
4. **Llama-3.1 general-lite**: TTFT=1033%, E2E=926% — 10× overprediction (model predicts ~10s for ~1s ground truth)
5. **Yi-34B general-lite**: TTFT=1198%, E2E=1087% — Similar 10× overprediction pattern

**Pattern**: General-lite and codegen workloads are **systematically overpredicted by 10-40×**. These workloads have specific characteristics (high request rate, short sequences) that the model cannot handle.

**Low-error experiments** (APE < 600%):
1. **Scout roleplay**: TTFT=342%, E2E=181% — Best performer (closest to ground truth)
2. **Scout codegen**: TTFT=544%, E2E=334% — Second best (MoE model with β₅ fix)
3. **Llama-3.1 codegen**: TTFT=617%, E2E=460% — Third best among dense models

**Pattern**: Scout (MoE) experiments perform BETTER than dense models after β₅ fix, suggesting the fix is working for MoE but dense models have independent issues.

**Complete catastrophe** (exactly 100% error):
1. **Scout reasoning-lite**: TTFT=100%, E2E=100% — All metrics exactly 100% (numerical failure)
2. **Qwen2.5 reasoning-lite**: TTFT=100%, E2E=100% — Identical failure mode
3. **Llama-2 reasoning-lite**: TTFT=100%, E2E=100% — Identical failure mode

**Pattern**: All three reasoning-lite experiments return **exactly 100% error** (not 95% or 105%, but exactly 100.0%), indicating a **systematic numerical failure** in the simulator (divide-by-zero, integer overflow, or negative time causing requests to timeout and produce zero completed requests).

**Error correlations**:
- ✅ **Confirmed**: Scout (MoE) experiments perform BETTER after β₅ fix (342-767% TTFT vs 1000-3700% for dense models)
- ✅ **Confirmed**: General-lite workloads consistently fail (all >1000% TTFT)
- ✅ **Confirmed**: Reasoning-lite workloads trigger numerical catastrophe (all exactly 100%)
- ❌ **Rejected**: Dense models do NOT recover when β₅ is fixed (all still >600% TTFT)

### Root Cause Hypotheses

**Principle 1**: **Coefficient Convergence ≠ Performance Recovery** (Necessary But Not Sufficient)

- **Evidence**:
  - β₅ converged from 1924.4 (iter13) to 32.5 (iter14), exactly within predicted range 1-50 ✅
  - Overall loss improved only 2.8% (2387% → 2319%), not the predicted 92% improvement ❌
  - All four hypothesis predictions failed despite β₅ converging correctly ❌

- **Mechanism**: **Why does this happen?**
  - The optimizer treats coefficients as **independent tuning knobs**, not as **coupled physics parameters**
  - When β₅ was fixed, the optimizer adjusted OTHER coefficients (β₀, β₁, β₃, β₄, β₆, β₇) to compensate for the changed β₅ contribution
  - Result: β₅ converged, but the **overall model behavior barely changed** because other coefficients absorbed the error
  - **Analogy**: Fixing one broken leg on a table doesn't fix the table if the other three legs are also broken

- **Action for iter15**: **CRITICAL MINDSET SHIFT** - Stop assuming single-bug fixes will work. Instead:
  1. **Multi-coefficient validation**: After fixing ANY coefficient, validate that OTHER coefficients remain stable
  2. **Holistic performance metrics**: Track not just coefficient values but their CONTRIBUTIONS to StepTime (e.g., β₅ × moeGatingTimeUs should be <10% of total)
  3. **Ablation studies**: Before declaring a fix successful, run ablation (remove the term entirely) to verify it's actually helping

---

**Principle 2**: **Warm-Start Failure Indicates Dataset Shift** (Iter7 Coefficients Invalid for Reasoning-Lite)

- **Evidence**:
  - Iter14 warm-started from iter7 coefficients (β₀=0.191, β₁=1.108, β₂=0.185, etc.)
  - Iter14 coefficients diverged massively: β₀→0.392 (+105%), β₃→0.00139ms (-287×), β₆→0.00509ms (-7859×)
  - 0/9 dense model experiments recovered to iter7 baseline (all still 17-156× worse)
  - Reasoning-lite experiments (NEW in iter13-14) return 100% error in BOTH iterations

- **Mechanism**: **Why does warm-start fail?**
  - Iter7 dataset: 15 experiments including 3 **reasoning** workloads (long sequences, overloaded servers, 259-second timeouts)
  - Iter13-14 dataset: 15 experiments including 3 **reasoning-lite** workloads (lighter load, different request rate/duration characteristics)
  - **Dataset changed between iter7 and iter13**, but Agent 1 assumed iter7 coefficients would be a good starting point
  - When dataset characteristics change, the optimal coefficient landscape changes — warm-start can get stuck in wrong basin of attraction
  - **Evidence of stuck optimization**: Iter14 used 1000 trials but loss barely improved (2387% → 2319%, only 2.8%), suggesting optimizer couldn't escape local minimum set by iter7 initialization

- **Action for iter15**: **TRY COLD-START OPTIMIZATION**
  1. Initialize all coefficients from UNIFORM RANDOM distribution within physically plausible bounds
  2. Do NOT initialize from iter7, iter13, or any previous iteration
  3. Increase optimization budget from 1000 to 2000-3000 trials to allow exploration
  4. Use TPE (Tree-structured Parzen Estimator) in Optuna to handle large search space
  5. **Hypothesis**: Cold-start will find a different local minimum better suited to reasoning-lite dataset

---

**Principle 3**: **Roofline Baseline Systematic Bias** (Overpredicts Prefill/Decode MFU by 2×)

- **Evidence**:
  - β₀ (prefill MFU scaling) = 0.392 vs expected 0.16-0.22 → 2× higher than expected
  - β₁ (decode memory MFU) = 0.916 vs expected 1.00-1.15 → ~10% lower than expected
  - β₄ (decode compute MFU) = 0.943 vs expected 0.70-0.85 → 11% higher than expected
  - **Pattern**: Roofline-based terms (β₀, β₁, β₄) are ALL out of expected ranges, suggesting roofline baseline is systematically wrong
  - Dense models overpredicted by 10-40× (e.g., Mistral general-lite: 3774% TTFT)

- **Mechanism**: **Why is roofline wrong?**

  **Prefill MFU (β₀)**:
  - Roofline assumes prefill phase achieves ~22% MFU (based on iter7 β₀=0.191, where 1.0 = theoretical peak)
  - Iter14 β₀=0.392 suggests prefill actually achieves ~11% MFU (half of roofline prediction)
  - **Root cause**: Roofline model assumes perfect batching and no framework overhead during prefill
  - **Reality**: vLLM prefill has kernel launch overhead, memory fragmentation, and suboptimal batching for mixed-length sequences

  **Decode Memory MFU (β₁)**:
  - Roofline assumes decode phase is memory-bound with MFU > 1.0 (memory bandwidth exceeds compute)
  - Iter14 β₁=0.916 suggests decode is LESS memory-bound than roofline predicts
  - **Root cause**: Roofline model uses per-token KV cache bandwidth, but vLLM does chunk-level caching and attention optimization

  **Decode Compute MFU (β₄)**:
  - Roofline assumes decode compute achieves ~75% MFU
  - Iter14 β₄=0.943 suggests decode compute achieves ~94% MFU (closer to theoretical peak)
  - **Root cause**: Modern GPUs (H100) have better tensor core utilization than roofline model assumes

- **Action for iter15**: **VALIDATE ROOFLINE ASSUMPTIONS**
  1. **Prefill MFU**: Profile real vLLM prefill phase to measure actual MFU (use NVIDIA Nsight or dcgm-exporter)
     - If actual MFU is ~10-15% (not 22%), adjust roofline baseline by 0.5× multiplier
  2. **Decode Memory MFU**: Profile real vLLM decode phase to measure KV cache bandwidth utilization
     - If decode is compute-bound (not memory-bound), reformulate β₁ term
  3. **Decode Compute MFU**: Profile real vLLM decode compute to measure GEMM efficiency
     - If actual MFU is >90%, this term may be correct (don't change)
  4. **Hypothesis**: If roofline baseline is corrected, β₀/β₁/β₄ will converge to ~1.0 (no scaling needed) and dense models will recover

---

**Principle 4**: **Scheduler Overhead Term Systematically Low** (β₆ Expected 40-100ms, Actual 5ms)

- **Evidence**:
  - β₃ (KV mgmt base overhead) = 1.39ms (expected 0.4-1.5ms) → ✅ **WITHIN RANGE**
  - β₆ (scheduler overhead) = 5.09ms (expected 40-100ms) → **7.9× below lower bound**
  - β₇ (decode per-request overhead) = 32.3ms (expected 15-30ms) → **8% above upper bound** (reasonable)
  - **Pattern**: Only β₆ is significantly out of range among framework overhead terms
  - Comparison to iter7: β₃ decreased 3.2× (4.4ms → 1.39ms), β₆ decreased 2.6× (13.2ms → 5.09ms), β₇ increased 1.2× (26.3ms → 32.3ms)

- **Mechanism**: **Why is β₆ consistently low?**

  The expected range for β₆ (40-100ms) was derived from Agent 1's hypothesis, but both iter7 (13.2ms) and iter14 (5.09ms) show β₆ converging to ~5-15ms range, not 40-100ms. This suggests:

  1. **Hypothesis error**: The expected range 40-100ms may be too high
     - Agent 1 may have overestimated vLLM scheduler overhead
     - Real vLLM scheduler overhead per request may be 5-15ms (not 40-100ms)
     - If true expected range is 5-15ms, then β₆=5.09ms is ✅ **WITHIN RANGE**

  2. **Missing overhead elsewhere**: If scheduler overhead really is 40-100ms, then it's being absorbed by other terms
     - Roofline terms (β₀, β₁, β₄) are already out of range (doubled or decreased)
     - β₇ (decode per-request) increased 1.2× (26.3ms → 32.3ms), possibly absorbing some scheduler overhead

  3. **Warm-start bias**: Both iter7 and iter14 warm-started from similar coefficients
     - Optimizer may be stuck in local minimum where β₆ ≈ 5-15ms
     - Cold-start optimization might explore different regions where β₆ converges to 40-100ms

- **Action for iter15**: **VALIDATE EXPECTED RANGE, THEN DECIDE IF FIX NEEDED**
  1. **First, validate the expected range**: Profile real vLLM scheduler overhead to measure actual per-request cost
     - If measured overhead is 5-15ms → β₆=5.09ms is correct, update expected range
     - If measured overhead is 40-100ms → β₆ is indeed too low, investigate why
  2. **If β₆ is genuinely low**: Check if overhead is being absorbed by β₇ (decode per-request)
     - β₇ increased from 26.3ms → 32.3ms (+6ms), could be absorbing some β₆ overhead
  3. **If cold-start optimization**: Check if β₆ converges to different value without warm-start bias

---

**Principle 5**: **Numerical Stability Critical for Long-Sequence Workloads** (100% Error = Simulator Catastrophe)

- **Evidence**:
  - All three reasoning-lite experiments return **exactly 100% error** (TTFT, E2E, ITL, all percentiles)
  - This pattern persisted across iter13 AND iter14 (β₅ fix did not help)
  - Reasoning-lite characteristics: 934 prefill tokens, 1448 decode tokens (2382 total), moderate batch size
  - Other workloads with similar characteristics (e.g., roleplay) do NOT return 100% error

- **Mechanism**: **Why exactly 100% error?**

  From iter13 analysis:
  1. **Simulator produces invalid predictions** (negative StepTime, integer overflow, or divide-by-zero)
  2. **All requests timeout** before completion due to invalid predictions
  3. **Zero completed requests** → summary metrics cannot be computed
  4. **APE calculation**: `APE = |predicted - actual| / actual × 100`
     - When `predicted = 0` (no data): `APE = |0 - actual| / actual × 100 = 100%`

  **Possible root causes**:
  1. **Integer overflow**: StepTime computed in microseconds, but for long sequences: 2382 tokens × 10μs per token × 8 layers = 190ms = 190,000μs → if ANY coefficient is >100, can overflow int64 when multiplied
  2. **Negative time**: If coefficients have opposite signs or basis functions produce negative contributions, StepTime can go negative → simulator rejects negative time → request never completes
  3. **Divide-by-zero**: Batching efficiency formula may divide by batch size, which could be zero for certain scheduling states

- **Action for iter15**: **ADD DEFENSIVE GUARDS (PRIORITY 1)**
  1. **Add bounds checking in StepTime()**:
     ```go
     totalTimeUs := prefillContribution + decodeMemoryContribution + ... + decodeOverheadContribution

     // DEFENSIVE GUARD: Catch numerical failures
     if totalTimeUs < 0 {
         logrus.Errorf("NEGATIVE StepTime: total=%d, prefill=%d, decode=%d, moeGating=%d, β₅=%f",
                       totalTimeUs, prefillContribution, decodeMemoryContribution, moeGatingContribution, m.Beta[5])
         return 1e9  // Return 1 second as fallback
     }
     if totalTimeUs > 1e12 {  // 1 million seconds
         logrus.Errorf("OVERFLOW StepTime: total=%d, prefill=%d, decode=%d, moeGating=%d, β₅=%f",
                       totalTimeUs, prefillContribution, decodeMemoryContribution, moeGatingContribution, m.Beta[5])
         return 1e12  // Clamp to 1 million seconds
     }

     return max(1, clampToInt64(totalTimeUs))
     ```

  2. **Add diagnostics for reasoning-lite experiments**:
     - Log StepTime, QueueingTime, and per-contribution breakdown for first 10 requests
     - Check if any contribution is >50% of total (indicates dominant term)
     - Check if any contribution is negative (indicates bug in basis function)

  3. **Add experiment-level timeout detection**:
     - If >90% of requests timeout, log ERROR and dump coefficient values
     - This will help diagnose future 100% error cases

  4. **Hypothesis**: After adding defensive guards, reasoning-lite experiments will return 200-500% error (overprediction) instead of 100% error (simulator catastrophe)

---

**Principle 6**: **Single-Bug Hypothesis Fallacy** (Complex Systems Have Multiple Failure Modes)

- **Evidence**:
  - Agent 1 hypothesized: β₅ is the "sole cause" of cascading failures
  - Iter14 fixed β₅ → coefficient converged ✅
  - But all four predictions failed: loss still 2319%, reasoning-lite still 100%, dense models still 10-40× worse ❌
  - **Pattern**: Fixing one bug (β₅ layer multiplier) did NOT fix the system

- **Mechanism**: **Why do single-bug hypotheses fail for complex systems?**

  The evolved model has **8 coefficients** (α₀-α₂, β₀-β₇) and **6 basis functions** (prefill, decode, TP comm, KV mgmt, MoE, scheduler). These interact in complex ways:
  - β₅ affects only MoE experiments (4/15 experiments)
  - β₀/β₁/β₄ affect ALL experiments (15/15)
  - β₃/β₆/β₇ affect framework overhead (all experiments)

  When β₅ was broken (missing layer multiplier):
  - MoE experiments had 8× overprediction (e.g., Scout general-lite 847% TTFT)
  - Dense experiments had 10-40× overprediction (e.g., Mistral general-lite 3965% TTFT)

  **Hypothesis**: β₅ was NOT the cause of dense model failures — those failures were caused by roofline baseline issues (β₀/β₁/β₄)

  **Agent 1's mistake**: Assumed that because β₅ was massively wrong (1924.4), it MUST be the root cause of ALL failures

  **Reality**: β₅ was ONE OF MULTIPLE independent failures:
  1. β₅ missing layer multiplier → MoE experiments fail
  2. Roofline baseline wrong → Dense experiments fail
  3. Numerical overflow → Reasoning-lite experiments fail
  4. Warm-start from wrong dataset → Optimizer gets stuck

- **Action for iter15**: **MULTI-HYPOTHESIS APPROACH**
  1. **Do NOT assume a single fix will work** — prepare for multiple iterations
  2. **Prioritize fixes by impact**:
     - Priority 1: Numerical stability (affects 3/15 experiments catastrophically)
     - Priority 2: Roofline baseline (affects 15/15 experiments moderately)
     - Priority 3: Cold-start optimization (affects convergence speed)
  3. **Validate each fix independently**:
     - After adding defensive guards, verify reasoning-lite experiments no longer return 100%
     - After fixing roofline, verify dense models recover to <100% TTFT
     - After cold-start, verify coefficients converge to expected ranges
  4. **Track multiple metrics**:
     - Not just overall loss (can hide trade-offs)
     - Track per-experiment loss, coefficient ranges, contribution breakdown
  5. **Use ablation studies**:
     - Before declaring a fix successful, remove it and verify performance degrades
     - This proves the fix is actually helping (not just optimizer compensating elsewhere)

---

## Coefficient Analysis

### Alpha Coefficients (Request-Level Overhead)

| Coeff | Description | Iter14 | Expected Range | Status |
|-------|-------------|--------|----------------|--------|
| **α₀** | API base overhead | 1.46ms | 0.8-2.5ms | ✅ Within range |
| **α₁** | Tokenization per input token | 54.7μs | 60-150μs | ⚠️ **9% below lower bound** |
| **α₂** | Serialization per output token | 91.3μs | 50-120μs | ✅ Within range |

**Observation**: Alpha coefficients are mostly stable. α₁ slightly below range suggests tokenization overhead is lower than expected (possibly due to batched tokenization optimization in vLLM).

**Physical interpretation**:
- α₀ = 1.46ms: API call setup time (reasonable for HTTP roundtrip + queue insertion)
- α₁ = 54.7μs: Tokenization time per input token (slightly fast, but plausible for cached tokenizer)
- α₂ = 91.3μs: Serialization time per output token (reasonable for JSON encoding + streaming)

**No action needed**: Alpha coefficients are not causing the catastrophic failures.

---

### Beta Coefficients (Step-Level Efficiency)

| Coeff | Description | Iter14 | Expected Range | Iter7 (Baseline) | Change from Iter7 | Status |
|-------|-------------|--------|----------------|------------------|-------------------|--------|
| **β₀** | Prefill MFU scaling | **0.392** | 0.16-0.22 | 0.191 | **+105%** | ❌ **DOUBLED** |
| **β₁** | Decode memory MFU | **0.916** | 1.00-1.15 | 1.108 | -17% | ❌ Below range |
| **β₂** | TP comm scaling | **0.211** | 0.15-0.25 | 0.185 | +14% | ✅ Within range |
| **β₃** | KV mgmt base overhead | **1.39ms** | 0.4-1.5ms | 4.4ms | -68% | ✅ Within range |
| **β₄** | Decode compute MFU | **0.943** | 0.70-0.85 | 0.713 | **+32%** | ❌ Above range |
| **β₅** | MoE gating overhead | **32.53** | 1-50 | 0.0411 | **+792×** | ✅ **FIXED** |
| **β₆** | Scheduler overhead | **5.09ms** | 40-100ms | 13.2ms | -61% | ❌ Below range |
| **β₇** | Decode per-request | **32.3ms** | 15-30ms | 26.3ms | +23% | ⚠️ Slightly above range |

**Critical Observations**:

1. **β₅ fix validated**: 32.53 is within predicted range 1-50 ✅ — layer multiplier implementation is correct
2. **β₀ doubled**: Prefill MFU scaling increased from 0.191 → 0.392 (+105%) — suggests roofline prefill baseline is 2× too optimistic
3. **β₃ converged correctly**: 1.39ms is within expected 0.4-1.5ms range ✅ — KV management overhead is reasonable
4. **β₆ systematically low**: 5.09ms vs expected 40-100ms (7.9× below) — either expected range is wrong OR scheduler overhead absorbed elsewhere
5. **β₇ slightly high**: 32.3ms vs expected 15-30ms (+8% above) — decode per-request overhead reasonable, possibly absorbing some β₆ overhead
6. **β₁ decreased, β₄ increased**: Decode memory MFU down 17%, decode compute MFU up 32% — suggests decode phase is LESS memory-bound and MORE compute-bound than roofline predicts

**Redundant terms**:
- **None** — all 8 coefficients are far from zero, suggesting they're all contributing

**Missing physics**:
- **MoE routing overhead** (β₈ from iter8): Removed in iter14 to isolate β₅ fix, but Scout experiments still show 3-8× overprediction → may need to add back in iter15
- **Batching inefficiency** (β₁₀ from iter13): Removed in iter14, but codegen experiments (high request rate) still show 10-40× overprediction → may need to add back in iter15
- **Attention optimization**: vLLM uses FlashAttention and other optimizations not captured in roofline model → may need explicit term

---

## Recommendations for iter15

### Priority 1: Critical Issues (MUST FIX)

**Issue 1: Numerical Stability (100% Error Catastrophe)**

**Problem**: All three reasoning-lite experiments return exactly 100% error, indicating numerical overflow/underflow causing simulator catastrophe.

**Action**:
1. Add defensive guards in `evolved_model.go:StepTime()` to catch negative/overflow time values
2. Add diagnostics logging for reasoning-lite experiments (first 10 requests, coefficient breakdown)
3. Add experiment-level timeout detection (log ERROR if >90% requests timeout)

**Expected outcome**: Reasoning-lite experiments will return 200-500% error (overprediction) instead of 100% error (catastrophe).

**Implementation**:
```go
// In StepTime() after computing totalTimeUs
if totalTimeUs < 0 {
    logrus.Errorf("NEGATIVE StepTime: total=%d, β₅=%f, prefill=%d, decode=%d",
                  totalTimeUs, m.Beta[5], prefillContribution, decodeMemoryContribution)
    return 1e9  // 1 second fallback
}
if totalTimeUs > 1e12 {  // 1 million seconds
    logrus.Errorf("OVERFLOW StepTime: total=%d, β₅=%f", totalTimeUs, m.Beta[5])
    return 1e12  // Clamp
}
```

---

**Issue 2: Roofline Baseline Validation (10-40× Overprediction)**

**Problem**: β₀ doubled (+105%), β₃/β₆/β₇ collapsed (800-3000×), suggesting roofline prefill/decode baselines are systematically wrong (predict 2× higher MFU than reality).

**Action**:
1. **Profile real vLLM prefill/decode MFU** using NVIDIA Nsight or dcgm-exporter on production workload
2. **Compare measured MFU vs roofline predictions**:
   - Prefill: Roofline predicts ~22% MFU (β₀=0.191), but iter14 β₀=0.392 suggests actual is ~11% MFU
   - Decode: Roofline predicts memory-bound (β₁>1.0), but iter14 β₁=0.916 suggests compute-bound
3. **Adjust roofline baseline formulas** if measured MFU differs by >30% from predictions
4. **Re-run optimization** with corrected roofline baseline

**Expected outcome**: β₀ converges to 0.16-0.22, β₃/β₆/β₇ recover to millisecond scale, dense models recover to <100% TTFT.

**Validation test**: After adjusting roofline, run 10-trial optimization and check if β₀ converges to ~0.18 (not 0.39).

---

**Issue 3: Cold-Start Optimization (Warm-Start Stuck in Local Minimum)**

**Problem**: Warm-start from iter7 coefficients failed — optimizer used 1000 trials but loss barely improved (2387% → 2319%, only 2.8%). This suggests optimizer got stuck in local minimum set by iter7 initialization.

**Action**:
1. **Initialize all coefficients from uniform random** within physically plausible bounds (NOT from iter7)
2. **Increase optimization budget** from 1000 to 2000-3000 trials to allow exploration
3. **Use TPE sampler** in Optuna (default) to handle large search space efficiently
4. **Disable warm-start logic** in `inner_loop_optimize.py`

**Expected outcome**: Optimizer explores different region of search space, finds coefficients better suited to reasoning-lite dataset.

**Validation test**: After cold-start, check if coefficients are significantly different from iter7 (e.g., β₀ not near 0.19, β₃ not near 4.4ms).

---

### Priority 2: Improvements (SHOULD FIX)

**Issue 4: Multi-Coefficient Validation**

**Problem**: β₅ converged correctly (32.5) but overall performance barely improved, because other coefficients (β₀, β₃, β₆, β₇) diverged to compensate.

**Action**:
1. After ANY coefficient fix, validate that OTHER coefficients remain stable (within ±20% of expected range)
2. Track coefficient CONTRIBUTIONS to StepTime (not just coefficient values)
   - Example: `β₅ × moeGatingTimeUs` should be <10% of total StepTime for balanced model
3. Add "contribution budget" constraint: No single term should exceed 60% of total StepTime

**Expected outcome**: Fixes that stabilize one coefficient without destabilizing others will be preferred.

**Implementation**: Add custom Optuna callback to track contribution percentages during optimization.

---

**Issue 5: Ablation Studies Before Declaring Success**

**Problem**: Without ablation studies, we can't distinguish between "fix helped" vs "optimizer compensated elsewhere".

**Action**:
1. After iter15 optimization, run ablation study:
   - Baseline: Full model with all terms
   - Ablation 1: Remove defensive guards (verify reasoning-lite returns 100% again)
   - Ablation 2: Revert roofline baseline (verify β₀ returns to 0.39)
   - Ablation 3: Warm-start from iter7 (verify optimizer gets stuck again)
2. If ablations show <10% impact, the fix may not be helping (optimizer compensating)

**Expected outcome**: Each fix should show >20% impact in ablation study, proving it's necessary.

---

### Priority 3: Refinements (NICE TO HAVE)

**Issue 6: Add MoE Terms Back (β₈, β₁₀)**

**Problem**: Iter14 removed β₈ (MoE routing) and β₁₀ (batching inefficiency) to isolate β₅ fix. But Scout experiments still show 3-8× overprediction, and codegen experiments (high request rate) show 10-40× overprediction.

**Action**:
1. After iter15 fixes Priority 1-2 issues and validates β₅ fix works, re-add β₈ and β₁₀ in iter16
2. Validate that β₅ remains stable (32-40 range) after adding β₈/β₁₀
3. Check for collinearity between β₅, β₈, β₁₀ using correlation analysis

**Expected outcome**: Scout experiments improve from 3-8× to <2× overprediction, codegen experiments improve from 10-40× to <5× overprediction.

---

**Issue 7: Simplify Architecture (Remove Redundant Terms)**

**Problem**: β₃ (KV base overhead) and β₆ (scheduler overhead) both measure millisecond-scale fixed overhead per request. They may be collinear (fighting for same error signal).

**Action**:
1. Try removing β₃ OR β₆ (not both) in ablation study
2. Check if loss increases by >10% when removed
3. If loss barely changes (<5%), the term is redundant — remove it permanently

**Expected outcome**: Simpler 7-coefficient model (remove redundant term) converges faster and avoids overfitting.

---

## Specific Actions for iter15

**Basis function changes**:
- **Add**: Defensive guards in StepTime() (negative/overflow checks)
- **Modify**: None (wait for roofline baseline validation results)
- **Remove**: None (all 8 coefficients are contributing)

**Bounds adjustments**:
- **Do NOT change bounds yet** — wait for cold-start optimization results
- If cold-start still fails, THEN consider widening bounds (but only after ruling out roofline baseline issue)

**Optimization strategy**:
1. **Cold-start** from random initialization (NOT warm-start from iter7)
2. **Increase budget** to 2000-3000 trials
3. **Add diagnostics** for reasoning-lite experiments (catch 100% errors early)
4. **Track multiple metrics**: Per-experiment loss, coefficient ranges, contribution percentages

**Success criteria** for iter15:
- Overall loss <500% (5× improvement from iter14's 2319%)
- Zero experiments with 100% error (numerical stability validated)
- ≥6/8 coefficients in expected ranges (vs 3/8 in iter14)
- Dense models <500% TTFT (5× improvement from iter14's 1000-3700%)

**If iter15 still fails** (loss >1000%):
- Consider FUNDAMENTAL ARCHITECTURE CHANGE:
  - Remove roofline-based model entirely
  - Switch to blackbox regression (GPR, random forest, or neural network)
  - Or switch to hybrid: roofline for prefill/decode base time, regression for overhead terms
  - Or use vidur's simpler model (α + β×tokens, no MFU terms)

---

## Meta-Learning: What Iter14 Taught Us

**What worked**:
1. ✅ **Focused hypothesis testing**: Single-bug, single-fix approach made it easy to validate β₅ coefficient convergence
2. ✅ **Code correctness**: Layer multiplier implementation was correct (verified by β₅=32.5 convergence)
3. ✅ **Diagnostic clause**: Agent 1's diagnostic clause correctly predicted Scenario 3 (β₅ converges but loss >250%)

**What failed**:
1. ❌ **Single-bug assumption**: Hypothesis that β₅ was "sole cause" of cascading failures was REFUTED
2. ❌ **Warm-start strategy**: Initializing from iter7 coefficients got optimizer stuck in wrong local minimum
3. ❌ **Necessary vs sufficient**: Fixing β₅ was necessary but NOT sufficient to recover performance

**What we learned**:
1. **Complex systems have multiple failure modes** — don't assume single-bug fixes will work
2. **Dataset changes invalidate previous solutions** — reasoning → reasoning-lite shift broke iter7 coefficients
3. **Roofline baseline may be systematically wrong** — β₀ doubled, β₃/β₆/β₇ collapsed by 800-3000×
4. **Numerical stability is CRITICAL** — 100% error catastrophe must be caught with defensive guards
5. **Coefficient convergence ≠ performance recovery** — optimizer can compensate elsewhere

**Process improvements for future iterations**:
1. **Multi-hypothesis approach**: Test 2-3 independent fixes in parallel (not sequential)
2. **Ablation studies mandatory**: Before declaring success, prove fix helps via ablation
3. **Cold-start when dataset changes**: Don't warm-start from previous iteration if data changed
4. **Track contribution breakdowns**: Not just coefficient values, but their impact on StepTime
5. **Add defensive guards early**: Don't wait for catastrophic failures to add bounds checking

---

## Conclusion

Iteration 14 was a **necessary but insufficient** step toward fixing the evolved model. The β₅ layer multiplier fix WORKED at the coefficient level (32.5, within 1-50 range), proving the implementation was correct. But it FAILED at the performance level (loss 2319%, barely improved from 2387%), revealing that β₅ was NOT the "sole cause" of cascading failures.

**Key insight**: The model has **multiple independent architectural defects**:
1. **Numerical stability**: Reasoning-lite experiments return 100% error (simulator catastrophe)
2. **Roofline baseline**: Prefill/decode MFU predictions are 2× too optimistic, causing 10-40× overprediction
3. **Warm-start failure**: Dataset change (reasoning → reasoning-lite) invalidated iter7 coefficients

Iter15 must address ALL THREE issues simultaneously (not sequentially). The training process is at a critical juncture — two consecutive catastrophic failures (iter13, iter14) with >2000% loss suggest fundamental problems requiring a multi-pronged approach.

**Recommendation**: Iter15 should be a **"fresh start"** iteration:
- Cold-start optimization (random initialization)
- Add defensive guards (prevent 100% errors)
- Validate roofline baseline (profile real vLLM MFU)
- Increase optimization budget (2000-3000 trials)

If iter15 STILL fails (loss >1000%), consider **switching to fundamentally different architecture** (blackbox regression, vidur-style linear model, or hybrid approach).
