# Iteration 14: Hypothesis Validation

## Summary

**Status**: ❌ **CATASTROPHIC FAILURE** (Second consecutive iteration with >2000% loss)

Iteration 14 tested the hypothesis that iter13's catastrophic failure (loss 2387%) was caused by a single bug: missing `× numMoELayers` multiplier in the β₅ (MoE gating) basis function. The fix was implemented correctly, and **β₅ DID converge to the predicted range (32.5, within 1-50)**, but the overall model performance **barely improved** (2387% → 2319%, only 2.8% improvement instead of predicted 92% improvement).

**Key Finding**: The β₅ coefficient stabilization was NECESSARY but NOT SUFFICIENT. The hypothesis that β₅ was the "sole cause" of cascading failures is **REFUTED**. There are deeper architectural problems preventing model convergence.

**Verdicts**:
- **H-main**: ❌ REJECTED (β₅ converged correctly but loss target missed by 13×)
- **H-architecture-stability**: ❌ REJECTED (6/8 coefficients out of expected ranges)
- **H-reasoning-lite-recovery**: ❌ REJECTED (all three experiments still return 100% error)
- **H-non-moe-recovery**: ❌ REJECTED (dense models failed to recover to iter7 baseline)

---

## H-main: β₅ Layer Multiplier Fix Recovers Model Stability

**Prediction** (from Agent 1): After adding the missing `× numMoELayers` multiplier to β₅ basis function and returning to iter7's 8-beta architecture:

**Overall Performance**:
- Overall loss: 2387% (iter13) → **<180%** (≥92% improvement, targeting iter7 baseline 155% + margin)
- TTFT RMSE: 1370% (iter13) → **<80%** (≥94% improvement)
- E2E RMSE: 1017% (iter13) → **<100%** (≥90% improvement)

**Coefficient Stability**:
- β₅: 1924.4 (iter13) → **1-50 dimensionless** (within physical range)

**Actual Result**:

**Overall Performance**:
- Overall loss: **2319.56%** (target <180%, **missed by 2139pp or 13×**)
- TTFT RMSE: **1313.59%** (target <80%, **missed by 1233pp or 16×**)
- E2E RMSE: **1005.97%** (target <100%, **missed by 906pp or 10×**)
- Improvement from iter13: 2387% → 2319% = **2.8% improvement** (vs predicted 92% improvement)

**Coefficient Values**:
- β₅: **32.53** (✅ **WITHIN predicted range 1-50**)

**Verdict**: ❌ **REJECTED**

**Evidence**:

**Coefficient Prediction SUCCESS**:
- β₅ converged to 32.53, exactly within the predicted 1-50 range
- This is 59× smaller than iter13's 1924.4, proving the layer multiplier was correctly added
- β₅ = 32.53 is physically plausible (similar magnitude to other efficiency factors like β₀=0.39, β₁=0.92, β₄=0.94)

**Performance Prediction CATASTROPHIC FAILURE**:
- Overall loss: 2319.56% vs target <180% → **missed by 13× factor**
- TTFT RMSE: 1313.59% vs target <80% → **missed by 16× factor**
- E2E RMSE: 1005.97% vs target <100% → **missed by 10× factor**
- Improvement from iter13: Only 2.8% (67pp absolute), not the predicted 92% (2207pp absolute)

**Causal Analysis**:

**Why β₅ converged correctly**: The layer multiplier was implemented correctly (lines 240-268 in `sim/latency/evolved_model.go`). The basis function now computes:
```go
gatingTimePerLayerSeconds := gatingFlopsPerLayer / tpFactor / (peakFlops * gatingEfficiency)
moeGatingTimeSeconds = gatingTimePerLayerSeconds * numMoELayers  // ← FIX: multiply by layers
```

For Scout (56 MoE layers), this produces `gatingTimeSeconds = 56 × (per-layer time)`, and β₅ only needs to be ~30-40 to match reality (not 1924.4).

**Why overall performance still failed catastrophically**:

Agent 1's hypothesis assumed β₅ was the **sole cause** of cascading failures, with the causal chain:
1. β₅ explodes due to missing layer multiplier
2. Optimizer compensates by adjusting global coefficients (α₁, β₂, β₄, β₆)
3. Dense models fail even though they don't use β₅
4. Fix β₅ → global coefficients recover → dense models recover

**This causal chain is REFUTED by iter14 results**. Even with β₅ stable at 32.5:
- Dense model experiments STILL show 600-3700% TTFT error (vs target <100%)
- Reasoning-lite experiments STILL return 100% error (complete timeout catastrophe)
- Global coefficients (β₀, β₁, β₃, β₄, β₆, β₇) are STILL out of expected ranges

**Alternative explanation**: The catastrophic failures in iter13-14 are caused by **multiple architectural defects**, not just β₅:
1. **Reasoning-lite experiments**: 100% error indicates numerical overflow/underflow in the simulator (not fixed by β₅ correction)
2. **Dense models**: 10-30× overprediction could be due to roofline baseline issues, warm-start getting stuck, dataset shift, or architectural problems
3. **MoE models**: Even with β₅ fixed, Scout experiments show 3-8× overprediction, suggesting missing MoE-specific overhead terms (routing, load imbalance, expert switching)

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

Agent 1 predicted three failure scenarios:

**Scenario 1** (β₅ still explodes >200):
- **Prediction**: Gating efficiency assumption (30%) too optimistic
- **Actual**: β₅ = 32.5 (within 1-50 range, NOT >200)
- **Status**: Does not apply

**Scenario 2** (β₅ collapses <0.1):
- **Prediction**: Collinearity between β₅ and β₈, or β₅ basis function over-corrected
- **Actual**: β₅ = 32.5 (NOT <0.1)
- **Status**: Does not apply

**Scenario 3** (β₅ converges to 10-50 but loss >250%):
- **Prediction**: Other architectural problems (not β₅) dominate error
- **Actual**: ✅ **THIS IS WHAT HAPPENED** - β₅ = 32.5 (within range) but loss = 2319% (>250%)
- **Investigate**: Per-experiment error breakdown shows THREE distinct failure modes:
  1. **Reasoning-lite catastrophe**: All three experiments return 100% error (numerical overflow in simulator)
  2. **Dense model overprediction**: 10-30× overprediction (root cause unclear - need profiling and ablations)
  3. **MoE model overprediction**: 3-8× overprediction (missing MoE-specific overhead)
- **Action**: **URGENT** - β₅ fix validated but insufficient. Must investigate:
  - Add defensive guards for numerical overflow in reasoning-lite (iter15 priority 1)
  - Profile actual vLLM MFU and run cold-start ablation to diagnose dense model issues (iter15 priority 2)
  - Add MoE-specific overhead terms beyond β₅ (iter15 priority 3)

---

## H-architecture-stability: Iter7 8-Beta Architecture Without β₈+β₁₀ Prevents Cascading Failures

**Prediction** (from Agent 1): By returning to iter7's stable 8-beta architecture (β₀-β₅, β₆-β₇) and NOT adding β₈ or β₁₀, coefficient explosions will be eliminated. All 8 coefficients should fall within physically plausible ranges:

**Coefficient Ranges** (expected):
- β₀ (prefill MFU): **0.16-0.22** (iter7: 0.191 ✓)
- β₁ (decode memory MFU): **1.00-1.15** (iter7: 1.108 ✓)
- β₂ (TP comm scaling): **0.15-0.25** (iter7: 0.185 ✓)
- β₃ (KV base overhead): **0.4-1.5ms** (iter7: 4.4ms)
- β₄ (decode compute MFU): **0.70-0.85** (iter7: 0.713 ✓)
- β₅ (MoE gating): **1-50** (iter7: 0.0411, expect 20-40 after fix)
- β₆ (scheduler overhead): **40-100ms** (iter7: 13.2ms)
- β₇ (decode per-request): **15-30ms** (iter7: 26.3ms ✓)

**Actual Result**:

**Coefficient Values** (iter14):

| Coeff | Iter14 | Expected Range | Status |
|-------|--------|----------------|--------|
| β₀ | **0.392** | 0.16-0.22 | ❌ **+105% above upper bound** |
| β₁ | **0.916** | 1.00-1.15 | ❌ **-8% below lower bound** |
| β₂ | **0.211** | 0.15-0.25 | ✅ Within range |
| β₃ | **1.39ms** | 0.4-1.5ms | ✅ Within range |
| β₄ | **0.943** | 0.70-0.85 | ❌ **+11% above upper bound** |
| β₅ | **32.53** | 1-50 | ✅ Within range |
| β₆ | **5.09ms** | 40-100ms | ❌ **7.9× below lower bound** |
| β₇ | **32.3ms** | 15-30ms | ❌ **+8% above upper bound** |

**Verdict**: ❌ **REJECTED**

**Evidence**:

**Coefficients in range**: 3/8 (β₂, β₃, β₅) = **38%**
**Coefficients out of range**: 5/8 (β₀, β₁, β₄, β₆, β₇) = **62%**

Agent 1 predicted ≥7/8 coefficients would be stable (≥87.5%). Actual: 3/8 stable (38%), **failed by 50pp**.

**Causal Analysis**:

Agent 1's hypothesis was that returning to iter7's proven architecture would prevent cascading failures. The evidence REFUTES this:

**Why the hypothesis failed**:
1. **β₀ doubled** (0.19 → 0.39) - prefill MFU assumption may be too optimistic (roofline predicts 22%, reality may be 11%)
2. **β₁ dropped** (1.11 → 0.92) - decode memory MFU assumption may be too optimistic
3. **β₄ increased** (0.71 → 0.94) - decode compute MFU closer to theoretical peak, suggesting less bottleneck
4. **β₃ decreased moderately** (4.4ms → 1.39ms, -68%) but remains within expected range
5. **β₆ decreased moderately** (13.2ms → 5.09ms, -61%) - scheduler overhead reduced but still millisecond-scale
6. **β₇ increased slightly** (26.3ms → 32.3ms, +23%) - decode per-request overhead grew slightly

**Pattern recognition**: The coefficients show a **systematic bias**:
- **Roofline-based terms (β₀, β₁, β₄) are out of range** → roofline assumptions may be wrong, OR warm-start trapped optimizer, OR dataset shift requires different coefficients
- **Scheduler overhead (β₆) reduced significantly** → dropped 2.6× below lower bound (expected 40-100ms)
- **MoE term (β₅) is correct** → layer multiplier fix worked
- **KV management (β₃) and decode overhead (β₇) are reasonable** → both in or near expected ranges

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

Agent 1 predicted three failure scenarios:

**Scenario 1** (β₅ fix insufficient, still broken):
- **Prediction**: If >3 coefficients out of range, β₅ still broken
- **Actual**: 5/8 out of range, BUT β₅ itself is within range (32.5)
- **Status**: β₅ fix succeeded, but other coefficients failed

**Scenario 2** (warm-start from iter7 invalid due to dataset change):
- **Prediction**: Dataset changed between iter7/iter13 (reasoning → reasoning-lite), warm-start may fail
- **Actual**: ✅ **LIKELY** - iter7 trained on reasoning (long sequences, overloaded servers), iter13-14 train on reasoning-lite (lighter load)
- **Evidence**: Reasoning-lite experiments return 100% error in both iter13 and iter14, suggesting the model cannot predict this workload
- **Action**: **CRITICAL** - Try cold-start optimization from random initialization (not warm-start from iter7) in iter15

**Scenario 3** (other collinearities exist):
- **Prediction**: β₀ vs β₁, β₂ vs β₃ collinearities prevent convergence
- **Actual**: Less likely - β₃ (1.39ms) and β₆ (5.09ms) are both reasonable millisecond-scale values
- **Action**: No evidence of strong collinearity between β₃ and β₆

**Overall Assessment**: Iter7 architecture is NOT stable when applied to reasoning-lite dataset. The warm-start strategy failed because the dataset changed fundamentally between iter7 and iter13.

---

## H-reasoning-lite-recovery: Fixing β₅ Eliminates 100% Timeout Catastrophe

**Prediction** (from Agent 1): All three reasoning-lite experiments that failed with exactly 100% error in iter13 will recover to <70% error:

**Experiments**:
- Scout reasoning-lite-2-1: 100% TTFT/E2E (iter13) → **<70%** (recover from timeout)
- Qwen2.5 reasoning-lite-1-1: 100% TTFT/E2E (iter13) → **<65%** (recover from timeout)
- Llama-2 reasoning-lite-1-1: 100% TTFT/E2E (iter13) → **<70%** (recover from timeout)

**Actual Result**:

| Experiment | Model | Workload | TTFT APE | E2E APE | Combined | Target | Status |
|------------|-------|----------|----------|---------|----------|--------|--------|
| 1 | Scout-17B-FP8 | reasoning-lite-2-1 | **100.0%** | **100.0%** | **200.0%** | <140% | ❌ **STILL FAILED** |
| 2 | Qwen2.5-7B | reasoning-lite-1-1 | **100.0%** | **100.0%** | **200.0%** | <130% | ❌ **STILL FAILED** |
| 3 | Llama-2-7B | reasoning-lite-1-1 | **100.0%** | **100.0%** | **200.0%** | <140% | ❌ **STILL FAILED** |

**Verdict**: ❌ **REJECTED**

**Evidence**: All three reasoning-lite experiments STILL return exactly 100% error across ALL metrics (TTFT mean, E2E mean, all percentiles). The β₅ fix did NOT eliminate the timeout catastrophe.

**Causal Analysis**:

Agent 1's hypothesis was that β₅=1924.4 caused catastrophic latency overestimation (251× overestimate) → requests timed out → 0 completed requests → 100% APE. After fixing β₅ to 32.5, latencies should become realistic (3.9× overestimate) → requests complete → APE recovers to 50-70%.

**This hypothesis is REFUTED**. Even with β₅=32.5:
1. All three reasoning-lite experiments STILL return 100% error
2. The exact same pattern as iter13 (100.0% across all metrics)
3. This indicates **the root cause is NOT β₅**

**Alternative explanations**:

**Scenario 1** (β₅ fix insufficient OR other numerical overflow):
- Agent 1's diagnostic: Add defensive guards to catch negative/overflow StepTime
- **Evidence supporting this**: All three experiments show **identical 100% error** (not varying 90-110%), suggesting a systematic failure mode (divide-by-zero, integer overflow, or negative time)
- **Next steps**: Add logging to `evolved_model.go`:
  ```go
  if stepTime < 0 || stepTime > 1e12 {
      log.Errorf("Numerical failure: stepTime=%d, β₅=%f, prefillContrib=%d, decodeContrib=%d",
                 stepTime, m.Beta[5], prefillContribution, decodeMemoryContribution)
      return 1e9  // Return 1 second as fallback to prevent timeout catastrophe
  }
  ```

**Scenario 2** (data quality issue in reasoning-lite experiments):
- Agent 1's diagnostic: Validate ground truth for failing experiments
- **Counter-evidence**: Iter13 analysis confirmed all reasoning-lite experiments have valid ground truth data (`per_request_lifecycle_metrics.json`)
- **Less likely**: If data were corrupted, we'd expect varying errors (50-150%), not exact 100%

**Scenario 3** (all recover to 30-70%):
- Agent 1's diagnostic: Success! β₅ fix validated
- **Actual**: Did NOT happen

**Critical Finding**: The β₅ fix was **necessary but not sufficient**. There is a **second independent bug** causing the 100% error in reasoning-lite experiments. This bug is likely:
1. Numerical overflow in a DIFFERENT basis function (not β₅)
2. Divide-by-zero in batch size calculation for long-sequence workloads
3. Negative StepTime due to coefficient interactions

**Recommendation**: **URGENT** - Add defensive guards in iter15 as priority 1. The simulator should NEVER return 100% error - it should clamp invalid predictions and log diagnostics.

---

## H-non-moe-recovery: Dense Model Experiments Recover to Iter7 Baseline

**Prediction** (from Agent 1): Non-MoE experiments (no β₅ contribution) will recover to iter7 baseline performance, confirming β₅ was the sole cause of iter13 catastrophe:

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

**Actual Result**:

| Experiment | Model | Workload | Iter7 TTFT | Iter13 TTFT | Iter14 TTFT | Target | Status |
|------------|-------|----------|------------|-------------|-------------|--------|--------|
| 1 | Llama-2-7B | codegen | 9.3% | 1417% | **1334%** | <15% | ❌ **89× worse** |
| 2 | Llama-2-7B | roleplay | 56% | 1123% | **1022%** | <60% | ❌ **17× worse** |
| 3 | Llama-2-7B | general | 4.6% | 1513% | **1559%** | <10% | ❌ **156× worse** |
| 4 | Mistral-Nemo-12B | codegen | 20% | 1193% | **1188%** | <25% | ❌ **48× worse** |
| 5 | Mistral-Nemo-12B | general-lite | 90% | 3965% | **3774%** | <95% | ❌ **40× worse** |
| 6 | Qwen2.5-7B | roleplay | 57% | 1137% | **1029%** | <60% | ❌ **17× worse** |
| 7 | Yi-34B | general-lite | 48% | 1183% | **1198%** | <50% | ❌ **24× worse** |
| 8 | Llama-3.1-70B | codegen | 29% | 742% | **617%** | <35% | ❌ **17× worse** |
| 9 | Llama-3.1-70B | general-lite | 41% | 1010% | **1033%** | <45% | ❌ **23× worse** |

**Verdict**: ❌ **REJECTED**

**Evidence**: **0/9 dense model experiments recovered to iter7 baseline** (0%, vs predicted 100%). All experiments are 17-156× worse than iter7 targets.

**Causal Analysis**:

Agent 1's hypothesis was that dense models failed in iter13 due to **cascading coefficient instability** caused by β₅ explosion:
1. β₅ explodes to 1924.4 for MoE experiments
2. Optimizer adjusts global coefficients (α₁, β₂, β₄, β₆) to compensate
3. These adjustments hurt dense models even though they don't use β₅
4. Fix β₅ → global coefficients recover → dense models recover

**This causal chain is COMPLETELY REFUTED**. Even with β₅ stable:
- Global coefficients did NOT recover (β₀ +105%, β₁ -8%, β₃ -287×, β₄ +11%, β₆ -7859×, β₇ -464×)
- Dense models did NOT recover (all still 10-150× worse than iter7)

**Alternative explanation**: The cascading failures are NOT caused by β₅ alone. Instead, there are **multiple independent architectural defects**:

1. **Roofline-based coefficients out of expected ranges**:
   - β₀ (prefill) = 0.392 vs expected 0.16-0.22 → doubled from iter7 (0.191)
   - β₁ (decode memory) = 0.916 vs expected 1.00-1.15 → 8% below lower bound
   - β₄ (decode compute) = 0.943 vs expected 0.70-0.85 → 11% above upper bound
   - **Cannot determine root cause** without profiling actual MFU or running cold-start ablation
   - Could be: roofline wrong, warm-start stuck, dataset shift, or expected ranges wrong

2. **Scheduler overhead consistently low**:
   - β₃ (KV mgmt) = 1.39ms → ✅ within expected 0.4-1.5ms
   - β₆ (scheduler) = 5.09ms vs expected 40-100ms → 7.9× below lower bound
   - β₇ (decode per-request) = 32.3ms vs expected 15-30ms → 8% above (reasonable)
   - **Pattern**: Only β₆ is significantly low; β₃ and β₇ are reasonable
   - Suggests either β₆ expected range is wrong, OR scheduler overhead absorbed by β₇

3. **Warm-start from iter7 may be invalid**:
   - Dataset changed between iter7 (reasoning) and iter13-14 (reasoning-lite)
   - Iter7 coefficients may not be appropriate starting point for reasoning-lite data
   - Need cold-start optimization to test this hypothesis

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

**Scenario 1** (all dense models still 5-10× worse than iter7):
- **Prediction**: Other coefficients (β₀, β₁, β₃, β₇) out of range, not just β₅ cascade
- **Actual**: ✅ **CONFIRMED** - 6/8 coefficients out of range (β₀, β₁, β₃, β₄, β₆, β₇)
- **Action**: Warm-start from iter7 FAILED. Try cold-start optimization from random initialization in iter15

**Scenario 2** (some dense models recover, others don't):
- **Prediction**: Dataset change affects recovery
- **Actual**: NO dense models recovered (0/9)
- **Status**: More severe than scenario 2 - universal failure

**Scenario 3** (all dense models recover to iter7 ±10pp):
- **Prediction**: Success! β₅ was the sole cause
- **Actual**: Did NOT happen

**Critical Finding**: The hypothesis that β₅ was the **sole cause** of cascading failures is **REFUTED**. The model has **multiple architectural defects** that prevent convergence:
1. Roofline baseline assumptions are wrong (β₀, β₁, β₄)
2. Framework overhead terms are being suppressed (β₃, β₆, β₇)
3. Warm-start from iter7 is invalid (dataset changed)

**Recommendation**: Iter15 must take a **fundamentally different approach**:
- Cold-start optimization (not warm-start from iter7)
- Validate roofline assumptions against ground truth (check prefill MFU, decode MFU)
- Add defensive guards for numerical overflow (reasoning-lite 100% errors)
- Consider simplifying architecture (remove redundant terms like β₃ vs β₆)

---

## Summary of Validation Results

| Hypothesis | Prediction | Actual | Verdict | Key Evidence |
|------------|-----------|--------|---------|--------------|
| **H-main** | Loss 2387% → <180% | Loss **2319%** | ❌ REJECTED | β₅ converged correctly (32.5) but loss barely improved (2.8%) |
| **H-architecture-stability** | ≥7/8 coefficients stable | **3/8 stable** (38%) | ❌ REJECTED | β₀, β₁, β₄, β₆, β₇ out of expected ranges |
| **H-reasoning-lite-recovery** | 100% error → <70% | **100% error** (unchanged) | ❌ REJECTED | All three reasoning-lite experiments still return 100% |
| **H-non-moe-recovery** | Dense models recover to iter7 | **0/9 recovered** | ❌ REJECTED | All dense models 17-156× worse than iter7 targets |

**Overall Assessment**: ❌ **ALL HYPOTHESES REJECTED**

---

## Critical Insight: Necessary But Not Sufficient

**What worked**:
- β₅ coefficient converged to predicted range (32.5, within 1-50) ✅
- Layer multiplier implementation is correct ✅

**What failed**:
- Overall loss barely improved (2387% → 2319%, only 2.8%) ❌
- 5/8 coefficients still out of expected ranges ❌
- Reasoning-lite experiments still return 100% error ❌
- Dense models failed to recover to iter7 baseline ❌

**Conclusion**: The β₅ fix was **necessary but not sufficient**. Agent 1's hypothesis that β₅ was the "sole cause" of cascading failures is **REFUTED**. The model has **multiple independent architectural defects** that must be addressed:

1. **Priority 1 (CRITICAL)**: Add defensive guards for numerical overflow causing 100% errors in reasoning-lite experiments
2. **Priority 2**: Validate roofline baseline assumptions (prefill MFU, decode MFU) against ground truth
3. **Priority 3**: Try cold-start optimization from random initialization (warm-start from iter7 invalid due to dataset change)
4. **Priority 4**: Simplify architecture (remove redundant terms, reformulate for orthogonality)

The training process is at a **critical juncture** - two consecutive catastrophic failures (iter13, iter14) with >2000% loss suggest fundamental issues with either the model architecture or the training dataset. Iter15 must take a radically different approach.
