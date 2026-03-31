# Iteration 13: Catastrophic Failure Analysis

## Executive Summary

**Status**: ❌ **CATASTROPHIC FAILURE** (loss 2387%, 15.4× worse than iter7 baseline of 155%)

Iteration 13 attempted to return to the stable iter7 baseline (155% loss) and incrementally add β₈ (MoE routing) and β₁₀ (batching inefficiency). Instead, it produced the **worst results in training history**, exceeding even iter10's 4267% loss when measured by the magnitude of overprediction.

**Key Findings**:
1. **β₅ explosion**: MoE gating coefficient exploded to 1924.4 (46,800× increase from iter7's 0.0411), causing massive overprediction
2. **Complete simulator failure**: Three reasoning-lite experiments returned exactly 100% error (simulator produced garbage predictions)
3. **Hypothesis refutation**: All three H-main predictions failed catastrophically
4. **Warm-start strategy failed**: Returning to iter7 baseline did NOT provide stability
5. **Dataset changed**: Between iter7 and iter13, reasoning experiments were converted to reasoning-lite (shorter sequences, different workload characteristics)

---

## Overall Performance vs Baseline

| Metric | Iter7 (Baseline) | Iter13 | Change | Status |
|--------|------------------|--------|--------|--------|
| **Overall Loss** | 155.37% | **2387.25%** | +1436× | ❌ CATASTROPHIC |
| **TTFT RMSE** | 64.04% | **1370.44%** | +21.4× | ❌ CATASTROPHIC |
| **E2E RMSE** | 91.33% | **1016.80%** | +11.1× | ❌ CATASTROPHIC |

**Observation**: Every metric deteriorated by an order of magnitude or more.

---

## Hypothesis Evaluation

### H-main: Iter7 Baseline + β₁₀ Recovers Sequence-Length Prediction

**Prediction**: Overall loss 155% → <140% (≥15pp improvement)

**Result**: Overall loss 155% → **2387%** (+1436pp deterioration)

**Status**: ❌ **REFUTED CATASTROPHICALLY**

**Evidence**:
- Predicted codegen improvements → Actual: codegen catastrophic (1200-2700% TTFT)
- Predicted Scout general-lite <85% TTFT → Actual: 847% TTFT (10× prediction)
- Predicted stability → Actual: worst iteration in training history

**Root Cause**: β₅ (MoE gating) exploded to 1924.4, causing massive overprediction across all experiments. The hypothesis that β₁₀ would recover sequence-length prediction was invalidated by the β₅ explosion absorbing all error signal.

---

### H-beta5-anomaly: β₅ Will Increase 9000-24000× to Recover from Collapse

**Prediction**: β₅: 0.0411 (iter7) → 300-1000 dimensionless (9000-24000× increase)

**Result**: β₅: 0.0411 (iter7) → **1924.4** (46,800× increase)

**Status**: ⚠️ **PARTIALLY CORRECT BUT EXPLODED BEYOND BOUNDS**

**Evidence**:
- Predicted range: 300-1000 → Actual: 1924.4 (1.9× upper bound)
- Increase magnitude: Predicted 9000-24000× → Actual: 46,800× (2× upper prediction)
- Physical interpretation: β₅ is dimensionless scaling factor for MoE gating roofline estimate
  - Basis function computes ~0.04μs per layer (single-layer gating)
  - With β₅=1924.4: contribution = 1924.4 × 0.04μs ≈ **77μs per layer**
  - Scout has ~56 MoE layers → **4.3ms total gating overhead** (physically implausible)

**Diagnostic**: The β₅ explosion suggests one of three fundamental problems:
1. **MoE gating basis function underestimates by 2× factor** (missing layers or operations)
2. **β₅ absorbing non-gating overhead** (batching delays, routing, or scheduler overhead)
3. **Collinearity with β₈ and β₁₀** causing optimizer to inflate β₅ to compensate

---

### H-baseline-insight: Learn from Vidur's Success

**Prediction**: Codegen workloads will improve most (currently worst performers), framework overhead terms (β₆, β₇, β₁₀) will make model behave like vidur

**Result**: Codegen workloads catastrophically failed (1200-2700% TTFT), framework terms did NOT help

**Status**: ❌ **REFUTED COMPLETELY**

**Evidence**:
- Llama-2 codegen: iter7 9.3% TTFT → iter13 **1417% TTFT** (152× worse)
- Mistral codegen: iter7 20% TTFT → iter13 **1193% TTFT** (60× worse)
- Llama-3.1 codegen: iter7 29% TTFT → iter13 **742% TTFT** (25× worse)

**Root Cause**: Instead of reducing TTFT overestimation (as vidur does), iter13 amplified it by 25-152×. The β₅ explosion caused the model to massively overpredict all experiments, not just codegen.

---

## Coefficient Analysis

### Alpha Coefficients (Request-Level Overhead)

| Coeff | Iter7 | Iter13 | Change | Expected Range | Status |
|-------|-------|--------|--------|----------------|--------|
| **α₀** | 1.32ms | **0.939ms** | -29% | 0.8-2.5ms | ✅ In range |
| **α₁** | 118μs | **54.3μs** | -54% | 60-150μs | ⚠️ Below range |
| **α₂** | 90.5μs | **103μs** | +14% | 50-120μs | ✅ In range |

**Observation**: α₁ (tokenization overhead) dropped below physical range, suggesting optimizer compensating for β₅ overprediction by reducing input-dependent overhead.

---

### Beta Coefficients (Step-Level Efficiency)

| Coeff | Description | Iter7 | Iter13 | Change | Expected Range | Status |
|-------|-------------|-------|--------|--------|----------------|--------|
| **β₀** | Prefill MFU | 0.191 | **0.187** | -2% | 0.16-0.22 | ✅ Stable |
| **β₁** | Decode memory MFU | 1.108 | **1.219** | +10% | 1.00-1.15 | ⚠️ Above range |
| **β₂** | TP comm scaling | 0.185 | **0.362** | +96% | 0.15-0.25 | ❌ **EXPLOSION** |
| **β₃** | KV mgmt base overhead | 4.4ms | **3.4ms** | -23% | 0.4-1.5ms | ⚠️ Above range |
| **β₄** | Decode compute MFU | 0.713 | **0.562** | -21% | 0.70-0.85 | ⚠️ Below range |
| **β₅** | MoE gating overhead | 0.0411 | **1924.4** | +46,800× | 300-1000 | ❌ **CATASTROPHIC EXPLOSION** |
| **β₆** | Scheduler overhead | 13.2ms | **10.0ms** | -24% | 40-100ms | ⚠️ Below range |
| **β₇** | Decode per-request | 26.3ms | **30.0ms** | +14% | 15-30ms | ✅ Stable |
| **β₈** | MoE routing | 30μs (iter8) | **13.3μs** | -56% | 25-80μs | ⚠️ Below range |
| **β₁₀** | Batching inefficiency | NEW | **0.100μs** | N/A | 0.1-1.0μs | ✅ In range |

**Critical Observations**:
1. **β₅ explosion is catastrophic**: 1924.4 is 46,800× iter7 and 1.9× predicted upper bound
2. **β₂ doubled**: TP communication scaling went from 0.185 → 0.362 (96% increase)
3. **β₈ halved**: MoE routing went from 30μs → 13.3μs (56% decrease), opposite of expected behavior
4. **β₁₀ converged correctly**: 0.100μs is exactly at lower bound of expected 0.1-1.0μs range, suggesting formula is correct but insufficient

---

## Per-Experiment Disaster

### Worst Failures (Top 5)

| Experiment | Model | Workload | TTFT APE | E2E APE | Combined Loss | Iter7 Comparison |
|------------|-------|----------|----------|---------|---------------|------------------|
| 1 | Mistral-Nemo-12B | general-lite-2-1 | **3965%** | **2893%** | **6858%** | 90 → 6858 (76× worse) |
| 2 | Llama-2-7B | codegen | **1417%** | **1242%** | **2659%** | 95 → 2659 (28× worse) |
| 3 | Yi-34B | general-lite-2-1 | **1183%** | **1081%** | **2264%** | 142 → 2264 (16× worse) |
| 4 | Mistral-Nemo-12B | codegen-1-1 | **1193%** | **986%** | **2179%** | 105 → 2179 (21× worse) |
| 5 | Qwen2.5-7B | roleplay-1-1 | **1137%** | **953%** | **2090%** | 131 → 2090 (16× worse) |

### Complete Simulator Failure (100% Error)

| Experiment | Model | Workload | TTFT APE | E2E APE | Combined Loss | Notes |
|------------|-------|----------|----------|---------|---------------|-------|
| 1 | Scout-17B-FP8 | reasoning-lite-2-1 | **100%** | **100%** | **200%** | Exact 100% across all percentiles |
| 2 | Qwen2.5-7B | reasoning-lite-1-1 | **100%** | **100%** | **200%** | Exact 100% across all percentiles |
| 3 | Llama-2-7B | reasoning-lite-1-1 | **100%** | **100%** | **200%** | Exact 100% across all percentiles |

**Observation**: Three reasoning-lite experiments returned exactly 100% error for ALL metrics (mean, p90, p99 TTFT/E2E/ITL). This suggests a **catastrophic failure mode** in the evolved model:
- Possible numerical overflow/underflow in β₅ computation
- Divide-by-zero in batching inefficiency formula (β₁₀)
- Simulator returning zero or negative latency predictions

**CRITICAL CONTEXT**: Between iter7 and iter13, the dataset changed — reasoning experiments were converted to **reasoning-lite** (lighter load, different workload characteristics). These are **NEW experiments** that iter7 never trained on:
- Iter7 had: reasoning workloads (original)
- Iter13 has: reasoning-lite workloads (lighter load version - same sequence lengths, different rate/duration characteristics)
- **Verified**: All 3 reasoning-lite experiments have valid ground truth data and are included in training (15/15 experiments)

**Implication**: The 100% error is NOT due to missing training data. Instead, it indicates **catastrophic numerical failure** in the model:
- Likely numerical overflow in β₅ computation (1924.4 × 0.04μs × 56 layers = 4.3ms may overflow int64 when converted to nanoseconds)
- OR divide-by-zero in β₁₀ formula for long sequences with small batch sizes
- OR negative StepTime returned due to coefficient interactions

**Diagnostic Action Required**:
1. ✅ **VERIFIED**: Reasoning-lite experiments exist and have valid ground truth (per_request_lifecycle_metrics.json)
2. ✅ **VERIFIED**: All 15 experiments included in training (not excluded)
3. ⚠️ **REQUIRED**: Add defensive guards in `evolved_model.go` to detect and log numerical failures before they produce 100% error

---

## Scout Experiments Analysis

Scout experiments were expected to improve most from β₈ (MoE routing) and β₁₀ (batching inefficiency), but instead showed catastrophic deterioration:

| Experiment | Workload | Iter7 TTFT | Iter13 TTFT | Change | Status |
|------------|----------|------------|-------------|--------|--------|
| Scout general-lite | general-lite-2-1 | **100%** | **847%** | +747pp | ❌ 8.5× worse |
| Scout reasoning-lite | reasoning-lite-2-1 | **98%** | **100%** | +2pp | ❌ Complete failure |
| Scout codegen | codegen-2 | **92%** | **559%** | +467pp | ❌ 6× worse |
| Scout roleplay | roleplay-2 | **79%** | **384%** | +305pp | ❌ 4.8× worse |

**Key Insight**: The hypothesis that β₈ + β₁₀ would address Scout's sequence-length bottleneck is **completely refuted**. Instead, the β₅ explosion caused Scout to be overpredicted by 4-8×, exactly the opposite of the intended effect.

---

## Optimization Behavior

| Metric | Value |
|--------|-------|
| **Trials** | 738 |
| **Converged Early** | ✅ Yes |
| **Optimization Time** | 20.5 minutes (1229 seconds) |
| **Num Errors** | 0 |

**Observation**: Optimizer converged early (738 trials vs 1000 budget), suggesting it found a local optimum and couldn't escape. The zero-improvement convergence criterion (iter12 change) may have triggered too early.

**Best Trial**: Trial 537 achieved loss 2387.25%, indicating the optimizer explored ~738 trials but couldn't find better coefficients. This suggests:
1. The search space contains no good solutions with current architecture
2. β₅ bounds [0, 2000] are too wide, allowing catastrophic explosions
3. Collinearity between β₅, β₈, and β₁₀ prevents stable convergence

---

## Root Cause Analysis

### Primary Cause: β₅ MoE Gating Formula Fundamentally Wrong

The β₅ coefficient exploded to 1924.4, suggesting the **MoE gating basis function is systematically underestimating** by a large factor (2-5×). Three possible explanations:

1. **Missing layer multiplier**: Basis function computes single-layer gating time (~0.04μs), but β₅ should be ~1.0 (not 1924.4) if the function correctly accounts for all 56 Scout MoE layers.
   - **Diagnostic**: Check if `moe_gating_time` in `evolved_model.go` multiplies by `numMoELayers`
   - **Expected fix**: If missing, adding layer multiplier should reduce β₅ by 56× → β₅ ≈ 34 (still 10× too high)

2. **Wrong efficiency assumption**: Basis function assumes 30% MFU for gating network, but real efficiency may be 1-5% (memory-bound, small GEMMs with poor tensor core utilization).
   - **Diagnostic**: Profile Scout gating network to measure actual MFU
   - **Expected fix**: Reduce gating efficiency → basis function computes larger time → β₅ decreases

3. **β₅ absorbing non-gating overhead**: MoE routing (β₈), batching inefficiency (β₁₀), or scheduler delays (β₆) are being absorbed into β₅ due to collinearity.
   - **Diagnostic**: Correlation analysis between β₅, β₈, β₁₀ across trials
   - **Expected fix**: Reformulate basis functions to be orthogonal (e.g., β₈ per expert-token instead of per routed-token)

---

### Secondary Cause: β₁₀ Insufficient for Sequence-Length Bottleneck

β₁₀ converged to 0.100μs (exactly at lower bound of 0.1-1.0μs range), suggesting:
1. **Formula is correct but contribution too small**: For Scout general-lite (500 tokens, batch_size=4):
   - β₁₀ contribution = 0.100μs × (500²/4) = 0.100μs × 62,500 = **6.25ms**
   - This is only 6ms, but Scout general-lite needs ~100ms correction to reach target TTFT
   - **Shortfall**: 94ms missing overhead NOT captured by β₁₀

2. **β₅ explosion masked β₁₀ effect**: The massive β₅ overprediction (+4.3ms per Scout prefill) swamped the 6ms β₁₀ contribution, making it invisible in the loss function.

---

### Tertiary Cause: Optimizer Trapped by Zero-Improvement Convergence

Iteration 13 converged early at 738 trials (26% below 1000 budget), suggesting the zero-improvement convergence criterion from iter12 triggered prematurely:
- **Criterion**: Stop if best loss doesn't improve for N consecutive trials
- **Problem**: With β₅ causing catastrophic overprediction, optimizer may have explored local region around trial 537 and found no improvements, triggering early stop
- **Evidence**: Loss 2387% is clearly NOT a good solution, but optimizer couldn't escape

**Recommended fix for iter14**: Increase convergence patience or require minimum improvement threshold (e.g., "stop if no ≥5% improvement for 200 trials").

---

## Comparison with Recent Iterations

| Iteration | Loss | Key Change | Outcome |
|-----------|------|------------|---------|
| **Iter7** | **155%** | Added β₇ (decode overhead) | ✅ Best stable baseline |
| Iter8 | 155% | Added β₈ (MoE routing) | ⚠️ 0pp improvement, mechanism real |
| Iter9 | 161% | Added β₉ (FP8), rejected | ⚠️ Coefficient explosions began |
| Iter10 | 4267% | Added β₁₀ (batching inefficiency) | ❌ Catastrophic, warm-started from iter9 |
| Iter11 | 4084% | Audited β₁₀ (proven correct) | ❌ Still catastrophic |
| Iter12 | 2590% | Widened β₃' bounds | ❌ β₃' collapsed |
| **Iter13** | **2387%** | Returned to iter7 + β₈ + β₁₀ | ❌ **CATASTROPHIC** |

### Implications of Dataset Change (Iter7 → Iter13)

**What Changed**:
- **Iter7 dataset**: Original reasoning workloads
- **Iter8-13 dataset**: Reasoning-lite workloads (lighter load version - same sequence lengths, different rate/duration/arrival characteristics)
- **3 experiments replaced**: Scout/Qwen2.5/Llama-2 reasoning → reasoning-lite

**What This Means for Analysis**:

1. **Direct comparison is imperfect**: Iter7 (155% loss) was optimized for different experiments than iter13. The 15.4× deterioration (155% → 2387%) includes both:
   - Architectural problems (β₅ explosion - primary cause)
   - Dataset distribution shift (reasoning → reasoning-lite workload characteristics)

2. **Warm-start strategy complications**: Iter7 coefficients may not be optimal for reasoning-lite experiments. For example:
   - β₆ (scheduler overhead) was tuned for original reasoning workload arrival patterns
   - β₇ (decode per-request overhead) may have different characteristics under lite load

3. **100% error interpretation**: All 3 reasoning-lite experiments failed with exactly 100% error. This is NOT due to missing training data (verified: all experiments present with valid ground truth). Instead, it's a **numerical failure** (overflow, negative values) in the evolved model when predicting these NEW experiments.

4. **"Stable baseline" caveat**: Iter7 appeared stable (155% loss) but was never tested on reasoning-lite workloads. We cannot assume iter7's stability would hold on the new dataset with different load characteristics.

**Key Insight**: Returning to iter7 baseline did NOT provide stability even accounting for dataset change. Iter13 is only marginally better than iter12 (2387% vs 2590%, 8% improvement), and the β₅ explosion (46,800×) occurred regardless of warm-start source. This suggests the **fundamental problem is architectural** (β₅ basis function bug), not initialization or dataset shift.

**Critical Realization**: The pattern across iter9-13 shows that adding ANY new coefficient (β₉, β₁₀, β₃', or even returning to iter7) triggers β₅ explosions. This suggests **β₅ is a "garbage collector" coefficient** that absorbs all unmodeled error when new terms are added.

---

## Diagnostic Questions for Iter14

### Question 1: Is the β₅ explosion a basis function bug?

**Test**: Manually compute MoE gating time for Scout:
```python
# Scout: hiddenDim=3584, numExperts=16, numMoELayers=56
tokens = 100
gating_flops_per_layer = 2 * tokens * 3584 * 16  # 11.5M FLOPs
peak_flops = 989e12  # H100 FP16
gating_efficiency = 0.30
gating_time_per_layer = gating_flops_per_layer / (peak_flops * gating_efficiency)
total_gating_time = gating_time_per_layer * 56  # All MoE layers

print(f"Single layer: {gating_time_per_layer*1e6:.2f} μs")
print(f"All 56 layers: {total_gating_time*1e6:.2f} μs")
print(f"If β₅=1924, contribution: {1924 * gating_time_per_layer*1e6:.2f} μs")
```

**Expected output**:
- Single layer: 0.0388 μs (matches hypothesis)
- All 56 layers: 2.17 μs (if basis function multiplies by numMoELayers)
- If β₅=1924: 74.6 μs (physically implausible for gating alone)

**Action**: If basis function does NOT multiply by numMoELayers, add it in iter14.

---

### Question 2: Are β₅, β₈, and β₁₀ collinear?

**Test**: Correlation analysis across all 738 Optuna trials:
```python
import optuna
study = optuna.load_study(study_name="iter13_evolved", storage="sqlite:///optuna.db")
trials_df = study.trials_dataframe()

# Compute pairwise correlations
corr_matrix = trials_df[['params_beta_5', 'params_beta_8', 'params_beta_9']].corr()
print(corr_matrix)
```

**Expected**: If correlation >0.7 between any pair, collinearity is confirmed.

**Action**: If collinear, reformulate one term (e.g., β₈ per expert-token instead of per routed-token).

---

### Question 3: Why do reasoning-lite experiments return exactly 100% error?

**DEBUGGED**: Root cause identified through BLIS simulation run:

**Mechanism**:
1. β₅ = 1924.4 causes catastrophic latency overestimation
   - Estimated per-token contribution: ~4.3ms/token (vs ~0.017ms actual)
   - For reasoning-lite (934 prefill + 1448 decode = 2382 tokens): ~10,268ms predicted
   - Ground truth E2E: 40.835ms → **251× underestimate**

2. All requests timeout before completion
   - BLIS simulation: 23 requests injected, **0 completed** (all timed out)
   - Summary metrics calculated from completed requests only
   - Result: `e2e_mean_ms = 0`, `ttft_mean_ms = 0`

3. APE calculation produces exactly 100%
   - `APE = |predicted - actual| / actual × 100`
   - `E2E APE = |0 - 40.835| / 40.835 × 100 = 100%`
   - `TTFT APE = |0 - 0.138| / 0.138 × 100 = 100%`

**Conclusion**: The 100% error is NOT a data quality issue—it's a numerical catastrophe. β₅ is so large that the simulator predicts latencies 251× higher than reality, causing complete request timeout and zero completed requests. The APE denominator uses ground truth (non-zero), but the numerator has predicted=0 due to timeout, yielding exactly 100%.

---

### Question 4: Is β₁₀ contribution too small?

**Test**: Compute β₁₀ contribution for Scout general-lite vs target correction:
- Scout general-lite: β₁₀ × (500²/4) = 0.100μs × 62,500 = 6.25ms
- Required correction: Iter7 TTFT 100% error → need ~100ms reduction
- **Shortfall**: 93.75ms missing

**Action**: If β₁₀ formula correct but insufficient, need additional term (β₃' with wide bounds) OR increase β₁₀ bounds to [0.1μs, 10μs].

---

## Recommendations for Iter14

### Option A: Fix β₅ Basis Function (Recommended)

**Hypothesis**: MoE gating basis function missing numMoELayers multiplier OR wrong efficiency assumption.

**Action**:
1. Audit `evolved_model.go` lines ~250-280 (MoE gating computation)
2. Add `× numMoELayers` if missing
3. OR reduce gating efficiency from 0.30 → 0.05 (profile-informed)
4. Tighten β₅ bounds to [0, 100] (prevent explosion)

**Expected outcome**: β₅ converges to 1-50 (physically plausible), loss 2387% → 120-180% (10-20× improvement).

---

### Option B: Remove β₅ Entirely

**Hypothesis**: β₅ is fundamentally untrainable due to collinearity with β₈ and β₁₀.

**Action**:
1. Remove β₅ from architecture
2. Keep β₈ (MoE routing) and β₁₀ (batching inefficiency)
3. Warm-start from iter7 (β₀-β₄, β₆-β₇) + iter8 (β₈) + new β₁₀

**Expected outcome**: Loss 2387% → 140-180% (simpler architecture, fewer collinearities), but may still underpredict Scout.

---

### Option C: Revert to Pure Iter7 (Safest)

**Hypothesis**: ANY addition to iter7 triggers coefficient explosions.

**Action**:
1. Pure iter7 architecture (β₀-β₇ only, 8 beta coefficients)
2. Do NOT add β₈ or β₁₀
3. Accept 155% loss as current capability ceiling

**Expected outcome**: Loss 2387% → 155% (recover iter7 baseline), but Scout problems remain unsolved.

---

### Option D: Reformulate β₁₀ with Higher Upper Bound

**Hypothesis**: β₁₀ formula correct but contribution too small (converged at lower bound 0.1μs).

**Action**:
1. Keep iter7 + β₈ + β₁₀ architecture
2. Increase β₁₀ bounds from [0.1μs, 2.0μs] → [0.1μs, 10μs]
3. Fix β₅ basis function (Option A) simultaneously

**Expected outcome**: β₁₀ converges to 1-5μs, contributing 30-150ms for long sequences, loss 2387% → 100-140%.

---

## Conclusion

Iteration 13's catastrophic failure (2387% loss, 15.4× worse than iter7) definitively proves that:

1. **Returning to iter7 baseline does NOT guarantee stability** when new coefficients are added
2. **β₅ MoE gating formula is fundamentally broken**, causing 46,800× explosion
3. **β₁₀ batching inefficiency is insufficient** (contributes only 6ms vs 100ms needed)
4. **Warm-start strategy is irrelevant** - the problem is architectural, not initialization

The path forward requires **fixing the β₅ basis function** (Option A) before attempting any other additions. Without this fix, every iteration will trigger β₅ explosions and produce catastrophic results.

**Next iteration must focus on a single goal: Make β₅ converge to a physically plausible value (1-50) before adding any complexity.**
