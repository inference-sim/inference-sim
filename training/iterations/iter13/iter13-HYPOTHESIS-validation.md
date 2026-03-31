# Iteration 13: Hypothesis Validation

## Summary

**Overall Verdict**: ❌ **ALL HYPOTHESES CATASTROPHICALLY REFUTED**

All three hypotheses were completely wrong. The strategy to return to iter7's stable baseline and incrementally add β₈ + β₁₀ resulted in the **worst iteration in training history** (loss 2387%, 15.4× worse than iter7's 155%). The core hypothesis that returning to iter7 would provide stability was invalidated — coefficient explosions occurred regardless of warm-start source.

**Critical Discovery**: β₅ (MoE gating) exploded 46,800× to 1924.4, suggesting the **basis function is fundamentally broken** (likely missing `× numMoELayers` multiplier). This single bug likely caused the entire catastrophic failure.

**Important Context**: Between iter7 and iter13, the dataset changed — reasoning experiments were converted to **reasoning-lite** (shorter sequences, different workload characteristics). This makes direct comparison with iter7 more complicated and may explain some failures (particularly the 3 reasoning-lite experiments that returned 100% error).

---

## H-main: Iter7 Baseline + β₁₀ Recovers Sequence-Length Prediction

**Status**: ❌ **COMPLETELY REFUTED**

### Prediction

After adding β₁₀ (batching inefficiency) to iter7's stable architecture:

**Overall Performance**:
- Overall loss: 155.4% (iter7) → **<140%** (≥15pp improvement)
- TTFT RMSE: 64.0% (iter7) → **<55%** (≥9pp improvement)
- E2E RMSE: 91.3% (iter7) → **<85%** (≥6pp improvement)

**Long-Sequence Experiments** (primary target):
- Scout general-lite (iter7: 100% TTFT) → **<85%** (≥15pp improvement)
- Mistral general-lite (iter7: 91% TTFT) → **<75%** (≥16pp improvement)
- Yi-34B general-lite (iter7: 78% TTFT) → **<65%** (≥13pp improvement)
- Llama-3.1 general-lite (iter7: 77% TTFT) → **<65%** (≥12pp improvement)

**Coefficient Stability** (all within expected ranges):
- β₅: 0.0411 → **300-1000** (9000-24000× increase to recover from collapse)
- β₆: 13.2ms → **40-100ms** (may increase for cold-start overhead)
- β₁₀: NEW → **0.1-1.0μs** per (token²/batch)

### Result

**Overall Performance**:
- Overall loss: **2387.25%** (15.4× WORSE than iter7, not better!)
- TTFT RMSE: **1370.44%** (21.4× worse than iter7)
- E2E RMSE: **1016.80%** (11.1× worse than iter7)

**Long-Sequence Experiments**:
- Scout general-lite: 100% → **847% TTFT** (747pp deterioration, 8.5× worse)
- Mistral general-lite: 91% → **3965% TTFT** (3874pp deterioration, 43× worse)
- Yi-34B general-lite: 48% → **1183% TTFT** (1135pp deterioration, 25× worse)
- Llama-3.1 general-lite: 41% → **1010% TTFT** (969pp deterioration, 25× worse)

**Coefficient Stability**:
- β₅: 0.0411 → **1924.4** (46,800× increase, 1.9× BEYOND upper bound!)
- β₆: 13.2ms → **10.0ms** (24% decrease, OPPOSITE direction from prediction)
- β₁₀: NEW → **0.100μs** (at lower bound, converged correctly but insufficient)

### Verdict: ❌ COMPLETELY REFUTED

**Why hypothesis failed**:

1. **Loss exploded 15.4× instead of improving 1.1×**:
   - Predicted: 155% → <140% (15pp improvement)
   - Actual: 155% → 2387% (+1436pp deterioration)
   - **This is the WORST iteration ever**, exceeding even iter10's 4267% loss

2. **Long-sequence experiments catastrophically failed**:
   - ALL four target experiments deteriorated by 25-43×
   - Scout general-lite: Expected <85%, got 847% (10× prediction)
   - Mistral general-lite: Expected <75%, got 3965% (53× prediction)
   - **Opposite of predicted improvement**

3. **β₅ exploded beyond hypothesis prediction**:
   - Predicted: 300-1000 (9000-24000× increase)
   - Actual: 1924.4 (46,800× increase, 1.9× upper bound)
   - Physical interpretation: 1924.4 × 0.04μs/layer × 56 layers = **4.3ms gating overhead** (physically implausible)

4. **β₁₀ converged but was insufficient**:
   - Converged to 0.100μs (exactly at lower bound of 0.1-1.0μs)
   - For Scout general-lite (500 tokens, batch_size=4): 0.1μs × (500²/4) = **6.25ms**
   - **But needed ~100ms correction** (iter7 had 100% TTFT error)
   - **Shortfall: 93.75ms unaccounted**

5. **Warm-start strategy failed**:
   - Hypothesis: "Clean warm-start from iter7 (not inflated iter9-12) prevents trapped local optimum"
   - Reality: Iter13 (2387%) only marginally better than iter12 (2590%)
   - **Conclusion**: Problem is architectural, not initialization

**Root Cause**: β₅ explosion absorbed all error signal, preventing β₁₀ from working as intended. The hypothesis that β₁₀ would address sequence-length bottleneck was invalidated by the β₅ basis function bug.

**Evidence of Basis Function Bug**: β₅ = 1924.4 / 56 layers = **34.4** (much closer to expected 1-50 range), suggesting the MoE gating basis function is missing a `× numMoELayers` multiplier.

---

## H-beta5-anomaly: β₅ Will Increase 9000-24000× to Recover from Collapse

**Status**: ⚠️ **PARTIALLY CORRECT BUT EXPLODED BEYOND BOUNDS**

### Prediction

Iter7 β₅ = 0.0411 (dimensionless) collapsed to essentially zero contribution. After adding β₈ (MoE routing) and β₁₀ (batching inefficiency):

- β₅: 0.0411 (iter7) → **300-1000 dimensionless** (9000-24000× increase)

**Causal Mechanism**:
- Iter7's β₅=0.0411 gave essentially ZERO contribution (0.0411 × 0.04μs = 0.0016μs)
- Scout underpredicted 79-100% because missing overhead went uncaptured
- After adding β₈+β₁₀, β₅ can recover to physical value (300-1000)

### Result

- β₅: 0.0411 (iter7) → **1924.4** (46,800× increase)
- Predicted range: 300-1000 → Actual: **1.9× upper bound**
- Increase magnitude: Predicted 9000-24000× → Actual: **46,800× (2× upper prediction)**

**Physical Interpretation**:
- β₅ is dimensionless scaling factor for MoE gating roofline estimate
- Basis function computes ~0.04μs per layer (single-layer gating)
- With β₅=1924.4: contribution = 1924.4 × 0.04μs ≈ **77μs per layer**
- Scout has ~56 MoE layers → 77μs × 56 = **4.3ms total gating overhead** (physically implausible)

**Profiling Evidence**: MoE gating networks are small linear projections (hiddenDim × numExperts), expected ~1-10μs total, not 4300μs.

### Verdict: ⚠️ PARTIALLY CORRECT BUT EXPLODED BEYOND BOUNDS

**What was correct**:
- ✅ Direction: β₅ DID increase (not stay collapsed)
- ✅ Magnitude order: Predicted 9000-24000×, got 46,800× (same order of magnitude)
- ✅ Mechanism: β₅ was absorbing missing overhead (hypothesis was right that it was collapsed due to missing β₈/β₁₀)

**What went wrong**:
- ❌ Exploded 1.9× beyond predicted upper bound (1924 vs 1000)
- ❌ Physical implausibility: 4.3ms gating overhead is 100-1000× too high
- ❌ Caused catastrophic overprediction across ALL experiments

**Root Cause (Three Hypotheses)**:

**Hypothesis 1: Missing Layer Multiplier** (Most Likely):
- If basis function computes single-layer gating time (~0.04μs), but β₅ should multiply by `numMoELayers`
- **Evidence**: 1924.4 / 56 layers = **34.4** (within expected 1-50 range!)
- **Test**: Check if `evolved_model.go` multiplies by `numMoELayers` in gating calculation
- **Expected Fix**: Adding layer multiplier should reduce β₅ by 56× → β₅ ≈ 34 (within physical range)

**Hypothesis 2: Wrong Gating Efficiency Assumption**:
- Basis function assumes 30% MFU for gating (same as dense attention)
- Reality: Gating is memory-bound, small GEMMs with poor tensor core utilization → MFU ~1-5%
- If true: Basis function underestimates by 6-30× → β₅ inflates to compensate
- **Expected Fix**: Reduce gating efficiency 0.30 → 0.05 should decrease β₅ by 6× → β₅ ≈ 320 (within predicted 300-1000 range)

**Hypothesis 3: Collinearity with β₈ and β₁₀**:
- MoE models have multiple overhead sources: gating (β₅), routing (β₈), batching inefficiency (β₁₀)
- If collinear (all scale with sequence length or token count), optimizer inflates β₅ while collapsing β₈
- **Evidence**: β₈ decreased 56% from 30μs → 13.3μs (opposite of expected)
- **Expected Fix**: Reformulate basis functions to be orthogonal

**Diagnostic Clause Result**:

**Scenario 1 (β₅ <100)**: Did NOT occur (β₅ = 1924.4, far above 100)

**Scenario 2 (β₅ >2000)**: NEARLY occurred (β₅ = 1924.4, just below 2000 upper bound)
- **Indicates**: Basis function underestimates OR β₅ absorbing non-gating overhead
- **Investigate**: Roofline calculation efficiency (30% too optimistic?), β₅ correlation with decode requests

**Scenario 3 (β₅ stabilizes at 300-1000)**: Did NOT occur

**Conclusion**: Hypothesis was directionally correct (β₅ would increase dramatically) but underestimated the explosion magnitude. The β₅ explosion suggests a **fundamental bug in the basis function** (missing layer multiplier OR wrong efficiency), not just collinearity.

---

## H-baseline-insight: Learn from Vidur's Success

**Status**: ❌ **COMPLETELY REFUTED**

### Prediction

After studying baseline_errors.json patterns:
- **Codegen workloads will improve most** (currently worst performers)
- **Scout workloads will show consistency** (currently wildly inconsistent -99% to +66%)
- **ITL predictions will improve** (currently underestimated -20% to -70% universally)

**Causal Mechanism**:
- Vidur avoids catastrophic TTFT overestimation (-14% to -32%) while all analytical models fail (+330% to +3031%)
- Vidur likely models framework overhead, scheduler delays, or queuing effects that pure roofline misses
- Iter13 adds these framework terms: β₆ (scheduler), β₇ (decode overhead), β₁₀ (batching inefficiency)
- Expected cascade: Adding framework terms should make our model behave more like vidur

### Result

**Codegen Workloads** (predicted: "will improve most"):
- Llama-2 codegen: iter7 9.3% TTFT → iter13 **1417% TTFT** (152× WORSE)
- Mistral codegen: iter7 20% TTFT → iter13 **1193% TTFT** (60× WORSE)
- Llama-3.1 codegen: iter7 29% TTFT → iter13 **742% TTFT** (25× WORSE)
- **Pattern**: ALL codegen workloads catastrophically failed, 25-152× worse than iter7

**Scout Workloads** (predicted: "will show consistency"):
- Scout general-lite: 100% → **847% TTFT** (8.5× worse)
- Scout reasoning-lite: 98% → **100% TTFT** (complete failure)
- Scout codegen: 92% → **559% TTFT** (6× worse)
- Scout roleplay: 79% → **384% TTFT** (4.8× worse)
- **Pattern**: Still wildly inconsistent (100-847% range), 5-8× worse than iter7

**ITL Predictions** (predicted: "will improve"):
- Not directly measured in results, but overall E2E RMSE 91.3% → **1016.8%** (11× worse)
- If ITL improved, E2E would improve (it didn't)
- **Pattern**: ITL likely worsened along with all other metrics

### Verdict: ❌ COMPLETELY REFUTED

**Why hypothesis failed**:

1. **Codegen workloads deteriorated 25-152×, not improved**:
   - Hypothesis: "Codegen will improve most (currently worst performers)"
   - Reality: Codegen became catastrophically worse (1200-2700% TTFT)
   - **Opposite of prediction**

2. **Scout workloads became MORE inconsistent, not less**:
   - Hypothesis: "Scout will show consistency"
   - Reality: Scout errors range 100-847% (8.5× spread, was 79-100% = 1.3× spread in iter7)
   - **6.5× MORE inconsistent**

3. **Framework overhead terms did NOT help**:
   - Added β₆ (scheduler), β₇ (decode overhead), β₁₀ (batching inefficiency)
   - Expected: Behavior like vidur (avoid overestimation)
   - Reality: Amplified overestimation by 15-150×
   - **Framework terms made it WORSE, not better**

4. **Universal pattern: ALL experiments worse**:
   - Best experiment (Scout roleplay): 79% → 384% TTFT (4.8× worse)
   - Worst experiment (Mistral general-lite): 90% → 3965% TTFT (43× worse)
   - **NO experiment improved** (0/15 success rate)

**Root Cause**: Instead of adding missing mechanisms that vidur captures, the framework overhead terms (β₆, β₇, β₁₀) were swamped by β₅'s catastrophic explosion. The β₅ explosion caused massive overprediction (+4.3ms per Scout prefill), overwhelming any benefit from framework terms.

**Why vidur succeeds while we fail**:
- Vidur likely uses empirical corrections, not purely analytical roofline
- Vidur may not model MoE gating separately (avoids β₅ explosion trap)
- Vidur's framework overhead terms may be calibrated against real measurements, not trained
- **Key insight**: Analytical roofline + learned coefficients is fundamentally unstable when MoE gating basis function is wrong

**Diagnostic Clause Results**:

**Scenario 1 (codegen DON'T improve <10pp)**: OCCURRED
- **Indicates**: Missing mechanism beyond framework overhead
- **Action**: Study vidur implementation for additional terms
- **Status**: Correct diagnosis, but cannot proceed until β₅ fixed

**Scenario 2 (Scout stays wildly inconsistent)**: OCCURRED
- **Indicates**: MoE-specific bottleneck not captured
- **Action**: Profile Scout vs dense models on same workload
- **Status**: Correct diagnosis — MoE gating (β₅) is the bottleneck

**Conclusion**: Hypothesis was based on correct observation (vidur succeeds by modeling framework overhead), but execution failed catastrophically due to β₅ explosion. Cannot evaluate whether framework terms help until β₅ is fixed.

---

## Summary of Hypothesis Results

| Hypothesis | Type | Prediction | Key Metric | Result | Verdict |
|------------|------|------------|------------|--------|---------|
| **H-main** | Architectural baseline + sequence-length term | Iter7 (155%) + β₁₀ recovers long-sequence prediction | Overall loss <140% | 2387% (15× worse) | ❌ REFUTED |
| **H-beta5-anomaly** | Coefficient correction | β₅ increases 9000-24000× to 300-1000 after β₈+β₁₀ | β₅ = 300-1000 | β₅ = 1924.4 (1.9× upper bound) | ⚠️ PARTIAL |
| **H-baseline-insight** | Learn from successful simulator | Framework overhead terms make us behave like vidur | Codegen improvement | Codegen 25-152× worse | ❌ REFUTED |

**Overall**: 0/3 hypotheses confirmed, 2/3 completely refuted, 1/3 partially correct (but exploded beyond bounds).

---

## Three Critical Failures Explained

### Failure 1: β₅ Exploded 46,800× (Worst Coefficient Explosion Ever)

**Predicted**: 0.0411 → 300-1000 (9000-24000× increase)
**Actual**: 0.0411 → 1924.4 (46,800× increase)

**Root Cause**: MoE gating basis function likely missing `× numMoELayers` multiplier:
- **Evidence**: 1924.4 / 56 layers = 34.4 (within expected 1-50 range)
- **Test**: Check `evolved_model.go` lines ~250-280 for layer multiplier
- **Expected Fix**: Add `× numMoELayers` should reduce β₅ by 56× → β₅ ≈ 34

**Impact**:
- Caused 4.3ms overprediction per Scout MoE layer (physically implausible)
- Triggered cascading collapses in other coefficients (α₁, β₄, β₆, β₈)
- Made ALL experiments catastrophically fail (0/15 success rate)

**Fix for iter14**: Audit and fix MoE gating basis function BEFORE any other changes.

### Failure 2: Three Reasoning-Lite Experiments Failed Completely (100% APE)

**Experiments**:
- Scout reasoning-lite-2-1: 100% TTFT, 100% E2E
- Qwen2.5 reasoning-lite-1-1: 100% TTFT, 100% E2E
- Llama-2 reasoning-lite-1-1: 100% TTFT, 100% E2E

**Pattern**: ALL reasoning-lite experiments returned exactly 100% error across all metrics (mean, p90, p99 TTFT/E2E/ITL)

**CRITICAL CONTEXT: Dataset Change**:
- **Iter7 dataset**: Original reasoning workloads
- **Iter8-13 dataset**: Reasoning workloads converted to **reasoning-lite** (lighter load version - same sequence lengths, different rate/duration/arrival characteristics)
- **Implication**: These are **NEW experiments** that iter7 never trained on
- **VERIFIED**: All 3 reasoning-lite experiments have valid ground truth data (per_request_lifecycle_metrics.json) and are included in training (15/15 experiments)

**Root Cause** (DEBUGGED via BLIS simulation run):

**Mechanism**:
1. β₅ = 1924.4 causes catastrophic latency overestimation
   - Per-token contribution: ~4.3ms/token (vs ~0.017ms actual)
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

**Conclusion**: The 100% error is NOT a numerical overflow or data quality issue—it's a request timeout catastrophe. β₅ is so large that the simulator predicts latencies 251× higher than reality, causing all requests to timeout before completion. The APE denominator uses ground truth (non-zero), but the numerator has predicted=0 due to timeout, yielding exactly 100%.

**Fix for iter14**:
1. ✅ **VERIFIED**: Reasoning-lite experiments exist with valid ground truth (not a data issue)
2. ⚠️ **REQUIRED**: Add defensive guards in `evolved_model.go`:
```go
if stepTime < 0 || stepTime > 1e12 { // >1000s = clearly wrong
    log.Printf("ERROR: StepTime overflow: %d ns", stepTime)
    return 0 // or clamp to max value
}
```

### Failure 3: Warm-Start Strategy Failed (Iter13 ≈ Iter12)

**Hypothesis**: "Clean warm-start from iter7 (not inflated iter9-12) prevents trapped local optimum"
**Predicted**: Loss 2590% (iter12) → <200% (iter13) via stable warm-start
**Actual**: Loss 2590% (iter12) → 2387% (iter13) — only 8% improvement

**Root Cause**: Problem is **architectural** (β₅ basis function bug), not initialization (warm-start source)

**Evidence**:
- Iter7 had β₅=0.0411 (collapsed, stable)
- Iter13 warm-started from iter7 β₅=0.0411
- Iter13 β₅ exploded to 1924.4 (46,800× increase)
- **Conclusion**: Explosion happens regardless of warm-start source when architecture includes β₈+β₁₀

**Pattern Across Iter9-13**:
- Iter9: Added β₉ (FP8) → β₅ stayed low, other coefficients exploded
- Iter10: Added β₁₀ → Loss 4267%
- Iter11: Audited β₁₀ (proven correct) → Loss 4084%
- Iter12: Widened β₃' bounds → Loss 2590%
- Iter13: Reverted to iter7 + β₈ + β₁₀ → Loss 2387%

**IMPORTANT CONFOUND: Dataset Changed Between Iter7 and Iter13**:
- Iter7 trained on: Original reasoning workloads (longer sequences, heavier load)
- Iter13 trained on: Reasoning-lite workloads (shorter sequences, lighter load)
- **Implication**: "Returning to iter7" means returning to iter7 *architecture*, but the dataset is different
- **Impact**: Direct comparison complicated — iter7 coefficients were optimized for different experiments than iter13
- **Evidence**: All 3 reasoning-lite experiments failed with 100% error (these experiments didn't exist in iter7)
- **HOWEVER**: Data verified present and valid (15/15 experiments included in training) — the 100% error is a numerical failure, not missing data

**Key Insight**: ANY addition to iter7 triggers β₅ explosion. The problem is that β₅'s basis function is fundamentally broken — it cannot coexist with β₈+β₁₀ without exploding. Additionally, the dataset change makes it unclear if iter7 coefficients are a valid warm-start for the new reasoning-lite experiments.

**Fix for iter14**: Cannot warm-start away from the problem. Must fix β₅ basis function or remove β₅ entirely. Also verify reasoning-lite experiments are properly included in training data.

---

## Recommendations for Iter14

### Priority 1: Fix β₅ Basis Function (CRITICAL)

**Action**: Audit `sim/latency/evolved_model.go` lines ~250-280:

```go
// Current (SUSPECTED BUG - missing layer multiplier):
moe_gating_time := gating_flops / (peak_flops * 0.30)

// Should be (WITH layer multiplier):
moe_gating_time := (gating_flops * numMoELayers) / (peak_flops * 0.30)
```

**Expected Outcome**: β₅ converges to 1-50 (physically plausible), loss 2387% → 120-180% (10-20× improvement)

**Alternative**: If no layer multiplier missing, reduce gating efficiency:
```go
// Reduce from 30% → 5% (profile-informed):
moe_gating_time := gating_flops / (peak_flops * 0.05)
```

### Priority 2: Tighten β₅ Bounds

**Action**: Change β₅ bounds from [0, 2000] → **[0, 100]** to prevent future explosions

**Rationale**: Even if basis function fixed, bounds should prevent β₅ from exceeding 100 (physical upper limit)

### Priority 3: Add Diagnostic Logging

**Action**: Add logging to `evolved_model.go` after computing `moe_gating_time`:
```go
if m.modelConfig.NumMoELayers > 0 && totalPrefillTokens > 0 {
    moe_gating_contrib := m.Beta[5] * moe_gating_time
    if moe_gating_contrib > 0.001 { // Log if >1ms
        fmt.Fprintf(os.Stderr, "DEBUG: β₅=%.2f, basis=%.6fs, contrib=%.6fs\n",
            m.Beta[5], moe_gating_time, moe_gating_contrib)
    }
}
```

### Priority 4: Run Correlation Test (5 minutes)

**Action**: Check if β₅, β₈, β₁₀ are collinear:
```python
import optuna
study = optuna.load_study(study_name="iter13_evolved", storage="sqlite:///optuna.db")
trials_df = study.trials_dataframe()
corr = trials_df[['params_beta_5', 'params_beta_8', 'params_beta_9']].corr()
print(corr)
```

**If r(β₅, β₈) > 0.7**: Consider removing β₅ entirely (Option B) instead of fixing basis function

---

## Process Lessons

### 1. Returning to Stable Baseline Does NOT Guarantee Stability

**Iter13 Mistake**: Assumed warm-starting from iter7 (loss 155%, stable) would prevent catastrophic failures

**Problem**: Iter7 was stable because it did NOT have β₈+β₁₀. Adding β₈+β₁₀ to iter7 architecture triggers β₅ explosion regardless of warm-start source.

**Additional Confound**: Dataset changed between iter7 and iter13 (reasoning → reasoning-lite), complicating "stable baseline" comparison:
- Iter7 optimized for: Original reasoning workloads
- Iter13 optimized for: Reasoning-lite workloads (lighter load version - same sequence lengths, different rate/duration/arrival characteristics)
- **Impact**: "Returning to iter7" returns architecture BUT coefficients were optimized for different workload characteristics
- **Verified**: All 3 reasoning-lite experiments have valid ground truth and are included in training (not a data issue)

**Implications of Dataset Change**:

1. **Loss comparison imperfect**: Iter7's 155% loss was on *different* experiments than iter13's 2387%. The 15.4× deterioration includes:
   - Architectural problems (β₅ explosion, primary cause)
   - Dataset distribution shift (reasoning → reasoning-lite load characteristics, minor secondary effect)

2. **Warm-start may not be optimal**: Iter7 coefficients (e.g., β₆ scheduler overhead, β₇ decode overhead) were tuned for original reasoning workload arrival patterns, may not generalize well to lite load characteristics

3. **100% error is numerical failure**: All 3 reasoning-lite experiments failed with exactly 100% error, but NOT due to missing data (verified present). This is overflow/underflow in evolved model computation.

**Correct Approach**:
- Stability comes from architecture, not initialization or dataset
- If adding new coefficients causes explosions on ANY dataset, the architecture is wrong
- Must fix basis functions BEFORE adding terms, not after
- Account for dataset changes when comparing across iterations, but don't let this distract from the primary issue (β₅ explosion)

### 2. Partial Hypothesis Confirmation Can Be Worse Than Complete Refutation

**Iter13 Mistake**: H-beta5-anomaly was "partially correct" (β₅ increased as predicted), but exploded 1.9× beyond bounds

**Problem**: Partial confirmation masked the fundamental problem (basis function bug). We thought "hypothesis mostly right, just needs bounds tuning" when actually "basis function fundamentally broken."

**Correct Approach**:
- Treat "partial confirmation with explosion" as REFUTATION
- If coefficient converges outside expected range, basis function is wrong
- Don't tune bounds when you should fix formulas

### 3. Unit Tests Must Be Run BEFORE Training, Not After

**Iter13 Mistake**: Did not add unit tests for β₅ basis function before training

**Problem**: 20-minute training run (738 trials × 1.67s/trial) wasted because β₅ basis function was wrong from start

**Correct Approach**: Pre-training validation checklist:
```bash
# 1. Unit tests for ALL new basis functions
go test ./sim/latency -run "TestBeta.*" -v

# 2. Smoke test on single experiment
python inner_loop.py --dry-run --experiment trainval_data/21-llama-4-scout-17b-16e-tp2-roleplay-2

# 3. Coefficient sanity check (expected ranges)
grep "beta.*:" coefficient_bounds.yaml

# 4. Correlation test if adding multiple terms
python scripts/check_collinearity.py
```

Total time: <5 minutes. Can save 20+ minutes of training + hours of analysis.

---

## Conclusion

All three hypotheses were catastrophically refuted. The strategy to return to iter7's stable baseline and incrementally add β₈+β₁₀ produced the **worst iteration in training history** (loss 2387%, 15.4× worse than iter7).

**Root Cause**: β₅ (MoE gating) basis function is fundamentally broken, likely missing a `× numMoELayers` multiplier. This caused a 46,800× explosion (1924.4 vs 0.0411 in iter7), triggering cascading failures across all experiments.

**Key Evidence**: β₅ = 1924.4 / 56 layers = 34.4 (within expected 1-50 range), strongly suggesting missing layer multiplier.

**Next Steps for Iter14**:
1. ✅ **Audit β₅ basis function** (check for layer multiplier)
2. ✅ **Fix basis function** (add `× numMoELayers` OR reduce gating efficiency 0.30 → 0.05)
3. ✅ **Tighten β₅ bounds** ([0, 2000] → [0, 100])
4. ✅ **Run correlation test** (if r(β₅, β₈) > 0.7, consider removing β₅ entirely)
5. ⚠️ **Add unit tests** for β₅ basis function before training
6. ⚠️ **Add diagnostic logging** to catch explosions early

**Expected Outcome**: If β₅ fixed, loss 2387% → 120-180% (10-20× improvement, 13-34× improvement from iter7).

**Bottom Line**: Cannot make progress on Scout bottleneck (sequence-length dependency) until β₅ explosion is resolved. Iter14 must focus on **fixing β₅ ONLY**, not adding new features or coefficients. One bug, one fix, one iteration.
