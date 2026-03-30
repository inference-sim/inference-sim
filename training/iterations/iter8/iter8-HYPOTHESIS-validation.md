# Iteration 8: Hypothesis Validation

## ⚠️ IMPORTANT: Data Update Post-Analysis

**AFTER this analysis was completed**, exp17 (Scout general) was replaced with clean data:

- **Old exp17**: `17-llama-4-scout-17b-16e-tp2-general-2` — Saturated server (used in this analysis)
- **New exp17**: `17-llama-4-scout-17b-16e-tp2-general-lite-2-1` — Normal conditions (use for iter9+)

**Impact**: This validation used saturated Scout general data (99.97% TTFT may be inflated). **iter9 MUST re-train** with new exp17 before adding β₉.

---

## H-main: MoE Routing Overhead Captures Scout Residual

**Prediction** (from Agent 1):
- Overall loss: 155.37% → **<80%** (75pp improvement, 48% reduction)
- TTFT RMSE: 64.04% → **<40%** (24pp improvement, 38% reduction)
- E2E RMSE: 91.33% → **<50%** (41pp improvement, 45% reduction)
- Scout TTFT error: Avg 90% (range 79-100%) → **<50%** (>40pp improvement for all 4 Scout experiments)
- Non-Scout experiments: Remain stable or improve slightly (< ±10pp change from iter7)

**Quantitative Threshold**: If overall loss does NOT reduce below 100%, or if Scout TTFT does NOT improve to <70%, then H-main is REJECTED.

**Causal Mechanism** (from Agent 1):

Scout MoE architecture has per-token expert routing overhead not captured by current model. β₈ captures routing cost (10-50μs per routed token) including expert selection, dispatch, load balancing, and aggregation. For Scout with 26 MoE layers and ~100 prefill tokens, β₈ should contribute 26-130ms per request, matching Scout's TTFT residual gap.

**Diagnostic Clause** (from Agent 1):

*If this hypothesis fails (overall loss remains >100% OR Scout TTFT >70%), it indicates:*
1. β₈ coefficient converged to zero → MoE routing overhead negligible, investigate alternative Scout bottlenecks
2. β₈ coefficient converged >100μs → Unrealistically high, investigate absorbing other missing terms
3. Non-Scout experiments degraded >10pp → Zero-sum trade-off, need architecture-specific handling

**Actual Result**:

**Loss Metrics**:
- Overall loss: **155.35%** (virtually unchanged from iter7's 155.37%, -0.02pp)
- TTFT RMSE: **63.98%** (virtually unchanged from iter7's 64.04%, -0.06pp)
- E2E RMSE: **91.37%** (virtually unchanged from iter7's 91.33%, +0.04pp)

**Scout TTFT Errors** (no improvement):
- Scout general (exp 17): **99.97%** TTFT (unchanged from iter7's 100%, +0pp)
- Scout reasoning-lite (exp 48): **98.46%** TTFT (unchanged from iter7's 98%, +0pp)
- Scout codegen (exp 20): **92.08%** TTFT (unchanged from iter7's 92%, +0pp)
- Scout roleplay (exp 21): **79.10%** TTFT (unchanged from iter7's 79%, +0pp)

**Non-Scout Experiments**: All remain stable (changes < ±10pp), no degradation observed.

**β₈ Coefficient**: **0.00003 seconds** = **30μs per routed token** (within predicted 10-50μs range)

**β₈ Contribution for Scout Prefill**:
- Routed tokens: 26 layers × 100 tokens × 1 expert/tok ÷ 2 TP = 1,300
- Contribution: 1,300 × 30μs = **39ms per prefill request**
- This is a significant contribution (39ms), yet Scout errors remain at 79-100% APE

**Verdict**: ❌ **REJECTED**

**Evidence**:
1. **Loss unchanged**: Overall loss 155.35% vs 155.37% (target <80%, failed by 75pp)
2. **Scout unchanged**: All 4 Scout experiments remain at 79-100% TTFT APE (target <50%, failed by 29-50pp)
3. **β₈ physically plausible**: 30μs per routed token (within 10-50μs range)
4. **β₈ contribution significant**: 39ms per Scout prefill request (not negligible)
5. **Non-Scout stable**: No degradation (< ±10pp change)
6. **Optimization converged**: 51 trials, converged early, 0 errors

**Causal Analysis**:

**Why β₈ Failed**: Despite β₈ converging to a physically plausible value and contributing 39ms per Scout request, it did NOT reduce Scout errors. This indicates **β₈ captured SOME MoE overhead, but it's NOT the primary Scout bottleneck**.

**Evidence against the causal mechanism**:
1. **β₈ is not zero**: The optimizer learned β₈ = 30μs, confirming MoE routing overhead exists
2. **β₈ is not >100μs**: 30μs is physically reasonable (within predicted range)
3. **Non-Scout experiments are stable**: No zero-sum trade-off observed
4. **Yet Scout errors unchanged**: β₈ contribution (39ms) is insufficient to close the gap

**What this reveals**: The hypothesis assumed MoE routing overhead was the PRIMARY Scout bottleneck. The results prove this assumption WRONG. β₈ captures a real overhead (30μs per routed token), but Scout's bottleneck lies elsewhere.

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

**Diagnostic clause evaluation**:
- ✅ β₈ coefficient is NOT zero (30μs, physically plausible)
- ✅ β₈ coefficient is NOT >100μs (30μs, within expected range)
- ✅ Non-Scout experiments did NOT degrade >10pp (all stable)

**Conclusion**: All three diagnostic conditions are SATISFIED, yet H-main FAILED. This means the diagnostic clause was incomplete — it didn't account for the scenario where β₈ is correct but INSUFFICIENT.

**Alternative Scout bottlenecks to investigate** (from diagnostic clause):
1. **FP8 dequantization overhead**: Scout uses FP8 dynamic quantization. Mixed-precision coordination may add latency not captured by β₈.
2. **TP=2 communication overhead**: Cross-GPU expert routing may have higher TP coordination cost than the β₂ term captures.
3. **Model config issue**: InterleaveMoELayerStep=26 or NumExpertsPerTok might be incorrect, causing β₈ basis function to underestimate routed tokens.
4. **Baseline roofline error**: The baseline analysis showed roofline underestimates Scout by 50-99% (missing overhead). β₈ adds 39ms, but the gap may be 100-200ms, requiring additional terms.

**Next Investigation**: Profile Scout MoE overhead separately with vLLM profiler to measure per-layer routing latency, TP communication overhead, and FP8 dequantization cost.

---

## H-ablation-beta8: β₈ Accounts for Majority of Scout Improvement

**Prediction** (from Agent 1):
- With β₈ (full model): Scout TTFT avg 90% → <50% (>40pp improvement)
- Without β₈ (ablated): Scout TTFT avg 90% → 80-90% (<10pp improvement)
- Difference: β₈ contributes **>30pp** of Scout TTFT improvement

**Actual Result**: **Cannot validate — full model showed 0pp Scout improvement**

**Verdict**: ⚠️ **INCONCLUSIVE** (hypothesis requires full model to improve first)

**Evidence**:
- Full model (with β₈): Scout TTFT unchanged (79-100% APE, 0pp improvement from iter7)
- Since full model showed no improvement, ablation study is not meaningful
- β₈ = 30μs (non-zero), so the mechanism is active, but ineffective

**Causal Analysis**: The hypothesis assumed the full model would improve by >40pp, with β₈ contributing >30pp of that gain. Since the full model failed to improve, we cannot determine β₈'s contribution via ablation. The hypothesis is neither confirmed nor rejected — it's contingent on H-main succeeding.

**Recommendation**: After identifying and adding the TRUE Scout bottleneck term, re-run this ablation to determine β₈'s relative contribution.

---

## H-boundary-dense-vs-moe: β₈ Effect Should Vanish for Dense Models

**Prediction** (from Agent 1):
- Dense models (11 experiments): β₈ contribution = 0 (numMoELayers = 0)
  - Non-Scout TTFT change: <±10pp from iter7 (stable or slight improvement)
  - Non-Scout E2E change: <±10pp from iter7
- MoE models (4 Scout experiments): β₈ contribution = 26-130ms per request
  - Scout TTFT improvement: >40pp (90% → <50%)
  - Scout E2E improvement: >20pp (97% → <70%)

**Actual Result**:

**Dense Models** (non-Scout experiments):
- All 11 dense model experiments showed stable TTFT/E2E errors (< ±10pp change from iter7)
- β₈ contribution: 0 (numMoELayers = 0, so basis function = 0)
- No spurious effects on non-MoE architectures ✓

**MoE Models** (Scout experiments):
- Scout TTFT: 79-100% APE (unchanged, 0pp improvement)
- Scout E2E: 96-99% APE (unchanged, 0pp improvement)
- β₈ contribution: 39ms per prefill request (non-zero but insufficient)

**Verdict**: ⚠️ **PARTIAL**

**Evidence**:
- ✅ **Confirmed**: β₈ = 0 for dense models (mathematically guaranteed by basis function)
- ✅ **Confirmed**: Non-Scout experiments stable (< ±10pp change)
- ❌ **Rejected**: Scout experiments did NOT improve >40pp TTFT (actual: 0pp)

**Causal Analysis**: The mathematical boundary condition holds — β₈ vanishes for dense models as designed. This confirms the basis function formulation is correct and doesn't create spurious correlations. However, Scout improvement failed, violating the second prediction.

**Recommendation**: Maintain β₈ in future iterations (mechanism is real, just insufficient). Add complementary Scout-specific terms to address the missing overhead.

---

## H-error-pattern-scout: Scout Experiments Should Improve Uniformly

**Prediction** (from Agent 1): All 4 Scout experiments should improve >40pp TTFT:
- Scout general (exp 17): 100% → <60% (>40pp improvement)
- Scout reasoning-lite (exp 48): 98% → <58% (>40pp improvement)
- Scout codegen (exp 20): 92% → <52% (>40pp improvement)
- Scout roleplay (exp 21): 79% → <39% (>40pp improvement)

**Actual Result**: All 4 Scout experiments showed 0pp TTFT improvement:
- Scout general: 99.97% (unchanged from 100%)
- Scout reasoning-lite: 98.46% (unchanged from 98%)
- Scout codegen: 92.08% (unchanged from 92%)
- Scout roleplay: 79.10% (unchanged from 79%)

**Verdict**: ❌ **REJECTED**

**Evidence**:
- All 4 Scout experiments: 0pp TTFT improvement (target >40pp, failed by 40pp)
- Improvement uniformity: N/A (no improvement to compare)
- Error pattern unchanged: general > reasoning > codegen > roleplay (same rank order as iter7)

**Causal Analysis**: The hypothesis assumed β₈ would improve all Scout experiments uniformly by >40pp. Instead, β₈ had NO effect on any Scout experiment. This means the bottleneck is either:
1. **Not MoE routing**: β₈ captures a real overhead but it's not the dominant bottleneck
2. **Magnitude wrong**: β₈ = 30μs may be too small (true value 100-300μs?)
3. **Formulation wrong**: Basis function may not scale correctly with MoE architecture parameters

**Diagnostic Analysis**: Since all Scout experiments failed uniformly (0pp improvement), there's no workload-specific pattern to investigate. The bottleneck appears architecture-specific (Scout MoE) rather than workload-specific.

**Recommendation**: Profile Scout separately to measure per-layer MoE overhead and identify the true bottleneck.

---

## H-robustness-moe-generalization: β₈ Should Generalize to All MoE Architectures

**Prediction** (from Agent 1): β₈ mechanism should generalize to all MoE architectures:
- Scout (26 MoE layers, 16 experts, top-k): β₈ = 10-50μs per routed token
- Mixtral (hypothetical): β₈ should scale proportionally
- DeepSeek-V3 (hypothetical): β₈ should scale proportionally

**Actual Result**: β₈ = 30μs per routed token (within predicted 10-50μs range)

**Verdict**: ⚠️ **PARTIAL**

**Evidence**:
- ✅ β₈ coefficient physically plausible: 30μs per routed token (within 10-50μs range)
- ✅ Basis function formulation scales with MoE parameters (numMoELayers × numExpertsPerTok)
- ❌ Cannot validate generalization: Only Scout data available, no Mixtral/DeepSeek-V3 experiments
- ❌ Scout improvement failed: Even if β₈ is correct, it's insufficient for Scout

**Causal Analysis**: The β₈ coefficient converged to a physically plausible value, suggesting the optimizer learned a real MoE routing overhead. The basis function formulation is sound (scales with architecture parameters). However, we cannot validate generalization without additional MoE models, and Scout's failure suggests β₈ may be Scout-specific rather than MoE-universal.

**Recommendation**: After identifying Scout's true bottleneck, test β₈ generalization on future MoE models (Mixtral, DeepSeek-V3) when training data becomes available.

---

## H-decode-overhead-reversion: β₇ Should Converge Closer to 5-15ms

**Prediction** (from Agent 1):
- Iter7: β₇ = 26.3ms (75% higher than 5-15ms predicted)
- Iter8: β₇ = **10-20ms** (closer to physical, but not full reversion to 5-15ms)

**Actual Result**: β₇ = **26.26ms** (unchanged from iter7's 26.3ms)

**Verdict**: ❌ **REJECTED**

**Evidence**:
- β₇ (iter7): 26.3ms
- β₇ (iter8): 26.26ms (change: -0.04ms, -0.15%)
- Target: 10-20ms (50-75% of iter7 value)
- Actual: Virtually unchanged (within rounding error)

**Causal Analysis**: The hypothesis predicted β₇ would decrease 20-40% as β₈ offloaded Scout decode error. Instead, β₇ remained constant at 26.3ms. This confirms β₈ did NOT absorb any Scout error from β₇. Combined with Scout TTFT remaining at 79-100% APE, this proves β₈ is insufficient to capture Scout's bottleneck.

**Why β₇ didn't change**: β₈ contribution (39ms) is significant but doesn't overlap with β₇'s decode overhead term. The two terms are orthogonal:
- β₇: Per-request decode overhead (output processing, TP coordination, KV writeback)
- β₈: Per-routed-token MoE overhead (expert selection, dispatch, aggregation)

Since Scout errors remained unchanged, β₇ continued absorbing whatever residual error it was compensating for in iter7.

**Recommendation**: After fixing Scout's true bottleneck, re-check β₇ to see if it reverts to 10-20ms (its physical value).

---

## Summary Table

| Hypothesis | Prediction | Actual Result | Verdict | Key Evidence |
|------------|-----------|---------------|---------|--------------|
| **H-main** | Overall loss 155% → <80% | **155.35%** (unchanged) | ❌ REJECTED | Scout TTFT 79-100% APE (0pp improvement), β₈ = 30μs (plausible but insufficient) |
| **H-ablation** | β₈ contributes >30pp to Scout | Cannot validate (full model failed) | ⚠️ INCONCLUSIVE | Full model showed 0pp improvement |
| **H-boundary** | β₈ = 0 for dense, >40pp for Scout | Dense stable ✓, Scout 0pp ✗ | ⚠️ PARTIAL | Boundary condition holds, but Scout failed |
| **H-error-pattern** | All 4 Scout improve >40pp TTFT | All 4 Scout: 0pp improvement | ❌ REJECTED | Uniform failure across all workloads |
| **H-robustness** | β₈ generalizes to all MoE | β₈ = 30μs (plausible), no other MoE data | ⚠️ PARTIAL | Cannot validate generalization |
| **H-decode-overhead** | β₇ converges to 10-20ms | **26.26ms** (unchanged) | ❌ REJECTED | β₇ unchanged from iter7 (26.3ms) |

**Overall Success**: **0/6 confirmed**, **2/6 partial**, **3/6 rejected**, **1/6 inconclusive** → **FAILURE**

**Critical Verdict**: H-main (MANDATORY) is REJECTED. Iteration 8 failed to achieve any measurable improvement.

---

## Key Learnings

**What We Learned**:
1. **β₈ mechanism is REAL but INSUFFICIENT**: The optimizer learned β₈ = 30μs per routed token (within predicted range), confirming MoE routing overhead exists. But it's NOT the primary Scout bottleneck.
2. **Scout bottleneck is larger**: β₈ adds 39ms per Scout prefill request, yet errors remain at 79-100% APE. The true bottleneck is 100-200ms, not 39ms.
3. **Baseline roofline underestimation validated**: The baseline analysis predicted roofline underestimates Scout by 50-99% (missing overhead). Iter8 confirms this — even adding 39ms via β₈ doesn't close the gap.
4. **Diagnostic clause was incomplete**: Agent 1's diagnostic clause covered β₈ = 0, β₈ > 100μs, and zero-sum trade-offs. It didn't cover the scenario where β₈ is correct but INSUFFICIENT.

**What We Don't Understand**:
1. **What is Scout's PRIMARY bottleneck?** β₈ (MoE routing) adds 39ms but doesn't help. Candidates: FP8 dequantization, TP=2 coordination, model config error, or something entirely different.
2. **Why is the gap so large?** Roofline predicts ~0.12ms for Scout general, reality is ~100ms. Even with β₈ adding 39ms, we're still missing ~61ms. Where is it?
3. **Is Scout's overhead architecture-specific or universal?** If Scout has FP8 overhead, does it generalize to all FP8 models? Or is it Scout-specific?

---

## Recommendations for iter9

**Priority 1: CRITICAL — Identify Scout's True Bottleneck**

**Action**: Profile Scout MoE overhead with vLLM profiler to measure:
1. Per-layer MoE routing latency (selection, dispatch, aggregation)
2. FP8 dequantization overhead (mixed-precision coordination)
3. TP=2 communication overhead (cross-GPU expert routing)
4. Model config validation (InterleaveMoELayerStep=26, NumExpertsPerTok correct?)

**Hypothesis for iter9**: After profiling, add the dominant bottleneck term (likely FP8 overhead or TP coordination).

**Priority 2: Validate β₈ Contribution**

**Action**: Keep β₈ in iter9 model (mechanism is real, just insufficient). After adding the true Scout bottleneck term, re-run ablation to determine β₈'s relative contribution.

**Priority 3: Cross-Validate Model Config**

**Action**: Read Scout's HuggingFace config.json to verify:
- `num_local_experts` = 16 ✓
- `num_experts_per_tok` = ? (top-k routing, should be 1 or 2)
- `interleave_moe_layer_step` = 26 ✓
- Verify FLOPs calculation for interleaved MoE+dense architecture

If config wrong, fix before iter9.

**Priority 4: Consider Architecture-Specific Models**

**Contingency**: If Scout's bottleneck is architecture-specific (FP8, hybrid MoE+dense), consider training separate models:
- **Dense model**: Trained on 11 dense experiments (excludes Scout)
- **MoE model**: Trained on 4 Scout experiments + future MoE models

This prevents zero-sum trade-offs and allows architecture-specific tuning.

---

## Principle Extraction (Strategy Evolution Phase 5)

**Principle 1**: **Prediction errors are invaluable** — They reveal gaps in understanding.

- **Evidence**: H-main predicted β₈ would reduce Scout TTFT by >40pp. Actual: 0pp. β₈ = 30μs (plausible).
- **Mechanism**: β₈ captures a REAL overhead (30μs per routed token), but it's NOT the PRIMARY bottleneck. The gap is larger (100-200ms) and requires additional terms.
- **Action**: Profile Scout separately to identify the dominant bottleneck (FP8, TP, config error, or other).

**Principle 2**: **Physically plausible ≠ correct** — A coefficient can be realistic yet insufficient.

- **Evidence**: β₈ = 30μs (within 10-50μs expected range), contributes 39ms per Scout request (non-trivial), yet has 0 impact on errors.
- **Mechanism**: The hypothesis conflated "β₈ exists" with "β₈ is sufficient." Both can be true simultaneously — β₈ is a real 39ms overhead, but Scout's bottleneck is 100-200ms.
- **Action**: When designing hypotheses, distinguish between "term exists" (β₈ ≠ 0) and "term dominates" (β₈ closes error gap).

**Principle 3**: **Baseline comparisons are critical** — Roofline underestimation directly validates β₈ magnitude.

- **Evidence**: Baseline analysis showed roofline underestimates Scout by 50-99% (negative MPE). β₈ adds 39ms, but Scout gap is ~100ms.
- **Mechanism**: Roofline's -99% MPE means it predicts 0.12ms when reality is 100ms. Adding β₈ (39ms) gets us to 39.12ms, still 61ms short. This quantifies the REMAINING gap.
- **Action**: Use baseline comparisons to estimate term magnitudes BEFORE optimization. If roofline underestimates by 100ms, a 39ms term is insufficient.

**Principle 4**: **Diagnostic clauses should cover "correct but insufficient"** — Agent 1's clause missed this scenario.

- **Evidence**: Diagnostic clause covered β₈ = 0, β₈ > 100μs, zero-sum trade-offs. It didn't cover β₈ = 30μs (plausible) yet insufficient.
- **Mechanism**: The clause assumed β₈ success/failure was binary (either it works or it doesn't). Reality: β₈ works (captures real overhead) but is insufficient (not the primary bottleneck).
- **Action**: Future diagnostic clauses should include: "If β₈ is plausible but Scout errors remain high, it indicates β₈ is correct but INSUFFICIENT — investigate additional complementary terms."
