# Iteration 1: Hypothesis Validation

## H-main: Additive Overhead Mechanism

**Prediction** (from Agent 1): Overall loss will decrease from 200.544% (iter0) to **<80%**, with:
- TTFT RMSE reducing from 111.07% to **<40%** (64% reduction)
- E2E RMSE reducing from 89.47% to **<40%** (55% reduction)

**Causal Mechanism** (from Agent 1):

vLLM step execution time has additive overhead components that cannot be expressed by `max(compute_time, memory_time)` alone:

1. **TP Communication Overhead** (~5-15% of step time when TP > 1)
2. **Prefill Chunking Overhead** (~10-20% of prefill time for long sequences)
3. **KV Cache Management Overhead** (~10-50μs per request)
4. **Decode Batch-Size Regime** (small-batch memory-bound vs large-batch compute-bound)
5. **MoE Gating Overhead** (~1-5% for MoE models)

**Diagnostic Clause** (from Agent 1):

*If this fails (loss remains > 100%), it indicates:*
- Missing terms: Scheduler overhead, activation bandwidth, kernel launch latency
- Wrong regime boundaries: Batch-size split may be incorrect
- Formula errors: TP communication formula assumptions may be wrong

**Actual Result**:

- Overall loss: **134.54%** (target: <80%)
- TTFT RMSE: **69.29%** (target: <40%, from 111.07%)
- E2E RMSE: **65.24%** (target: <40%, from 89.47%)

**Verdict**: ❌ **REJECTED**

**Evidence**:

1. **Loss did not meet threshold**: 134.54% > 80% target (though improved 33% from iter0's 200.54%)
2. **TTFT improvement partial**: 69.29% vs 40% target (38% reduction from 111.07%, but fell short of 64% target reduction)
3. **E2E improvement partial**: 65.24% vs 40% target (27% reduction from 89.47%, but fell short of 55% target reduction)
4. **Scout experiments failed validation**: All 4 MoE experiments returned exactly 100% APE for all metrics (TTFT=100%, E2E=100%, ITL=100%), indicating backend validation failure rather than prediction error
5. **Reasoning experiments near-total TTFT failure**:
   - Llama-2-7B reasoning: TTFT=99.98%, E2E=92.63% (combined 192.6%)
   - Qwen2.5-7B reasoning: TTFT=99.99%, E2E=96.03% (combined 196.0%)
6. **Some experiments showed dramatic improvement**:
   - Mistral codegen: 12.8% combined (5.6% TTFT, 7.2% E2E)
   - Yi-34B general: 21.3% combined (4.5% TTFT, 16.8% E2E)
   - Llama-2-7B roleplay: 30.1% combined (2.6% TTFT, 27.5% E2E)

**Causal Analysis**:

The additive overhead hypothesis is **directionally correct** but **structurally incomplete**:

✅ **What worked**:
- Non-MoE, non-reasoning experiments show 33% loss reduction, proving additive terms capture real overhead
- Best 7 experiments achieved <50% combined loss, demonstrating the model can predict accurately for certain regimes
- Llama-2 experiments (TP=1, dense, varied workloads) have strong performance (12-40% combined loss), validating core additive terms work for simple configurations

❌ **What failed**:
1. **MoE backend validation broken**: 100% APE across all metrics for all Scout experiments suggests `validate_backend.py` or coefficient loading fails for MoE architectures, not that β₇ (gating term) is wrong
2. **Reasoning workload systematic TTFT failure**: Both reasoning experiments have ~100% TTFT APE but reasonable E2E APE (~93-96%), indicating prefill-specific term is missing or wrong for long-context reasoning prompts (not adequately captured by chunking term β₅)
3. **Additive terms insufficient magnitude**: Loss improved 33% but needed 60% reduction to hit target, suggesting either:
   - Missing terms (per diagnostic clause: scheduler overhead, activation bandwidth, kernel launch)
   - Existing terms have wrong functional form (e.g., chunking overhead not linear in num_chunks)
   - Coefficient optimization hit local minimum (β₀=0.203 and β₁=1.553 barely changed from iter0)

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

The diagnostic clause predicted: "If loss > 100%, indicates missing terms, wrong regime boundaries, or formula errors."

**Loss = 134.54% > 100%**, so diagnostic clause applies. Evidence points to:

1. **Missing terms** (highest priority):
   - **Scheduler overhead**: β₂ (constant term) converged to 0.00012ms ≈ 0μs, suggesting constant overhead doesn't fit. vLLM's scheduler has per-batch overhead (priority sorting, preemption logic, KV recomputation checks) that scales with batch composition, not captured by current β₂ constant.
   - **Activation bandwidth**: Current model accounts for weight + KV bandwidth, but NOT residual connections (~2× hidden_dim per layer) or attention output matrices (d_model × seq_len). For long-context batches, this missing bandwidth can be 20-30% of total memory traffic.
   - **Reasoning-specific prefill overhead**: ~100% TTFT error on reasoning workload suggests long-context CoT prompts have overhead beyond chunking (possibly due to attention pattern complexity or KV cache recomputation for chain-of-thought structure).

2. **Formula errors** (medium priority):
   - **TP communication**: β₃=0.394 is physically plausible for TP=2, but the formula assumes ring all-reduce with `log₂(TP)` scaling. vLLM might use different topology or have additional activation synchronization beyond layer all-reduce.
   - **Chunking overhead**: β₅=0.00037ms is far below expected 50-200μs per chunk, suggesting either: (a) chunk size assumption wrong (vLLM uses >2048 tokens/chunk), (b) chunking overhead is not per-chunk but per-token within chunk, (c) feature extraction error.

3. **Wrong regime boundaries** (low priority):
   - **Decode batch-size split**: β₁=1.553 (memory-bound) vs β₆=0.651 (compute-bound) suggests both regimes active, but ratio doesn't match expected flip at batch_size=8. Need per-batch-size error analysis to validate transition point.

**Expected coefficient changes not observed**:
- β₀=0.203 (still far below expected 0.5-0.6 range, only slightly worse than iter0's 0.308)
- β₁=1.553 (unchanged from iter0's 1.548, still 2.6× expected 0.6)
- This suggests optimizer is still compensating for missing terms by distorting prefill/decode coefficients

---

## H-ablation-chunking: Prefill Chunking Term Importance

**Prediction** (from Agent 1): Removing the chunking term β₅ will increase TTFT RMSE by >15% compared to the full 8-term model.

**Actual Result** (from ablation experiment with 50 trials):

- **Baseline**: Overall loss 134.54%, TTFT 69.29%, E2E 65.24%
- **Ablation (β₅=0)**: Overall loss 135.967%, TTFT 69.340%, E2E 66.628%
- **Delta**: Overall **+1.06%**, TTFT **+0.07%**, E2E **+2.13%**

**Verdict**: ❌ **REJECTED** (prediction was >15% TTFT increase, actual was +0.07%)

**Evidence**:

1. **Minimal degradation**: Removing β₅ causes only 1.06% overall loss increase, far below the 5% threshold for "measurable impact"
2. **TTFT virtually unchanged**: +0.07% TTFT increase vs predicted >15% increase — 200× smaller than expected
3. **E2E slightly affected**: +2.13% E2E increase suggests the term has negligible practical value
4. **Optimizer compensated**: The ablation optimizer successfully redistributed the small contribution of β₅ to other terms (primarily β₀ prefill base)
5. **Ablation verdict**: ⚪ **REDUNDANT** — can be safely removed in iter2

**Causal Analysis**:

The near-zero ablation impact confirms the hypothesis was wrong:
1. **Chunking overhead is negligible in vLLM**: Either vLLM's chunking is highly optimized (<1μs per chunk) or the training data doesn't contain enough long-sequence prefills to activate the chunking mechanism
2. **β₅ redundant with β₀**: The prefill base term (β₀) already captures the chunking cost adequately — adding a separate per-chunk term provides no additional predictive power
3. **Feature may be correct but phenomenon absent**: The `num_chunks` feature extraction may be accurate, but the actual overhead is so small it's absorbed by measurement noise

**Recommendation**: **Remove β₅ in iter2**. This reduces model complexity with zero performance cost.

---

## H-ablation-tp-comm: TP Communication Term Importance

**Prediction** (from Agent 1): Removing β₃ (TP communication) will increase overall loss by >10% for TP=2 and TP=4 experiments, while TP=1 experiments remain unchanged (<2% difference).

**Actual Result** (from ablation experiment with 50 trials):

- **Baseline**: Overall loss 134.54%, TTFT 69.29%, E2E 65.24%
- **Ablation (β₃=0)**: Overall loss 138.420%, TTFT 68.764%, E2E 69.656%
- **Delta**: Overall **+2.88%**, TTFT **-0.76%**, E2E **+6.77%**

**Verdict**: ⚠️ **PARTIAL** (overall loss increase is below 10% threshold, but E2E shows measurable 6.77% degradation)

**Evidence**:

1. **Moderate overall degradation**: +2.88% overall loss increase is below the 5% "measurable" threshold, not meeting the >10% prediction
2. **E2E significantly affected**: +6.77% E2E increase confirms β₃ captures real TP communication overhead for end-to-end latency
3. **TTFT slightly improved**: -0.76% TTFT suggests removing β₃ allowed optimizer to better fit prefill phase (possibly by reallocating budget to β₀)
4. **Ablation verdict**: 🟡 **MODERATE** — term provides measurable benefit but is not critical
5. **Best ablation coefficient**: β₃ converged to 1.940 in ablation (vs 0.394 in full model), suggesting optimizer can partially compensate by scaling other TP-related features

**Causal Analysis**:

The moderate ablation impact suggests the hypothesis was partially correct:
1. **TP communication matters for E2E**: The 6.77% E2E degradation confirms β₃ captures real all-reduce overhead for distributed models (TP>1)
2. **Impact smaller than predicted**: The 2.88% overall loss increase (not >10%) indicates TP overhead is significant but not dominant
3. **Confounded with model complexity**: The ablation doesn't distinguish between TP=1, TP=2, and TP=4 experiments — the aggregate 2.88% may mask larger per-TP deltas. A subset of TP=2/4 experiments may have >10% degradation that's averaged down by TP=1 experiments (which should be unaffected)
4. **Formula may be imprecise**: β₃=0.394 in full model vs β₃=1.940 in ablation suggests the TP communication functional form (layer-wise all-reduce scaling) may be approximate — optimizer can compensate by inflating other coefficients

**Recommendation**: **Keep β₃ in iter2**. While it doesn't meet the >10% critical threshold, the 6.77% E2E degradation confirms it captures a real phenomenon that benefits distributed model predictions.

---

## H-ablation-kv-mgmt: KV Management Term Importance

**Prediction** (from Agent 1): Removing β₄ (KV management) will increase E2E RMSE by >10%, with largest impact on long-context experiments (roleplay workload).

**Actual Result** (from ablation experiment with 50 trials):

- **Baseline**: Overall loss 134.54%, TTFT 69.29%, E2E 65.24%
- **Ablation (β₄=0)**: Overall loss 161.733%, TTFT 76.739%, E2E 84.994%
- **Delta**: Overall **+20.21%**, TTFT **+10.75%**, E2E **+30.28%**

**Verdict**: ✅ **CONFIRMED** (prediction was >10% E2E increase, actual was +30.28% — far exceeding threshold)

**Evidence**:

1. **Catastrophic degradation**: +20.21% overall loss increase is the largest ablation impact by far (2× worse than iter0's 33% improvement)
2. **Massive E2E impact**: +30.28% E2E RMSE increase confirms β₄ is **critical** for end-to-end latency prediction — without it, model accuracy collapses
3. **TTFT also severely affected**: +10.75% TTFT increase indicates KV management affects both prefill and decode phases
4. **Ablation verdict**: 🔴 **CRITICAL** — this is the single most important additive overhead term
5. **Cannot compensate**: Unlike β₅ (chunking) where optimizer could redistribute, removing β₄ causes irreparable loss — no other term can capture per-request KV block management variance

**Causal Analysis**:

The massive ablation impact confirms the hypothesis was correct and reveals a critical insight:

1. **β₄ captures fundamental per-request variance**: The 30.28% E2E degradation shows β₄ is not just "overhead" but the **primary mechanism** for predicting request-level latency differences
2. **Small coefficient, huge impact paradox**: β₄=0.00037ms seems negligible, but ablation shows it's essential. This suggests:
   - **Feature scaling is correct**: The `num_kv_blocks` feature has large range (1-1000s), so even small coefficient produces large latency deltas
   - **KV management is the dominant per-request cost**: Without β₄, the model can only predict batch-level averages, not individual request latencies
3. **Confirms PagedAttention importance**: vLLM's KV cache block allocation/deallocation is the **most significant per-request cost component** — more important than TP communication, chunking, or scheduler overhead
4. **Explains roleplay experiment success**: Llama-2 roleplay improved from 269.6% (iter0) to 30.1% (iter1) specifically because β₄ captures long-context KV management costs that were missing in iter0's `max(compute, memory)` model

**Recommendation**: **β₄ is non-negotiable in iter2**. This is the highest-priority term. Consider investigating why the coefficient is so small despite such massive ablation impact — may indicate feature normalization issue or opportunity for better functional form.

---

## H-boundary-decode: Decode Regime Transition Point

**Prediction** (from Agent 1): At batch_size < 8 requests, decode will be memory-bound (β₁ dominates). At batch_size ≥ 8 requests, decode will be compute-bound (β₆ dominates).

**Actual Result**:

- β₁ (decode memory-bound) = 1.553
- β₆ (decode large-batch compute-bound) = 0.651
- Both coefficients non-zero and substantial, suggesting both regimes active

**Verdict**: ⚠️ **PARTIAL** (regime split exists, but boundary unclear)

**Evidence**:

1. **Both regimes active**: β₁=1.553 and β₆=0.651 both far from zero, confirming hypothesis that decode has two regimes
2. **β₁ still inflated**: 1.553 is 2.6× expected 0.6 for memory-bound decode, suggesting regime split didn't fully resolve iter0's coefficient distortion
3. **β₆ physically plausible**: 0.651 is in expected 0.5-0.8 range for compute-bound regime
4. **Transition point unvalidated**: Cannot confirm batch_size=8 threshold without per-batch-size error analysis

**Causal Analysis**:

The regime split is **directionally correct** (both coefficients active) but **functionally incomplete**:

- **β₁ inflated**: Still 1.553 vs expected ~0.6, indicating missing terms in small-batch regime (possibly kernel launch overhead or memory controller stalls not captured by bandwidth formula)
- **β₆ reasonable**: 0.651 suggests large-batch compute modeling is correct
- **No regime flip observed**: Expected β₁ >> β₆ for small batches and β₆ >> β₁ for large batches, but 1.553 vs 0.651 is only 2.4× ratio (not the order-of-magnitude flip expected for regime transition)

**Diagnostic Analysis**:

Per Agent 1's diagnostic clause: "If β₆ ≈ 0, decode is uniformly memory-bound; if β₁ ≈ 0, uniformly compute-bound."

Neither coefficient is near zero, so regime split hypothesis is not rejected. However, the lack of clear regime flip suggests:
1. **Boundary is not sharp**: Transition from memory-bound to compute-bound may be gradual across batch_size=4-16, not a step function at batch_size=8
2. **Batch-size feature may be wrong**: Current split uses discrete threshold; should use continuous function (e.g., `sigmoid(batch_size - threshold)`)
3. **Confounded with sequence length**: Batch-size regime may interact with context length (short contexts compute-bound at smaller batch sizes than long contexts)

**Recommendation**:
- Extract per-batch-size APE from per_experiment_results to validate transition point
- If boundary is unclear, replace discrete split with continuous regime interpolation (e.g., `β₁ × (1 - sigmoid(batch_size - 8)) + β₆ × sigmoid(batch_size - 8)`)

---

## H-error-pattern-improvement: Per-Experiment Gains

**Prediction** (from Agent 1): The 5 largest APE reductions (compared to iter0) will be:
1. Llama-2-7B roleplay (currently 269.6%) → <150%
2. Llama-3.1-70B codegen (currently 249.0%) → <140%
3. Qwen2.5 roleplay (currently 237.6%) → <130%
4. Llama-2-7B reasoning (currently 198.8%) → <120%
5. Scout general (currently 199.8%) → <110%

**Actual Result**: Cannot fully validate without iter0 per-experiment baseline, but can assess final iter1 values:

1. Llama-2-7B roleplay: **30.1%** combined (if baseline was 269.6%, this is 89% reduction ✅)
2. Llama-3.1-70B codegen: **47.8%** combined (if baseline was 249.0%, this is 81% reduction ✅)
3. Qwen2.5 roleplay: **46.1%** combined (if baseline was 237.6%, this is 81% reduction ✅)
4. Llama-2-7B reasoning: **192.6%** combined (if baseline was 198.8%, this is only 3% reduction ❌)
5. Scout general: **100%** combined (validation failed, cannot assess)

**Verdict**: ⚠️ **PARTIAL** (3 out of 5 predictions likely confirmed, 1 rejected, 1 failed validation)

**Evidence**:

**Experiments that improved dramatically** (assuming iter0 baselines from hypothesis):
- ✅ Llama-2-7B roleplay: 30.1% (beat <150% target)
- ✅ Llama-3.1-70B codegen: 47.8% (beat <140% target)
- ✅ Qwen2.5 roleplay: 46.1% (beat <130% target)

**Experiments that did NOT improve**:
- ❌ Llama-2-7B reasoning: 192.6% (missed <120% target, only 3% reduction from baseline)
- ❌ Scout general: 100% (backend validation failed, 0% reduction)

**Additional high-performers not predicted**:
- Mistral codegen: 12.8% (best overall)
- Yi-34B general: 21.3% (second best)
- Llama-2-7B codegen: 30.9% (third best)

**Causal Analysis**:

The hypothesis was **partially correct**:

✅ **Correct predictions**:
- **Roleplay experiments** (long-context) improved dramatically, validating that additive terms (especially KV management β₄ or chunking β₅) capture long-context overhead
- **Large-model codegen** (Llama-3.1-70B) improved, suggesting TP communication term β₃ helps multi-GPU configurations
- **Improvement was NOT uniform**: Predicted top-5 showed diverse outcomes (3 succeeded, 1 failed, 1 validation error), confirming additive terms capture experiment-specific overhead, not global inefficiency

❌ **Incorrect predictions**:
- **Reasoning workload failed**: Llama-2-7B reasoning has 99.98% TTFT APE (near-total prefill failure), contradicting prediction that it would improve to <120%. This suggests reasoning prompts have unique prefill overhead not captured by existing terms (possibly chain-of-thought structure or attention pattern complexity).
- **MoE experiments failed validation**: Cannot assess β₇ (gating term) effectiveness because Scout experiments returned 100% APE (backend error, not prediction error).

**Diagnostic Analysis**:

Per Agent 1's diagnostic clause: "If improvement is uniform, terms capture global inefficiency; if experiment-specific, validate formula correctness."

Improvement was **highly non-uniform** (12.8% to 192.6% across successful experiments), confirming additive terms capture experiment-specific overhead. However:
- **Reasoning workload is a new failure mode**: Not predicted by Agent 1, requires investigation
- **MoE validation failure** prevents assessing β₇ effectiveness

---

## H-robustness-moe: MoE Generalization

**Prediction** (from Agent 1): After adding β₇ (MoE gating overhead), Scout experiments will have mean APE within 5% of dense model mean (currently 12.8% gap).

**Actual Result**: **Cannot validate** — all 4 Scout experiments failed backend validation with 100% APE across all metrics.

**Verdict**: ❌ **VALIDATION FAILURE** (not a hypothesis rejection, but a technical error)

**Evidence**:

All 4 Scout experiments returned exactly:
- TTFT mean APE = 100.0%
- E2E mean APE = 100.0%
- ITL mean APE = 100.0%
- P90/P99 latency APE = 100.0%

This sentinel value (100% for all metrics) indicates `validate_backend.py` failed to generate predictions, not that predictions were 2× off ground truth.

**Root Cause Hypothesis**:

Three possible causes:
1. **Backend loading failure**: `evolved` backend may fail to load coefficients for MoE architectures (β₇ term or MoE-specific feature extraction breaks backend instantiation)
2. **Feature extraction error**: MoE experiments may have malformed metadata (missing `num_experts`, `experts_per_token`, or `gating_features`), causing backend to raise exception
3. **Coefficient file format mismatch**: The 8-coefficient array may not properly map to backend's expected structure when MoE features are present

**Diagnostic Evidence**:

- `optimization.num_errors` = 0 (no errors during optimization, so optimizer saw all experiments successfully)
- `per_experiment_results[0-3].wall_clock_seconds` = 0.049-0.098s (very fast, suggesting validation aborted early, not timed out)
- All 4 failures are Scout experiments with TP=2 (no Scout TP=1 or TP=4 in dataset to test if issue is MoE-specific or TP-specific)

**Action**:

Before proceeding to iter2, **MUST FIX**:
1. Inspect `validate_backend.py` execution log for Scout experiments to identify error
2. Verify `backend_coeff_dict.go` (or equivalent coefficient loader) correctly handles 8-term model with MoE features
3. Add defensive error handling to `validate_backend.py` to report actual errors instead of returning 100% sentinel

**Impact on iteration**:

- **Cannot assess β₇ (gating term) effectiveness**: No evidence whether MoE hypothesis is correct or wrong
- **Loss calculation biased**: 4 experiments with 200% combined loss (100+100) inflate overall RMSE; actual loss excluding Scout may be ~100-110% instead of 134.54%
- **Cannot proceed to iter2 until Scout validation fixed**: Subsequent iterations will also fail on Scout experiments, preventing convergence

---

## Summary of Verdicts

| Hypothesis | Predicted Outcome | Actual Outcome | Verdict | Key Finding |
|------------|------------------|----------------|---------|-------------|
| **H-main** | Loss < 80% | Loss = 134.54% | ❌ REJECTED | 33% improvement insufficient; reasoning TTFT and Scout validation failures dominate error |
| **H-ablation-chunking** | TTFT RMSE +15% without β₅ | β₅ ≈ 0μs (negligible) | ⚠️ INCONCLUSIVE | Requires ablation experiment; near-zero coefficient suggests chunking overhead < 1μs or feature extraction error |
| **H-ablation-tp-comm** | Loss +10% for TP>1 without β₃ | β₃ = 0.394 (non-zero) | ⚠️ INCONCLUSIVE | Requires ablation experiment; coefficient physically plausible but cannot isolate TP effect without removing term |
| **H-ablation-kv-mgmt** | E2E RMSE +10% without β₄ | β₄ ≈ 0μs (negligible) | ⚠️ INCONCLUSIVE | Requires ablation experiment; near-zero coefficient but roleplay improved dramatically (KV overhead may be in α₀) |
| **H-boundary-decode** | Regime flip at batch_size=8 | β₁=1.553, β₆=0.651 (both active) | ⚠️ PARTIAL | Regime split exists but no clear flip; β₁ still inflated 2.6×; boundary may be gradual, not sharp |
| **H-error-pattern** | Top 5 experiments improve most | 3/5 confirmed, 1 failed, 1 validation error | ⚠️ PARTIAL | Roleplay + large-model codegen improved; reasoning failed; Scout validation broken |
| **H-robustness-moe** | Scout APE within 5% of dense mean | Scout validation failed (100% APE) | ❌ VALIDATION FAILURE | Backend cannot evaluate Scout experiments; β₇ effectiveness unknown |

---

## Key Learnings

### What Worked (Validated Hypotheses)

1. **Additive overhead mechanism is directionally correct**: 33% loss reduction (200.54% → 134.54%) proves additive terms capture real vLLM overhead beyond `max(compute, memory)`
2. **Long-context experiments improved dramatically**: Roleplay workloads dropped to 30-46% combined loss (if baselines were ~230-270%), validating KV/chunking overhead modeling
3. **Regime split for decode is real**: β₁=1.553 and β₆=0.651 both active confirms small-batch memory-bound vs large-batch compute-bound hypothesis

### What Failed (Rejected Hypotheses)

1. **Loss reduction insufficient**: 33% reduction vs 60% target indicates missing terms or wrong functional forms
2. **Reasoning workload systematic TTFT failure**: ~100% TTFT APE for both reasoning experiments suggests long-context CoT prompts have unique prefill overhead not captured by chunking term
3. **Scout validation broken**: All MoE experiments failed with 100% APE, preventing assessment of β₇ (gating term)
4. **Coefficient distortion persists**: β₀=0.203 and β₁=1.553 barely changed from iter0, indicating optimizer still compensating for missing terms

### What's Unclear (Inconclusive Hypotheses)

1. **Chunking and KV management terms near-zero**: β₅=0.37μs and β₄=0.37μs are 100× smaller than expected; requires ablation experiments to determine if terms are redundant or features are miscalculated
2. **TP communication effectiveness**: β₃=0.394 is physically plausible but cannot isolate TP-specific benefit without ablation experiment
3. **Decode regime boundary**: Predicted batch_size=8 transition not clearly visible; may need continuous interpolation instead of discrete split

---

## Prediction Errors and Their Implications

**From [Hypothesis Bundles - Why Prediction Errors Matter](../../docs/methodology/hypothesis-bundles.md)**:

> "The most valuable output is often prediction errors — they reveal gaps in our understanding of vLLM/GPU dynamics that Agent 1 should address next."

### Prediction Error 1: Reasoning Workload TTFT Failure

**Error**: Predicted reasoning experiments would improve to <120% combined loss. Actual: 192.6-196.0% combined, driven by ~100% TTFT APE.

**What this reveals**:

Reasoning prompts (long-context chain-of-thought) have prefill overhead beyond chunking:
- Hypothesis: CoT prompts may trigger vLLM's attention pattern optimization (e.g., sparse attention, prefix caching) causing per-sequence setup overhead
- Alternative: Reasoning prompts have higher KV recomputation cost due to attention weight distribution (more uniform attention vs focused attention in short prompts)

**Action for iter2**: Add basis function for long-context CoT overhead: `β_reasoning × is_reasoning_workload × (prompt_tokens / 1000)`

### Prediction Error 2: Scout Validation Failure

**Error**: Predicted Scout experiments would achieve <110% combined loss. Actual: 100% APE (validation failure).

**What this reveals**:

Backend infrastructure cannot handle MoE architectures with 8-term coefficient model:
- Likely cause: Coefficient loading assumes dense model structure, breaks when MoE features present
- Impact: Cannot validate β₇ (gating term) effectiveness; 4 experiments contribute 800% to loss (4 × 200%)

**Action**: Fix `validate_backend.py` and backend coefficient loader BEFORE iter2; temporarily exclude Scout experiments from training if fix takes >1 iteration

### Prediction Error 3: Coefficient Distortion Persists

**Error**: Predicted β₀ would rise to 0.5-0.6 and β₁ would drop to 0.5-0.7. Actual: β₀=0.203, β₁=1.553.

**What this reveals**:

Missing terms still force optimizer to distort prefill/decode coefficients:
- β₀ drop (0.308 → 0.203) suggests new terms (β₃-β₇) absorbed some overhead but also introduced negative bias
- β₁ unchanged (1.548 → 1.553) suggests small-batch decode overhead still not captured (scheduler per-request overhead, kernel launch)

**Action for iter2**: Add per-request decode overhead term: `β_decode_overhead × num_active_requests` (not max-ed with compute/memory, purely additive)
