# Iteration 4: Findings and Principles

## Summary

Iter4 tested the hypothesis that **activation memory bandwidth** is the missing prefill term causing β₀ to be artificially low (0.169, far below physical 0.40-0.55). The hypothesis was **conclusively rejected** by multiple lines of evidence:

- Overall loss improved by only 3.93% (133.13% → 129.20%), missing the <110% target by 19 percentage points
- Reasoning experiments (the key test case) showed **ZERO improvement** (stayed at 99.98-99.99% TTFT)
- β₀ **decreased** from 0.169 → 0.165 instead of rising to 0.25-0.35
- Other coefficients destabilized dramatically: β₁ +73.8%, β₂ +328%, β₅ +160%

**Key learning**: The 3.93% improvement came from **simplification** (removing ineffective β₂/β₇ terms), not from adding β₆ (activation bandwidth). The new β₆ term is misspecified and causing coefficient drift across the model.

**Critical insight**: Reasoning experiments' 0% improvement despite adding a term that should scale with prompt length indicates the missing overhead is **qualitatively different** — likely a fixed per-operation cost (kernel launch) or algorithmic switch (attention kernel for long contexts), not another memory bandwidth term.

---

## Error Analysis

### Systematic Patterns

**By workload category**:
1. **Catastrophic (>80% combined loss)**: Reasoning (4 exps) + Scout general/reasoning (2 exps)
2. **High-error (50-80% combined loss)**: Scout codegen/roleplay (2 exps), Mistral TP=2 general-lite (1 exp)
3. **Moderate-error (30-50% combined loss)**: Llama-2 roleplay/codegen (2 exps), Llama-3.1 TP=4 general-lite (1 exp), Qwen roleplay (1 exp)
4. **Low-error (<30% combined loss)**: Yi-34B (1 exp), Llama-3.1 TP=4 codegen (1 exp), Mistral codegen (1 exp)

**Key observation**: Error strongly correlates with **workload type** and **architecture**, NOT with coefficient values or optimization quality.

### High-Error Experiments (APE > 80%)

**Reasoning workload** (catastrophic underestimation):
- Qwen2.5-7B reasoning-1-1: TTFT=99.99%, E2E=95.22%, combined=195.21%
- Llama-2-7B reasoning: TTFT=99.98%, E2E=92.49%, combined=192.47%
- Scout reasoning-2: TTFT=99.99%, E2E=93.09%, combined=193.08%

**Why reasoning fails**:
1. **TTFT is 1000× underestimated** (99.99% APE = model predicts ~1ms, actual ~1000ms)
2. **E2E is also underestimated** (92-95% APE), but less dramatically
3. **Prefill is the bottleneck**: TTFT error drives E2E error for reasoning workload

**Scout MoE experiments** (general/reasoning):
- Scout general-2: TTFT=99.97%, E2E=94.91%, combined=194.88%
- Scout reasoning-2: TTFT=99.99%, E2E=93.09%, combined=193.08%

**Why Scout fails**:
- Interleaved MoE+dense architecture not captured by single β₀ (prefill) or β₅ (MoE gating)
- Current model assumes uniform layer type (all dense OR all MoE)
- Scout alternates MoE and dense layers → need per-layer-type coefficients

**Pattern**: All reasoning experiments + Scout general/reasoning share:
1. Long prompts (8K-16K tokens)
2. Prefill-dominated workload (TTFT >> ITL)
3. TTFT underestimation by 100× to 1000×

### Medium-Error Experiments (APE 50-80%)

**Scout shorter-context workloads**:
- Scout codegen-2: TTFT=89.69%, E2E=83.98%, combined=173.67%
- Scout roleplay-2: TTFT=84.00%, E2E=75.61%, combined=159.61%

**Mistral TP=2 general-lite**:
- Mistral-Nemo TP=2 general-lite-2-1: TTFT=76.90%, E2E=64.56%, combined=141.46%

**Pattern**: Medium-length prompts (2K-4K tokens) with MoE or TP=2 configs show 50-80% TTFT error, suggesting:
- Missing TP-dependent prefill overhead (not communication, which was rejected in iter3)
- MoE routing overhead during prefill (current β₅ only captures decode gating)

### Low-Error Experiments (APE < 30%)

**Best performers**:
- Yi-34B TP=2 general-lite-2-1: TTFT=11.29%, E2E=29.09%, combined=40.38%
- Llama-3.1-70B TP=4 codegen-4-1: TTFT=22.70%, E2E=12.58%, combined=35.28%
- Llama-2-7B codegen: TTFT=39.43%, E2E=5.96%, combined=45.39%

**What makes these easy to predict**:
- **Decode-dominated** (high output tokens): E2E is low-error (5.96-12.58%)
- **Short-to-medium prefill** (1K-3K tokens): TTFT is moderate-error (11-40%)
- **Dense models with good TP scaling** (Yi-34B, Llama-3.1-70B)

**Pattern**: Decode-dominated workloads with short prefill are well-modeled by current β₁ (decode memory) and β₄ (decode compute).

### Error Correlations

**✅ Confirmed correlations**:
1. **Prompt length → TTFT error**: Longer prompts (>4K) have higher TTFT error
   - Reasoning (8K-16K): 99.98-99.99% TTFT
   - General-lite (2K-4K): 29-77% TTFT
   - Codegen/roleplay (<2K): 10-40% TTFT

2. **Workload type → E2E error**: Reasoning has highest E2E error (92-95%) due to TTFT dominating
   - Reasoning: 92-95% E2E (prefill-dominated)
   - General-lite: 28-64% E2E (mixed prefill/decode)
   - Codegen: 6-42% E2E (decode-dominated)

3. **Architecture → overall error**: MoE experiments have systematically higher error
   - Scout (MoE): 159-195% combined loss (4 experiments)
   - Dense models: 35-141% combined loss (11 experiments)

**❌ Rejected correlations**:
1. **TP degree → error**: TP=1/TP=2/TP=4 experiments show similar error distributions
   - TP=1: 40-195% combined loss (7 experiments)
   - TP=2: 40-195% combined loss (5 experiments)
   - TP=4: 35-60% combined loss (3 experiments)
   - **Conclusion**: TP communication is NOT the dominant error source (consistent with iter3 findings)

2. **Model size → error**: Large models (70B) don't systematically have higher error than small models (7B)
   - Llama-2-7B: 45-192% combined loss
   - Yi-34B: 40% combined loss (best!)
   - Llama-3.1-70B: 35-60% combined loss
   - **Conclusion**: Model size is not predictive of error

### Root Cause Hypotheses

Based on the error patterns, three root causes emerge:

#### **Principle 1**: Reasoning workload has a qualitatively different prefill bottleneck

**Evidence**:
- 4 reasoning experiments: 99.98-99.99% TTFT (1000× underestimation)
- 0% improvement from activation bandwidth term (should improve >25% if BW-limited)
- β₀ decreased instead of rising (0.169 → 0.165)
- E2E error follows TTFT error (92-95% E2E for reasoning)

**Mechanism**:
The 1000× underestimation cannot be explained by any continuous bottleneck (compute, memory bandwidth, communication). Physical constraints:
- Memory bandwidth: max 3-5× slowdown (HBM limits)
- Compute: max 2-3× slowdown (MFU limits)
- Communication: max 2× slowdown (NVLink limits)

A 1000× slowdown requires either:
1. **Algorithmic switch**: vLLM uses different attention kernel for long contexts (>8K tokens)
   - FlashAttention-2 → slower kernel with better memory efficiency
   - Or: PagedAttention overhead scales super-linearly with context length
2. **Kernel launch overhead**: 50μs × 200 kernels × 80 layers = 800ms (close to observed 1000ms)
3. **Scheduler batching overhead**: vLLM may batch reasoning requests differently (larger batches → longer wait times)

**Action for iter5**:
- **CRITICAL**: Profile vLLM reasoning experiments with `nsys profile` to identify actual bottleneck
- Measure: kernel launch count, attention kernel type, scheduler queue depth, batch formation latency
- If kernel launch: Add `β_kernel_launch = num_layers × num_kernels_per_layer × 50μs`
- If attention algorithm: Add separate β₀ for long-context prefill (context > 8K)

---

#### **Principle 2**: Activation bandwidth hypothesis is wrong, but simplification strategy is validated

**Evidence from H-simplification (⚠️ PARTIAL)**:
- Removing β₂ (scheduler ≈ 0) and β₇ (TP prefill comm ≈ 0) improved loss by 3.93%
- Pattern from iter2→iter3→iter4: Removing ineffective terms ALWAYS improves results
  - Iter3: Removed β₇/β₈ (very long context + per-request decode) → +3.06% improvement
  - Iter4: Removed β₂/β₇ (scheduler + TP prefill comm) → +3.93% improvement
- No experiments degraded by >5% after removal

**Evidence from H-main (❌ REJECTED)**:
- β₆ (activation BW) converged to 1.818, not 3.0-6.0 (40-70% lower than expected)
- Reasoning experiments improved by 0%, not >25%
- β₀ decreased instead of rising (0.169 → 0.165)
- Other coefficients destabilized: β₁ +73.8%, β₂ +328%, β₅ +160%

**Mechanism**:
The improvement came from **pruning ineffective terms** (reducing 10 → 8 parameters), NOT from adding β₆. The activation bandwidth formula is misspecified:

1. **Wrong scale factor**: k=4-6 may be too high (actual k~1.5-2)
2. **Wrong basis**: Activation writes may not be the dominant overhead
3. **Collinearity**: β₆ overlaps with β₀ (both scale with tokens × layers), causing coefficient drift

**Why did coefficients explode?**
- **β₁ (decode memory) +73.8%**: Absorbing error from misspecified β₆
- **β₂ (TP comm) +328%**: Absorbing error that correlates with TP configs
- **β₅ (MoE gating) +160%**: Absorbing error that correlates with MoE experiments

This is classic **coefficient drift** caused by adding a misspecified term.

**Action for iter5**:
- **Remove β₆** (activation bandwidth) entirely — current formula is harmful
- **Continue simplification**: Check if any other terms have coefficients near zero
- **Before adding new terms**: Ensure they don't create collinearity with existing terms (especially β₀)

---

#### **Principle 3**: Scout experiments confirm need for per-layer-type coefficients

**Evidence**:
- 4 Scout experiments: 159-195% combined loss (worst in training set)
- Scout architecture: Interleaved MoE and dense layers (novel to LLaMA-4-Scout family)
- Current model: Single β₀ (prefill) and β₅ (MoE gating) assume uniform layer type
- Pattern: Scout codegen/roleplay (shorter context) slightly better than Scout general/reasoning (longer context)

**Mechanism**:
LLaMA-4-Scout alternates between:
1. **Dense layers**: Standard transformer (FLOPs = 2 × M × I × O)
2. **MoE layers**: Sparse routing (FLOPs = 2 × M × I × O / num_experts_per_token)

Current β₀ computes FLOPs assuming ALL layers are dense OR all layers are MoE (per `ModelConfig.MoELayerFraction`). But Scout has:
- 50% dense layers
- 50% MoE layers
- No way to separate their compute times

**Why this causes 159-195% error**:
- Dense layers may achieve 40-50% MFU
- MoE layers may achieve 20-30% MFU (lower due to routing overhead)
- Current β₀ = 0.165 (16.5% MFU) is a bad compromise — too low for dense, too high for MoE

**Action for iter5**:
- **Add per-layer-type coefficients**:
  - β₀_dense: Prefill MFU for dense layers
  - β₀_moe: Prefill MFU for MoE layers
  - Formula: `prefill_time = β₀_dense × dense_flops + β₀_moe × moe_flops`
- **Requires backend changes**: `evolved_model.go` needs to split FLOPs by layer type
- **Alternative**: Use end-to-end validation for Scout (accept 150-200% error during training)

---

## Coefficient Analysis

**Alpha [α₀, α₁, α₂]** (request-level overheads):
- α₀ = 0.001498 ms = 1.498 ms fixed API overhead per request
- α₁ = 0.0001247 ms/token = 124.7 μs per input token
- α₂ = 0.00003599 ms/token = 36.0 μs per output token

**Physical interpretation**:
- α₀ (1.5 ms): Reasonable for API call, JSON parsing, request validation
- α₁ (125 μs/token): Matches tokenization + input validation overhead
- α₂ (36 μs/token): Matches output detokenization + response formatting

**Outliers**: None. Alpha coefficients are stable and physically plausible.

---

**Beta [β₀, β₁, β₂, β₃, β₄, β₅, β₆]** (step-level basis functions):

**β₀ = 0.1654** (prefill compute MFU, down from 0.1688 in iter3)
- **Physical interpretation**: 16.54% MFU during prefill (far below ideal 40-55%)
- **Trend**: Decreased by 2% instead of rising to 0.25-0.35 as predicted
- **Problem**: Still far from physical plausibility, and moving in WRONG direction
- **Conclusion**: β₀ cannot rise until we identify the missing prefill overhead

**β₁ = 1.8016** (decode memory-bound MFU, up 73.8% from 1.0372 in iter3)
- **Physical interpretation**: 1.80× theoretical memory-bound time (80% slower)
- **Trend**: Exploded from 1.037 → 1.802 when β₆ was added
- **Problem**: Physically implausible — decode cannot be 80% slower than memory bandwidth allows
- **Conclusion**: β₁ is absorbing error from misspecified β₆ term

**β₂ = 1.3598** (TP decode communication, up 328% from 0.3185 in iter3)
- **Physical interpretation**: 1.36× theoretical all-reduce time (36% slower)
- **Trend**: Exploded from 0.318 → 1.360 when β₆ was added (was iter3's β₃)
- **Problem**: 4.3× increase is physically implausible for NVLink communication
- **Conclusion**: β₂ is absorbing error from misspecified β₆ term

**β₃ = 0.0004953** (KV cache management, up 20.7% from 0.0004102 in iter3)
- **Physical interpretation**: 0.495 ms per request for block allocation/deallocation
- **Trend**: Stable across iterations (±20% is reasonable noise)
- **Conclusion**: β₃ is correctly modeling KV cache overhead

**β₄ = 0.9182** (decode compute-bound MFU, up 15.3% from 0.7963 in iter3)
- **Physical interpretation**: 91.8% of theoretical compute time (8% overhead)
- **Trend**: Increased from 0.796 → 0.918 when β₆ was added (was iter3's β₅)
- **Problem**: Should have decreased if β₆ was absorbing prefill overhead, but increased instead
- **Conclusion**: β₄ is also absorbing error from misspecified β₆

**β₅ = 0.0304** (MoE gating overhead, up 160% from 0.0117 in iter3)
- **Physical interpretation**: 30.4 ms per step for expert routing (was 11.7 ms)
- **Trend**: Exploded from 0.0117 → 0.0304 when β₆ was added (was iter3's β₆)
- **Problem**: 2.6× increase suggests absorbing error correlated with MoE experiments
- **Conclusion**: β₅ is absorbing error from misspecified β₆

**β₆ = 1.8177** (NEW: activation write bandwidth, expected 3.0-6.0)
- **Physical interpretation**: 1.82× theoretical activation write time
- **Trend**: Converged to 1.818, well below expected 3.0-6.0 range
- **Problem**: Lower-than-expected coefficient + 0% improvement in reasoning experiments + destabilized other coefficients
- **Conclusion**: β₆ formula is wrong — either capturing wrong overhead or using wrong scale factor

**Redundant terms**: None identified (all β coefficients > 0.0001)
- β₃ (KV mgmt) is smallest at 0.000495, but this is physically meaningful (0.5 ms per request)
- No candidates for removal in iter5

**Missing physics**:
1. **Kernel launch overhead**: Reasoning experiments need fixed per-operation cost, not another memory bandwidth term
2. **Per-layer-type coefficients**: Scout experiments need separate β₀_dense and β₀_moe
3. **Long-context attention switch**: Reasoning may need separate β₀ for context > 8K tokens

---

## Recommendations for iter5

### Priority 1: Critical Issues (address confirmed rejections)

**1.1 Remove β₆ (activation bandwidth) — IMMEDIATE**
- **Rationale**: β₆ is misspecified and destabilizing other coefficients (β₁ +73.8%, β₂ +328%, β₅ +160%)
- **Evidence**: Reasoning improved 0% (should improve >25%), β₆ converged to 1.818 (expected 3.0-6.0)
- **Action**: Revert to 7 beta terms (β₀-β₅) for iter5, removing activation bandwidth entirely
- **Expected impact**: Coefficients should stabilize back to iter3 ranges (β₁ ≈ 1.0, β₂ ≈ 0.3, β₅ ≈ 0.01)

**1.2 Profile reasoning experiments to identify actual bottleneck — CRITICAL**
- **Rationale**: 99.98-99.99% TTFT error (1000× underestimation) cannot be memory bandwidth
- **Action**: Run `nsys profile` on vLLM reasoning experiments (Llama-2-7B, Qwen2.5-7B, 8K-16K tokens)
- **Measure**:
  - Kernel launch count and timing (hypothesis: 50μs × 200 kernels × 80 layers)
  - Attention kernel type for long contexts (hypothesis: algorithmic switch at >8K tokens)
  - Scheduler queue depth and batch formation latency
  - Memory allocator overhead (prefix cache, KV cache block swapping)
- **Expected outcome**: One of four bottlenecks will dominate:
  1. Kernel launch: 800-1000ms from 50μs × 200 × 80
  2. Attention algorithm: Different kernel for long contexts
  3. Scheduler batching: Reasoning requests batched differently
  4. KV cache preemption: Swapping blocks to CPU for long contexts

**1.3 Investigate coefficient explosion when β₆ added — ANALYSIS**
- **Rationale**: β₁ +73.8%, β₂ +328%, β₅ +160% when β₆ added suggests collinearity or formula error
- **Action**: Check for collinearity between β₆ and existing terms (correlation matrix of basis functions)
- **Hypothesis**: β₆ formula overlaps with β₀ (both scale with tokens × layers), creating gradient masking
- **Expected outcome**: High correlation (>0.7) between β₆ and β₀ basis functions

### Priority 2: Improvements (address partial confirmations)

**2.1 Continue simplification strategy — VALIDATED**
- **Rationale**: Iter2→iter3→iter4 all improved by 3-4% from removing ineffective terms
- **Action**: After removing β₆, check if any remaining terms have coefficients near zero
- **Candidates**: β₃ (KV mgmt) = 0.000495 is very small, but physically meaningful (0.5 ms per request)
- **Expected impact**: 6 beta terms (β₀-β₅) should suffice for iter5

**2.2 Expand β₀ bounds to allow higher MFU — EXPLORATION**
- **Rationale**: β₀ = 0.165 is far below physical plausibility (ideal 0.40-0.55)
- **Action**: Widen β₀ bounds from [0.15, 0.55] to [0.15, 0.70] to allow optimizer more freedom
- **Caveat**: Only expand bounds if we've identified and added the missing prefill overhead term
- **Expected impact**: With correct prefill term, β₀ should rise to 0.30-0.50 range

**2.3 Check for TP-dependent prefill overhead (not communication) — HYPOTHESIS**
- **Rationale**: Mistral TP=2 (76.90% TTFT) and Llama-3.1 TP=4 (29.33% TTFT) have moderate prefill error
- **Action**: Review TP=2/TP=4 experiments to see if prefill error correlates with TP degree
- **Hypothesis**: TP increases prefill overhead via kernel launch or scheduler batching, not all-reduce
- **Expected outcome**: If TP-dependent, add β_tp_prefill = TP × num_layers × overhead

### Priority 3: Refinements (minor tweaks)

**3.1 Scout experiments: Accept 150-200% error or add per-layer-type coefficients — DECISION**
- **Rationale**: Scout's interleaved MoE+dense architecture cannot be modeled with single β₀
- **Option A**: Accept Scout error during training, use end-to-end validation for Scout models
  - Pros: Simple, no backend changes
  - Cons: 4 experiments (25% of training set) have catastrophic error
- **Option B**: Add per-layer-type coefficients (β₀_dense, β₀_moe)
  - Pros: Correctly models interleaved architectures
  - Cons: Backend changes, +1 parameter (6 beta → 7 beta)
- **Recommendation**: Choose Option A for iter5 (accept Scout error), revisit Option B if Scout models become more common

**3.2 Warm-start iter5 from iter3, not iter4 — STABILITY**
- **Rationale**: Iter4 coefficients drifted due to misspecified β₆; iter3 was more stable
- **Action**: Use iter3's best_params as initial values for iter5 (revert β₁/β₂/β₅ to iter3 ranges)
- **Expected impact**: Faster convergence, more stable coefficients

---

## Basis Function Changes for Iter5

**Remove**:
- β₆ (activation write bandwidth) — Wrong formula, destabilizes other coefficients

**Add** (after profiling reasoning experiments):
- **If kernel launch dominates**: β_kernel = num_layers × num_kernels_per_layer × fixed_overhead_us
- **If attention algorithm switches**: β₀_short (context ≤ 8K) and β₀_long (context > 8K)
- **If TP-dependent prefill**: β_tp_prefill = TP × num_layers × overhead

**Keep**:
- β₀: Prefill compute MFU
- β₁: Decode memory-bound MFU
- β₂: TP decode communication
- β₃: KV cache management
- β₄: Decode compute-bound MFU
- β₅: MoE gating overhead

**Total parameters**: 9 (3 alpha + 6 beta) for iter5, assuming one new term from profiling

---

## Bounds Adjustments for Iter5

**Revert to iter3 warm-start values** (before β₆ destabilization):
- β₀: [0.15, 0.55], initial 0.169 (from iter3)
- β₁: [0.8, 2.0], initial 1.037 (from iter3)
- β₂: [0.0, 1.5], initial 0.318 (from iter3)
- β₃: [0.0, 0.01], initial 0.00041 (from iter3)
- β₄: [0.4, 1.5], initial 0.796 (from iter3)
- β₅: [0.0, 0.05], initial 0.0117 (from iter3)

**Add new term bounds** (after profiling identifies bottleneck):
- If kernel launch: β_kernel: [0.0, 200.0] μs, initial 50.0 μs
- If long-context: β₀_long: [0.10, 0.50], initial 0.15
- If TP prefill: β_tp_prefill: [0.0, 100.0] μs, initial 20.0 μs

**Rationale**: Iter3 coefficients were stable and physically plausible. Iter4's coefficient drift came from misspecified β₆, so reverting to iter3 baseline is safer than warm-starting from iter4.

---

## Cross-Validation Decision

**Criteria for CV**:
- ✅ All hypotheses confirmed (every hypothesis ✅ verdict)
- Overall loss < 80% (ideally < 50%)
- No experiment with TTFT or E2E APE > 100%
- Coefficients physically plausible (no bounds violations)

**Iter4 Status**:
- ❌ 4 out of 5 hypotheses REJECTED, 1 PARTIAL
- ❌ Overall loss = 129.20% (far above 80% threshold)
- ❌ 6 experiments with TTFT > 80% (reasoning + Scout)
- ❌ Coefficients not physically plausible (β₁ = 1.802, β₂ = 1.360)

**Decision**: **DO NOT proceed to CV**. Iter5 required to address critical issues (remove β₆, profile reasoning, fix coefficient drift).

**Expected iter5 outcome**: With β₆ removed and correct prefill term added, expect:
- Overall loss: 120-130% → 100-110% (if kernel launch is correct)
- TTFT RMSE: 66.49% → 50-55%
- Reasoning TTFT: 99.98% → 70-85% (if profiling identifies bottleneck)
- Coefficients: β₁ → 1.0-1.1, β₂ → 0.3-0.4, β₅ → 0.01-0.012 (revert to iter3 stability)

---

## Lessons Learned

**What worked**:
1. **Simplification strategy**: Removing ineffective terms (β₂/β₇) improved loss by 3.93%, continuing the pattern from iter3
2. **Diagnostic clauses**: Agent 1's diagnostic clauses correctly identified failure modes (β₀ < 0.22 → activation BW wrong)
3. **Warm-starting from previous iteration**: Convergence to 185 trials despite harder search space

**What didn't work**:
1. **Activation bandwidth hypothesis**: β₆ formula was wrong, caused coefficient drift, 0% improvement in reasoning
2. **Formula without profiling**: Adding terms without profiling real vLLM behavior leads to misspecified basis functions
3. **Collinearity risk**: β₆ overlapped with β₀ (both scale with tokens × layers), causing gradient masking

**Key insight for iter5**:
- **Profile before hypothesizing**: Don't add new terms based on physics intuition alone — profile vLLM to identify actual bottlenecks
- **Test for collinearity**: Before adding new term, check correlation with existing basis functions (correlation matrix)
- **Use diagnostic clauses**: They worked! β₀ < 0.22 correctly triggered "activation BW is wrong" diagnostic

**Strategy Evolution validation**:
- **Prediction errors are valuable**: The H-main rejection revealed that activation BW is NOT the bottleneck, guiding iter5 toward kernel launch or attention algorithm
- **Causal mechanisms must be testable**: H-boundary's prediction (long prompts improve >25%) provided clear test that refuted the hypothesis
- **Diagnostic clauses guide next iteration**: Each diagnostic clause pointed to specific alternatives (kernel launch, O(n²) attention, KV preemption)

**Next iteration (iter5) must**:
1. Remove β₆ (activation bandwidth) immediately
2. Profile reasoning experiments with `nsys` to identify actual bottleneck
3. Add correct prefill overhead term based on profiling evidence
4. Warm-start from iter3 (not iter4) to avoid coefficient drift
5. Test for collinearity before adding new terms
