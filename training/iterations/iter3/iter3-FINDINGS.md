# Iteration 3: Findings and Principles

## Summary

Iter3 achieved modest improvement (3.06% loss reduction: 136.19% → 133.13%) through **simplification** (removing ineffective β₇, β₈), but the new β₇ (TP-dependent prefill communication) **failed to activate** (coefficient ≈ 0). This reveals that TP prefill overhead is either: (1) not communication-dominated, (2) has wrong functional form, or (3) already captured by existing terms. The iteration validated that systematic failures (reasoning at ~100% TTFT, Scout at 174-197%) cannot be solved by adding formulas without understanding root causes.

**What worked:**
- Simplification reduced parameter bloat (12 → 10 params), improving optimization quality
- β₁ normalized significantly (1.553 → 1.037), moving closer to physical expectation

**What didn't work:**
- New β₇ (TP prefill comm) rejected by optimizer (coefficient ≈ 0)
- β₀ (prefill MFU) worsened (0.203 → 0.169), moving further from physical plausibility
- Reasoning and Scout failures persist unchanged

**What we learned:**
- Adding terms without validating assumptions wastes optimization budget
- Removing ineffective terms helps remaining coefficients converge better
- TP prefill overhead requires investigation (profiling), not speculation

---

## Error Analysis

### Systematic Patterns

**High-error experiments** (TTFT > 100%):
- **All 4 reasoning experiments**: 99.97-99.99% TTFT (model predicts near-zero prefill time)
  - Qwen2.5-7B reasoning-1-1: 99.99% TTFT, 96.22% E2E
  - Scout reasoning-2: 99.99% TTFT, 94.85% E2E
  - Llama-2-7B reasoning: 99.98% TTFT, 93.69% E2E
  - **Pattern**: Massive prefill underestimation (predicted ~1ms, actual ~1000ms), but decode is closer (~90-96% E2E error vs ~100% TTFT error)
  - **Why**: Long prompts (4K-16K tokens) with quadratic attention memory bandwidth, KV cache write bursts, or preemption overhead not captured by model

- **All 4 Scout experiments** (TTFT): 89.52-99.98% TTFT, but E2E varies (84.54-97.43%)
  - Scout general-2: 99.98% TTFT, 97.43% E2E
  - Scout reasoning-2: 99.99% TTFT, 94.85% E2E
  - Scout codegen-2: 93.85% TTFT, 90.50% E2E
  - Scout roleplay-2: 89.52% TTFT, 84.54% E2E
  - **Pattern**: Interleaved MoE+dense architecture not captured by single β₀ (prefill MFU)
  - **Why**: MoE layers have lower MFU (routing overhead, expert load imbalance) than dense layers, but model uses single β₀ for all layers

**Medium-error experiments** (TTFT 50-90%):
- Mistral TP=2 general-lite: 79.61% TTFT, 66.27% E2E
- Llama-3.1-70B TP=4 general-lite: 70.90% TTFT, 5.17% E2E (E2E excellent!)
- **Pattern**: TP=2 and TP=4 have asymmetric errors — high TTFT but low-to-medium E2E
- **Why**: Prefill TP overhead underestimated (β₇ = 0 didn't help), but decode TP overhead captured by β₃

**Low-error experiments** (TTFT < 20%):
- Llama-3.1-70B TP=4 codegen: 3.86% TTFT, 32.49% E2E (TTFT excellent!)
- Yi-34B TP=2: 14.69% TTFT, 2.55% E2E (both excellent!)
- Llama-2-7B roleplay: 13.07% TTFT, 22.71% E2E
- Qwen2.5-7B roleplay: 36.16% TTFT, 2.14% E2E (E2E excellent!)
- Mistral codegen: 9.44% TTFT, 7.38% E2E (both excellent!)
- **Pattern**: TP=1/TP=2 with short prompts and balanced prefill/decode mix
- **What makes these easy**: Batch composition matches model's sweet spot (moderate prompt lengths, good prefill/decode balance, no extreme TP or MoE overhead)

**Error correlations**:
- ✅ **Confirmed**: Long prompts (>4K tokens) correlate with high TTFT error (reasoning: ~100%)
- ✅ **Confirmed**: MoE architecture correlates with high errors (Scout: 174-197% combined loss)
- ✅ **Confirmed**: TP=4 with long prompts correlates with high TTFT (70.90%) BUT low E2E (5.17%)
- ❌ **Rejected**: TP degree alone does NOT predict error (TP=4 codegen: 3.86% TTFT excellent, TP=4 general-lite: 70.90% TTFT high)
- ❌ **Rejected**: Workload type alone does NOT predict error (codegen ranges from 3.86% to 93.85% TTFT)

### Root Cause Hypotheses

**Principle 1: TP Prefill Overhead is Not Communication-Dominated (or Formula is Wrong)**
- **Evidence**:
  - β₇ (TP prefill comm) = 2.78e-07 ≈ 0 (optimizer rejected the term)
  - Formula: `β₇ × TP × num_layers × (prompt_tokens / 1000)`
  - Predicted coefficient: 5-15 μs per (TP × layer × K-token)
  - Actual coefficient: 0.0000002778 μs (rejected)
  - TP=4 experiments improved (18.9pp and 38.08pp) but NOT due to β₇
- **Mechanism**:
  - Agent 1 hypothesized that TP prefill is communication-dominated (all-reduce per layer per token)
  - Optimizer rejected this → either overhead is NOT communication-dominated OR formula is wrong
  - Alternative explanations:
    1. **Memory-bandwidth-bound**: Activation writes (residual connections, attention outputs) dominate, not communication
    2. **Compute-bound**: Prefill MFU is low (β₀=0.169), suggesting compute is bottleneck not communication
    3. **Wrong functional form**: Real TP overhead may be logarithmic log₂(TP) or constant + linear (C + k×TP)
    4. **Already captured**: Existing β₃ (TP communication) or β₀ (prefill MFU) already account for TP prefill overhead
- **Action**:
  - **Profile TP=4 prefill**: `nsys profile vllm --model llama-3.1-70b --tp 4` to identify actual bottleneck
  - **Validate formula**: If communication-bound, check all-reduce scaling, bandwidth saturation, network congestion
  - **Test alternative forms**: Try logarithmic TP scaling or constant + linear
  - **Ablation test**: Temporarily set β₀ and β₃ to zero, refit β₇ to check if collinearity is masking signal

---

**Principle 2: Removing Ineffective Terms Improves Optimization Quality**
- **Evidence**:
  - Removed β₇ (very long context) = 1.0 (at initial value in iter2)
  - Removed β₈ (per-request decode) = 3e-05 (at initial value in iter2)
  - Overall loss improved by 3.06% (136.19% → 133.13%)
  - β₁ normalized significantly (1.553 → 1.037) after removing β₈
  - Optimization converged at trial 84 (vs iter2: trial 51, but with better loss)
  - Reasoning/Scout experiments unchanged (confirming β₇/β₈ were ineffective)
- **Mechanism**:
  - Terms that don't move during optimization provide no gradient signal but increase parameter space
  - Bayesian optimization samples density improves in lower-dimensional space (10D vs 12D)
  - Removing dead weight allows optimizer to focus on effective terms
  - Parameter density improved: 0.67 params/experiment (vs 0.80 in iter2), reducing overfitting risk
- **Action**:
  - **Monitor coefficient convergence**: Track which terms don't move from initial values (candidates for removal)
  - **Set removal threshold**: If |β_i - β_i_initial| < 0.05 × |β_i_initial| after 50+ trials → remove in next iteration
  - **Validate removal**: Check that experiments don't degrade >5% after removal (as H-simplification verified)

---

**Principle 3: Systematic Failures Require Investigation, Not Formulas**
- **Evidence**:
  - Reasoning experiments: Stayed at 99.97-99.99% TTFT (unchanged from iter2)
  - Scout experiments: Stayed at 174-197% combined loss (unchanged from iter2)
  - Adding β₇ (targeting very long context) did NOT help reasoning (stayed at 100% TTFT)
  - Adding β₈ (targeting Scout decode) did NOT help Scout (stayed at 174-197%)
  - Removing β₇ and β₈ did NOT degrade these experiments (confirming they were ineffective)
- **Mechanism**:
  - Adding formulas based on speculation (without profiling or validation tests) wastes optimization budget
  - Reasoning ~100% TTFT suggests massive prefill underestimation (predicted ~1ms, actual ~1000ms)
  - Possible root causes: Quadratic attention memory bandwidth O(n²) for n>4K, KV cache preemption overhead, prefix cache miss rate
  - Scout 174-197% suggests interleaved MoE+dense architecture not captured by single β₀
  - Possible root cause: MoE layers have lower MFU (routing overhead) than dense layers, but model uses one β₀ for all layers
- **Action**:
  - **For reasoning**: Profile vLLM reasoning experiments to identify actual bottleneck (memory? attention? KV cache?)
  - **For Scout**: Create end-to-end latency validation test (not just FLOPs test) that catches MoE-specific overhead
  - **For Scout**: Consider per-layer-type basis functions (β₀_dense for dense layers, β₀_moe for MoE layers)
  - **General principle**: Do NOT add terms without validating the assumption via profiling or ablation tests

---

**Principle 4: β₀ (Prefill MFU) Remains Far Below Physical Plausibility**
- **Evidence**:
  - β₀ = 0.169 (iter3) vs 0.203 (iter2) — **worsened** by 0.034
  - Physical expectation: 0.40-0.55 (40-55% MFU for prefill)
  - Actual: 0.169 (16.9% MFU) — 2.4× to 3.3× too low
  - β₇ (new TP prefill comm) = 0 did NOT absorb overhead as predicted
  - All 4 reasoning experiments have ~100% TTFT (massive prefill underestimation)
- **Mechanism**:
  - Prefill time formula: `prefill_flops / (β₀ × gpu_peak_flops)`
  - With β₀ = 0.169, model assumes prefill achieves only 16.9% MFU
  - This is too low even for memory-bound workloads (typical: 30-40% MFU)
  - Missing overhead terms are artificially lowering β₀ to compensate
  - Agent 1's hypothesis (TP prefill comm) was one candidate, but β₇ = 0 rejected it
- **Action**:
  - **Identify missing prefill term**: Profile prefill to find dominant overhead (memory bandwidth? scheduler? KV write bursts?)
  - **Candidate terms** (from Agent 1's diagnostic):
    1. **Activation memory bandwidth**: Residual connections, attention outputs, layer norm writes
    2. **Large-model-specific scheduler overhead**: Batch preparation, memory allocation for 70B models
    3. **KV cache write bursts**: Writing large KV cache during prefill (4K-16K tokens × 80 layers × 8192 hidden_dim)
  - **Test approach**: Add one candidate term at a time, check if β₀ rises toward 0.30-0.45
  - **Alternative**: Use separate prefill/decode MFU (β₀_prefill, β₀_decode) if overhead is regime-specific

---

**Principle 5: β₁ Normalization via Simplification (Unexpected Success)**
- **Evidence**:
  - β₁ = 1.037 (iter3) vs 1.553 (iter2) — **improved** by 0.516 (closer to ideal 1.0)
  - Physical expectation: ~1.0 for memory-bound decode (100% efficiency)
  - β₁ > 1.0 in iter2 indicated either: (a) decode FLOPs undercounted, or (b) discrete regime split wrong
  - Simplification (removing β₈) allowed β₁ to normalize
- **Mechanism**:
  - Agent 1 predicted β₁ would "stay ~1.5-1.6" because β₈ was negligible
  - Actual result: β₁ improved significantly (1.553 → 1.037)
  - Why: Removing β₈ eliminated a red herring term, allowing optimizer to fit β₁ more accurately
  - With fewer parameters (10 vs 12), optimizer had better sample density in remaining dimensions
  - β₁ = 1.037 is still slightly high (ideal ~1.0), but much closer than 1.553
- **Action**:
  - **Continue monitoring β₁**: If it rises again in iter4, investigate decode FLOPs formula or regime split
  - **Hypothesis**: β₁ ≈ 1.0 may be correct if decode is truly memory-bound (bandwidth-limited)
  - **Validate**: Compare predicted decode time to actual vLLM decode step time for memory-bound cases

---

**Principle 6: TP=4 Improvements Came from β₁ Normalization, Not β₇**
- **Evidence**:
  - TP=4 experiments improved by 18.9pp (general-lite) and 38.08pp (codegen) on average
  - β₇ (TP prefill comm) = 2.78e-07 ≈ 0 (did NOT contribute)
  - β₁ decreased from 1.553 → 1.037 (improved decode predictions)
  - TP=4 E2E errors are low (5.17% and 32.49%), but TTFT errors mixed (70.90% and 3.86%)
- **Mechanism**:
  - TP=4 experiments have large models (70B) with long decode sequences
  - β₁ normalization improved E2E predictions (decode-heavy), which propagated to TTFT via E2E-TTFT coupling
  - One TP=4 experiment (codegen) achieved excellent TTFT (3.86%), suggesting batch composition matters more than TP degree alone
  - Other TP=4 experiment (general-lite) still high TTFT (70.90%), suggesting missing prefill term (same as reasoning)
- **Action**:
  - **Investigate TP=4 general-lite**: Why TTFT 70.90% when codegen is 3.86%? Batch composition? Prompt length distribution?
  - **Hypothesis**: Codegen has shorter prompts or better prefill/decode balance, reducing impact of missing β₀ overhead
  - **Test**: Compare prompt length distribution between codegen (TTFT 3.86%) and general-lite (TTFT 70.90%)

---

## Coefficient Analysis

**Alpha [α₀, α₁, α₂]** from `best_params.alpha`: [Fixed API overhead, per-input-token, per-output-token]
- Optimal values: α₀ = 0.00111 (1.11ms), α₁ = 5.27e-05 (52.7μs/token), α₂ = 8.49e-05 (84.9μs/token)
- Physical interpretation:
  - α₀ = 1.11ms: Fixed API overhead per request (plausible for vLLM request processing)
  - α₁ = 52.7μs/token: Input token overhead (tokenization, KV cache allocation) — reasonable
  - α₂ = 84.9μs/token: Output token overhead (detokenization, response streaming) — reasonable
- Outliers: None, all within plausible ranges
- **Change from iter2**: α₀ 1.16ms → 1.11ms, α₁ 42.5μs → 52.7μs, α₂ 95.7μs → 84.9μs (minor adjustments)

**Beta [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇]** from `best_params.beta`: [Step-level basis functions]
- **β₀ = 0.169** (Prefill MFU): **Too low** (ideal: 0.40-0.55). Indicates missing prefill overhead term. **Action**: Identify missing term via profiling.
- **β₁ = 1.037** (Decode memory-bound MFU): **Improved** from 1.553 (iter2). Now close to ideal ~1.0. **Action**: Monitor in iter4, validate against vLLM decode step times.
- **β₂ = 9.97e-05** (Scheduler overhead per request): **Negligible** (~0.1μs per request). Either genuinely small or captured elsewhere (β₀ or β₁). **Action**: Consider removing in iter4 if stays near zero.
- **β₃ = 0.318** (TP communication overhead): **Decreased** from 0.394 (iter2). Captures decode TP overhead (all-reduce per layer per decode step). **Action**: Stable, no changes needed.
- **β₄ = 0.00041** (Batch formation overhead): **Stable** (~0.41μs per prefill token). Captures cost of forming batches. **Action**: No changes needed.
- **β₅ = 0.796** (KV cache write overhead): **Increased** from 0.651 (iter2). May be absorbing missing prefill overhead. **Action**: Monitor if continues rising in iter4.
- **β₆ = 0.0117** (Attention overhead): **Increased** from 0.008 (iter2). May be absorbing missing prefill overhead. **Action**: Monitor if continues rising in iter4.
- **β₇ = 2.78e-07** (TP prefill communication): **Rejected** by optimizer (≈ 0). **Action**: Remove in iter4 (redundant term).

**Redundant terms**: β₂ ≈ 0 (scheduler overhead) and β₇ ≈ 0 (TP prefill comm) are candidates for removal in iter4.

**Missing physics**: β₀ = 0.169 far below physical expectation suggests major missing prefill term (activation memory bandwidth? KV cache write bursts? large-model scheduler overhead?). β₅ and β₆ increasing may indicate they're absorbing this missing overhead.

---

## Recommendations for iter4

### Priority 1: Critical Issues

**1. Investigate Reasoning Failures (100% TTFT underestimation)**
- **Issue**: All 4 reasoning experiments predict near-zero prefill time (actual ~1000ms, predicted ~1ms)
- **Root cause unknown**: Quadratic attention? KV cache preemption? Prefix cache misses?
- **Action**: Profile vLLM reasoning experiments with long prompts (8K-16K tokens):
  ```bash
  nsys profile vllm serve --model qwen2.5-7b --max-model-len 16384 < reasoning_requests.jsonl
  ```
  - Identify dominant overhead (memory bandwidth? attention kernel? scheduler?)
  - Measure actual vs expected prefill time for 8K-16K token prompts
  - Check if overhead scales linearly (O(n)) or quadratically (O(n²)) with prompt length
- **Do NOT add formulas** until root cause is validated

**2. Add End-to-End Validation Test for Scout (MoE Architecture)**
- **Issue**: Scout experiments consistently fail (174-197% combined loss) despite FLOPs tests passing
- **Root cause hypothesis**: Single β₀ (prefill MFU) cannot represent MoE (lower) vs dense layers (higher)
- **Action**:
  - Create end-to-end latency validation test (compare predicted vs actual vLLM latency for Scout)
  - If test fails, consider per-layer-type basis functions: β₀_dense (dense layers) and β₀_moe (MoE layers)
  - Measure actual Scout prefill time vs dense model of similar size to quantify MoE overhead

**3. Identify Missing Prefill Overhead Term (β₀ = 0.169 << 0.40-0.55)**
- **Issue**: Prefill MFU far below physical plausibility, reasoning experiments at 100% TTFT
- **Candidates** (from Agent 1's diagnostic + new analysis):
  1. **Activation memory bandwidth**: Residual connections, attention outputs, layer norm writes
  2. **KV cache write overhead**: Already have β₅, but may need prefill-specific variant (β₅ currently captures decode KV write)
  3. **Large-model scheduler overhead**: 70B models may have different scheduler cost than 7B models
  4. **Long-context attention memory bandwidth**: O(n²) scaling for n>4K tokens
- **Action**: Profile prefill to identify dominant overhead, add one candidate term at a time

### Priority 2: Improvements

**4. Remove Redundant Terms (β₂ ≈ 0, β₇ ≈ 0)**
- **Issue**: Two terms with coefficients near zero provide no predictive value
- **Action**: Remove β₂ (scheduler overhead) and β₇ (TP prefill comm) in iter4
- **Expected impact**: Reduce parameters from 10 to 8, improve optimization quality (following Principle 2)
- **Validation**: Ensure removal doesn't degrade any experiments by >5% (as H-simplification validated)

**5. Investigate TP=4 Asymmetry (High TTFT, Low E2E)**
- **Issue**: TP=4 general-lite has 70.90% TTFT but 5.17% E2E (decode excellent, prefill poor)
- **Why interesting**: TP=4 codegen has 3.86% TTFT (excellent), suggesting batch composition matters
- **Action**: Compare prompt length distribution and prefill/decode ratio between:
  - TP=4 codegen: TTFT 3.86% (excellent)
  - TP=4 general-lite: TTFT 70.90% (poor)
- **Hypothesis**: Codegen has shorter prompts → less impact from missing β₀ overhead

**6. Monitor β₅ and β₆ Drift (May be Absorbing Missing Prefill Overhead)**
- **Issue**: β₅ increased 0.651 → 0.796, β₆ increased 0.008 → 0.0117 (both rising)
- **Hypothesis**: With β₀ too low and missing prefill term, optimizer is pushing overhead into β₅ (KV write) and β₆ (attention)
- **Action**: After adding missing prefill term in iter4, check if β₅ and β₆ decrease back to iter2 levels
- **Expected**: If hypothesis correct, β₅ and β₆ should decrease when proper prefill term is added

### Priority 3: Refinements

**7. Validate β₁ ≈ 1.0 Physical Interpretation**
- **Issue**: β₁ = 1.037 is close to ideal ~1.0 but still slightly high
- **Action**: Compare predicted decode time to actual vLLM decode step time for memory-bound cases
- **If β₁ ≈ 1.0 is correct**: Decode is truly memory-bound (bandwidth-limited), no changes needed
- **If β₁ should be <1.0**: Decode FLOPs formula may overcount operations or regime split may be wrong

**8. Test Alternative TP Scaling Functional Forms**
- **Issue**: Linear TP scaling (β₇ × TP × layers × tokens) was rejected (β₇ = 0)
- **Alternative forms to test**:
  1. Logarithmic: `β_tp_prefill × log₂(TP) × layers × tokens`
  2. Constant + linear: `(β_c + β_k × TP) × layers × tokens`
  3. Ring all-reduce: `β_tp_prefill × (TP - 1) / TP × layers × tokens`
- **Action**: Add one alternative form in iter4, check if optimizer accepts it (coefficient > 0)
- **Note**: Only test if profiling confirms TP prefill is communication-dominated (Priority 1 action 1)

### Basis Function Changes

**Add**:
1. **Missing prefill overhead term** (Priority 1, issue 3): Activation memory bandwidth OR long-context attention overhead
2. **Per-layer-type MFU** (Priority 1, issue 2): β₀_dense and β₀_moe for Scout (if end-to-end test confirms need)

**Remove**:
1. **β₂ (scheduler overhead)**: Coefficient ≈ 0, redundant (Priority 2, issue 4)
2. **β₇ (TP prefill comm)**: Coefficient ≈ 0, rejected by optimizer (Priority 2, issue 4)

**Modify**:
1. **None**: Wait for profiling results before modifying existing terms

### Bounds Adjustments

**Current bounds** (from `coefficient_bounds.yaml`):
- β₀: [0.15, 0.55] — β₀ = 0.169 near lower bound, consider expanding downward if missing term not found
- β₁: [0.8, 2.0] — β₁ = 1.037 within range, bounds OK
- β₇: [0.0, 50.0] — β₇ = 2.78e-07 rejected, remove term in iter4

**Adjustments for iter4**:
- **Keep β₀ bounds [0.15, 0.55]** until missing prefill term is added (then check if β₀ rises toward 0.40)
- **Keep β₁ bounds [0.8, 2.0]** (β₁ = 1.037 is stable and plausible)
- **No other adjustments needed** (all other coefficients within plausible ranges)

---

## Expected Outcome for Iter4

Based on findings from iter3:

**If Priority 1 actions succeed** (identify missing prefill term, validate Scout end-to-end):
- Overall loss: 100-120% (improvement from 133%)
- TTFT RMSE: 50-60% (improvement from 71%)
- Reasoning experiments: 50-80% TTFT (down from 100%)
- Scout experiments: 120-150% combined loss (down from 174-197%)
- β₀ rises to 0.30-0.45 (closer to physical plausibility)

**If Priority 1 actions fail** (profiling reveals unexpected root cause):
- May need multiple iterations to converge
- Consider alternative approaches: per-request-type models, regime-specific MFU, or hybrid physics+blackbox

**If only Priority 2 actions succeed** (remove redundant terms, investigate TP=4):
- Overall loss: 125-135% (minor improvement from 133%)
- Simplification helps optimization quality but doesn't solve systematic failures

**Critical success factor**: **Do NOT add formulas without validating assumptions.** Iter3 validated this principle (β₇ rejected because assumption was wrong). Iter4 must profile before proposing terms.
