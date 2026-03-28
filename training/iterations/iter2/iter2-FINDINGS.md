# Iteration 2: Findings and Principles

## Summary

Iteration 2 **degraded** performance by 12%, achieving 150.20% overall loss vs iter1's 134.54%. The very long context (β₇) + per-request decode overhead (β₈) hypothesis failed completely:

- **β₇ ineffective**: Large coefficient (1.507) but reasoning experiments still at 99% TTFT APE
- **β₈ negligible**: Converged to 0.000042 (effectively zero)
- **β₀ degraded**: Prefill efficiency dropped from 0.203 to 0.155
- **Scout MoE validation failure**: 4 experiments at 100% APE contribute 53% of total loss

**Key insight**: Testing TWO independent mechanisms simultaneously (β₇ + β₈) made it impossible to isolate which failed. Both failed, suggesting the additive overhead model structure needs fundamental rethinking.

---

## Error Analysis

### Systematic Patterns

**High-error experiments** (TTFT or E2E APE > 80%):

1. **Scout MoE experiments** (4): All 100% APE
   - **Pattern**: All TP=2, all Scout 17B-16E MoE model
   - **Root cause**: Validation failure (ground truth latencies incorrect or model name parsing broken)
   - **Contribution**: 800% to loss (53%)

2. **Reasoning experiments** (2): Qwen 99%, Llama-2 99% TTFT APE
   - **Pattern**: Longest prompts in dataset (expected >4096 tokens)
   - **Root cause**: β₇ mechanism is WRONG - overhead not proportional to `(tokens - 4096)`
   - **Contribution**: ~390% to loss (26%)

3. **Mistral-Nemo TP=2 general-lite** (1): 83% TTFT, 90% E2E
   - **Pattern**: Medium-length prompts, TP=2
   - **Root cause**: Unknown - not long-context, not Scout MoE
   - **Contribution**: 174% to loss (12%)

**Combined**: These 7 experiments (47% of dataset) contribute 1364% to loss (91%).

**Low-error experiments** (TTFT APE < 10%):

1. **Llama-2 roleplay** (TP=1): 5.7% TTFT, 60.4% E2E
2. **Qwen2.5 roleplay** (TP=1): 6.8% TTFT, 48.3% E2E
3. **Llama-2 codegen** (TP=1): 7.7% TTFT, 50.7% E2E
4. **Yi-34B general-lite** (TP=2): 10.3% TTFT, 70.7% E2E

**Pattern**: All have good TTFT prediction (<11%) but moderate E2E error (48-71%). This suggests:
- Prefill model is correct for **short-to-medium contexts**
- Decode model has systematic bias (E2E error 6-10× higher than TTFT error)

**Error correlations**:

✅ **Confirmed correlations** (features associated with low error):
- **TP=1 non-reasoning experiments**: Mean 20.6% TTFT APE (excluding reasoning's 99%)
- **Short-medium contexts**: <4096 tokens have lower TTFT error
- **Small-medium models**: 7B-34B models perform better than 70B or MoE

❌ **Rejected correlations** (features that DON'T explain error):
- **Workload labels**: Low-error experiments span codegen, roleplay, general-lite (no systematic pattern)
- **Batch size**: Both small and large batch experiments have mixed error
- **TP degree alone**: TP=4 (mean 32% TTFT) not clearly better than TP=1 (mean 33% TTFT, excluding reasoning)

---

### Root Cause Hypotheses

#### Principle 1: Very Long Context Overhead Has Wrong Functional Form

**Evidence**:
- β₇ = 1.507 (from `best_params.beta[7]`) - large, non-negligible coefficient
- Reasoning experiments STILL at 99% TTFT APE (from `per_experiment_results[4,5]`)
- Expected: β₇ would reduce reasoning TTFT from ~100% to <50%
- Actual: No improvement despite large coefficient

**Mechanism**:

Agent 1's hypothesis assumed linear scaling: `β₇ × max(0, prompt_tokens - 4096) / 1000 × num_layers`

This implies overhead = constant × (tokens above threshold) × depth, suggesting prefill overhead grows **linearly** beyond 4096 tokens.

**However**, vLLM's attention kernels (FlashAttention/FlashInfer) have non-linear behavior:
- **Below L2 cache threshold** (~2048-4096 tokens): Linear scaling, memory-bound
- **Exceeding L2 but within HBM**: Quadratic attention cost starts dominating
- **Triggering chunking**: vLLM splits into chunks, introducing per-chunk overhead (non-linear step function)
- **KV eviction**: Recomputation overhead is discrete (either 0 or 100% of chunk), not linear

The linear `(tokens - threshold)` feature cannot capture:
1. Quadratic scaling of attention beyond cache limits
2. Discrete chunking boundaries (e.g., every 8192 tokens)
3. KV cache eviction (depends on memory pressure, not just length)
4. Prefix cache miss rate (depends on cache state, not just length)

**Alternative functional forms to test**:
- **Quadratic**: `max(0, (prompt_tokens - 4096)²) / 1e6`
- **Piecewise linear**: Different slopes for 0-2048, 2048-4096, 4096-8192, >8192
- **Log scaling**: `log(1 + max(0, prompt_tokens - 4096))`
- **Chunking step function**: `floor(prompt_tokens / chunk_size) × chunk_overhead`

**Action for iter3**: Replace `max(0, tokens - 4096)` with quadratic or piecewise term. Test multiple thresholds (2048, 4096, 8192) in parallel.

---

#### Principle 2: Per-Request Decode Overhead is Negligible or Constant

**Evidence**:
- β₈ = 0.000042 (from `best_params.beta[8]`) - effectively zero
- β₁ = 1.316 (from `best_params.beta[1]`) - still inflated (expected 0.6-0.9)
- Expected: β₈ absorbs per-request overhead, normalizing β₁
- Actual: β₈ converged to zero, β₁ still inflated

**Mechanism**:

Agent 1 predicted ~10-50μs per-request overhead for:
1. Scheduler iteration (vLLM `_schedule_running()`)
2. FlashAttention metadata setup (KV block tables, query offsets)
3. Kernel launch overhead (especially for small batches)

The model was `β₈ × num_requests_in_batch`.

**However**, optimizer drove β₈ to zero, proving one of:
1. **Overhead is constant**: Same overhead regardless of request count (captured by α₀ or β₂)
2. **Overhead is negligible**: <1μs per request, below measurement noise
3. **Overhead scales differently**: Scales with total context length `Σ(context_length)`, not request count
4. **Already captured**: β₁ (memory-bound decode) already includes this overhead

**Counter-evidence for "already captured"**: β₁ = 1.316 is still 2× too high. If β₈ is zero, what's causing β₁ inflation?

**Alternative hypotheses**:
- **Per-token decode overhead**: `β × Σ(context_length_per_request)` instead of `β × num_requests`
- **Batch-size dependent**: Overhead varies with batch size (small batches have proportionally higher overhead)
- **TP-dependent**: Per-request overhead increases with TP degree (communication setup)
- **KV cache fragmentation**: Overhead depends on KV block allocation pattern, not request count

**Action for iter3**: Test per-token decode overhead `Σ(context_length)` instead of per-request. Investigate β₁ inflation root cause independently.

---

#### Principle 3: Removing Redundant Terms Can Degrade Model

**Evidence**:
- Iter1: β₅ (chunking) ablation showed +1.06% loss increase (deemed redundant)
- Iter2: Removed β₅, added β₇ + β₈
- Result: **Overall loss increased 12%** (iter1: 134.54% → iter2: 150.20%)
- β₀ degraded from 0.203 to 0.155 (23% drop in prefill efficiency)

**Mechanism**:

Agent 1 concluded β₅ was redundant because ablating it caused only +1.06% loss increase. However:

1. **Confounded with other terms**: β₅ may have been absorbing real overhead that other terms couldn't capture
2. **Coefficient redistribution**: Removing β₅ forced optimizer to redistribute its contribution to other terms, causing distortion
3. **Spurious correlation**: β₅ may have been fitting noise, but removing it exposed missing signal elsewhere

**Evidence β₅ may have been useful**:
- β₀ got WORSE after removing β₅ (0.203 → 0.155)
- E2E RMSE got WORSE after removing β₅ (65.24% → 80.90%)
- Iter1 ablation showed +1.06% loss, which is SMALL but NON-ZERO

**Why removing a "redundant" term can degrade performance**:
- **Overfitting to ablation test set**: Ablation may have tested on subset of experiments where β₅ was truly redundant
- **Optimization landscape change**: Removing a term changes the loss surface, potentially making it harder to find optimal coefficients for remaining terms
- **Interaction effects**: β₅ may have been interacting with other terms (e.g., reducing β₁ inflation)

**Action for iter3**:
1. **Do NOT remove terms based on small ablation deltas** - even +1% loss may indicate real signal
2. **Test ablations on FULL dataset**, not subset
3. **Check coefficient stability** - if removing one term causes large shifts in other coefficients, terms are confounded
4. **Consider keeping β₅** - restore chunking term in iter3 and test if it stabilizes β₀

---

#### Principle 4: Scout MoE Validation Failure is Critical Blocker

**Evidence**:
- 4 Scout MoE experiments: All 100% TTFT and E2E APE (from `per_experiment_results[0-3]`)
- Contribution: 800% to loss (53% of total)
- Model: RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic (MoE architecture)
- TP=2 for all Scout experiments

**Mechanism**:

Scout experiments have been failing since iter0 (originally >100% APE, now exactly 100%). This suggests:

1. **Ground truth corruption**: Observed latencies in Scout experiments may be incorrect
2. **Model name parsing failure**: MoE model parameters (num_experts, expert_size) may not be parsed correctly from model name
3. **FP8 quantization modeling**: FP8-dynamic quantization may not be correctly handled in weight bandwidth calculations
4. **MoE routing overhead**: Expert routing, load imbalance, and expert specialization overhead may be missing from model

**Why 100% APE specifically**:
- 100% APE means `|predicted - observed| / observed = 1.0`
- This implies `predicted ≈ 2 × observed` or `predicted ≈ 0`
- More likely: `predicted ≈ 0` (model predicting near-zero latency, indicating parsing failure)

**Action for iter3**:
1. **CRITICAL PRIORITY**: Debug Scout MoE validation BEFORE iterating model
2. **Check**: Print predicted vs observed latencies for Scout experiments
3. **Investigate**: Model name parsing for "Llama-4-Scout-17B-16E" format
4. **Verify**: FP8-dynamic quantization handling in `EffectiveWeightBytesPerParam()`
5. **Add**: MoE-specific basis function if routing overhead is missing

**Impact if not fixed**: Scout experiments contribute 53% of loss. Even if all other experiments achieve 20% APE, overall loss would be `(0.53 × 200 + 0.47 × 20) / 1.0 ≈ 115%`, failing <80% target.

---

#### Principle 5: Testing Multiple Hypotheses Simultaneously Masks Root Causes

**Evidence**:
- Iter2 added TWO new terms: β₇ (very long context) + β₈ (per-request decode)
- Changed decode regime transition (discrete → sigmoid)
- Removed one term: β₅ (chunking)
- Result: Both new terms ineffective, overall loss increased

**Mechanism**:

Scientific method requires testing **one variable at a time**. Iter2 violated this by:
1. Adding β₇ to fix reasoning experiments
2. Adding β₈ to normalize β₁
3. Removing β₅ based on iter1 ablation
4. Changing regime transition function

**Why this masks root causes**:
- **Confounded effects**: If loss increases, which change caused it? Can't isolate.
- **Interaction effects**: β₇ and β₈ may interact (e.g., both targeting decode overhead in different ways)
- **Optimization instability**: Adding multiple terms simultaneously can create local minima in loss surface

**Evidence of confounding**:
- β₁ = 1.316 (still inflated) - was β₈ supposed to fix this, or did removing β₅ cause it?
- β₀ = 0.155 (degraded) - did removing β₅ cause this, or did adding β₇ distort it?
- E2E RMSE increased 24% - was this β₈ failing, sigmoid transition, or β₅ removal?

**Correct approach** (single-variable testing):
1. **Iter2a**: Add ONLY β₇ (very long context), keep everything else from iter1
2. **Iter2b**: Add ONLY β₈ (per-request), keep everything else from iter1
3. **Iter2c**: Remove ONLY β₅ (chunking), keep everything else from iter1
4. **Iter2d**: Change ONLY regime transition (sigmoid), keep everything else

Then combine ONLY the changes that improved loss.

**Action for iter3**: Test ONE mechanism at a time. Do NOT add multiple basis functions in a single iteration.

---

#### Principle 6: Coefficient Magnitude Predicts Term Importance

**Evidence**:

| Term | Coefficient | Interpretation | Expected | Match? |
|------|-------------|----------------|----------|--------|
| β₀ | 0.155 | Prefill MFU | 0.4-0.5 | ❌ 3× too low |
| β₁ | 1.316 | Decode memory-bound MFU | 0.6-0.9 | ❌ 1.5× too high |
| β₂ | 0.000093 | Constant step overhead | ~0.0001 | ✅ |
| β₃ | 0.119 | TP communication | 0.05-0.15 | ✅ |
| β₄ | 0.000043 | KV mgmt overhead | 0.0003-0.0004 (iter1) | ❌ 10× too low |
| β₅ | 0.956 | Large-batch compute MFU | 0.6-0.8 | ⚠️ 1.3× high |
| β₆ | 0.135 | Memory mgmt overhead | ? | ? |
| β₇ | 1.507 | Very long context overhead | 0.5-2.0 | ✅ In range |
| β₈ | 0.000042 | Per-request decode overhead | 0.00001-0.00005 | ✅ In range |

**Observations**:

**Terms with expected magnitude but INEFFECTIVE**:
- β₇ = 1.507 (in expected range 0.5-2.0) but reasoning experiments still at 99% APE
- β₈ = 0.000042 (in expected range) but β₁ still inflated
- **Conclusion**: Magnitude in expected range does NOT prove mechanism is correct

**Terms with wrong magnitude**:
- β₀ = 0.155 (3× too low): Prefill is grossly under-accounted
- β₁ = 1.316 (1.5× too high): Decode memory-bound is grossly over-accounted
- β₄ = 0.000043 (10× too low vs iter1): KV mgmt importance collapsed

**Why β₇ and β₈ have "expected" magnitudes but don't work**:

The optimizer fits coefficients to minimize RMSE. If a basis function has the WRONG functional form but the RIGHT order of magnitude, the optimizer will assign a plausible-looking coefficient even though the term doesn't capture real physics.

**Example**: β₇ = 1.507 suggests ~1.5μs overhead per (token - 4096) / 1000 × layer. For reasoning experiments with 4000 excess tokens and 30 layers:
```
Overhead = 1.507 × 4 × 30 = 180μs
```

This is a plausible prefill overhead (10-30% of typical TTFT). However, reasoning experiments STILL have 99% APE, proving the 180μs is being added in the WRONG place or at the WRONG time.

**Principle**: **Coefficient magnitude in expected range is NECESSARY but NOT SUFFICIENT to prove mechanism is correct. Always validate with per-experiment analysis.**

**Action for iter3**: When adding a new term, verify it:
1. Has expected coefficient magnitude
2. **AND** improves error on target experiments (e.g., β₇ should fix reasoning experiments)
3. **AND** doesn't degrade error on other experiments

---

## Coefficient Analysis

### Alpha [α₀, α₁, α₂]: Fixed API + Per-Token Overhead

**Optimal values** (from `best_params.alpha`):
- α₀ = 0.00484 (4.84μs fixed API overhead)
- α₁ = 0.000071 (0.071μs per input token)
- α₂ = 0.000032 (0.032μs per output token)

**Physical interpretation**:

**α₀ = 4.84μs**: Fixed API overhead (request parsing, response formatting, logging). Expected: 10-100μs. **SEVERELY UNDERESTIMATED**.

Possible causes:
- Most API overhead already captured by β₂ (constant step overhead)
- Ground truth latencies may not include API overhead (measured at kernel level, not API level)
- Optimizer driving α₀ down to compensate for inflated β₁

**α₁ = 0.071μs/token**: Per-input-token cost. For a 1000-token prompt:
```
1000 × 0.071 = 71μs
```
This is plausible for tokenization + input buffer setup.

**α₂ = 0.032μs/token**: Per-output-token cost. For a 100-token response:
```
100 × 0.032 = 3.2μs
```
This is plausible for detokenization + response streaming.

**Trend**: All Alpha coefficients are VERY SMALL, suggesting most latency is in Beta terms (compute/memory/overhead).

---

### Beta [β₀, β₁, ..., β₈]: Step-Level Basis Functions

**β₀ = 0.155 (Prefill MFU)**

Expected: 0.4-0.5 (40-50% of roofline peak)
Actual: 0.155 (15.5% of peak)
**Status**: ❌ **SEVERELY DEGRADED** (worse than iter1's 0.203)

**Interpretation**: Model predicts prefill achieves only 15.5% of theoretical FLOPs throughput. This is implausible - typical prefill MFU is 40-60% on H100.

**Root cause**: β₀ is being driven down to compensate for OVER-prediction in other terms. Either:
1. Prefill FLOPs calculation is wrong (over-counting)
2. β₇ is adding too much overhead
3. Removing β₅ eliminated a term that was compensating for β₀ under-prediction

**Action**: Investigate prefill FLOPs formula. May need separate MFU coefficients for short vs long prefills.

---

**β₁ = 1.316 (Decode Memory-Bound MFU)**

Expected: 0.6-0.9 (memory bandwidth limited)
Actual: 1.316 (131.6% of bandwidth limit)
**Status**: ❌ **INFLATED** (worse than iter1's 1.553 → improved to 1.316, but still high)

**Interpretation**: Model predicts decode achieves 131.6% of memory bandwidth, which is physically impossible.

**Root cause**: β₁ is compensating for missing decode overhead terms. Candidates:
- **Activation memory bandwidth**: MLP output, attention output, residual adds (not modeled)
- **KV cache fragmentation**: Non-contiguous KV blocks cause extra memory transactions
- **Synchronization overhead**: AllReduce/AllGather latency for TP>1 during decode
- **Scheduler overhead**: vLLM scheduling decisions between steps

**Why β₈ didn't fix this**: β₈ converged to 0.000042 (negligible), proving per-request overhead hypothesis is wrong.

**Action**: Test per-token decode overhead `Σ(context_length)` instead of per-request. Investigate TP communication in decode path.

---

**β₂ = 0.000093 (Constant Step Overhead)**

Expected: ~0.0001 (0.1μs per step)
Actual: 0.000093
**Status**: ✅ **PLAUSIBLE**

**Interpretation**: ~0.1μs overhead per decode step, regardless of batch size or tokens. This captures:
- Event loop overhead
- Scheduler invocation
- Metric logging

Magnitude is reasonable.

---

**β₃ = 0.119 (TP Communication Overhead)**

Expected: 0.05-0.15 (depends on TP degree and interconnect)
Actual: 0.119
**Status**: ✅ **PLAUSIBLE**

**Interpretation**: ~0.12μs overhead per TP communication event. For TP=4, each prefill/decode step has ~3-4 AllReduce operations:
```
4 steps × 0.119 = 0.476μs TP overhead per prefill
```

This is in line with NVLink latency (1-2μs per operation).

---

**β₄ = 0.000043 (KV Management Overhead)**

Expected: 0.0003-0.0004 (based on iter1 = 0.37μs, but iter1 may have been spurious)
Actual: 0.000043
**Status**: ⚠️ **COLLAPSED** (10× smaller than iter1)

**Interpretation**: Iter1 claimed β₄ was CRITICAL (+30.28% E2E degradation when removed). Iter2 drove it to near-zero, suggesting:
1. Iter1's ablation was spurious (overfitting)
2. β₄ was confounded with β₅ (chunking) - removing β₅ eliminated β₄'s role
3. β₇ or β₈ absorbed β₄'s contribution

**Instability**: A "CRITICAL" term collapsing to zero in the next iteration indicates coefficient instability or overfitting.

**Action**: Re-test β₄ ablation on iter2 baseline. If removing β₄ now causes <5% loss increase, iter1's result was spurious.

---

**β₅ = 0.956 (Large-Batch Compute-Bound MFU)**

Expected: 0.6-0.8 (large-batch decode saturates compute)
Actual: 0.956 (after renumbering - this was β₆ in iter1)
**Status**: ⚠️ **SLIGHTLY INFLATED** (up from iter1's 0.651)

**Interpretation**: Model predicts large-batch decode achieves 95.6% of compute throughput. This is higher than expected (typical: 60-80%).

**Why it increased**: Removing β₅ (chunking) and sigmoid regime transition may have shifted more decode overhead onto β₅ (compute-bound) term.

**Action**: Monitor if this stays stable or continues increasing.

---

**β₆ = 0.135 (Memory Management Overhead)**

Expected: Unknown (iter1 called this β₇)
Actual: 0.135
**Status**: ⚠️ **UNKNOWN INTERPRETATION**

**Interpretation**: ~0.135μs overhead for memory management operations. Need to check code to understand what this term captures.

**Action**: Document β₆ basis function definition. May be capturing KV cache allocation, prefill chunking, or GPU memory pool management.

---

**β₇ = 1.507 (Very Long Context Overhead) - NEW TERM**

Expected: 0.5-2.0
Actual: 1.507
**Status**: ✅ **IN EXPECTED RANGE** but ❌ **INEFFECTIVE**

**Interpretation**: ~1.5μs overhead per `(tokens - 4096) / 1000 × num_layers`. For reasoning experiments (4000 excess tokens, 30 layers):
```
1.507 × 4 × 30 = 180μs overhead
```

This is a plausible overhead magnitude (10-30% of TTFT). **However, reasoning experiments still have 99% TTFT APE**, proving the mechanism is wrong.

**Why it's ineffective**:
- Wrong functional form (linear vs quadratic)
- Wrong threshold (4096 vs 2048/8192)
- Wrong feature (prompt length vs KV cache state)

**Action**: Replace with quadratic or piecewise term.

---

**β₈ = 0.000042 (Per-Request Decode Overhead) - NEW TERM**

Expected: 0.00001-0.00005 (10-50μs per request)
Actual: 0.000042
**Status**: ✅ **IN EXPECTED RANGE** but ❌ **INEFFECTIVE**

**Interpretation**: ~0.042μs per request = 42 nanoseconds. For a batch of 8 requests:
```
8 × 0.042 = 0.34μs total overhead
```

This is NEGLIGIBLE compared to typical decode step times (100-1000μs). **β₈ is effectively zero.**

**Why it's ineffective**:
- Overhead is constant (already in α₀ or β₂), not per-request
- Overhead scales with context length, not request count
- Overhead is negligible (<1μs total)

**Action**: Replace with per-token decode overhead `Σ(context_length)` or remove entirely.

---

### Redundant Terms

**Candidates for removal** (coefficients near zero):
- β₂ = 0.000093 (constant step overhead) - plausible but tiny
- β₄ = 0.000043 (KV mgmt) - collapsed from iter1, likely redundant now
- β₈ = 0.000042 (per-request) - proven ineffective

**Should NOT remove**:
- β₇ = 1.507 - large coefficient, even though ineffective, suggests optimizer is trying to fit SOME real overhead (wrong functional form)

**Action**:
1. Remove β₈ (per-request) - clearly redundant
2. Keep β₇ but change functional form (quadratic or piecewise)
3. Re-test β₄ ablation before removing

---

### Missing Physics

**Coefficient magnitudes suggest missing terms**:

1. **β₀ = 0.155 (too low)**: Prefill MFU under-predicted by 3×
   - **Missing**: Separate MFU for short vs long prefills
   - **Missing**: Prefill memory bandwidth term (current model only has compute-bound)

2. **β₁ = 1.316 (too high)**: Decode memory-bound over-predicted by 1.5×
   - **Missing**: Activation memory bandwidth (residual connections, MLP output)
   - **Missing**: KV cache fragmentation overhead
   - **Missing**: Per-token decode overhead (not per-request)

3. **β₇ = 1.507 (ineffective)**: Wrong functional form for very long context
   - **Missing**: Quadratic attention scaling
   - **Missing**: Discrete chunking boundaries
   - **Missing**: KV eviction/recomputation model

**Priority for iter3**: Add per-token decode overhead to reduce β₁ inflation.

---

## Recommendations for iter3

### Priority 1: Critical Blockers (Must Fix to Achieve <80% Loss)

**1. Fix Scout MoE Validation** (800% contribution, 53% of loss)
- **Action**: Debug why all Scout experiments have exactly 100% APE
- **Investigate**: Model name parsing, FP8 quantization, MoE expert routing
- **Verification**: Print predicted vs observed latencies for Scout experiments
- **Timeline**: BEFORE designing iter3 hypothesis

**2. Re-Examine Reasoning Experiment Root Cause** (390% contribution, 26% of loss)
- **Action**: Investigate WHY reasoning experiments have 99% TTFT APE
- **Test**: Are prompts actually >4096 tokens? Check ground truth prompt lengths.
- **Hypotheses**:
  - Ground truth latencies incorrect (validation failure like Scout)
  - Prompts shorter than expected (β₇ never activated)
  - Mechanism is wrong (quadratic, not linear)
- **Verification**: Manually inspect one reasoning experiment end-to-end
- **Timeline**: BEFORE designing iter3 hypothesis

---

### Priority 2: Model Structure Improvements (Required for <50% Loss)

**3. Replace β₇ (Very Long Context) with Better Functional Form**
- **Action**: Test quadratic, piecewise, or log scaling instead of linear
- **Candidates**:
  - Quadratic: `β × max(0, (tokens - threshold)²) / 1e6`
  - Piecewise: `β × (0 if tokens<2048, 1×(tokens-2048) if <4096, 2×(tokens-4096) if >=4096)`
  - Chunking: `β × floor(tokens / chunk_size) × num_layers`
- **Verification**: Check if reasoning experiments improve to <50% TTFT APE
- **Timeline**: iter3

**4. Add Per-Token Decode Overhead (Replace β₈)**
- **Action**: Replace per-request `β₈ × num_requests` with per-token `β₉ × Σ(context_length)`
- **Rationale**: β₈ converged to zero, but β₁ still inflated (missing decode overhead)
- **Expected**: β₁ drops from 1.316 to 0.6-0.9, β₉ converges to 0.00001-0.0001
- **Verification**: Check if E2E RMSE decreases
- **Timeline**: iter3

**5. Restore β₅ (Prefill Chunking Overhead)**
- **Action**: Re-add the removed β₅ term from iter1
- **Rationale**: β₀ degraded from 0.203 to 0.155 after removing β₅, suggesting β₅ was absorbing real overhead
- **Expected**: β₀ rises back to 0.2-0.3
- **Verification**: Check if TTFT RMSE decreases
- **Timeline**: iter3 or iter4 (if iter3 still fails)

---

### Priority 3: Process Improvements (Prevent Future Failures)

**6. Test ONE Mechanism at a Time**
- **Action**: DO NOT add multiple basis functions in a single iteration
- **Rationale**: Adding β₇ + β₈ + removing β₅ + changing sigmoid made it impossible to isolate which change caused degradation
- **Protocol**: iter3 should test ONLY the highest-priority mechanism (fix reasoning experiments)
- **Timeline**: Immediate (process change)

**7. Validate Ablation Results Before Removing Terms**
- **Action**: DO NOT remove terms based on <5% ablation loss increase
- **Rationale**: Removing β₅ caused 12% degradation, despite iter1 ablation showing only +1.06% increase
- **Protocol**: Only remove terms with <0.5% ablation impact OR coefficient <1e-6
- **Timeline**: Immediate (process change)

**8. Check Coefficient Stability Across Iterations**
- **Action**: Track coefficient evolution across iterations (β₄: 0.37μs → 0.000043)
- **Rationale**: Dramatic coefficient changes indicate instability or overfitting
- **Metric**: If any coefficient changes by >50% between iterations, investigate root cause before accepting new model
- **Timeline**: Immediate (add to analysis workflow)

---

### Priority 4: Hypothesis Design Guidelines (Improve iter3 Success Rate)

**9. Require Per-Experiment Analysis in Hypothesis**
- **Action**: Agent 1 must predict specific experiments that will improve and by how much
- **Example**: "β₇ will reduce reasoning experiments (Qwen, Llama-2) from 99% to <50% TTFT APE"
- **Rationale**: H-main predicted <55% overall loss but didn't specify which experiments would improve
- **Verification**: Agent 3 can immediately check if target experiments improved
- **Timeline**: iter3 hypothesis

**10. Require Diagnostic Thresholds for Coefficients**
- **Action**: Agent 1 must specify "if β < X, mechanism is absent; if X < β < Y, mechanism is working"
- **Example**: "if β₇ < 0.01, very long context overhead is negligible; if 0.5 < β₇ < 2.0, working as intended"
- **Rationale**: β₇ = 1.507 was "in range" but ineffective - need stronger validation than magnitude alone
- **Verification**: Agent 3 checks both coefficient magnitude AND per-experiment improvement
- **Timeline**: iter3 hypothesis

**11. Limit Hypothesis Count to 3 Core Hypotheses**
- **Action**: Agent 1 should write H-main + 2-3 ablation hypotheses, NOT 6 hypotheses
- **Rationale**: 6 hypotheses in iter2 created validation burden (3 experimental, 3 observable)
- **Protocol**: H-main + H-ablation-X + H-ablation-Y only
- **Timeline**: iter3 hypothesis

---

## Basis Function Changes for iter3

**Remove**:
- β₈ (per-request decode overhead) - proven ineffective (coefficient 0.000042)

**Modify**:
- β₇ (very long context) → Replace `max(0, tokens - 4096)` with quadratic or piecewise

**Add**:
- β₉ (per-token decode overhead) → `β × Σ(context_length_per_request)` to normalize β₁

**Maybe restore**:
- Old β₅ (prefill chunking overhead) → If β₀ doesn't improve, restore this term

**Final count**: 9 Beta terms (if only modify β₇, add β₉, remove β₈) OR 10 Beta terms (if also restore old β₅)

---

## Bounds Adjustments

**Tighten bounds** (coefficients stable):
- β₂: [0.00005, 0.0002] → stable at ~0.0001
- β₃: [0.05, 0.20] → stable at ~0.12

**Expand bounds** (coefficients hit limits):
- β₀: [0.1, 0.8] → currently 0.155, allow down to 0.05 or up to 1.0
- β₁: [0.3, 2.0] → currently 1.316, allow up to 3.0 if needed (though ideally should decrease)

**New term bounds**:
- β₇ (quadratic long context): [0.0001, 0.01] (quadratic scales faster, need smaller coefficient)
- β₉ (per-token decode): [1e-6, 1e-4] (per-token overhead is tiny)

---

## Overall Assessment

**Model readiness**: ❌ **NOT READY** - Loss increased 12%

**Strengths**:
- β₃ (TP communication) stable and plausible
- β₂ (constant step overhead) stable
- Optimization converged reliably (142 trials, 0 errors)

**Weaknesses**:
- Scout MoE validation failure blocks 53% of progress
- Reasoning experiments unchanged despite β₇ targeting them
- β₁ inflation persists despite β₈ targeting it
- β₀ degraded significantly
- Testing multiple hypotheses simultaneously masked root causes

**Recommendation**:

**DO NOT PROCEED TO ITER3 UNTIL**:
1. ✅ Scout MoE validation is fixed (blocks 53% of progress)
2. ✅ Reasoning experiment root cause is identified (blocks 26% of progress)
3. ✅ Validation that β₇ functional form is correct (check actual prompt lengths)

**After fixes, iter3 should**:
1. Test ONLY ONE mechanism (either fix long context OR add per-token decode, not both)
2. Require per-experiment predictions (not just overall loss target)
3. Use stricter hypothesis validation (coefficient magnitude + target experiment improvement)

**Timeline estimate**: Fixing Scout MoE + reasoning investigation: 1-2 days. Iter3 design + execution: 3-5 days.
