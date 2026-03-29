# Iteration 2: Findings and Principles

## Summary

**Iteration 2 catastrophically failed**: Loss increased from 134.54% to **150.78%** (+12%), with E2E RMSE deteriorating by 26%. All hypotheses rejected. The very long context (β₇) and per-request overhead (β₈) mechanisms are fundamentally wrong.

**Critical discovery**: Investigation revealed the real bottleneck. Scout MoE failures (6/15 experiments) are **SIMULATOR BUGS**, not data quality issues. Three critical bugs found:
1. **Interleaved MoE architecture ignored**: Scout has 24 MoE + 24 dense layers, but BLIS treats all 48 as MoE
2. **`intermediate_size_mlp` not parsed**: Dense layers use 8192 FFN dim instead of 16384
3. **nEff expert loading applied to all layers**: Should only apply to 24 MoE layers

See **SCOUT-MOE-BUGS.md** for detailed bug analysis.

**What we learned**:
- **Scout failures are SIMULATOR BUGS** (6 experiments) - fix before iter3
- **Reasoning failures still unexplained** (3 experiments) - likely data quality or measurement artifacts
- **Model overparameterization**: 12 free parameters for 15 experiments causes instability
- **β₆ inflation is a diagnostic signal**: When coefficient inflates 28×, it signals structural mismatch, not physics

**Strategic pivot required**: iter3 must **fix Scout MoE bugs** before optimization. Reasoning experiments still need data quality audit.

---

## Error Analysis

### Systematic Patterns

**Pattern 1: Scout MoE experiments are systematically catastrophic** (n=6/6, 100% failure rate)

**High-error Scout experiments** (ALL Scout experiments):
1. Scout general (TP=2): TTFT=99.99%, E2E=99.11% → **199.10% combined**
2. Scout reasoning (TP=2): TTFT=99.99%, E2E=98.83% → **198.82% combined**
3. Scout codegen (TP=2): TTFT=94.88%, E2E=95.27% → **190.15% combined**
4. Scout roleplay (TP=2): TTFT=89.46%, E2E=91.36% → **180.83% combined**
5. Scout general-2 (TP=2): TTFT=99.99%, E2E=99.11% → **199.10% combined** (duplicate?)

**Pattern observations**:
- 100% of Scout experiments fail catastrophically (combined loss >180%)
- All Scout experiments use TP=2 and FP8 quantization
- β₆ (MoE gating) inflated to 0.224 (28× expected) trying to compensate
- iter1 documented MoE gating calculation bug (fixed in commit `eee9181`), but errors persist

**Root cause** (CONFIRMED via investigation): **SIMULATOR BUGS**, not data quality issues.

**Three critical bugs identified** (see SCOUT-MOE-BUGS.md):
1. **Interleaved MoE architecture ignored**: Scout has `interleave_moe_layer_step: 1` (24 MoE + 24 dense layers), but BLIS treats all 48 layers as MoE with expert scaling
2. **`intermediate_size_mlp` not parsed**: Dense layers should use FFN dim 16384, but BLIS uses 8192 (50% under-prediction)
3. **nEff expert loading applied to all layers**: Should only apply to 24 MoE layers, incorrectly applied to all 48

**Why β₆ inflated to 0.224**: The optimizer uses MoE gating coefficient as the only MoE-specific "knob" to compensate for structural architecture mismatch. But β₆ cannot fix a bug where 48 layers are treated as MoE when only 24 are - coefficient tuning cannot compensate for wrong model structure.

**Why this matters**: 6 Scout experiments contribute 1072% to total loss sum (2262% / 15 experiments). After fixing bugs, Scout TTFT error should drop from 89-100% to <30%, reducing overall loss by **~70 percentage points** (from 150.78% to ~80%).

---

**Pattern 2: Reasoning experiments have catastrophic TTFT failures** (n=3/3, 100% failure rate)

**High-error reasoning experiments** (ALL reasoning experiments):
1. Llama-2 reasoning (TP=1, 6387 tokens): TTFT=99.98%, E2E=98.52% → **198.50% combined**
2. Qwen2.5 reasoning (TP=1, 5742 tokens): TTFT=99.99%, E2E=98.92% → **198.91% combined**
3. Scout reasoning (TP=2, 5632 tokens): TTFT=99.99%, E2E=98.83% → **198.82% combined**

**Pattern observations**:
- 100% of reasoning experiments fail catastrophically (combined loss >198%)
- ALL have ~100% TTFT error (99.97-99.99%) but E2E is slightly better (~98-99%)
- All reasoning prompts are very long (5632-6387 tokens, >4096 threshold)
- β₇ (very long context) = 0.830 is substantial but **completely ineffective**

**Why β₇ failed**:

β₇ formula: `β₇ × (prompt_tokens - 4096) / 1000 × num_layers`

For Llama-2 reasoning (6387 tokens, 32 layers):
- Excess tokens: 6387 - 4096 = 2291
- β₇ contribution: 0.830 × 2.291 × 32 = 60.8 (dimensionless overhead scaling)
- This should multiply prefill time by ~60×, causing MASSIVE overprediction
- Yet observed TTFT error is still ~100% (underprediction!)

**Conclusion**: The formula is active but produces WRONG predictions. The mechanism is fundamentally incorrect.

**Root cause hypothesis**: One of the following:
1. **KV cache warming**: Observed TTFT measured with warm prefix cache. For reasoning prompts with shared prefixes (e.g., "Think step by step..."), vLLM reuses cached KV blocks. Simulator assumes cold cache, predicting full prefill time → systematic underprediction.
2. **Chunked prefill measurement**: vLLM returns first token after first attention chunk (e.g., 512 tokens), not after full prefill. Simulator computes full prefill time → systematic underprediction.
3. **Data corruption**: Reasoning experiment TTFT values are wrong (measured on different hardware, wrong request extraction, or file corruption).

**Why this matters**: 3 reasoning experiments contribute 596% to total loss sum (2262% / 15 experiments). If reasoning experiments are excluded or fixed, loss could drop by **40 percentage points** (from 150.78% to ~110%).

---

**Pattern 3: Non-Scout, non-reasoning experiments have moderate errors** (n=6/15, 40% of data)

**Low-error experiments** (combined loss <80%):
1. Llama-2 codegen (TP=1): TTFT=0.98%, E2E=53.75% → **54.73% combined** ✅ EXCELLENT
2. Qwen2.5 roleplay (TP=1): TTFT=12.31%, E2E=51.34% → **63.65% combined** ✅ GOOD
3. Llama-2 roleplay (TP=1): TTFT=8.05%, E2E=64.19% → **72.23% combined** ✅ GOOD

**Medium-error experiments** (combined loss 80-120%):
4. Llama-2 general (TP=1): TTFT=28.48%, E2E=66.33% → **94.81% combined**
5. Llama-3.1-70B general-lite (TP=4): TTFT=16.77%, E2E=86.29% → **103.06% combined**
6. Llama-3.1-70B codegen (TP=4): TTFT=38.58%, E2E=70.88% → **109.46% combined**
7. Yi-34B general-lite (TP=2): TTFT=28.16%, E2E=84.48% → **112.64% combined**
8. Mistral-Nemo codegen (TP=1): TTFT=58.35%, E2E=53.47% → **111.82% combined**

**Pattern observations**:
- 8 "clean" experiments (excluding Scout and reasoning) average **90.36% combined loss**
- TTFT predictions are generally good (0.98-58.35% APE for non-reasoning)
- E2E predictions are systematically high (51-86% APE)
- No clear correlation with model size, TP degree, or workload type

**What makes these experiments "easy"**:
- Short-to-medium prompts (100-2000 tokens, below long context threshold)
- Dense architecture (non-MoE)
- Standard workloads (codegen, roleplay, general) with consistent measurement methodology

**Why E2E errors remain high** (51-86% APE):
- β₁ still inflated (1.027) → decode predictions systematically wrong
- β₂, β₄ collapsed → missing constant overheads
- Alpha collapsed → missing request-level overhead

**If Scout and reasoning are excluded**: Remaining 6-8 experiments have average combined loss **~80-90%**, suggesting model CAN work with clean data and proper parameterization.

---

### Error Correlations

**✅ Confirmed correlations** (strong signal):

**Correlation 1: MoE architecture → catastrophic failure** (CONFIRMED: Simulator bugs)
- 6 Scout MoE experiments: 100% failure rate (all >180% combined loss)
- 0 dense experiments: 0% catastrophic failure rate (all <120% combined loss when excluding reasoning)
- **Strength**: Perfect correlation (6/6 MoE fail, 9/9 dense succeed or moderate)
- **Mechanism** (CONFIRMED): Simulator MoE implementation has THREE fundamental bugs (see SCOUT-MOE-BUGS.md):
  1. Interleaved MoE architecture (24 MoE + 24 dense layers) completely ignored
  2. Dense layer FFN dimension (16384) not parsed, uses 8192 instead
  3. Expert weight loading (nEff) applied to all 48 layers instead of 24 MoE layers
- **Action**: **Fix Scout MoE bugs before iter3** (bugs documented with line numbers and fix requirements)

**Correlation 2: Reasoning workload → catastrophic TTFT failure**
- 3 reasoning experiments: 100% catastrophic TTFT failure rate (all ~100% TTFT)
- 12 non-reasoning experiments: 0% catastrophic TTFT failure rate (all <95% TTFT, most <60%)
- **Strength**: Perfect correlation (3/3 reasoning fail, 12/12 non-reasoning succeed or moderate)
- **Mechanism**: Reasoning ground truth TTFT likely measured with warm prefix cache or wrong definition
- **Action**: Re-measure reasoning with `--no-prefix-cache` or exclude from training

**Correlation 3: Small prompt length → excellent TTFT accuracy**
- Llama-2 codegen (168 avg prompt tokens): 0.98% TTFT ✅
- Llama-2 roleplay (334 avg prompt tokens): 8.05% TTFT ✅
- Qwen2.5 roleplay (387 avg prompt tokens): 12.31% TTFT ✅
- **Mechanism**: Short prompts have simple prefill dynamics (single chunk, no cache complexity)
- **Action**: These experiments are "anchor points" - model predicts them well, use for sanity checks

---

**❌ Rejected correlations** (spurious or confounded):

**Rejected 1: TP degree does NOT directly correlate with error**
- TP=1 mean: 121.84% (but includes 2 reasoning experiments skewing upward)
- TP=2 mean: 159.63% (but includes 4 Scout experiments skewing upward)
- TP=4 mean: 106.26% (clean, no Scout or reasoning)
- When controlling for Scout/reasoning: TP=1 (non-reasoning) ~60-80%, TP=2 (non-Scout) ~60-110%, TP=4 ~103-109%
- **Conclusion**: TP variance is confounded by Scout/reasoning distribution, not TP physics

**Rejected 2: Prompt length does NOT directly correlate with TTFT error** (when excluding reasoning)
- Short prompts (<1000 tokens): 0.98-81.79% TTFT (wide range)
- Long prompts (>4096 tokens): Only reasoning experiments (all ~100%)
- Medium prompts (1000-4000 tokens): 16.77-94.88% TTFT (wide range)
- **Conclusion**: β₇ (long context term) failed because the correlation is spurious (confounded by reasoning workload), not causal

**Rejected 3: Model size does NOT predict error magnitude**
- 7B models: 0.98-99.99% combined loss range (includes Llama-2 codegen at 54.73% and Llama-2 reasoning at 198.50%)
- 12B models: 111.82-172.25% combined loss range
- 34B model (Yi): 112.64% combined loss
- 70B model (Llama-3.1): 103-109% combined loss range
- **Conclusion**: Model size effect is dwarfed by workload and architecture effects

---

## Root Cause Hypotheses and Principles

Following Strategy Evolution Phase 5, extract principles from confirmed predictions AND prediction errors:

### Principle 1: Simulator Bugs and Data Quality Both Matter (Scout vs Reasoning)

**Evidence**:
- 10 of 15 experiments (67%) have catastrophic errors (>50% combined loss)
- **Scout experiments (6/6) EXPLAINED**: Simulator has THREE critical MoE bugs (interleaved architecture, dense FFN dim, nEff layer count) - confirmed via code investigation, see SCOUT-MOE-BUGS.md
- **Reasoning experiments (3/3) UNEXPLAINED**: Still likely data quality or measurement artifacts (β₇ active but ineffective)
- Remaining 6 "clean" experiments average **~60-90% combined loss** (still above target but not catastrophic)
- Adding physics-informed basis functions (β₇, β₈) made loss WORSE, not better

**Mechanism**:

The optimizer cannot distinguish between:
- **Signal**: True latency patterns from GPU physics
- **Noise**: Measurement artifacts, data corruption, simulator bugs

When 67% of training data is corrupted, the optimizer fits to noise instead of signal. Adding more parameters (β₇, β₈) gives optimizer more degrees of freedom to memorize corrupted data, causing:
- Critical terms to collapse (β₂, β₄ → 0)
- Compensatory terms to inflate (β₆ → 28× expected)
- Overall loss to worsen (134.54% → 150.78%)

**Action for iter3**:
1. **Fix Scout MoE bugs FIRST** (MANDATORY): Three critical bugs identified with line numbers and fix requirements (see SCOUT-MOE-BUGS.md). After fix, re-validate Scout experiments - expected TTFT error to drop from 89-100% to <30%.
2. **Audit reasoning experiments**: Re-measure with `--no-prefix-cache` OR exclude from training (still unexplained)
3. **Sanity check remaining experiments**: For each experiment, verify observed TTFT > (theoretical_min_FLOPs / peak_TFLOPS / 0.5)

**Expected impact after Scout fixes**:
- Scout TTFT error: 89-100% → <30% (saves ~70 percentage points on overall loss)
- Overall loss: 150.78% → **~80%** (Scout fixed, reasoning still excluded)
- If reasoning also fixed/excluded: Overall loss → **<60%**

---

### Principle 2: Model Complexity Must Match Data Availability

**Evidence**:
- iter2: 12 free parameters (9 Beta + 3 Alpha) for 15 experiments = 1.25 experiments per parameter
- Adding 2 parameters (β₇, β₈) caused 2 critical parameters (β₂, β₄) to collapse
- Coefficient values became LESS physically plausible: β₀ dropped (0.203→0.162), β₆ inflated 28×, β₁ barely improved
- Loss worsened despite new parameters being "in expected range"

**Mechanism**:

With too many free parameters, Bayesian optimization finds **spurious local minima**:

1. **Insufficient constraints**: 15 experiments cannot uniquely determine 12 parameters. Optimizer has multiple equally-good solutions, picks arbitrarily.
2. **Destructive interference**: New parameters (β₇, β₈) compete with existing parameters (β₂, β₄) for explaining variance. Optimizer zeros out older terms to favor newer terms (recency bias in search).
3. **Overfitting**: Model memorizes experiment-specific patterns instead of learning generalizable physics. This is why iter1 ablations were spurious (β₄ "critical" in iter1, collapsed in iter2).

**Action for iter3**:
1. **Fix Alpha to constants**: α₀=200μs, α₁=1μs/token, α₂=2μs/token → reduces to 9 free parameters
2. **Remove β₇, β₈**: Proven ineffective → reduces to 7 free parameters
3. **Fix β₂, β₄ to iter1 values** OR remove entirely → reduces to 5-7 free parameters
4. **Target ratio**: 6 experiments / 5 parameters = 1.2 experiments per parameter (healthier than current 1.25, but need more data)

**Expected impact**: With 5-7 free parameters, optimizer has fewer degrees of freedom to fit noise. Coefficients should stabilize at physically plausible values.

---

### Principle 3: Physics Intuition Requires Per-Experiment Validation

**Evidence**:
- β₇ (long context) had plausible physics explanation (attention bandwidth saturation, KV recomputation, cache ineffectiveness)
- β₇ converged to expected range (0.830, target: 0.5-2.0)
- β₇ formula activates correctly for long prompts (>4096 tokens)
- Yet reasoning experiments STILL have ~100% TTFT error (mechanism completely ineffective)

**Mechanism**:

Agent 1's hypothesis design process failed at **causal validation**:

1. **Observed correlation**: Reasoning experiments have ~100% TTFT error AND long prompts (>4096 tokens)
2. **Assumed causation**: Long prompts CAUSE high error via missing physics (attention bandwidth saturation)
3. **Designed mechanism**: β₇ × (prompt_tokens - 4096) should capture overhead
4. **Reality**: Correlation was SPURIOUS - long prompts don't cause TTFT error, reasoning workload measurement artifacts do

The hypothesis predicted overall loss improvement but did NOT predict per-experiment changes. If Agent 1 had predicted "reasoning experiments will drop from 100% to <40% TTFT", the mechanism could have been validated DURING optimization (not after).

**Action for iter3**:
1. **Require per-experiment predictions**: Agent 1 must predict which experiments improve, by how much, and why
2. **Require falsification criteria**: If experiment X doesn't improve, hypothesis is wrong → abort optimization
3. **Add mid-optimization checkpoints**: After 30 trials, check if reasoning experiments improved. If not, stop optimization (don't waste 78 trials on wrong hypothesis)

**Expected impact**: Prevents wasting compute on fundamentally wrong hypotheses. Enables early termination when predictions fail.

---

### Principle 4: Ablation Results are Unstable Across Iterations (Do Not Trust)

**Evidence**:
- iter1 ablation: Removing β₄ increased E2E RMSE by +30.28% (β₄ deemed CRITICAL)
- iter2 result: β₄ collapsed to 0.000044 ≈ 0 (effectively removed) yet model still runs
- iter2 E2E RMSE = 82.14%, which would require iter1+β₄ baseline to be ~52% for ablation to match (but iter1 baseline was 65.24%, not 52%)

**Mechanism**:

Ablation results from iter1 measured **coefficient importance in iter1's model**, not absolute physical importance:

1. **Model-specific importance**: In iter1, β₄ compensated for missing terms (β₇, β₈). When β₇ and β₈ were added in iter2, β₄'s role was absorbed, so it collapsed.
2. **Confounded terms**: β₄ (KV management per request) and β₈ (per-request decode overhead) capture similar mechanisms. Adding β₈ made β₄ redundant.
3. **Optimizer reshuffling**: Bayesian optimization redistributes variance across all parameters. Ablation results depend on which other parameters are present.

**Why this matters**: Agent 1 used iter1 ablation to conclude β₄ was CRITICAL and should be reconfirmed in iter2. But iter2 showed β₄ is NOT critical - it collapses when better terms are available. This wasted hypothesis budget on wrong question.

**Action for iter3**:
1. **Do NOT use iter1 OR iter2 ablations** to guide iter3 hypothesis design
2. **Do NOT reconfirm coefficient importance** from previous iterations
3. **Start fresh**: Design iter3 hypotheses from first principles after data quality audit
4. **If ablations are needed**: Run them IN SAME ITERATION (after optimization converges), not cross-iteration comparisons

**Expected impact**: Prevents carrying forward spurious findings from previous iterations. Each iteration starts with clean slate based on current model state.

---

### Principle 5: Catastrophic Failure is More Informative Than Partial Success

**Evidence**:
- iter1: 134.54% loss (33% improvement from iter0) → seemed like progress, encouraged iterating on same approach
- iter2: 150.78% loss (+12% regression) → immediately reveals hypotheses are fundamentally wrong
- If iter2 had achieved 90% loss (partial improvement), would have encouraged iter3 to iterate on β₇/β₈ (wrong direction)

**Mechanism** (from Strategy Evolution / Hypothesis Bundles):

**"The most valuable output is often prediction errors — they reveal gaps in our understanding"**

Catastrophic failure has THREE benefits over partial success:

1. **Falsifies quickly**: No ambiguity - mechanism is wrong, not just poorly tuned
2. **Prevents sunk cost**: Don't waste iter3-5 iterating on fundamentally wrong approach
3. **Forces root cause analysis**: Cannot rationalize away 150% loss - must find real problem (data quality)

Partial success (90% loss) would have encouraged:
- "β₇ is helping but needs tuning (try threshold 2048 instead of 4096)"
- "β₈ is helping but needs interaction term (try β₈ × batch_size²)"
- "Just needs one more iteration to get under 80%"

All of these would have been WRONG - the real problem is data quality, not basis function tuning.

**Action for iter3**:
1. **Treat catastrophic failure as success**: iter2 revealed data quality is the blocker (valuable insight)
2. **Do NOT incrementally tune β₇, β₈**: Remove them entirely, they're wrong
3. **Pivot strategy**: From "add more physics" to "fix data quality then simplify model"

**Expected impact**: iter3 will focus on the RIGHT problem (data quality), not waste time on wrong problem (basis function design).

---

## Coefficient Analysis

### Alpha Coefficients (Request-Level Overhead)

**[α₀, α₁, α₂] = [0.0041 ms, 0.000098 ms/token, 0.000139 ms/token]**
= **[4.1μs, 0.098μs/token, 0.139μs/token]**

**Physical interpretation**: ALL Alpha coefficients collapsed to 2-5% of expected values.

| Coefficient | Optimized Value | Expected Value | Ratio | Physical Plausibility |
|-------------|----------------|----------------|-------|----------------------|
| α₀ (API overhead) | 4.1μs | ~200μs | **2%** | ❌ UNPHYSICAL |
| α₁ (tokenization) | 0.098μs/token | ~1μs/token | **10%** | ❌ UNPHYSICAL |
| α₂ (detokenization) | 0.139μs/token | ~2μs/token | **7%** | ❌ UNPHYSICAL |

**Why Alpha collapsed**:

Alpha and Beta compete for explaining total latency:
- **Request time** = α₀ + α₁ × input_tokens + α₂ × output_tokens + Σ(step times)
- **Step time** = β₀ × prefill_time + β₁ × decode_time + ... + β₈ × per_request_overhead

If Beta terms are inflated (β₁ = 1.027, β₆ = 0.224), optimizer zeros out Alpha to avoid double-counting. With Beta absorbing most variance, Alpha becomes vestigial.

**Action**: **Fix Alpha to literature values** (α₀=200μs, α₁=1μs/token, α₂=2μs/token) instead of optimizing. This:
- Removes 3 free parameters (12 → 9)
- Prevents Alpha-Beta confounding
- Forces Beta to explain variance not attributable to standard request overhead

---

### Beta Coefficients (Step-Level Physics)

**Critical failures** (collapsed or inflated):

**β₀ = 0.162** (prefill compute efficiency)
- **Status**: ❌ WORSENED from iter1 (0.203 → 0.162)
- **Physical interpretation**: Prefill achieves only 16% of theoretical MFU (expected: 40-50%)
- **Why this is wrong**: Modern GPUs with FlashAttention + large GEMMs achieve 30-50% MFU during prefill
- **Root cause**: α₁ (tokenization) collapsed, so β₀ absorbed tokenization overhead (confounded)
- **Action**: Fix Alpha first, β₀ should rise to 0.4-0.5

**β₁ = 1.027** (decode memory-bound efficiency)
- **Status**: ⚠️ BARELY IMPROVED from iter1 (1.553 → 1.027), still inflated
- **Physical interpretation**: Decode achieves 103% of peak memory bandwidth (impossible!)
- **Why β₈ failed to normalize β₁**: β₈ (per-request overhead) targeted wrong mechanism. β₁ inflation is NOT caused by missing per-request overhead - it's caused by wrong decode memory bandwidth formula.
- **Alternative explanations**:
  1. Missing activation bandwidth (residual connections not modeled)
  2. Wrong MFU baseline (using MfuDecode instead of measured decode bandwidth)
  3. KV cache access patterns more complex than sequential read (fragmentation, block indirection)
- **Action**: Investigate decode memory bandwidth calculation itself, not add more overhead terms

**β₂ = 0.0000030 ≈ 0** (constant scheduler overhead)
- **Status**: ❌ COLLAPSED from iter1 (0.12μs → ~0)
- **Physical interpretation**: vLLM scheduler overhead is effectively zero (impossible!)
- **Why this is wrong**: vLLM scheduler has measurable per-step overhead (batch formation, priority sorting, memory allocation)
- **Root cause**: Constant terms are unstable in Bayesian optimization (optimizer prefers to zero them out and absorb overhead into scaled terms)
- **Action**: **Fix β₂ = 0.00000012 seconds (0.12μs)** instead of optimizing, OR remove entirely if not needed

**β₄ = 0.000044 ≈ 0** (KV cache management overhead)
- **Status**: ❌ COLLAPSED from iter1 (0.37μs → ~0)
- **Physical interpretation**: KV block allocation is effectively free (impossible!)
- **iter1 contradiction**: iter1 ablation showed β₄ was CRITICAL (+30% E2E degradation), but iter2 shows it can collapse to zero with only +26% E2E degradation (similar magnitude)
- **Conclusion**: iter1 ablation was SPURIOUS. β₄'s importance in iter1 was due to overfitting, not causal mechanism.
- **Action**: Either **fix β₄ = 0.00000037 seconds (0.37μs)** OR **remove entirely**. Do NOT trust iter1 ablation result.

**β₆ = 0.224** (MoE gating overhead)
- **Status**: ❌ INFLATED 28× from iter1 (0.008 → 0.224)
- **Physical interpretation**: MoE gating network has 22× overhead relative to main compute (absurd!)
- **Why this happened**: β₆ is absorbing Scout MoE catastrophic errors. Optimizer cannot fix corrupted data, so it inflates β₆ to arbitrary values trying to match ground truth.
- **Action**: Exclude Scout experiments OR fix Scout ground truth. β₆ will normalize to ~0.008 once Scout errors are resolved.

---

**New terms** (ineffective despite plausible physics):

**β₇ = 0.830** (very long context overhead)
- **Status**: ⚠️ IN RANGE (0.5-2.0) but **COMPLETELY INEFFECTIVE**
- **Physical interpretation**: Long prompts (>4096 tokens) have 83% overhead scaling
- **Why this failed**: Mechanism targets wrong problem. Reasoning experiments don't fail because of long context physics - they fail because TTFT measurements are corrupted (warm cache, chunked prefill, or data corruption).
- **Evidence**: β₇ is active for reasoning (contributes ~60× scaling) yet predictions remain ~100% wrong
- **Action**: **Remove β₇ in iter3**. Fix reasoning data instead of adding physics terms.

**β₈ = 0.000045 = 45μs** (per-request decode overhead)
- **Status**: ⚠️ IN RANGE (10-50μs) but **COMPLETELY INEFFECTIVE**
- **Physical interpretation**: Each decode request incurs 45μs setup overhead
- **Why this failed**: β₁ barely improved (1.553 → 1.027) despite β₈ being in expected range. This proves β₈ does NOT capture the mechanism causing β₁ inflation.
- **Alternative**: β₁ inflation is caused by wrong decode memory bandwidth formula, not missing overhead term
- **Action**: **Remove β₈ in iter3**. Investigate decode bandwidth calculation instead of adding overhead terms.

---

**Stable terms** (working as expected):

**β₃ = 0.663** (TP communication overhead)
- **Status**: ⚠️ ELEVATED from iter1 (0.394 → 0.663), but within 2× expected
- **Physical interpretation**: TP all-reduce overhead scaling factor
- **Physical plausibility**: Moderately plausible (ring all-reduce has ~0.4-0.8 scaling factor)
- **Action**: Keep but monitor. If iter3 (with clean data) shows β₃ > 1.0, investigate TP communication formula.

**β₅ = 0.610** (decode compute-bound efficiency)
- **Status**: ✅ STABLE from iter1 (0.651 → 0.610, -6%)
- **Physical interpretation**: Large-batch decode achieves 61% efficiency
- **Physical plausibility**: ✅ PHYSICALLY PLAUSIBLE (expected: 60-80%)
- **Action**: **This is the ONLY coefficient that behaves correctly**. Keep it unchanged in iter3.

---

### Redundant Terms

**Zero or near-zero coefficients** (candidates for removal):

| Coefficient | Value | Expected | Interpretation |
|-------------|-------|----------|----------------|
| β₂ | 0.0000030 | 0.12μs | Collapsed - either redundant or needs fixing |
| β₄ | 0.000044 | 0.37μs | Collapsed - either redundant or needs fixing |
| β₈ | 0.000045 | 45μs | In range but ineffective - wrong mechanism |

**Decision tree for iter3**:

**Option A: Fix to iter1 values** (conservative)
- β₂ = 0.00000012 (0.12μs)
- β₄ = 0.00000037 (0.37μs)
- Remove β₇, β₈ (proven ineffective)
- Optimize remaining 5 Beta terms: β₀, β₁, β₃, β₅, β₆
- **Rationale**: Constant terms may be real but unstable during optimization

**Option B: Remove collapsed terms** (aggressive)
- Remove β₂, β₄, β₇, β₈ entirely
- Optimize 5 Beta terms: β₀, β₁, β₃, β₅, β₆
- **Rationale**: If terms keep collapsing, they're not needed. Simpler model may generalize better.

**Recommendation**: Try **Option A first**. If iter3 still shows β₂, β₄ collapsing or loss not improving, switch to Option B for iter4.

---

## Recommendations for iter3

### Phase 1: Pre-Optimization (Fix Scout MoE Bugs)

**CRITICAL: DO THIS BEFORE HYPOTHESIS DESIGN**

**Step 1: Fix Scout MoE bugs** (MANDATORY - bugs confirmed, see SCOUT-MOE-BUGS.md)

Three critical bugs identified:

**Bug 1: Interleaved MoE architecture ignored**
- Location: `sim/model_hardware_config.go` (missing field), `sim/latency/roofline.go:102-105`
- Fix: Add `InterleaveMoELayerStep int` field, parse from config.json, split FLOPs/bandwidth into MoE vs dense layer calculations
- Formula: `numMoELayers = numLayers / (1 + interleaveMoELayerStep)` when interleave > 0

**Bug 2: `intermediate_size_mlp` not parsed**
- Location: `sim/latency/config.go:240`
- Fix: Add `DenseIntermediateDim int` field, parse `intermediate_size_mlp`, use for dense layers

**Bug 3: nEff expert loading applied to all layers**
- Location: `sim/latency/roofline.go:151-170`
- Fix: Apply nEff only to MoE layers (requires Bug 1 fix first)

**Validation after fix**:
```bash
# Re-run Scout experiments through fixed simulator
./blis run --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic --latency-model evolved ...

# Expected: TTFT predictions improve from 89-100% error to <30% error
# Expected: β₆ drops from 0.224 to ~0.008 after re-optimization
```

**Decision criteria**:
- If post-fix error drops to <30%: Include Scout in iter3 training (bugs were root cause)
- If post-fix error remains >50%: Additional investigation needed (may have secondary bugs)

**Step 2: Audit reasoning experiments** (STILL NEEDED - unexplained failures)
```bash
# Re-measure reasoning with cold prefix cache
blis observe --model meta-llama/Llama-2-7b-hf --no-prefix-cache \
  --workload-spec reasoning.yaml --trace-header reasoning-cold.yaml --trace-data reasoning-cold.csv

blis observe --model Qwen/Qwen2.5-7B-Instruct --no-prefix-cache \
  --workload-spec reasoning-1-1.yaml --trace-header reasoning-qwen-cold.yaml --trace-data reasoning-qwen-cold.csv

# Sanity check: theoretical minimum TTFT
# For Llama-2 reasoning (6387 tokens, 32 layers, 4096 hidden_dim, A100 80GB):
# FLOPs = 2 × 32 × 6387 × 4096 × (12 × 4096) ≈ 1.0e13 FLOPs
# Min time = FLOPs / (312 TFLOPS × 0.5 MFU) ≈ 64ms
# If observed TTFT < 64ms, data is corrupted

# Compare new observations to existing ground truth
# If cold-cache TTFT matches simulator predictions: warm cache was the issue → replace files
# If still mismatched: TTFT definition inconsistency or data corruption → exclude from training
```

**Decision criteria**:
- If re-measurement matches simulator: Replace ground truth, include reasoning in iter3
- If re-measurement still mismatches: **Exclude reasoning from iter3 training** (data unreliable)

**Step 3: Sanity check remaining experiments**

For each of the 6-8 "clean" experiments (non-Scout, non-reasoning):
```bash
# For each experiment, verify:
# observed_TTFT > theoretical_min_TTFT = (prefill_FLOPs / peak_TFLOPS / 0.5)
# If violated, data is corrupted → exclude

python -m experiment.ground_truth --sanity-check --data-dir trainval_data/
```

**Expected result**: 0-2 additional experiments flagged as corrupted. Final clean training set: **12-15 experiments** (Scout now included after bug fixes, reasoning may be excluded).

---

### Phase 2: Hypothesis Design (After Data Audit Completes)

**Constraint**: Assume clean training set has **N = 6-10 experiments** (after excluding Scout and/or reasoning).

**Model complexity target**: **5-7 free parameters** (N/2 < params < N*0.7 for healthy fit)

**Fixed parameters** (remove from optimization):
1. α₀ = 0.0002 seconds (200μs API overhead)
2. α₁ = 0.000001 seconds/token (1μs/token tokenization)
3. α₂ = 0.000002 seconds/token (2μs/token detokenization)
4. β₂ = 0.00000012 seconds (0.12μs scheduler overhead) - IF keeping, else remove
5. β₄ = 0.00000037 seconds (0.37μs KV management) - IF keeping, else remove

**Free parameters** (optimize these):
- β₀ (prefill compute efficiency): [0.3, 0.7]
- β₁ (decode memory efficiency): [0.5, 1.0]
- β₃ (TP communication): [0.2, 0.8]
- β₅ (decode compute efficiency): [0.5, 0.9]
- β₆ (MoE gating): [0.001, 0.05] - only if MoE experiments included

**Result**: 4-5 free Beta parameters + 0 Alpha parameters = **4-5 total free parameters**

**Regularization**:
```python
priors = {"beta0": 0.45, "beta1": 0.75, "beta3": 0.4, "beta5": 0.7, "beta6": 0.008}
regularization_penalty = 0.1 × sum((beta[i] - priors[i])**2 for i in range(len(beta)))
total_loss = training_loss + regularization_penalty
```

**Expected iter3 loss** (with Scout bugs fixed + reduced model + regularization):
- If 9 experiments (Scout + reasoning excluded, before bugs fixed): **<50% overall loss**
- If 12 experiments (Scout bugs fixed, reasoning excluded): **<60% overall loss** ✅ RECOMMENDED
- If 15 experiments (Scout bugs fixed, reasoning data fixed): **<70% overall loss**

---

### Phase 3: Improved Hypothesis Validation

**New requirement**: Agent 1 must provide **per-experiment predictions** in HYPOTHESIS.md:

**Example H-main format for iter3**:
```markdown
## H-main: [Hypothesis Title]

**Prediction**: Overall loss < 50%, with per-experiment breakdown:
- Llama-2 codegen: Combined loss from 54.73% to <30% (TTFT already excellent at 0.98%, E2E improves from 53.75% to <25%)
- Llama-2 roleplay: Combined loss from 72.23% to <40% (TTFT from 8.05% to <5%, E2E from 64.19% to <35%)
- [Continue for each experiment in training set]

**Causal Mechanism**: [Explain WHY each experiment improves]
- Llama-2 codegen improves because β₁ normalization fixes decode predictions
- Llama-2 roleplay improves because fixed Alpha prevents request overhead collapse
- [Continue for each experiment]

**Diagnostic Clause**: If experiment X doesn't improve by >20%, mechanism is wrong for that experiment type → investigate [specific alternative hypothesis]
```

**Why this helps**: Enables **mid-optimization validation**. After 30-50 trials, check if predicted experiments are improving. If not, abort optimization and revise hypothesis (don't waste full 78 trials).

---

### Phase 4: Optimization Configuration

**Hyperparameters**:
- `n_trials`: 100 (reduce from 250 to save compute if hypothesis is wrong)
- `convergence_patience`: 20 (stop after 20 trials without improvement)
- `regularization_lambda`: 0.1 (prevent extreme coefficient values)

**Early stopping criteria** (check after 30 trials):
- If loss > iter2 baseline (150.78%): Hypothesis wrong → abort
- If predicted experiments not improving by >10%: Mechanism wrong → abort
- If any coefficient hits bounds: Model under-constrained → expand bounds and restart

**Checkpointing**: Save intermediate results every 10 trials to enable post-hoc analysis if optimization diverges.

---

## Success Criteria for iter3

**Minimum viable success** (proceed to iter4):
- Overall loss < 100% (currently 150.78%, need 34% reduction)
- TTFT RMSE < 50% (currently 68.64%, need 27% reduction)
- E2E RMSE < 60% (currently 82.14%, need 27% reduction)
- All coefficients physically plausible (β₀ > 0.3, β₁ < 1.0, β₆ < 0.05)
- At most 2 experiments with combined loss > 120%

**Target success** (proceed to CV):
- Overall loss < 60%
- TTFT RMSE < 30%
- E2E RMSE < 35%
- All coefficients in expected ranges
- No experiment with combined loss > 100%

**Ideal success** (production-ready):
- Overall loss < 40%
- TTFT RMSE < 20%
- E2E RMSE < 20%
- All coefficients physically interpretable
- CV tests pass (CV1/CV2/CV3 all < 15-20% MAPE)

**Most likely outcome for iter3**: Minimum viable success (loss ~80-100%) IF data quality audit identifies and fixes/excludes corrupted experiments. Target success (<60%) requires both clean data AND correct basis function design.
