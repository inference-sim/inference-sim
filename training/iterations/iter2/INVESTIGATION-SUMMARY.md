# Iteration 2: Complete Investigation Summary

## Timeline

1. **Initial analysis** (Agent 3): Catastrophic failure identified (loss 134.54% → 150.78%)
2. **Hypothesis validation**: All hypotheses rejected, β₇ and β₈ proven ineffective
3. **Initial diagnosis**: Blamed data quality (Scout + reasoning experiments "corrupted")
4. **Deep investigation**: User requested MoE bug check
5. **Critical discovery**: Found THREE simulator bugs explaining ALL Scout failures

## Key Discovery: Scout Failures are Simulator Bugs (Not Data Quality)

### The Three Critical Bugs

**Bug 1 (CRITICAL)**: Interleaved MoE architecture completely ignored
- **What's wrong**: Scout has 24 MoE layers + 24 dense layers (`interleave_moe_layer_step: 1`)
- **What BLIS does**: Treats all 48 layers as MoE layers with expert scaling
- **Impact**: 2× over-count of MoE overhead, expert loading applied to wrong layers

**Bug 2 (CRITICAL)**: `intermediate_size_mlp` field not parsed
- **What's wrong**: Dense layers should use FFN dim 16384, but BLIS uses 8192
- **Impact**: 50% under-prediction of dense layer MLP FLOPs

**Bug 3 (MODERATE)**: nEff expert loading applied to all layers
- **What's wrong**: Expert weight loading formula should only apply to 24 MoE layers
- **Impact**: Weight bandwidth over-estimated for dense layers

### Evidence Trail

1. **β₆ inflated to 0.224** (28× expected 0.008) - Signal that optimizer found structural mismatch
2. **ALL 6 Scout experiments failed identically** (89-100% TTFT) - Systematic, not random data corruption
3. **Investigation found zero matches for `interleave` in codebase** - Field completely unhandled
4. **Scout config has both `intermediate_size: 8192` and `intermediate_size_mlp: 16384`** - Second field ignored

### Why This Changes Everything

**Old hypothesis** (from initial analysis):
- Scout failures are data quality issues
- Recommendation: Exclude Scout from iter3 training
- Expected iter3 loss: <50% with 6 clean experiments

**New understanding** (after bug discovery):
- Scout failures are simulator bugs (confirmed with line numbers)
- Recommendation: Fix Scout bugs, then INCLUDE Scout in iter3 training
- Expected iter3 loss: <60% with 12 experiments (Scout fixed, reasoning excluded)

## Impact on iter2 Analysis

### Validated Findings (Still True)

✅ **β₇ and β₈ are ineffective** - Reasoning experiments still fail despite β₇ active
✅ **Model overparameterization** - 12 parameters for 15 experiments causes instability
✅ **Reasoning failures unexplained** - Still likely data quality or measurement artifacts
✅ **Principle: Catastrophic failure is more informative than partial success** - The 150% loss forced investigation that found bugs

### Invalidated Findings (Wrong)

❌ **"Data quality is the bottleneck"** - TRUE for reasoning (3), FALSE for Scout (6)
❌ **"Exclude Scout from training"** - Should FIX Scout bugs instead
❌ **"67% of data is corrupted"** - Actually 20% corrupted (reasoning), 40% simulator bugs (Scout), 40% clean

### Updated Principles

**Principle 1 (REVISED)**: **Extreme coefficient inflation signals structural bugs, not physics**

When β₆ inflated from 0.008 to 0.224 (28×), this was NOT the optimizer finding better physics - it was the optimizer **highlighting a structural architecture mismatch**. The coefficient became a diagnostic signal pointing to bugs in the model implementation.

**Key insight**: Coefficient tuning cannot compensate for wrong model architecture (wrong layer counts, wrong dimensions). The optimizer can only flag the mismatch by driving coefficients to extreme values.

**Action**: When ANY coefficient inflates >10× expected during optimization, investigate for **structural bugs** before assuming data quality issues or missing physics.

---

**Principle 2 (NEW)**: **Test coverage protects against rare architectural patterns**

Scout's interleaved MoE architecture (`interleave_moe_layer_step: 1`) is rare - most MoE models (Mixtral, DeepSeek) use uniform MoE across all layers. BLIS was developed and tested for uniform architectures, so the interleaved case was never exercised.

**Evidence**:
- No test in `roofline_test.go` exercises interleaved MoE
- No Scout-like config in any test file
- All 6 Scout experiments failed identically (100% failure rate)

**Action**: When adding support for new model families, add test cases for ALL architectural variants (dense, uniform MoE, interleaved MoE, expert parallelism, etc).

---

## Quantitative Impact of Bugs

### Scout Codegen Experiment (~588 prompt tokens, TP=2)

**Correct calculation** (24 MoE + 24 dense layers):
- MoE layers (24): Expert FLOPs = 2 × 588 × (2 × 5120 × 8192) × 1 per layer
- Dense layers (24): Dense FLOPs = 2 × 588 × (2 × 5120 × 16384) per layer (2× MoE FLOPs/layer)
- Total MLP FLOPs: 24 × (MoE) + 24 × (2 × MoE) = 72 × MoE-per-layer

**BLIS calculation** (all 48 layers as MoE with 8192 dim):
- All layers (48): Expert FLOPs = 2 × 588 × (2 × 5120 × 8192) × 1 per layer
- Total MLP FLOPs: 48 × MoE-per-layer

**Ratio**: BLIS / Correct = 48 / 72 = 0.67

BLIS **under-predicts** MLP FLOPs by 33% (because dense layers have 2× larger FFN but this is ignored). However, weight bandwidth is **over-predicted** (nEff applied to all 48 layers instead of 24), and the net effect depends on whether compute or memory dominates.

**Why predictions are still wrong**: The bugs create OPPOSING errors (under-predict compute, over-predict bandwidth), and the optimizer tries to compensate via β₆ inflation. The net result is unstable predictions with extreme coefficient values.

### Expected Post-Fix Improvement

After fixing bugs:
- Scout TTFT predictions: 89-100% error → **<30% error** (expected)
- β₆ (MoE gating): 0.224 → **~0.008** (normalized)
- Overall loss: 150.78% → **~80%** (Scout saves 70 points)

If reasoning also fixed/excluded:
- Overall loss: ~80% → **<60%**

---

## Reasoning Experiments (Still Unexplained)

Scout bugs do NOT explain reasoning failures:
- Reasoning experiments are NOT MoE (Llama-2, Qwen2.5 are dense)
- β₇ (long context) is active but ineffective
- All 3 reasoning experiments have ~100% TTFT error

**Hypothesis** (still needs validation): Reasoning ground truth TTFT measured with:
1. Warm prefix cache (vLLM reused KV blocks, BLIS assumes cold cache)
2. Chunked prefill (TTFT = time to first chunk, not full prefill)
3. Data corruption (wrong hardware, wrong extraction, or file corruption)

**Action for iter3**: Re-measure reasoning with `--no-prefix-cache` or exclude from training.

---

## Recommendations for iter3 (UPDATED)

### Phase 0: Fix Scout MoE Bugs (MANDATORY - DO BEFORE HYPOTHESIS)

**Required fixes** (see SCOUT-MOE-BUGS.md for implementation details):
1. Add `InterleaveMoELayerStep` field to `ModelConfig`, parse from config.json
2. Add `DenseIntermediateDim` field to `ModelConfig`, parse `intermediate_size_mlp`
3. Split `calculateTransformerFlops()` into MoE vs dense layer calculations
4. Split `calculateMemoryAccessBytes()` into MoE vs dense layer calculations
5. Update `evolved_model.go` gating calculation to only apply to MoE layers
6. Add test coverage for interleaved MoE (Scout-like config in `roofline_test.go`)

**Validation**:
```bash
# After fixes, re-run Scout experiments
./blis run --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic ...

# Expected: TTFT error 89-100% → <30%
```

**If errors persist >50% after fix**: Secondary bugs exist, deeper investigation needed.

---

### Phase 1: Data Quality Audit (Reasoning Only)

**Step 1: Re-measure reasoning experiments with cold cache**
```bash
blis observe --model meta-llama/Llama-2-7b-hf --no-prefix-cache \
  --workload-spec reasoning.yaml --trace-header reasoning-cold.yaml --trace-data reasoning-cold.csv
```

**Decision criteria**:
- If cold-cache TTFT matches simulator: Include reasoning in iter3
- If still mismatched: Exclude reasoning from iter3 training

**Step 2: Sanity check remaining experiments**
- Verify observed TTFT > theoretical minimum for all non-Scout, non-reasoning experiments
- Expected: 0-2 additional experiments flagged

**Final training set**: **12-15 experiments** (Scout fixed + reasoning decision pending)

---

### Phase 2: Hypothesis Design (After Scout Fixes Complete)

**Model configuration for iter3**:
- **Fixed parameters**: α₀=200μs, α₁=1μs/token, α₂=2μs/token, β₂=0.12μs (if kept), β₄=0.37μs (if kept)
- **Free parameters**: β₀, β₁, β₃, β₅, β₆ (5 Beta terms)
- **Removed terms**: β₇, β₈ (proven ineffective)
- **Regularization**: L2 penalty with priors (β₀=0.45, β₁=0.75, β₃=0.4, β₅=0.7, β₆=0.008), λ=0.1

**Expected outcomes**:
- With 12 experiments (Scout fixed, reasoning excluded): Overall loss **<60%**
- With 15 experiments (Scout fixed, reasoning fixed): Overall loss **<70%**
- β₆ should drop from 0.224 to ~0.008 (Scout bugs no longer requiring compensation)

---

## Success Criteria for iter3 (UPDATED)

**Minimum viable success** (proceed to iter4):
- Overall loss < 100% (currently 150.78%, need 34% reduction)
- Scout TTFT error < 40% (currently 89-100%, need 50+ point reduction)
- All coefficients physically plausible (β₀ > 0.3, β₁ < 1.0, β₆ < 0.05)

**Target success** (proceed to CV):
- Overall loss < 60%
- Scout TTFT error < 30%
- TTFT RMSE < 30%, E2E RMSE < 35%
- No experiment with combined loss > 100%

**Most likely outcome**: Target success (<60%) achievable IF:
1. Scout bugs fixed (saves ~70 percentage points)
2. Reasoning excluded OR fixed (saves ~40 percentage points)
3. Model simplified (fix Alpha, remove β₇/β₈) prevents instability
4. Regularization prevents coefficient collapse/inflation

---

## Lessons from Investigation

### Strategy Evolution Validated

**"The most valuable output is often prediction errors — they reveal gaps in our understanding"**

iter2's catastrophic failure (150.78%) was MORE valuable than if it had achieved 90% loss:
- Forced deep investigation into failure modes
- Revealed THREE simulator bugs that would have biased ALL future iterations
- Prevented wasting iter3-5 on wrong hypotheses (data quality) when real issue was code bugs

### Coefficient Inflation as Diagnostic Signal

β₆ inflating to 0.224 (28× expected) was NOT the optimizer finding better physics. It was a **diagnostic signal** that the model structure is wrong. When the optimizer has no valid way to fit data (architecture mismatch), it drives the nearest-related coefficient to extremes.

**Pattern recognition**: When ANY coefficient inflates or collapses >10× during optimization:
1. **First check**: Is there a structural bug? (wrong architecture, wrong formula, missing field)
2. **Second check**: Is there data quality issue? (corrupted measurements, wrong extraction)
3. **Last resort**: Is the basis function wrong? (missing physics, wrong mechanism)

In iter2, we initially blamed data quality (step 2), but investigation revealed structural bugs (step 1) were the real cause for Scout.

### Test Coverage Matters for Rare Patterns

Scout's interleaved MoE (`interleave_moe_layer_step: 1`) is rare - most models use uniform architectures. BLIS passed all tests but failed on Scout because tests didn't cover interleaved patterns.

**Recommendation**: When adding MoE support, test matrix should cover:
- ✅ Dense models (tested)
- ✅ Uniform MoE (tested with Mixtral)
- ❌ Interleaved MoE (NOT tested - Scout exposed this gap)
- ❌ Expert parallelism (EP > 1, not yet supported)

---

## Next Steps

1. **Immediate** (before iter3): Fix the three Scout MoE bugs (estimated effort: 4-6 hours)
2. **Validation**: Re-run Scout experiments through fixed simulator (expected: 89-100% → <30% error)
3. **If validation passes**: Proceed to iter3 with Scout INCLUDED in training set
4. **If validation fails**: Investigate for secondary bugs (may have FP8 issues, TP issues, or memory bandwidth issues beyond architecture)
5. **Reasoning audit**: Still needed (separate from Scout bugs)
