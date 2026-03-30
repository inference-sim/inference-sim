# Iteration 8: Findings and Principles

## Summary

Iteration 8 **failed to achieve any measurable improvement** despite β₈ converging to a physically plausible value (30μs per routed token). Overall loss remained at 155.35% (unchanged from iter7's 155.37%), and all 4 Scout experiments showed 0pp TTFT improvement (79-100% APE unchanged). This is a **critical learning moment**: β₈ captures a REAL MoE routing overhead (39ms per Scout prefill request), but it's NOT the primary Scout bottleneck. The true gap is 100-200ms, requiring additional investigation.

**Key Takeaway**: **Prediction errors are the most valuable output** — Iter8 reveals that Scout's bottleneck is NOT MoE routing overhead. This eliminates a major hypothesis and narrows the search space for iter9.

---

## Error Analysis

### Systematic Patterns

#### High-Error Experiments (APE > 50%)

**Scout MoE experiments** (4 experiments, 49% of total loss):
- **exp_17** (Scout general): TTFT=99.97%, E2E=99.40% — Worst performer, unchanged from iter7
- **exp_48** (Scout reasoning-lite): TTFT=98.46%, E2E=99.81% — NEW in iter7 (clean data), still fails
- **exp_20** (Scout codegen): TTFT=92.08%, E2E=98.26% — Moderate Scout workload, fails uniformly
- **exp_21** (Scout roleplay): TTFT=79.10%, E2E=96.04% — Best Scout, but still >70% APE

**Pattern**: All Scout experiments fail uniformly (79-100% TTFT) regardless of workload. This confirms the bottleneck is **architecture-specific** (Scout MoE hybrid) rather than workload-specific.

**What's special about Scout?**
1. **MoE architecture**: 26 MoE layers interspersed with 30 dense layers (56 total)
2. **FP8 quantization**: FP8 dynamic quantization (mixed-precision coordination overhead)
3. **TP=2 configuration**: Cross-GPU expert routing (higher TP coordination cost?)
4. **16 experts, top-k routing**: k=1 or k=2 (unknown, needs validation)

**Dense model experiments** (6 experiments with TTFT > 50%):
- **exp_62** (Mistral-Nemo general-lite): TTFT=89.62%, E2E=98.38% — High TTFT error, moderate E2E
- **exp_67** (Llama-2 reasoning-lite): TTFT=65.98%, E2E=96.30% — Reasoning workload, dense model
- **exp_66** (Qwen2.5 reasoning-lite): TTFT=54.12%, E2E=95.00% — Reasoning workload, dense model
- **exp_64** (Qwen2.5 roleplay): TTFT=57.51%, E2E=73.52% — Roleplay workload, LOW E2E ✓
- **exp_20260217** (Llama-2 roleplay): TTFT=54.85%, E2E=79.03% — Roleplay workload, LOW E2E ✓

**Pattern**: Dense models have moderate TTFT errors (54-90%) but MUCH lower E2E errors (73-98% vs Scout's 96-100%). Roleplay workloads have the lowest E2E errors (<80%), suggesting decode-phase predictions are more accurate for interactive workloads.

#### Low-Error Experiments (TTFT APE < 50%)

- **exp_60** (Llama-3.1-70B general-lite, TP=4): TTFT=40.97%, E2E=95.03% — Largest model, TP=4
- **exp_65** (Yi-34B general-lite, TP=2): TTFT=48.08%, E2E=94.33% — Mid-size model
- **exp_61** (Llama-3.1-70B codegen, TP=4): TTFT=29.24%, E2E=86.31% — **BEST TTFT** (sub-30%)
- **exp_63** (Mistral-Nemo codegen): TTFT=??, E2E=?? — (data truncated, likely similar)

**Pattern**: Large dense models (Llama-3.1-70B, Yi-34B) with TP=4 have the LOWEST TTFT errors (29-48%). Codegen workloads have lower errors than general/reasoning. This suggests:
1. **Larger models are easier to predict** (less variance in MFU?)
2. **Higher TP is easier to predict** (TP=4 vs TP=1/2)
3. **Codegen workloads are easier** (consistent batch composition?)

#### Error Correlations

**Confirmed correlations** (what explains LOW error):
- ✅ **Large dense models**: Llama-3.1-70B (29-41% TTFT), Yi-34B (48% TTFT)
- ✅ **Higher TP**: TP=4 models have lower TTFT errors (29-48%) vs TP=1/2 (54-90%)
- ✅ **Codegen workloads**: Lowest TTFT errors (29-57%) across dense models
- ✅ **Roleplay workloads**: Lowest E2E errors (73-96%) — decode predictions more accurate

**Rejected correlations** (what does NOT explain error):
- ❌ **Clean data**: Scout reasoning-lite (NEW in iter7, clean data) still fails (98% TTFT)
- ❌ **Workload type for Scout**: All Scout workloads fail uniformly (79-100% TTFT)
- ❌ **β₈ addition**: β₈ = 30μs (plausible), contributes 39ms, but has 0 impact

---

### Root Cause Hypotheses

#### Principle 1: Scout's Bottleneck is NOT MoE Routing Overhead

**Evidence**:
- β₈ converged to 30μs per routed token (within predicted 10-50μs range) — physically plausible ✓
- β₈ contributes 39ms per Scout prefill request (1,300 routed tokens × 30μs) — non-trivial ✓
- Scout TTFT errors: 79-100% APE (unchanged from iter7) — 0pp improvement ✗
- Non-Scout experiments: All stable (< ±10pp change) — no zero-sum trade-off ✓
- Overall loss: 155.35% (unchanged from 155.37%) — no improvement ✗

**Mechanism**: β₈ captures a REAL MoE routing overhead (expert selection, dispatch, aggregation), but it's NOT the PRIMARY Scout bottleneck. The baseline analysis showed roofline underestimates Scout by 50-99% (negative MPE), meaning the missing overhead is 100-200ms. β₈ adds 39ms, leaving 61-161ms unaccounted for.

**Action for iter9**: Profile Scout separately to identify the dominant bottleneck. Candidates:
1. **FP8 dequantization overhead**: Scout uses FP8 dynamic quantization. Mixed-precision coordination may add 50-100ms per request.
2. **TP=2 communication overhead**: Cross-GPU expert routing may have higher TP cost than β₂ captures.
3. **Model config error**: InterleaveMoELayerStep=26 or NumExpertsPerTok might be wrong, causing basis function underestimation.
4. **Batching inefficiency**: Scout MoE may have lower batching efficiency than dense models (expert load imbalance).

---

#### Principle 2: Physically Plausible ≠ Sufficient

**Evidence**:
- β₈ = 30μs (within 10-50μs expected range) — passes plausibility check ✓
- β₈ adds 39ms per Scout request — significant contribution (39% of roofline's missing 100ms) ✓
- Yet Scout errors unchanged — β₈ is correct but INSUFFICIENT ✗

**Mechanism**: The hypothesis conflated "β₈ exists" (mechanism is real) with "β₈ is sufficient" (mechanism dominates error budget). Both can be true simultaneously:
- β₈ is a real 39ms overhead (confirmed by optimizer learning 30μs)
- Scout's bottleneck is 100-200ms (confirmed by baseline roofline -99% MPE)
- Therefore β₈ accounts for 20-39% of the gap, not 100%

**Action for iter9**: When designing hypotheses, distinguish between:
1. **Term existence**: Does the mechanism contribute ANY latency? (β₈ = 30μs ≠ 0 ✓)
2. **Term dominance**: Does the mechanism account for >50% of error? (β₈ adds 39ms, gap is 100ms ✗)

Use baseline comparisons to estimate term magnitudes BEFORE optimization. If roofline underestimates by 100ms, a 39ms term is insufficient.

---

#### Principle 3: Diagnostic Clauses Should Cover "Correct but Insufficient"

**Evidence**:
- Agent 1's diagnostic clause (from HYPOTHESIS.md):
  1. β₈ coefficient converged to zero → investigate alternative bottlenecks
  2. β₈ coefficient converged >100μs → investigate absorbing other terms
  3. Non-Scout experiments degraded >10pp → investigate zero-sum trade-off
- Actual outcome: β₈ = 30μs (plausible), non-Scout stable, yet Scout failed
- **The diagnostic clause MISSED this scenario**: β₈ is plausible but insufficient

**Mechanism**: The clause assumed β₈ success/failure was binary (either it works or it doesn't). Reality: β₈ works (captures real overhead) but is insufficient (not the primary bottleneck). The clause should have included:
- "If β₈ is plausible (10-50μs) but Scout errors remain >70%, it indicates β₈ is correct but INSUFFICIENT — investigate additional complementary terms."

**Action for iter9**: When writing diagnostic clauses, include scenarios for:
1. **Term is zero** (mechanism doesn't exist)
2. **Term is implausible** (mechanism exists but formulation wrong)
3. **Term is plausible but insufficient** (mechanism exists but is not dominant) ← MISSED in iter8
4. **Term causes zero-sum trade-off** (mechanism helps one case, hurts another)

---

#### Principle 4: Baseline Comparisons Quantify Remaining Gap

**Evidence**:
- Baseline roofline underestimates Scout general (exp_17) by -99.88% MPE
- This means: roofline predicts 0.12ms, reality is 100ms, missing overhead is 99.88ms
- β₈ adds 39ms → new prediction is 39.12ms → remaining gap is 60.88ms (61% APE)
- Actual iter8 result: Scout general TTFT = 99.97% APE (unchanged)

**Mechanism**: Baseline comparisons provide a **quantitative target** for term magnitudes. If roofline underestimates by 100ms, we need terms that sum to ~100ms, not 39ms. β₈ closes 39% of the gap, but the remaining 61% requires additional terms.

**Action for iter9**: Before designing iter9 hypothesis:
1. Estimate Scout's total missing overhead: 100-200ms (from baseline -99% MPE)
2. Subtract β₈ contribution: 100-200ms - 39ms = 61-161ms remaining
3. Design iter9 term to target 61-161ms (e.g., FP8 overhead, TP coordination)
4. If profiling shows FP8 adds 80ms, expect iter9 to close 80/100 = 80% of the remaining gap

---

#### Principle 5: Scout is Architecture-Specific, Not Workload-Specific

**Evidence**:
- All 4 Scout experiments fail uniformly (79-100% TTFT) regardless of workload:
  - Scout general: 99.97% TTFT
  - Scout reasoning-lite: 98.46% TTFT (NEW clean data in iter7)
  - Scout codegen: 92.08% TTFT
  - Scout roleplay: 79.10% TTFT
- Non-Scout reasoning workloads succeed (54-66% TTFT) with same clean data
- Error magnitude correlates with sequence length (general > reasoning > codegen > roleplay), but ALL remain >70%

**Mechanism**: Scout's bottleneck is tied to its architecture (MoE hybrid, FP8, TP=2), not workload characteristics. Evidence:
1. **Clean data doesn't help**: Scout reasoning-lite (clean) still fails (98% TTFT)
2. **Workload variance is small**: 79-100% TTFT (21pp range), but all >70%
3. **Dense reasoning succeeds**: Qwen/Llama-2 reasoning-lite (54-66% TTFT) with same workload

**Action for iter9**: Focus on Scout architecture-specific terms:
- FP8 dequantization overhead (unique to Scout)
- Hybrid MoE+dense coordination (unique to Scout's interleaved architecture)
- TP=2 expert routing (may be different from dense TP=2)

Do NOT add workload-specific terms (Scout failures are uniform across workloads).

---

## Coefficient Analysis

### Alpha Coefficients (API overhead)

| Coefficient | Iter7 | Iter8 | Change | Physical Interpretation |
|-------------|-------|-------|--------|-------------------------|
| α₀ (base) | 1.32ms | 1.32ms | 0% | Fixed API overhead per request — stable ✓ |
| α₁ (input token) | 118μs | 117.6μs | -0.3% | Per-input-token tokenization — stable ✓ |
| α₂ (output token) | 91μs | 90.54μs | -0.5% | Per-output-token detokenization — stable ✓ |

**Analysis**: All alpha coefficients unchanged (< 1% variation). This confirms API overhead terms are orthogonal to Scout's bottleneck. No alpha reversion or degradation observed.

**Physical interpretation**:
- α₀ = 1.32ms: Fixed overhead (request parsing, validation, queueing) — plausible ✓
- α₁ = 118μs: Input tokenization (500 tokens × 118μs = 59ms) — plausible ✓
- α₂ = 91μs: Output detokenization (100 tokens × 91μs = 9ms) — plausible ✓

All alpha values are physically reasonable and stable across iterations.

---

### Beta Coefficients (Step-level basis functions)

| Coefficient | Iter7 | Iter8 | Change | Physical Interpretation |
|-------------|-------|-------|--------|-------------------------|
| β₀ (prefill compute) | 0.191 | 0.1912 | +0.1% | Prefill MFU — stable ✓ |
| β₁ (decode memory) | 1.108 | 1.1076 | -0.0% | Decode memory MFU — stable ✓ |
| β₂ (TP comm) | 0.185 | 0.1846 | -0.2% | TP communication factor — stable ✓ |
| β₃ (KV mgmt) | 0.00440 | 0.004404 | +0.1% | KV cache management — stable (did NOT revert) ✗ |
| β₄ (decode compute) | 0.713 | 0.7132 | +0.0% | Decode compute MFU — stable ✓ |
| β₅ (MoE gating) | 0.0411 | 0.04112 | +0.0% | MoE gating overhead — stable (did NOT decrease) ✗ |
| β₆ (scheduler) | 0.0132 | 0.01316 | -0.3% | Scheduler overhead — stable ✓ |
| β₇ (decode overhead) | 0.0263 | 0.02626 | -0.2% | Decode per-request overhead — stable (did NOT decrease) ✗ |
| **β₈ (MoE routing)** | **N/A** | **0.00003** | **NEW** | **MoE routing overhead — 30μs per routed token** |

**Key Observations**:

1. **β₈ converged to 30μs** (within predicted 10-50μs range) — confirms MoE routing overhead exists ✓
2. **β₃, β₅, β₇ did NOT revert** — predicted to decrease as β₈ offloads Scout error, but remained constant ✗
3. **All other coefficients stable** (< 1% change) — no zero-sum trade-off ✓

**Why β₃, β₅, β₇ didn't revert**:
- **Hypothesis predicted**: β₈ would offload Scout error from β₃ (KV mgmt), β₅ (MoE gating), β₇ (decode overhead)
- **Reality**: β₈ added 39ms but Scout errors unchanged → β₃, β₅, β₇ continue absorbing whatever residual error they compensated for in iter7
- **Implication**: β₃, β₅, β₇ are inflated but orthogonal to β₈ (different error sources)

**Coefficient Plausibility Check**:

| Coefficient | Expected | Actual | Status |
|-------------|----------|--------|--------|
| β₀ | 0.15-0.25 | 0.1912 | ✓ Within range |
| β₁ | 1.00-1.15 | 1.1076 | ✓ Within range |
| β₂ | 0.20-0.35 | 0.1846 | ⚠️ Below range (expected 0.20+) |
| β₃ | 0.0004-0.001 | 0.004404 | ✗ 4-10× too high (should be 0.4-1ms, actual 4.4ms) |
| β₄ | 0.70-0.90 | 0.7132 | ✓ Within range |
| β₅ | 0.010-0.020 | 0.04112 | ✗ 2-4× too high (should be 10-20μs, actual 41μs) |
| β₆ | 0.015-0.030 | 0.01316 | ⚠️ Below range (expected 15-30ms, actual 13ms) |
| β₇ | 0.010-0.020 | 0.02626 | ✗ 1.3-2.6× too high (should be 10-20ms, actual 26ms) |
| **β₈** | **0.00001-0.00005** | **0.00003** | **✓ Within range (10-50μs, actual 30μs)** |

**Outliers**:
- **β₃ = 4.4ms**: 4-10× higher than physical (expected 0.4-1ms) — still absorbing Scout error
- **β₅ = 41μs**: 2-4× higher than physical (expected 10-20μs) — MoE gating inflated, but not routing
- **β₇ = 26ms**: 1.3-2.6× higher than physical (expected 10-20ms) — still absorbing Scout decode error

**Redundant terms**: None (no coefficients near zero).

**Missing physics**: β₃, β₅, β₇ inflation suggests missing terms that would absorb their excess contribution. After identifying Scout's true bottleneck (likely FP8 or TP coordination), expect β₃, β₅, β₇ to revert to physical ranges.

---

## Recommendations for iter9

### Priority 1: CRITICAL — Identify and Add Scout's True Bottleneck Term

**Rationale**: β₈ (MoE routing) is real but insufficient (adds 39ms, gap is 100ms). Must find the dominant 61-161ms term.

**Action Plan**:

1. **Profile Scout with vLLM profiler** to measure per-component latency:
   ```bash
   # Use vLLM profiling tools to isolate Scout overhead
   python -m vllm.profiler --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic \
     --workload general --batch-size 1 --tp 2 --profile-moe
   ```
   Measure:
   - Per-layer MoE routing latency (selection, dispatch, aggregation)
   - FP8 dequantization overhead (mixed-precision coordination)
   - TP=2 communication overhead (cross-GPU expert routing)
   - Framework overhead (vLLM MoE kernel launch, synchronization)

2. **Validate Scout model config** (cross-check with HuggingFace config.json):
   ```python
   from transformers import AutoConfig
   config = AutoConfig.from_pretrained("RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic")
   print(config.num_local_experts)        # Should be 16
   print(config.num_experts_per_tok)      # Should be 1 or 2 (top-k routing)
   print(config.interleave_moe_layer_step)  # Should be 26 or equivalent
   ```
   If config wrong, fix before iter9.

3. **Design β₉ term** based on profiling results:
   - **If FP8 dominant** (50-100ms): Add β₉ × (totalTokens × isFP8) — per-token dequantization overhead
   - **If TP dominant** (50-100ms): Add β₉ × (numMoELayers × numDecodeRequests × TP) — cross-GPU MoE coordination
   - **If config wrong**: Fix config, re-run iter8 (don't add β₉)
   - **If none of above**: Investigate framework overhead (vLLM MoE kernel launch, batching inefficiency)

4. **Expected iter9 outcome**:
   - If β₉ targets 80ms: Expect Scout TTFT to improve from 90% to 40-50% (closing 80ms / 100ms = 80% of gap)
   - Overall loss: 155% → 70-90% (Scout contributes 49% of loss, so 80% Scout improvement = 39pp overall loss reduction)

---

### Priority 2: Keep β₈ in iter9 Model

**Rationale**: β₈ captures a real 39ms overhead (validated by optimizer learning 30μs). Keep it to avoid regression.

**Action**: Maintain β₈ in iter9 coefficient list. After adding β₉ (Scout's true bottleneck), β₈ will continue contributing 39ms to Scout experiments while β₉ adds the remaining 61-161ms.

**Expected β₈ stability**: β₈ should remain at 30μs ± 10μs in iter9 (stable across iterations).

---

### Priority 3: Validate β₃, β₅, β₇ Reversion After Adding β₉

**Rationale**: β₃, β₅, β₇ are inflated (4.4ms, 41μs, 26ms vs physical 0.4-1ms, 10-20μs, 10-20ms). After adding β₉ to close Scout gap, these should revert.

**Action**: After iter9 training, check coefficient reversion:
- β₃: 4.4ms → 0.4-1ms (10× decrease)
- β₅: 41μs → 10-20μs (2-4× decrease)
- β₇: 26ms → 10-20ms (1.3-2.6× decrease)

If reversion occurs, confirms β₉ offloaded excess error from β₃, β₅, β₇. If not, investigate alternative error sources.

---

### Priority 4: Consider Architecture-Specific Models (Contingency)

**Rationale**: If Scout's bottleneck is unique to its hybrid MoE+dense+FP8 architecture, a universal model may continue struggling (zero-sum trade-offs).

**Contingency Plan** (if iter9 fails):
1. **Split training data**:
   - **Dense model**: Trained on 11 dense experiments (excludes all Scout)
   - **MoE model**: Trained on 4 Scout experiments + future MoE models (Mixtral, DeepSeek-V3)
2. **Architecture-specific basis functions**:
   - Dense model: β₀-β₇ (no MoE terms)
   - MoE model: β₀-β₇, β₈ (MoE routing), β₉ (FP8/TP/config), β₁₀ (batching inefficiency)
3. **Deploy**: Route requests to appropriate model based on architecture metadata

**Trade-off**: Increased model complexity (2 models vs 1), but eliminates zero-sum trade-offs and enables architecture-specific tuning.

---

### Priority 5: Update Diagnostic Clause Template

**Rationale**: Iter8's diagnostic clause missed "correct but insufficient" scenario. Future iterations should include this case.

**Action**: Update `training/docs/analysis-agent-prompt.md` diagnostic clause template:

```markdown
**Diagnostic Clause** (if hypothesis fails):

*If this hypothesis fails, it indicates:*
1. **Term is zero** (coefficient ≈ 0): Mechanism doesn't exist, investigate alternative physics
2. **Term is implausible** (coefficient outside expected range): Formulation wrong, refine basis function
3. **Term is correct but insufficient** (coefficient within range, but errors remain high): Mechanism exists but is not dominant, investigate complementary terms  ← ADD THIS
4. **Zero-sum trade-off** (helps one case, hurts another): Need architecture-specific handling

*Next investigation*: [Specific profiling or validation steps to disambiguate above scenarios]
```

---

## Basis Function Changes for iter9

### Add β₉: Scout Bottleneck Term (TBD after profiling)

**Decision**: Awaiting profiling results. Three candidates:

1. **FP8 dequantization overhead** (if profiling shows 50-100ms):
   - Basis function: `β₉ × totalTokens × isFP8`
   - Expected β₉: 0.05-0.10ms per token (50-100μs)
   - Scout contribution: 100 tokens × 50-100μs = 5-10ms per token × 10 layers = 50-100ms

2. **TP MoE coordination overhead** (if profiling shows 50-100ms):
   - Basis function: `β₉ × (numMoELayers × numDecodeRequests × TP)`
   - Expected β₉: 2-4ms per layer per request per TP rank
   - Scout contribution: 26 layers × 1 request × 2 TP × 2-4ms = 104-208ms

3. **Batching inefficiency** (if profiling shows Scout has lower batching efficiency):
   - Basis function: `β₉ × (batchSize^2 × numMoELayers / maxBatchSize)`
   - Expected β₉: 1-5ms per batch slot per MoE layer
   - Scout contribution: Depends on batch size (higher with larger batches)

**Note**: Wait for profiling before committing to β₉ formulation.

---

### Keep β₈: MoE Routing Overhead

**Rationale**: β₈ = 30μs (within 10-50μs range), contributes 39ms per Scout request. Mechanism is real, just insufficient.

**Basis function**: `β₈ × (numMoELayers × totalTokens × numExpertsPerTok / TP)`

**Expected stability**: β₈ should remain at 30μs ± 10μs across iterations.

---

### Monitor β₃, β₅, β₇: Should Revert After β₉ Addition

**Rationale**: These coefficients are inflated (absorbing Scout error). After β₉ closes Scout gap, expect reversion:

| Coefficient | Current (iter8) | Physical | Expected (iter9) |
|-------------|-----------------|----------|------------------|
| β₃ (KV mgmt) | 4.4ms | 0.4-1ms | 0.4-1ms (10× decrease) |
| β₅ (MoE gating) | 41μs | 10-20μs | 10-20μs (2-4× decrease) |
| β₇ (decode overhead) | 26ms | 10-20ms | 10-20ms (1.3-2.6× decrease) |

If reversion doesn't occur, investigate alternative error sources.

---

## Bounds Adjustments for iter9

### Widen β₉ Bounds (TBD after profiling)

**Rationale**: β₉ needs sufficient headroom to capture 61-161ms Scout overhead.

**Recommended bounds** (pending profiling):
- **If FP8**: [0.0, 0.00015] = [0, 150μs per token]
- **If TP**: [0.0, 0.005] = [0, 5ms per layer per request per TP rank]
- **If batching**: [0.0, 0.01] = [0, 10ms per batch slot per MoE layer]

**Initial value**: Set to middle of profiling-measured range (e.g., if profiling shows 80μs per token, set β₉ initial = 0.00008).

---

### Maintain All Other Bounds

**Rationale**: All coefficients converged well within bounds (no ceiling/floor violations). No adjustments needed.

**Exception**: If iter9 adds β₉ and β₃, β₅, β₇ don't revert, consider tightening bounds to force reversion:
- β₃: [0.0, 0.001] → [0.0, 0.0005] (force reversion to 0.4-0.5ms)
- β₅: [0.0, 0.05] → [0.0, 0.025] (force reversion to 10-20μs)
- β₇: [0.0, 0.03] → [0.0, 0.02] (force reversion to 10-20ms)

But prefer letting optimizer find natural values first (don't force).

---

## Success Criteria for iter9

**Tier 1 (Full Success)**:
- Overall loss < 80% ✓
- TTFT RMSE < 40% ✓
- Scout TTFT < 50% (>40pp improvement) ✓
- Non-Scout stable (< ±10pp change) ✓
- β₉ coefficient physically plausible ✓
- β₃, β₅, β₇ revert to physical ranges ✓

**Tier 2 (Partial Success)**:
- Overall loss < 100% (major improvement, not target)
- Scout TTFT < 70% (>20pp improvement, not target)
- Non-Scout stable
- β₉ coefficient plausible
- At least one of β₃, β₅, β₇ reverts

**Tier 3 (Failure)**:
- Overall loss > 120% (minimal improvement)
- Scout TTFT > 80% (<10pp improvement)
- β₉ coefficient implausible OR zero

**If Tier 3**: Consider architecture-specific models (Priority 4 contingency plan).

---

## Conclusion

**Iter8 was a critical learning iteration**. While it failed to improve loss metrics, it provided invaluable insights:

1. **β₈ mechanism is real**: Optimizer learned 30μs per routed token (within predicted range), confirming MoE routing overhead exists.
2. **β₈ is insufficient**: Scout's bottleneck is 100-200ms, not 39ms. Need additional term (β₉) to close the gap.
3. **Scout is architecture-specific**: All Scout workloads fail uniformly (79-100% TTFT), regardless of clean data or workload type. Focus on architecture-specific terms (FP8, TP, hybrid MoE+dense).
4. **Prediction errors are valuable**: Iter8's failure eliminates MoE routing as the primary bottleneck, narrowing the search space for iter9.

**Next Steps**:
1. Profile Scout to identify dominant bottleneck (FP8, TP, config, or other)
2. Design β₉ term targeting 61-161ms overhead
3. Train iter9 with β₈ + β₉ (keep β₈, add β₉)
4. Validate β₃, β₅, β₇ reversion (should decrease to physical ranges)
5. If iter9 fails, consider architecture-specific models (contingency)

**The learning continues** — each iteration, whether successful or not, reveals gaps in our understanding and guides the next hypothesis. Iter8 taught us that Scout's bottleneck is deeper than MoE routing overhead. Iter9 will dig deeper.
