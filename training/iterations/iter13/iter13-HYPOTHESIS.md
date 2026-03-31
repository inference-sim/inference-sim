# Iteration 13: Return to Iter7 Baseline + Sequence-Length Term

## Executive Summary

**Strategy**: Start from iter7 stable baseline (155% loss) + add β₈ (Scout MoE routing, iter8) + add β₁₀ (batching inefficiency, addresses iter9's sequence-length discovery).

**Critical Learning from Iter9-12 Failures**: The catastrophic failures (161% → 4267% → 4084% → 2590%) were caused by:
1. **Warm-starting from inflated coefficients** (iter9 had β₆=99ms, β₂=0.82, β₈=73μs)
2. **Collinearity** between β₁₀ and β₃' (both scale with sequence length)
3. **Missing the stable foundation** - iter7 (155%) was more stable than iter6 (162%)

**Iter13 Design**:
- **Architecture**: 3 alpha + 10 beta (β₀-β₇ from iter7, β₈ from iter8, β₁₀ new)
- **Warm-start**: Iter7 optimal coefficients (NOT iter9-12 inflated values)
- **Key addition**: β₁₀ (batching inefficiency) addresses iter9's sequence-length discovery
- **Key omission**: NO β₃' (avoids collinearity with β₁₀)

**Expected Outcome**: Loss 155% → 120-140% (15-35pp improvement, stable progress)

---

## H-main: Iter7 Baseline + β₁₀ Recovers Sequence-Length Prediction

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
- β₀: 0.191 → **0.16-0.22** (stable, may decrease slightly)
- β₁: 1.108 → **1.00-1.15** (stable, already in range)
- β₂: 0.185 → **0.15-0.25** (stable, already in range)
- β₃: 4.4ms → **0.4-1.5ms** (expect decrease after β₁₀ offloads overhead)
- β₄: 0.713 → **0.70-0.85** (stable)
- β₅: 0.0411 → **300-1000** dimensionless (expect dramatic increase 7300-24000×, was collapsed in iter7)
- β₆: 13.2ms → **40-100ms** (may increase, iter7 value too low for cold-start overhead)
- β₇: 26.3ms → **15-30ms** (stable or slight decrease)
- β₈: 30μs → **25-80μs** (stable, captures Scout MoE routing)
- β₁₀: NEW → **0.1-1.0μs** per (token²/batch) (new term)

### Causal Mechanism

**Why Iter7 is the Right Baseline**:

Iter7 (155% loss) > Iter6 (162% loss) because:
1. **β₇ stabilized decode coefficients**: β₁ (1.851 → 1.108), β₄ (1.451 → 0.713)
2. **Lower loss**: 7pp better than iter6
3. **Has β₈ from iter8**: Captures real 39ms Scout MoE routing mechanism

**Why Iter9-12 Failed**:

1. **Iter9**: Added β₉ (FP8), rejected → triggered coefficient explosions:
   - β₆: 13ms → 99ms (+654%)
   - β₂: 0.18 → 0.82 (+343%)
   - β₈: 30μs → 73μs (+143%)

2. **Iter10-12**: Started from iter9's inflated coefficients → optimizer trapped in bad local optimum → catastrophic losses (4267%, 4084%, 2590%)

**Why Iter13 Will Succeed**:

1. **Clean warm-start**: Use iter7's stable coefficients (not iter9-12 inflated)
2. **Proven architecture**: Keep all iter7 terms (β₀-β₇) + iter8's β₈
3. **Address sequence-length discovery**: Add β₁₀ (batching inefficiency from iter9's key insight)
4. **Avoid collinearity**: Do NOT add β₃' (would fight with β₁₀ for same error)

**β₁₀ Mechanism** (batching inefficiency):

Long sequences consume disproportionate batch capacity:
- Batch constraint: Σ(prefill_tokens + kv_cache_blocks) ≤ max_num_batched_tokens
- Long sequences (500 tokens) consume 10× more than short (50 tokens)
- Fewer requests per batch → lower GPU utilization → longer queue waits

Formula: `β₁₀ × Σ(prefillTokens² / batchSize)`
- Quadratic scaling: 500² = 250,000 vs 50² = 2,500 (100× difference)
- Division by batchSize: Amplifies effect for long sequences (lower batch efficiency → smaller denominator)

**Expected β₁₀ contribution**:
- Scout general-lite (500 tokens, batch_size=4): β₁₀ × (500²/4) = 0.5μs × 62,500 ≈ 31ms
- Scout roleplay (100 tokens, batch_size=32): β₁₀ × (100²/32) = 0.5μs × 312 ≈ 0.16ms
- Ratio: 194× difference (matches observed long-sequence vs short-sequence error pattern)

### Code Citations

**Iter7 stable baseline**:
- `training/iterations/iter7/inner_loop_results.json`: Loss 155.37%, coefficients proven stable
- β₇ (decode overhead) stabilized β₁/β₄: `iter7-FINDINGS.md` line ~30-50

**Iter8 β₈ captures real Scout mechanism**:
- `training/iterations/iter8/inner_loop_results.json`: β₈ = 30μs, captures 39ms Scout contribution
- 0pp improvement overall but mechanism is real: `iter8-FINDINGS.md`

**Iter9 sequence-length discovery**:
- Scout short-sequence improved: roleplay -53pp, codegen -34pp
- Scout long-sequence failed: general-lite 0pp, reasoning-lite -8pp
- **Inverse correlation**: Longer sequences → worse performance
- Evidence: `training/iterations/iter9/iter9-FINDINGS.md` line ~20-80

**Iter10 β₁₀ formula validated**:
- Unit tests: 0% error (iter11 audit)
- Basis function implementation CORRECT
- `training/iterations/iter11/iter11-HYPOTHESIS-validation.md` line ~200-250

**Why warm-start from iter7 NOT iter9**:
- Iter9 had inflated coefficients after β₉ rejection
- Iter10-12 started from inflated values → trapped in bad optimum
- `training/iterations/TRAINING_JOURNEY.md` line ~100-150

### Diagnostic Clause

**If H-main is refuted (loss does NOT improve to <140%)**:

**Scenario 1**: Loss improves modestly (155% → 145-150%, 5-10pp)
- **Indicates**: β₁₀ helps but insufficient, need additional terms
- **Investigate**:
  1. Check coefficient convergence: Are all 10 coefficients within expected ranges?
  2. Which long-sequence experiments still fail (>80% TTFT)?
  3. Is β₅ still anomalously high (>1ms vs expected 15-30μs)?
- **Next steps**: If β₅ >1ms, investigate MoE gating formula; if Scout still fails, consider adding β₃' with very wide bounds [0.05-5.0μs]

**Scenario 2**: Loss worsens (155% → >160%)
- **Indicates**: β₁₀ causing collinearity or formula error
- **Investigate**:
  1. Did β₁₀ converge to expected 0.1-1.0μs range?
  2. Did other coefficients explode (β₃, β₆, β₂)?
  3. Check β₁₀ contribution: Does it scale correctly with sequence length?
- **Next steps**: If β₁₀ formula wrong, audit implementation; if collinearity, remove β₁₀ and try β₃' alone in iter14

**Scenario 3**: β₅ stays >10ms (40-100× above expected 15-30μs)
- **Indicates**: MoE gating formula wrong OR absorbing unintended overhead
- **Investigate**:
  1. Profile Scout experiments: What is actual MoE gating time?
  2. Check basis function: `2 × tokens × hiddenDim × numExperts` may be wrong
  3. Is β₅ absorbing decode per-request overhead (β₇ too low)?
- **Next steps**: Reformulate β₅ with corrected FLOPs OR increase β₇ bounds to decouple

**Scenario 4**: β₆ stays <20ms (below profiling-observed 40-100ms range)
- **Indicates**: Scheduler overhead is being absorbed elsewhere (β₁₀, β₃, or Alpha)
- **Investigate**:
  1. Check Alpha values: Did α₀ or α₁ inflate to compensate?
  2. Check β₁₀ contribution: Is it capturing scheduler batching delays?
  3. Profiling data shows 40-100ms cold-start overhead - where is it going?
- **Next steps**: Widen β₆ bounds to [0.04, 0.15] (40-150ms) if profiling evidence strong

**Scenario 5**: Long-sequence experiments DON'T improve (<5pp gain)
- **Indicates**: Batching inefficiency is NOT the bottleneck OR β₁₀ formula inadequate
- **Investigate**:
  1. What mechanism causes long-sequence failure? Memory bandwidth? KV cache pressure?
  2. Check baseline simulators: How does vidur handle long sequences?
  3. Is it actually a Scout-specific problem (MoE + long-seq interaction)?
- **Next steps**: Add β₃' (KV seq-len scaling) with VERY wide bounds [0.05-5.0μs] to capture bandwidth saturation

**Success Criteria** (if H-main is confirmed):
- Overall loss: **<140%** (15pp improvement from iter7)
- Long-sequence experiments: **≥4/5 improve by ≥12pp** (Scout/Mistral/Yi/Llama-3.1 general-lite)
- Coefficient stability: **≥8/10 within expected ranges** (no explosions)
- Foundation for iter14: Stable baseline for incremental β₃' addition if needed

---

## H-beta5-anomaly: β₅ Will Increase 9000-24000× to Recover from Collapse

### Prediction

Iter7 β₅ = 0.0411 (dimensionless) collapsed to essentially zero contribution.

**Dimensional analysis**:
- β₅ is dimensionless scaling factor (multiplies roofline estimate)
- Roofline estimate for Scout (100 tokens, single layer): ~0.04μs
- Iter7 contribution: 0.0411 × 0.04μs = **0.0016μs ≈ ZERO**
- Expected contribution: 15-30μs per step (accounting for all MoE layers)
- Required β₅: (15-30μs) / (0.04μs) = **375-750 dimensionless**

After adding β₈ (MoE routing) and β₁₀ (batching inefficiency):
- β₅: 0.0411 (iter7) → **300-1000 dimensionless** (9000-24000× increase)

**Why β₅ collapsed in iter7**:
- Iter7 had 4 Scout experiments with 79-100% TTFT APE (severe UNDERprediction)
- Model predicted too LOW, but had no mechanism to capture MoE overhead
- β₅ collapsed to near-zero (0.0411) because missing overhead (β₈ routing, β₁₀ batching) couldn't be captured elsewhere
- The optimization couldn't fit Scout data, so β₅ gave up

### Causal Mechanism

**MoE gating physics** (what β₅ should capture):

Gating network computes routing probabilities for all experts:
- FLOPs per layer: `2 × tokens × hiddenDim × numExperts` (linear projection)
- For Scout (100 tokens): `2 × 100 × 3584 × 16 = 11.5M FLOPs per layer`
- At 30% MFU: `11.5M / (989 TFLOPS × 0.3) = 0.0388μs per layer`
- Scout has ~56 MoE layers, but basis function only computes single-layer time
- **β₅ must scale to account for ALL MoE layers**: β₅ ≈ 56 layers × (overhead factor)

**Why β₅ = 0.0411 is anomalous**:

With β₅ = 0.0411:
- Contribution per step: 0.0411 × 0.04μs = 0.0016μs (essentially ZERO)
- This explains Scout's 79-100% TTFT underprediction → no MoE overhead captured!

**Why iter13 will recover β₅**:

Adding β₈ (MoE routing) and β₁₀ (batching inefficiency) captures missing overhead:
- β₈ captures per-token expert routing (39ms per Scout prefill from iter8)
- β₁₀ captures long-sequence queue delays (31ms for 500-token Scout)
- With these in place, β₅ can increase to physical value (300-1000) to capture gating overhead across all layers

### Diagnostic Clause

**If β₅ converges to <100 (still too low)**:
- **Indicates**: Basis function not accounting for all MoE layers, or β₈/β₁₀ absorbing gating overhead
- **Investigate**:
  1. Check if basis function multiplies by numMoELayers (should be ~56 for Scout)
  2. Is β₈ too high (absorbing gating overhead instead of routing)?
  3. Profile Scout: What is actual total MoE overhead (gating + routing)?
- **Next steps**: Add numMoELayers multiplication to basis function in iter14

**If β₅ converges to >2000 (too high)**:
- **Indicates**: Basis function underestimates FLOPs OR β₅ absorbing non-gating overhead
- **Investigate**:
  1. Roofline calculation: Is gating efficiency (30%) too optimistic?
  2. Is β₅ correlated with decode requests (should only correlate with total tokens)?
  3. Check if gating network has additional overhead beyond FLOP count
- **Next steps**: Lower gating efficiency assumption or add separate overhead term

**If β₅ stabilizes at 300-1000 (SUCCESS)**:
- **Confirms**: β₈ + β₁₀ removed missing-overhead pressure, β₅ captures actual MoE gating
- **Validates**: Iter7 β₅=0.0411 collapsed because model couldn't fit Scout without β₈/β₁₀
- **Expected contribution**: 300-1000 × 0.04μs = 12-40μs (within physical range)

---

## H-baseline-insight: Learn from Vidur's Success

### Prediction

Vidur avoids catastrophic TTFT overestimation (-14% to -32%) while all analytical models fail (+330% to +3031%).

After studying baseline_errors.json patterns:
- **Codegen workloads will improve most** (currently worst performers)
- **Scout workloads will show consistency** (currently wildly inconsistent -99% to +66%)
- **ITL predictions will improve** (currently underestimated -20% to -70% universally)

### Causal Mechanism

**Universal failure mode** (roofline, llm-optimizer, aiconfigurator):

All analytical models overestimate TTFT by 10-30× on codegen:
- Mistral codegen: roofline +1031%, llm-optimizer +1971%
- Llama-2 codegen: roofline +587%
- Llama-3.1 codegen: roofline +912%

**Root cause hypothesis**: Pure roofline ignores:
1. Framework overhead (vLLM scheduler, batching, queuing)
2. Cold-start effects (first 4-10 requests have 2-7× higher TTFT)
3. Batching inefficiency (long sequences → fewer requests per batch → queue delays)

**Why vidur succeeds**: Likely models framework overhead, scheduler delays, or queuing effects

**Iter13 addresses these**:
- β₆ (scheduler overhead in QueueingTime): Captures batch formation + KV allocation
- β₇ (decode per-request overhead): Captures vLLM framework per-request costs
- β₁₀ (batching inefficiency): Captures queue delays from long sequences

**Expected cascade**: Adding these framework terms should make our model behave more like vidur (avoid catastrophic overestimation)

### Diagnostic Clause

**If codegen workloads DON'T improve (<10pp gain)**:
- **Indicates**: Missing mechanism beyond framework overhead
- **Investigate**: Study vidur implementation for additional terms
- **Next steps**: Add memory bandwidth saturation term (β₃' with wide bounds) in iter14

**If Scout stays wildly inconsistent (errors still range -90% to +50%)**:
- **Indicates**: MoE-specific bottleneck not captured by current terms
- **Investigate**: Profile Scout vs dense models on same workload
- **Next steps**: Add Scout-specific calibration or separate MoE overhead term

---

## Summary of Hypotheses

| Hypothesis | Type | Prediction | Key Metric | Success Threshold |
|------------|------|------------|------------|-------------------|
| **H-main** | Architectural baseline + sequence-length term | Iter7 (155%) + β₁₀ recovers long-sequence prediction | Overall loss | <140% (≥15pp improvement) |
| **H-beta5-anomaly** | Coefficient correction | β₅ decreases 1000-2000× after β₈+β₁₀ remove inflation | β₅ convergence | 15-30μs (from 41.1ms) |
| **H-baseline-insight** | Learn from successful simulator | Framework overhead terms make us behave like vidur | Codegen TTFT improvement | ≥10pp gain on codegen workloads |

---

## Changes from Iter12

**What's REMOVED** (catastrophic iter12 architecture):
- ❌ **β₉ (FP8 dequant)**: Rejected in iter9 (converged to 0.14μs ≈ 0)
- ❌ **β₃' (KV seq-len scaling)**: Causes collinearity with β₁₀, collapsed in iter12
- ❌ **Warm-start from iter9-12**: Inflated coefficients trapped optimizer

**What's ADDED** (back to stable iter7):
- ✅ **Iter7 baseline**: Loss 155% (7pp better than iter6), proven stable
- ✅ **β₈ from iter8**: Captures Scout MoE routing (30μs), real 39ms mechanism
- ✅ **β₁₀ NEW**: Batching inefficiency, addresses iter9 sequence-length discovery

**Architecture**: 3 alpha + 10 beta
- **From iter7** (β₀-β₇): Proven stable, loss 155%
- **From iter8** (β₈): Scout MoE routing, 0pp improvement but real mechanism
- **NEW** (β₁₀): Batching inefficiency, formula validated in iter11

**Warm-start from**: Iter7 optimal coefficients (clean baseline, no inflation)

**Expected outcome**: Loss 155% → 120-140% (15-35pp improvement), stable foundation for iter14+

---

## Risk Mitigation

**Risk 1**: β₁₀ causes collinearity with other terms
- **Mitigation**: Iter10-11 unit tests proved β₁₀ formula correct (0% error)
- **Contingency**: If collinearity appears, remove β₁₀ and try β₃' alone in iter14

**Risk 2**: β₅ stays inflated (>10ms)
- **Mitigation**: β₈ + β₁₀ should remove inflation pressure
- **Contingency**: If β₅ >10ms, reformulate MoE gating basis function in iter14

**Risk 3**: Loss doesn't improve from iter7 (<10pp gain)
- **Mitigation**: Iter7 is stable baseline, small improvements acceptable
- **Contingency**: If <10pp gain, add β₃' with VERY wide bounds in iter14

**Risk 4**: Coefficient explosions like iter9-12
- **Mitigation**: Clean warm-start from iter7 (not inflated iter9-12)
- **Contingency**: If explosions occur, revert to pure iter7 + β₈ only

**Success path**: Iter13 (155% → 120-140%) establishes stable foundation → Iter14 can incrementally add β₃' if needed → Iter15+ continues steady progress

---

## Pre-Training Checklist (MANDATORY)

Before starting iter13 training, verify:

- [x] **Baseline simulator analysis**: Studied baseline_errors.json patterns (vidur success, codegen failure, Scout inconsistency)
- [x] **Journey review**: Analyzed ALL iterations 0-12 (not just recent failures)
- [x] **Iter7 is better than iter6**: Confirmed 155% < 162%, β₇ stabilized coefficients
- [x] **β₈ captures real mechanism**: Iter8 showed 0pp improvement but 39ms Scout contribution real
- [x] **Sequence-length discovery**: Iter9 proved Scout bottleneck is sequence-length-dependent (not FP8)
- [x] **β₁₀ formula validated**: Iter11 unit tests proved 0% error
- [x] **Warm-start from iter7**: Extracted iter7 optimal coefficients (NOT iter9-12 inflated)
- [ ] **Coefficient bounds validated**: Check all 10 beta bounds match expected physical ranges
- [ ] **Index mapping correct**: β₆ in QueueingTime at correct index, all other terms in StepTime
- [ ] **Build verification**: Code compiles with 10 beta coefficients

**Green light criteria**: All checks passed → Proceed with iter13 training

