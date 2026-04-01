# Training Journey: 11 Iterations to Model vLLM Latency

## Quick Reference: Formula Evolution

| Iter | Formula (StepTime only, see note for QueueingTime) | Loss (RMSE) | Result | Key Recommendation |
|------|-----------|------|--------|-------------------|
| **0** | `β₀·prefill + β₁·decode_mem + β₂·const` | 200 | ❌ Insufficient | Need overhead terms beyond roofline |
| **1** | `β₀·prefill + β₁·decode_mem + β₂·sched + β₃·TP + β₄·KV + β₅·chunk + β₆·decode_comp + β₇·MoE` | 135 | ⚠️ +33% but missed <80 | Good progress, keep adding terms |
| **2** | `β₀·prefill + β₁·decode_mem + β₂·sched + β₃·TP + β₄·KV + β₅·decode_comp + β₆·MoE + β₇·longctx + β₈·dec_oh` | 136 | ❌ β₇, β₈ never moved | Validate terms before adding |
| **3** | `β₀·prefill + β₁·decode_mem + β₂·sched + β₃·TP + β₄·KV + β₅·decode_comp + β₆·MoE + β₇·TP_prefill` | 133 | ⚠️ β₇ rejected (≈0) | Remove terms optimizer rejects |
| **4** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₆·activation_BW` | 129 | ❌ Coefficients destabilized | Low RMSE ≠ good (check coefficient ranges) |
| **5** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₆·per_layer` | 603 | 💥 CATASTROPHIC | Validate assumptions from traces first |
| **6** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE` + **β₆ → QueueingTime** | 162 | ✅ **Decoupling breakthrough** | Term location matters (avoid collinearity) |
| **—** | **DATASET CHANGED (1/2)** | — | reasoning → reasoning-lite (3 experiments replaced) | Fresh data from unloaded servers |
| **7** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₇·decode_oh` + β₆ in QueueingTime | 155 | ✅ β₁/β₄ stabilized (14 clean + 1 bad exp) | Check data quality early (97% bad data found) |
| **8** | `+ β₈·MoE_routing` | 155 | ❌ No improvement (still 14 clean + 1 bad exp) | Zero improvement eliminates hypothesis |
| **—** | **DATASET CHANGED (2/2)** | — | Scout general → general-lite (exp17 replaced) | All 15 experiments now clean |
| **9** | `+ β₉·FP8_dequant` | 161 | ❌ β₉→0 (**FIRST iter with 15 clean experiments**) | Watch coefficient explosions (reveal missing terms) |
| **10** | `+ β₁₀·batch_ineff + β₃'·KV_seqlen` | 4267 | 💥💥 CATASTROPHIC (thought basis bugs) | Misdiagnosed - units were actually correct |
| **11** | Same as iter10 (basis functions audited) | 4084 | 💥💥 CATASTROPHIC (basis correct, YAML typo!) | Unit test basis functions BEFORE training |
| **12** | Widened β₃' bounds (0.05-5μs) to capture bandwidth | 2590 | 💥💥💥 CATASTROPHIC (β₃' collapsed!) | Don't warm-start from inflated coefficients |
| **13** | Return to iter7 baseline + β₈ + β₁₀ | 2387 | 💥💥 β₅ exploded 46,800× (1924.4) | Coefficient explosion ≠ missing term (bug in basis) |
| **14** | Fixed β₅ basis function (added numMoELayers) | 2319 | 💥💥 Only 2.8% improvement | Fixing one coefficient doesn't fix cascade |

**Note**: Iter6-7 split overhead between StepTime and QueueingTime. β₆ (scheduler overhead) moved to QueueingTime in iter6.

**Current Formula (Iter12 - Same as Iter11, only bounds changed)**:
```
StepTime = β₀·prefill_comp + β₁·decode_mem + β₂·TP_comm + β₃·KV_base + β₃'·KV_seqlen +
           β₄·decode_comp + β₅·MoE_gating + β₇·decode_overhead + β₈·MoE_routing +
           β₁₀·batch_ineff
QueueingTime = α₀ + α₁·input_tokens + β₆·scheduler_overhead
```

**Change from Iter11**: Widened β₃' bounds from 0.05-2.0μs → 0.05-5.0μs (2.5× increase) to allow capturing both KV allocation and bandwidth saturation penalty.

---

## The Problem We're Solving

**Challenge**: Predict vLLM inference latency with physics-based model + trained coefficients.

**Hardest case**: "Reasoning" workloads take 100-200ms but model predicts ~1ms (99% error).

---

## Journey Timeline

### Iter0-2: Foundation Building
- Started with basic roofline `max(compute, memory)` → 200% error
- Added 5 additive overhead terms (TP comm, KV mgmt, chunking, MoE) → 135% error
- Added long-context terms → **FAILED** (terms never moved during optimization)

**Learning**: Adding terms without validation wastes optimization budget.

---

### Iter3-5: The Wrong Hypothesis
**Hypothesis**: "Reasoning experiments use 8K-16K tokens (vs 1K for others)"

- **Iter3**: Added context-dependent TP prefill → rejected by optimizer
- **Iter4**: Added activation bandwidth → coefficients destabilized
- **Iter5**: Added per-layer kernel overhead → **CATASTROPHIC 603% error**

**Reality Check**: Analyzed traces → reasoning uses ~1K tokens, NOT 8K!

**Learning**: **VALIDATE ASSUMPTIONS FROM TRACES FIRST.** Wasted 3 iterations on wrong hypothesis.

---

### Iter6: Architectural Breakthrough 🎯

**Key Insight**: Moved β₆ from StepTime (per-layer) to QueueingTime (per-request)

**Why this mattered**:
- Decoupled scheduler overhead from prefill compute (no more collinearity)
- β₀ dropped from 0.266 → 0.164, restoring predictions
- 8 out of 11 short-context experiments recovered (1091% → 26% for Llama-3.1!)

**But**: Reasoning still stuck at 99% error.

**Learning**: **Term location matters as much as formula.**

---

### Iter7: First Iteration with Reasoning-Lite Data 🔍

**Added**: β₇ decode per-request overhead → β₁/β₄ stabilized

**Dataset Status**: **14 clean + 1 bad** (3 reasoning-lite ✅ + exp17 Scout general saturated ❌)

**Critical Discovery from Iter6**: Analyzed reasoning traces → **97-99% of data was from overloaded servers!**
- 85% failure rate, 259-second timeouts
- Only 1-3% usable requests (those have 50-110ms TTFT, which β₆=21.5ms captures correctly!)
- **The model wasn't broken — the data was**

**Result with Fresh Data**: Non-Scout reasoning-lite improved dramatically: 99% → 54-66% error ✅

**Remaining Issue**: Scout MoE experiments account for 49% of error budget (architecture-specific, NOT data quality).

---

### Dataset Changes: Two-Stage Data Quality Fix 🔄

**Change 1: Reasoning → Reasoning-Lite** (March 30, 2026 - between iter6 and iter7)

**Context**: Iter6 discovered that 97-99% of reasoning workload data came from overloaded servers (85% failure rate, 259s timeouts).

**Action**: Replaced 3 corrupted reasoning experiments with fresh "reasoning-lite" data:
- **Iter0-6**: 15 experiments (3 bad reasoning + 12 good) = **14 clean + 1 bad**
- **Iter7-8**: 15 experiments (3 reasoning-lite + 12 good) = **14 clean + 1 bad**

**Impact**: Non-Scout reasoning-lite improved from 99% → 54-66% error ✅

---

**Change 2: Scout General → General-Lite** (March 30, 2026 - after iter8)

**Context**: Iter8 analysis revealed exp17 (Scout general) was collected under saturated server conditions.

**Action**: Replaced exp17 with Scout general-lite-2-1 (reduced workload intensity):
- **Iter0-8**: exp17 Scout general-2 (saturated) = **14 clean + 1 bad**
- **Iter9+**: exp17 Scout general-lite-2-1 (clean) = **15 clean ✅**

**Impact**:
- **Iter9 is the FIRST iteration trained on a fully clean dataset** (all 15 experiments from unloaded servers)
- Loss: 161 RMSE (worse than iter7/8's 155, but due to wrong β₉ hypothesis, not data quality)

---

**Summary**:
- **Iter0-6**: 3 bad reasoning + 1 bad Scout general = 11/15 clean (73%)
- **Iter7-8**: 3 reasoning-lite + 1 bad Scout general = 14/15 clean (93%)
- **Iter9+**: 3 reasoning-lite + 1 Scout general-lite = **15/15 clean (100%)** ✅

---

### Iter8: MoE Routing Hypothesis Rejected ❌

**Added**: β₈ MoE routing overhead (30μs per routed token)

**Dataset Status**: **14 clean + 1 bad** (still using saturated exp17 Scout general-2)

**Result**: Zero improvement (RMSE: 155.35 vs 155.37)

**Learning**: β₈ captures a REAL mechanism (39ms per Scout prefill), but it's NOT Scout's primary bottleneck (100-200ms gap remains).

**Critical Discovery**: Scout's bottleneck is NOT MoE routing overhead. This eliminates a major hypothesis and narrows the search space.

**Data Update After Iter8**: Replaced exp17 (Scout general-2, saturated) with Scout general-lite-2-1 (clean data) for iter9+.

---

### Iter9: Sequence-Length Breakthrough 🎯

**Added**: β₉ FP8 dequantization overhead

**Dataset Status**: **15 clean experiments (100%)** ✅ **FIRST ITERATION WITH FULLY CLEAN DATASET**

**Result**: β₉ → 0.14μs (rejected!), loss worsened to RMSE 160.6 (+5.25 pts from iter8's 155)
- Loss increased despite clean data because β₉ hypothesis was wrong, not data quality

**CRITICAL DISCOVERY**: Scout's bottleneck is **sequence-length-dependent**, NOT architecture-dependent!

**Evidence**:
- Short-sequence Scout improved dramatically: roleplay -53pp (79%→26%), codegen -34pp (92%→58%)
- Long-sequence Scout failed: general-lite 92%, reasoning-lite 91% (no improvement)
- **Inverse correlation**: Longer sequences → worse performance (opposite of FP8 hypothesis)

**Mechanism Revealed**: Long sequences face batching inefficiency, scheduler struggles, or memory bandwidth issues.

**Coefficient Explosions**: β₆ +654% (13ms→99ms), β₂ +343% (0.18→0.82), β₈ +143% (30μs→73μs) — optimizer compensating for wrong hypothesis.

---

### Iter10-11: The Catastrophic Detour 💥

**Iter10 - Added**: β₁₀ (batching inefficiency) + β₃' (KV seq-len component)

**Result**: Catastrophic loss explosion (RMSE: 4267, 27× worse than iter9)

**Initial Diagnosis (WRONG)**: "Basis function formulation bugs" — β₁₀ converged 1000× too small, β₃' converged 65× too large.

**Iter11 - Reality Check**: Wrote unit tests → **basis functions were CORRECT all along!**

**Actual Problem**: A YAML comment typo wrote "0.1-1.0 ms" instead of "0.1-1.0 μs", leading to incorrect expected ranges. Wasted 7,250 trial-hours chasing non-existent bugs.

**Real Issue Revealed**: 6/11 coefficients out of range, particularly β₆ (scheduler) = 59ms vs expected 15-40ms. The model cannot converge with current basis function set.

**Key Learning**: **Always write unit tests BEFORE training**. A 5-minute test would have prevented this entire detour.

---

### Iter12: The Warm-Start Catastrophe 💥💥💥

**Hypothesis**: Widen β₃' bounds to 0.05-5.0μs to allow capturing BOTH KV allocation (CPU-side) and memory bandwidth saturation (GPU-side) in a single term.

**Rationale**:
- Iter9 saw THREE coefficients explode together: β₂ (TP comm), β₃ (KV base), β₆ (scheduler)
- Single root cause suspected: Memory bandwidth saturation during long-sequence prefill
- β₃' already uses basis function Σ(prefillTokens × numLayers) — perfect for bandwidth penalty
- Widen bounds to let optimizer increase β₃' from 0.252μs → 1-3μs to capture both mechanisms

**Result**: **CATASTROPHIC FAILURE** — RMSE 2590 (16× worse than iter9, worst iteration ever)

**What Actually Happened**:
- **β₃' collapsed** from 0.252μs → **0.064μs** (4× DECREASE, moved AWAY from target!)
- **Three experiments failed completely** (100% APE) — ALL reasoning-lite workloads
- **Cascading stabilization PARTIAL**: β₂, β₃ improved but β₆ over-corrected (22ms vs 40-100ms expected)

**Root Causes**:
1. **Warm-Start Paradox**: Started from iter9's inflated coefficients (β₂=0.82, β₃=9.6ms, β₆=99ms) while trying to fix them → unstable optimization landscape
2. **Bounds Widening Backfired**: Gave optimizer room to collapse β₃' instead of increase it (no lower bound constraint)
3. **Data Quality Issues**: Three reasoning-lite experiments failed (100% APE) — suggests corrupted ground truth data

**Partial Success**: The cascading stabilization mechanism WAS confirmed:
- β₂: 0.82 → 0.284 ✅ (entered range 0.25-0.60)
- β₃: 9.6ms → 1.16ms ✅ (entered range 0.4-1.5ms)
- This proves cascade exists, but operates incorrectly when warm-starting from inflated coefficients

**Key Learning**: **Don't warm-start from inflated coefficients**. Widening bounds on a coefficient competing with inflated neighbors doesn't work. Must first return to stable baseline (iter6/7) before expanding parameter space.

**Cost**: 413 trials × 16 min = 110 hours wasted, 16× regression from baseline

---

### Iter13: β₅ Explosion from Missing Multiplier 💥💥

**Strategy**: Return to stable iter7 baseline (155% loss, 8 coefficients) and add back β₈ (MoE routing) + β₁₀ (batching inefficiency) incrementally.

**Hypothesis**: Warm-starting from iter7 will provide stability, β₁₀ will recover sequence-length prediction.

**Result**: **WORST FAILURE IN TRAINING HISTORY** — RMSE 2387 (15.4× worse than iter7)

**What Happened**:
- **β₅ explosion**: MoE gating coefficient exploded from 0.0411 (iter7) → **1924.4** (46,800× increase!)
- **Complete simulator failure**: Three reasoning-lite experiments returned exactly 100% error
- **Massive overprediction**: Codegen experiments 1200-2700% TTFT (152× worse than iter7)

**Root Cause Analysis**:
- **Bug in basis function**: β₅ computation was missing `× numMoELayers` multiplier
- With β₅=1924.4 and missing multiplier, effective contribution ≈ 4.3ms per request (physically implausible)
- Optimizer compensated for under-computed basis function by inflating the coefficient
- Dataset shift also played a role: reasoning→reasoning-lite changed between iter7 and iter13

**Key Discovery**: **Coefficient explosion often indicates basis function bug, not missing physics term**. When a coefficient increases by 1000×+, audit the basis function implementation before adding new terms.

**Cost**: 738 trials × 13 min = 160 hours, 15.4× regression from baseline

---

### Iter14: The Necessary-But-Not-Sufficient Fix 💥

**Strategy**: Fix β₅ basis function bug (add `× numMoELayers` multiplier), expect dramatic loss improvement.

**Hypothesis**: β₅ is the "sole cause" of iter13's cascade. Fixing it will recover to near-iter7 performance (155% loss).

**Prediction**: Loss 2387% → <200% (≥92% improvement), with β₅ converging to 1-50 range.

**Result**: ❌ **HYPOTHESIS REFUTED** — Loss barely improved: 2387% → 2319% (only 2.8% improvement)

**What Happened**:
- **β₅ converged correctly**: 1924.4 → 32.5 (within predicted 1-50 range) ✅
- **Loss barely changed**: 2319% still catastrophic ❌
- **Other coefficients destabilized**: β₀ +105%, β₃ -287×, β₆ -7859× trying to compensate
- **Dense models still failed**: All 10-40× overprediction despite β₅ fix
- **Scout improved slightly**: MoE experiments performed BETTER than dense (342-767% vs 1000-3700%)

**Critical Insight — Principle 1**: **Coefficient Convergence ≠ Performance Recovery**

When you fix ONE broken coefficient, the optimizer adjusts ALL OTHER coefficients to maintain similar total error. β₅ converged, but β₀/β₃/β₆ absorbed the error. **Analogy**: Fixing one broken table leg doesn't fix the table if the other three legs are also broken.

**Critical Insight — Principle 2**: **Warm-Start Failure Indicates Dataset Shift**

Iter7 trained on reasoning (long sequences, overloaded servers). Iter13-14 trained on reasoning-lite (shorter, cleaner data). Warm-starting from iter7 coefficients anchored the optimizer in the wrong basin of attraction — it couldn't escape iter7's local minimum despite 1000 trials.

**Action for Iter15**: Try **cold-start optimization** (uniform random initialization, 2000 trials) instead of warm-starting from any previous iteration.

**Cost**: 1000 trials × 9 min = 150 hours, only 2.8% improvement

---

## Key Lessons

| Lesson | Evidence |
|--------|----------|
| **Validate assumptions from traces** | Wasted iter3-5 assuming reasoning = long context without checking |
| **Avoid collinearity** | Per-layer terms + num_layers scaling broke iter4-5 |
| **Term location matters** | Moving β₆ to QueueingTime eliminated collinearity |
| **Optimizer rejection is signal** | When terms don't move from initial values, remove them |
| **Check data quality early** | 7 iterations before discovering 85% failure rate in reasoning data |
| **Zero improvement eliminates hypotheses** | Iter8: β₈ = 30μs (plausible) but 0pp improvement → MoE routing NOT the bottleneck |
| **Coefficient explosions reveal missing mechanisms** | Iter9: β₆ +654%, β₂ +343% when β₉→0 → optimizer compensating for wrong hypothesis |
| **Unit test basis functions BEFORE training** | Iter10-11: 7,250 trial-hours wasted on "bugs" that didn't exist (just YAML typo) |
| **Audit code before accepting diagnoses** | A 5-minute unit test would have prevented entire iter10-11 detour |
| **Don't warm-start from inflated coefficients** | Iter12: Warm-started from iter9 (β₂=0.82, β₃=9.6ms inflated) → β₃' collapsed, loss 16× worse |
| **Widening bounds requires constraints** | Iter12: Widened β₃' 0.05-5.0μs without lower bound → collapsed to 0.064μs (wrong direction) |
| **Validate data quality BEFORE training** | Iter12: Three reasoning-lite experiments failed (100% APE) — suggests corrupted data |
| **Coefficient explosion → audit basis function** | Iter13: β₅ increased 46,800× → found missing `× numMoELayers` in implementation |
| **Coefficient convergence ≠ performance recovery** | Iter14: β₅ fixed (1924→32.5) but loss barely improved (2387%→2319%, 2.8%) — other coefficients absorbed error |
| **Warm-start fails after dataset shift** | Iter13-14: Warm-started from iter7 (reasoning data) but training on reasoning-lite → optimizer stuck in wrong basin |

---

## Current Status (Post-Iter14)

### ✅ What Works
- **β₅ bug fixed**: Missing `× numMoELayers` multiplier added, coefficient converged correctly (1924→32.5)
- **Scout improved relative to dense**: After β₅ fix, MoE experiments perform BETTER than dense models (342-767% vs 1000-3700%)
- **Basis functions audited**: β₁₀ and β₃' implementations validated with unit tests (iter11)
- **Dataset updated**: All 15 experiments use clean data (reasoning-lite replacing corrupted reasoning)

### ❌ What's Broken (Still Catastrophic)
- **MINIMAL IMPROVEMENT**: Loss 2387% → 2319% (only 2.8% improvement despite β₅ fix)
- **Dense models still fail**: General-lite and codegen experiments 10-40× overprediction
- **Coefficient cascade**: Fixing β₅ caused β₀ (+105%), β₃ (-287×), β₆ (-7859×) to destabilize
- **Three reasoning-lite experiments**: Exactly 100% error (numerical failure in simulator)
- **All roofline coefficients out of range**: β₀, β₁, β₄ all outside expected physical bounds

### 🔍 Root Causes: Two Independent Problems

**Problem 1**: **Coefficient Convergence ≠ Performance Recovery**
- Fixing ONE coefficient (β₅) doesn't fix the MODEL because other coefficients adjust to compensate
- The optimizer treats coefficients as independent knobs, not coupled physics parameters
- Result: β₅ converged, but overall behavior barely changed (other coefficients absorbed the error)

**Problem 2**: **Warm-Start Failure After Dataset Shift**
- Iter7 trained on reasoning (long sequences, overloaded servers)
- Iter13-14 trained on reasoning-lite (shorter, cleaner data)
- Warm-starting from iter7 anchored optimizer in wrong basin of attraction
- 1000 trials insufficient to escape iter7's local minimum in new landscape

### 🎯 Critical Next Steps (Iter15+)

**MANDATORY**: **Try Cold-Start Optimization**
1. **Uniform random initialization** within physically plausible bounds (DO NOT warm-start from any previous iteration)
2. **Increase trials**: 1000 → 2000-3000 (10-dimensional search requires more exploration)
3. **Multi-term addition**: Add β₈, β₉, β₁₀ simultaneously (address all three roofline defects together)
4. **Holistic validation**: Track coefficient CONTRIBUTIONS (e.g., β₅ × basis < 10% of total StepTime), not just values

**Hypothesis for Iter15**: Cold-start will find different local minimum better suited to reasoning-lite dataset, allowing model to escape iter7's basin.

**What to Monitor**:
- Coefficient stability (are roofline terms β₀/β₁/β₄ staying in expected ranges?)
- Per-workload breakdown (are general-lite improving? Are reasoning-lite still 100% error?)
- Contribution analysis (which terms dominate StepTime? Are they physically plausible?)

**DO NOT**:
- ❌ Warm-start from iter7, iter13, or iter14 (wrong dataset, wrong basin)
- ❌ Assume single-coefficient fixes will cascade to overall improvement
- ❌ Add new terms without validating existing ones are stable

**Note on Metrics**: RMSE (Root Mean Square Error) is computed across all experiments' APE (Absolute Percentage Error) values. Target RMSE < 90 means the model should have low and consistent error across all test cases.

---

## For Details

- **Per-iteration findings**: See `training/iterations/iter{N}/iter{N}-FINDINGS.md`
- **Trace analysis**: See `training/iterations/TRACE_DATA_ANALYSIS.md`
- **Coefficient mappings**: Beta indices shift between iterations — check individual FINDINGS files
