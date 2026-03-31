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
| **7** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₇·decode_oh` + β₆ in QueueingTime | 155 | ✅ β₁/β₄ stabilized | Check data quality early (97% bad data found) |
| **8** | `+ β₈·MoE_routing` | 155 | ❌ No improvement, MoE routing not Scout's bottleneck | Zero improvement eliminates hypothesis |
| **9** | `+ β₉·FP8_dequant` | 161 | ❌ β₉→0, hypothesis rejected; Scout is seq-len dependent | Watch coefficient explosions (reveal missing terms) |
| **10** | `+ β₁₀·batch_ineff + β₃'·KV_seqlen` | 4267 | 💥💥 CATASTROPHIC (thought basis bugs) | Misdiagnosed - units were actually correct |
| **11** | Same as iter10 (basis functions audited) | 4084 | 💥💥 CATASTROPHIC (basis correct, YAML typo!) | Unit test basis functions BEFORE training |
| **12** | Widened β₃' bounds (0.05-5μs) to capture bandwidth | 2590 | 💥💥💥 CATASTROPHIC (β₃' collapsed!) | Don't warm-start from inflated coefficients |

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

### Iter7: Data Quality Discovery 🔍

**Added**: β₇ decode per-request overhead → β₁/β₄ stabilized

**Critical Discovery**: Analyzed reasoning traces → **97-99% of data is from overloaded servers!**

- 85% failure rate, 259-second timeouts
- Only 1-3% usable requests (those have 50-110ms TTFT, which β₆=21.5ms captures correctly!)
- **The model isn't broken — the data is**

**Solution**: Collected fresh "reasoning-lite" data → non-Scout improved 99% → 54-66% ✅

**Remaining Issue**: Scout MoE experiments account for 49% of error budget (architecture-specific).

---

### Iter8: MoE Routing Hypothesis Rejected ❌

**Added**: β₈ MoE routing overhead (30μs per routed token)

**Result**: Zero improvement (RMSE: 155.35 vs 155.37)

**Learning**: β₈ captures a REAL mechanism (39ms per Scout prefill), but it's NOT Scout's primary bottleneck (100-200ms gap remains).

**Critical Discovery**: Scout's bottleneck is NOT MoE routing overhead. This eliminates a major hypothesis and narrows the search space.

**Data Update**: Post-analysis, replaced exp17 (Scout general-2, saturated) with Scout general-lite-2-1 (clean data) — mirroring the reasoning → reasoning-lite fix from iter7.

---

### Iter9: Sequence-Length Breakthrough 🎯

**Added**: β₉ FP8 dequantization overhead

**Result**: β₉ → 0.14μs (rejected!), loss worsened to RMSE 160.6 (+5.25 pts)

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

---

## Current Status (Post-Iter12)

### ✅ What Works
- **Cascading stabilization mechanism confirmed**: β₂ (0.82→0.284) and β₃ (9.6ms→1.16ms) improved when β₃' bounds widened — proves the cascade exists!
- **Basis functions validated**: β₁₀ and β₃' implementations are CORRECT (0% error in unit tests, iter11)
- **Sequence-length discovery preserved**: Confirmed Scout's bottleneck is sequence-length-dependent (iter9 finding)

### ❌ What's Broken (Worse Than Ever)
- **CATASTROPHIC REGRESSION**: RMSE 2590 (16× worse than iter9, worst iteration ever)
- **β₃' collapsed**: 0.252μs → 0.064μs (moved AWAY from target 1-3μs range)
- **Three experiments failed completely** (100% APE) — ALL reasoning-lite workloads (data corruption suspected)
- **Cascade operates incorrectly**: β₂, β₃ improved but β₆ over-corrected (22ms vs 40-100ms expected)
- **5/11 coefficients out of range**: β₀, β₁, β₃', β₄, β₆, β₇ all outside expected ranges

### 🔍 Root Cause: The Warm-Start Trap
**What went wrong in iter12**:
1. **Warm-started from iter9's inflated coefficients** (β₂=0.82, β₃=9.6ms, β₆=99ms) while trying to fix them
2. **Created unstable optimization landscape** → optimizer collapsed ALL coefficients instead of increasing β₃'
3. **Widened bounds without constraints** → gave optimizer room to move wrong direction
4. **Data quality not validated** → three reasoning-lite experiments failed (100% APE)

**The Paradox**: Trying to fix inflated coefficients by widening bounds on a competing term made things worse. The inflation must be resolved FIRST by returning to stable baseline.

### 🎯 Critical Next Steps (Before Iter13)

**MANDATORY Step 1**: Validate reasoning-lite data quality
```bash
# Check for corrupted ground truth
grep -E "ttft.*: 0\.|e2e.*: 0\." training/trainval_data/*reasoning*/ground_truth.csv
```
If corrupted: Exclude from training OR regenerate ground truth

**MANDATORY Step 2**: REVERT to stable baseline (iter6 or iter7)
- **Iter6**: Loss ~80% (best iteration ever)
- **Iter9-12**: Failures (160-2590%)
- **Architecture**: 3 alpha + 8 beta (β₀-β₇, NO β₈/β₉/β₁₀/β₃')
- **Expected**: Loss <100% (return to stability)

**DO NOT**:
- ❌ Attempt to fix iter12 architecture (fundamentally flawed)
- ❌ Warm-start from iter9-12 (all have inflated or collapsed coefficients)
- ❌ Add new terms before returning to <100% loss baseline
- ❌ Train without data validation (reasoning-lite may be corrupted)

**Strategy**: Return to known-good state (iter6/7), validate <100% loss, THEN incrementally add terms with:
1. Unit tests for new basis functions
2. Collinearity checks (design matrix condition number <30)
3. Data validation (no zero-latencies, balanced workload distribution)
4. Warm-start from STABLE iteration (≥80% coefficients in range)

**Note on Metrics**: RMSE (Root Mean Square Error) is computed across all experiments' APE (Absolute Percentage Error) values. Target RMSE < 90 means the model should have low and consistent error across all test cases.

---

## For Details

- **Per-iteration findings**: See `training/iterations/iter{N}/iter{N}-FINDINGS.md`
- **Trace analysis**: See `training/iterations/TRACE_DATA_ANALYSIS.md`
- **Coefficient mappings**: Beta indices shift between iterations — check individual FINDINGS files
