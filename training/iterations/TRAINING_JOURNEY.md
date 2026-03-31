# Training Journey: 11 Iterations to Model vLLM Latency

## Quick Reference: Formula Evolution

| Iter | Formula (StepTime only, see note for QueueingTime) | Loss (RMSE) | Result |
|------|-----------|------|--------|
| **0** | `β₀·prefill + β₁·decode_mem + β₂·const` | 200 | ❌ Insufficient |
| **1** | `β₀·prefill + β₁·decode_mem + β₂·sched + β₃·TP + β₄·KV + β₅·chunk + β₆·decode_comp + β₇·MoE` | 135 | ⚠️ +33% but missed <80 |
| **2** | `β₀·prefill + β₁·decode_mem + β₂·sched + β₃·TP + β₄·KV + β₅·decode_comp + β₆·MoE + β₇·longctx + β₈·dec_oh` | 136 | ❌ β₇, β₈ never moved |
| **3** | `β₀·prefill + β₁·decode_mem + β₂·sched + β₃·TP + β₄·KV + β₅·decode_comp + β₆·MoE + β₇·TP_prefill` | 133 | ⚠️ β₇ rejected (≈0) |
| **4** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₆·activation_BW` | 129 | ❌ Coefficients destabilized |
| **5** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₆·per_layer` | 603 | 💥 CATASTROPHIC |
| **6** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE` + **β₆ → QueueingTime** | 162 | ✅ **Decoupling breakthrough** |
| **7** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₇·decode_oh` + β₆ in QueueingTime | 155 | ✅ β₁/β₄ stabilized |
| **8** | `+ β₈·MoE_routing` | 155 | ❌ No improvement, MoE routing not Scout's bottleneck |
| **9** | `+ β₉·FP8_dequant` | 161 | ❌ β₉→0, hypothesis rejected; Scout is seq-len dependent |
| **10** | `+ β₁₀·batch_ineff + β₃'·KV_seqlen` | 4267 | 💥💥 CATASTROPHIC (thought basis bugs) |
| **11** | Same as iter10 (basis functions audited) | 4084 | 💥💥 CATASTROPHIC (basis correct, YAML typo!) |

**Note**: Iter6-7 split overhead between StepTime and QueueingTime. β₆ (scheduler overhead) moved to QueueingTime in iter6.

**Current Formula (Iter11)**:
```
StepTime = β₀·prefill_comp + β₁·decode_mem + β₂·TP_comm + β₃·KV_base + β₃'·KV_seqlen +
           β₄·decode_comp + β₅·MoE_gating + β₇·decode_overhead + β₈·MoE_routing +
           β₁₀·batch_ineff
QueueingTime = α₀ + α₁·input_tokens + β₆·scheduler_overhead
```

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

---

## Current Status (Post-Iter11)

### ✅ What Works
- **Iter9 sequence-length discovery**: Confirmed Scout's bottleneck is sequence-length-dependent (not architecture-specific)
- **Short-sequence predictions improved**: Scout roleplay (26% TTFT), codegen (58% TTFT) — dramatic improvement from iter8
- **Basis functions validated**: β₁₀ and β₃' implementations are CORRECT (0% error in unit tests)
- **5/11 coefficients in range**: β₂, β₃', β₅, β₈, β₁₀ all converged to physical ranges

### ❌ What's Broken
- **Catastrophic convergence failure**: RMSE 4084 (45× worse than target, unchanged from iter10)
- **Long-sequence experiments**: Scout general-lite (92% TTFT APE), reasoning-lite (91% TTFT APE) still fail
- **6/11 coefficients out of range**: β₀ (+30%), β₁ (-8%), β₃ (-50%), β₄ (+25%), β₆ (+48-295% ⚠️), β₇ (-38-75%)
- **β₆ explosion**: 59.3ms (expected 15-40ms) — scheduler overhead absorbing unexplained error

### 🔍 Root Cause Analysis Needed
**Key Questions**:
1. Why does β₆ (scheduler overhead) keep exploding? (59ms vs 15-40ms expected)
2. What mechanism explains long-sequence Scout failures? (100-200ms unexplained gap)
3. Are basis functions correct but INSUFFICIENT? (missing critical terms?)
4. Is training data quality still an issue? (saturated servers, non-representative workloads?)

**Hypotheses for Iter12+**:
- **Option 1**: Profile scheduler code path for long-sequence requests (vllm/core/scheduler.py)
- **Option 2**: Add memory bandwidth saturation term (HBM throughput × seq_len scaling)
- **Option 3**: Investigate batch formation efficiency (why long sequences delay others?)
- **Option 4**: Collect targeted Scout profiling data (nsys/nvprof on long vs short sequences)

### 🎯 Next Steps
**DO NOT proceed to iter12 without**:
1. ✅ Root cause verification (profiling data or trace analysis)
2. ✅ Unit tests for any new basis functions
3. ✅ Validation that β₆ explosion is NOT absorbing missing physics
4. ⚠️ Consider: Is the current basis function set fundamentally insufficient?

**Note on Metrics**: RMSE (Root Mean Square Error) is computed across all experiments' APE (Absolute Percentage Error) values. Target RMSE < 90 means the model should have low and consistent error across all test cases.

---

## For Details

- **Per-iteration findings**: See `training/iterations/iter{N}/iter{N}-FINDINGS.md`
- **Trace analysis**: See `training/iterations/TRACE_DATA_ANALYSIS.md`
- **Coefficient mappings**: Beta indices shift between iterations — check individual FINDINGS files
