# Training Journey: 8 Iterations to Model vLLM Latency

## Quick Reference: Formula Evolution

| Iter | Formula (StepTime only, see note for QueueingTime) | Loss | Result |
|------|-----------|------|--------|
| **0** | `β₀·prefill + β₁·decode_mem + β₂·const` | 200% | ❌ Insufficient |
| **1** | `β₀·prefill + β₁·decode_mem + β₂·sched + β₃·TP + β₄·KV + β₅·chunk + β₆·decode_comp + β₇·MoE` | 135% | ⚠️ +33% but missed <80% |
| **2** | `β₀·prefill + β₁·decode_mem + β₂·sched + β₃·TP + β₄·KV + β₅·decode_comp + β₆·MoE + β₇·longctx + β₈·dec_oh` | 136% | ❌ β₇, β₈ never moved |
| **3** | `β₀·prefill + β₁·decode_mem + β₂·sched + β₃·TP + β₄·KV + β₅·decode_comp + β₆·MoE + β₇·TP_prefill` | 133% | ⚠️ β₇ rejected (≈0) |
| **4** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₆·activation_BW` | 129% | ❌ Coefficients destabilized |
| **5** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₆·per_layer` | 603% | 💥 CATASTROPHIC |
| **6** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE` + **β₆ → QueueingTime** | 162% | ✅ **Decoupling breakthrough** |
| **7** | `β₀·prefill + β₁·decode_mem + β₂·TP + β₃·KV + β₄·decode_comp + β₅·MoE + β₇·decode_oh` + β₆ in QueueingTime | 155% | ✅ β₁/β₄ stabilized |

**Note**: Iter6-7 split overhead between StepTime and QueueingTime. β₆ (scheduler overhead) moved to QueueingTime in iter6.

**Current Formula (Iter7)**:
```
StepTime = β₀·prefill_comp + β₁·decode_mem + β₂·TP_comm + β₃·KV_mgmt +
           β₄·decode_comp + β₅·MoE_gating + β₇·decode_overhead
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

## Key Lessons

| Lesson | Evidence |
|--------|----------|
| **Validate assumptions from traces** | Wasted iter3-5 assuming reasoning = long context without checking |
| **Avoid collinearity** | Per-layer terms + num_layers scaling broke iter4-5 |
| **Term location matters** | Moving β₆ to QueueingTime eliminated collinearity |
| **Optimizer rejection is signal** | When terms don't move from initial values, remove them |
| **Check data quality early** | 7 iterations before discovering 85% failure rate in reasoning data |

---

## Current Status (Post-Iter7)

### ✅ What Works
- **11/15 experiments**: 5-66% error (good!)
- **Coefficient stability**: β₁=1.108, β₄=0.713 returned to physical ranges
- **Clean data validated**: Non-Scout reasoning-lite improved dramatically

### ❌ What's Broken
- **Scout MoE**: 4 experiments, 79-100% TTFT error, 49% of total error budget
- **Root cause**: MoE routing overhead not captured by current 8 terms

### 🎯 Next Steps (Iter8)
**Add β₈**: MoE routing overhead = `β₈ × (MoE_layers × tokens × experts_per_token / TP)`
- Expected: 10-50μs per routed token
- Should capture Scout's 767% combined loss
- Keep all 15 experiments in training (including Scout)

---

## For Details

- **Per-iteration findings**: See `training/iterations/iter{N}/iter{N}-FINDINGS.md`
- **Trace analysis**: See `training/iterations/TRACE_DATA_ANALYSIS.md`
- **Coefficient mappings**: Beta indices shift between iterations — check individual FINDINGS files
