# Iteration 2: Very Long Context Overhead + Per-Request Decode Overhead + Smooth Regime Transition

## Iteration Context

**Previous iteration (iter1) outcomes**:
- Overall loss: 134.54% (TTFT RMSE=69.29%, E2E RMSE=65.24%)
- **What worked**: Dense non-reasoning experiments achieved <50% combined loss (7/15 experiments)
- **What failed**:
  - Reasoning workload ~100% TTFT error (2 experiments)
  - Scout MoE validation failure (4 experiments) - **now fixed via simulator bug patches**
  - Coefficient distortion persists (β₀=0.203, β₁=1.553 still physically implausible)

**Iter1 ablation results** (50 trials each):
- **β₅ (chunking)**: +1.06% overall loss, +0.07% TTFT RMSE → **REDUNDANT, remove**
- **β₄ (KV mgmt)**: +30.28% E2E degradation → **CRITICAL, must keep**
- **β₃ (TP comm)**: +6.77% E2E degradation → **MODERATE, keep**

**Iter2 changes from iter1**:
1. **Remove β₅ (chunking overhead)**: Ablation confirmed redundancy
2. **Add β₇ (very long context prefill overhead)**: Captures reasoning experiment failures (>4096 tokens)
3. **Add β₈ (per-request decode overhead)**: Normalizes inflated β₁ by capturing scheduler per-request work
4. **Replace discrete decode regime split with sigmoid interpolation**: Smooth transition from memory-bound to compute-bound

**New Beta coefficient mapping** (9 terms total):
- β₀: Prefill compute time
- β₁: Decode memory-bound time × memory_weight(batch_size)
- β₂: Constant scheduler overhead
- β₃: TP communication overhead
- β₄: KV cache management overhead per request
- β₅: Decode compute-bound time × compute_weight(batch_size) [formerly β₆]
- β₆: MoE gating overhead [formerly β₇]
- β₇: Very long context prefill overhead (>4096 tokens) [NEW]
- β₈: Per-request decode overhead [NEW]

---

## H-main: Very Long Context + Per-Request Overhead Mechanism

**Prediction**: Overall loss will decrease to <80% (from 134.54% in iter1), with:
- **TTFT RMSE** reducing from 69.29% to <40%
- **E2E RMSE** reducing from 65.24% to <40%
- **Target experiments**: Reasoning experiments (Llama-2, Qwen2.5) will improve from ~100% TTFT to <60% TTFT
- **Scout experiments**: Now that simulator bugs are fixed, Scout experiments will achieve <60% combined loss (previously 200%)

**Causal Mechanism**:

Iter1's failures stem from two missing overhead terms that inflate other coefficients as compensation:

1. **Very long context prefill overhead (β₇)**: Reasoning prompts (>4096 tokens) have additional prefill overhead beyond standard FLOPs accounting:
   - **Attention memory bandwidth saturation**: Very long sequences cause intermediate attention matrices to spill to HBM, adding bandwidth overhead beyond weight/KV reads
   - **KV cache recomputation**: vLLM preemption logic triggers KV recomputation for long contexts under memory pressure, introducing per-chunk recomputation cost
   - **Reduced prefix cache effectiveness**: Unique long prompts (reasoning CoT chains) have lower cache hit rates, forcing full prefill instead of cached prefix reuse

   The β₇ term captures this as: `(prompt_tokens - 4096) / 1000 × num_layers` — naturally scaling with excess context length and layer depth, without using workload labels.

2. **Per-request decode overhead (β₈)**: Each active decode request incurs fixed overhead independent of compute/memory bottlenecks:
   - **Scheduler per-request work**: Priority checks, sequence state updates (~5-20μs per request)
   - **Attention state setup**: Query offsets, KV block tables, sequence lengths (~10-50μs per request)
   - **Kernel launch overhead**: Small-batch decode has higher per-request kernel launch cost (~10-30μs per request)

   The β₈ term captures this as: `num_decode_requests × per_request_time` — absorbing the overhead currently compensated by inflated β₁.

3. **Smooth regime transition**: Replacing discrete `if batch_size < 8` with sigmoid interpolation `memory_weight(n) = 1/(1+exp((n-8)/2))` prevents discontinuities in medium-batch predictions (5-12 requests).

**Code Citations**:
- **vLLM long-context overhead**: `vllm/attention/backends/flashinfer.py:prefill_with_paged_kv()` shows memory bandwidth saturation for long sequences; `vllm/core/scheduler.py:_schedule()` triggers KV recomputation under memory pressure
- **vLLM per-request overhead**: `vllm/core/scheduler.py:_schedule_running()` iterates all active requests for priority checks; `vllm/attention/backends/flashinfer.py:begin_forward()` prepares per-request KV block tables
- **BLIS implementation**: `sim/latency/evolved_model.go:StepTime()` lines 249-284 (β₇ term), lines 269-284 (β₈ term), lines 173-180 (sigmoid interpolation)

**Diagnostic Clause**:
- *If β₇ converges to near-zero (<0.1)*: Long-context overhead is negligible or already captured by β₀ prefill term. Reasoning failures have a different root cause (e.g., attention pattern complexity not bandwidth).
- *If β₈ converges to near-zero (<5μs)*: Per-request overhead is negligible, and β₁ inflation has a different cause (e.g., decode FLOPs formula is wrong).
- *If reasoning TTFT remains >80%*: Very long context overhead formula is insufficient; may need attention-pattern-specific term or batch-heterogeneity term.
- *If Scout experiments still fail*: Simulator bugs not fully resolved, or MoE gating formula (β₆) is structurally wrong.

---

## H-ablation-1: β₇ (Very Long Context) Importance

**Prediction**: Removing β₇ (very long context overhead) will degrade:
- Overall loss by >10 percentage points (from <80% to >90%)
- TTFT RMSE by >15 percentage points (from <40% to >55%)
- Reasoning experiments specifically will degrade by >30% TTFT (from <60% to >90%)

**Causal Mechanism**: β₇ is the ONLY term capturing overhead for prompts >4096 tokens. Without it, reasoning experiments (which have the longest prompts) will revert to catastrophic TTFT underprediction (~100% as in iter1), forcing β₀ to stay at physically implausible 0.203.

**Diagnostic Clause**: *If removing β₇ degrades loss by <5%*: Very long context overhead is already captured by another term (likely β₀ or β₂), making β₇ redundant. Consider merging with β₀ as a piecewise prefill efficiency term.

---

## H-ablation-2: β₈ (Per-Request Decode Overhead) Importance

**Prediction**: Removing β₈ (per-request decode overhead) will degrade:
- Overall loss by >5 percentage points (from <80% to >85%)
- E2E RMSE by >8 percentage points (from <40% to >48%)
- Small-batch decode experiments (batch_size <8) will degrade by >15% ITL

**Causal Mechanism**: β₈ normalizes the currently inflated β₁=1.553. Without β₈, the optimizer will keep β₁ inflated (>1.5) to compensate for missing per-request overhead, preventing β₁ from reaching physically plausible range (0.6-0.9).

**Diagnostic Clause**: *If removing β₈ degrades loss by <2%*: Per-request overhead is negligible (<5μs per request) or already absorbed by β₂ (constant scheduler overhead). The β₁ inflation has a different cause (e.g., decode FLOPs formula or memory bandwidth formula is wrong).

---

## H-coefficient-normalization: Physical Plausibility Recovery

**Prediction**: With β₇ and β₈ added, the following coefficients will move toward physically plausible ranges:
- **β₀ (prefill MFU)**: Will rise from 0.203 to **0.40-0.55** (still below ideal 0.6-0.8, but improved)
- **β₁ (decode memory-bound MFU)**: Will drop from 1.553 to **0.60-0.90** (physically plausible range)
- **β₂ (scheduler overhead)**: Will rise from 0.12μs to **5-50μs** (physically plausible range)

**Causal Mechanism**:
- β₇ absorbs reasoning prefill overhead that was forcing β₀ down
- β₈ absorbs decode per-request overhead that was forcing β₁ up
- With these two terms capturing missing physics, the optimizer no longer needs to distort β₀, β₁, β₂ to minimize loss

**Diagnostic Clause**:
- *If β₀ remains <0.3*: Still missing a major prefill overhead term (possibly activation bandwidth or prefix cache dynamics)
- *If β₁ remains >1.3*: Per-request overhead is larger than expected, or decode FLOPs formula has structural issues
- *If β₂ remains <1μs*: Constant scheduler overhead is genuinely negligible, or overhead is truly per-request (captured by β₈) not constant

---

## H-scout-recovery: Post-Bug-Fix MoE Validation

**Prediction**: With simulator bugs fixed (interleaved MoE architecture, `intermediate_size_mlp` parsing, MoE gating FLOPs calculation), Scout experiments will:
- Achieve <60% combined loss (vs 200% in iter1 validation failure)
- Have TTFT APE <50% (vs 100% in iter1)
- Have E2E APE <50% (vs 100% in iter1)
- β₆ (MoE gating) will converge to **0.005-0.015** (vs 0.008 in iter1, which was inflated by bug compensation)

**Causal Mechanism**:
- Iter1 Scout failures were NOT data quality issues — they were simulator bugs:
  1. **Interleaved architecture ignored**: BLIS treated all 48 layers as MoE (should be 24 MoE + 24 dense)
  2. **Dense FFN dim wrong**: BLIS used 8192 for all layers (should be 16384 for dense layers)
  3. **MoE gating FLOPs wrong**: Used dimensionless count × 1e6 instead of actual FLOPs / peak_TFLOPS

- With bugs fixed, Scout experiments will validate properly, and β₆ (MoE gating) will reflect true overhead (~0.5-1.5% of step time)

**Code Citations**:
- **Bug fix commits**: `8bd79f56` (identifies bugs), `2308663e` (fixes MoE gating calculation)
- **Scout architecture**: 24 MoE layers (16 experts, top-1, 8192 FFN per expert) + 24 dense layers (16384 FFN)
- **BLIS fixes**: `sim/model_hardware_config.go:InterleaveMoELayerStep` field added, `sim/latency/roofline.go` now computes separate FLOPs/bandwidth for MoE vs dense layers

**Diagnostic Clause**:
- *If Scout experiments still achieve >100% combined loss*: Bugs not fully fixed, or a NEW bug introduced by the fixes
- *If β₆ converges to >0.03*: MoE gating overhead is higher than expected, or gating FLOPs formula still wrong
- *If β₆ converges to <0.003*: MoE gating overhead is negligible (unlikely given H-moe-parity in iter0 showed 12.8% overhead)

---

## H-boundary: Sigmoid Interpolation Smoothness

**Prediction**: Experiments with medium batch sizes (5-12 decode requests) will have:
- <30% E2E APE (improved from iter1's potential discontinuity at batch_size=8 threshold)
- Smoother error distribution across batch sizes 4-16 (no jump at batch_size=8)

**Causal Mechanism**: Discrete regime split (`if batch_size < 8 use β₁ else use β₆`) causes prediction discontinuity: a batch with 7 requests and 8 requests have identical compute/memory characteristics but get different formulas. Sigmoid interpolation `memory_weight(n) = 1/(1+exp((n-8)/2))` smoothly transitions from memory-bound (small batch) to compute-bound (large batch), preventing discontinuity artifacts.

**Diagnostic Clause**: *If medium-batch experiments (5-12 requests) still have >40% E2E APE*: Transition is not smooth enough (adjust sigmoid slope parameter) OR the two-regime model is fundamentally wrong (decode may always be a mix of memory and compute, not a regime switch).

---

## H-error-pattern: Workload-Specific Improvements

**Prediction**: Error pattern changes from iter1:
- **Reasoning experiments**: TTFT will improve from ~100% to <60% (β₇ captures very long context overhead)
- **Scout experiments**: Combined loss will improve from 200% to <60% (simulator bugs fixed)
- **Roleplay experiments**: Maintain <50% combined loss (already well-predicted in iter1: 30-46%)
- **Codegen/general experiments**: Maintain <50% combined loss for TP=1,2 (already well-predicted in iter1: 12-48%)

**Causal Mechanism**:
- Reasoning: β₇ captures the unique prefill overhead for prompts >4096 tokens (CoT chains)
- Scout: Simulator bugs fixed, so MoE experiments will validate properly
- Roleplay/codegen/general: Already predicted well in iter1 (β₄ KV management + β₃ TP comm worked), so iter2 changes should maintain accuracy

**Diagnostic Clause**:
- *If reasoning TTFT remains >80%*: Very long context overhead formula is insufficient or wrong functional form
- *If Scout combined loss remains >100%*: Simulator bugs not fully resolved
- *If roleplay/codegen/general degrade by >20%*: New terms (β₇, β₈) or sigmoid interpolation introduced regression

---

## H-robustness: TP Configuration Generalization

**Prediction**: Iter2 will maintain accuracy across TP configurations:
- **TP=1 experiments** (Llama-2, Qwen, Mistral): <50% combined loss
- **TP=2 experiments** (Yi-34B, Scout): <60% combined loss
- **TP=4 experiments** (Llama-3.1-70B): <80% combined loss

**Causal Mechanism**: β₃ (TP communication) from iter1 captures all-reduce overhead scaling with TP. Iter2 doesn't modify β₃, so TP generalization should remain intact. The new terms (β₇, β₈) are TP-independent (very long context and per-request overhead don't scale with TP).

**Diagnostic Clause**: *If TP=4 experiments degrade by >20%*: TP communication formula (β₃) is wrong or insufficient. May need TP-dependent variants of new terms (e.g., per-request overhead increases with TP due to synchronization).

---

## Expected Iteration 2 Outcomes

**Overall loss target**: <80% (from 134.54% in iter1)
- TTFT RMSE target: <40% (from 69.29% in iter1)
- E2E RMSE target: <40% (from 65.24% in iter1)

**Per-experiment targets**:
- Reasoning experiments: TTFT <60% (from ~100%)
- Scout experiments: Combined loss <60% (from 200% validation failure)
- Dense non-reasoning experiments: Maintain <50% combined loss (7/11 already achieved this in iter1)

**Coefficient targets**:
- β₀: 0.40-0.55 (up from 0.203, toward physical range)
- β₁: 0.60-0.90 (down from 1.553, toward physical range)
- β₂: 5-50μs (up from 0.12μs, toward physical range)
- β₃: 0.3-0.5 (stable from iter1's 0.394)
- β₄: Maintain critical importance (ablation showed +30.28% E2E degradation without it)
- β₅: 0.5-0.8 (stable from iter1's β₆=0.651)
- β₆: 0.005-0.015 (MoE gating, down from iter1's 0.008 inflated by bug compensation)
- β₇: 0.5-2.0 (NEW, very long context overhead scaling)
- β₈: 10-50μs per request (NEW, per-request decode overhead)

**If loss remains >90%**: Missing terms are activation bandwidth, batch heterogeneity overhead, or attention pattern complexity. Consider Phase 0.5 experiment: profile vLLM to measure actual overhead breakdown.

**If iter2 hits <80% target**: Proceed to iteration 3 with further refinements (potentially add activation bandwidth term or batch composition variance term).
