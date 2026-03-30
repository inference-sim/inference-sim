# Iteration 6: Scheduler Overhead Hypothesis

## H-main: Per-Request Scheduler Overhead

**Prediction**: Overall loss will decrease to <110% (from 603% in iter5, from 129% in iter4), with TTFT RMSE <50% (from 519% in iter5, from 66% in iter4) and E2E RMSE <60% (from 84% in iter5, from 63% in iter4).

**Target experiments**: Reasoning workloads will see largest improvement in TTFT APE:
- Llama-2-7B reasoning: 99.76% → <40% TTFT APE
- Qwen2.5-7B reasoning: 99.85% → <45% TTFT APE
- Scout reasoning: 99.66% → <40% TTFT APE

Short-context experiments will recover to iter4 levels:
- Llama-3.1-70B codegen: 1091% → <25% TTFT APE (iter4: 3.86%)
- Mistral codegen: 834% → <35% TTFT APE (iter4: 30.69%)
- Llama-2 roleplay: 823% → <40% TTFT APE (iter4: 29.20%)

**Causal Mechanism**:

vLLM's continuous batching scheduler introduces **per-request fixed overhead** (50-150ms) that dominates TTFT for short-context prefills (1K tokens, theoretical ~1-5ms):

1. **Request arrival → scheduler queue**: Incoming requests wait in priority queue until batch formation threshold reached (typical: wait for 4-8 requests or 10-50ms timeout)

2. **Batch formation overhead**: Scheduler computes can-schedule decision (KV cache capacity check, GPU memory validation, priority ordering) for each request. For reasoning workloads with high concurrency (multi-turn chat), this adds 20-50ms per request.

3. **KV cache block allocation**: PagedAttention allocates physical blocks from block manager (typical: 10-20 blocks for 1K tokens × 1 beam = 80-400KB). Allocation traverses free list, updates block tables, zeros memory. This adds 10-30ms per request.

4. **Variance from batching**: Requests arriving when batch is full wait for next batch cycle (explains p10=0.13ms for immediate processing, p90=215ms for waiting 200ms). Reasoning workload (multi-turn, high concurrency) experiences longer waits than codegen (single-turn, lower concurrency).

**Why iter5 failed**: Added per-layer overhead (`β₆ × num_layers × (1.0 + tokens/2048)`) that:
- Scaled with num_layers (32-80 layers) → over-predicted large models by 100-500ms
- Applied base factor (1.0) to ALL contexts including 1K tokens → catastrophically broke short-context predictions (1091% TTFT for Llama-3.1)
- Converged too low (521μs vs expected 2000μs) because optimizer penalized by 11 short-context experiments

**Why iter6 should work**: Replace per-layer overhead with per-request scheduler overhead:
- Fixed cost per request (50-150ms) independent of num_layers → no over-prediction for large models
- Only applies during queuing phase (QueueingTime), not step execution (StepTime) → no interaction with prefill compute β₀
- Physically grounded: vLLM scheduler overhead is per-request (batch formation, KV allocation), not per-layer (kernel launch)

**Code Citations**:
- vLLM scheduler: `vllm/core/scheduler.py:Scheduler._schedule()` (batch formation, ~10-50ms per cycle)
- Block allocation: `vllm/core/block_manager.py:BlockSpaceManager.allocate()` (allocates KV blocks, ~10-30ms per request)
- Priority queue: `vllm/core/scheduler.py:SchedulerRunningOutputs` (processes waiting queue)
- BLIS: `sim/latency/evolved_model.go:QueueingTime()` (currently only models α₀ + α₁×input_len, missing scheduler overhead)

**Diagnostic Clause**: *If this fails, it indicates that reasoning's 100-200ms TTFT is NOT due to scheduler overhead, but rather: (1) prefix cache misses (shared system prompt not cached), (2) attention kernel startup cost (FlashAttention-2 fixed cost), or (3) memory allocation beyond KV blocks (activation buffers, temporary tensors). Investigate by decomposing traces into scheduler time vs kernel execution time.*

---

## H-ablation-scheduler: Scheduler Overhead Dominates for 1K-Token Prefills

**Prediction**: The new β₆ (scheduler overhead per request) will converge to 50-150ms, accounting for 50-75% of reasoning's 100-200ms TTFT gap (simulator currently predicts ~1-5ms for 1K-token prefill).

**Mechanism**: For short prefills (1K tokens), scheduler overhead (batch formation + KV allocation) should dominate over prefill compute time:
- Theoretical prefill compute: ~1-5ms (1K tokens, 32 layers, β₀=0.25-0.35 MFU)
- Actual TTFT: 100-200ms
- Scheduler overhead should be: 95-195ms (captures 95% of gap)

**Evidence**: Trace analysis shows reasoning experiments have huge variance:
- Llama-2 reasoning: p10=0.13ms, p90=215ms (1650× variance)
- This variance pattern matches batching delay (immediate vs waiting for batch formation)

**Diagnostic Clause**: *If β₆ converges to <30ms, scheduler overhead is not the dominant factor. Check for: (1) prefix cache misses, (2) attention kernel overhead, or (3) memory bandwidth bottleneck during KV write.*

---

## H-boundary: Scheduler Overhead is Per-Request, Not Per-Layer

**Prediction**: Short-context experiments (1K tokens) will recover to iter4 accuracy levels (<40% TTFT APE) because new β₆ is per-request (no scaling with num_layers):
- Llama-3.1-70B (80 layers, 1K tokens): 1091% → <25% TTFT
- Llama-2-7B (32 layers, 1K tokens): 328% → <35% TTFT

The improvement should be **uniform across all num_layers** (32, 40, 56, 80 layers), unlike iter5 where degradation was proportional to num_layers.

**Mechanism**: Iter5's β₆ scaled with num_layers → 80-layer models degraded catastrophically (1091% TTFT). Iter6's β₆ is per-request (independent of num_layers) → should improve all short-context experiments uniformly.

**Diagnostic Clause**: *If degradation still correlates with num_layers, the per-request overhead model is wrong. Investigate whether scheduler overhead actually scales with model complexity (e.g., larger models require more GPU memory validation, longer block allocation).*

---

## H-error-pattern: Reasoning Improves While Short-Context Recovers

**Prediction**: Reasoning experiments (currently 99% TTFT) will see 55-60pp improvement (99% → 40-45%), while short-context experiments (currently 200-1000% TTFT) will recover 150-900pp (back to iter4 levels of 4-40%).

**Mechanism**: Two orthogonal improvements:
1. **Reasoning improvement**: Adding β₆ (50-150ms scheduler overhead) captures the missing 100-200ms gap for 1K-token prefills
2. **Short-context recovery**: Removing iter5's per-layer overhead (which applied `1.0 + tokens/2048` base factor) eliminates catastrophic over-prediction for 1K tokens

**Evidence**: Iter5 showed inverse effect:
- Reasoning barely improved (99.75% → 99.45%, only 0.3pp)
- Short-context catastrophically degraded (4-77% → 200-1091%, 150-1000pp worse)

Iter6 should show opposite:
- Reasoning improves significantly (scheduler overhead captures gap)
- Short-context recovers (no per-layer overhead applied to 1K tokens)

**Diagnostic Clause**: *If reasoning improves but short-context doesn't recover, β₀ may have risen too high (>0.35). If short-context recovers but reasoning doesn't improve, scheduler overhead β₆ is too low (<30ms) or wrong mechanism.*

---

## H-coefficient-stability: Coefficients Revert to Iter3 Physical Ranges

**Prediction**: With scheduler overhead decoupled from prefill compute, other coefficients will stabilize to iter3 ranges:
- β₀: 0.15-0.25 (iter3: 0.169, iter4: 0.165, iter5: 0.266 too high)
- β₁: 1.00-1.10 (iter3: 1.037, iter4: 1.802, iter5: 1.449 improving)
- β₂: 0.30-0.35 (iter3: 0.318, iter4/iter5: 1.36-1.37 stuck high)
- β₃: Recover from 0.000013 → 0.0004-0.0005 (KV management overhead)
- β₄: 0.75-0.85 (iter3: 0.796, iter5: 0.620 implausibly low)
- β₅: 0.01-0.012 (iter3: 0.0117, iter5: 0.0149 close)

**Mechanism**: Iter5's collinearity between β₀ and per-layer β₆ caused coefficient drift:
- β₀ rose to 0.266 (predicted time decreased 38%)
- β₃ collapsed to 0.000013 (KV management became redundant)
- β₄ dropped to 0.620 (physically implausible, balancing β₁ being too high)

Iter6 decouples scheduler overhead (in QueueingTime) from prefill compute (in StepTime), eliminating collinearity.

**Diagnostic Clause**: *If β₀ doesn't drop to 0.15-0.25, collinearity persists (scheduler overhead may be leaking into StepTime). If β₂ stays at 1.36, missing TP-dependent prefill overhead exists (not just decode communication).*

---

## Implementation Summary

**Changes from iter5**:

1. **Move β₆ from StepTime to QueueingTime**: Scheduler overhead is per-request (queuing phase), not per-step (kernel execution)
2. **Remove chunking scale factor**: No `(1.0 + tokens/2048)` term - scheduler overhead is constant per request
3. **Expected β₆ range**: 50-150ms per request (not 500-3000μs per layer)
4. **Revert Alpha to iter4**: α₁=0.000125, α₂=0.000036 (iter5 Alpha exploded to absorb error)
5. **Warm-start Beta from iter5** (except β₀): β₁=1.449, β₂=1.368, β₃=0.0005 (iter4, not iter5's 0.000013), β₄=0.796 (iter3), β₅=0.0149
6. **Constrain β₀ bounds**: [0.10, 0.35] (prevent rising too high like iter5)

**Expected iter6 outcome**:
- Overall loss: 603% → 90-110% (recovering iter4 short-context + gaining reasoning)
- TTFT RMSE: 519% → 40-50% (reasoning 99% → 40-45%, short-context recover to iter4)
- E2E RMSE: 84% → 55-60% (following TTFT improvement)
- All coefficients: Return to physical plausibility (β₀: 0.15-0.25, β₁: 1.00-1.10, β₃/β₄ recover)

**Risk mitigation**: If β₆ converges to <30ms (scheduler overhead insufficient), diagnostic clause directs investigation to: (1) prefix cache misses, (2) attention kernel startup, (3) memory bandwidth bottleneck. Post-iter6 analysis will guide iter7 hypothesis.
