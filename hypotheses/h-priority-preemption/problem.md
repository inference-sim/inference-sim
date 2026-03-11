# Strategy Evolution: Priority-Based Preemption (Iteration 3)

**Iteration:** 3 (breakthrough: batch composition layer, not queue ordering)
**Date:** 2026-03-10
**Branch:** `hypothesis-playground`

---

## Phase 1: Problem Framing

### The Bottleneck (from Iterations 1-2)

Critical TTFT P99 at 120% capacity is ~128ms. Decomposition:
- Alpha overhead: ~5ms
- Prefill step time: ~25ms (round-2, 1024 tokens)
- **Queue wait: ~98ms** (waiting for batch slots to free up)

The queue wait exists because `VLLMBatchFormation` Phase 2 will not schedule new requests when `len(RunningBatch) >= MaxRunningReqs`. Even though critical requests are FIRST in the queue (thanks to StaticClassWeight), they cannot enter the batch until a running request completes and frees a slot. Each step takes ~7-12ms, so 8-10 steps of waiting = ~98ms.

**Neither scheduling (Iter 1) nor admission (Iter 2) can address this.** Scheduling reorders the queue but can't bypass the batch capacity gate. Admission reduces total traffic but critical requests still wait for batch slots occupied by running sheddable requests.

### The Breakthrough: Priority Preemption

When a high-priority request is waiting and the batch is full, **evict the lowest-priority running request** to make room. The preempted request goes back to the queue with reset progress (BLIS recomputation mode).

This operates at the **batch composition layer** — a fundamentally different lever than queue ordering (scheduling) or traffic management (admission).

**Predicted effect:** Critical requests enter the batch in the NEXT step (~7ms) instead of waiting for natural batch turnover (~98ms). Critical TTFT should drop from ~128ms to ~37ms (alpha + prefill + 1 step wait) — a **~70% improvement**.

### Baselines

| ID | Priority | Admission | Batch Formation | Purpose |
|----|----------|-----------|----------------|---------|
| **B2** | StaticClassWeight(10,5,1) | AlwaysAdmit | VLLMBatchFormation (standard) | Iteration 1 winner |
| **T3** | StaticClassWeight(10,5,1) | AlwaysAdmit | VLLMBatchFormation (priority preemption enabled) | Treatment |

Single dimension varied: priority preemption on/off (ED-1 compliant).

### Parameters (1)

| Parameter | Value | Derivation |
|-----------|-------|-----------|
| `PriorityPreemptionMargin` | 5.0 | critical(10) - sheddable(1) = 9 > 5.0: triggers. standard(5) - sheddable(1) = 4 < 5.0: does not trigger. Only critical preempts sheddable. |

**S3 compliance:** 1 parameter (well under ≤7).

### Risk: Livelock (S15/R19)

S15 found "no moderate regime" for KV-pressure preemption. Priority preemption is different:
- **Bounded trigger rate:** Only critical requests (20% of traffic) can trigger preemption.
- **Bounded cascade depth:** Critical can preempt sheddable, but sheddable can't preempt anyone. Max 1 level.
- **Preemption cost:** Recomputation mode resets ProgressIndex to 0. For sheddable with 256-token input: ~11ms wasted. For round-2 sheddable (1024 tokens): ~25ms wasted.
- **Circuit breaker:** If preemption count per step exceeds 3, stop preempting (prevents burst-induced cascades).

### Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Critical TTFT P99 improvement: T3 vs B2 | >50% | Batch-level preemption should dramatically reduce queue wait |
| Sheddable TTFT P99 degradation: T3 vs B2 | <100% (2x) | Sheddable pays the cost but within bounds |
| Throughput change | <10% | Some throughput loss from recomputation overhead acceptable |

### Hypothesis Bundle (3 arms)

**H-main:** T3 reduces critical TTFT P99 by >50% over B2 at 120% because priority preemption eliminates the batch-occupancy queue wait.
*If this fails, the batch occupancy is not the bottleneck — queue wait comes from elsewhere.*

**H-zero-sum:** T3's cluster TTFT P99 is within 20% of B2 — the critical improvement outweighs the sheddable degradation for cluster-level metrics.

**H-control-negative:** With uniform SLO (all 'standard', priority margin not met), T3 produces <5% difference from B2.
