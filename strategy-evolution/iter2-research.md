# Iteration 2 Research: Building on SLO-Gated Priority Cascade

## Context: What Iteration 1 Proved

Iteration 1 established that **SLO-tiered priority is the primary differentiator** for orthogonal workloads. The results:

| Metric | Baseline | Iter 1 | Delta |
|--------|----------|--------|-------|
| Critical TTFT P99 | 268.8 ms | 132.3 ms | **-50.8%** |
| Standard TTFT P99 | 271.0 ms | 189.9 ms | -29.9% |
| Sheddable TTFT P99 | 264.8 ms | 466.1 ms | **+76.0%** |
| Cluster TTFT P99 | 269.1 ms | 437 ms | **+62.4%** |
| SLO Gap | 0.99x | 3.52x | +256% |
| Throughput | 17,198 tps | 17,213 tps | ~0% |

**Key finding**: Priority reordering is a zero-sum game at the scheduling level -- improving critical TTFT comes directly at the cost of sheddable TTFT. The cluster-wide P99 is dominated by the sheddable tail, which got 76% worse. Throughput is perfectly preserved because reordering does not discard work.

**The Iteration 2 challenge**: Maintain or improve the 3.52x SLO gap while bringing cluster-wide P99 back below baseline (269 ms). This requires either:
- (a) Making sheddable requests faster WITHOUT reducing the critical advantage, or
- (b) Creating a non-zero-sum mechanism that improves the entire system, with critical requests benefiting more.

## Root Cause Analysis: Why Cluster P99 Degraded

The cluster-wide TTFT P99 is the max across all SLO tiers weighted by population:
- 30% of requests are sheddable
- Sheddable TTFT P99 went from 264.8 ms to 466.1 ms (+201 ms)
- This dominates the cluster P99 (437 ms) because 30% > 1% tail

The degradation mechanism: `PriorityFCFSScheduler` pushes sheddable requests to the back of every queue ordering. During burst arrivals (gamma CV=2.0), a sheddable request arriving during a burst sits behind ALL critical and standard requests that arrived in the same burst window. The threshold mechanism (200ms grace before urgency activates) means sheddable requests accumulate no urgency advantage for the first 200ms, exactly when they need it most.

**Structural insight**: Priority reordering is a fixed pie -- scheduling delay removed from critical requests is added to sheddable requests. The only way to improve cluster-wide P99 is to make the pie smaller (reduce total scheduling delay for everyone) or to selectively improve throughput.

## Three Strategies for Iteration 2

---

## Idea A: Bounded Priority with Admission-Gated Shedding

### Core Mechanism

This strategy keeps the proven SLO-tiered priority from Iteration 1 but addresses the sheddable degradation through two new mechanisms:

1. **Bounded priority inversion ratio**: Instead of unlimited priority separation, cap the maximum scheduling delay ratio between any two tiers. When sheddable requests' average TTFT exceeds `degradation_cap * baseline_sheddable_ttft_estimate`, the priority policy dynamically compresses the base score gap. This creates a self-correcting feedback loop: as sheddable degradation approaches the cap, the priority gap shrinks, allowing some sheddable requests through, which reduces their tail latency.

   Concrete formula:
   ```
   effective_base[class] = base[class] * dampening_factor
   dampening_factor = min(1.0, degradation_cap / observed_ratio)
   observed_ratio = moving_avg_sheddable_wait / moving_avg_critical_wait
   ```
   When `observed_ratio < degradation_cap`, dampening = 1.0 (full priority separation). When it exceeds the cap, dampening < 1.0 (compressed priority).

2. **SLO-aware admission shedding**: Replace `always-admit` with a new `SLOGatedAdmission` policy that tracks cluster-wide load and selectively delays or rejects sheddable requests during extreme bursts. When cluster `avg(EffectiveLoad) > load_threshold`, sheddable requests are rejected (they can retry). This reduces queue depth at the source, benefiting ALL requests in the system.

   This is a non-zero-sum mechanism: by reducing the total request population during bursts, average scheduling delay decreases for everyone. Critical requests benefit from shorter queues. Sheddable requests that DO get admitted face less competition.

3. **Retain all Iter 1 components**: `slo-tiered` priority policy, `priority-fcfs` scheduler, `slo-priority` scorer, router priority bridge.

### Why It Improves on Iteration 1

- **Addresses cluster P99 directly**: The admission gate removes the worst-case burst pileups. During a gamma burst, sheddable requests that would have accumulated 400+ ms of scheduling delay are instead rejected at admission (instantaneous), preventing them from clogging the pipeline.
- **Non-zero-sum**: Admission shedding reduces queue depth for ALL tiers. This is the only mechanism that can improve both critical AND sheddable P99 simultaneously.
- **Bounded degradation**: The dampening factor prevents runaway sheddable degradation even without admission gating. The cap provides a tunable guarantee.
- **Throughput trade-off is explicit**: Rejected sheddable requests reduce throughput by at most `sheddable_fraction * rejection_rate`. With 30% sheddable traffic and 10% rejection rate, throughput drops ~3%.

### Parameterized Template (8 parameters)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `base_critical` | float64 | [5.0, 15.0] | Priority base score for critical |
| `base_standard` | float64 | [2.0, 8.0] | Priority base for standard |
| `base_sheddable` | float64 | [0.0, 3.0] | Priority base for sheddable |
| `age_weight` | float64 | [1e-6, 1e-4] | Shared age escalation rate |
| `threshold_sheddable` | int64 (us) | [50000, 500000] | Grace period before sheddable urgency |
| `degradation_cap` | float64 | [1.5, 4.0] | Max allowed sheddable/critical wait ratio |
| `load_threshold` | float64 | [3.0, 10.0] | Avg effective load triggering admission shedding |
| `slo_load_bias_critical` | float64 | [0.5, 1.0] | Load vs cache blend for critical routing |

### Experimentally Verifiable Hypotheses

- **HA1**: Cluster-wide TTFT P99 < 269 ms (baseline), while critical TTFT P99 < 160 ms (within 20% of Iter 1 best).
- **HA2**: At `degradation_cap=2.0`, sheddable TTFT P99 < 538 ms (2x baseline), regardless of other parameters.
- **HA3**: Admission rejection rate < 15% of sheddable requests (i.e., throughput drops < 5% from baseline).
- **HA4**: Removing admission gating (always-admit) while keeping bounded priority still improves cluster P99 vs Iter 1 (tests the dampening mechanism in isolation).

### Implementation Plan

| Step | File(s) | Change |
|------|---------|--------|
| 1 | `sim/priority.go` | Add `BoundedSLOPriority` policy with dampening factor. Needs a moving-average tracker (small ring buffer of per-step wait ratios). |
| 2 | `sim/priority.go` | Register `bounded-slo` in `NewPriorityPolicy` factory |
| 3 | `sim/admission.go` | Add `SLOGatedAdmission` policy that reads `req.SLOClass` and `state.Snapshots` |
| 4 | `sim/admission.go` | Register `slo-gated` in `NewAdmissionPolicy` factory |
| 5 | `sim/bundle.go` | Add `bounded-slo` to `validPriorityPolicies`, `slo-gated` to `validAdmissionPolicies` |
| 6 | `cmd/root.go` | CLI flags: `--degradation-cap`, `--load-threshold` |
| 7 | Retain | `slo-priority` scorer, priority bridge, `priority-fcfs` scheduler from Iter 1 |

Touch points: 4 files (6-7 changes). Medium complexity -- the ring-buffer moving average in BoundedSLOPriority is new state management.

### Self-Critique

1. **Moving average introduces statefulness in priority policy**: Current priority policies are pure functions of (req, clock). BoundedSLOPriority needs historical wait-time data, which either requires passing it through a side channel or making the priority policy struct mutable. This is an architectural change -- priority policies have been stateless until now.
2. **Observed_ratio is a lagging indicator**: The dampening factor reacts to PAST wait ratios, not current ones. During a sharp burst, the dampening may not engage until the burst has already caused damage.
3. **Admission shedding requires careful calibration of load_threshold**: Too low and you reject too much traffic (throughput hit). Too high and the gate never triggers (no benefit). The optimal value depends on the specific workload arrival rate, which varies.
4. **Request rejection changes INV-1 accounting**: Rejected requests increment `rejected_requests` counter. Conservation formula: `num_requests == injected_requests + rejected_requests`. This is already handled by the existing TokenBucket admission path, but SLOGatedAdmission needs to follow the same protocol.
5. **Throughput target (>5% improvement) becomes harder to hit**: Admission shedding reduces throughput by design. Need to ensure the improved scheduling efficiency compensates.

---

## Idea B: Preemption-Aware SLO Batch Formation

### Core Mechanism

This strategy attacks the problem at the batch formation layer -- the one module Iteration 1 did not touch. The insight: `VLLMBatchFormation` preempts from the batch TAIL (last-added request), which is SLO-oblivious. When KV pressure forces preemption, a critical request that was last to join the batch gets evicted just as readily as a sheddable one. Moreover, the conservative rule "stop dequeuing new requests if any preemption occurred" halts all new admissions for the step, which hurts critical requests waiting in the queue.

The new `SLOAwareBatchFormation` makes two targeted modifications:

1. **SLO-ordered preemption**: When preemption is needed, evict the lowest-priority request in the running batch instead of the tail. This is a single sort comparison change: instead of `RunningBatch.Requests[len-1]` (tail), find `argmin(req.Priority for req in RunningBatch.Requests)`. Critical requests are protected from eviction as long as any sheddable request is in the batch.

2. **Priority-gated dequeuing after preemption**: Relax the conservative "stop all dequeuing after preemption" rule. After preemption, continue dequeuing ONLY if the next queued request has priority > `min_dequeue_priority_after_preemption` (a tunable threshold). This allows high-priority critical requests to enter the batch even when a preemption just occurred, while still blocking low-priority sheddable requests (maintaining the safety property).

3. **Retain all Iter 1 components**: The same priority scores computed by `slo-tiered` priority policy drive both scheduling order AND preemption order AND post-preemption dequeue gating. This is genuine cross-layer synergy -- priority flows from router -> scheduler -> batch formation.

### Why It Improves on Iteration 1

- **Non-zero-sum at the batch level**: Currently, preemption is random with respect to SLO class (tail eviction). Making it SLO-aware means preemptions preferentially remove sheddable work, freeing KV for critical work. The total preemption count may stay the same, but the distribution shifts toward sheddable (which has higher tolerance).
- **Addresses the "preemption halts everything" bottleneck**: The current conservative rule means ONE preemption blocks ALL dequeuing. With priority-gated dequeuing, critical requests bypass this block. This is most impactful during KV-pressure episodes.
- **Works even at zero preemptions**: At the baseline workload (0 preemptions with 132K blocks), the preemption change has no effect. But the priority-gated dequeuing STILL matters because it interacts with the token budget exhaustion path (Phase 2 loop termination condition). The gating logic applies to any dequeue-blocking condition, not just preemption.
- **Amplifies Iter 1 priority**: Iter 1's priority only affects queue ordering (which request is NEXT). This extends priority into batch composition (which request STAYS in the batch). The signal coherence strengthens as priority penetrates deeper into the pipeline.

### Parameterized Template (7 parameters)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `base_critical` | float64 | [5.0, 15.0] | Priority base score for critical (from Iter 1) |
| `base_standard` | float64 | [2.0, 8.0] | Priority base for standard (from Iter 1) |
| `base_sheddable` | float64 | [0.0, 3.0] | Priority base for sheddable (from Iter 1) |
| `age_weight` | float64 | [1e-6, 1e-4] | Shared age escalation rate (from Iter 1) |
| `threshold_sheddable` | int64 (us) | [50000, 500000] | Grace period for sheddable urgency (from Iter 1) |
| `min_dequeue_priority` | float64 | [3.0, 12.0] | Priority floor for post-preemption dequeuing |
| `preemption_kv_headroom` | float64 | [0.85, 0.95] | Lower offload threshold when batch has mixed SLO |

### Experimentally Verifiable Hypotheses

- **HB1**: Under KV pressure (reduce `--total-kv-blocks` to 5000), critical preemption rate < 0.2x of sheddable preemption rate (SLO-ordered preemption discriminates correctly).
- **HB2**: At baseline KV blocks (132K, zero preemptions), priority-gated dequeuing still improves critical TTFT P99 by >5% vs Iter 1 (non-preemption dequeue path benefits).
- **HB3**: Critical TTFT P99 < 130 ms while sheddable TTFT P99 < 400 ms (improving BOTH vs Iter 1: critical from 132.3 ms, sheddable from 466.1 ms).
- **HB4**: At 80% capacity, throughput within 3% of baseline (preemption reordering does not discard more work than tail eviction).

### Implementation Plan

| Step | File(s) | Change |
|------|---------|--------|
| 1 | `sim/batch_formation.go` | New `SLOAwareBatchFormation` struct implementing `BatchFormation` interface |
| 2 | `sim/batch_formation.go` | `preemptForTokens` override: find argmin(Priority) instead of batch tail |
| 3 | `sim/batch_formation.go` | Phase 2 loop: replace `!result.PreemptionHappened` with priority-gated condition |
| 4 | `sim/batch_formation.go` | `NewBatchFormation` factory: accept name parameter, return `VLLMBatchFormation` or `SLOAwareBatchFormation` |
| 5 | `sim/config.go` | Add `BatchFormationPolicy string` to `BatchConfig` |
| 6 | `sim/bundle.go` | Validation: `validBatchFormationPolicies` map |
| 7 | `cmd/root.go` | CLI flags: `--batch-formation`, `--min-dequeue-priority`, `--preemption-kv-headroom` |
| 8 | Retain | All Iter 1 components (priority policy, scorer, scheduler, bridge) |

Touch points: 4 files (7-8 changes). Medium-High complexity -- new BatchFormation implementation is the highest-friction extension type.

### Self-Critique

1. **Batch formation is the highest-friction extension point** (4+ touch points per the extension recipes). This is architecturally justified but implementation-heavy. A second `BatchFormation` implementation is actually architecturally positive (R13: multi-impl interfaces), but it's more work.
2. **At baseline load (0 preemptions), the preemption ordering change has zero effect**: The primary value of this strategy requires KV pressure, which our baseline workload does not produce (132K blocks is abundant). HB2 tests whether the dequeue-gating path alone provides value at baseline load. If HB2 is refuted, this strategy only adds value under KV pressure scenarios.
3. **argmin(Priority) preemption is O(n) per preemption event**: With batch sizes of 100-200 requests, this is negligible. But it changes the preemption contract from O(1) to O(n). At very large batch sizes, this could matter.
4. **Priority-gated dequeuing may starve sheddable requests more aggressively than Iter 1**: If `min_dequeue_priority` is set above sheddable base priority, sheddable requests can only enter the batch when no preemption has occurred in that step. This is actually more restrictive than Iter 1 (where dequeuing stops entirely for all tiers). The net effect on sheddable TTFT depends on whether the indirect benefit (shorter queues from critical requests draining faster) outweighs the direct cost (blocked dequeuing).
5. **Requires `req.Priority` to be set BEFORE batch formation**: This is already the case -- `scheduleBatch()` computes priorities and reorders before calling `FormBatch()`. But it creates a tighter coupling between the priority policy and the batch formation strategy.

---

## Idea C: Load-Regime Adaptive Priority with Opportunistic Sheddable Batching

### Core Mechanism

This strategy directly addresses Iteration 1's fundamental weakness: priority differentiation is wasteful at sub-saturation and too aggressive near saturation. H23 proved that all policies converge within 4.4% at low load. H7 proved that near-saturation effects are super-linear. The insight: **the optimal priority gap is a function of current load, not a fixed parameter**.

Three components:

1. **Load-regime-adaptive priority**: The priority base score gap between critical and sheddable scales with the current load regime. At sub-saturation (all instances have EffectiveLoad < `low_load_threshold`), the gap compresses toward zero (approaching FCFS, which H23 shows is optimal). At high load (any instance has EffectiveLoad > `high_load_threshold`), the gap expands to full Iter 1 levels. Between thresholds, linear interpolation.

   ```
   load_factor = clamp((max_instance_load - low_threshold) / (high_threshold - low_threshold), 0, 1)
   effective_gap = base_gap * load_factor
   priority = base_sheddable + effective_gap * class_weight[class] + urgency
   ```

   When `load_factor = 0`: all classes get `base_sheddable + urgency` (effectively FCFS with age tiebreaking). When `load_factor = 1`: full SLO differentiation (Iter 1 behavior).

2. **Opportunistic sheddable batching via reduced prefill chunking**: When the system is at high load AND sheddable requests are accumulating in the queue, temporarily increase `longPrefillTokenThreshold` for sheddable requests. This allows their prefills to complete in fewer steps (fewer beta0 overheads), improving their throughput at the cost of slight HOL blocking. Since critical requests have priority and enter the batch first, they are unaffected by the longer sheddable prefill chunks.

   Mechanism: `SLOAwareBatchFormation` (from Idea B) checks the priority of each request entering Phase 2. For requests with priority below `sheddable_priority_threshold`, it uses a higher `PrefillTokenThreshold` (e.g., 512 instead of 256). This reduces the number of steps needed for sheddable prefills from ceil(768/256)=3 to ceil(768/512)=2, saving one beta0 (6.9ms) per sheddable request.

3. **Load-aware router signal injection**: The priority policy needs to observe cluster-wide load to compute `load_factor`. But priority policies only see `(req, clock)`. Solution: inject the load signal through the existing router priority bridge. The `WeightedScoring` router already computes `EffectiveLoad` per instance. Extend the bridge to set `req.Priority` incorporating the load regime, and have the instance-level priority policy use this as a hint.

   Alternatively (simpler): make the priority policy observe instance-local queue depth. Each instance's priority policy can read `len(WaitQ.Items())` as a proxy for load. This avoids cross-layer communication entirely -- the priority policy just needs access to the queue it already operates on.

### Why It Improves on Iteration 1

- **Eliminates unnecessary sheddable degradation at sub-saturation**: When the system is lightly loaded (most of the time in bursty workloads between bursts), sheddable requests face no priority penalty. They only face priority competition during bursts, when differentiation actually matters. This directly reduces sheddable TTFT P99 because P99 is dominated by burst-period behavior, and the inter-burst periods now contribute baseline-equivalent latency.
- **Addresses the fixed-pie problem**: By adapting the priority gap to load, the system spends less total "priority budget" on differentiation. At sub-saturation, zero budget is spent (FCFS). At high load, full budget. The time-averaged sheddable degradation is lower than Iter 1's constant full-gap approach.
- **Opportunistic batching improves throughput**: Reducing per-request step count for sheddable requests (via higher chunking threshold) directly improves throughput. Each saved beta0 overhead (6.9ms per step) is pure efficiency gain. With 30% sheddable traffic and 1 step saved per request, this is ~30% * 6.9ms / E2E per request throughput improvement.
- **Builds on proven H23 and H7 findings**: H23 (low-load equivalence) proves FCFS is optimal at sub-saturation. H7 (super-linear scaling) proves that reducing effective load at near-saturation yields outsized benefits. This strategy directly implements both findings.

### Parameterized Template (8 parameters)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `base_gap` | float64 | [5.0, 15.0] | Maximum priority gap (critical - sheddable) at full load |
| `base_floor` | float64 | [0.0, 3.0] | Base priority for all classes at zero load |
| `age_weight` | float64 | [1e-6, 1e-4] | Shared age escalation rate |
| `low_load_threshold` | int | [1, 5] | Queue depth below which gap is zero |
| `high_load_threshold` | int | [10, 50] | Queue depth above which gap is maximum |
| `sheddable_prefill_threshold` | int64 | [256, 1024] | Higher chunking threshold for sheddable prefills |
| `threshold_sheddable` | int64 (us) | [50000, 500000] | Grace period for sheddable urgency |
| `critical_class_weight` | float64 | [0.7, 1.0] | Fraction of gap allocated to critical (vs standard) |

### Experimentally Verifiable Hypotheses

- **HC1**: At sub-saturation (rate=500, 25% capacity), SLO gap < 10% (approaching FCFS equivalence from H23), while at near-saturation (rate=2000, baseline), SLO gap > 2.5x.
- **HC2**: Cluster-wide TTFT P99 < 300 ms at baseline load (better than Iter 1's 437 ms), while critical TTFT P99 < 160 ms.
- **HC3**: Sheddable TTFT P99 < 350 ms at baseline load (better than Iter 1's 466 ms), demonstrating reduced degradation from load-adaptive gap.
- **HC4**: Throughput > 17,500 tps (>1.5% improvement over baseline) from the opportunistic batching efficiency gain.

### Implementation Plan

| Step | File(s) | Change |
|------|---------|--------|
| 1 | `sim/priority.go` | New `LoadAdaptivePriority` policy. Observes queue depth via a callback (injected at construction) to compute `load_factor`. |
| 2 | `sim/priority.go` | Register `load-adaptive` in `NewPriorityPolicy` factory |
| 3 | `sim/batch_formation.go` | New `SLOAwareBatchFormation` (shared with Idea B) with SLO-aware chunking threshold |
| 4 | `sim/batch_formation.go` | Phase 2 loop: check `req.Priority` for sheddable threshold, use higher `PrefillTokenThreshold` |
| 5 | `sim/simulator.go` | Wire queue-depth callback from Simulator to priority policy at construction |
| 6 | `sim/config.go` | Add `SheddablePrefillThreshold int64` to `BatchConfig` |
| 7 | `sim/bundle.go` | Validation maps |
| 8 | `cmd/root.go` | CLI flags: `--load-adaptive-low-threshold`, `--load-adaptive-high-threshold`, `--sheddable-prefill-threshold` |

Touch points: 5 files (8 changes). Medium-High complexity -- the callback injection and batch formation changes are the main risks.

### Self-Critique

1. **Queue depth as load proxy is noisy for individual instances**: With 8 instances, queue depths vary significantly. A burst on one instance triggers high-load mode for that instance while others remain at sub-saturation. The load-adaptive behavior is per-instance, not cluster-wide, which creates inconsistent priority gaps across instances.
2. **The opportunistic batching benefit may be marginal**: With our workload (input Gaussian(256,100) + prefix 512), total input is ~768 tokens. At threshold=256, that is 3 chunks. At threshold=512, that is 2 chunks. Saving one beta0 (6.9ms) per sheddable request is ~0.1% of E2E (5.4s). The throughput improvement may be immeasurable within noise.
3. **Load factor computation adds per-step overhead**: Computing `max_instance_load` requires iterating the queue. This is O(1) per instance (just `len(WaitQ.Items())`), but adds to the priority computation hot path.
4. **SLO-aware chunking threshold violates the current BatchFormation contract**: `PrefillTokenThreshold` is a property of `BatchContext`, not per-request. Making it per-request requires modifying the Phase 2 loop to check each request's priority and select a threshold. This is a meaningful change to the batch formation logic.
5. **Two load thresholds double the calibration requirement**: `low_load_threshold` and `high_load_threshold` must be calibrated against the specific workload's queue depth distribution. The wrong values cause either never-engaging (too high) or always-engaged (too low) priority adaptation.

---

## Executive Summary

### Rankings

| Rank | Idea | Expected Impact | Complexity | Key Risk |
|------|------|----------------|------------|----------|
| **1** | **C: Load-Regime Adaptive** | High | Medium-High | Queue depth proxy may be noisy |
| 2 | B: Preemption-Aware Batch | Medium-High | High | Zero preemptions at baseline = no effect from preemption ordering |
| 3 | A: Bounded Priority + Shedding | Medium | Medium | Admission rejection trades throughput for latency |

### Recommendation: Idea C (Load-Regime Adaptive Priority)

**Why C wins**: It's the only strategy that addresses the fundamental Iteration 1 flaw — constant priority differentiation is wasteful at sub-saturation. By adapting the priority gap to current load, it reduces unnecessary sheddable degradation between bursts while maintaining full SLO differentiation during bursts. It directly implements two proven findings (H23 low-load equivalence + H7 super-linear scaling).

**Why not B**: At our baseline workload (132K blocks, 0 preemptions), the preemption ordering change has zero effect. The strategy's primary value requires KV pressure that doesn't exist at this load level. The dequeue-gating path alone is worth exploring but is a smaller lever.

**Why not A**: Admission shedding reduces throughput by design — the 5% throughput improvement target becomes harder. The bounded priority dampening is interesting but adds stateful complexity (moving-average tracker) that may not be worth it.

### Implementation Plan for Idea C

Phase 1: `LoadAdaptivePriority` policy (adapts gap to load). Phase 2: Wire queue-depth callback. Phase 3: Bayesian optimize. Defer the opportunistic sheddable batching (high friction, marginal gain per self-critique #2) to a later iteration if needed.
