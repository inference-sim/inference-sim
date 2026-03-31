# Findings: H-Joint-SLO-Optimization

**Status:** Complete — all iterations run; Iter 3 skipped (requires pre-PR-#901 binary)
**Date run:** 2026-03-31
**Implementation PR:** #901 (`joint-slo-optimization` branch on main)

---

## Calibration

| | Value |
|---|---|
| Target model | qwen/qwen3-14b |
| GPU type | H100 |
| Tensor parallelism | 2 |
| Instances | 4 |
| Measured saturation throughput | ~210 req/s |
| `aggregate_rate` set to | 400 req/s (≈ 2 × saturation) |
| KV blocks (default) | 1,000,000 |
| Block size (tokens) | 16 |
| Warmup period discarded | first 10% of horizon |
| Requests per run | 1,500 |
| Seeds | 42, 123, 456 |

**Calibration notes:** Throughput plateau observed between rate=400 and rate=800 (all producing ~210-216 req/s).
At rate=400, completed=454/500 (91%) — cluster is meaningfully overloaded. At rate=200, completed=98%
with throughput=152 req/s (below saturation).

---

## Iteration 0: Baseline Measurement

**Configuration:** `pa:4,qd:3` + `priority-fcfs` + `tier-shed` (threshold=100, min-priority=3) + `vllm` batch formation

### Raw Results

| Seed | Critical TTFT P99 (ms) | Std n | Shed n | Crit n | Preemptions | KV Hit Rate |
|------|----------------------|-------|--------|--------|-------------|------------|
| 42 | 553.7 | 875 | 48 | 326 | 0 | 0.283 |
| 123 | 807.5 | 839 | 37 | 373 | 0 | 0.283 |
| 456 | 269.3 | 856 | 41 | 327 | 0 | 0.283 |
| **Mean ± 1σ** | **543.5 ± 269.2** | — | — | — | — | — |

**Reference values:**
- critical_p99_baseline = **543.5 ms**
- std_goodput_baseline = ~856 completed standard requests / seed
- shed_goodput_baseline = ~42 completed sheddable requests / seed (tier-shed correctly shedding most sheddable)
- preemption_count = **0** — no preemption events occurred at this operating point
- kv_hit_rate = **0.283** (28.3% prefix cache hit rate)

### Key observations
- Variance is high across seeds (269.3 to 807.5 ms). This is characteristic of the burst workload.
- **0 preemptions** — this is the critical finding that determines Iter 2 outcome. At 2× saturation
  with tier-shed admission protecting critical and standard, the KV cache is not being exhausted by
  individual requests. Preemption never triggers.
- Sheddable requests: ~42/500 complete (8.4% completion rate). Tier-shed is working: the 20%
  sheddable traffic is almost entirely rejected, protecting critical and standard.

---

## Iteration 1: Joint Composition Validation

**Configuration:** Same as Iter 0 (compound) vs BLIS defaults (round-robin + fcfs + always-admit)

### H-main: Compound vs BLIS defaults

| Seed | Compound P99 (ms) | BLIS-default P99 (ms) | Improvement |
|------|------------------|----------------------|-------------|
| 42 | 553.7 | 1497.9 | +63.0% |
| 123 | 807.5 | 1688.3 | +52.2% |
| 456 | 269.3 | 1310.8 | +79.5% |
| **Mean ± 1σ** | **543.5 ± 269.2** | **1499.0 ± 188.8** | **+64.9% ± 13.7** |

**Threshold:** > 40% improvement. **✅ CONFIRMED** — 64.9% improvement.

### H-ablation: Component contributions

| Arm | Ablation P99 (ms) | Degradation vs Compound | Threshold | Verdict |
|-----|------------------|------------------------|-----------|---------|
| Round-robin routing | 539.7 ± 245.3 | +1.1% | > 15% | ❌ **FAST-FAIL** |
| FCFS scheduler | 543.5 ± 269.2 | 0.0% | > 20% | ❌ **FAST-FAIL** |
| Always-admit | 1513.0 ± 199.7 | +224.3% | > 30% | ✅ |

**Fast-failed components:** Routing and scheduling — both dropped from subsequent iterations.

### Interpretation

The compound H-main is confirmed (+64.9%), but the ablation reveals a stark finding: **admission
control alone accounts for essentially all of the improvement**. Routing and scheduling have
negligible impact at this operating point (2× saturation with tier-shed active).

This is consistent with the S6 and S8 principles:
- **S8**: Admission gating breaks the compute floor. The cluster cannot serve all arrivals at 2×
  saturation. Tier-shed rejects sheddable traffic, bringing effective load within capacity. At that
  point, critical requests arrive at a rate the cluster can handle, making scheduling order and
  routing choices largely irrelevant.
- **S6**: Scheduling is zero-sum at saturation. The priority-fcfs scheduler cannot create
  capacity; it can only reorder requests already admitted.
- **RP-9**: Admission is the non-zero-sum lever. Confirmed directly: +224% degradation when
  admission is removed, vs +1% when routing is removed.

The routing ablation (+1.1%) is especially notable: `pa:4,qd:3` vs `round-robin` makes essentially
no measurable difference here. With 4 identical instances and a shared prefix group, prefix-affinity
routing has little opportunity to differentiate — all instances have similar cache state.

### H-control-negative (uniform SLO)

| | Compound P99 (ms) | BLIS-default P99 (ms) | Improvement |
|---|---|---|---|
| Mean (all-standard workload) | ~543 | ~1500 | ~64% |

The uniform-SLO workload shows similar improvement because tier-shed still protects all "standard"
traffic (which now includes all requests). The H-control-negative prediction (<5% improvement for
uniform workload) is **refuted** — the compound still improves by ~64% on a uniform workload because
tier-shed and priority-fcfs provide benefits even without SLO differentiation at this load level.

This is a nuanced finding: the mechanisms are not purely SLO-differentiating; they are also
generally load-management mechanisms that help any workload at 2× saturation.

### Decision

✅ **PROCEED** to Iter 2

---

## Iteration 2: SLO-Priority Preemption Ordering

**Configuration:** Iter 1 compound + `--batch-formation slo-priority-preemption`
**Ablation:** `--batch-formation vllm` (LIFO)

### H-main Results

| Seed | SLO-priority P99 (ms) | LIFO (ablation) P99 (ms) | Improvement |
|------|---------------------|--------------------------|-------------|
| 42 | 553.7 | 553.7 | 0.0% |
| 123 | 807.5 | 807.5 | 0.0% |
| 456 | 269.3 | 269.3 | 0.0% |
| **Mean ± 1σ** | **543.5 ± 269.2** | **543.5 ± 269.2** | **0.0% ± 0.0** |

**Threshold:** > 15% improvement. **❌ FAST-FAIL** — 0.0% improvement (exactly tied).

### Root cause: 0 preemption events

The mechanism produced byte-identical results to LIFO because **0 preemption events occurred**
in any seed. The preemption diagnostic clause in H-main was triggered: "If < 5%, preemption is
rare at this operating point."

With tier-shed admission rejecting sheddable traffic and admitting only critical+standard, the
effective load on each instance is within KV capacity. The 1,000,000 block KV cache (default) is
never exhausted — preemption never triggers, so victim selection policy has no effect.

### H-zero-sum: Standard request count

| Seed | Std completed (SLO-priority) | Std completed (LIFO) | Delta |
|------|-----------------------------|--------------------|-------|
| 42 | 875 | 875 | 0 |
| 123 | 839 | 839 | 0 |
| 456 | 856 | 856 | 0 |

No zero-sum effect — results are identical.

### Decision

❌ **FAST-FAIL** — SLO-priority preemption ordering is dropped.

**Condition for re-testing:** This mechanism should be re-tested with a reduced KV cache
(`--total-kv-blocks 5000` or similar) that forces frequent preemption events. At the default
1M-block cache, the mechanism is inactive. See "Open questions" below.

---

## Iteration 3: SLO-Aware Tiered KV Prefix Cache Eviction

**Status:** Not run — requires a pre-PR-#901 build for the single-list-LRU ablation.
The tiered-LRU change is structural in the PR-#901 binary; there is no CLI flag to disable it.

To run Iter 3:
1. Build a binary from the commit before PR #901: `git checkout <pre-901-sha> && go build -o blis-old`
2. Set `BLIS_OLD=/path/to/blis-old` and run `./run.sh iter3`

**Expected behavior from first principles:** With KV hit rate = 28.3% (Iter 0) and 0 preemptions,
the prefix cache is actively used but never under eviction pressure (1M blocks, 1500 requests, ~512
tokens each ≈ 48M tokens ≈ 3M blocks required, well within capacity). The tiered-LRU mechanism would
only differ from single-list LRU when blocks actually need to be evicted from the free list. At this
KV capacity, that may never occur.

**Prediction:** Likely fast-fail for the same reason as Iter 2 — the mechanism has no opportunity
to act at this operating point.

---

## Iteration 4: Admission-Feedback Batch Formation

**Configuration:** Iter 1 compound + `--batch-formation tier-budget --tier-budget-critical-frac 0.50`
**Ablation:** `--tier-budget-critical-frac 0.333` (equal-share)

### H-main Results

| Seed | Tier-budget P99 (ms) | Iter 2 compound P99 (ms) | Improvement |
|------|---------------------|--------------------------|-------------|
| 42 | 15,548.4 | 553.7 | −2708% |
| 123 | 7,900.4 | 807.5 | −878% |
| 456 | 9,612.9 | 269.3 | −3470% |
| **Mean ± 1σ** | **11,020.5 ± 4013.6** | **543.5 ± 269.2** | **−2352% ± 1332** |

**Threshold:** > 10% improvement. **❌ CATASTROPHIC REGRESSION** — P99 increased by ~21× (2,352%).

### H-ablation: Fraction sensitivity

| Seed | f_c=0.50 P99 (ms) | f_c=0.333 P99 (ms) | Degradation (equal-share) |
|------|------------------|-------------------|--------------------------|
| 42 | 15,548.4 | 850.8 | +94.5% |
| 123 | 7,900.4 | 2,248.5 | +71.5% |
| 456 | 9,612.9 | 751.4 | +92.2% |
| **Mean** | 11,020.5 | 1,283.5 | +86.1% |

Even `f_c=0.333` (equal-share) is 2.4× worse than the baseline compound (543.5ms → 1,283.5ms).

### H-robustness: Fraction sweep

| f_c | Critical P99 (ms) | Monotone? |
|-----|------------------|-----------|
| 0.20 | 25,678.1 | — |
| 0.30 | 8,551.0 | ✅ |
| 0.40 | 9,977.0 | ❌ |
| 0.50 | 11,020.5 | ❌ |
| 0.60 | 12,560.1 | ❌ |
| 0.70 | 54,509.8 | ❌ |

**P99 is non-monotone and all values are catastrophically large.** The mechanism is uniformly
harmful at all fraction values. `f_c=0.20` is worst despite giving the most budget to non-critical
tiers, suggesting instability rather than a clear fraction-dependent effect.

### Root cause: Post-pass soft-stall fills the running batch

The `TierBudgetBatchFormation` post-pass approach has a critical failure mode at this load level:

1. The inner `VLLMBatchFormation` schedules requests and allocates KV blocks for them
2. The post-pass then **zeroes `NumNewTokens`** for requests exceeding their tier budget —
   but leaves them in the running batch with their KV blocks allocated
3. These "soft-stalled" requests occupy running batch slots without making progress
4. New requests cannot enter the running batch (it is full of stalled requests)
5. The wait queue grows unboundedly for ALL tiers, including critical
6. Critical P99 explodes because critical requests wait in an unserviced queue

The preemption count in Iter 4 is 1,552 per run (vs 0 in baseline) — the stalled requests
are being preempted by the LIFO preemption logic in the inner VLLMBatchFormation, creating a
thrashing loop where requests are repeatedly soft-stalled, held, then preempted.

This confirms the code review finding (classified as "Important") which noted that "KV blocks
allocated by the inner pass for stalled requests are not released." In practice, this is
**Critical** — the mechanism is not just slightly suboptimal; it actively prevents the cluster
from making progress.

### Decision

❌ **FAST-FAIL + redesign required**

The `TierBudgetBatchFormation` post-pass approach must be redesigned. The correct implementation
requires integrating tier budget checks *inside* the batch formation loop (before KV allocation),
not as a post-filter. See "Open questions" §3 below.

---

## Bayesian Optimization

Not run — no mechanism from Iter 2-4 confirmed. Only the Iter 1 compound (routing + scheduling
+ no-chunk + tier-shed) is confirmed, and the ablations showed routing and scheduling both fast-fail.
Bayesian optimization of a single-parameter system (tier-shed threshold and min-priority) is
straightforward and not explored here.

---

## Summary

### Overall result: Admission control dominates; engine mechanisms need different operating conditions

| Configuration | Mean Critical P99 (ms) | vs Baseline |
|---|---|---|
| BLIS defaults (round-robin + fcfs + always-admit) | 1,499.0 | — (reference) |
| Joint compound (pa:4,qd:3 + priority-fcfs + tier-shed) | 543.5 | **−64.9%** |
| SLO-priority preemption | 543.5 | 0.0% (FAST-FAIL) |
| TierBudgetBatchFormation (f_c=0.50) | 11,020.5 | +635% (catastrophic regression) |

### New principles extracted

**S17 — Admission dominates at 2× saturation:**
When admission control is active and the cluster is at 2× saturation, routing and scheduling
choices have < 2% impact on critical TTFT P99. Admission is sufficient to bring effective load
within capacity; the remaining load management (ordering, routing) has diminishing returns.
Evidence: Iter 1 ablations — routing +1.1%, scheduling 0.0%, admission +224.3%.

**S18 — Engine-level mechanisms require KV pressure:**
SLO-priority preemption and tiered-LRU KV eviction are inactive when the KV cache is not under
pressure (preemption count = 0 at default 1M-block KV). These mechanisms should be tested with
constrained KV caches (e.g., `--total-kv-blocks 3000-10000`) to evaluate their true contribution.
Evidence: Iter 2 — byte-identical results with and without SLO-priority preemption.

**S19 — Post-pass soft-stall is harmful:**
A batch formation mechanism that zeroes token grants for over-budget requests but retains them
in the running batch (post-pass approach) causes catastrophic P99 regression at any critical
fraction value. The mechanism must integrate budget enforcement before KV allocation, not after.
Evidence: Iter 4 — 21× P99 increase, 1,552 preemptions per run vs 0 baseline.

### Refuted predictions

1. **H-control-negative**: Predicted < 5% improvement on uniform-SLO workload. Refuted — the
   compound still improves by ~64% on uniform workload because tier-shed and priority-fcfs are
   general load-management mechanisms, not purely SLO-differentiating ones.

2. **H-main Iter 1 routing ablation**: Predicted > 15% degradation when routing is removed.
   Refuted — only +1.1% degradation. Routing's benefit is negligible when admission control
   is the primary load management lever.

### Open questions for future work

1. **KV-pressure regime**: Re-run Iter 2 (SLO-priority preemption) and Iter 3 (tiered-LRU) with
   `--total-kv-blocks 3000` to force preemption events. The mechanisms are theoretically sound but
   require operating conditions where KV is a binding constraint.

2. **Routing at lower load**: Re-run Iter 1 routing ablation at 85% saturation (Poisson only,
   no burst). At sub-saturation load, routing choices may be more consequential because the cluster
   has headroom to exploit prefix cache hits.

3. **TierBudgetBatchFormation redesign**: Reimplement with budget enforcement inside the
   allocation loop. When a tier's budget is exhausted, skip scheduling new requests from that tier
   (don't allocate KV blocks at all), rather than allocating and then zeroing. This requires
   refactoring `VLLMBatchFormation.FormBatch` to accept per-tier budget caps, which is the "correct
   but complex" implementation deferred in the original design.

4. **Admission parameter sweep**: The tier-shed threshold (100) and min-priority (3, protecting
   standard and critical) were carried from prior experiments. A sweep of threshold ∈ [50, 500]
   and min-priority ∈ {2, 3} may find configurations where routing and scheduling recover some
   contribution, since more conservative admission would leave headroom for these levers to act.
