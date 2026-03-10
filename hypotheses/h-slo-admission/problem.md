# Strategy Evolution: SLO-Gated Admission Control (Iteration 2)

**Iteration:** 2 (builds on Iteration 1 findings: S17 static weights sufficient, S6 scheduling is zero-sum)
**Date:** 2026-03-10
**Worktree:** `.worktrees/hypothesis-playground`
**Branch:** `hypothesis-playground`

---

## Phase 1: Problem Framing

### Goal

Compose `StaticClassWeight` (proven in Iteration 1: 92.6% critical TTFT improvement) with SLO-gated admission control to break the zero-sum scheduling barrier (S6). Iteration 1 confirmed that scheduling alone is zero-sum at saturation — improving critical by 92% costs +6.5% cluster-wide degradation. Admission control is the non-zero-sum "third lever" (S8/S9 from prior Strategy Evolution).

**Mechanism:** Under overload, reject sheddable requests before they enter the queue, reducing total queue depth for all remaining classes. This is non-zero-sum because rejected sheddable requests never consume scheduling or compute resources.

**Implementation scope:** Requires implementing `SLOGatedAdmission` as a new `AdmissionPolicy` template. This policy accepts all critical/standard requests and applies token-bucket rate limiting only to sheddable requests.

### Baselines

| ID | Scheduler | Priority Policy | Admission | Purpose |
|----|-----------|----------------|-----------|---------|
| **B2** | PriorityFCFSScheduler | StaticClassWeight(10,5,1) | AlwaysAdmit | Iteration 1 proven winner (class-aware, no admission) |
| **T2** | PriorityFCFSScheduler | StaticClassWeight(10,5,1) | SLOGatedAdmission | Treatment: class weights + SLO-gated admission |

B0 and B1 from Iteration 1 are retained for reference but the primary comparison is T2 vs B2 (isolates the admission mechanism, ED-1 compliant — single dimension varied).

### Target Workload

Same as Iteration 1 for comparability:
- **Arrival:** Gamma CV=2.0, 3-round multi-turn, 500ms think time, context accumulation
- **SLO mix:** 20% critical, 40% standard, 40% sheddable
- **Shapes:** Orthogonal (input mean=256, output mean=128)
- **Rate:** 80% (~200 req/s) and 120% (~300 req/s). Drop 30% (Iteration 1 showed scheduling irrelevant at sub-saturation).
- **Instances:** 4, routing `pa:3,qd:2`
- **Seeds:** 42, 123, 456
- **Num requests:** 1500 per rate point
- **CLI:** `--tp 2 --hardware H100`

### Capacity Estimate

Same as Iteration 1: ~250 req/s for 4 instances (corrected for context accumulation).

### Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Critical TTFT P99 improvement: T2 vs B2 | >20% | Admission must add value beyond class weights alone |
| Cluster TTFT P99: T2 vs B2 | Improved or <5% worse | Non-zero-sum: admission should benefit ALL admitted requests |
| Sheddable rejection rate at 120% | 10-40% | Enough to reduce queue depth, not so much that sheddable is fully excluded |
| Throughput of admitted requests | >B2 throughput | Fewer requests competing = higher per-request throughput |

### Hard Constraints

- **S6:** Admission control is the only non-zero-sum lever at saturation
- **S8:** Admission gating breaks the "compute floor"
- **S17 (new):** Static class weights are the minimal sufficient scheduling mechanism
- **S18 (new):** Time-dependent priority is ineffective — don't add it
- **INV-1:** `num_requests == injected + rejected` — must track rejected requests
- **INV-9:** Admission policy must not read `Request.OutputTokens`

---

## Phase 2: Hypothesis Bundle Design

### The Strategy: SLOGatedAdmission

New `AdmissionPolicy` implementation:

```
Admit(req, state):
  if req.SLOClass == "critical" or req.SLOClass == "standard":
    return true  // always admit critical and standard
  // Sheddable: apply load-based gating
  totalQueueDepth = sum(snapshot.QueueDepth for all instances)
  if totalQueueDepth > queueThreshold:
    return false, "slo-gated: sheddable rejected under load"
  return true
```

**Classification:** Cross-policy comparative / Validation / Statistical(Dominance)

**Parameters (2):**

| Parameter | Value | Derivation |
|-----------|-------|-----------|
| `protectedClasses` | ["critical", "standard"] | Always admitted; these are the SLO classes that must meet latency targets |
| `queueThreshold` | 100 | At 4 instances, threshold=100 means reject sheddable when average queue > 25/instance. At 80% load, avg queue ≈ 10-20; at 120%, avg queue ≈ 50-100. Threshold triggers at moderate-to-heavy overload. |

**S3 compliance:** 2 parameters (well under ≤7 limit).

**Key properties:**
- **INV-9 compliant:** Reads `SLOClass` only (class membership), not `OutputTokens`
- **INV-1 compliant:** Rejected requests tracked via `RejectedRequests` counter
- **Non-zero-sum mechanism:** Rejected requests free compute and memory for admitted requests
- **Selective shedding:** Only sheddable requests are rejected; critical/standard always admitted

### Hypothesis Bundle

This is a **single-component mechanism** (SLO-gated admission), so per the bundle size guide: H-main, H-control-negative, 1-2 H-robustness.

#### H-main — SLO-gated admission mechanism

**Classification:** Cross-policy comparative / Validation / Statistical(Dominance)

> "Adding SLO-gated admission (rejecting sheddable under load) to StaticClassWeight will reduce critical TTFT P99 by >20% over B2 (StaticClassWeight alone) at 120% capacity, because shedding sheddable requests reduces total queue depth, allowing critical/standard requests to be scheduled faster. Additionally, cluster-wide TTFT P99 for admitted requests will improve (non-zero-sum per S8).
>
> *If this fails, queue depth at 120% is dominated by critical/standard requests (which are not shed), meaning sheddable shedding has insufficient impact. Diagnostic: check sheddable fraction of queue at 120%. If <30%, shedding is too selective.*"

**Primary metric:** T2 vs B2 critical TTFT P99 at 120%.

**Experiment design:**
- Treatment T2: `StaticClassWeight(10,5,1)` + `SLOGatedAdmission(threshold=100)` + `PriorityFCFS`
- Control B2: `StaticClassWeight(10,5,1)` + `AlwaysAdmit` + `PriorityFCFS`
- Rate: 80%, 120%
- Seeds: 42, 123, 456
- Num requests: 1500
- **ED-3 precondition:** Assert sheddable rejection rate > 5% at 120% (admission is active)

#### H-zero-sum-broken — Non-zero-sum verification

> "T2's cluster-wide TTFT P99 (for admitted requests only) will be BETTER than B2's cluster-wide TTFT P99 by >5%, because admission reduces total queue depth — a non-zero-sum benefit that improves all admitted requests, not just critical.
>
> *If cluster P99 is unchanged or worse, the admission threshold is too high (not enough shedding) or the sheddable requests were not actually the bottleneck.*"

#### H-control-negative — Mechanism specificity

> "With uniform SLO class (all requests 'standard'), SLO-gated admission has no effect (all requests are 'standard' = protected), producing <5% difference from B2 at 120%.
>
> *If >5% difference, the admission policy has an SLO-independent effect we haven't identified.*"

#### H-threshold-sensitivity — Parameter robustness

> "The mechanism maintains >15% critical TTFT P99 improvement over B2 across queue thresholds [50, 100, 200] at 120%, confirming the mechanism is not threshold-fragile.
>
> *If improvement collapses at threshold=200 (permissive), the mechanism depends on aggressive shedding. If improvement collapses at threshold=50 (aggressive), over-shedding causes starvation of sheddable sessions.*"

### Bundle Summary

| Arm | Type | Prediction | Key metric |
|-----|------|-----------|------------|
| H-main | Core claim | >20% crit TTFT P99 improvement over B2 | critical TTFT P99 |
| H-zero-sum-broken | Side-effect | >5% cluster P99 improvement (non-zero-sum) | cluster TTFT P99 |
| H-control-negative | Specificity | <5% with uniform SLO | all metrics |
| H-threshold-sensitivity | Robustness | >15% across thresholds [50, 100, 200] | critical TTFT P99 |
