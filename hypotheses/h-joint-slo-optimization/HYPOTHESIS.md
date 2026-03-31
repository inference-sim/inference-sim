# H-Joint-SLO-Optimization: Joint Admission, Routing, and Engine Mechanisms

**Status:** Pending — experiments not yet run
**Date:** 2026-03-31
**Seeds:** 42, 123, 456
**Full problem statement:** `problem.md`
**Implementation:** PR #901 (`joint-slo-optimization` branch)

---

## Overview

This investigation tests four stacked hypotheses. Each builds on the confirmed result of
the prior iteration. The dependency chain is:

```
Iter 0 (baseline measurement)
  → Iter 1 (joint composition of known-best strategies)
    → Iter 2 (+ SLO-priority preemption ordering)
      → Iter 3 (+ tiered LRU KV eviction)
        → Iter 4 (+ admission-feedback batch formation)
```

A **fast-fail rule** applies at each step: if the mechanism's ablation contribution is
< 5% of critical TTFT P99, it is dropped and not carried into subsequent iterations.
The Bayesian optimization phase (if all four pass fast-fail) then jointly optimizes the
continuous parameters of the confirmed compound.

---

## Iteration 0: Baseline Measurement

**Purpose:** Establish reference values for the joint compound under the mixed workload.
No new mechanisms — this is pure measurement.

**Configuration:** See `problem.md` §3 for exact parameters.

**What to record:**
- Critical TTFT P99 (sustained phase and burst phase)
- Standard goodput (% of injected standard requests completed)
- Sheddable goodput
- Preemption count (to understand how often preemption occurs — informs Iter 2 impact)
- KV cache hit rate (to understand prefix cache utilization — informs Iter 3 impact)
- Saturation throughput (to calibrate `aggregate_rate` in `workload.yaml`)

---

## Iteration 1: Joint Composition Validation

**Research question:** Does the best-known compound strategy (`pa:4,qd:3` + priority-fcfs +
no-chunk-critical + tier-shed) improve critical TTFT P99 over the BLIS defaults when all
four components are active simultaneously under the mixed sustained+burst workload?

### H-main

> "The compound strategy will improve critical TTFT P99 by **>40%** over BLIS defaults
> (`round-robin` + `fcfs` + `always-admit`) under the mixed sustained+burst workload,
> because each layer addresses a distinct bottleneck: routing reduces KV misses, scheduling
> prioritizes critical at the instance, no-chunk eliminates prefill overhead for critical,
> and admission prevents instance overload.
>
> *Diagnostic if < 15%: the burst overlay neutralizes the benefit of any single layer;
> investigate whether preemption count or KV miss rate is abnormally high.*"

### H-ablation (four arms)

| Arm | What is removed | Prediction | Diagnostic |
|---|---|---|---|
| abl-routing | Revert to `round-robin` | > 15% P99 degradation | If < 5%: routing benefit is negligible under this workload; consider whether KV pressure is too low |
| abl-scheduling | Revert to `fcfs` | > 20% P99 degradation during burst phase | If < 5%: admission is doing all the work (S6 generalization holds here too) |
| abl-nochunk | Revert all tiers to chunked prefill | > 10% P99 degradation | If < 5%: admission already controls queue depth; prefill overhead is irrelevant |
| abl-admission | Revert to `always-admit` | > 30% P99 degradation | If < 30%: burst intensity is insufficient; recalibrate aggregate_rate upward |

### H-super-additivity (routing × admission)

> "Routing + admission together produce **> 10% additional** improvement over the sum of
> their individual contributions, because routing directs critical requests to cache-warm
> instances while admission ensures those instances have headroom — a capacity-amplification
> effect neither achieves alone.
>
> *Diagnostic if interaction term is negative: routing and admission are substitutes at this
> burst intensity; admission shedding eliminates the traffic that makes routing decisions
> consequential.*"

### H-control-negative

> "On a **uniform-SLO workload** (all requests labeled `standard`, identical shapes), the
> compound strategy produces **< 5% improvement** over BLIS defaults, confirming the
> mechanism requires SLO differentiation.
>
> *Diagnostic if > 10%: the routing/admission policies have an SLO-independent benefit
> (e.g., prefix-affinity improving cache hits regardless of SLO class).*"

### H-robustness

> "Critical TTFT P99 improvement remains **> 20% at burst CV=3.0** and the degradation
> curve is monotone as CV increases from 2.0 to 4.0. Below CV=1.0, improvement drops
> toward zero because burst dynamics are absent.
>
> *Diagnostic if collapse between CV=2.0 and CV=3.0: the strategy is tuned too tightly
> to the CV=2.0 burst shape used in H-main.*"

### Termination

| Outcome | Decision |
|---|---|
| H-main > 15%, ≥ 3 ablations > 5% | PROCEED to Iter 2 |
| H-main confirmed, abl-admission < 30% | REVISE: recalibrate aggregate_rate upward |
| H-main < 15% | RESTART: diagnose interaction effects before Iter 2 |

---

## Iteration 2: SLO-Priority Preemption Ordering

**Mechanism:** When KV blocks run out during batch formation and a running request must be
evicted, select the lowest-SLO running request as the victim rather than the most-recently-
scheduled one (LIFO).

**What this is NOT:** This is not proactive preemption (interrupting running low-SLO requests
to make room for *incoming* high-SLO ones). S15 found proactive preemption has no moderate
regime under BLIS's recomputation model. This mechanism only changes *which* request bears
the cost of an eviction that was already going to happen.

**CLI flag:** `--batch-formation slo-priority-preemption`

**Baseline for this iteration:** Iter 1 confirmed compound.

### H-main

> "SLO-priority victim selection reduces critical TTFT P99 by **> 15%** over the Iter 1
> compound, because LIFO may evict a recently-scheduled critical request; SLO-priority
> ordering ensures sheddable requests absorb the preemption cost instead.
>
> *Diagnostic if < 5%: preemption is rare at this operating point — LIFO and SLO-priority
> are equivalent in practice. Check preemption count from Iter 0; if < 1% of steps have
> preemption, the mechanism has no opportunity to act.*"

### H-zero-sum

> "Standard and sheddable TTFT P99 degrade by **≤ 20%** over the Iter 1 compound,
> because preempted sheddable requests re-queue and complete later — delayed but not
> abandoned.
>
> *Diagnostic if standard degrades > 30%: SLO-priority preemption is directly transferring
> latency from critical to standard; consider whether sheddable requests are being
> repeatedly preempted (a starvation indicator).*"

### H-control-negative

> "Under abundant KV blocks (≥ 4× the working set), the mechanism produces **< 3%**
> improvement, confirming it is activated only under KV pressure.
>
> *Diagnostic if > 5%: the mechanism has an SLO-independent side effect in the batch
> formation path.*"

**Fast-fail threshold:** If abl-admission contribution in Iter 1 was < 5% (admission was
dropped), re-examine whether KV pressure is occurring at all before running Iter 2.

---

## Iteration 3: SLO-Aware Tiered KV Prefix Cache Eviction

**Mechanism:** The GPU prefix cache uses per-tier LRU lists (`freeTierHead[5]`). When the
cache needs to evict a block, it drains lower-SLO-tier blocks first (background → batch →
sheddable → standard → critical). High-SLO prefix entries are preserved under memory
pressure. This mechanism is **structural** (always active in the PR #901 build) — there is
no CLI flag to toggle it.

**What this changes:** Previously, all released blocks entered a single LRU list regardless
of SLO tier. Now, a standard request's prefix block in the free list is only evicted after
all sheddable and batch blocks are exhausted.

**Baseline for this iteration:** Iter 2 confirmed compound (or Iter 1 if Iter 2 was
fast-failed).

**Ablation strategy:** Compare Iter 3 results against a build compiled WITHOUT the tiered-LRU
changes (the pre-PR #901 `sim/kv/cache.go`). Since this is structural, the ablation requires
two different binary builds, not a CLI flag.

### H-main

> "Tiered LRU eviction improves critical prefix cache hit rate by **> 10%** and critical
> TTFT P99 by **> 15%** over the Iter 2 compound, because under memory pressure the
> single-list LRU indiscriminately evicts high-value critical prefix blocks.
>
> *Diagnostic if hit rate improves but TTFT P99 does not: the bottleneck has shifted
> elsewhere — likely the token budget or scheduling is now the limiting factor.*"

### H-super-additivity (Iter 2 × Iter 3)

> "Tiered LRU combined with SLO-priority preemption produces **> 5% additional** improvement
> beyond each alone, because they protect critical requests at different pipeline stages:
> preemption ordering protects in-flight token budget; KV eviction protects prefix cache
> residency. These are non-overlapping bottlenecks.
>
> *Diagnostic if interaction term is negative: protecting critical at the preemption stage
> leaves few eviction-eligible blocks for the KV policy to act on — the mechanisms are
> substitutes under this workload's KV pressure level.*"

### H-robustness (prefix sharing rate)

> "The mechanism maintains **> 10% improvement** across prefix-sharing rates from 30% to
> 90%. At 0% prefix sharing (all unique tokens), improvement drops to **< 5%** — expected,
> since there is no prefix cache to protect.
>
> *Diagnostic if improvement collapses at 50% sharing: tiered LRU has a threshold effect
> tied to prefix density. Record the knee point in FINDINGS.md.*"

**Validity note:** The mixed workload in `workload.yaml` uses a single shared prefix group
with `prefix_length=256`. This is a moderate prefix-sharing scenario. Results may differ
significantly in workloads with many prefix groups or no prefix sharing.

---

## Iteration 4: Admission-Feedback Batch Formation

**Mechanism:** A new batch formation policy (`TierBudgetBatchFormation`) partitions the
per-step token budget by SLO tier. Critical requests get first claim up to fraction `f_c`
of `MaxScheduledTokens`; standard gets `f_s × (1 - f_c)`; sheddable takes the remainder.
Requests that exceed their tier's budget for a given step receive 0 new tokens that step
(soft stall) but remain in the running batch and retry next step.

**How this differs from scheduling:** `priority-fcfs` controls the *order* in which requests
enter the running batch (queue ordering). TierBudgetBatchFormation controls *how many
tokens* each tier receives per step once already in the batch. A critical request that wins
scheduling priority can still be under-served if standard requests consume the step budget
first; this mechanism prevents that.

**CLI flags:** `--batch-formation tier-budget`, `--tier-budget-critical-frac 0.50`,
`--tier-budget-standard-frac 0.70`

**Baseline for this iteration:** Iter 3 confirmed compound.

### H-main

> "Tier-partitioned token budget (`f_c=0.50`, `f_s=0.70`) reduces critical TTFT P99 by
> **> 10%** over the Iter 3 compound, because even with SLO-priority scheduling and KV
> protection, critical requests share per-step token budget with lower-priority requests —
> budget partitioning ensures proportional throughput at step granularity.
>
> *Diagnostic if < 5%: scheduling and KV protection already ensure critical requests
> dominate the batch; step-level budget partitioning is redundant at this load level.
> This is the most likely fast-fail candidate.*"

### H-ablation (fraction sensitivity)

> "Reducing `f_c` from 0.50 to 0.333 (equal-share) degrades critical TTFT P99 by **> 8%**,
> confirming the token budget fraction is the active mechanism rather than the partitioning
> structure itself.
>
> *Diagnostic if degradation < 3%: the optimization is insensitive to the exact fraction
> within this range — the mechanism is robust but the specific parameterization does not
> matter much.*"

### H-super-additivity (full 7-component compound)

> "The full compound (Iter 1 base + SLO-priority preemption + tiered LRU + tier-budget)
> produces **> 5% additional** improvement over the Iter 3 compound, with **larger gain
> during burst phases than sustained phases**, because admission-feedback batch formation
> is most useful when the batch is contended — a condition burst intervals create more
> than sustained load.
>
> *Diagnostic if burst phase shows < 5% additional improvement: the preceding layers have
> already eliminated batch contention during bursts.*"

### H-robustness (fraction sweep)

> "Critical TTFT P99 is **monotone-decreasing** as `f_c` increases from 0.20 to 0.70.
> The knee occurs near 0.50; marginal improvement above `f_c=0.60` is < 3%. Standard
> goodput floor holds at ≥ 85% for all `f_c ≤ 0.60`.
>
> *Diagnostic if the knee is below 0.40: the mechanism is more aggressive than necessary —
> recalibrate for safety margin before production deployment.*"

**Known limitation:** The post-pass soft-stall implementation holds KV blocks for stalled
requests during the stall step. This causes temporary over-allocation but does not corrupt
ProgressIndex (fixed in commit `3c87916c`). At `f_c=0.50`, the over-allocation affects
a small fraction of steps. The H-zero-sum arm will catch any systematic goodput harm.

---

## Bayesian Optimization Phase (after all iterations)

If all four iterations confirm their H-main (or if some are fast-failed, the confirmed
subset), a Gaussian Process optimization runs over the confirmed compound's continuous
parameters:

| Parameter | Range | Constraint | Prior |
|---|---|---|---|
| `w_pa` (prefix-affinity weight) | [2, 6] | `w_pa / w_qd ≤ 1.5` (RP-10) | 4 |
| `w_qd` (queue-depth weight) | [1, 4] | — | 3 |
| `overloadThreshold` | [50, 600] | integer | from Iter 10 |
| `minAdmitPriority` | {1, 2, 3} | discrete | 2 |
| `f_c` (critical budget fraction) | [0.30, 0.70] | `f_s ≥ 0.20`, `f_sh ≥ 0.10` | 0.50 |
| critical prefill threshold | [128, ∞] | ∞ = no-chunk | ∞ |

30 total evaluations (8 Latin-hypercube warm-start + 22 BO). Stop when Expected Improvement
falls below 1% of current best for 5 consecutive iterations. Report the Pareto frontier of
critical P99 vs. standard goodput for the top 5 feasible points.

---

## Stopping Criteria

| Condition | Action |
|---|---|
| BO converges, all confirmed H-main arms pass | **Natural stop** — report winning compound + new principles |
| Iters 2, 3, 4 each produce < 5% additional improvement | **Plateau stop** — Iter 1 compound is the practical recommendation |
| Any mechanism triggers H-zero-sum failure (standard > 30% degradation) | **Halt that iteration** — redesign with explicit goodput floor before resuming |
| 6 total iteration cycles reached | **Hard budget stop** — report best confirmed compound |
