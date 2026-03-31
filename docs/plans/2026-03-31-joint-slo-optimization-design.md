# Design: Joint SLO-Aware Optimization — Admission, Routing, and Engine Mechanisms

**Status:** Approved
**Date:** 2026-03-31
**Type:** Strategy Evolution — Problem Framing + Phase 2 Hypothesis Bundle Design
**Methodology:** [`docs/methodology/strategy-evolution.md`](../methodology/strategy-evolution.md)

---

## Context

The two prior Strategy Evolution tracks — routing (19 iterations, PR #447) and scheduling (11 iterations,
PR #452) — were run in isolation. Both converged independently on the same insight: admission control is
the non-zero-sum lever at high load (RP-9, S6, S8). However, the winning strategies from each track have
never been tested together, and neither track explored engine-level mechanisms: preemption victim ordering,
KV prefix cache eviction priority, or step-level token budget allocation.

This design governs a new investigation that pursues two goals simultaneously:

1. **Composition validation** — verify that the best-known routing, scheduling, prefill, and admission
   policies compose correctly under a new mixed-regime workload they were never jointly tested against.
2. **Engine mechanism discovery** — layer three novel engine mechanisms on top of the confirmed compound
   and measure their incremental contribution.

The workload regime — sustained load near saturation with superimposed gamma burst spikes — is harder than
either prior track tested. It combines the continuous pressure that makes scheduling matter (S6) with the
transient spikes that amplify admission benefit (RP-13).

---

## Phase 1: Problem Framing

### Baseline Configuration

The baseline is the best-known compound strategy from prior isolated tracks, measured across seeds 42, 123,
and 456 before any iteration begins. This baseline is *not* BLIS defaults — it is the ceiling from prior
work, applied jointly for the first time.

| Layer | Policy | Parameters |
|---|---|---|
| Routing | `pa:4,qd:3` (no kv-util) | Prefix-affinity weight=4, queue-depth weight=3 |
| Scheduling | `PriorityFCFSScheduler` | Base scores: critical=10, standard=5, sheddable=1 |
| Prefill | No-chunk for critical | Per-SLO threshold: critical=∞, standard/sheddable=chunked |
| Admission | `TierShedAdmission` | Best-known `overloadThreshold` and `minAdmitPriority` from prior Iter 8/10 |
| Batch formation | `VLLMBatchFormation` | LIFO preemption, default chunk size |
| KV eviction | LRU | GPU-only, no tiered cache |

### Target Workload

**Mixed sustained + burst regime:** Two arrival processes superimposed.

- **Sustained base:** Poisson arrival process at 85% of measured saturation throughput. Leaves 15%
  headroom that burst spikes will overwhelm.
- **Burst overlay:** Gamma arrival process, CV=2.0 (consistent with RP-13 prior experiments),
  superimposed on the Poisson base. Burst intensity set to push total instantaneous arrival rate to
  2× saturation throughput during peak. Simulation horizon spans at least 3 full burst cycles.
- **SLO tier mix:** 20% critical, 60% standard, 20% sheddable. **Orthogonal:** all tiers share
  identical request shapes (prompt length, output length). SLO class is the only differentiator,
  preventing strategies from proxying tier via token length.

This workload was deliberately designed to be harder than either prior track: the sustained base ensures
scheduling trade-offs are always active, while the burst overlay ensures admission and routing signals are
stressed simultaneously.

### Success Criteria

**Primary metric:** Critical TTFT P99.

**Hard constraints (anti-gaming guards):**

- Standard goodput ≥ 85% of the Iteration 0 baseline measurement
- Sheddable goodput ≥ 60% of the Iteration 0 baseline measurement
- No strategy may improve critical P99 primarily by rejecting standard traffic; the goodput floors
  enforce this structurally.

**Success thresholds by phase:**

| Phase | Threshold | Rationale |
|---|---|---|
| Iteration 1 (composition) | >15% critical P99 improvement | Conservative: mixed regime may be harder than pure burst |
| Iterations 2–4 (engine) | >5% incremental improvement per mechanism | Incremental bar; mechanisms build on a strong compound |
| Bayesian optimization | Confirms optimum within ±3% across 3 seeds | Stability check, not a new improvement bar |

**Phase-separated reporting (RP-7):** All metrics must be reported separately for sustained-base intervals
and burst intervals. A strategy that helps only during bursts is not a general solution.

### Prior Knowledge Inventory

These principles from prior tracks narrow the design space and constrain mechanism design.

| Principle | Implication |
|---|---|
| RP-9 | Admission is the non-zero-sum lever at high load — engine mechanisms must not break admission's effect |
| RP-10 | PA:QD routing ratio ≤ 1.5 is the empirical safety bound for Bayesian search |
| RP-13 | Bursty arrivals amplify admission benefit — expect larger effects in burst phase than sustained phase |
| S6 | Scheduling is zero-sum at saturation — joint strategy must not sacrifice standard/sheddable to gain critical |
| S8 | Admission gating breaks the compute floor — engine mechanisms layered on top get a head start |
| S15 | SLO-aware *proactive* preemption has no moderate regime under BLIS recomputation model — reactive SLO-priority victim ordering is untested and in scope |
| RP-7 | Regime-dependent optimum — measure sustained and burst phases separately |

---

## Phase 2: Hypothesis Bundle Design

### Iteration Structure

This investigation uses **Approach B: two-phase with a hard gate.**

- **Iteration 1:** Validate the joint composition as a single hypothesis bundle. Apply a fast-fail rule:
  any component with ablation contribution <5% is dropped before engine mechanism work begins.
- **Iterations 2–4:** One engine mechanism per iteration, in bottom-up dependency order through the
  execution stack. Each iteration's bundle is designed after seeing the prior finding.
- **Bayesian optimization:** Runs after Iteration 4 over the confirmed compound's continuous parameter
  space.

---

### Iteration 1: Joint Composition Validation

The four components (routing, scheduling, no-chunk prefill, admission) have been independently validated
but never tested together under the mixed sustained+burst workload. Two specific interaction effects are
uncharted: (a) does routing to cache-warm instances reduce the benefit of no-chunk prefill, and (b) does
admission shedding at the cluster level make instance-level scheduling less important?

#### H-main

> "The compound strategy (`pa:4,qd:3` + `PriorityFCFSScheduler` + no-chunk-critical +
> `TierShedAdmission`) will improve critical TTFT P99 by **>40%** over the baseline compound under the
> mixed sustained+burst workload, because each layer addresses a distinct bottleneck: routing reduces KV
> misses, scheduling prioritizes critical requests at the instance, no-chunk eliminates prefill overhead
> for critical, and admission ensures the instance is never overwhelmed.
>
> *If <15%: the components are redundant at this workload regime — the burst overlay neutralizes the
> benefit of any single layer.*"

Seeds: 42, 123, 456. Rate: 85% Poisson + 2× burst gamma CV=2.0.

#### H-ablation — four arms

| Arm | What is removed | Prediction | Diagnostic clause |
|---|---|---|---|
| abl-routing | Revert to `pa:3,qd:2,kv:2` | >15% P99 degradation | If <5%: RP-6 dominates (kv-util harm cancels pa benefit in this regime) |
| abl-scheduling | Revert to `FCFSScheduler` | >20% P99 degradation during burst phase | If <5%: admission is doing all the work (S6 generalizes to mixed regime) |
| abl-nochunk | Remove no-chunk-critical threshold | >10% P99 degradation | If <5%: admission already controls queue depth, prefill overhead irrelevant |
| abl-admission | Disable `TierShedAdmission` | >30% P99 degradation | If <30%: burst intensity is insufficient to trigger admission benefit; recalibrate workload |

**Fast-fail rule:** Any component with ablation degradation <5% is dropped from all subsequent iterations.
Its experimental slot is reallocated to additional engine mechanism ablation arms.

#### H-super-additivity — routing × admission interaction

> "The combination of `pa:4,qd:3` routing and `TierShedAdmission` will produce **>10% additional**
> improvement beyond the sum of their individual contributions, because routing directs critical requests
> to cache-warm instances while admission ensures those instances have headroom — a capacity-amplification
> effect neither achieves alone.
>
> *If the interaction term is negative: routing and admission are substitutes at this burst intensity.
> Admission alone controls queue depth; routing has nowhere to improve.*"

#### H-control-negative — mechanism specificity

> "On a uniform-SLO workload (all requests labeled `standard`, identical request shapes), the compound
> strategy will produce **<5% improvement** over BLIS defaults, confirming the mechanism requires SLO
> differentiation to function.
>
> *If >10% on uniform workload: the routing/admission policies have an SLO-independent benefit that has
> not been identified.*"

#### H-robustness — burst CV generalization

> "Critical TTFT P99 improvement remains **>20% at CV=3.0** and the degradation curve is monotone as CV
> increases from 2.0 to 4.0. Below CV=1.0 (near-Poisson), improvement drops toward zero because burst
> dynamics are absent.
>
> *If improvement collapses between CV=2.0 and CV=3.0: the strategy is tuned too tightly to the specific
> burst shape used in H-main.*"

#### Termination criteria

| Outcome | Decision |
|---|---|
| H-main confirmed (>15%), ≥3 ablations show >5% contribution | PROCEED to engine mechanisms |
| H-main confirmed but abl-admission shows <30% degradation | REVISE: recalibrate burst intensity upward, re-run before engine work |
| H-main refuted (<15%) | RESTART: joint composition has an unexpected interaction; diagnose before engine work |

---

### Iteration 2: SLO-Priority Preemption Ordering

**What changes:** `VLLMBatchFormation` currently selects the eviction victim using LIFO ordering — the
most-recently-scheduled running request, regardless of SLO tier. The new policy selects the lowest-SLO
running request as the victim. This is a change to victim *selection*, not preemption *frequency*.

**Critical distinction from S15:** S15 tested proactive preemption — interrupting running low-SLO requests
to make room for incoming high-SLO requests. This iteration tests reactive victim ordering only: when KV
blocks run out during batch formation, the lowest-SLO running request is evicted first. This mechanism
does not trigger additional preemption events; it changes which request bears the cost of unavoidable ones.

**Interface:** New `BatchFormation` implementation behind the existing interface. No changes to `KVStore`
or `InstanceScheduler` interfaces.

#### H-main

> "SLO-priority victim selection will reduce critical TTFT P99 by **>15%** vs the Iteration 1 compound,
> because under KV pressure LIFO may evict a recently-scheduled critical request; SLO-priority ordering
> ensures sheddable requests absorb the preemption cost instead.
>
> *If <5%: preemption is rare at the workload's operating point — LIFO and SLO-priority are equivalent in
> practice. Iteration 3 (KV eviction priority) may produce higher impact.*"

#### H-zero-sum — side-effect detection

> "Standard and sheddable TTFT P99 will not degrade by more than **20%** vs Iteration 1, because
> preempted sheddable requests re-queue and complete later — delayed but not abandoned.
>
> *If >30% standard degradation: SLO-priority preemption is directly transferring latency from critical
> to standard. A non-zero-sum complement (e.g., increased KV capacity) is needed.*"

#### H-control-negative

> "Under abundant KV blocks (132K, well above the working set), the mechanism produces **<3%
> improvement**, confirming it is only active under KV pressure.
>
> *If >5%: the mechanism has an SLO-independent side effect in the batch formation path.*"

---

### Iteration 3: SLO-Aware KV Prefix Cache Eviction

**What changes:** The GPU prefix cache evicts blocks using pure LRU with no awareness of which request
tier owns them. The new policy implements tiered LRU: sheddable-tier blocks are evicted before standard,
standard before critical, with LRU ordering within each tier. Under memory pressure, critical-tier prefix
blocks are the last to be evicted.

**Architectural note:** This is the most interface-invasive change in the investigation. The `KVStore`
eviction path must receive tier information at allocation time. The design must thread tier context through
`AllocateKVBlocks` or attach it as block metadata — this choice belongs in the micro-plan, not here.

#### H-main

> "Tiered LRU eviction will improve critical prefix cache hit rate by **>10%** and critical TTFT P99 by
> **>15%** vs the Iteration 2 compound, because LRU indiscriminately evicts high-value critical-request
> prefix blocks to serve sheddable requests under memory pressure.
>
> *If prefix hit rate improves but TTFT P99 does not: cache hit rate and TTFT are decoupled — the
> bottleneck has shifted elsewhere (likely the scheduler or the token budget).*"

#### H-super-additivity — Iteration 2 × Iteration 3 interaction

> "Tiered LRU combined with SLO-priority preemption will produce **>5% additional** improvement beyond
> each mechanism alone, because they protect critical requests at different pipeline stages: preemption
> ordering protects in-flight token budget; KV eviction protects prefix cache residency. The two
> mechanisms do not share a bottleneck.
>
> *If the interaction term is negative: the mechanisms are substitutes. Protecting critical requests at
> the preemption stage leaves few eviction-eligible blocks for the KV eviction policy to act on.*"

#### H-robustness — prefix-sharing rate

> "The mechanism maintains **>10% improvement** across prefix-sharing rates from 30% to 90%. At 0%
> prefix sharing, improvement drops to <5% — expected, since there is no prefix cache to protect.
>
> *If improvement collapses at 50% sharing: tiered LRU has a threshold effect tied to prefix density
> that constrains its applicability.*"

---

### Iteration 4: Admission-Feedback Batch Formation

**What changes:** A new `BatchFormation` variant partitions the per-step token budget by SLO tier.
Critical requests get first claim up to fraction `f_c` of the token budget; standard requests take `f_s`
of the remainder; sheddable takes what is left. This operates at step granularity — finer than cluster-
level admission shedding, coarser than individual request scheduling.

**Why this is distinct from scheduling:** `PriorityFCFSScheduler` controls the order in which requests
enter the running batch. Admission-feedback batch formation controls how much of each step's compute each
tier receives once already in the batch. A critical request that wins scheduling priority can still be
starved of tokens if standard requests consume the budget in that step. This mechanism closes that gap.

**Interface:** New `BatchFormation` implementation with per-tier budget fraction parameters. The
`BatchContext` struct requires one new field for tier fractions. No changes to `KVStore` or scheduler
interfaces.

#### H-main

> "Tier-partitioned token budget (`f_c=0.50`, `f_s=0.35`, `f_sh=0.15`) will improve critical TTFT P99
> by **>10%** vs the Iteration 3 compound, because even with SLO-priority scheduling and KV protection,
> critical requests share step-level token budget with lower-priority requests — budget partitioning
> ensures proportional throughput at step granularity.
>
> *If <5%: scheduling and KV protection already ensure critical requests dominate the batch; step-level
> budget partitioning is redundant at this load level.*"

#### H-ablation — fraction sensitivity

> "Reducing `f_c` from 0.50 to 0.33 (equal-share) will degrade critical TTFT P99 by **>8%**, confirming
> the token budget fraction is the active mechanism rather than the partitioning structure itself."

#### H-super-additivity — full compound interaction

> "The 7-component compound (Iteration 1 + preemption ordering + tiered LRU + admission-feedback) will
> produce **>5% additional** improvement vs the Iteration 3 compound, and the additional improvement will
> be larger during burst phases than sustained phases, because admission-feedback batch formation is most
> useful when the batch is contended — a condition burst intervals create more than sustained load.
>
> *If burst phase shows <5% additional improvement: the preceding layers have already eliminated batch
> contention during bursts.*"

#### H-robustness — budget fraction sweep

> "Critical TTFT P99 is monotone-decreasing as `f_c` increases from 0.20 to 0.70. The knee occurs near
> 0.50; marginal improvement above `f_c=0.60` is <3%. Standard goodput floor holds at ≥85% for all
> `f_c ≤ 0.60`.
>
> *If the knee is below 0.40: the mechanism is more aggressive than needed — recalibrate for safety
> margin before production use.*"

---

## Phase 4: Bayesian Optimization

After Iteration 4 confirms the 7-component compound, parameter optimization searches the continuous space
jointly. The prior hand-tuned values become warm-start priors, not fixed points.

### Search Space — 6 Parameters

| Parameter | Range | Constraint | Prior |
|---|---|---|---|
| `w_pa` (prefix-affinity weight) | [2, 6] | `w_pa / w_qd ≤ 1.5` | 4 |
| `w_qd` (queue-depth weight) | [1, 4] | paired with `w_pa` | 3 |
| `overloadThreshold` (admission trigger, in-flight requests) | [50, 600] | integer | best from Iter 8/10 |
| `minAdmitPriority` (minimum admitted tier under overload) | {1, 2, 3} | discrete | 2 |
| `f_c` (critical token budget fraction per step) | [0.30, 0.70] | `f_s ≥ 0.20`, `f_sh ≥ 0.10` | 0.50 |
| critical prefill chunk threshold | [128, ∞] | continuous with ∞ sentinel | ∞ |

`f_s = (1 - f_c) × 0.70` and `f_sh = (1 - f_c) × 0.30` are derived; they are not free parameters.

### Optimizer

Gaussian Process with Expected Improvement acquisition function. 8 Latin-hypercube warm-start evaluations,
then 22 BO iterations = 30 total evaluations. Each evaluation runs seeds 42, 123, 456; objective = mean
critical TTFT P99 across seeds subject to the goodput floor hard constraints.

**Convergence criterion:** Stop when Expected Improvement falls below 1% of the current best for 5
consecutive iterations, or after 30 evaluations — whichever comes first. Report the Pareto frontier
(critical P99 vs standard goodput) for the top 5 feasible points.

**Shallow plateau rule:** If the top 5 feasible points lie within 3% of each other in critical P99, the
compound has reached a performance ceiling for this workload. Report the safest operating point (highest
standard goodput within the cluster) rather than the strict minimum.

**RP-7 requirement:** The optimizer uses the full-run metric as its objective. The final report must
additionally break the winning configuration's performance into sustained-base and burst-phase windows.

---

## Phase 5: Measurement Harness and Stopping Criteria

### Per-Evaluation Protocol

1. **Warmup discard:** First 10% of simulation horizon excluded from all metrics. Allows the KV prefix
   cache to reach steady-state occupancy and prevents cold-start artifacts from distorting routing signals.
2. **Phase-separated windows:** Metrics recorded separately for sustained-base intervals (Poisson-only)
   and burst intervals (Poisson + gamma overlay). A burst interval is defined as any 60-second window
   where arrival rate exceeds 120% of saturation throughput.
3. **Seed aggregation:** A result is *confirmed* only when the lower bound of mean ± 1σ across 3 seeds
   exceeds the stated threshold. If seeds diverge by >20% of the mean, a fourth seed (789) is added and
   the result re-evaluated before a decision is issued.

### Per-Iteration Artifacts

| Artifact | Content |
|---|---|
| `findings/iter-N-bundle.md` | Hypothesis bundle with quantitative predictions, written before running |
| `findings/iter-N-results.md` | Raw metric tables, seed breakdown, phase-separated P99 and goodput |
| `findings/iter-N-principles.md` | New principles (RP-N or S-N format) or existing principles confirmed/updated |
| `findings/iter-N-decision.md` | PROCEED / REVISE / RESTART verdict with rationale and fast-fail outcomes |

### Stopping Criteria

| Condition | Action |
|---|---|
| BO converges + all 4 H-main arms confirmed | **Natural stop** — report winning compound + principles |
| Iterations 2, 3, 4 each produce <5% incremental improvement | **Plateau stop** — report Iteration 1 compound as practical recommendation; engine mechanisms are collectively diminishing |
| Any engine mechanism triggers H-zero-sum failure (standard degradation >30%) | **Halt that iteration** — redesign mechanism with explicit goodput floor enforcement before resuming |
| 6 total iteration cycles reached | **Hard budget stop** — report best confirmed compound regardless of convergence |

### INV Compliance Checkpoints

| Invariant | Required check |
|---|---|
| INV-1 (request conservation) | `injected == completed + queued + running + dropped + timedout` across all seeds |
| INV-4 (KV conservation) | `allocated + free == total` after every step in Iteration 3 (tiered LRU changes the eviction path) |
| INV-9 (oracle boundary) | Engine mechanism implementations must not read `Request.OutputTokens` for scheduling, eviction, or budget decisions |
| INV-10/11 (session causality and completeness) | No sessions silently abandoned when admission-feedback batch formation starves the sheddable tier |

---

## Open Questions for Micro-Planning

These questions are deliberately left for the micro-plan phase. They require implementation-level
decisions that should not be pre-empted by this design document.

1. **KV tier threading (Iteration 3):** Should tier context be threaded through `AllocateKVBlocks` as a
   new parameter, or should it be attached as metadata to `KVBlock` at request scheduling time? The
   choice affects whether the interface extension is additive or breaking.
2. **Preemption ordering integration (Iteration 2):** Should SLO-priority victim selection be a new
   `BatchFormation` implementation or a configurable strategy injection into `VLLMBatchFormation`? The
   former is cleaner; the latter avoids code duplication of the chunked-prefill logic.
3. **Token budget enforcement (Iteration 4):** Should the tier budget fractions be enforced as hard caps
   per step or as soft weights that allow critical to exceed `f_c` when standard/sheddable slots are
   empty? Hard caps are simpler; soft weights are more efficient under light critical load.
