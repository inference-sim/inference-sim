# Strategy Evolution: Deadline-Aware SLO Scheduling

**Iteration:** 1 (evolving beyond the winning strategy from prior 11-iteration scheduling track)
**Date:** 2026-03-10
**Worktree:** `.worktrees/hypothesis-playground`
**Branch:** `hypothesis-playground`

---

## Phase 1: Problem Framing

### Goal

Evolve the SLO-aware scheduling mechanism beyond the prior winning strategy (SLO-tiered static priority + SLO-gated admission + per-SLO prefill thresholds). The prior strategy used fixed base scores per SLO class with linear age-weighting. This iteration replaces the priority policy with deadline-aware hyperbolic urgency, deriving scheduling priority from explicit per-class TTFT targets.

**Implementation scope:** This experiment requires implementing `DeadlineAwarePriority` as a new `PriorityPolicy` template (extension type: policy template, per Extension Recipes). The new policy must be registered in `NewPriorityPolicy`, added to `validPriorityPolicies`, and accept parameters via the policy bundle YAML (`--policy-config`). This is a feature implementation + experiment, not a parameter-only experiment.

### Baselines

| ID | Scheduler | Priority Policy | Admission | Purpose |
|----|-----------|----------------|-----------|---------|
| **B0** | FCFSScheduler | Constant(0) | AlwaysAdmit | Zero-effort baseline (no scheduling) |
| **B1** | PriorityFCFSScheduler | SLOBased(base=0, age=1e-6) | AlwaysAdmit | Current BLIS age-only priority (does NOT use SLOClass — `priority.go:26`) |
| **B2** | PriorityFCFSScheduler | StaticClassWeight(critical=10, standard=5, sheddable=1) | AlwaysAdmit | Class-aware but no deadline mechanism (isolates class-awareness from deadline-awareness) |

**Why three baselines:** B1 (`SLOBasedPriority`) uses only age, not `SLOClass`. The treatment varies two dimensions: (1) SLO-class-awareness and (2) deadline-based urgency shape. B2 isolates the deadline mechanism by providing class differentiation without deadline urgency. Comparison matrix: Treatment vs B2 tests deadline mechanism; Treatment vs B1 tests compound effect; B2 vs B1 tests class-awareness alone.

**Note:** B2 (`StaticClassWeight`) also requires implementation as a new PriorityPolicy: `Compute() = classWeight(req.SLOClass)`. This is a minimal extension (constant return per class, no time dependence).

Admission control is AlwaysAdmit for all baselines and treatment to isolate the scheduling mechanism. Admission control can be composed later (per S8).

### Target Workload

**Bursty multi-turn with mixed SLOs:**

- **Arrival process:** Gamma (CV=2.0)
- **Sessions:** 3 rounds, 500ms think time, context accumulation
- **SLO mix:** 20% critical, 40% standard, 40% sheddable
- **Request shapes:** Orthogonal — identical token distributions across all SLO classes
  - Input: Gaussian, mean=256, std=64, min=32, max=1024
  - Output: Gaussian, mean=128, std=32, min=16, max=512
- **Rate:** Sweep at 30%, 80%, and 120% of corrected capacity estimate (see below)
- **Instances:** 4 (cluster mode, routing: `pa:3,qd:2` per RP-7, omitting kv-utilization per RP-6 — KV blocks abundant at 132139)
- **Seeds:** 42, 123, 456
- **Num requests:** 1500 per rate point (yields ~300 critical-class requests per seed, adequate for P99)
- **CLI requirements:** Must explicitly pass `--tp 2 --hardware H100` in all run commands (default is TP=1 per `defaults.yaml` line 9, which maps to different coefficients)

### Capacity Estimate (corrected for context accumulation)

With 3-round multi-turn and `context_growth: accumulate`, effective input tokens grow per round:
- Round 0: ~256 tokens (base input, all cache-miss on first visit)
- Round 1: ~640 tokens (256 original + 128 output + 256 new). With prefix-affinity routing (session stickiness), ~256 tokens cached → ~384 cache-miss
- Round 2: ~1024 tokens (accumulated). With caching, ~640 cached → ~384 cache-miss

Using beta coefficients (llama-3.1-8b, H100, TP=2): `stepTime = 6910 + 17.67 * cacheMissTokens + 2.84 * decodeTokens`

| Scenario | Weighted mean cache-miss tokens | Step time (prefill) | Capacity/instance | 4-instance total |
|----------|--------------------------------|--------------------|--------------------|-----------------|
| No caching | (256+640+1024)/3 = 640 | ~18.6ms | ~53.8 req/s | ~215 req/s |
| Perfect caching | (256+384+384)/3 = 341 | ~13.3ms | ~75.2 req/s | ~301 req/s |

Conservative estimate: **~250 req/s** (moderate caching, accounting for decode batching overhead).

Rate sweep points: 30% (~75 req/s), 80% (~200 req/s), 120% (~300 req/s).

**Note on think time:** The 500ms think time between rounds produces staggered intra-session arrivals on top of the Gamma inter-session process. Burst clusters at time T produce follow-up round-1 arrivals at T+500ms, creating a secondary burst. The capacity estimate above accounts for per-request service time but not this secondary burst effect; the actual saturation point may be lower.

### Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Critical TTFT P99 improvement over B2 | >15% | Deadline mechanism adds value beyond static class weights |
| Critical TTFT P99 improvement over B1 | >20% | Compound effect beats age-only baseline |
| Cluster TTFT P99 degradation vs B0 | <50% | Bounded zero-sum effect (S6) |
| Cluster TTFT P99 degradation: Treatment vs B2 | Measured | Quantifies whether deadline cap reduces zero-sum vs static class weights |
| Throughput change vs B1 | <5% | No throughput sacrifice |

### Hard Constraints (from 30 discovered principles)

- **S3:** ≤7 parameters (treatment has 7: at boundary)
- **S5:** Starvation quantified with concrete crossover times (see analytical derivation below)
- **S6:** Must measure cluster-wide impact; scheduling is zero-sum at saturation. **At 120% capacity (overload), all requests will exceed their deadlines, and urgency saturates at `classWeight / epsilon`. The policy degenerates to class-weight-only ordering: critical=1000 > standard=500 > sheddable=100. This is explicitly expected and represents the design's saturation behavior — the deadline curve adds value during the transient approach to overload, not at steady-state overload.**
- **S7:** Binary load-adaptive mechanisms don't work at sustained near-saturation (continuous urgency growth is a fundamentally different mechanism — gradual, not switched)
- **INV-6:** Float comparison in `PriorityFCFSScheduler` (`scheduler.go:28-29`) warns against division-based priority. Implementation must either: (a) ensure urgency values are sufficiently separated to avoid float comparison hazards, or (b) add epsilon-band comparison. Analysis: minimum urgency difference between any two SLO classes is `classWeight.sheddable / max(eps, ...) - classWeight.sheddable / max(eps, ...)` — since classWeights differ by 5x and 10x, and requests at similar elapsed times produce urgency proportional to classWeight, the separation is inherently 5x+, well above float64 precision. Only same-class requests at near-identical elapsed times could collide — these are broken by ArrivalTime then ID (deterministic).
- **INV-9:** Priority policy must NOT read `Request.OutputTokens`
- **RP-5:** Under low load, scheduling effect should vanish (routing dominates) — tested via 30% capacity arm

### Prior Knowledge Inventory

Key findings from prior 11-iteration scheduling track:

- S1: Priority policy is the primary differentiator (50.8% critical TTFT improvement)
- S6: Scheduling is zero-sum at saturation (+62.4% cluster P99 degradation)
- S8: Admission gating breaks the compute floor (benefits ALL tiers)
- S9/S10: Per-SLO prefill thresholds are a zero-cost lever
- S15: SLO-aware KV preemption has no moderate regime in recomputation mode
- S16: No-chunk benefits all tiers on multi-turn workloads

### Starvation Crossover Analysis (S5)

The DeadlineAwarePriority formula: `urgency = classWeight / max(epsilon, 1.0 - elapsed / deadline)`.

**Crossover time** (when a sheddable request's urgency exceeds a fresh critical request's urgency):

A sheddable request at elapsed time `t_s` has urgency `1.0 / max(0.01, 1.0 - t_s / 2,000,000)`.
A fresh critical request at elapsed time `0` has urgency `10.0 / max(0.01, 1.0 - 0) = 10.0`.

Crossover when: `1.0 / (1.0 - t_s / 2,000,000) > 10.0` → `1.0 - t_s / 2,000,000 < 0.1` → `t_s > 1,800,000 μs` = **1.8 seconds**.

At the urgency cap (past deadline): sheddable urgency = `1.0 / 0.01 = 100`, critical urgency = `10.0 / 0.01 = 1000`. Even at the cap, critical still dominates sheddable 10:1. **Sheddable can only overtake fresh critical requests, not past-deadline critical requests.** This means under sustained overload where all classes exceed their deadlines, the class-weight ordering is preserved and sheddable starvation is bounded by the steady-state queue drain rate.

**Alpha overhead note:** `QueueingTime` delays enqueue by ~2,500 μs (`alpha0 + alpha1 * 256 = 1601 + 899`). The urgency clock starts at `ArrivalTime`, which is ~2.5ms before the request enters `WaitQ`. For a 100ms critical deadline, this consumes 2.5% of the deadline before the request is even eligible for scheduling. This is architecturally realistic (modeling real pre-processing time) and does not materially affect the crossover analysis.

### DES-Specific Notes

- **Priority recomputation scope:** Priority is recomputed only for queued requests (`WaitQ.Items()`) each step, not for running requests already in the batch. The urgency mechanism works through scheduling order (queued-to-scheduled transition), not preemption.
- **Precondition check (ED-3):** `run.sh` must include a diagnostic assertion that average queue depth per instance exceeds 1 at the measurement rate, confirming the priority policy has material to reorder.
- **Multi-turn confound:** Later rounds have larger inputs due to context accumulation (round 0: ~256, round 2: ~1024). This creates a correlation between request age/round-index and request size. A single-turn control at one rate point (no context accumulation) will be included to verify the mechanism works without session dynamics.

---

## Phase 2: Hypothesis Bundle Design

### The Strategy: DeadlineAwarePriority

New `PriorityPolicy` implementation:

```
urgency = classWeight(SLOClass) / max(epsilon, 1.0 - elapsed / deadline(SLOClass))
```

**Classification:** Cross-policy comparative / Validation / Statistical(Dominance).

**Deadline clock semantics:** The formula uses `elapsed = clock - req.ArrivalTime` where `ArrivalTime` is the **per-round** arrival time (each multi-turn round generates a separate `Request` with its own `ArrivalTime` per `reasoning.go:70,91-93`). Each round gets a fresh deadline budget — a round-1 critical request starts with the full 100ms, not the residual from round 0. This per-round reset is the intended behavior; a per-session interpretation would cause all post-R0 requests to immediately saturate at max urgency.

**Parameters (7):**

| Parameter | Value | Analytical derivation |
|-----------|-------|----------------------|
| `classWeight.critical` | 10.0 | Matches prior winning strategy base score; 10:5:1 ratio creates clear 2x and 10x separation |
| `classWeight.standard` | 5.0 | Mid-tier; urgency at t=0 is 5.0, growing to 500 at cap |
| `classWeight.sheddable` | 1.0 | Lowest tier; urgency at cap (100) still below fresh standard (5.0) until sheddable elapsed > 0.96 * deadline |
| `deadline.critical` | 100,000 μs (100ms) | Derived: service time for 256-token prefill ≈ 11.4ms. At 80% utilization, expected queue wait ≈ 4 × service time ≈ 45ms (Erlang-C approximation). 100ms = ~2.2x expected TTFT, providing urgency ramp during the 55ms margin. |
| `deadline.standard` | 500,000 μs (500ms) | 5x critical deadline, matching the classWeight ratio (5:1). Provides ~455ms of urgency ramp. |
| `deadline.sheddable` | 2,000,000 μs (2s) | 20x critical deadline. Generous: sheddable requests only become urgent after ~1.8s, well past the saturation regime where S6 applies. |
| `epsilon` | 0.01 | Caps urgency at 100x classWeight. At cap: critical=1000, standard=500, sheddable=100. The 10:5:1 class ordering is preserved even at saturation, preventing priority inversion. |

**Saturation behavior (S6 acknowledgment):** At 120% capacity (sustained overload), most requests will exceed their deadlines within seconds. Urgency saturates at `classWeight / epsilon` for all classes. The policy degenerates to static class-weight ordering (critical=1000 > standard=500 > sheddable=100). The deadline curve's value is concentrated in the **transient approach to saturation** (during gamma bursts) and at **moderate overload** (where some but not all requests exceed deadlines). At steady-state deep overload, the mechanism is equivalent to B2 (static class weights). This is explicitly predicted and tested via the rate sweep.

### Hypothesis Bundle

#### H-main — Deadline-aware urgency mechanism

**Classification:** Cross-policy comparative / Validation / Statistical(Dominance)

> "DeadlineAwarePriority with deadlines [critical=100ms, standard=500ms, sheddable=2s] will reduce critical TTFT P99 by >15% over B2 (static class weights) at 120% capacity with bursty multi-turn workload, because hyperbolic urgency growth allows critical requests approaching their tight 100ms deadline to rapidly overtake sheddable requests still far from their generous 2s deadline — creating stronger priority separation during transient overload than static class weights alone. **Secondary prediction:** >20% improvement over B1 (compound effect of class-awareness + deadline shape).
>
> *If this fails against B2, the deadline curve does not provide meaningful differentiation beyond static class weights — the transient urgency ramp is too brief or too weak at this operating point. If it fails against B1 but succeeds against B2, class-awareness (not deadline shape) drives the improvement. In either failure case, admission control (S8) is required before deadline-aware scheduling adds value.*"

**Primary metric:** Treatment vs B2 (single-dimension test of deadline mechanism, per ED-1). The B1 comparison is a secondary/composite metric reported alongside but not part of the pass/fail criteria.

**Experiment design:**
- Treatment: `DeadlineAwarePriority` + `PriorityFCFSScheduler` + `AlwaysAdmit`
- Control B0: `ConstantPriority` + `FCFSScheduler` + `AlwaysAdmit`
- Control B1: `SLOBasedPriority` + `PriorityFCFSScheduler` + `AlwaysAdmit`
- Control B2: `StaticClassWeight(10,5,1)` + `PriorityFCFSScheduler` + `AlwaysAdmit`
- Rate: 30%, 80%, 120% of corrected capacity (~75, ~200, ~300 req/s)
- Seeds: 42, 123, 456
- Num requests: 1500 per rate point
- **ED-2 vanishing point:** At 30% capacity, queues are near-empty and all policies should produce <5% difference (RP-5 confirmation)
- **ED-3 preconditions:** (a) Assert average queue depth per instance > 1 at 120% rate point; (b) Assert that ≥20% of steps have requests from 2+ SLO classes in the wait queue simultaneously (confirms the urgency mechanism has multi-class material to differentiate)

#### H-ablation-deadline — Per-class deadline differentiation

**Classification:** Cross-policy comparative / Validation / Statistical(Dominance)

> "Replacing per-class deadlines with a single uniform deadline (500ms for all classes) will degrade critical TTFT P99 by >15% compared to the differentiated treatment, because uniform deadlines eliminate the mechanism's core advantage: tight critical deadlines create urgency faster than generous sheddable deadlines. The degradation has two sources: (1) loosened critical deadline (100ms → 500ms) slows critical urgency growth, and (2) tightened sheddable deadline (2s → 500ms) accelerates sheddable urgency growth — a pincer effect.
>
> *If degradation is <5%, the urgency benefit comes from the hyperbolic shape alone (not deadline differentiation), and per-class deadlines are redundant. If degradation is 5-15%, the pincer effect is weaker than predicted — consider which direction (loosened critical or tightened sheddable) dominates.*"

**Experiment design:**
- Treatment: Per-class deadlines [100ms, 500ms, 2000ms]
- Ablation: Uniform deadline [500ms, 500ms, 500ms]
- Same classWeights for both
- Rate: 80% and 120% capacity (80% is where deadline differentiation has maximum leverage — queues are deep enough for urgency to matter but not so deep that all requests exceed their deadlines. At 120%, the S6 degeneration prediction means differentiated and uniform may converge as all urgencies saturate.)

#### H-zero-sum — Cluster-wide side-effect (application-specific)

**Classification:** Cross-policy comparative / Validation / Statistical(Dominance)

> "The treatment at 120% capacity will NOT degrade cluster-wide TTFT P99 by more than 40% compared to B0 (FCFS), where degradation is defined as `(Treatment_cluster_P99 - B0_cluster_P99) / B0_cluster_P99`.
>
> **Sub-prediction A (comparative):** Treatment's cluster P99 degradation vs B0 will be LESS than B2's cluster P99 degradation vs B0, because deadline-aware urgency's saturation degeneration produces less sustained priority pressure than static class weights.
>
> **Sub-prediction B (zero-sum verification):** The per-class weighted mean TTFT change relative to B0 satisfies `|Σ(class_fraction_i × (mean_TTFT_treatment_i - mean_TTFT_B0_i) / mean_TTFT_B0_i)| < 0.10` (10% tolerance), confirming the scheduling redistribution is approximately zero-sum.
>
> *Conflict resolution: If the absolute bound passes (<40%) but the comparative fails (Treatment ≥ B2), the deadline mechanism creates MORE cluster disruption than static class weights during the transient-to-saturation approach — the urgency ramp itself is destabilizing. If the absolute bound fails (>40%) but the zero-sum check passes (<10%), the mechanism correctly redistributes latency but the total redistribution is larger than acceptable.*"

**Experiment design:**
- Measure cluster-wide TTFT P99 for Treatment, B0, B1, B2 at 120% capacity
- Compute per-class mean TTFT changes relative to B0: `Σ(class_fraction_i × ΔTTFT_mean_i / TTFT_B0_mean_i)`, accept if `|sum| < 0.10`

#### H-control-negative — Mechanism specificity

**Classification:** Cross-policy comparative / Validation / Equivalence

> "With uniform SLO class (all requests labeled 'standard', using deadline=500ms and classWeight=5 for all), the treatment-with-uniform-labels will produce <5% difference from treatment-with-differentiated-labels at 30% capacity, confirming that the mechanism requires both SLO differentiation AND sufficient queueing to produce an effect.
>
> *If difference exceeds 5% even at sub-saturation with uniform labels, the hyperbolic shape provides an SLO-independent benefit — likely from the urgency growth curve itself (even within a single class) producing different scheduling decisions than linear age-weighting.*"

**Experiment design:**
- Treatment-uniform: DeadlineAwarePriority with all requests labeled 'standard'
- Treatment-differentiated: DeadlineAwarePriority with mixed SLO labels
- Rate: 30% capacity (where RP-5 predicts null effect)
- **Note:** This arm compares two variants of the treatment, not treatment vs B1, to avoid conflating formula shape differences with SLO differentiation.

#### H-robustness-burst — Burst intensity scaling

**Classification:** Cross-policy comparative / Validation / Statistical(Dominance)

> "The treatment maintains >15% critical TTFT P99 improvement over B2 across gamma CV values [1.5, 2.0, 3.5], because deadline-aware urgency's advantage grows under bursty arrivals — temporal spikes create deeper queues where deadline-based differentiation matters more.
>
> *If improvement drops below 10% at CV=3.5, the mechanism is burst-insensitive — queue depth during sustained bursts overwhelms deadline-based ordering. If improvement drops at CV=1.5 (less bursty), the mechanism depends on burst-induced transient overload and does not generalize to steady-state.*"

**Experiment design:**
- Treatment vs B2 at 3 gamma CV values: 1.5, 2.0, 3.5
- Rate: 120% capacity
- All three CV values share the same seeds for valid comparison

#### H-single-turn-control — Multi-turn confound isolation

**Classification:** Cross-policy comparative / Validation / Equivalence

> "The treatment achieves >10% critical TTFT P99 improvement over B2 on a single-turn workload (no context accumulation) at 120% capacity, confirming the deadline mechanism works independently of session dynamics.
>
> *If improvement is <5% on single-turn, the deadline mechanism's value depends on multi-turn context growth (larger later-round requests amplify the scheduling effect). The mechanism still works for multi-turn but does not generalize to single-turn workloads.*"

**Experiment design:**
- Single-turn workload: same token distributions (input mean=256, output mean=128), same SLO mix (20/40/40), same Gamma CV=2.0
- **Rate matching:** Use the same aggregate request rate (req/s) as the multi-turn 120% rate point. Since single-turn has no context accumulation, the effective per-request service time is lower (~11.4ms vs ~13.3-18.6ms), so the same aggregate rate represents a lower utilization. This is a known confound: removing multi-turn simultaneously removes context growth AND secondary burst structure AND changes effective load. The arm tests the **combined** effect of these removals.
- Treatment vs B2 at the matched aggregate rate
- 1 rate point, 3 seeds

### Bundle Summary

| Arm | Type | Classification | Prediction | Key metric |
|-----|------|---------------|-----------|------------|
| H-main | Core claim | Dominance | **Primary:** >15% improvement over B2; **Secondary:** >20% over B1 | critical TTFT P99 |
| H-ablation-deadline | Component isolation | Dominance | >15% degradation with uniform deadlines | critical TTFT P99 |
| H-zero-sum | Side-effect detection | Dominance | <40% cluster P99 degradation vs B0; less than B2's degradation | cluster TTFT P99 |
| H-control-negative | Specificity | Equivalence | <5% difference with uniform SLO at sub-saturation | all TTFT metrics |
| H-robustness-burst | Generalization | Dominance | >15% improvement at CV=1.5, 2.0, 3.5 | critical TTFT P99 |
| H-single-turn-control | Confound isolation | Dominance | >10% improvement on single-turn | critical TTFT P99 |

---

## Iteration 1 Results (Ledger)

| Iter | Strategy | Crit TTFT P99 Δ% vs B2 | Crit TTFT P99 Δ% vs B1 | Key Mechanism | Prediction Accuracy | Status |
|------|----------|------------------------|------------------------|---------------|-------------------|--------|
| 0 | B0 (FCFS) | — | — | None | — | Baseline |
| 0 | B1 (age-only) | — | — | Linear age | — | Baseline |
| 0 | B2 (static class weights) | — | — | Class lookup | — | Baseline |
| 1 | DeadlineAwarePriority | +7.7% (worse) | -92.6% | Hyperbolic urgency | Primary REFUTED; Secondary CONFIRMED | Bundle verified |

**Principles extracted:** S17 (static weights sufficient), S18 (time-dependent priority ineffective in DES)

---

### Configurations Required

| Config | CLI flags (beyond common) |
|--------|--------------------------|
| Common | `--tp 2 --hardware H100 --num-instances 4 --routing-scorers prefix-affinity:3,queue-depth:2 --num-requests 1500` |
| B0 | `--scheduler fcfs --priority-policy constant` |
| B1 | `--scheduler priority-fcfs --priority-policy slo-based` |
| B2 | `--scheduler priority-fcfs --priority-policy static-class-weight --policy-config <B2 bundle>` |
| Treatment | `--scheduler priority-fcfs --priority-policy deadline-aware --policy-config <treatment bundle>` |
| Treatment-uniform | Same as Treatment but workload YAML has all SLO classes set to 'standard' |
