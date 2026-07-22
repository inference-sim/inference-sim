# LoRA Placement-Policy Seams — Design

**Date:** 2026-07-22
**Status:** APPROVED (human-approved by tantawi, 2026-07-22) — convergence review PASSED (6 rounds, converged 0 CRITICAL / 0 IMPORTANT, 8 perspectives on sonnet)
**Species:** System Overview (multi-PR feature spanning Routing, Eviction, Creation modules)
**Feature branch:** `008-lora-policy-seams`
**Spec:** [`specs/008-lora-policy-seams/spec.md`](../../specs/008-lora-policy-seams/spec.md)

**Builds on:** the merged LoRA control-plane subsystem (adapter identity & registry,
per-instance resident set with LRU + pin/capacity, cold-load creation gate, adapter
cost model, static HBM accounting, `lora-affinity` scorer, per-adapter metrics —
[`docs/plans/2026-07-15-lora-control-plane-design.md`](2026-07-15-lora-control-plane-design.md),
which establishes invariants **INV-L1…INV-L7** that this feature preserves and extends).

**Source input:** Part B of the LoRA policy-seams design authored in the `lora-control`
repo (`docs/superpowers/specs/2026-07-22-lora-policy-seams-design.md`) and lora-control
issue #12. This document is the BLIS-native design that Part B seeds; it supersedes
Part B's illustrative Go shapes with behavioral contracts grounded in the real code.

> **Naming disclaimer.** Trigger names (`Initial`, `OnRoute`, …), policy names
> (`route-to-holder`, `pre-placement`, …), and seam labels in this document are
> **conceptual labels, not frozen Go identifiers**. Exact type/method names are a
> micro-plan decision (design-guidelines §3.4). The **Status** column in §7 tracks
> which decisions are Proposed / Implemented / Superseded so divergence during the
> 7-PR rollout is visible rather than silent (anti-pattern §6.1 "Silent Staleness").

---

## 1. Motivation

BLIS models LoRA adapter placement as purely *emergent*: instances start empty,
adapters become resident on demand through a cold-load gate, eviction is hardcoded
least-recently-used, and the `lora-affinity` router only *biases* routing toward warm
instances — it never forbids. A researcher cannot pre-assign adapters to instances,
cannot route strictly to holders, and cannot choose eviction victims by anything other
than recency. Published static-placement policies therefore cannot be reproduced, and
there is no orthogonal surface on which to run placement-policy experiments or automated
policy search.

This feature makes adapter **placement policy** a first-class, selectable concern along
three independent knobs — **Routing**, **Eviction**, **Creation** — each defaulting to
today's exact behavior. The harness layer (already landed in `lora-control`) can already
*name* these policies; this feature is the BLIS-side mechanism those names resolve to.

**Analysis questions this design answers:**
1. *(AQ1 — Static placement)* How does static pre-placement + strict route-to-holder compare to
   emergent placement on cold-load rate, adapter churn, and TTFT/throughput?
2. *(AQ2 — Eviction ablation)* Under skewed adapter popularity, how does a rank/cost-aware evictor
   change adapter load/eviction counts and TTFT relative to LRU, holding routing and
   creation fixed?

These are treated as two independent experiment classes (the three knobs are orthogonal),
not one bundled comparison.

## 2. Scope

**In scope:**
- Three named policy seams with baseline defaults byte-identical to today: **Routing**
  (which instance serves a request), **Eviction** (victim under capacity pressure),
  **Creation** (start-of-run residency + admit-on-miss).
- Three seed policies: **route-to-holder** (strict routing), **rank/cost-aware eviction**,
  **pre-placement** (declared adapter→instance seeding at t=0).
- A **(trigger → policy)** taxonomy distinguishing reactive triggers (implemented) from a
  periodic trigger (scaffolded, inert — §9, D5).
- **Named strategy bundles**: a config name resolving to a {routing, eviction, creation}
  triple, with per-knob override.
- **Provenance**: the effective (post-expansion) policy triple recorded in run output,
  omitted for all-baseline runs (D8).

**Explicitly out:**
- Migration and scaling *policies* (the periodic trigger is scaffolded; no policy consumes
  it this round).
- **Cross-scale adapter re-placement.** The AutoScaler is **merged and operational** today
  (self-rescheduling `ScalingTickEvent`/`ScaleActuationEvent`; enabled via
  `--model-autoscaler-interval-us`), so once this feature ships a run can combine LoRA pre-placement
  with live scaling using existing flags — no further scaling-integration work. What this feature defers is **cross-scale re-placement**: keeping the
  *declared* adapter→instance assignment synchronized as the topology changes mid-run. This
  is safe to defer because **per-instance composition already works without modification** —
  an instance scaled in mid-run starts with an empty resident set and falls through to
  normal Creation admit-on-miss (Creation's own baseline default, the same liveness-preserving stance
  D1 applies to routing), exactly as the control-plane design
  established (2026-07-15, the "Autoscaler / horizontal-scaling composability" bullet: new
  instance = empty residency, per-instance invariants unmodified). Because the pre-placement +
  autoscaler combination is reachable via existing flags (no future scaling-integration PR needed),
  §11 adds a validation item for it rather than deferring it.
- Any runtime policy language / DSL or out-of-process policy server — policies are named
  entries in a compiled catalog.
- Dynamic (runtime-negotiated) adapter↔KV memory tradeoffs — HBM accounting stays static.
- The harness/experiment side (PolicySpec, bundle *research* tables, provenance manifest) —
  that is Part A, already landed in `lora-control`.

**Deferred:** proactive eviction, prefetch, migration, and scaling policies — all of which
attach to the periodic trigger the taxonomy designs in now.

## 3. Concept Model

Placement policy is decomposed into three orthogonal knobs, each a **named entry in a
registry** with a registered baseline default:

```
                    ┌──────────── request lifecycle ────────────┐
   t=0  ──▶ Creation: seed residency (Initial trigger)
            │
   arrival ─┼──▶ Routing: pick instance (OnRoute) ──▶ instance wait queue
            │         │
            │         └─(request needs an absent adapter)─▶ Creation: admit on miss (OnResidentMiss)
            │                                                    │
            │                                     (slot needed)  ▼
            └───────────────────────────────────▶ Eviction: pick victim (OnCapacityPressure)
```

- **Routing** decides *which instance* serves a request. Baseline = the existing
  scorer-based weighted routing; new = **route-to-holder** (strict).
- **Eviction** decides *which resident adapter is evicted* when a slot is needed. Baseline =
  LRU; new = **rank/cost-aware**.
- **Creation** decides *which adapters are resident at start-of-run* and *whether to admit
  an adapter on a miss*. Baseline = on-demand (empty seed, always admit); new =
  **pre-placement** (declared seeding at t=0).

A **strategy bundle** is a named config that binds a triple of these; it resolves to three
registry lookups and introduces no new mechanism.

A **trigger** is the point at which a policy is invoked. Triggers are orthogonal to
policies: the taxonomy (§9) names reactive triggers (start-of-run, on-route,
on-resident-miss, on-capacity-pressure) and a periodic (interval-driven) trigger, so a
future proactive policy attaches to the periodic trigger without repainting the framework.

**Parallel development.** The three seams share no *runtime* mutable state (per the §6 contracts), so once each seam's
registry PR lands, its seed-policy PR can proceed concurrently with the others, coordinated
only by that seam's behavioral-contract tests (§13 sequences this explicitly). (The eviction and
creation *registry* PRs, B-3/B-5, land in the same instance-construction region of code, so they
carry ordinary file-level merge coordination — not shared runtime state.)

## 4. Modeling Decisions

Each "loses:" clause names the real-system behavior the simplification foregoes
(design-guidelines §2.1 checklist).

| Concern | Modeled | Simplified | Omitted (this round) |
|---|---|---|---|
| Routing strictness | route-to-holder forbids non-holders when a holder exists | best-scored holder reuses existing scorer composition among holders (loses: a source policy's own multi-holder tie-break rule, e.g. least-loaded-holder, if it differs from the configured scorer profile); no-holder → unconstrained fallback (D1) | soft/probabilistic strictness knobs (loses: tunable holder-preference weighting between "always" and "never") |
| Eviction cost | victim chosen by adapter rank→reload-cost from the existing cost model | cost ranking is a pure function of static rank; deterministic tie-break by id (loses: observed/variable reload latency and access-frequency weighting) | dynamic/learned cost models (loses: cost estimation that improves over time from telemetry — a feedback loop or learned predictor — a categorically different mechanism from the static-vs-observed gap in the Simplified column) |
| Pre-placement | declared adapter→instance-index assignment, resident at t=0 with no cold-load charge (D4) | seeding bypasses the load event entirely — instantaneous at t=0 (loses: real one-time provisioning/weight-transfer latency at startup) | mid-run migration, replication (loses: adapters moving/copying between instances after t=0 in response to load) |
| Signal freshness (D7) | ground-truth-immediate residency for the strict routing decision | route-to-holder pins `ResidentAdapters` to Immediate freshness, overriding the tiered Periodic default (loses: propagation-delay fidelity for this one field — a real distributed gateway could mis-route on a stale view, which BLIS's single-threaded Immediate read makes structurally impossible) | modeling stale-view mis-routing as an experimental variable (loses: the ability to study, as a controlled factor, how routing quality degrades under realistic snapshot propagation delay — the Immediate override makes that path structurally unreachable this round) |
| Triggers | reactive triggers invoke policies at their existing decision points | periodic trigger is a declared, deterministic interval, inert this round (loses: nothing this round — no action fires) | any periodic *action* (prefetch/migrate/scale) (loses: proactive, request-independent placement adjustments) |
| Provenance | run-level effective triple recorded in output, omitted when all-baseline (D8) | one triple per run — policies are run-scoped | per-request policy attribution (loses: the ability to attribute a specific request's routing/eviction outcome to a per-request policy decision — not modeled, as policies are run-scoped) |
| Randomness | none new — routing reuses the existing `SubsystemRouter` tie-break; eviction is deterministic by id tie-break; creation is deterministic by declared, ordered seeding (no tie to break) | — | randomized tie-breaking in the new policies (loses: stochastic exploration a search agent might later want — would add a named RNG subsystem then) |

**Per-seam model scoping (Banks et al. six criteria).** Each seed policy is defended below;
criteria 1 (accuracy impact), 5 (what breaks if omitted), and 6 (simplest version) are the
load-bearing ones:

- **route-to-holder** — (1) changes which instance serves a request under static placement, the
  central variable of analysis question 1; (5) omitting it makes the static-placement
  reproduction impossible (the whole motivation); (6) simplest form is a hard candidate filter
  reusing existing scoring (D1).
- **rank/cost-aware eviction** — (1) changes churn/cold-load rate under skew, analysis question
  2; (5) omitting it leaves only LRU, so the ablation cannot be run; (6) simplest form reuses the
  already-fitted rank→cost model (D2), no new fidelity claim.
- **pre-placement** — (1) removes demand cold-loads for seeded adapters (SC-002 metric); (5)
  omitting it leaves only emergent residency; (6) simplest form is direct t=0 seeding (D4).
- **periodic trigger** — answers **zero** analysis questions this round; it is included as a
  deliberate, flagged exception under the design-guidelines "Target module" precedent (§4.2 —
  e.g. AutoScaler existed as scoped-but-inert scaffolding before PR11), justified because the
  trigger→policy taxonomy is cheap to define now and expensive to retrofit later (D5). Its inert
  form changes zero behavior (INV-PS3), so it is not "fidelity for its own sake" — it adds a
  taxonomy slot, not a modeled mechanism.

## 5. Real-System Correspondence

Not every seed policy reproduces an existing production system; some are research proposals.
This table makes the distinction explicit (design-guidelines §4.4).

| Seam / policy | llm-d | vLLM / S-LoRA / Punica (single-instance engines) | Nature |
|---|---|---|---|
| Routing: existing scorers | Endpoint Picker (biasing) | N/A (single instance, no cross-instance routing) | Reproduces llm-d EPP bias |
| Routing: route-to-holder | strict gateway placement | N/A | Reproduces a **gateway/control-plane** concept (`Tantawi2025`); not an engine concept |
| Eviction: LRU | N/A (engine-internal) | S-LoRA / Punica use LRU-style adapter eviction | Reproduces mainstream engine practice |
| Eviction: rank/cost-aware | N/A | departs from LRU norm | **Research proposal**, BLIS-native — *inspired by* the rank-aware signal in `Li2025` (Toppings), applied to the eviction seam. Note: the merged 2026-07-15 design characterizes Toppings as rank-aware *routing*; whether the paper prescribes a victim/eviction rule at all is tracked as an open item in §14 (not yet confirmed). This is a repurposing of the rank-aware idea into eviction, **not a direct reproduction** of a published evictor. |
| Creation: on-demand | dynamic load | vLLM/S-LoRA dynamic adapter loading | Reproduces engine default |
| Creation: pre-placement | static offline placement | N/A (engines load on demand) | Reproduces a **control-plane** concept (`Tantawi2025`) |

**Implication:** route-to-holder and pre-placement live at the gateway/control-plane layer
(llm-d-adjacent), not inside a single vLLM/SGLang engine — which is exactly why BLIS (a
cluster simulator) is the right place to model them. rank/cost-aware eviction is explicitly a
research departure, so its validation target (§11) is mechanism-fidelity to the cited policy,
not agreement with a mainstream system.

## 6. Module Contracts

Per the design-guidelines module-contract template (observes / controls / owns / invariants /
events / extension friction). Extension-type attribution per seam is in §8.

### 6.1 Routing policy — Policy Template over the frozen `RoutingPolicy` interface (+ a narrow cluster-layer freshness override for the strict variant, D7)

| Aspect | Contract |
|---|---|
| **Observes** | Request adapter identity; per-instance routing snapshots including the resident-adapter membership set (at **Immediate** freshness for the strict policy — D7, a documented **INV-7** exception for this one field); existing routing signals (queue depth, KV utilization, cache hit rate, in-flight). |
| **Controls** | Target-instance selection. route-to-holder restricts the candidate set to holders when non-empty; among holders, selection reuses the existing scorer composition (including its existing `SubsystemRouter` tie-break). |
| **Owns** | No mutable state (stateless policy; consistent with the existing Router contract). |
| **Invariants** | Selects from the non-empty candidate list; **when route-to-holder is the active policy and ≥1 holder exists, never selects a non-holder (INV-PS1)** — the baseline weighted-routing policy is unaffected and may route to a non-holder (it biases, never forbids); reads adapter id/residency only, never `OutputTokens` (**INV-L6**/INV-9); deterministic given inputs + RNG state. |
| **Events** | Consumes the routing-decision point; produces the target selection. No new event type. |
| **Extension friction** | A biasing scorer = ~2 touch points (registry entry + validation name). A *strict* candidate-constraining policy = ~3–4 touch points (policy + selection wiring + validation + the D7 cluster-layer freshness override; see §10). |
| **Failure modes** | No holder for the requested adapter → documented fallback (D1). Empty candidate list → existing router error path, unchanged. |

### 6.2 Eviction policy — Backend Swap extracting a seam from the hardcoded LRU

| Aspect | Contract |
|---|---|
| **Observes** | The set of currently-resident, **unpinned** adapters and their eviction-relevant attributes (recency for LRU; rank→reload-cost, via the eviction context of D2, for rank-aware). |
| **Controls** | The victim id when a slot is needed. |
| **Owns** | No policy state; recency ordering remains owned by the resident set. The **eviction context** (D2) is owned/constructed by the instance simulator (which already owns the resident set and cost model) and read by the policy. |
| **Invariants** | Never selects a pinned adapter (**INV-L5**); preserves `\|resident\| ≤ capacity` (**INV-L2**); returns "no victim" iff every resident adapter is pinned (no eligible candidate remains); reads adapter id/rank/residency only, never `Request.OutputTokens` (**INV-L6**/INV-9 — the eviction context's rank data is a new surface that must not leak the oracle); deterministic (explicit id tie-break). |
| **Events** | Invoked synchronously at the capacity-pressure decision point inside the cold-load path; produces no new event. |
| **Extension friction** | ~3–4 touch points (policy + eviction-context/rank plumbing + validation). Over the ~3-file target by ~1 because rank is not reachable at the eviction site today (D2); justified in §10. |
| **Failure modes** | **No victim (all candidates pinned)** → the cold-load does **not** proceed this tick; the waiting request stays queued and **INV-8 (work-conserving)** guarantees a later `StepEvent` retries once a pin clears — the no-deadlock guarantee already tested for LRU (2026-07-15 §15, INV-L5 verification scenario (a)). A rank-aware policy with a degenerate comparator must still terminate and must still return "no victim" only when all are pinned. |

### 6.3 Creation policy — Subsystem Module over start-of-run residency + admit-on-miss

| Aspect | Contract |
|---|---|
| **Observes** | The declared adapter→instance assignment (pre-placement) and the adapter registry; at miss time, the routed request and instance snapshot. |
| **Controls** | Which adapters are resident at t=0 on each instance (via the `Initial` trigger); whether to admit an adapter on a resident miss (via `OnResidentMiss`). |
| **Owns** | No mutable state beyond the residency it seeds into the resident set. |
| **Invariants** | Seeded residency respects per-instance capacity (**INV-L2**; rejected at startup otherwise, **INV-PS2**); seeded adapters must be registered; on-demand default seeds nothing and always admits (today's behavior, **INV-L1**); a pre-placed adapter incurs no cold-load latency and no load-count (D4; **INV-L3** — t=0 seeding is not a charged cold load); reads adapter id/registry/residency only, never `Request.OutputTokens` (**INV-L6**/INV-9). |
| **Events** | `Initial` runs at instance construction (a seeding hook, not an event-loop trigger) — at **both** the up-front build loop and the deferred node-ready construction of initial-topology instances (D3). `OnResidentMiss` runs at the existing cold-load decision. No new event type this round. |
| **Extension friction** | ~4–5 touch points (policy + cluster-scoped placement config + seeding at both initial-topology construction sites + validation). Over target by ~1–2 because per-instance targeting is a cluster-topology concern absent from today's instance-agnostic config, and correctness requires seeding both construction sites (D3); justified in §10. |
| **Multi-method justification** | Creation is deliberately a **two-entry-point** contract (`Initial` seeding + `OnResidentMiss` admission) rather than two separate seams because both express one coherent *placement stance* — static vs. on-demand. Splitting them would allow an incoherent mix (e.g. pre-placement seeding coexisting with an on-demand admit that re-loads elsewhere), which no experiment wants. The baseline pairs (nothing seeded, always admit); pre-placement pairs (seed the assignment, then admit-on-miss for any non-seeded adapter — the same liveness-preserving stance that D1 applies to routing). |
| **Failure modes** | Over-capacity or unregistered assignment, or an out-of-range instance index → **rejected at startup** with a clear error (INV-PS2). |

### 6.4 Invariants introduced & preserved

**New invariants (canonical statements).** Each is added to `docs/contributing/standards/invariants.md`
in the PR that first makes it enforceable:

- **INV-PS1 (strict-routing honesty).** *When route-to-holder is the active routing policy and at least
  one instance holds the requested adapter at the routing-decision instant, the served instance is a
  holder.* Scoped to the selection instant (ground-truth residency via D7's Immediate freshness); an
  eviction on the chosen instance *after* selection but *before* service is **not** a violation — it is
  handled by the ordinary `OnResidentMiss` admit-on-miss path. The baseline routing policy is exempt
  (it biases, never forbids). *Added in B-2.*
- **INV-PS2 (pre-placement conservation).** *Every declared (adapter, instance) pre-placement pair for an
  initial-topology instance is resident on that instance at its construction, or the configuration is
  rejected at startup.* No pre-placement is silently dropped or truncated — including for deferred
  initial-topology instances (D3). *Added in B-5/B-6.*
- **INV-PS3 (periodic-trigger inertness).** *A declared periodic trigger produces output byte-identical to
  a run without it, this round.* Trivial: no event is scheduled (D5). *Added in B-7.*

**Preserved invariants.** INV-1 (request conservation — the new routing fallback (D1) and creation
admission (D9) touch the admission/routing path, so conservation is re-verified per §11), INV-6
(determinism / no-op byte-identity — the primary safety property, via INV-L1 and D8), INV-8
(work-conserving — the eviction no-victim path, §6.2), INV-9 / INV-L6 (oracle boundary), INV-13
(run/replay parity), INV-L1/L2/L3/L5 (LoRA no-op inertness, capacity bound, cold-load-charged-once —
including that D4's t=0 seeding is *not* a charged load, §11 — and no eviction of in-use), and INV-7
(signal freshness — preserved for all signals except the `ResidentAdapters` Immediate override under
route-to-holder, D7). All hold under every policy selection. **INV-L4** (adapter HBM memory
conservation) and **INV-L7** (latency-backend parity) are **untouched** by this feature — HBM accounting
stays static (§2 out-of-scope) and no seam alters the per-step overhead factor — so the "preserves and
extends INV-L1…L7" claim in the header holds with L4/L7 preserved trivially.

## 7. Decisions with Trade-offs

**Status legend:** all decisions below are **Proposed** (no code written yet); the column is
updated to **Implemented** as each PR in §13 lands, or **Superseded** if a later decision
replaces it. This is the freshness mechanism that guards against the "Silent Staleness"
anti-pattern for a multi-PR rollout.

| # | Decision | Status |
|---|---|---|
| D1 | route-to-holder is a routing *policy* (candidate-set restriction), not a scorer | Proposed (fallback provisional — §14) |
| D2 | rank-aware eviction consumes an eviction context wired from the adapter registry | Proposed (victim criterion/tie-break provisional — §14) |
| D3 | pre-placement assignment is cluster-scoped, resolved to per-instance subsets | Proposed |
| D4 | pre-placed adapters resident at t=0 with no load latency and no load count | Proposed |
| D5 | periodic trigger reuses the existing self-rescheduling clock pattern; inert this round | Proposed |
| D6 | provenance is a run-level effective triple | Proposed |
| D7 | route-to-holder pins `ResidentAdapters` to Immediate freshness | Proposed |
| D8 | provenance field omitted for all-baseline runs (INV-6 byte-identity) | Proposed |
| D9 | Creation is one two-entry-point seam (`Initial` + `OnResidentMiss`), not two seams | Proposed |

### D1 — route-to-holder is a routing *policy*, not merely a scorer

**Problem.** Part B's illustrative shape registered route-to-holder as "a new named scorer."
But the existing router composes scorers as a *weighted sum* and picks the argmax: a scorer
that scores holders 1.0 and non-holders 0.0 only *biases* toward holders — a non-holder can
still win when other scorers dominate. That cannot express *strict* placement.

**Decision.** route-to-holder constrains the **candidate set** to holders (when non-empty)
*before* the existing scorer composition runs, then selects the best-scored holder using the
current weighted scoring (inheriting its existing `SubsystemRouter` tie-break). Strictness is
a set restriction, not a score.

**No-holder fallback (provisional, pending §14 paper confirmation).** When **no** instance
holds the requested adapter, route-to-holder falls back to **unconstrained weighted routing
over all instances** — i.e. exactly baseline behavior, turning the first request for an
adapter into a normal on-demand cold-load. This preserves liveness and trivially subsumes
INV-6 in the no-adapter case. (Note: because `lora-affinity`, if present in the profile,
scores identically across an all-holder candidate set, it contributes nothing to the argmax
once the set is restricted — harmless, not load-bearing, in strict mode.)

**Rationale.** Strictness is a hard constraint; hard constraints belong in candidate
filtering, not in a soft additive score. Reusing the existing scorer composition among holders
keeps tie-breaking and determinism identical to today.

**Alternatives considered.**
- *Very large scorer weight* — rejected: still probabilistic, not strict; brittle; violates
  INV-PS1 in adversarial cases.
- *Reject the request outright when no holder exists (no fallback)* — rejected: breaks liveness
  and makes the reproduction experiment un-runnable for any adapter not pre-placed; the fallback
  is the liveness-preserving choice.
- *Defer/queue the request until some instance becomes a holder* — rejected: adds a new blocking
  state and starvation risk for no benefit this round (the first request to any not-yet-placed
  adapter would stall indefinitely absent a loader); the unconstrained fallback (which triggers a
  normal cold-load) achieves placement without a new queue.

**What breaks if wrong.** If strictness were a scorer, INV-PS1 could not be guaranteed and the
reproduction experiment (US1) would be invalid.

### D2 — rank/cost-aware eviction requires wiring the adapter registry to the eviction site

**Problem.** Today the eviction decision has no access to adapter rank: the resident-set
entries carry only id + pin count, and the cost accessor reachable at the eviction site exposes
load latency *by id* but not a rank ranking. The adapter registry (which does expose rank) is
defined but never wired into the instance simulator.

**Decision.** Introduce an **eviction context** that exposes, at the decision site, the unpinned
candidate ids together with the rank/reload-cost needed to order them — sourced from the
already-defined adapter registry, wired into the instance simulator alongside the existing cost
model. Baseline LRU ignores the context (byte-identical); rank-aware consumes it. The rank→cost
relationship is **not a new fidelity claim** — it reuses the cost model fitted in the merged
control-plane work (load latency = base + ⌈footprint/bandwidth⌉, footprint ∝ rank). (The exact
victim-selection criterion and tie-break for the rank-aware policy are **provisional** pending
paper confirmation against `Li2025` — tracked in §14.)

**Rationale.** Extending the *narrow* cost accessor with a rank method would change a frozen
production contract and its test doubles; wiring the existing registry interface is additive and
reuses a seam that already exists for exactly this data (R13 — no new single-impl interface). The
eviction context keeps the resident set policy-agnostic: it need not learn about rank.

**Alternatives considered.**
- *Widen the cost accessor with a rank method* — rejected: mutates a frozen interface + mocks for
  one consumer.
- *Store rank inside resident-set entries* — rejected: couples the LRU data structure to cost
  semantics it should not own; changes the resident-set construction contract and its byte-identity.

**What breaks if wrong.** Storing rank in resident entries would make the resident set carry
policy-specific state, breaking clean baseline/rank-aware substitution and LRU byte-identity.

### D3 — pre-placement is a cluster-scoped concern, resolved to per-instance subsets

**Problem.** The LoRA sub-config is *instance-agnostic*: an identical copy is handed to every
instance at construction. "Assign adapter X to instance i" cannot be expressed there without
every instance seeing every other instance's assignment.

**Decision.** The adapter→instance assignment lives at **cluster/deployment scope** (beside node
pools and the routing profile). The **cluster layer resolves each instance's own subset** wherever
an instance belonging to the **initial topology** is constructed, and hands only that subset across
the boundary into instance-local state — mirroring the existing precedent where the cluster resolves
a per-instance GPU type from the node-pool placement and never embeds the full node-pool slice into
every instance. Instance-level (`sim/`) code never receives the full cross-instance map
(principles.md: "never leak cluster state to instance-level code").

**Two initial-topology construction sites (both must seed).** Instance identity/index is resolved
in **two** places, not one: (1) the up-front construction loop at build time, and (2) the deferred
path for instances whose placement waits on node-pool capacity, constructed mid-event-loop when the
node becomes ready. **Both build instances of the *initial, fixed* topology** and both must apply
pre-placement seeding — otherwise a deferred instance silently drops its pre-placed adapters
(violating INV-PS2 and defeating SC-002 in exactly the NodePools + capacity-pressure configuration
the feature must support). This is distinct from the genuinely-scaled path (autoscaler scale-up),
which stays out of scope (§2, cross-scale re-placement): a scaled-in instance starts empty and falls
through to admit-on-miss.

**Implementation caveat (shared low-level constructor).** The deferred (node-ready) path and the
out-of-scope autoscaler scale-up path funnel through the **same** low-level instance constructor, which
carries no flag distinguishing the caller. The upstream distinction *does* exist (only startup-deferred
initial-topology instances pass through the pending-placement queue; autoscaler scale-up never does), so
the seeding hook **must be placed at the deferred path's caller site (after construction returns), not
inside the shared constructor** — otherwise seeding leaks onto autoscaler-added instances, violating §2.
This is exactly the canonical-constructor / R4 "shotgun surgery" trap the guidelines flag. Because seeding
must run at both initial-topology sites (and be gated to fire only at the deferred-path caller, never at the shared constructor's other, autoscaler-scale-up caller),
Creation's touch-point count carries a further +1–2 (§8/§10). §11's pre-placement + live-autoscaler
validation item is the guard that would catch a wrong placement.

**Rationale.** Placement is inherently a cluster-topology decision; the deployment config already
hosts sibling cluster-scoped concerns (instance count, node pools, routing profile). Resolving to
a per-instance subset at the boundary keeps instance code topology-agnostic.

**Alternatives considered.**
- *Per-instance LoRA config list* — rejected: structurally cannot target a specific instance;
  would require post-hoc filtering by index inside instance-agnostic code.
- *Thread the full cross-instance map into every instance's config* — rejected: leaks cluster
  topology into instance-local state, violating separation of concerns.

**What breaks if wrong.** Putting the assignment in instance-agnostic config makes per-instance
targeting impossible and silently seeds every adapter on every instance.

### D4 — pre-placed adapters are resident at t=0 with no cold-load latency and no load count

**Problem.** Should a pre-placed adapter count as a "load" in metrics, and incur cold-load latency?

**Decision.** Pre-placement seeds residency directly at t=0, before any request, incurring
**neither** cold-load latency **nor** a load-count increment. Load counts remain a measure of
*demand-driven* cold loads (the quantity the reproduction experiment compares).

**Rationale.** The value of static placement is precisely that it *avoids* demand cold loads;
counting seeding as a load would mask that in the headline metric (SC-002). t=0 seeding is
instantaneous by construction — there is no request to block, so no latency is meaningful.

**Alternatives considered.**
- *Count seeds as loads* — rejected: conflates provisioning with demand churn; defeats SC-002.
- *Charge a one-time provisioning latency* — deferred (Banks criterion 5): no analysis question
  needs it this round; addable later without changing the seam.

**What breaks if wrong.** Counting seeds as loads would make SC-002 unmeasurable and the
reproduction comparison misleading.

### D5 — the periodic trigger reuses the existing self-rescheduling clock pattern; inert this round

**Problem.** Migration/scaling policies (future) need a periodic, request-independent trigger.

**Prior art (corrected).** BLIS **already has** such a mechanism: `ScalingTickEvent` (a
self-rescheduling clock event driving the autoscaler pipeline on declared simulation-time
intervals). The cluster event-priority space is currently packed **within 0–9**, though the same
`ClusterEvent` scheme already extends below zero for infra-lifecycle events
(`priorityNodeLifecycle = -2`, `priorityInstanceLifecycle = -1`) — so a new slot is a
deliberate-but-low-risk *extension* (below 0 or above 9), not a scarce resource requiring
renumbering of existing constants. The periodic *trigger* this
round is therefore **not a novel event mechanism** — it is the taxonomy slot that a future
placement periodic policy would fill by **reusing/generalizing the `ScalingTickEvent`
self-rescheduling pattern**.

**Decision.** This round, scaffold **only the taxonomy** (the trigger *type* and its config), and
schedule **no event** — nothing is inserted into the event heap, so no priority constant is
consumed and INV-PS3 (byte-identity) holds trivially. When a future policy activates the periodic
trigger, it reuses the `ScalingTickEvent` pattern and must be assigned a **deliberate** priority
slot (packed within 0–9, but the scheme already extends to negatives — a low-risk extension, not a
renumbering; flagged now).

**Rationale.** Defining the taxonomy is cheap; retrofitting the trigger→policy framework later is
expensive. Scheduling no event this round means the scaffold is provably inert (no ordering
guarantee to get right yet) while still fixing the framework shape. Determinism holds because any
future interval is declared config on simulation time.

**Alternatives considered.**
- *Defer the taxonomy entirely* — rejected: guarantees a framework repaint when migration lands.
- *Implement a periodic action now (or schedule an inert heap event)* — rejected: an action is out
  of scope with no consuming experiment; even an inert heap event would consume a priority slot and
  create an ordering guarantee to validate for no benefit.
- *Reuse `ScalingTickEvent` directly rather than a sibling* — deferred to the activating design:
  the two may need to fire independently of whether the autoscaler is enabled; that is a decision
  for when the periodic policy is built, not now.

**What breaks if wrong.** Implementing a periodic action (or heap event) now would add unvalidated
behavior and priority-ordering risk (this repo has prior event-ordering bugs); deferring the
taxonomy would force a later repaint.

### D6 — provenance is a run-level effective triple, recorded in output

**Problem.** No existing output field records which policies a run used; the harness needs the
*effective* (post-bundle-expansion) triple for reproducibility.

**Decision.** Record the effective {routing, eviction, creation} selection as a run-level field in
the metrics output, **computed once at policy/bundle resolution before the event loop starts** (not
accumulated per-event — it is a config-time constant, keeping it out of any state-mutation path).
Policies are run-scoped, so one triple per run suffices; no per-request attribution.

**Rationale.** Reproducibility requires the resolved triple, not just a bundle name (a bundle table
can change). Run-level matches the granularity at which policies are selected and keeps per-request
records unchanged.

**Alternatives considered.**
- *Per-request policy attribution* — rejected (Banks criterion 6): policies are run-scoped;
  per-request records would bloat with a constant.
- *Rely on the harness's own manifest only* — rejected: the run output should be self-describing for
  reproduction from the artifact alone (SC-006).

**What breaks if wrong.** Without a recorded triple, a run's artifact alone cannot be reproduced — a
reader would have to cross-reference the harness manifest, which can drift from what the run actually
resolved (e.g. a bundle name redefined between manifest authoring and execution). See D8 for the
byte-identity constraint on this field.

### D7 — route-to-holder pins `ResidentAdapters` to Immediate freshness

**Problem.** route-to-holder converts the resident-adapter membership set from a *soft bias*
(as the `lora-affinity` scorer uses it) into a *hard candidate filter* that INV-PS1 depends on. That
signal has **tiered freshness**: it is **Periodic by default** (the `--snapshot-refresh-interval` flag
defaults to 50000µs = 50ms, matching CLAUDE.md INV-7 "Periodic by default at 50ms"), and only
**Immediate** when `--snapshot-refresh-interval 0` is explicitly passed (R17/INV-7). So the stale-view
risk is present in the **default** configuration, not merely an opt-in: under Periodic freshness the
router's cached view can claim an instance is a holder after it has evicted the adapter (or miss one it
just loaded), which would let a strict route land on a non-holder — breaking INV-PS1.

**Decision.** When route-to-holder is active, `ResidentAdapters` is read at **Immediate** freshness
(synchronous re-read at the routing decision, like `InFlightRequests`) **regardless** of the global
`--snapshot-refresh-interval`. In BLIS's single-threaded DES an Immediate read reflects exact
ground-truth residency at the routing event, so INV-PS1 is stated and holds in **ground truth**, not
merely "as of the last snapshot."

**Rationale.** A hard constraint requires fresh data; a strict policy reading a stale-by-design
signal is a latent correctness bug. Immediate freshness for one field, only when the strict policy is
selected, is a narrow, local override that leaves every other signal's freshness (and the baseline's
byte-identity) untouched.

**Alternatives considered.**
- *Leave `ResidentAdapters` at the global freshness and reframe INV-PS1 as snapshot-relative* —
  rejected: weakens the invariant to "holder as of the snapshot," which does not honor the reproduction
  experiment's intent (a request must reach an actual holder) and would need the ground-truth race
  window documented as an accepted gap.
- *Force Immediate globally whenever LoRA is configured* — rejected: needlessly changes the freshness
  of the soft `lora-affinity` path and could perturb its behavior.

**What breaks if wrong.** Under Periodic freshness without this override, INV-PS1 is violable and the
static-placement reproduction can silently mis-route to a non-holder.

### D8 — provenance field omitted for all-baseline runs

**Problem.** D6's output field must not break the INV-6/INV-L1 byte-identity golden (the primary
safety property), which requires an adapter-blind / all-baseline run to produce byte-identical stdout.

**Decision.** The effective-triple field is **omitted from stdout whenever every seam is at baseline**
(the all-baseline / adapter-blind run), mirroring the established pattern where per-adapter metrics and
the HBM reservation are omitted/zero when the subsystem is inert. When any non-baseline policy is
active, the field is present. This makes the golden test pass **unchanged** and is a firm decision, not
a micro-plan open item.

**Rationale.** Byte-identity is non-negotiable; an always-present field would change baseline stdout.
Omission-when-inert is the same convention the control-plane subsystem already uses.

**Alternatives considered.**
- *Always emit the triple (carrying an explicit "baseline" value)* — rejected: changes baseline stdout,
  breaking INV-6 unless the golden is regenerated, which defeats the safety check.

**What breaks if wrong.** An always-present field silently breaks the INV-6 golden — the exact
regression this feature's first test exists to catch.

### D9 — Creation is one two-entry-point seam, not two seams

**Problem.** Creation governs two behaviors — `Initial` (t=0 seeding) and `OnResidentMiss` (admit-on-miss).
Should these be one policy with two entry points, or two orthogonal single-method seams?

**Decision.** One **two-entry-point** Creation policy (design-guidelines §5.3 item 2 asks that a
multi-method interface be justified). Both entry points express one coherent *placement stance*: baseline
pairs (seed nothing, always admit); pre-placement pairs (seed the declared assignment, then admit-on-miss
for any non-seeded adapter — the same liveness-preserving stance that D1 applies to routing).

**Rationale.** Splitting them would allow an incoherent mix (e.g. pre-placement seeding coexisting with an
on-demand admit that re-loads the same adapter elsewhere), which no experiment wants. Coupling them under
one named policy keeps the stance internally consistent.

**Alternatives considered.**
- *Two separate seams (seeding policy + admission policy)* — rejected: permits incoherent combinations and
  doubles the config surface for no experimental benefit this round.

**What breaks if wrong.** Two independent seams could be configured into a self-contradictory stance
(seed-static + admit-and-reload-elsewhere), producing placement behavior that matches no intended policy.

## 8. Extension Points & Types (registries)

Each seam becomes a **named registry** so adding a policy is a localized, additive change — the
minimal diff surface both a human and an automated search agent want (spec FR-005, SC-004). The three
seams are **different extension types**, each with its own recipe and acceptance gate:

| Seam | Extension type | Recipe / gate |
|---|---|---|
| Routing (route-to-holder) | **Policy Template** (§5.2) — new algorithm behind the already-frozen `RoutingPolicy` interface, **+ a narrow cluster-layer freshness override** (D7) | Implements the interface; deterministic; handles empty/edge candidate lists; the D7 override is the one non-pure-template touch (§10). |
| Eviction (LRU→registry) | **Backend Swap** (§5.4) — extract an interface from currently-hardcoded logic | **Phase-A gate**: all existing tests pass, no behavior change, factory returns the existing LRU by default (byte-identical). Phase-B adds rank-aware. |
| Creation (new state+triggers) | **Subsystem Module** (§5.3) — new module with its own state, triggers, config | Module contract (§6.3); no-op default; startup validation; testable with mocks. |

The overall feature is classified **Subsystem Module** (its most demanding component) for
process/complexity purposes (§10).

**Touch-point targets** (design-guidelines §4.5 reference: policy template ≈ 3 files):

| Seam | Add one more policy | Notes |
|---|---|---|
| Routing (biasing scorer) | ~2 files | Meets/beats target (mature seam). |
| Routing (strict, i.e. route-to-holder incl. D7 freshness) | ~3–4 files | Exceeds by ~1; justified §10. D7's Immediate-freshness override is **mandatory whenever route-to-holder ships** (INV-PS1 is unenforceable without it), so this is the only strict-routing deliverable in scope — not an optional add-on. |
| Eviction | ~3–4 files | Exceeds by ~1 (eviction-context/rank plumbing, D2); justified §10. |
| Creation | ~4–5 files | Exceeds by ~1–2 (cluster-scoped placement config + seeding at both initial-topology construction sites, D3); justified §10. |

## 9. Trigger Taxonomy

**Class 1 — reactive (implemented this round):**

| Trigger | Fires when | Class | Policy invoked |
|---|---|---|---|
| `Initial` | instance construction — the up-front build loop, **and** the deferred-instance path (node-ready) for initial-topology instances awaiting capacity (D3) | **construction-time seeding hook** (not an event-loop trigger; needs no priority) | Creation → seed residency |
| `OnRoute` | request routing (existing decision point) | reactive/endogenous | Routing → select instance |
| `OnResidentMiss` | routed request needs an absent adapter (existing cold-load decision) | reactive/endogenous | Creation → admit-on-miss |
| `OnCapacityPressure` | slot needed and resident set full (existing eviction point) | reactive/endogenous | Eviction → pick victim |

No new *reactive* event type is introduced — reactive triggers reuse existing decision points.

**Class 2 — periodic (scaffolded, inert this round):**

| Trigger | Fires when | This round |
|---|---|---|
| `Periodic` (interval Δt) | every Δt of simulation time, request-independent | **type + config only; no event scheduled** (INV-PS3). Future activation reuses the `ScalingTickEvent` self-rescheduling pattern and needs a deliberately-assigned priority (packed within 0–9, but the scheme extends to negatives — a low-risk extension). Unlocks proactive eviction, prefetch, migration, scaling. |

## 10. Complexity Tracking

This feature is processed as a **Subsystem Module** (design-guidelines §5.3) and exceeds the ≤3-file
policy-template target for two of three seams. Justification:

| Over-target area | Extra cost | Why justified |
|---|---|---|
| Eviction rank plumbing (D2) | +1 file (eviction context + registry wiring) | Rank is genuinely unreachable at the eviction site today; the alternative (mutating the frozen cost accessor) is worse. One-time cost; every future eviction policy reuses the context. |
| Cluster-scoped pre-placement (D3) | +1–2 files (deployment-config field + resolution at **both** initial-topology construction sites — up-front loop and deferred node-ready path — with seeding gated at the deferred *caller* site so it does not fire in the shared constructor's autoscaler-scale-up caller) | Per-instance targeting is a cluster-topology concern with no home in instance-agnostic config; seeding both construction sites (while the shared low-level constructor has exactly two callers — the deferred node-ready path and the out-of-scope autoscaler scale-up — so the hook goes at the deferred *caller*, not inside the shared constructor) is required for correctness (INV-PS2) under NodePool capacity pressure. |
| Strict-routing freshness override (D7) | +1 touch (cluster-layer `ResidentAdapters` freshness override when route-to-holder is active) | route-to-holder's selection logic is a pure Policy Template, but the Immediate-freshness guarantee (INV-PS1 ground truth) requires a narrow, policy-conditioned change in the cluster snapshot layer, similar in spirit to how `InFlightRequests` is exempt from periodic staleness by construction. One-time; the mechanism choice is deferred to micro-plan (§14). |

**Touch-point yardstick.** Design-guidelines §4.5 publishes numeric references by *concrete example*
(policy template ~3, KV tier ~4, config parameter ~2, observable metric ~3, latency-model backend ~2,
batch-formation strategy ~2) but none labeled by extension-*taxonomy* type. The ~3-file policy-template
bar is the general yardstick used above; for the Eviction Backend Swap the tighter latency-model-backend
row (~2 files) is the closer analog. Either way these are borrowed references, not official targets for
Backend Swap / Subsystem Module.

These are one-time structural costs paid once per seam, not per policy. After they land, adding a
further eviction, creation, or routing policy returns to the ~2–3-file target.

## 11. Validation Strategy

**Verification (correctness — which invariants):**
- **No-op golden (INV-L1/INV-6, SC-001):** the existing adapter-blind byte-identity test must pass
  **unchanged** — guaranteed by D8 (provenance omitted when all-baseline). A companion test asserts
  byte-identity when every seam is *explicitly* set to baseline. First test written per PR (TDD).
- **Strict routing (INV-PS1):** given a holder exists, a request is never routed to a non-holder —
  tested under **both** freshness modes, including the **default Periodic-freshness** scenario where a
  concurrent eviction would stale the snapshot, asserting D7's Immediate override (which route-to-holder
  forces) holds the invariant in ground truth; given no holder, the D1 fallback applies (provisional per §14). A **property test** (≥100 random
  holder-configurations, per the repo's `TestProperty_*` precedent) complements the example-based tests.
- **Eviction pin-safety + no-deadlock (INV-L5, INV-L2, INV-8):** given a full set with one unpinned
  candidate, only that candidate can be evicted under any policy; given several unpinned candidates of
  differing rank, rank-aware picks the declared victim with a deterministic id tie-break; given **all
  candidates pinned**, no victim is chosen and the waiting request runs once a pin clears (liveness — the
  INV-L5(a) no-deadlock scenario, re-run for the pluggable policies).
- **Pre-placement conservation (INV-PS2):** valid assignment → adapters resident at t=0 with zero
  load-count; over-capacity, unregistered adapter, **or out-of-range instance index** → startup fails
  with a clear error.
- **Pre-placement under deferred construction (INV-PS2 / SC-002):** a fixture-based test (pattern:
  `TestNodeReadyEvent_*`, since the deferred path is not live-CLI-reachable today — §14) constructs an
  initial-topology instance via the deferred node-ready path and confirms its declared pre-placement is
  resident with zero cold-loads — guarding D3's identified single point of failure (seeding must apply
  at **both** initial-topology construction sites, never on the shared constructor's autoscaler caller).
- **Oracle boundary (INV-L6/INV-9):** both legs of the INV-9 precedent — a behavioral test that a
  route/evict/admit decision is invariant to `OutputTokens`, and a static/CI code-path check that the
  new routing/eviction-context/creation files never reference `OutputTokens` (the D2 rank data and D3/D4
  seeding are new surfaces that could tempt such a read).
- **Periodic inertness (INV-PS3):** a declared periodic trigger yields byte-identical output (trivial —
  no event scheduled, D5).
- **Run/replay parity (INV-13):** export→replay with identical policy selection yields identical
  per-request metrics. **Mechanism:** the new policy names, bundle name, and pre-placement assignment
  are added to the run/replay **sync-point** field set, with a `logrus.Fatalf` fail-fast for any
  selection replay cannot honor (per invariants.md INV-13), and a coverage-matrix parity case
  (à la the existing run/replay byte-identity matrix). Assigned to B-7 (§13).
- **Conservation (INV-1):** a request-conservation matrix check across {routing, eviction, creation}
  selections (precedent: the H12 conservation-across-routing-policies check), since the new routing
  fallback and creation admission touch the admission/routing path.
- **Pre-placement + live autoscaler (reachable via existing flags once shipped):** with pre-placement configured *and* the
  autoscaler enabled (`--model-autoscaler-interval-us`), a run in which an instance is added or drained
  mid-run must preserve INV-PS2/INV-1/INV-L2 — a scaled-in instance starts empty and admits on-miss
  (no seeding of the *declared* assignment onto it this round; cross-scale re-placement is out of
  scope, §2). This case is verified as part of this rollout because the combination is reachable via existing flags, not
  gated behind a future module.
- **Registry validation (FR-004):** selecting an unregistered policy name for any seam — or an unknown
  bundle name — fails at startup with a clear error listing the valid names for that seam.
- **Provenance correctness (SC-006, FR-015/FR-016, US3):** a bundle name expands to the correct triple;
  an explicit per-knob override changes only that knob (others take the bundle's values); and the recorded
  effective triple round-trips to an equivalent run configuration (reconstructable from the record alone,
  without the original command line).
- **Cold-load charged-once under mixed pre-placement (INV-L3):** in a run where pre-placement is active,
  a demand miss for a **non**-pre-placed adapter is still charged exactly one cold-load (the seeding and
  demand-load paths are disjoint; D4 must not perturb INV-L3).
- **Determinism + parity (SC-005):** same seed + same policy selection → identical routes, victims,
  seedings across two runs, and an export→replay round-trip (INV-13, above) yields identical per-request
  metrics. **Common-random-numbers caveat:** strict CRN validity for the AQ1/AQ2 paired comparisons
  requires positional (RNG-free) router tie-breaking — see the §12 randomness caveat.
- **SC-004 (extension friction)** is verified by inspection against the touch-point tables in §8/§10, not
  an automated test.

**Validation (fidelity — mechanism, not published numbers):** the **AQ1 reproduction experiment**
(in `lora-control`) compares static pre-placement + route-to-holder against the landed
`llmd-affinity-baseline`. Because that baseline is itself a BLIS-simulated bundle (not real telemetry
or the papers' reported figures), this is **structural/mechanism fidelity** — it confirms the seams
implement the *algorithm* the cited policies describe (route-to-holder and pre-placement trace to a
cited passage in `Tantawi2025`; the rank-aware eviction↔`Li2025` correspondence is **provisional** per
D2/§14 and may resolve to "inspired by" rather than "traces to"), **not** agreement with any paper's
reported metric. Absolute-fidelity claims
against the Digital Twin remain bounded exactly as in the prior LoRA work (base-mismatch caveats
documented there). This distinction is stated so no reader mistakes the ablation for external
validation.

**Falsification:** INV-PS1 is falsified by any strict-mode route to a non-holder while a holder exists
(the D7 stale-snapshot test targets this); SC-002 is falsified by any nonzero cold-load for a pre-placed
adapter (note: the *first* request to a **non**-pre-placed adapter is D1's fallback path — an expected
cold-load, **not** an SC-002 violation); SC-003 by the reload-cost monotonicity assertion below.

**Measurable success criteria** (tightening the spec where "measurably" was vague): SC-003 — in a
scenario **deliberately constructed so LRU-recency order and rank/cost order disagree** (skewed
adapter popularity + adapters of differing rank, per spec.md's precondition), holding routing/creation
fixed and seed fixed, the rank-aware policy's chosen victim **tracks the reload-cost criterion** (a
monotonicity assertion: the victim is the **lowest-reload-cost** unpinned candidate — per §4/D2's
rank→reload-cost model — not merely "differs from LRU"; the exact victim rule is **provisional** per
D2/§14 and this test tracks whatever rule paper-confirmation settles on). This mirrors the control-plane design's
rank-sensitivity gate (2026-07-15 §15, "Rank/uniqueness sensitivity") and specifically defeats a
rank-*ignoring* bug that a mere "differs from LRU" check would pass. Deterministic, so a single seed
suffices. SC-002 — 100% of requests for pre-placed adapters served by a holder AND zero cold-loads for
pre-placed adapters (an in-repo, comparison-free check; distinct from the **AQ1 reproduction
experiment** above, which is the cross-run comparison in `lora-control`).

## 12. DES Design Review Checklist

| Question | Answer |
|---|---|
| What analysis questions does this design help answer? | Two, stated in §1: static-placement comparison, and eviction-policy ablation under skew. |
| What is modeled, simplified, omitted? | §4 table, each Simplified/Omitted cell naming the behavior lost. |
| What events are introduced/modified? Exogenous/endogenous? | No new reactive event (reuses existing decision points). `Initial` is a construction-time hook. Periodic trigger is scaffolded taxonomy only — **no event scheduled** this round. |
| How do new events interact with tie-breaking? | None scheduled this round → no ordering guarantee needed. A future periodic policy reuses `ScalingTickEvent`'s pattern and gets a deliberately-assigned priority (packed within 0–9; scheme extends to negatives, so a low-risk extension). |
| What new state is introduced? Who owns it? | Cluster-scoped adapter→instance assignment (deployment config), resolved to per-instance subsets (D3); the eviction context (instance simulator, D2); seeded residency (resident set, already owned). No new per-request state. |
| What new metrics are derived? Incremental or on-demand? | Run-level effective policy triple (provenance), **computed once at policy resolution before the event loop — not accumulated per-event** (D6). Per-adapter load/eviction counts already exist. |
| How will correctness be verified? | §11 verification: no-op golden + per-seam behavioral + property + determinism + parity + conservation, INV-1 + INV-L1/L2/L3/L5/L6 + INV-6/7/8/9/13 + INV-PS1/PS2/PS3. |
| How will fidelity be validated? | §11 validation: mechanism-fidelity reproduction vs `llmd-affinity-baseline` (explicitly not paper-number validation). |
| New randomness? Which subsystem? | **None new.** Eviction deterministic by id; creation deterministic by ordered seeding; routing reuses the existing `SubsystemRouter` tie-break. A future randomized policy would add a named subsystem then. **CRN (common random numbers, §2.5):** the design adds no new RNG stream, but it is not automatically CRN-valid — in cluster mode the existing `SubsystemRouter` tie-break RNG is active by default, and route-to-holder's candidate-set restriction changes router tie multiplicity (and, for the AQ2 eviction ablation, differing residency histories feed the `lora-affinity` score and change ties indirectly), desynchronizing the shared stream between arms. Strict CRN would need RNG-free (positional) router tie-breaking; the router *constructor* already supports this (a nil RNG selects positional tie-break), but no CLI flag or cluster-config field threads nil through to it today — tracked as a §14 item, not something `lora-control` can enable now. This is a comparison-hygiene caveat, not a correctness issue: INV-6/INV-PS1 hold within any single run regardless. |
| Simplest version that answers the same questions? | Baseline defaults byte-identical; each seam ships exactly one real policy beyond baseline; periodic trigger scaffolded but fires nothing. |

## 13. Suggested PR Roadmap (Small-tier, each no-op-safe)

Refines Part B §B7 with the friction findings. Each PR keeps the no-op default byte-identical.
**Parallelism:** once a seam's registry PR lands (B-1, B-3, B-5), its seed-policy PR (B-2, B-4, B-6)
can proceed concurrently with the other seams' work, coordinated only by behavioral-contract tests.

| PR | Scope | Extension type / notes |
|---|---|---|
| **B-1** | Routing: registry framing; register existing scorers/policies as defaults (byte-identical). | Mature seam; smallest PR. |
| **B-2** | Routing: add route-to-holder strict policy (candidate-set restriction, D1) + Immediate-freshness override (D7). | Policy Template (§5.2). |
| **B-3** | Eviction: extract the eviction-policy seam; register `lru` default (byte-identical); introduce the eviction context (D2). | **Backend Swap** (§5.4) — Phase-A gate: existing tests pass, factory returns LRU by default. |
| **B-4** | Eviction: add rank/cost-aware policy (consumes eviction context). | Policy Template (§5.4 Phase-B). |
| **B-5** | Creation: creation-policy seam with on-demand default; seat `Initial` + `OnResidentMiss`; cluster-scoped placement config resolved per-instance (D3). | Subsystem Module (§5.3) — includes structural placement-config plumbing. |
| **B-6** | Creation: pre-placement policy (t=0 seeding, D4) + startup validation (INV-PS2). | Policy Template over the B-5 seam. |
| **B-7** | Periodic-trigger taxonomy scaffold (inert, D5/INV-PS3) + strategy bundles + provenance (D6/D8) + INV-13 sync-point/`Fatalf` + docs. | No behavior change beyond bundle resolution + omitted-when-baseline output field. |

**Grouping:** one tracking/epic issue ("LoRA placement-policy seams") with B-1…B-7 as sub-issues,
generated by `/speckit.taskstoissues` from `tasks.md`.

## 14. Open Items to Resolve at Micro-Plan Time

- **Paper confirmation (provisional decisions):** confirm the route-to-holder no-holder fallback (D1)
  against `Tantawi2025`, and the rank/cost-aware victim criterion/tie-break (D2) against `Li2025`
  (Toppings) — including whether Toppings prescribes an *eviction* rule at all, or only rank-aware
  *routing* (per the 2026-07-15 design's characterization); if the latter, D2/§5 stay labeled a
  BLIS-native repurposing rather than a reproduction. D1's fallback and D2's victim rule are written as
  provisional; a check must trace the implemented mechanism to a cited passage from each paper. If either
  paper prescribes different behavior, update the decision and its Status.
- Finalize the eviction-context contents against the real eviction call site (exact rank source, id
  tie-break key).
- **Pre-placement seeding site (shared-constructor gating).** Place the `Initial` seeding hook at the
  deferred path's caller (after the low-level instance constructor returns), or thread an explicit
  initial-topology discriminator to the caller — either way the seeding call itself stays **outside** the
  shared constructor (a discriminator must not reintroduce seeding logic inside it, which would re-open
  the R4 shotgun-surgery risk), so seeding never fires for the autoscaler scale-up caller (D3, §10). The
  call-graph shape (two callers) is confirmed in D3/§10; only the gating mechanism is the micro-plan
  choice. Confirm at micro-plan against the actual construction call sites.
- **Housekeeping (cross-doc).** Two stale references in governance docs, to update separately (out of
  scope for this PR series): (a) the design-guidelines module map (§4.2) still lists the AutoScaler as
  "planned / Target — PR11" though it is merged and operational; (b) `invariants.md`'s INV-7 freshness
  table predates the `lora-affinity` scorer (#1469) and has no `ResidentAdapters` row — it should gain
  one so INV-7's canonical statement covers the field D7 overrides.
- Confirm the deployment-config shape for the adapter→instance assignment and the exact startup-validation
  error messages (over-capacity, unregistered, out-of-range index).
- Confirm whether route-to-holder's Immediate-freshness override (D7) is wired per-field at policy
  selection or via a dedicated snapshot mode — a micro-plan mechanism choice; the behavioral contract
  (ground-truth residency at the routing event) is fixed here.
- **CRN lever does not exist yet.** Strict common-random-numbers validity for the paired experiments
  (§12) needs RNG-free/positional router tie-breaking, but BLIS currently always wires the
  `SubsystemRouter` RNG into the router. The constructor's positional path already exists (nil RNG), but
  no CLI flag or cluster-config field threads nil through to it, so enabling the §12 recommendation
  requires a small net-new knob — a separate change, not assumed available to `lora-control` today.
- **Deferred-construction reachability.** The deferred node-ready construction site (D3's second
  initial-topology site) is not triggered by any live `blis run`/`replay`/`observe` today — initial
  nodes are constructed Ready with no provisioning delay, and the node-provisioning path is exercised
  only via test-harness injection. So an INV-PS2-under-NodePool-pressure test uses the same fixture
  pattern as the existing `TestNodeReadyEvent_*` tests, not an end-to-end CLI run. The caller-site
  seeding fix (D3) is correct regardless of what triggers that caller.
