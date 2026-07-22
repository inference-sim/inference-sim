# LoRA Placement-Policy Seams — Design

**Date:** 2026-07-22
**Status:** Draft (pending convergence review + human approval)
**Species:** System Overview (multi-PR feature spanning Routing, Eviction, Creation modules)
**Feature branch:** `008-lora-policy-seams`
**Spec:** [`specs/008-lora-policy-seams/spec.md`](../../specs/008-lora-policy-seams/spec.md)

**Builds on:** the merged LoRA control-plane subsystem (adapter identity & registry,
per-instance resident set with LRU + pin/capacity, cold-load creation gate, adapter
cost model, static HBM accounting, `lora-affinity` scorer, per-adapter metrics —
`docs/plans/2026-07-15-lora-control-plane-design.md`).

**Source input:** Part B of the LoRA policy-seams design authored in the `lora-control`
repo (`docs/superpowers/specs/2026-07-22-lora-policy-seams-design.md`) and lora-control
issue #12. This document is the BLIS-native design that Part B seeds; it supersedes
Part B's illustrative Go shapes with behavioral contracts grounded in the real code.

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

## 2. Scope

**In scope:**
- Three named policy seams with baseline defaults byte-identical to today: **Routing**
  (which instance serves a request), **Eviction** (victim under capacity pressure),
  **Creation** (start-of-run residency + admit-on-miss).
- Three seed policies: **route-to-holder** (strict routing), **rank/cost-aware eviction**,
  **pre-placement** (declared adapter→instance seeding at t=0).
- A **(trigger → policy)** taxonomy distinguishing reactive triggers (implemented) from a
  periodic trigger (scaffolded, inert).
- **Named strategy bundles**: a config name resolving to a {routing, eviction, creation}
  triple, with per-knob override.
- **Provenance**: the effective (post-expansion) policy triple recorded in run output.

**Explicitly out:**
- Migration and scaling *policies* (the periodic trigger is scaffolded; no policy consumes
  it this round).
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
   t=0  ──▶ Creation.Initial (seed residency)
            │
   arrival ─┼──▶ Routing (pick instance) ──▶ instance wait queue
            │         │
            │         └─(request needs an absent adapter)─▶ Creation.AdmitOnMiss
            │                                                    │
            │                                     (slot needed)  ▼
            └───────────────────────────────────▶ Eviction (pick victim)
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

A **trigger** is the simulation event that invokes a policy. Triggers are orthogonal to
policies: the taxonomy names reactive triggers (start-of-run, on-route, on-resident-miss,
on-capacity-pressure) and a periodic (interval-driven) trigger, so a future proactive
policy attaches to the periodic trigger without repainting the framework.

## 4. Modeling Decisions

| Concern | Modeled | Simplified | Omitted (this round) |
|---|---|---|---|
| Routing strictness | route-to-holder forbids non-holders when a holder exists | "best-scored holder" reuses existing scorer composition among holders | soft/probabilistic strictness knobs |
| Eviction cost | victim chosen by adapter rank→reload-cost from the existing cost model | cost ranking is a pure function of static rank; deterministic tie-break by id | dynamic/learned cost, access-frequency models |
| Pre-placement | declared adapter→instance-index assignment, resident at t=0 with no cold-load charge | seeding bypasses the load event entirely (instantaneous at t=0) | mid-run migration, replication policies |
| Triggers | reactive triggers invoke policies at their existing decision points | periodic trigger is a declared, deterministic interval | any periodic *action* (prefetch/migrate/scale) |
| Provenance | run-level effective triple recorded in output | one triple per run (policies are run-scoped, not per-request) | per-request policy attribution |
| Randomness | none — all seed policies are deterministic functions of state + config | — | randomized tie-breaking (no new RNG subsystem) |

**Model-scoping note (Banks et al.):** each seam ships baseline + ≥1 real policy, so no
interface is speculative (R13). The periodic trigger is the one deliberately-inert
addition; it is included now only because retrofitting the trigger taxonomy later would
force a framework repaint (a known, high-cost refactor), and its no-op form changes zero
behavior (INV-6 preserved).

## 5. Module Contracts

Per the design-guidelines module-contract template (observes / controls / owns / invariants
/ events / extension friction).

### 5.1 Routing policy (extends the existing Router module)

| Aspect | Contract |
|---|---|
| **Observes** | Request adapter identity; per-instance routing snapshots including the resident-adapter membership set; existing routing signals (queue depth, KV utilization, cache hit rate, in-flight). |
| **Controls** | Target-instance selection. route-to-holder restricts the candidate set to holders when non-empty; among holders, selection reuses the existing scorer composition. |
| **Owns** | No mutable state (stateless policy; consistent with the existing Router contract). |
| **Invariants** | Must select from the non-empty candidate list; when ≥1 holder exists, must never select a non-holder; deterministic given same inputs + RNG state. |
| **Events** | Consumes the routing-decision event; produces the target selection. No new event type. |
| **Extension friction** | A biasing scorer = 2 touch points (registry entry + validation name). A *strict* policy that constrains the candidate set = ~3 touch points (policy + selection wiring + validation). |

### 5.2 Eviction policy (extracts a seam from the resident set)

| Aspect | Contract |
|---|---|
| **Observes** | The set of currently-resident, **unpinned** adapters and their eviction-relevant attributes (recency for LRU; rank→reload-cost for rank-aware). |
| **Controls** | The victim id when a slot is needed. |
| **Owns** | No policy state; recency ordering remains owned by the resident set. |
| **Invariants** | Never selects a pinned adapter; preserves `loaded ≤ capacity`; returns "no victim" iff every candidate is pinned; deterministic (explicit tie-break). |
| **Events** | Invoked synchronously at the capacity-pressure decision point inside the cold-load path; produces no new event. |
| **Extension friction** | ~3–4 touch points (policy + the eviction-context plumbing that exposes rank at the decision site + validation). Higher than a pure policy template because rank is not reachable at the eviction site today (see Decision D2). |

### 5.3 Creation policy (new seam over start-of-run residency + admit-on-miss)

| Aspect | Contract |
|---|---|
| **Observes** | The declared adapter→instance assignment (pre-placement) and the adapter registry; at miss time, the routed request and instance snapshot. |
| **Controls** | Which adapters are resident at t=0 on each instance; whether to admit an adapter on a resident miss. |
| **Owns** | No mutable state beyond the residency it seeds into the resident set. |
| **Invariants** | Seeded residency respects per-instance capacity (rejected at startup otherwise); seeded adapters must be registered; on-demand default seeds nothing and always admits (today's behavior); seeding a pre-placed adapter incurs no cold-load latency and no load-count (see Decision D4). |
| **Events** | `Initial` runs at simulator build (before any request); `OnResidentMiss` runs at the existing cold-load decision. The periodic trigger is defined but fires no creation action this round. |
| **Extension friction** | ~4 touch points (policy + cluster-scoped placement config + per-instance seeding wiring + validation). Higher than a pure policy template because per-instance targeting is a cluster-topology concern absent from today's instance-agnostic config (see Decision D3). |

## 6. Invariants

Existing invariants this feature must preserve:

- **INV-6 (determinism / no-op byte-identity)** — the primary safety property. With no
  placement-policy options set, or with all seams explicitly at baseline, stdout is
  byte-identical to the pre-feature goldens for the same seed and configuration. Every seed
  policy is a deterministic function of simulation state + declared config; no wall-clock,
  no new randomness.
- **INV-9 (oracle knowledge boundary)** — routing/creation/eviction decisions read control-
  plane state (adapter identity, residency, rank, capacity) but never a request's realized
  `OutputTokens`.
- **INV-13 (run/replay parity)** — any policy selection supported by both run and replay
  produces identical per-request metrics across an export→replay round-trip.
- **Resident-set pin/capacity invariant** — `loaded ≤ capacity`; pinned (in-flight) adapters
  are never evicted, under *every* eviction policy.

New invariants introduced by this feature:

- **INV-PS1 (strict-routing honesty)** — under route-to-holder, if at least one instance
  holds the requested adapter, the served instance is a holder. When no holder exists, the
  documented fallback (Decision D1) applies; the request is never silently dropped.
- **INV-PS2 (pre-placement conservation)** — every pre-placed (adapter, instance) pair
  declared in config is either resident on that instance at t=0 or the configuration is
  rejected at startup; no pre-placement is silently dropped or truncated.
- **INV-PS3 (periodic-trigger inertness)** — a declared periodic trigger produces output
  byte-identical to a run without it, this round.

## 7. Decisions with Trade-offs

### D1 — route-to-holder is a routing *policy*, not merely a scorer

**Problem.** Part B's illustrative shape registered route-to-holder as "a new named scorer."
But the existing router composes scorers as a *weighted sum* and picks the argmax: a scorer
that scores holders 1.0 and non-holders 0.0 only *biases* toward holders — a non-holder can
still win when other scorers dominate or when weights are unfortunate. That cannot express
*strict* placement.

**Decision.** route-to-holder constrains the **candidate set** to holders (when non-empty)
*before* the existing scorer composition runs, then selects the best-scored holder using the
current weighted scoring. Strictness is a set restriction, not a score.

**Rationale.** Strictness is a hard constraint; hard constraints belong in candidate
filtering, not in a soft additive score. Reusing the existing scorer composition *among
holders* keeps tie-breaking and determinism identical to today.

**Alternatives considered.**
- *Very large scorer weight* — rejected: still probabilistic, not strict; brittle to other
  weights; violates INV-PS1 in adversarial cases.
- *A dedicated top-level routing policy parallel to `weighted`* — viable and is effectively
  what this is; the candidate-filter framing keeps the door open to composing strictness
  with the existing scorer profile rather than replacing it.

**What breaks if wrong.** If strictness were left as a scorer, INV-PS1 could not be
guaranteed and the reproduction experiment (US1) would be invalid.

### D2 — rank/cost-aware eviction requires wiring the adapter registry to the eviction site

**Problem.** Today the eviction decision has no access to adapter rank: the resident-set
entries carry only id + pin count, and the cost interface reachable at the eviction site
exposes load latency *by id* but not a rank ranking. The adapter registry (which does expose
rank) is defined but never wired into the instance simulator.

**Decision.** Introduce an **eviction context** that exposes, at the decision site, the
unpinned candidate ids together with the rank/reload-cost needed to order them — sourced from
the already-defined adapter registry, wired into the instance simulator alongside the
existing cost model. Baseline LRU ignores the context (byte-identical); rank-aware consumes
it.

**Rationale.** Extending the *narrow* cost interface with a rank accessor would change a
frozen production contract and its test doubles; wiring the existing registry interface is
additive and reuses a seam that already exists for exactly this data (R13 — no new single-
impl interface). The eviction context keeps the resident set policy-agnostic: it need not
learn about rank.

**Alternatives considered.**
- *Widen the cost interface with `RankOf`* — rejected: mutates a frozen interface + mocks for
  one consumer.
- *Store rank inside resident-set entries* — rejected: couples the LRU data structure to
  cost semantics it should not own; changes the resident-set construction contract.

**What breaks if wrong.** Choosing to store rank in resident entries would make the resident
set carry policy-specific state, breaking the clean baseline/rank-aware substitution and the
byte-identity of the LRU default.

### D3 — pre-placement is a cluster-scoped concern, not per-instance LoRA config

**Problem.** The LoRA sub-config is *instance-agnostic*: an identical copy is handed to every
instance at construction. "Assign adapter X to instance i" cannot be expressed there without
every instance seeing every other instance's assignment.

**Decision.** The adapter→instance assignment lives at **cluster/deployment scope** (beside
node pools and the routing profile) and is consulted in the per-instance construction loop,
which is the one place instance index/identity is known. Seeding happens as instances are
built.

**Rationale.** Placement is inherently a cluster-topology decision; the deployment config
already hosts sibling cluster-scoped concerns (instance count, node pools, routing profile).
Keeping it out of the per-instance sub-config avoids leaking cluster topology into instance-
local state.

**Alternatives considered.**
- *Per-instance LoRA config list* — rejected: structurally cannot target a specific instance;
  would require post-hoc filtering by index inside instance-agnostic code.
- *A construction parameter on the instance simulator* — this is the mechanism; the decision
  is *where the data originates* (deployment scope) and *who consumes it* (the construction
  loop).

**What breaks if wrong.** Putting assignment in instance-agnostic config makes per-instance
targeting impossible and silently seeds every adapter on every instance.

### D4 — pre-placed adapters are resident at t=0 with no cold-load latency and no load count

**Problem.** Should a pre-placed adapter count as a "load" in metrics, and should it incur
the cold-load latency?

**Decision.** Pre-placement seeds residency directly at t=0, before any request, incurring
**neither** cold-load latency **nor** a load-count increment. Load counts remain a measure of
*demand-driven* cold loads (the quantity the reproduction experiment compares).

**Rationale.** The value of static placement is precisely that it *avoids* demand cold loads;
counting the seeding as a load would mask that in the headline metric (SC-002: pre-placed
adapters incur zero cold loads). t=0 seeding is instantaneous by construction — there is no
request to block, so no latency is meaningful.

**Alternatives considered.**
- *Count seeds as loads* — rejected: conflates provisioning with demand churn; defeats the
  metric the experiment measures.
- *Charge a one-time provisioning latency* — deferred: no analysis question needs it this
  round (Banks criterion 5); can be added later without changing the seam.

**What breaks if wrong.** Counting seeds as loads would make SC-002 unmeasurable and the
reproduction comparison misleading.

### D5 — the periodic trigger is scaffolded but inert

**Problem.** Migration/scaling policies (future) need a periodic, request-independent trigger
that BLIS does not have. Adding it later would repaint the trigger→policy framework.

**Decision.** Define the trigger taxonomy now (reactive classes + a periodic class) and
scaffold a declared, deterministic periodic interval that fires *no* action this round.

**Rationale.** The taxonomy is cheap to define and expensive to retrofit; an inert periodic
trigger changes zero behavior (INV-PS3) while fixing the framework shape. Determinism holds
because the interval is declared config on simulation time, not wall-clock.

**Alternatives considered.**
- *Defer the taxonomy entirely* — rejected: guarantees a framework repaint when migration
  lands.
- *Implement a periodic action now* — rejected: out of scope; no seed policy needs it, and it
  would introduce behavior to validate with no consuming experiment.

**What breaks if wrong.** Implementing a periodic action now would add unvalidated behavior;
deferring the taxonomy would force a later repaint.

### D6 — provenance is a run-level effective triple, recorded in output

**Problem.** No existing output field records which policies a run used; the harness needs the
*effective* (post-bundle-expansion) triple for reproducibility.

**Decision.** Record the effective {routing, eviction, creation} selection as a run-level
field in the metrics output. Policies are run-scoped, so one triple per run suffices; no
per-request attribution.

**Rationale.** Reproducibility requires the resolved triple, not just a bundle name (a bundle
table can change). Run-level matches the granularity at which policies are selected and keeps
per-request records unchanged (INV-6 for adapter-free/baseline runs — the field is omitted or
carries the baseline triple, to be finalized so byte-identity holds; see Validation).

**Alternatives considered.**
- *Per-request policy attribution* — rejected as unnecessary (Banks criterion 6): policies are
  run-scoped; per-request records would bloat with a constant.
- *Rely on the harness's own manifest only* — rejected: the run output should be self-
  describing for reproduction from the artifact alone (SC-006).

## 8. Extension Points (registries)

Each seam becomes a **named registry** so adding a policy is a localized, additive change —
the minimal diff surface both a human and an automated search agent want (spec FR-005, SC-004).

- **Routing** already has a name→scorer catalog; route-to-holder adds a strict policy behind
  the existing router selection point.
- **Eviction** converts the single hardcoded LRU choice into a named policy with LRU as the
  registered default and rank-aware as the second entry.
- **Creation** introduces a name→policy catalog with on-demand as the default and pre-placement
  as the second entry, seated on the `Initial` and `OnResidentMiss` triggers.

**Touch-point targets** (design-guidelines §4.5 reference: policy template ≈ 3 files):

| Seam | Add one more policy | Notes |
|---|---|---|
| Routing (biasing scorer) | ~2 files | Meets/beats target (mature seam). |
| Routing (strict/candidate-constraining policy) | ~3 files | Meets target. |
| Eviction | ~3–4 files | Exceeds target by ~1 due to eviction-context/rank plumbing (D2); justified in §10. |
| Creation | ~4 files | Exceeds target by ~1 due to cluster-scoped placement config (D3); justified in §10. |

## 9. Trigger Taxonomy

**Class 1 — reactive / endogenous (implemented this round):**

| Trigger | Fires when | Policy invoked |
|---|---|---|
| `Initial` (t=0) | simulator build, before any request | Creation → seed residency |
| `OnRoute` | request routing | Routing → select instance |
| `OnResidentMiss` | routed request needs an absent adapter | Creation → admit-on-miss |
| `OnCapacityPressure` | slot needed and resident set full | Eviction → pick victim |

**Class 2 — proactive / scheduled (designed now, inert this round):**

| Trigger | Fires when | Unlocks (future) |
|---|---|---|
| `Periodic(Δt)` | every Δt of simulation time, request-independent | proactive eviction, prefetch, migration, scaling |

The periodic trigger, if declared, is a self-rescheduling clock event with a declared interval.
This round it invokes no policy (INV-PS3). Because Δt is declared config on simulation time,
determinism holds.

**Event classification:** no new *reactive* event type is introduced — reactive triggers reuse
existing decision points (routing, cold-load gate, capacity pressure). The periodic trigger, if
and when it carries an action, is an **endogenous, state-driven** event and will require a
priority constant assigned at that time; this round its scaffolding fires nothing and needs no
new ordering guarantee.

## 10. Complexity Tracking

This feature is a **Subsystem Module** (design-guidelines §5.3) and therefore exceeds the ≤3-file
policy-template target for two of three seams. Justification:

| Over-target area | Extra cost | Why justified |
|---|---|---|
| Eviction rank plumbing (D2) | +1 file (eviction context + registry wiring) | Rank is genuinely unreachable at the eviction site today; the alternative (mutating the frozen cost interface) is worse. One-time cost; every future eviction policy reuses the context. |
| Cluster-scoped pre-placement (D3) | +1 file (deployment-config field + construction-loop wiring) | Per-instance targeting is a cluster-topology concern with no home in instance-agnostic config; placing it correctly now avoids a later re-layering. |

Both are one-time structural costs paid once per seam, not per policy. After they land, adding a
further eviction or creation policy returns to the ~3-file target.

## 11. Validation Strategy

**Verification (correctness — which invariants):**
- **No-op golden (INV-6, SC-001):** the existing adapter-blind byte-identity test must pass
  unchanged; a companion test asserts byte-identity when every seam is *explicitly* set to
  baseline. This is the first test written for each PR (TDD).
- **Strict routing (INV-PS1):** behavioral GIVEN/WHEN/THEN — given a holder exists, a request is
  never routed to a non-holder; given no holder, the documented fallback (D1) applies.
- **Eviction pin-safety + determinism (resident-set invariant, INV-6):** given a full set with
  one unpinned candidate, only that candidate can be evicted under any policy; given several
  unpinned candidates of differing rank, the rank-aware policy picks the declared victim with a
  deterministic tie-break.
- **Pre-placement conservation (INV-PS2):** given a valid assignment, adapters are resident at
  t=0 with zero load-count; given an over-capacity or unregistered assignment, startup fails with
  a clear error.
- **Periodic inertness (INV-PS3):** a declared periodic trigger yields byte-identical output.
- **Run/replay parity (INV-13):** export→replay with identical policy selection yields identical
  per-request metrics.
- **Determinism:** same seed + same policy selection → identical routes, victims, seedings across
  two runs.

**Validation (fidelity — against what data):**
- **Reproduction experiment (SC-002, in `lora-control`):** static pre-placement + route-to-holder
  vs the landed `llmd-affinity-baseline` on the same scenarios/metrics. This is the fidelity
  check that the seams express the intended published policy behavior. Absolute-fidelity claims
  against the Digital Twin remain bounded as in the prior LoRA work (base-mismatch caveats
  documented there); this feature's validation target is *policy behavior*, not new latency
  physics.

**Randomness:** no new `PartitionedRNG` subsystem. All seed policies are deterministic; any tie-
break is by a stable key (id). If a *future* policy needs randomized tie-breaking it will add a
named subsystem then — flagged, not pre-built.

## 12. DES Design Review Checklist

| Question | Answer |
|---|---|
| What analysis questions does this design help answer? | How do static-placement and cost-aware policies compare to emergent placement on adapter churn, cold-load rate, and TTFT/throughput? |
| What is modeled, simplified, omitted? | See §4 table. |
| What events are introduced/modified? Exogenous/endogenous? | No new reactive event (reuses existing decision points). Periodic trigger is a scaffolded endogenous event, inert this round. |
| How do new events interact with tie-breaking? | Periodic trigger fires no action this round → no ordering guarantee needed yet; a priority constant is assigned if/when it carries an action. |
| What new state is introduced? Who owns it? | Cluster-scoped adapter→instance assignment (deployment config); seeded residency (resident set, already owned). No new per-request state. |
| What new metrics are derived? | Run-level effective policy triple (provenance). Per-adapter load/eviction counts already exist. |
| How will correctness be verified? | §11 verification: no-op golden + per-seam behavioral + determinism + parity, INV-6/9/13/PS1/PS2/PS3. |
| How will fidelity be validated? | §11 validation: reproduction experiment vs `llmd-affinity-baseline` in `lora-control`. |
| New randomness? Which subsystem? | None this round (all seed policies deterministic). |
| Simplest version that answers the same questions? | Baseline defaults byte-identical; each seam ships exactly one real policy beyond baseline; periodic trigger inert. |

## 13. Suggested PR Roadmap (Small-tier, each no-op-safe)

Refines Part B §B7 with the friction findings. Each PR keeps the no-op default byte-identical.

| PR | Scope | Notes |
|---|---|---|
| **B-1** | Routing: registry framing + register existing scorers/policies as defaults (byte-identical). | Mature seam; smallest PR. |
| **B-2** | Routing: add **route-to-holder** strict policy (candidate-set restriction, D1). | Policy template + selection wiring. |
| **B-3** | Eviction: extract **EvictionPolicy** seam; register **lru** default (byte-identical); introduce the eviction context. | Includes the D2 registry-to-eviction-site wiring; LRU ignores context. |
| **B-4** | Eviction: add **rank/cost-aware** policy (consumes eviction context). | Policy template. |
| **B-5** | Creation: **CreationPolicy** seam with **on-demand** default; seat `Initial` + `OnResidentMiss`; cluster-scoped placement config (D3). | Includes the structural placement-config plumbing. |
| **B-6** | Creation: **pre-placement** policy (t=0 seeding, D4). | Policy template + startup validation (INV-PS2). |
| **B-7** | Trigger-taxonomy scaffolding incl. the periodic trigger type (inert, INV-PS3) + strategy bundles + provenance (D6) + docs. | No behavior change beyond bundle resolution + output field. |

**Grouping:** one tracking/epic issue ("LoRA placement-policy seams") with B-1…B-7 as sub-issues,
generated by `/speckit.taskstoissues` from `tasks.md`.

## 14. Open Items to Resolve at Micro-Plan Time

- Finalize the eviction-context contents against the real eviction call site (rank source,
  tie-break key).
- Confirm the route-to-holder fallback (D1) against the source policy (`Tantawi2025`) and the
  rank-aware victim criterion (D4/§4) against `Li2025` (Toppings).
- Confirm whether the provenance field is omitted for baseline runs or carries an explicit
  baseline triple, such that INV-6 byte-identity holds either way.
- Confirm the deployment-config shape for the adapter→instance assignment and its startup
  validation messages.
