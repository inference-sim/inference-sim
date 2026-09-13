# Feature Specification: LoRA Placement-Policy Seams

**Feature Branch**: `008-lora-policy-seams`
**Created**: 2026-07-22
**Status**: Draft
**Input**: User description: "LoRA placement-policy seams — three pluggable seams (Routing, Eviction, Creation) with byte-identical no-op/baseline defaults, a (trigger→policy) taxonomy (reactive triggers now; a periodic trigger designed and scaffolded but unimplemented), and named strategy bundles. Seed policies: route-to-holder, pre-placement, rank/cost-aware evictor. Part B of the LoRA policy-seams design, building on the merged LoRA control-plane subsystem."

## Overview

BLIS today models LoRA adapter placement as purely *emergent*: instances start
empty, adapters become resident on demand via a cold-load gate, and the existing
`lora-affinity` router only *biases* routing toward warm instances — it never
forbids. There is no way to pre-assign adapters to instances, to route strictly
to holders, or to choose eviction victims by anything other than least-recently-used
order. As a result, published static-placement policies cannot be expressed or
reproduced, and there is no orthogonal surface on which to run placement-policy
experiments or automated policy search.

This feature makes adapter **placement policy** a first-class, selectable concern
along three independent knobs — **Routing** (which instance serves a request),
**Eviction** (which resident adapter is evicted under capacity pressure), and
**Creation** (when/where an adapter first becomes resident) — each defaulting to
today's exact behavior so that, unless a researcher opts in, nothing changes.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reproduce a static placement policy end-to-end (Priority: P1)

A capacity-planning researcher wants to reproduce a published static-placement
result: adapters are assigned to specific instances ahead of time, and every
request for an adapter is served only by an instance that holds it. The researcher
selects a **pre-placement** creation policy (seeding a declared adapter→instance
assignment before any request arrives) together with a **route-to-holder** routing
policy (requests go only to holders), runs a scenario, and reads back per-adapter
metrics.

**Why this priority**: This is the headline capability the whole feature exists to
unlock — the reproduction experiment that the harness layer (already landed) can
name but cannot currently run. It exercises two of the three seams and is the
single most valuable slice.

**Independent Test**: Configure a small cluster with a declared adapter→instance
assignment, select pre-placement + route-to-holder, and confirm that (a) the
assigned adapters are resident on their instances before the first request, (b)
every served request lands on a holder, and (c) cold-load counts for pre-placed
adapters are zero. Delivers a runnable reproduction scenario on its own.

**Acceptance Scenarios**:

1. **Given** a cluster with adapters pre-assigned to instances and pre-placement
   creation selected, **When** the simulation starts, **Then** each pre-assigned
   adapter is resident on its assigned instance(s) at time zero without incurring a
   cold-load charge.
2. **Given** route-to-holder routing and an adapter resident on exactly one
   instance, **When** a request for that adapter arrives, **Then** the request is
   dispatched to a holding instance and never to a non-holder.
3. **Given** route-to-holder routing and an adapter that no instance currently
   holds, **When** a request for that adapter arrives, **Then** the request is
   handled by a clearly defined, documented fallback (see FR-014) rather than
   silently dropped or mis-served.

---

### User Story 2 - Swap the eviction policy independently (Priority: P2)

A researcher studying adapter churn under skewed popularity wants to compare
eviction strategies without touching routing or creation. They run the same
workload twice — once with the default least-recently-used eviction, once with a
**rank/cost-aware** evictor that prefers to evict cheaper-to-reload adapters — and
compare load/eviction counts.

**Why this priority**: Proves the seams are genuinely orthogonal (one knob changes,
the others hold), and delivers the first ablation experiment. Independent of US1.

**Independent Test**: With a resident set held at capacity and a mix of adapters of
different ranks/reload costs, force an eviction and confirm the default picks the
least-recently-used unpinned adapter while the rank/cost-aware policy picks the
declared victim by its cost criterion — and that neither ever evicts a pinned
(in-flight) adapter.

**Acceptance Scenarios**:

1. **Given** a full resident set with the default eviction policy, **When** a new
   adapter must be admitted, **Then** the least-recently-used unpinned adapter is
   evicted (identical to today's behavior).
2. **Given** a full resident set with the rank/cost-aware eviction policy and
   several unpinned candidates of differing rank/reload cost, **When** a new adapter
   must be admitted, **Then** the victim chosen is the one the policy's cost
   criterion designates, and the choice is deterministic across repeated runs.
3. **Given** any eviction policy and a resident set where all-but-one candidates are
   pinned, **When** an eviction is forced, **Then** only the single unpinned
   candidate can be chosen and no pinned adapter is ever evicted.

---

### User Story 3 - Select a whole policy by one bundle name (Priority: P3)

A researcher reproducing a paper wants to select a single named **bundle** that
stands for a coherent triple of {routing, eviction, creation} policies, rather than
wiring each knob by hand. Selecting the bundle name expands to the three underlying
policies; leaving an individual knob explicitly set overrides just that knob; the
run's provenance records the *effective* triple actually used.

**Why this priority**: Convenience and reproducibility layer on top of US1/US2. It
adds no new mechanism — it names existing ones — so it is lower priority than the
seams themselves but valuable for faithful, auditable reproduction.

**Independent Test**: Select a known bundle name and confirm it resolves to the
expected three policies; override one knob and confirm only that knob changes;
confirm the run output records the resolved effective triple, not just the bundle
name.

**Acceptance Scenarios**:

1. **Given** a defined bundle name, **When** it is selected with no per-knob
   overrides, **Then** the run uses exactly the three policies the bundle names.
2. **Given** a bundle selected together with one explicit per-knob override, **When**
   the run starts, **Then** the overridden knob uses the explicit value and the
   other two use the bundle's values.
3. **Given** any policy selection (bundle or explicit triple), **When** the run
   completes, **Then** the effective (post-expansion) triple is recorded in the
   run's output so the result is reproducible from the record alone.

---

### User Story 4 - Preserve today's behavior by default (Priority: P1)

Every researcher who does *not* opt into placement policies — and every existing
BLIS user, experiment, and regression golden — must see byte-identical output. The
default routing, eviction, and creation behavior after this feature must equal the
behavior before it.

**Why this priority**: This is the non-negotiable safety property (INV-6). It is a
precondition for merging any part of the feature and is the first thing tested. It
shares P1 with US1 because neither is acceptable without the other.

**Independent Test**: Run the existing regression scenarios with no placement-policy
options set and confirm the output is byte-identical to the pre-feature goldens; run
again with every seam explicitly set to its baseline default and confirm the same.

**Acceptance Scenarios**:

1. **Given** a scenario with no placement-policy options configured, **When** it is
   run, **Then** its output is byte-identical to the pre-feature result for the same
   seed and configuration.
2. **Given** every seam explicitly set to its baseline (default routing, default
   eviction, on-demand creation), **When** the scenario is run, **Then** the output
   is byte-identical to the no-options run.
3. **Given** a scenario that configures no LoRA adapters at all, **When** it is run,
   **Then** placement policy has no observable effect on latency, memory, or routing.

---

### User Story 5 - Design-in a periodic trigger for future policies (Priority: P4)

A researcher planning future migration/scaling policies needs the trigger taxonomy
to already distinguish *reactive* triggers (fired by request events) from a
*periodic* trigger (fired on a declared simulation-time interval), so that later
proactive policies attach without repainting the framework. In this feature the
periodic trigger is defined and scaffolded but performs no action.

**Why this priority**: Pure forward-compatibility scaffolding with no behavior
change this round. Lowest priority; included so the framework is not reworked when
migration/scaling policies arrive.

**Independent Test**: Confirm the trigger taxonomy names both reactive and periodic
trigger classes, that a periodic trigger can be declared, and that declaring it
produces no change to simulation output (it is inert this round).

**Acceptance Scenarios**:

1. **Given** the trigger taxonomy, **When** it is inspected, **Then** it distinguishes
   reactive triggers (start-of-run, on-route, on-resident-miss, on-capacity-pressure)
   from a periodic (interval-driven) trigger.
2. **Given** a periodic trigger declared with an interval, **When** the simulation
   runs, **Then** output is byte-identical to a run without it (the trigger is inert
   this round).

---

### Edge Cases

- **Route-to-holder with no holder**: an adapter that no instance currently holds
  receives a request. The fallback behavior must be explicit and documented
  (FR-014), not silent.
- **Pre-placement over capacity**: a declared adapter→instance assignment names more
  adapters for an instance than its resident-set capacity allows. This must be
  rejected at startup with a clear error rather than silently truncating.
- **Pre-placement of an unregistered adapter**: an assignment names an adapter id
  not present in the adapter registry. Must be rejected at startup.
- **Eviction with every candidate pinned**: a slot is needed but all resident
  adapters are pinned (in-flight). The resident-set invariant (never evict a pinned
  adapter) must hold under *every* eviction policy, including the new one.
- **Bundle names an unknown policy** or a knob names a policy that is not registered:
  must be rejected at startup with a clear error listing valid names.
- **Determinism under a new policy**: the same seed plus the same policy selection
  must produce identical routes, victims, and seedings across repeated runs.
- **Cross-command parity**: a scenario exported and replayed with identical policy
  selection must produce identical per-request metrics (INV-13).

## Requirements *(mandatory)*

### Functional Requirements

#### Seam selection & defaults

- **FR-001**: The system MUST let a researcher select a **routing policy** by name,
  an **eviction policy** by name, and a **creation policy** by name, independently of
  one another.
- **FR-002**: Each of the three seams MUST have a **baseline default** whose behavior
  equals today's behavior: default routing = the existing scorer-based routing,
  default eviction = least-recently-used, default creation = on-demand cold-load with
  an empty initial resident set.
- **FR-003**: When no placement-policy option is set, the system MUST behave exactly
  as it did before this feature (INV-6): byte-identical output for the same seed and
  configuration.
- **FR-004**: Selecting a policy that is not registered MUST fail at startup with a
  clear error that lists the valid policy names for that seam.
- **FR-005**: Adding one more policy to any seam MUST be possible as a localized,
  additive change (one new self-contained policy plus its registration), without
  editing the other policies — so both a human and an automated search agent can drop
  in a policy with a minimal, mechanical diff.

#### Routing seam

- **FR-006**: The system MUST provide a **route-to-holder** routing policy that
  dispatches a request only to an instance that currently holds the requested
  adapter, never to a non-holder (when at least one holder exists).
- **FR-007**: The existing routing behavior (including the `lora-affinity` scorer,
  which biases but never forbids) MUST remain available and unchanged as the default.

#### Eviction seam

- **FR-008**: The system MUST provide a **rank/cost-aware** eviction policy that
  chooses the victim among unpinned candidates by a declared rank/reload-cost
  criterion.
- **FR-009**: No eviction policy — default or new — may ever choose a **pinned**
  (in-flight) adapter as a victim; the resident-set capacity/pin invariant
  (`loaded ≤ capacity`, pinned protected) MUST hold under every policy.

#### Creation seam

- **FR-010**: The system MUST provide a **pre-placement** creation policy that seeds a
  declared adapter→instance assignment as resident at time zero, before any request,
  without incurring a cold-load charge for the seeded adapters.
- **FR-011**: The default **on-demand** creation policy MUST start every instance with
  an empty resident set and admit adapters via the existing cold-load gate (today's
  behavior).
- **FR-012**: A pre-placement assignment that exceeds an instance's resident-set
  capacity, or that names an adapter absent from the adapter registry, MUST be
  rejected at startup with a clear error.

#### Triggers & composition

- **FR-013**: The system MUST bind policies to **triggers** as a first-class pairing,
  distinguishing reactive triggers (start-of-run, on-route, on-resident-miss,
  on-capacity-pressure) from a **periodic** (interval-driven) trigger. The periodic
  trigger MUST be defined and selectable but inert this round (no behavior change).
- **FR-014**: The route-to-holder policy's behavior when **no instance holds** the
  requested adapter MUST be explicit and documented (the assumed default is to fall
  back to on-demand admission at the best-scored instance; see Assumptions).
- **FR-015**: The system MUST let a researcher select a **named bundle** that expands
  to a {routing, eviction, creation} triple; an explicitly set per-knob value MUST
  override the bundle's value for that knob only; unset knobs MUST fall back to the
  baseline default.
- **FR-016**: The run output MUST record the **effective (post-expansion) triple** of
  policies actually used, so any result is reproducible from its record alone.

#### Correctness carriers

- **FR-017**: Every trigger and policy MUST be a deterministic function of simulation
  state and declared configuration only — no wall-clock time, and any randomness only
  via a partitioned RNG subsystem — so the same seed plus the same policy selection
  yields identical decisions across runs (INV-6 determinism).
- **FR-018**: For any policy selection supported by both direct simulation and
  trace-replay, a scenario exported and replayed with identical selection MUST produce
  identical per-request metrics (INV-13 run/replay parity).

### Key Entities *(include if feature involves data)*

- **Routing policy**: a named strategy that, given a request and the current per-
  instance state, decides which instance serves the request. Baseline = existing
  scorers; new = route-to-holder.
- **Eviction policy**: a named strategy that, given the unpinned resident adapters and
  eviction context, selects the victim when a slot is needed. Baseline = LRU; new =
  rank/cost-aware.
- **Creation policy**: a named strategy that decides which adapters are resident at
  start-of-run and whether to admit an adapter on a miss. Baseline = on-demand
  (empty seed, always admit); new = pre-placement (declared seeding at t=0).
- **Trigger**: a simulation event that invokes a policy — reactive (start-of-run,
  on-route, on-resident-miss, on-capacity-pressure) or periodic (interval-driven,
  inert this round).
- **Strategy bundle**: a named configuration binding a {routing, eviction, creation}
  triple; resolves to three policy selections with per-knob override.
- **Adapter→instance assignment**: the declared placement input consumed by the
  pre-placement creation policy.
- **Effective triple (provenance)**: the resolved {routing, eviction, creation}
  selection recorded in run output for reproducibility.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With no placement-policy options set (and with all seams explicitly at
  baseline), simulation output is byte-identical to the pre-feature result for the
  same seed and configuration — verified across the existing regression goldens.
- **SC-002**: A researcher can reproduce a static-placement scenario (pre-placement +
  route-to-holder) in which 100% of requests for pre-placed adapters are served by a
  holding instance and pre-placed adapters incur zero cold loads.
- **SC-003**: Under skewed adapter popularity, swapping only the eviction policy
  (LRU → rank/cost-aware) measurably changes adapter load/eviction counts while
  routing and creation behavior are unchanged — demonstrating knob orthogonality.
- **SC-004**: Adding one more policy to any seam requires changing only that new
  policy's self-contained definition plus its one-line registration — no edits to
  existing policies — confirmed by inspection of the change set for each seed policy.
- **SC-005**: The same seed plus the same policy selection produces identical routes,
  victims, and seedings across two independent runs (determinism), and an
  export→replay round-trip with identical selection produces identical per-request
  metrics (parity).
- **SC-006**: Every run records the effective policy triple in its output, and a
  reviewer can reconstruct the exact policy configuration from the record without the
  original command line.

## Assumptions

- **Route-to-holder fallback (FR-014)**: when no instance holds the requested adapter,
  the assumed default is to admit on-demand at the best-scored instance (turning the
  first request for an adapter into a normal cold-load), rather than dropping the
  request or forcing a specific instance. This preserves liveness and matches the
  "biases toward, then admits" spirit of existing routing. To be confirmed in the
  design doc against the source policy (`Tantawi2025`).
- **Rank/cost-aware criterion (FR-008)**: the victim cost criterion reuses the
  existing adapter cost model (rank → reload cost) already present in the LoRA
  subsystem; the exact tie-break and cost formula are finalized in the design doc
  against the source policy (`Li2025`, Toppings).
- **Pre-placement input shape (FR-010)**: the adapter→instance assignment is a
  per-instance-index declaration in LoRA configuration; instances beyond the declared
  set start empty (on-demand).
- **Periodic trigger scope (FR-013)**: `Periodic` is a declared, deterministic
  interval on simulation time; it is scaffolded (defined, selectable, documented) but
  performs no action this round. Migration/scaling policies that consume it are out of
  scope.
- **Interface economy (R13)**: no seam interface is introduced without at least two
  implementations (baseline + ≥1 real policy), and existing types are extended rather
  than minting single-implementation interfaces.

## Out of Scope

- Migration and scaling *policies* (the periodic trigger is scaffolded but no policy
  consumes it this round).
- A runtime policy language / DSL, or any out-of-process policy server — policies are
  selected by name from a fixed, compiled catalog.
- Adapter training, rank-accuracy tradeoffs, or dynamic (runtime-negotiated)
  adapter↔KV memory tradeoffs (HBM accounting stays static, as landed).
- The harness/experiment side (PolicySpec, bundle tables as research data, provenance
  manifest) — that is Part A and has already landed in the `lora-control` repo. This
  feature is the BLIS-side seams that Part A's names resolve to.

## Dependencies

- Builds directly on the merged LoRA control-plane subsystem (adapter identity &
  registry, per-instance resident set with LRU + pin/capacity, cold-load creation
  gate, adapter cost model, HBM accounting, `lora-affinity` scorer, per-adapter
  metrics). This feature branches from the `lora-integration` line where that
  subsystem is assembled.
- Consumed by the `lora-control` harness, which emits policy selections as
  configuration and reads per-adapter metrics back.
