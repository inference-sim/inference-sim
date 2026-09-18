# Feature Specification: Phase 1B — Service Tiers & Tenant Fairness

**Feature Branch**: `002-tier-tenant-fairness`
**Created**: 2026-03-23
**Status**: Draft
**Source**: [Discussion #402 comment v4 (T-1, T-2)](https://github.com/inference-sim/inference-sim/discussions/402#discussioncomment-15901661), tracking issue [#696](https://github.com/inference-sim/inference-sim/issues/696), sub-issues [#691](https://github.com/inference-sim/inference-sim/issues/691) and [#739](https://github.com/inference-sim/inference-sim/issues/739)

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Tier-Ordered Load Shedding Under Overload (Priority: P1)

A cluster operator runs a multi-tenant LLM simulation under heavy load. The cluster
is oversubscribed — more requests arrive than can be served. The operator expects that
lower-priority requests (Background, Batch, Sheddable) are rejected first, while
higher-priority requests (Standard, Critical) continue to be served. This matches how
the Gateway API Inference Extension and OpenAI service tiers describe priority ordering.

**Why this priority**: The 5 SLO tiers already exist on every request (Critical, Standard,
Sheddable, Batch, Background) but nothing in the admission path reads them. Without this
story, tier labels are cosmetic — they carry no operational meaning. This is the minimum
viable piece of Phase 1B.

**Independent Test**: A simulation where a constant-rate workload of mixed-tier requests
(equal volumes of each tier) is sent to a cluster sized for 40% of the load. The output
shows that Background requests are shed first and Critical requests are shed last, with
monotonically increasing survival rates from Background → Critical.

**Acceptance Scenarios**:

1. **Given** a cluster at capacity receiving a mix of Critical and Sheddable requests,
   **When** admission runs,
   **Then** Sheddable requests are rejected before any Critical request is rejected.

2. **Given** a cluster at capacity receiving Background and Batch requests,
   **When** admission runs,
   **Then** Background requests are rejected before any Batch request is rejected.

3. **Given** a cluster fully saturated with Standard requests already queued,
   **When** a new Critical request arrives,
   **Then** the Critical request is admitted (displacing or blocking a Standard request
   if necessary to maintain tier ordering).

4. **Given** a simulation with all five tiers present and load ramping from 50% to 200% of capacity,
   **When** the simulation completes,
   **Then** shed counts show: `Background ≥ Batch ≥ Sheddable ≥ Standard ≥ Critical` — the
   shedding order is monotonic throughout the ramp.

5. **Given** a shedding decision being made for any tier,
   **When** the decision logic runs,
   **Then** no predicted or actual output token count is consulted — only arrival-time
   metadata (tier, prompt token count, arrival timestamp) influences the decision.

---

### User Story 2 — Deferred Processing for Batch and Background Requests (Priority: P1)

An operator submits batch evaluation jobs and background research tasks alongside
real-time chat traffic. Batch and Background requests should not be rejected outright —
instead they should be parked in a deferred queue and executed only when the cluster
has spare capacity, much like how cloud providers handle spot/batch workloads.

**Why this priority**: Without a deferred queue, Batch and Background tiers are simply
shed like Sheddable, which loses their semantic meaning (they should complete eventually,
just not at the expense of synchronous traffic). This is the other half of T-1.

**Independent Test**: A simulation with a steady stream of Batch requests alongside a
real-time workload sized to consume 95% of capacity. After the real-time workload ends,
deferred Batch requests are picked up and completed. Batch requests never run concurrently
with real-time requests that are still queued.

**Acceptance Scenarios**:

1. **Given** a cluster serving real-time (Critical/Standard) traffic at 90% utilization
   and incoming Batch requests,
   **When** admission processes Batch requests,
   **Then** Batch requests enter a deferred queue rather than being rejected or queued
   alongside real-time traffic.

2. **Given** deferred Batch requests in the deferred queue and the real-time
   synchronous queue becoming empty,
   **When** the next scheduling step runs,
   **Then** deferred requests are promoted to the active queue and begin execution.

3. **Given** a deferred Background request waiting,
   **When** a new real-time request arrives and the cluster returns to high utilization,
   **Then** the deferred Background request is not promoted — it remains in the deferred
   queue until capacity is idle again.

4. **Given** the simulation horizon is reached with requests still in the deferred queue,
   **When** the simulation ends,
   **Then** those requests are accounted for in the metrics as "horizon-interrupted" (not
   dropped), and INV-1 (request conservation) holds.

---

### User Story 3 — Per-Tenant Fair-Share Budget Enforcement (Priority: P2)

A platform operator configures a multi-tenant cluster where Tenant A is allowed 40% of
capacity and Tenant B is allowed 60%. When Tenant A over-submits and exceeds their budget,
their lower-priority requests are preferentially shed while Tenant B's requests are
unaffected. This mirrors how cloud providers enforce tenant resource quotas.

**Why this priority**: Without fairness enforcement, a noisy tenant can crowd out others
even at equal SLO tier. This completes Phase 1B by adding the tenant dimension orthogonal
to tier priority. Depends on Story 1 being in place (tier-ordered shedding must exist
before per-tenant bias can be layered on top).

**Independent Test**: Two-tenant simulation: Tenant A (budget: 30%) and Tenant B (budget: 70%).
Both send Standard requests at equal rate. Cluster is sized for 80% of combined load.
Output shows Tenant A's requests shed at approximately twice the rate of Tenant B's until
Tenant A returns under budget.

**Acceptance Scenarios**:

1. **Given** two tenants with configured fair-share budgets and both under budget,
   **When** a mixed-tenant workload runs under moderate load,
   **Then** both tenants' requests are admitted at equal rate regardless of tenant identity.

2. **Given** Tenant A has consumed more than their fair-share of active request slots,
   **When** a new Sheddable request from Tenant A arrives alongside a Sheddable request
   from Tenant B (under budget),
   **Then** Tenant A's request is shed preferentially over Tenant B's.

3. **Given** a deployment config with no per-tenant budgets specified,
   **When** the simulation runs with a single tenant or with TenantID unset on all requests,
   **Then** the simulation produces byte-identical results to a run without tenant tracking
   configured (zero-value safe, INV-6 determinism preserved).

4. **Given** Tenant A is over-budget,
   **When** a Critical-tier request from Tenant A arrives,
   **Then** the Critical request is still admitted — budget enforcement does not override
   Critical-tier protection; only lower tiers are affected by over-budget status.

---

### User Story 4 — Per-Tenant Fairness Metrics in Simulation Output (Priority: P3)

An operator inspecting a simulation run wants to understand how equitably resources were
distributed across tenants. The metrics output includes a per-tenant breakdown of requests
served, tokens consumed, and a Jain fairness index so the operator can diagnose fairness
problems without re-running the simulation.

**Why this priority**: Observability is required for the Phase 1D hypothesis experiments
(H-TenantFairness). Without it, tenant fairness can only be inferred indirectly. Lower
priority because the enforcement logic (Story 3) must exist first, but metrics are
essential before declaring Phase 1B complete.

**Independent Test**: A two-tenant simulation with unequal load generates a metrics JSON
with a `per_tenant` section containing request counts, token totals, and a Jain fairness
index between 0.5 and 1.0 consistent with the actual distribution.

**Acceptance Scenarios**:

1. **Given** a simulation with two or more distinct TenantIDs,
   **When** the simulation completes,
   **Then** the metrics JSON output includes a `per_tenant` section with per-tenant
   request counts, total tokens served, and a Jain fairness index.

2. **Given** a perfectly balanced two-tenant workload,
   **When** the simulation completes,
   **Then** the reported Jain fairness index is ≥ 0.99 (near-perfect fairness).

3. **Given** a highly skewed workload where one tenant receives 10× more service than another,
   **When** the simulation completes,
   **Then** the Jain fairness index is < 0.70, correctly signaling unfair distribution.

4. **Given** a simulation with no TenantID set on any request,
   **When** the simulation completes,
   **Then** the `per_tenant` section is absent from the metrics JSON (no spurious empty section).

---

### Edge Cases

- What happens when a request has no tier set? It defaults to Standard tier — no admission
  path change required (existing behavior preserved).
- What happens when a request has no TenantID set? It is treated as belonging to a
  special "untenanted" group that is never subject to fair-share shedding.
- What happens when the deferred queue grows without bound (e.g., real-time load never drops)?
  Deferred requests are ultimately horizon-interrupted at simulation end; they are never
  silently dropped. INV-1 conservation holds.
- What happens when all tenants are over-budget simultaneously? Shedding falls back to
  tier-only ordering; no request receives preferential treatment based on tenant identity.
- What happens when a single tenant's budget is set to 100%? Effectively disables
  per-tenant enforcement for that tenant; other tenants with lower budgets are still
  tracked.
- What happens with Background-tier requests that are also from an over-budget tenant?
  They enter the deferred queue first (tier rule); tenant budget enforcement only applies
  within synchronous admission, not to the deferred queue.

---

## Requirements *(mandatory)*

### Functional Requirements

**T-1: Tier-Ordered Load Shedding**

- **FR-001**: Under overload, the admission system MUST shed requests in tier order:
  Background shed first, then Batch, then Sheddable, then Standard; Critical requests
  MUST be shed last.
- **FR-002**: Batch and Background requests MUST be routed to a deferred queue rather
  than rejected outright; they MUST be promoted to active processing only when the
  synchronous request queue is idle.
- **FR-003**: Tier-based shedding decisions MUST use only request arrival-time metadata
  (tier, prompt token count, arrival timestamp, tenant ID) — predicted or actual output
  token counts MUST NOT be consulted (INV-9: oracle knowledge boundary).
- **FR-004**: Simulation metrics MUST report per-tier counts for: admitted, shed, deferred,
  and horizon-interrupted requests.
- **FR-005**: The 5 canonical tier names (Critical, Standard, Sheddable, Batch, Background)
  MUST be the sole vocabulary used in code and config; legacy tier names (`realtime`,
  `interactive`) MUST NOT be introduced.
- **FR-006**: Request conservation (INV-1) MUST hold at simulation end with the deferred
  queue included in the accounting: `injected == completed + queued + running + shed + deferred_horizon_interrupted`.

**T-2: Per-Tenant Fair-Share**

- **FR-007**: The simulator MUST track per-tenant state: count of active (in-flight) requests
  and total tokens consumed in the current observation window.
- **FR-008**: Operators MUST be able to configure per-tenant fair-share budgets (as a
  fraction of total capacity) in the deployment config; tenants without explicit budgets
  share remaining capacity equally.
- **FR-009**: When a tenant exceeds their fair-share budget, their Sheddable, Batch, and
  Background requests MUST be shed preferentially over same-tier requests from tenants
  within budget; Critical and Standard requests from over-budget tenants MUST NOT be
  shed solely due to budget status.
- **FR-010**: Simulation metrics output MUST include a `per_tenant` section containing
  request counts, tokens served, and a Jain fairness index for each tenant.
- **FR-011**: A simulation with no TenantIDs set on any request MUST produce results
  identical to the same simulation prior to Phase 1B (zero-value safe); no `per_tenant`
  section appears in the output.
- **FR-012**: Tenant accounting MUST reset or decay between observation windows so a
  burst from one tenant does not permanently penalise that tenant for the rest of the
  simulation.

### Key Entities

- **SLO Tier**: One of five canonical priority levels (Critical, Standard, Sheddable,
  Batch, Background) assigned per request at arrival time. Determines shedding order
  and deferred-queue routing. Immutable after request creation.
- **Deferred Queue**: A separate queue holding Batch and Background requests awaiting
  idle-capacity windows. Distinct from the synchronous WaitQueue. Requests move from
  deferred to active only when synchronous queues drain.
- **Tenant State**: Per-tenant accounting record holding in-flight request count and
  tokens consumed in the current window. Used by admission to detect over-budget status.
  Resets on a configurable window boundary.
- **Fair-Share Budget**: Operator-configured fraction of cluster capacity reserved for
  a tenant. Stored in deployment config alongside per-tenant ID. Zero value = unlimited
  (no enforcement).
- **Fairness Index**: Jain's fairness index computed over per-tenant tokens-served
  distribution at simulation end. Ranges 0 (maximally unfair) to 1 (perfectly fair).

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Under a load ramp from 50% to 200% of cluster capacity with all five tiers
  present, shedding counts satisfy `Background ≥ Batch ≥ Sheddable ≥ Standard ≥ Critical`
  at every measurement point — monotonic tier ordering holds in 100% of overload scenarios.

- **SC-002**: Batch and Background requests are never dispatched to the execution engine
  while unserved requests exist in the synchronous WaitQueue; deferred requests begin
  executing within one scheduling step of the synchronous queue becoming empty.

- **SC-003**: When a tenant is 50% over their fair-share budget, their Sheddable requests
  are shed at a rate ≥ 1.5× a same-tier on-budget tenant's Sheddable requests, with no
  statistically significant impact on the on-budget tenant's admission rate.

- **SC-004**: The Jain fairness index reported in simulation output is within 2% of the
  value computed directly from the per-tenant token totals in the same output.

- **SC-005**: All existing simulation tests (pre-Phase-1B) pass unchanged with zero-value
  tier and tenant configuration — no regressions in single-tenant or tier-unset workloads.

- **SC-006**: Request conservation (INV-1) holds at simulation end in all four scenarios:
  (a) tier-only shedding, (b) deferred queue active, (c) tenant fairness active,
  (d) all three combined.

---

## Assumptions

- The 5 SLO tier labels (Critical, Standard, Sheddable, Batch, Background) are already
  defined in the workload spec v2 and are present on every `Request` struct; this feature
  wires them into the serving path, it does not add the labels themselves.
- `TenantID` is already a field on the `Request` struct (added in Phase 0 workload unification).
- "Overload" is defined as: the synchronous WaitQueue reaching its configured capacity
  limit (or the cluster's active-request count reaching the configured maximum).
- "Idle capacity window" is defined as: the synchronous WaitQueue length dropping to zero.
- Deferred queue capacity is unbounded within a single simulation run; the simulation
  horizon acts as the natural termination boundary.
- Per-tenant observation windows are a configurable parameter with a reasonable default
  (e.g., equivalent to 60 seconds of simulated time); the exact value is a tuning knob,
  not a correctness invariant.
- T-2 (tenant fairness) depends on T-1 (tier-ordered shedding) being in place; the two
  PRs are sequentially ordered, not parallel.
