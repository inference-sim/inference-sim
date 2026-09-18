# Feature Specification: Tier-Ordered Admission Shedding

**Feature Branch**: `003-tier-shed-admission`
**Created**: 2026-03-25
**Status**: Draft
**Tracking issue**: #696 | **Sub-issue**: #809 | **Phase**: 1B-1a

## User Scenarios & Testing

### User Story 1 — Protect high-priority requests under overload (Priority: P1)

A capacity planner running BLIS load tests wants the simulator to shed lower-priority requests first when the cluster is overloaded. Today all requests are treated equally. With this feature they configure `tier-shed` admission and observe that Sheddable requests are rejected before Standard, and Standard before Critical — matching real production serving behavior.

**Why this priority**: This is the core deliverable. All other stories depend on the tier priority mapping and policy being in place.

**Independent Test**: Run a two-tier workload (Critical + Sheddable, equal volumes) at 2× cluster capacity with `tier-shed` policy. Verify `shed(Sheddable) >> shed(Critical)` and `shed(Critical) == 0` until very high overload.

**Acceptance Scenarios**:

1. **Given** a cluster at 100% capacity with `tier-shed` policy, **When** Critical and Sheddable requests arrive at equal rates, **Then** Sheddable requests are rejected and Critical requests are admitted.
2. **Given** a cluster with effective load at or below the configured threshold, **When** any tier of request arrives, **Then** all requests are admitted regardless of tier.
3. **Given** `tier-shed` with default settings (MinAdmitPriority=3), **When** Standard requests arrive under overload, **Then** they are admitted (Standard priority 3 ≥ default minimum 3).

---

### User Story 2 — Batch and Background pass through the tier-shed policy (Priority: P1)

Batch and Background are the lowest-priority tiers and will be handled by a deferred queue in a later phase. The tier-shed policy must pass them through unchanged so that the deferred queue phase can intercept them without conflicting behavior.

**Why this priority**: Correctness dependency for the deferred queue PR (#810). If this policy shed Batch/Background, the deferred queue could never receive them.

**Independent Test**: Run a workload including Batch requests under heavy overload with `tier-shed` policy. Verify the admission policy reports admitted=true for Batch and Background (they may still be dropped at capacity, but not by this policy).

**Acceptance Scenarios**:

1. **Given** a cluster under overload with `tier-shed` policy, **When** a Batch request arrives, **Then** the admission policy returns admitted=true regardless of load level.
2. **Given** a cluster under overload with `tier-shed` policy, **When** a Background request arrives, **Then** the admission policy returns admitted=true regardless of load level.

---

### User Story 3 — Shedding order remains monotonic across the full load range (Priority: P2)

Under increasing load, the simulator must never shed a higher-priority tier more than a lower-priority tier. This invariant validates correct behavior throughout the entire load range, not just at a single point.

**Why this priority**: Required for the monotonic shedding invariant. Validates that tier priority mapping and policy logic are consistent end-to-end.

**Independent Test**: Run a five-tier workload at increasing load steps. At each measurement point verify: `shed(Sheddable) ≥ shed(Standard) ≥ shed(Critical)`.

**Acceptance Scenarios**:

1. **Given** a cluster with all five tiers sending traffic, **When** load increases from 0 to 3× capacity, **Then** shed counts remain monotonically ordered by tier at every measurement interval.
2. **Given** a request with an unknown or empty `SLOClass`, **When** the admission policy evaluates it, **Then** it is treated as Standard (priority 3) and never shed ahead of Standard requests.

---

### Edge Cases

- What happens when `SLOClass` is empty or unknown? Treated as Standard (priority 3) for backward compatibility with requests that predate SLO tier tagging.
- What happens when there are zero instances (empty cluster state)? Policy must not panic; all requests are admitted.
- What happens when `MinAdmitPriority` is 0? Equivalent to admitting all tiers under overload — same as no tier enforcement. This is a valid but almost certainly unintended configuration; the default must be 3 (Standard).
- What happens at exactly the threshold value? Overload triggers only when effective load is strictly greater than `OverloadThreshold`; requests at exactly the threshold are admitted.
- What happens to a simulation that does not use `tier-shed`? Behavior must be byte-identical to the pre-feature baseline.

---

## Requirements

### Functional Requirements

- **FR-001**: The system MUST provide a `tier-shed` admission policy that rejects requests whose tier priority falls below a configurable minimum when the cluster is overloaded.
- **FR-002**: The system MUST define a fixed priority ordering for the five SLO tiers: Critical (4) > Standard (3) > Sheddable (2) > Batch (1) > Background (0).
- **FR-003**: Under overload, requests with tier priority ≥ `MinAdmitPriority` MUST be admitted; requests below it MUST be rejected with a human-readable reason.
- **FR-004**: The `tier-shed` policy MUST always admit Batch and Background requests regardless of load; their deferral is handled by a later phase.
- **FR-005**: Unknown or empty `SLOClass` values MUST be treated as Standard priority (3) to preserve backward compatibility.
- **FR-006**: The overload signal MUST be derived from observable cluster state at admission time and MUST NOT read `req.OutputTokens` (INV-9 oracle boundary).
- **FR-007**: The system MUST track per-tier rejection counts so capacity planners can observe which tiers were shed and by how much.
- **FR-008**: The policy MUST be configurable via `tier_shed_threshold` (default 0) and `tier_shed_min_priority` (default 3).
- **FR-009**: Simulations that do not configure `tier-shed` MUST produce byte-identical output to a pre-feature baseline.
- **FR-010**: Shedding order MUST be monotonic: at any point during simulation, shed counts satisfy `shed(lower_tier) ≥ shed(higher_tier)` for all tier pairs.

### Key Entities

- **SLO Tier Priority**: An integer rank (0–4) assigned to each `SLOClass` string. Defines relative protection against admission shedding.
- **Tier-Shed Admission Policy**: A stateless policy that compares a request's tier priority against a configurable minimum, using per-instance effective load as the overload signal.
- **Overload Threshold**: The maximum per-instance effective load (queue depth + active batch + in-flight count) below which no tier-based shedding occurs.
- **MinAdmitPriority**: The lowest tier priority protected from shedding under overload. Requests with priority strictly below this are rejected when overloaded.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: In a two-tier (Critical + Sheddable) workload at 2× capacity, Critical shed count is 0 while Sheddable shed count equals total excess arrivals (100% selectivity).
- **SC-002**: Shedding monotonicity holds at every measurement step across a 0–3× load ramp for all five tiers.
- **SC-003**: A simulation not using `tier-shed` produces byte-identical output before and after this feature is introduced (zero regression).
- **SC-004**: Per-tier rejection counts appear in simulation output so planners can distinguish Sheddable shedding from Standard shedding.
- **SC-005**: Empty or unknown `SLOClass` requests are never shed ahead of Standard-tier requests under any load condition.

---

## Assumptions

- `Request.SLOClass` is already set at arrival time for all requests generated by the v2 workload spec. No changes to request generation are needed.
- The `AdmissionPolicy` interface (`Admit(req, state) (bool, string)`) is stable and will not change in this PR.
- Max per-instance effective load (QueueDepth + BatchSize + InFlightRequests) is the correct overload signal for admission decisions.
- The deferred queue for Batch/Background (PR #810) is a separate deliverable. This PR only ensures the pass-through behavior that deferred queue requires.
- `MinAdmitPriority` defaults to 3; callers must explicitly set a lower value to loosen enforcement.

---

## Out of Scope

- Deferred queue for Batch/Background requests (PR #810)
- Per-tenant fair-share enforcement (PR #811)
- Per-tenant Jain fairness metrics (PR #812)
- Dynamic or adaptive threshold adjustment (thresholds are static config)
- CLI flags for tier-shed parameters (YAML config only in this PR)
