# Feature Specification: Deferred Queue for Batch and Background Requests

**Feature Branch**: `004-deferred-queue-batch`
**Created**: 2026-03-26
**Status**: Draft
**Tracking Issue**: #696 (Phase 1B-1b, sub-issue #810)
**Depends on**: Phase 1B-1a (#809) — merged as #825

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Batch/Background Requests Complete Under Mixed Workload (Priority: P1)

A simulation operator runs a mixed workload with Critical, Standard, and Batch/Background requests against a cluster that is often busy with high-priority traffic. Currently, Batch and Background requests are admitted and may compete with real-time requests. The operator expects Batch/Background requests to eventually complete — just without competing against real-time traffic.

**Why this priority**: Core feature value. Without deferred promotion, Batch/Background requests either compete with real-time traffic or risk being dropped. This delivers the "spot/batch" contract: lower priority work eventually completes.

**Independent Test**: Run a simulation with 50 Batch requests and a busy cluster. Verify all 50 Batch requests either complete or appear in deferred-horizon-interrupted — zero are silently lost.

**Acceptance Scenarios**:

1. **Given** a busy cluster (real-time requests in flight), **When** a Batch request arrives, **Then** the request is held in the deferred queue and not counted as rejected.
2. **Given** deferred Batch requests in the queue, **When** all instance queues drain to idle, **Then** all deferred requests are promoted and enter normal processing.
3. **Given** deferred requests were promoted and the cluster refills, **When** more Batch requests arrive, **Then** they are again deferred (deferral is continuous, not one-shot).
4. **Given** the simulation horizon is reached with deferred requests still queued, **When** metrics are collected, **Then** the count of horizon-interrupted deferred requests appears in output and request conservation holds.

---

### User Story 2 — Real-Time Requests Unaffected by Batch Traffic (Priority: P2)

A simulation researcher compares two runs: one with only Critical/Standard requests, one with Critical/Standard plus Batch/Background. The researcher expects identical latency distributions for real-time traffic in both runs.

**Why this priority**: Correctness guarantee. The deferred queue must not introduce head-of-line blocking or resource contention for real-time tiers.

**Independent Test**: Run identical seeds with and without Batch traffic. Verify p50/p99 latency for Critical and Standard requests is equal across both runs.

**Acceptance Scenarios**:

1. **Given** the cluster is processing real-time requests, **When** deferred Batch requests are present, **Then** the deferred requests do not consume instance queue slots until the cluster is genuinely idle.
2. **Given** a deferred queue promotion fires, **When** more real-time requests arrive in the same tick, **Then** real-time requests are not blocked behind promoted Batch requests.

---

### User Story 3 — Request Conservation Remains Exact (Priority: P3)

A simulation auditor verifies that no requests are silently lost. Every injected request must be accounted for: completed, running, queued, shed, dropped, or deferred-horizon-interrupted.

**Why this priority**: Invariant correctness. Request conservation is a foundational guarantee of the simulator. Deferred requests represent a new accounting category that must be included.

**Independent Test**: Run any simulation with Batch/Background traffic and a finite horizon. Verify that `injected == completed + still_running + still_queued + shed + dropped_unservable + timed_out + rejected + deferred_horizon_interrupted` at simulation end.

**Acceptance Scenarios**:

1. **Given** a simulation ends with requests still in the deferred queue, **When** metrics are reported, **Then** `deferred_horizon_interrupted` equals the number of requests that were never promoted.
2. **Given** all deferred requests were promoted before the horizon, **When** metrics are reported, **Then** `deferred_horizon_interrupted == 0` and standard conservation holds.

---

### Edge Cases

- What happens when a Batch request arrives and the cluster has zero instances? → Cluster is not busy; request is admitted normally (not deferred).
- What happens when a Background request arrives during an idle cluster? → Cluster is not busy; request is admitted immediately and never enters the deferred queue.
- What happens when `deferredQueue` grows very large before promotion? → Promotion injects all entries at once; the simulator must handle the burst without error.
- What happens if promotion fires and a new real-time request arrives in the same tick? → Real-time request is not blocked; both enter the event queue and are scheduled by normal priority/arrival ordering.
- What happens when the simulation horizon arrives while deferred requests remain? → Requests still in the deferred queue at horizon are counted as `deferred_horizon_interrupted`; requests already promoted follow normal horizon logic.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The simulator MUST hold Batch and Background requests in a deferred queue rather than admitting them when the cluster has non-zero in-flight load.
- **FR-002**: The simulator MUST promote all deferred requests to normal processing within one scheduling step of the cluster becoming fully idle.
- **FR-003**: Promoted requests MUST re-enter the arrival pipeline at the current simulation clock, following the same routing and scheduling path as any newly arriving request.
- **FR-004**: Deferral MUST NOT increment the rejected-requests counter; deferred requests are neither admitted nor rejected.
- **FR-005**: The simulator MUST NOT access output token counts during deferral or promotion decisions.
- **FR-006**: Metrics output MUST include a `deferred_horizon_interrupted` count representing requests still deferred when the simulation horizon is reached.
- **FR-007**: Request conservation MUST hold at simulation end: `injected == completed + running + queued + shed + dropped + timed_out + deferred_horizon_interrupted`.
- **FR-008**: The deferred queue MUST activate without any new configuration — no new YAML fields or flags are needed.
- **FR-009**: The simulator MUST remain work-conserving: promotion must not stall or delay in-progress real-time work.
- **FR-010**: Critical, Standard, and Sheddable requests MUST be completely unaffected — same admission, routing, and latency behavior as before this feature.

### Key Entities

- **Deferred Queue**: An ordered collection of Batch/Background requests waiting for cluster idle capacity. Preserves arrival order; all entries are promoted atomically when the cluster becomes idle.
- **Cluster Idle State**: The condition where no instance has any in-flight, queued, or batched requests. The trigger for deferred queue promotion.
- **Deferred-Horizon-Interrupted Count**: A metric recording how many requests were still in the deferred queue when the simulation horizon ended — required for request conservation accounting.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of Batch and Background requests in a finite simulation either complete or appear in `deferred_horizon_interrupted` — zero silent losses.
- **SC-002**: p50 and p99 latency for Critical and Standard requests is identical (within floating-point tolerance) whether or not Batch/Background traffic is present in the same simulation run.
- **SC-003**: `deferred_horizon_interrupted + completed_batch_background == injected_batch_background` holds for every simulation run with a finite horizon.
- **SC-004**: No new configuration fields are required; existing YAML workload files continue to work without modification.
- **SC-005**: The full test suite passes with no regressions after the feature is added.

## Assumptions

- "Cluster idle" means all instances have zero in-flight, queued, and in-batch requests. A cluster with no instances is treated as idle.
- Deferred queue promotion is atomic: all queued requests are promoted in the same scheduling step.
- Deferral applies unconditionally to the `"batch"` and `"background"` SLO class strings with no configuration knob.
- Requests promoted from the deferred queue reuse the same request struct with unchanged fields.
- The pre-admission deferral intercept fires before the admission policy is evaluated, so the admission policy is never reached for deferred requests.

## Out of Scope

- Per-tenant fairness within the deferred queue (Phase 1B-2a/#811).
- Priority ordering within the deferred queue (FIFO is sufficient).
- Configurable deferred queue capacity limits (unbounded for now).
- Deferred queue persistence across simulation restart.
