# Feature Specification: Per-Tenant Jain Fairness Index

**Feature Branch**: `005-tenant-jain-fairness`
**Created**: 2026-03-30
**Status**: Draft
**Input**: Issue #812 — Phase 1B-2b: feat(metrics): per-tenant Jain fairness index in simulation output

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Inspect Tenant Fairness After Simulation (Priority: P1)

An operator runs a BLIS simulation with a multi-tenant workload (requests labelled with tenant identifiers). After the run completes, they read the simulation output and see a per-tenant summary that shows how many requests each tenant completed, how many output tokens were served per tenant, and a single Jain fairness index measuring how evenly tokens were distributed across tenants. The operator uses this to determine whether the cluster served all tenants equitably.

**Why this priority**: Fairness visibility is the core value of this feature. Without it, operators have no observable signal about tenant equity from a simulation run.

**Independent Test**: Run a two-tenant workload with equal request rates; the output must include a `per_tenant` section listing both tenants with their counts and tokens, plus a Jain index ≥ 0.99.

**Acceptance Scenarios**:

1. **Given** a simulation run with requests from tenants `alice` and `bob` split 50/50, **When** the simulation completes, **Then** the output includes a `per_tenant` section listing `alice` and `bob` with their completed request counts, token totals, and a Jain fairness index ≥ 0.99.
2. **Given** a simulation with requests from tenant `alice` receiving 3× more tokens than `bob`, **When** the simulation completes, **Then** the Jain fairness index reported is less than 1.0 and within 2% of the value computed directly from the same token counts.
3. **Given** a highly skewed workload where one tenant receives 10× more output tokens than another, **When** the simulation completes, **Then** the Jain fairness index is less than 0.70, correctly signaling severe unfairness.
4. **Given** a simulation where no request carries a tenant identifier, **When** the simulation completes, **Then** the output does NOT include a `per_tenant` section.

---

### User Story 2 — Inspect Tenant Fairness After Trace Replay (Priority: P2)

An operator replays a previously captured TraceV2 trace (which carries tenant labels) through the BLIS discrete-event simulator using `blis replay`. The replay output includes the same per-tenant fairness section as a fresh simulation run, so the operator can assess historical fairness without re-running the workload against a live server.

**Why this priority**: The replay pipeline must produce parity output with the run pipeline so operators can use the same analysis workflow for both live and historical data.

**Independent Test**: Replay a trace file containing requests from two distinct tenants; the output must include a `per_tenant` section with the same structure as produced by `blis run`.

**Acceptance Scenarios**:

1. **Given** a TraceV2 trace with tenant labels on every record, **When** `blis replay` finishes, **Then** the output includes a `per_tenant` section listing each tenant's request count, token total, and Jain fairness index.
2. **Given** a TraceV2 trace with no tenant labels, **When** `blis replay` finishes, **Then** the output does NOT include a `per_tenant` section.

---

### Edge Cases

- What happens when only a single tenant is present? → The fairness index should equal 1.0 (a single player is trivially fair).
- What happens when all tenants received zero output tokens? → The fairness index should be 1.0 (all-zero is perfectly fair by convention, matching the existing `JainFairnessIndex` implementation).
- What happens when some requests have tenant identifiers and some do not? → Only requests with non-empty tenant identifiers contribute to the per-tenant counts; requests without a tenant identifier are omitted.
- What happens when tenants are listed in the output? → Tenants MUST appear in deterministic (lexicographically sorted) order on every run with the same seed (INV-6).
- What happens to requests that were deferred (Batch/Background parked in the deferred queue) but never executed before the simulation horizon? → These deferred-horizon-interrupted requests have zero output tokens and are excluded from per-tenant counts entirely; they never ran so they never consumed the fairness resource.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: When at least one completed request carries a non-empty tenant identifier, the simulation output MUST include a per-tenant summary section.
- **FR-002**: The per-tenant summary MUST report, for each tenant: the number of completed requests attributed to that tenant and the total output tokens served to that tenant.
- **FR-003**: The per-tenant summary MUST report a single Jain fairness index computed over the per-tenant output token totals.
- **FR-004**: The reported Jain fairness index MUST be within 2% (relative error) of the value obtained by calling `JainFairnessIndex` directly on the same per-tenant token map.
- **FR-005**: The per-tenant summary MUST be absent (no section, no empty map) when no completed request carries a tenant identifier.
- **FR-006**: Tenants MUST be listed in lexicographically sorted order to ensure deterministic output across identical runs (INV-6).
- **FR-007**: The per-tenant summary MUST appear in the output of both `blis run` and `blis replay`.

### Key Entities

- **TenantID**: A non-empty string label on a request identifying which tenant originated it. Requests with empty TenantID are not attributed to any tenant.
- **Per-tenant metrics**: The aggregate statistics for a single tenant — completed request count and total output tokens served during the simulation window.
- **Jain fairness index**: A scalar in [1/N, 1.0] measuring how evenly output tokens were distributed across N tenants. Computed as (Σxi)² / (N·Σxi²) where xi = per-tenant token total.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A perfectly balanced two-tenant workload (equal output tokens per tenant) produces a reported Jain fairness index of at least 0.99.
- **SC-002**: The reported Jain fairness index deviates by no more than 2% (relative) from the value computed directly from the same per-tenant token totals.
- **SC-003**: The per-tenant section does not appear in the output of any run where all requests have empty tenant identifiers — backward-compatible with existing single-tenant workloads.
- **SC-004**: Per-tenant output appears in identical order across repeated runs with the same seed (determinism, INV-6 compliance).
- **SC-005**: Both `blis run` and `blis replay` produce per-tenant metrics sections with the same structure when tenant-labelled requests are present.
- **SC-006**: A highly skewed two-tenant workload where one tenant receives 10× more output tokens than the other produces a reported Jain fairness index less than 0.70, confirming the index is sensitive to severe imbalance (not just accurate near 1.0).

## Assumptions

- `JainFairnessIndex(map[string]float64)` is already implemented and correct in `sim/cluster/metrics.go`; this feature only wires its output into the printed metrics pipeline.
- Per-tenant metrics are computed from completed requests only; requests still in the queue or running at simulation end are excluded (consistent with how throughput is computed).
- Output tokens (not input tokens) are the fairness resource: they measure the compute actually delivered to each tenant.
- The per-tenant section is printed to stdout as a formatted plaintext section (following the `=== Per-SLO Metrics ===` / `=== Per-Model Metrics ===` pattern), not embedded in the `MetricsOutput` JSON block. Embedding in JSON is architecturally blocked: `JainFairnessIndex` and the compute function live in `sim/cluster/`, while `MetricsOutput` serialization lives in `sim/`; `sim/` cannot import `sim/cluster/` without creating a dependency cycle.
- Requests without a TenantID contribute to global metrics but are silently excluded from the per-tenant breakdown (no "unknown" bucket).
- Deferred-horizon-interrupted requests (Batch/Background requests that were never promoted from the deferred queue before the simulation horizon) are excluded from per-tenant counts; they consumed no output tokens and must not distort the Jain index.
