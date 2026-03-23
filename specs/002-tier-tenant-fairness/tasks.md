# Tasks: Phase 1B — Service Tiers & Tenant Fairness

**Input**: Design documents from `specs/002-tier-tenant-fairness/`
**Branch**: `002-tier-tenant-fairness`
**PRs**: #809 (US1), #810 (US2), #811 (US3), #812 (US4)

**Tests**: Included — project requires BDD/TDD per `docs/contributing/standards/principles.md`. Tests MUST be written first and must FAIL before implementation begins.

## Format: `[ID] [P?] [Story?] Description — file path`

- **[P]**: Can run in parallel (different files, no blocking dependencies)
- **[Story]**: User story label (US1–US4 map to spec.md priority order)

---

## Phase 1: Setup (Shared Foundation)

**Purpose**: Add the `sloTierPriority()` helper that US1 and US3 both depend on. Unblocks all subsequent phases.

- [ ] T001 Add `sloTierPriority(class string) int` function (Background=0 … Critical=4; empty→3) to `sim/admission.go`
- [ ] T002 Add `shedByTier map[string]int` field to `ClusterSimulator` struct and initialize to `make(map[string]int)` in `NewClusterSimulator()` in `sim/cluster/cluster.go`

**Checkpoint**: `go build ./...` passes. Foundation ready for all user stories.

---

## Phase 2: Foundational (Blocking Prerequisites)

No additional foundational work — T001 and T002 are sufficient. User story phases may begin.

---

## Phase 3: User Story 1 — Tier-Ordered Admission Shedding (Priority: P1) 🎯 PR #809

**Goal**: Under overload, shed Sheddable before Standard, Standard before Critical. Batch/Background pass through (deferred queue is US2). Closes issue #809.

**Independent Test**: Run a two-tier workload (Critical + Sheddable, equal volumes) at 2× cluster capacity with `--admission-policy tier-shed`. Verify `shed(Sheddable) >> shed(Critical)` and `shed(Critical) == 0` until very high overload.

### Tests for User Story 1

> **Write these first — they MUST FAIL before T008 is implemented.**

- [ ] T003 [US1] Table-driven unit tests for `sloTierPriority()`: all 5 canonical classes + empty string + unknown string — `sim/admission_tier_test.go` (new file)
- [ ] T004 [US1] Behavior test: `TierShedAdmission.Admit()` returns `(true,"")` for Critical and Standard under overload — `sim/admission_tier_test.go`
- [ ] T005 [US1] Behavior test: `TierShedAdmission.Admit()` returns `(false, reason)` for Sheddable when max effective load > threshold — `sim/admission_tier_test.go`
- [ ] T006 [US1] Behavior test: Batch and Background always return `(true,"")` from `TierShedAdmission` regardless of load — `sim/admission_tier_test.go`
- [ ] T007 [US1] Invariant test: monotonic shedding order under a load ramp — shed counts satisfy `Sheddable ≥ Standard ≥ Critical` at each measurement point — `sim/cluster/cluster_tier_test.go` (new file)

### Implementation for User Story 1

- [ ] T008 [US1] Implement `TierShedAdmission` struct + `Admit()` method in `sim/admission.go` (depends on T001; T003–T006 must fail first)
- [ ] T009 [US1] Register `"tier-shed"` in `validAdmissionPolicies`, add `TierShedThreshold int` and `TierShedMinPriority int` to `AdmissionConfig`, add `case "tier-shed":` in `NewAdmissionPolicy()` — `sim/bundle.go`
- [ ] T010 [US1] In `AdmissionDecisionEvent.Execute()` increment `cs.shedByTier[tier]++` on rejection (after `cs.rejectedRequests++`) — `sim/cluster/cluster_event.go`

**Checkpoint**: `go test ./sim/... ./sim/cluster/...` passes. T003–T007 now pass. PR #809 is ready.

---

## Phase 4: User Story 2 — Deferred Queue for Batch/Background (Priority: P1) PR #810

**Goal**: Batch and Background requests park in a deferred queue when cluster is busy; promoted when all instance queues drain. Closes issue #810.

**Depends on**: Phase 3 complete (PR #809 merged).

**Independent Test**: Workload with steady real-time (Standard) traffic at 95% capacity plus a stream of Batch requests. Verify Batch requests do not execute while Standard requests are queued, then complete after Standard traffic ends. `DeferredHorizonInterrupted` appears in metrics when horizon cuts off deferred work.

### Tests for User Story 2

> **Write these first — they MUST FAIL before T014–T016 are implemented.**

- [ ] T011 [US2] Behavior test: Batch request during busy cluster enters `deferredQueue` and is NOT counted as rejected — `sim/cluster/cluster_deferred_test.go` (new file)
- [ ] T012 [US2] Behavior test: deferred request is promoted (injected as ClusterArrivalEvent) within one scheduling step of all instance WaitQueues becoming empty — `sim/cluster/cluster_deferred_test.go`
- [ ] T013 [US2] Invariant test: INV-1 holds at simulation end with non-empty deferred queue at horizon (`injected == completed + running + queued + shed + dropped + deferred_horizon_interrupted`) — `sim/cluster/cluster_deferred_test.go`

### Implementation for User Story 2

- [ ] T014 [US2] Add `deferredQueue []*sim.Request` field, `isBusy() bool` method, `promoteDeferred()` method, and promotion check in `Run()` loop — `sim/cluster/cluster.go` (depends on T002)
- [ ] T015 [US2] Add pre-admission intercept in `AdmissionDecisionEvent.Execute()`: if tier is `"batch"` or `"background"` AND `cs.isBusy()`, append to `cs.deferredQueue` and return — `sim/cluster/cluster_event.go`
- [ ] T016 [US2] Add `DeferredHorizonInterrupted int` to `RawMetrics`, populate in `CollectRawMetrics()` from a new `deferredCount int` parameter (or `ClusterSimulator` accessor) — `sim/cluster/metrics.go`

**Checkpoint**: `go test ./sim/cluster/...` passes. T011–T013 now pass. PR #810 is ready.

---

## Phase 5: User Story 3 — Per-Tenant Fair-Share Enforcement (Priority: P2) PR #811

**Goal**: Track per-tenant in-flight counts; over-budget tenants have Sheddable-and-below requests shed preferentially. Critical and Standard from over-budget tenants are still admitted. Closes issue #811.

**Depends on**: Phase 3 complete (PR #809 merged). Can proceed in parallel with Phase 4.

**Independent Test**: Two-tenant workload, Tenant A budget=30%, Tenant B budget=70%, both sending Standard+Sheddable traffic at equal rates, cluster at 80% capacity. Verify Tenant A's Sheddable shed at higher rate than Tenant B's while both tenants' Standard traffic is unaffected. Single-tenant run without `tenant_budgets` produces byte-identical output to pre-Phase-1B baseline.

### Tests for User Story 3

> **Write these first — they MUST FAIL before T020–T023 are implemented.**

- [ ] T017 [US3] Unit tests for `TenantTracker`: `IsOverBudget` true/false, `OnStart`/`OnComplete` balance, zero-value safety (nil budgets → always false), empty TenantID → no-op — `sim/cluster/tenant_test.go` (new file)
- [ ] T018 [US3] Behavior test: over-budget tenant's Sheddable request rejected while on-budget tenant's Sheddable request admitted under same load — `sim/cluster/cluster_tenant_test.go` (new file)
- [ ] T019 [US3] Behavior test: over-budget tenant's Critical request is NOT rejected solely due to budget status — `sim/cluster/cluster_tenant_test.go`
- [ ] T020 [US3] Invariant test: simulation with `TenantBudgets: nil` produces byte-identical stdout to run without tenant tracking (INV-6) — `sim/cluster/cluster_tenant_test.go`

### Implementation for User Story 3

- [ ] T021 [P] [US3] Implement `TenantTracker` struct + `NewTenantTracker()` + `IsOverBudget()` + `OnStart()` + `OnComplete()` — `sim/cluster/tenant.go` (new file; depends on T017 failing first)
- [ ] T022 [P] [US3] Add `TenantBudgets map[string]float64 \`yaml:"tenant_budgets,omitempty"\`` to `DeploymentConfig` — `sim/cluster/deployment.go`
- [ ] T023 [US3] Add `tenantTracker *TenantTracker` field to `ClusterSimulator`; initialize in `NewClusterSimulator()` when `config.TenantBudgets != nil`; call `OnStart`/`OnComplete` at request dispatch and terminal state — `sim/cluster/cluster.go` (depends on T021, T022)
- [ ] T024 [US3] Add tenant budget override in `AdmissionDecisionEvent.Execute()`: after admission policy returns `admitted=true`, if `cs.tenantTracker.IsOverBudget(req.TenantID)` and `sloTierPriority(req.SLOClass) < 3`, override to rejected — `sim/cluster/cluster_event.go` (depends on T023)

**Checkpoint**: `go test ./sim/cluster/...` passes. T017–T020 now pass. PR #811 is ready.

---

## Phase 6: User Story 4 — Per-Tenant Jain Fairness Metrics (Priority: P3) PR #812

**Goal**: Emit `per_tenant` section in simulation output with per-tenant completed count, output tokens, and cluster-level Jain fairness index. Closes issue #812.

**Depends on**: Phase 5 complete (PR #811 merged).

**Independent Test**: Two-tenant balanced workload → Jain index ≥ 0.99 in output. Two-tenant 10:1 skewed workload → Jain index < 0.70. No TenantID set → no `per_tenant` section in output.

### Tests for User Story 4

> **Write these first — they MUST FAIL before T027–T028 are implemented.**

- [ ] T025 [US4] Unit tests for `ComputePerTenantMetrics()`: balanced two-tenant → Jain ≥ 0.99; skewed 10:1 → Jain < 0.70; all empty TenantID → returns nil; Jain value matches manual calculation within 0.001 — `sim/cluster/metrics_tenant_test.go` (new file)

### Implementation for User Story 4

- [ ] T026 [US4] Add `TenantMetrics` struct (json tags: `tenant_id`, `completed_requests`, `total_output_tokens`) and `ComputePerTenantMetrics(aggregated *sim.Metrics) (map[string]*TenantMetrics, float64)` to `sim/cluster/metrics.go`; return `nil, 0` when all TenantIDs are empty (depends on T025 failing first)
- [ ] T027 [US4] Add `printPerTenantMetrics(w io.Writer, metrics map[string]*TenantMetrics, jainIndex float64)` (no-op when nil) and call it after `printPerModelMetrics` in `cmd/root.go`

**Checkpoint**: `go test ./...` passes. T025 now passes. PR #812 is ready.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T028 [P] Run `go test ./... -count=1 -race` and fix any failures
- [ ] T029 [P] Run `golangci-lint run ./...` and fix all warnings
- [ ] T030 Update `docs/reference/configuration.md`: document `tier-shed` admission policy options (`tier_shed_threshold`, `tier_shed_min_priority`) and `tenant_budgets` deployment config field
- [ ] T031 Update `docs/guide/results.md`: document `deferred_horizon_interrupted` counter and `per_tenant` metrics section

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (T001–T002)
    └──► Phase 3 US1 / PR-1 (T003–T010)   ← #809
              ├──► Phase 4 US2 / PR-2 (T011–T016)  ← #810  (sequential after #809)
              └──► Phase 5 US3 / PR-3 (T017–T024)  ← #811  (can parallel with #810)
                        └──► Phase 6 US4 / PR-4 (T025–T027)  ← #812
                                  └──► Phase 7 Polish (T028–T031)
```

### User Story Dependencies

| Story | Depends on | Can parallel with |
|-------|-----------|------------------|
| US1 (PR-1, #809) | Phase 1 only | — |
| US2 (PR-2, #810) | US1 merged | US3 (after US1) |
| US3 (PR-3, #811) | US1 merged | US2 (after US1) |
| US4 (PR-4, #812) | US3 merged | — |

### Within Each User Story

1. Tests written first → confirmed FAILING
2. Implementation tasks run → tests turn GREEN
3. `go test ./...` + `golangci-lint` clean
4. PR ready for review

### Parallel Opportunities Within Stories

- **US3**: T021 (`tenant.go`) and T022 (`deployment.go`) are fully parallel [P]
- **Phase 7**: T028 and T029 are parallel [P]

---

## Parallel Execution Example: User Story 3

```text
After US1 (PR-1) merges:

Parallel stream A (US2 / PR-2):
  T011 → T012 → T013 (tests, write+fail)
  T014 → T015 → T016 (implement, turn green)

Parallel stream B (US3 / PR-3):
  T017 → T018 → T019 → T020 (tests, write+fail)
  T021 ──┐
  T022 ──┴──► T023 → T024 (implement, turn green)
```

---

## Implementation Strategy

### MVP (User Story 1 only, PR-1 / #809)

1. Phase 1: T001–T002 (foundation, ~10 min)
2. Phase 3 tests: T003–T007 (write failing tests, ~20 min)
3. Phase 3 impl: T008–T010 (make tests pass, ~30 min)
4. **VALIDATE**: `go test ./sim/... ./sim/cluster/...` green, `golangci-lint` clean
5. Open PR #809

### Full Phase 1B (all 4 PRs)

| PR | Issues | Est. new lines | New test files |
|----|--------|---------------|----------------|
| #809 | #809 | ~55 | `admission_tier_test.go`, `cluster_tier_test.go` |
| #810 | #810 | ~80 | `cluster_deferred_test.go` |
| #811 | #811 | ~95 | `tenant_test.go`, `cluster_tenant_test.go` |
| #812 | #812 | ~65 | `metrics_tenant_test.go` |

---

## Notes

- `[P]` = different files, no inter-task dependency → safe to implement simultaneously
- TDD order is non-negotiable per `docs/contributing/standards/principles.md`: test MUST fail before implementation
- Each PR should include only the tasks listed for that story — no cross-PR changes
- `sloTierPriority()` (T001) lives in `sim/` (not `sim/cluster/`) to keep tenant enforcement in the cluster layer able to call it without a circular import
- Speckit artifacts (`specs/002-tier-tenant-fairness/`) are committed as a separate PR before any implementation PRs
