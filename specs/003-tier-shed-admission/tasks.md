# Tasks: Phase 1B-1a — Tier-Ordered Admission Shedding

**Input**: Design documents from `specs/003-tier-shed-admission/`
**Branch**: `003-tier-shed-admission`
**PR**: #809

**Tests**: Included — project requires BDD/TDD per `docs/contributing/standards/principles.md`. Tests MUST be written first and must FAIL before implementation begins.

## Format: `[ID] [P?] [Story?] Description — file path`

- **[P]**: Can run in parallel (different files, no blocking dependencies)
- **[Story]**: User story label (US1–US3 map to spec.md priority order)

---

## Phase 1: Setup (Shared Foundation)

**Purpose**: Add the `SLOTierPriority()` helper and register the `"tier-shed"` policy name — both needed by all subsequent test and implementation tasks.

- [X] T001 Add `SLOTierPriority(class string) int` function (exported; Critical=4 … Background=0; empty/unknown→3) to `sim/admission.go`
- [X] T002 Add `"tier-shed": true` to `validAdmissionPolicies` map in `sim/bundle.go` (validation only — no factory case)
- [X] T003 [P] Add `TierShedThreshold int \`yaml:"tier_shed_threshold,omitempty"\`` and `TierShedMinPriority int \`yaml:"tier_shed_min_priority,omitempty"\`` fields to `DeploymentConfig` in `sim/cluster/deployment.go`
- [X] T004 [P] Add `shedByTier map[string]int` field to `ClusterSimulator` struct and initialize to `make(map[string]int)` in `NewClusterSimulator()` in `sim/cluster/cluster.go`

**Checkpoint**: `go build ./...` passes. Foundation ready for all user stories.

---

## Phase 2: Foundational (Blocking Prerequisites)

No additional foundational work — T001–T004 are sufficient. User story phases may begin.

---

## Phase 3: User Stories 1 & 2 — Tier Shedding Behavior (Priority: P1) 🎯 PR #809

**Goal**: `TierShedAdmission.Admit()` rejects Sheddable requests under overload while admitting Critical and Standard. Batch and Background always pass through. Per-tier shed counter populated on rejection.

**Independent Test**: Run a two-tier workload (Critical + Sheddable, equal volumes) at 2× cluster capacity with `--admission-policy tier-shed`. Verify `shed(Sheddable) >> shed(Critical)` and `shed(Critical) == 0`.

### Tests for User Stories 1 & 2

> **Write these first — they MUST FAIL before T012 is implemented.**

- [X] T005 [P] [US1] Table-driven unit tests for `SLOTierPriority()`: all 5 canonical classes + empty string + unknown string → `sim/admission_tier_test.go` (new file)
- [X] T006 [P] [US1] Behavior test: `TierShedAdmission.Admit()` returns `(true,"")` for Critical and Standard under overload — `sim/admission_tier_test.go`
- [X] T007 [P] [US1] Behavior test: `TierShedAdmission.Admit()` returns `(false, reason)` for Sheddable when maxLoad > OverloadThreshold — `sim/admission_tier_test.go`
- [X] T008 [P] [US1] Behavior test: `TierShedAdmission.Admit()` returns `(true,"")` for all tiers when maxLoad ≤ OverloadThreshold — `sim/admission_tier_test.go`
- [X] T009 [P] [US1] Behavior test: empty `state.Snapshots` → no panic, all requests admitted — `sim/admission_tier_test.go`
- [X] T010 [P] [US1] Behavior test: empty `SLOClass` treated as Standard (priority 3), never shed below Standard — `sim/admission_tier_test.go`
- [X] T011 [P] [US2] Behavior test: Batch and Background always return `(true,"")` from `TierShedAdmission` regardless of load — `sim/admission_tier_test.go`

### Implementation for User Stories 1 & 2

- [X] T012 [US1] Implement `TierShedAdmission` struct + `Admit()` method in `sim/admission.go` (depends on T001; T005–T011 must fail first)
- [X] T013 [US1] In `NewClusterSimulator()`, detect `config.AdmissionPolicy == "tier-shed"` and construct `&sim.TierShedAdmission{OverloadThreshold: config.TierShedThreshold, MinAdmitPriority: config.TierShedMinPriority}` directly (bypassing `NewAdmissionPolicy` factory) — `sim/cluster/cluster.go` (depends on T003, T004, T012)
- [X] T014 [US1] In `AdmissionDecisionEvent.Execute()`, after `cs.rejectedRequests++`, increment `cs.shedByTier[tier]++` (normalize empty SLOClass to `"standard"`) — `sim/cluster/cluster_event.go` (depends on T004, T013)

**Checkpoint**: `go test ./sim/...` passes. T005–T011 now green. Tier-shedding behavior complete.

---

## Phase 4: User Story 3 — Monotonic Shedding Invariant (Priority: P2)

**Goal**: Validate that shed counts satisfy `shed(Sheddable) ≥ shed(Standard) ≥ shed(Critical)` at every measurement point under a load ramp. Validate no-regression for simulations not using `tier-shed`.

**Independent Test**: Run a five-tier workload at increasing load steps; verify monotonicity at each step.

### Tests for User Story 3

> **Write these first — they MUST FAIL before Phase 3 implementation (T012–T014) is complete.**

- [X] T015 [US3] Invariant test: monotonic shedding order — `shed(Sheddable) ≥ shed(Standard) ≥ shed(Critical)` at each step of a load ramp — `sim/cluster/cluster_tier_test.go` (new file)
- [X] T016 [US3] Invariant test: simulation without `tier-shed` produces byte-identical stdout to pre-feature baseline (INV-6) — `sim/cluster/cluster_tier_test.go`

### Implementation for User Story 3

No additional implementation tasks — monotonic ordering is a property of the `TierShedAdmission` logic implemented in Phase 3. T015–T016 validate the composition.

**Checkpoint**: `go test ./sim/cluster/...` passes. T015–T016 now green. All invariants hold.

---

## Phase 5: Polish & Cross-Cutting Concerns

- [X] T017 [P] Run `go test ./... -count=1 -race` and fix any data-race failures
- [X] T018 [P] Run `golangci-lint run ./...` and fix all warnings

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (T001–T004)
    └──► Phase 3 US1/US2 (T005–T014)   ← #809
              └──► Phase 4 US3 (T015–T016)
                        └──► Phase 5 Polish (T017–T018)
```

### User Story Dependencies

| Story | Depends on | Notes |
|-------|-----------|-------|
| US1/US2 (Phase 3) | Phase 1 (T001–T004) | Tests written in parallel with foundation |
| US3 (Phase 4) | Phase 3 complete (T012–T014) | Invariant tests verify full composition |

### Within Each Phase

1. Tests written first → confirmed FAILING (compile or assertion failure)
2. Implementation tasks run → tests turn GREEN
3. `go test ./...` + `golangci-lint` clean
4. PR ready for review

### Parallel Opportunities

- **Phase 1**: T003 (`deployment.go`) and T004 (`cluster.go`) are fully parallel [P]
- **Phase 3 tests**: T005–T011 are all parallel [P] (same file, no dependencies between them)
- **Phase 5**: T017 and T018 are parallel [P]

---

## Parallel Execution Example

```text
Phase 1:
  T001 (admission.go: SLOTierPriority)
  T002 (bundle.go: register tier-shed)
  T003 ──┐ parallel
  T004 ──┘

Phase 3 tests (all parallel, write + confirm failing):
  T005 ─┐
  T006  │
  T007  │ all in sim/admission_tier_test.go
  T008  │
  T009  │
  T010  │
  T011 ─┘
  T015 ─┐ sim/cluster/cluster_tier_test.go
  T016 ─┘

Phase 3 implementation (sequential — same files):
  T012 → T013 → T014
```

---

## Implementation Strategy

### Full PR #809

1. Phase 1: T001–T004 (foundation, ~10 min)
2. Phase 3 tests: T005–T011, T015–T016 (write failing tests, ~20 min)
3. Phase 3 impl: T012–T014 (make tests pass, ~25 min)
4. **VALIDATE**: `go test ./sim/... ./sim/cluster/...` green, `golangci-lint` clean
5. Phase 5: T017–T018 (polish, ~5 min)
6. Open PR #809

| Phase | Est. new lines | New test files |
|-------|---------------|----------------|
| Foundation | ~41 | — |
| US1/US2 tests | ~80 | `admission_tier_test.go` |
| US1/US2 impl | ~55 | — |
| US3 tests | ~40 | `cluster_tier_test.go` |
| **Total** | **~216** | 2 new test files |

---

## Notes

- `[P]` = different files, no inter-task dependency → safe to implement simultaneously
- TDD order is non-negotiable per `docs/contributing/standards/principles.md`: test MUST fail before implementation
- `SLOTierPriority()` (T001) lives in `sim/` (not `sim/cluster/`) — exported so `sim/cluster/` can call it without circular import
- Do NOT add `case "tier-shed":` to `NewAdmissionPolicy()` factory — factory is float64-parameterized and cannot carry the int fields `TierShedAdmission` requires (research.md D-2)
- `shedByTier` is a runtime-mutated metrics map (not a validation map) — R8 applies to validation maps only; field is correctly unexported
