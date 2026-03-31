# Tasks: Defer Instance Construction Until After Placement

**Input**: Design documents from `specs/006-defer-instance-placement/`
**Branch**: `006-defer-instance-placement`

**Organization**: Tasks are grouped by user story. US3 (dynamic registration) is implemented before US2 (deferred path) because it is a direct technical prerequisite — US2 cannot complete without US3's `AddInstance` method.

**TDD**: All implementation tasks are preceded by a failing test task (BDD/TDD — Constitution Principle IV, non-negotiable).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other [P] tasks in the same phase
- **[Story]**: US1/US2/US3 maps to spec.md user stories
- Exact file paths included in every task

---

## Phase 1: Setup

**Purpose**: No project initialization needed — this fix modifies existing files within `sim/cluster/`. Phase 1 is empty.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: `PlaceInstance` signature change and `pendingInstance`/`placedInstance` struct extensions. These are the lowest-level changes that every user story phase depends on. No user story work can begin until this phase is complete.

**⚠️ CRITICAL**: Phases 3–5 all block on this phase.

- [ ] T001 Write failing test `TestPlaceInstance_ReturnsMatchedPoolGPUType` (table-driven: pool `gpu_type` matches, mismatches, error path — assert returned `matchedGPUType` equals pool's own value) in `sim/cluster/infra_placement_test.go`
- [ ] T002 Extend `PlaceInstance` signature in `sim/cluster/infra_placement.go`: add `matchedGPUType string` as third return value; set it to `poolState.config.GPUType` on success path; return `""` on all error paths
- [ ] T003 [P] Write failing test `TestRetryPendingInstances_PlacedInstanceHasGPUType` (assert `placedInstance.gpuType` equals the pool's GPU type after retry succeeds) in `sim/cluster/infra_placement_test.go`
- [ ] T004 [P] Add `gpuType string` field to `placedInstance` struct in `sim/cluster/infra_placement.go`; update `RetryPendingInstances` to capture `matchedGPUType` from `PlaceInstance` and populate `placedInstance.gpuType`
- [ ] T005 Write failing test `TestAddPending_StoresSimCfg` (call `AddPending` with a non-zero `simCfg`; verify it is accessible through the retry path) in `sim/cluster/infra_placement_test.go`
- [ ] T006 Add `simCfg sim.SimConfig` field to `pendingInstance` struct in `sim/cluster/infra_placement.go`; add `simCfg sim.SimConfig` parameter to `AddPending`; store in literal; add `simCfg sim.SimConfig` field to `placedInstance` struct; propagate from `pendingInstance` through `RetryPendingInstances`
- [ ] T007 Update all existing `PlaceInstance(...)` call sites in `sim/cluster/infra_placement_test.go` and `sim/cluster/infra_placement.go` to capture or discard the new `matchedGPUType` return value; run `go test ./sim/cluster/... -run TestPlace` to confirm green

**Checkpoint**: `go test ./sim/cluster/... -run TestPlace -run TestRetry -run TestAddPending` passes. Foundation ready — user story phases may begin.

---

## Phase 3: User Story 1 — Accurate Roofline Latency for Synchronous Placement (Priority: P1) 🎯 MVP

**Goal**: When `NodePools` are configured, every instance placed synchronously at startup has its latency model initialized from the pool's `gpu_type`, not the CLI `--gpu` flag. When `NodePools` is empty, behavior is byte-identical to the current behavior (INV-6).

**Independent Test**: `go test ./sim/cluster/... -run TestNewClusterSimulator_UsesPoolGPUType` passes; `go test ./sim/cluster/... -run TestNewClusterSimulator_NoNodePools_DeterminismPreserved` passes.

- [ ] T008 [US1] Write failing test `TestNewClusterSimulator_UsesPoolGPUType` (GIVEN `NodePools` with `gpu_type: A100` and `config.GPU = "H100"`, WHEN `NewClusterSimulator` runs, THEN each placed instance's `LatencyModel` step-time reflects A100 hardware coefficients — assert via observable latency output, not internal field access) in `sim/cluster/cluster_test.go`
- [ ] T009 [US1] Write failing test `TestNewClusterSimulator_NoNodePools_DeterminismPreserved` (GIVEN `NodePools` empty and same seed as a reference run, WHEN `NewClusterSimulator` runs, THEN simulation output metrics are byte-identical to the pre-refactor baseline — use the existing determinism test pattern) in `sim/cluster/cluster_test.go`
- [ ] T010 [US1] Implement unified construction loop in `sim/cluster/cluster.go`: move `NewInstanceSimulator` call to after `PlaceInstance` succeeds; set `simCfg.GPU = matchedGPUType` (pool-authoritative) before construction; handle no-NodePools path as immediate placement with `gpuType = config.GPU`; update `AddPending` call to pass `simCfg` (with `GPU` unset); remove pre-construction of instances for the pending path (instances in `pendingInsts` are not yet created)
- [ ] T011 [US1] Update `instanceMap` and `snapshotProvider` initialization in `sim/cluster/cluster.go` to reflect that pending instances are not yet constructed at startup (only placed instances enter `instanceMap` at construction time)
- [ ] T012 [US1] Update the `OnRequestDone` / `tenantTracker` callback wiring loop in `sim/cluster/cluster.go` to iterate only over instances that are actually constructed (placed instances); pending instances are wired when constructed in `NodeReadyEvent`
- [ ] T013 [US1] Update call sites of `AddPending` in `sim/cluster/cluster.go` to pass the pre-resolved `simCfg` (with `GPU` field left empty)

**Checkpoint**: `go test ./sim/cluster/... -run TestNewClusterSimulator` passes. User Story 1 is fully functional: synchronous placement uses pool GPU type; no-NodePools path is determinism-preserved.

---

## Phase 4: User Story 3 — Dynamic Instance Registration (Priority: P3)

**Goal**: The snapshot provider supports registering instances constructed after simulation startup. This is a prerequisite for User Story 2 (deferred placement). Implemented here, before US2, because US2 cannot complete without it.

**Independent Test**: `go test ./sim/cluster/... -run TestCachedSnapshotProvider_AddInstance` passes; a newly added instance appears in `Snapshot()` output.

- [ ] T014 [P] [US3] Write failing test `TestCachedSnapshotProvider_AddInstance` (GIVEN a provider with N instances, WHEN `AddInstance(newID, newInst)` is called, THEN `Snapshot(newID, clock)` returns a valid snapshot for the new instance; WHEN called again with the same ID, THEN it panics) in `sim/cluster/snapshot_test.go`
- [ ] T015 [P] [US3] Implement `AddInstance(id InstanceID, inst *InstanceSimulator)` on `CachedSnapshotProvider` in `sim/cluster/snapshot.go`: insert into `p.instances`, initialize `p.cache[id]` with `sim.NewRoutingSnapshot(string(id))`, initialize `p.lastRefresh[id]` with zero timestamps; panic if `id` already present (R1: no silent overwrite)

**Checkpoint**: `go test ./sim/cluster/... -run TestCachedSnapshotProvider_AddInstance` passes. Dynamic registration works in isolation.

---

## Phase 5: User Story 2 — Deferred Placement Uses Correct GPU Type (Priority: P2)

**Goal**: When a `NodeReadyEvent` fires and places a previously-pending instance, that instance is constructed using the matched pool's GPU type (from `placedInstance.gpuType`) — not a stale CLI value. The new instance is registered with the snapshot provider and is routable.

**Independent Test**: `go test ./sim/cluster/... -run TestNodeReadyEvent_DeferredConstruction` passes; pending instance is constructed with pool GPU type and appears in routing.

**Depends on**: Phase 4 (T015 — `AddInstance` must exist before wiring here).

- [ ] T016 [US2] Write failing test `TestNodeReadyEvent_DeferredConstruction_UsesPoolGPUType` (GIVEN a cluster with insufficient initial capacity so one instance is pending, WHEN a `NodeReadyEvent` fires at a later tick, THEN the newly constructed instance uses the pool's `gpu_type` for latency estimation — assert via observable latency output; THEN a subsequent request can be routed to the new instance) in `sim/cluster/cluster_test.go`
- [ ] T017 [US2] Rewrite `NodeReadyEvent.Execute` in `sim/cluster/infra_lifecycle_event.go`: replace the pre-constructed instance lookup loop with deferred construction — for each `placedInstance` returned by `RetryPendingInstances`, set `p.simCfg.GPU = p.gpuType`, call `NewInstanceSimulator(p.id, p.simCfg)`, wire `OnRequestDone`/`tenantTracker` callback (mirror startup path), call `cs.snapshotProvider.(*CachedSnapshotProvider).AddInstance(p.id, inst)`, append `inst` to `cs.instances`, initialize `cs.inFlightRequests[string(p.id)] = 0`, set `warmUpRemaining`, `TransitionTo(InstanceStateLoading)`, call `scheduleInstanceLoadedEvent(inst)`

**Checkpoint**: `go test ./sim/cluster/... -run TestNodeReadyEvent` passes. User Story 2 fully functional: deferred instances use pool GPU type and are routable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Caller updates, invariant verification, and the final acceptance gate.

- [ ] T018 [P] Update all remaining `AddPending(...)` call sites in test files (`sim/cluster/infra_placement_test.go`, `sim/cluster/pool_test.go`, any other test using `AddPending`) to pass a `simCfg` argument (use `sim.SimConfig{}` as zero-value where the config content is irrelevant to the test)
- [ ] T019 [P] Update all remaining `PlaceInstance(...)` call sites in test files to capture or explicitly discard (`_`) the new `matchedGPUType` return value; confirm no compile errors
- [ ] T020 Grep `sim/cluster/` for `config\.GPU` reads in the NodePools-active construction path; confirm zero occurrences after refactor (acceptance criterion SC-004); document result as a comment in `cluster.go` near the unified construction loop
- [ ] T021 Run full verification gate: `go build ./...` (exit 0), `go test ./... -count=1` (exit 0), `golangci-lint run ./...` (exit 0); fix any issues found before marking done

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 2 (Foundational)**: No dependencies — start immediately
- **Phase 3 (US1)**: Depends on Phase 2 completion
- **Phase 4 (US3)**: Depends on Phase 2 completion — can run in parallel with Phase 3
- **Phase 5 (US2)**: Depends on Phase 3 AND Phase 4 completion
- **Phase 6 (Polish)**: Depends on all phases complete

### User Story Dependencies

- **US1 (P1)**: After Phase 2 — no story dependencies
- **US3 (P3)**: After Phase 2 — no story dependencies, parallelizable with US1
- **US2 (P2)**: After US1 AND US3 — depends on both

### Within Each Phase

- Test tasks MUST be written first and MUST fail before implementation begins (BDD/TDD)
- Within Phase 2: T001→T002, T003→T004 (these two pairs parallelizable), then T005→T006, then T007
- Within Phase 3: T008+T009 (write tests in parallel) → T010→T011→T012→T013
- Within Phase 4: T014 and T015 can proceed in parallel (test + implementation, different concerns)
- Within Phase 5: T016 (test) → T017 (implementation)
- Within Phase 6: T018+T019+T020 parallelizable → T021 (gate)

### Parallel Opportunities

```
Phase 2: [T001→T002] || [T003→T004] → T005→T006 → T007
Phase 3+4 (after Phase 2): Phase 3 || Phase 4  (different files, no overlap)
Phase 5 (after Phase 3+4): T016→T017
Phase 6: [T018] || [T019] || [T020] → T021
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 2 (Foundational)
2. Complete Phase 3 (US1 — synchronous path)
3. **STOP and VALIDATE**: `go test ./sim/cluster/...` passes; grep confirms no `config.GPU` in NodePools path; existing golden tests unchanged
4. This alone fixes the most common case (synchronous placement with initial nodes)

### Full Delivery

1. Phase 2 → Phase 3 + Phase 4 (parallel) → Phase 5 → Phase 6
2. Each phase checkpoint independently verifiable
3. Total: ~21 tasks across 4 active phases

---

## Notes

- [P] tasks operate on different files or independent test functions — no coordination needed
- All test tasks must produce a RED (failing) run before the paired implementation task begins
- `go test ./sim/cluster/...` is the primary feedback loop throughout
- US3 (P3 in spec) is implemented before US2 (P2) due to technical dependency — this does not reflect priority, only construction order
- After T010, `cs.instances` at startup contains only placed instances (not pending); this is the correct behavior since unplaced instances should not appear in routing
