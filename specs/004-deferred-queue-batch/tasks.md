# Tasks: Deferred Queue for Batch and Background Requests

**Input**: Design documents from `/specs/004-deferred-queue-batch/`
**Branch**: `004-deferred-queue-batch` | **Sub-issue**: #810 | **Depends on**: #825 merged

**Tests**: TDD is NON-NEGOTIABLE per Constitution Principle IV. All test tasks MUST be written and confirmed FAILING before any implementation task begins.

**Organization**: Tasks grouped by user story. US1 (core deferral) is the MVP. US2 (real-time isolation) is a behavioral property of US1's correct implementation. US3 (conservation metrics) adds the CLI output.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)

---

## Phase 1: Setup

**Purpose**: Create test file scaffolding so all tests can be written in Phase 2.

- [X] T001 Create `sim/cluster/cluster_deferred_test.go` with package declaration `package cluster`, required imports (`fmt`, `math`, `testing`, `sim`), and helper `newDeferredTestRequests(n int, sloClass string) []*sim.Request` (arrivals every 10µs starting at t=0, InputTokens: 50 tokens, OutputTokens: 20 tokens)

---

## Phase 2: Foundational — Write ALL Tests (TDD: RED Phase)

**Purpose**: Write every behavioral contract test before any implementation. Each test MUST be run and confirmed FAILING (`go test ./sim/cluster/... -run TestDeferred -v`) before Phase 3 begins.

**⚠️ CRITICAL**: Do NOT write any implementation code until all tests in this phase are written AND confirmed failing.

- [X] T002 [US1] Write `TestDeferredQueue_BatchDeferredWhenBusy` in `sim/cluster/cluster_deferred_test.go` — Table-driven over `{"batch", "background"}`: GIVEN a 1-instance cluster overloaded with 30 standard requests (busy), WHEN a batch or background request arrives, THEN `cs.RejectedRequests() == 0` (not rejected, deferred) AND `completed + DeferredQueueLen() >= 31` (request reached a terminal state, not silently lost). Verify with `go test ./sim/cluster/... -run TestDeferredQueue_BatchDeferredWhenBusy` → must FAIL with compile or assertion error before impl

- [X] T003 [US1] Write `TestDeferredQueue_BatchAdmittedWhenIdle` in `sim/cluster/cluster_deferred_test.go` — GIVEN an idle cluster (no in-flight requests), WHEN a batch request arrives alone, THEN it is admitted normally (completes in metrics, `DeferredQueueLen() == 0`). Verify FAIL before impl

- [X] T004 [US1] Write `TestDeferredQueue_DeferredPromotedAfterIdle` in `sim/cluster/cluster_deferred_test.go` — GIVEN 5 standard requests and 5 batch requests (10 total), WHEN `Run()` completes, THEN conservation holds (all 10 accounted for), `RejectedRequests() == 0`, and `CompletedRequests == 10` (all requests complete after promotion fires — a deleted `promoteDeferred()` would leave 5 deferred and fail this assertion). Verify FAIL before impl

- [X] T005 [US2] Write `TestDeferredQueue_RealTimeUnaffected` in `sim/cluster/cluster_deferred_test.go` — GIVEN two identical runs (same seed, same critical requests): run A has only critical requests, run B has critical + batch requests, WHEN both `Run()` calls complete, THEN `AggregatedMetrics().CompletedRequests` for critical tier is identical across both runs (batch traffic does not affect real-time completion count). Verify FAIL before impl

- [X] T006 [US3] Write `TestDeferredQueue_INV1_Conservation` in `sim/cluster/cluster_deferred_test.go` — GIVEN 10 standard + 5 batch requests and `Horizon = 500` (short enough to guarantee at least one batch request remains deferred), WHEN `Run()` completes, THEN: `DeferredQueueLen() > 0` (deferred-at-horizon path is actually exercised) AND `injected == completed + still_running + still_queued + shed + dropped + timed_out + rejected + DeferredQueueLen()`. Use `newTestDeploymentConfig(1)` with reduced horizon. Verify FAIL before impl

- [X] T007 [US1] Write `TestDeferredQueue_EmptyClusterAdmitsNormally` in `sim/cluster/cluster_deferred_test.go` — GIVEN a ClusterSimulator with `NumInstances: 0` (empty cluster), WHEN batch and background requests are submitted, THEN `DeferredQueueLen() == 0` after `Run()` (all admitted normally since cluster is not busy). Verify FAIL before impl

- [X] T008 [US1] Write `TestDeferredQueue_DeferredQueueLenPanicsBeforeRun` in `sim/cluster/cluster_deferred_test.go` — GIVEN a newly constructed ClusterSimulator (Run() not yet called), WHEN `DeferredQueueLen()` is called, THEN it panics with a message containing "before Run". Use `defer func() { recover() }()` pattern. Verify FAIL before impl

**Checkpoint (RED)**: Run `go test ./sim/cluster/... -run TestDeferred -v` — ALL 7 tests MUST fail (compile error or assertion failure). Do NOT proceed to Phase 3 until confirmed.

---

## Phase 3: User Story 1 — Core Deferral Logic (Priority: P1) 🎯 MVP

**Goal**: Batch and Background requests park in the deferred queue when busy, promoted when idle. `DeferredQueueLen()` accessor available for metrics.

**Independent Test**: `go test ./sim/cluster/... -run TestDeferred` — T002, T003, T004, T007, T008 should all pass after this phase.

- [X] T009 [US1] Add `deferredQueue []*sim.Request` field to `ClusterSimulator` struct in `sim/cluster/cluster.go`; add `isBusy() bool` method (iterates `c.instances`, returns true if any `inst.QueueDepth()+inst.BatchSize()+c.inFlightRequests[string(inst.ID())] > 0`); add `promoteDeferred()` method (heap.Push each entry as `ClusterArrivalEvent{time: c.clock}` with `c.nextSeqID()`, then `c.deferredQueue = c.deferredQueue[:0]`); add `DeferredQueueLen() int` accessor (panics if `!c.hasRun`, returns `len(c.deferredQueue)`)

- [X] T010 [US1] Add idle-capacity check at the end of the main event loop body in `ClusterSimulator.Run()` in `sim/cluster/cluster.go`: after the `if clusterTime <= instanceTime { ... } else { ... }` block, add `if len(c.deferredQueue) > 0 && !c.isBusy() { c.promoteDeferred() }`

- [X] T011 [US1] Add pre-admission deferral intercept at the TOP of `AdmissionDecisionEvent.Execute()` in `sim/cluster/cluster_event.go`, BEFORE `buildRouterState()`: `if (e.request.SLOClass == "batch" || e.request.SLOClass == "background") && cs.isBusy() { cs.deferredQueue = append(cs.deferredQueue, e.request); return }` — add comment citing Phase 1B-1b and BC-D1

**Checkpoint (GREEN — US1)**: Run `go test ./sim/cluster/... -run TestDeferred -v` — T002, T003, T004, T007, T008 MUST pass. T005, T006 may still fail (need metrics wiring). Run `go test ./sim/cluster/...` to confirm no regressions.

---

## Phase 4: User Story 2 — Real-Time Requests Unaffected (Priority: P2)

**Goal**: Verify p50/p99 for Critical/Standard requests is identical with and without Batch traffic.

**Independent Test**: `go test ./sim/cluster/... -run TestDeferredQueue_RealTimeUnaffected` — T005 MUST pass.

**Note**: No new implementation required. US2 is a behavioral property of the correct US1 implementation — the deferral intercept fires before admission, so batch requests never consume instance queue slots while the cluster is busy. T005 verifies this property. If T005 fails after Phase 3, diagnose the deferral intercept placement.

**Checkpoint (GREEN — US2)**: `go test ./sim/cluster/... -run TestDeferredQueue_RealTimeUnaffected` MUST pass.

---

## Phase 5: User Story 3 — Request Conservation Metrics (Priority: P3)

**Goal**: `DeferredHorizonInterrupted` count appears in metrics output; INV-1 holds.

**Independent Test**: `go test ./sim/cluster/... -run TestDeferredQueue_INV1` — T006 MUST pass. CLI prints `Deferred (horizon-interrupted): N` when non-zero.

- [X] T012 [US3] Add `DeferredHorizonInterrupted int` field to `RawMetrics` struct in `sim/cluster/metrics.go` with comment: `// DeferredHorizonInterrupted: Batch/Background requests still deferred at horizon (Phase 1B-1b). INV-1 extended: injected == completed + running + queued + shed + dropped + deferred_horizon_interrupted`

- [X] T013 [P] [US3] Wire `rawMetrics.DeferredHorizonInterrupted = cs.DeferredQueueLen()` in `cmd/root.go` after the `rawMetrics.ShedByTier = cs.ShedByTier()` line; add `|| rawMetrics.DeferredHorizonInterrupted > 0` to the anomaly block condition; add `if rawMetrics.DeferredHorizonInterrupted > 0 { fmt.Printf("Deferred (horizon-interrupted): %d\n", rawMetrics.DeferredHorizonInterrupted) }` inside the anomaly block

- [X] T014 [P] [US3] Wire `rawMetrics.DeferredHorizonInterrupted = cs.DeferredQueueLen()` in `cmd/replay.go` after the `rawMetrics.ShedByTier = cs.ShedByTier()` line; add same anomaly condition and print as T013

**Checkpoint (GREEN — US3)**: Run `go test ./sim/cluster/... -run TestDeferredQueue_INV1` — T006 MUST pass. Run `go test ./...` — all packages green.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Verification gate before commit.

- [X] T015 Run `go build ./...` — must produce zero errors or warnings

- [X] T016 Run `go test ./... -count=1` — all packages must pass; confirm total runtime under 60 seconds

- [X] T017 Run `golangci-lint run ./...` — zero lint errors or warnings

- [X] T018 Validate quickstart.md Scenario 1 manually: `./blis run --model qwen/qwen3-14b` with a mixed workload YAML containing batch requests; confirm output shows no "Deferred (horizon-interrupted)" line (all batch requests complete in the default horizon)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 (tests written and confirmed RED)
- **US2 (Phase 4)**: Depends on Phase 3 complete — behavioral verification only, no new impl
- **US3 (Phase 5)**: Depends on Phase 3 (DeferredQueueLen() must exist); T013 and T014 can run in parallel [P]
- **Polish (Phase 6)**: Depends on all story phases complete

### Within Each Phase

- Tests MUST be written and confirmed FAILING before implementation
- T009, T010, T011 must be done in order (T009 defines isBusy/promoteDeferred needed by T010 and T011)
- T013 and T014 are [P] — they touch different files

---

## Parallel Opportunities

```
Phase 1 → Phase 2 (sequential: test scaffold before tests)
Phase 2 → Phase 3 (sequential: RED before GREEN)
Phase 3 → Phase 4 (sequential: verify after impl)
Phase 3 → Phase 5 T012 (can start T012 in parallel with Phase 4 verification)
Phase 5: T013 ‖ T014 (parallel: cmd/root.go and cmd/replay.go are independent files)
Phase 6: T015 ‖ T016 ‖ T017 (can run in parallel as independent checks)
```

---

## Implementation Strategy

### MVP (User Story 1 Only — Phases 1–3)

1. Phase 1: Create test file
2. Phase 2: Write all 7 tests, confirm RED
3. Phase 3: Implement core deferral (T009–T011)
4. **STOP**: Confirm T002–T004, T007, T008 pass; run `go test ./sim/cluster/...` for no regressions
5. US1 is independently deliverable at this point

### Full Delivery (All Stories — Phases 1–6)

1. MVP (above)
2. Phase 4: Verify T005 (no new code)
3. Phase 5: Add metrics field + CLI wiring (T012–T014)
4. Phase 6: Verification gate

---

## Notes

- [P] tasks = touch different files, no blocking dependencies
- All test tasks must be confirmed FAILING before their implementation tasks begin (Constitution IV)
- `DeferredQueueLen()` is the only new exported method — gated on `hasRun` (panics otherwise)
- No new YAML config fields; no new Go interfaces
- The deferral intercept fires BEFORE `buildRouterState()` — deferred requests never reach `admissionPolicy.Admit()`
- R4 check: `deferredQueue` field is nil-initialized; no struct literal construction site to update (nil slice is the correct zero value)
- R21 check: `promoteDeferred()` uses `range c.deferredQueue` — slice cannot shrink during this method (cleared atomically after loop)
