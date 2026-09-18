# Tasks: Phase 1A — Infrastructure: Nodes, GPUs, Instances

**Input**: Design documents from `/specs/001-infra-nodes-gpus-instances/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Included per BLIS constitution (BDD/TDD is mandatory — write tests before implementation, ensure they fail first).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1–US5)
- All file paths are relative to repository root

---

## Phase 1: Setup (Construction-Site Audit)

**Purpose**: Pre-flight audits required before adding fields to shared types (R4). No code written yet.

- [x] T001 Grep for all `Request{` and `sim.Request{` literals across the codebase to audit construction sites before adding `Model` field — run: `grep -rn "Request{" --include="*.go" .`
- [x] T002 Grep for all `RoutingSnapshot{` literals to audit construction sites before adding `Model` field — run: `grep -rn "RoutingSnapshot{" --include="*.go" .`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared types and config that ALL user stories depend on. Must be complete before any US phase begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T003 Add `Model string` field to `sim.Request` in `sim/request.go` (R4: update all construction sites found in T001; zero value `""` = single-model backward-compatible)
- [x] T004 Add `Model string` field to `sim.RoutingSnapshot` in `sim/routing.go` (R4: update all construction sites found in T002)
- [x] T005 [P] Create `NodeState` and `InstanceState` enums with `IsValidNodeState()` / `IsValidInstanceState()` accessors (unexported validation maps per R8) in `sim/cluster/infra_node.go`
- [x] T006 [P] Create `NodePoolConfig` struct with `IsValid()` method and `NewNodePoolConfig()` factory (validates GPUsPerNode ≥1, GPUMemoryGiB >0, InitialNodes ≤ MaxNodes, MinNodes ≤ MaxNodes, DistSpec per R3) in `sim/cluster/infra_config.go`
- [x] T007 [P] Create `InstanceLifecycleConfig` struct with `DrainPolicy` enum, `IsValidDrainPolicy()` accessor, and validation (WarmUpTTFTFactor ≥ 1.0 per R3) in `sim/cluster/infra_config.go`
- [x] T008 Create `Node` and `GPU` struct types in `sim/cluster/infra_node.go` (Node: ID, PoolName, GPUType, TotalGPUs, GPUs slice, State, CostStartTime; GPU: ID, NodeID, PoolName, Type, MemoryGiB, AllocatedTo InstanceID)
- [x] T009 Add `NodePools []NodePoolConfig` and `InstanceLifecycle InstanceLifecycleConfig` fields to `DeploymentConfig` in `sim/cluster/deployment.go` (R4: check all DeploymentConfig{} construction sites)
- [x] T010 Register `"node-provisioning"` and `"instance-loading"` subsystem names in `PartitionedRNG` initialization in `sim/cluster/cluster.go` (INV-6: deterministic RNG subsystems)

**Checkpoint**: Foundational types compiled — user story implementation can now begin.

---

## Phase 3: User Story 1 — Configure Node-Pool Infrastructure Topology (Priority: P1) 🎯 MVP

**Goal**: A simulation configured with N node pools materializes concrete nodes and GPUs with unique IDs. Each GPU is traceable to its parent node and pool.

**Independent Test**: A simulation with two node pools (H100 ×8, 2 nodes; A100 ×4, 1 node) produces 16 H100 GPUs and 4 A100 GPUs, all with distinct IDs. `PlacementManager.VerifyConservation()` returns nil.

### Tests for User Story 1 (write first, verify they FAIL before T015)

- [x] T011 [P] [US1] Write table-driven test: two-pool initialization produces correct node count and GPU inventory, GPU IDs unique and traceable to pool/node in `sim/cluster/infra_node_test.go`
- [x] T012 [P] [US1] Write invariant test: `VerifyConservation()` returns nil for freshly initialized pool; returns error when GPU count manually corrupted in `sim/cluster/infra_node_test.go`
- [x] T013 [P] [US1] Write test: `InitialNodes: 0` produces empty pool with zero nodes and zero GPUs in `sim/cluster/infra_node_test.go`
- [x] T014 [P] [US1] Write test: multi-pool GPUs never share ID prefix — cross-pool ID uniqueness in `sim/cluster/infra_node_test.go`

### Implementation for User Story 1

- [x] T015 [US1] Implement `PlacementManager` struct with `NewPlacementManager(pools []NodePoolConfig, rng *sim.PartitionedRNG)` — materializes `Node` and `GPU` instances from `NodePools` at construction time; assigns deterministic IDs `"{pool-name}-{index}"` / `"{node-id}-gpu-{index}"` in `sim/cluster/infra_placement.go`
- [x] T016 [US1] Implement `PlacementManager.VerifyConservation() error` — iterates all nodes, checks `allocated + free == total`, uses sorted node ID iteration (R2) in `sim/cluster/infra_placement.go`
- [x] T017 [US1] Implement `PlacementManager.NodeCount() int` and `PlacementManager.GPUCount(poolName string) int` observation methods in `sim/cluster/infra_placement.go`
- [x] T018 [US1] Integrate `PlacementManager` construction into `NewClusterSimulator()` when `cfg.NodePools` is non-empty; no-op when empty (backward-compatible mode) in `sim/cluster/cluster.go`

**Checkpoint**: Run `go test ./sim/cluster/... -run TestNodePool` — all US1 tests should pass.

---

## Phase 4: User Story 2 — Place Instances onto Nodes Using Bin-Packing (Priority: P1)

**Goal**: Instance creation triggers placement onto a node with sufficient free GPUs of the correct type. Requests for TP=4 on an 8-GPU node yield 2 placed + 1 pending.

**Independent Test**: Requesting 3 instances of TP=4 on a single 8-GPU node results in 2 instances placed and 1 instance in `Scheduling` (pending) state; node shows 0 free GPUs; `VerifyConservation()` passes.

### Tests for User Story 2 (write first, verify they FAIL before T023)

- [x] T019 [P] [US2] Write test: 2 TP=4 instances on 8-GPU node → both placed (GPUs 0–3 and 4–7), node has 0 free GPUs, INV-A holds in `sim/cluster/infra_placement_test.go`
- [x] T020 [P] [US2] Write test: 8-GPU node fully allocated, new TP=4 request → `PlaceInstance()` returns error, instance stays in `Scheduling` state in `sim/cluster/infra_placement_test.go`
- [x] T021 [P] [US2] Write test: place then `ReleaseInstance()` → GPUs returned to free pool, `FreeGPUCount` = original, INV-A holds in `sim/cluster/infra_placement_test.go`
- [x] T022 [P] [US2] Write test: H100-type instance not placed on A100 pool node — GPU type constraint enforced in `sim/cluster/infra_placement_test.go`

### Implementation for User Story 2

- [x] T023 [US2] Add `Model string`, `State InstanceState`, `nodeID string`, `allocatedGPUIDs []string`, `warmUpRemaining int` fields to `InstanceSimulator` in `sim/cluster/instance.go` (R4: check all InstanceSimulator{} literals)
- [x] T024 [US2] Implement `PlacementManager.PlaceInstance(id InstanceID, model, gpuType string, tpDegree int) (nodeID string, gpuIDs []string, err error)` — first-fit bin-packing within matching pool, select-then-commit atomicity (R5), returns error when no capacity in `sim/cluster/infra_placement.go`
- [x] T025 [US2] Implement `PlacementManager.ReleaseInstance(id InstanceID) error` — marks GPUs as free, validates INV-A via `VerifyConservation()` after release in `sim/cluster/infra_placement.go`
- [x] T026 [US2] Implement `PlacementManager.FreeGPUCount(nodeID string) int` — returns count of free GPUs for a node, uses sorted map iteration (R2) in `sim/cluster/infra_placement.go`
- [x] T027 [US2] Integrate `PlaceInstance()` into `NewClusterSimulator()` instance loop — set `InstanceState.Loading` on success, `InstanceState.Scheduling` when no capacity; call `ReleaseInstance()` in `ClusterSimulator` teardown in `sim/cluster/cluster.go`

**Checkpoint**: Run `go test ./sim/cluster/... -run TestPlacement` — all US2 tests should pass. US1 tests remain green.

---

## Phase 5: User Story 3 — Model Node Provisioning and Termination Lifecycle (Priority: P2)

**Goal**: Nodes transition through lifecycle states with configurable provisioning delays. Instances cannot be placed on provisioning nodes. Cost accrues from Provisioning through Draining.

**Independent Test**: Adding a node with a 120s provisioning delay means instances cannot be placed on it for 120 simulation seconds.

### Tests for User Story 3 (write first, verify they FAIL before T031)

- [x] T028 [P] [US3] Write test: `ProvisionNode()` at clock T schedules `NodeReadyEvent` at T+120s (constant delay); node in `Provisioning` state blocks placement until event fires in `sim/cluster/infra_node_test.go`
- [x] T029 [P] [US3] Write test: `Ready` node with zero instances → drain command transitions immediately to `Draining` then `Terminated`, GPUs removed from free pool in `sim/cluster/infra_node_test.go`
- [x] T030 [P] [US3] Write test: `Draining` node with 1 active instance → transitions to `Terminated` only after instance terminates (not before) in `sim/cluster/infra_node_test.go`

### Implementation for User Story 3

- [x] T031 [US3] Create `NodeReadyEvent` type implementing `ClusterEvent` interface (Priority 0, Execute: transition node Provisioning→Ready, call `retryPendingInstances()`) in `sim/cluster/infra_lifecycle_event.go`
- [x] T032 [US3] Implement `PlacementManager.ProvisionNode(poolName string, clock float64) (*Node, float64)` — creates new `Node` in Provisioning state, samples delay from DistSpec via PartitionedRNG, returns node + ready-time; caller schedules `NodeReadyEvent` in `sim/cluster/infra_placement.go`
- [x] T033 [US3] Implement `retryPendingInstances()` in `PlacementManager` — iterates pending instances (index-based per R21), attempts `PlaceInstance()` on newly-ready node; bounded loop with max-iter = len(pending) guard (R19) in `sim/cluster/infra_placement.go`
- [x] T034 [US3] Create `NodeDrainedEvent` type (Priority 0, Execute: transition Draining→Terminated, release all remaining GPU inventory) in `sim/cluster/infra_lifecycle_event.go`
- [x] T035 [US3] Implement `PlacementManager.DrainNode(nodeID string, clock float64)` — transitions Ready→Draining; if no instances, schedules `NodeDrainedEvent` immediately; else registers drain-on-last-instance-terminated callback in `sim/cluster/infra_placement.go`

**Checkpoint**: Run `go test ./sim/cluster/... -run TestNode` — all US3 tests pass. US1+US2 remain green.

---

## Phase 6: User Story 4 — Model Instance Startup Phases Including Warm-Up (Priority: P2)

**Goal**: Instances transition Loading→WarmingUp→Active with configurable delays. Only Active instances receive requests. The warm-up TTFT penalty is measurable for the first N requests.

**Independent Test**: A newly-Active instance's first-request TTFT is `WarmUpTTFTFactor` × a warm instance's TTFT for the same request profile.

### Tests for User Story 4 (write first, verify they FAIL before T040)

- [x] T036 [P] [US4] Write test: instance with 60s loading delay is not included in `buildRouterState()` snapshots before T+60s; included after `InstanceLoadedEvent` fires in `sim/cluster/instance_lifecycle_test.go`
- [x] T037 [P] [US4] Write test: first N requests to newly-Active instance have TTFT = raw × factor; request N+1 has normal TTFT — warm-up penalty measurable in `sim/cluster/instance_lifecycle_test.go`
- [x] T038 [P] [US4] Write test: WAIT drain policy — instance excluded from `buildRouterState()` during drain; in-flight requests complete; instance reaches Terminated in `sim/cluster/instance_lifecycle_test.go`
- [x] T039 [P] [US4] Write test: REDIRECT drain policy — queued (not yet scheduled) requests from draining instance are re-injected as new `ClusterArrivalEvent`s and served by other Active instances in `sim/cluster/instance_lifecycle_test.go`

### Implementation for User Story 4

- [x] T040 [US4] Add `IsRoutable() bool`, `IsWarmingUp() bool`, `ConsumeWarmUpRequest()`, `TransitionTo(state InstanceState)` methods to `InstanceSimulator` — `IsRoutable()` returns true for `Active` and `WarmingUp`; `TransitionTo()` panics on invalid transition in `sim/cluster/instance.go`
- [x] T041 [US4] Create `InstanceLoadedEvent` (Priority 1, Execute: transition Loading→WarmingUp) and `InstanceWarmUpCompleteEvent` (Priority 1, Execute: WarmingUp→Active); schedule LoadedEvent at construction time based on `LoadingDelay` DistSpec in `sim/cluster/infra_lifecycle_event.go`
- [x] T042 [US4] Apply warm-up TTFT factor in `ClusterSimulator` completion recording — when `inst.IsWarmingUp()`: multiply recorded TTFT by `WarmUpTTFTFactor`, then call `inst.ConsumeWarmUpRequest()` (apply factor consistently in all TTFT recording paths per R23) in `sim/cluster/cluster.go`
- [x] T043 [US4] Implement `DrainImmediate`, `DrainWait`, `DrainRedirect` as three `DrainPolicy` interface implementations (satisfying R13: ≥2 backends) — REDIRECT extracts pending requests from instance WaitQ and re-injects as `ClusterArrivalEvent`s in `sim/cluster/infra_lifecycle_event.go`
- [x] T044 [US4] Update `buildRouterState()` to filter by `inst.IsRoutable()` — only instances in Active or WarmingUp state appear in routing snapshots in `sim/cluster/cluster_event.go`

**Checkpoint**: Run `go test ./sim/cluster/... -run TestInstanceLifecycle` — all US4 tests pass. All prior phases remain green.

---

## Phase 7: User Story 5 — Route Requests to Active Instances of the Correct Model (Priority: P2)

**Goal**: Requests carry a model identifier and are routed only to Active instances of that model. Per-model metrics appear in the output JSON.

**Independent Test**: Cluster with 2 llama and 2 qwen instances; after 100 requests per model, `result.PerModel["meta-llama/Llama-3.1-8B"]` and `result.PerModel["qwen/Qwen3-14B"]` both present in output with distinct TTFT/E2E distributions.

### Tests for User Story 5 (write first, verify they FAIL before T048)

- [x] T045 [P] [US5] Write test: llama request routes only to llama instances — qwen instance snapshots absent from RouterState in `sim/cluster/cluster_test.go`
- [x] T046 [P] [US5] Write test: all instances of model M are non-Active → request for M receives empty RouterState (admission policy rejects or request queues) in `sim/cluster/cluster_test.go`
- [x] T047 [P] [US5] Write test: 100 llama + 100 qwen requests → `PerModel` map in output contains both model keys with correct TotalRequests counts and non-nil distributions in `sim/cluster/metrics_test.go`

### Implementation for User Story 5

- [x] T048 [US5] Filter `buildRouterState()` by `req.Model` — set `RoutingSnapshot.Model` from `inst.Model`; include snapshot only when `snapshot.Model == req.Model` (or both empty for backward-compat) in `sim/cluster/cluster_event.go`
- [x] T049 [US5] Create `ModelMetrics` struct (TTFT Distribution, E2E Distribution, ThroughputRPS, ThroughputTokensPerSec, TotalRequests) in `sim/cluster/metrics.go`
- [x] T050 [US5] Implement per-model request partitioning in `CollectRawMetrics()` — group completed requests by `req.Model`, compute `ModelMetrics` per group, collect into `map[string]*ModelMetrics` in `sim/cluster/metrics.go`
- [x] T051 [US5] Add `PerModelMetrics map[string]*ModelMetrics` to `EvaluationResult` in `sim/cluster/evaluation.go`; update JSON serialization to include `"per_model"` key in output

**Checkpoint**: Run `go test ./sim/cluster/... -run TestMultiModel` — all US5 tests pass. Full test suite: `go test ./...` green.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Verification, backward-compatibility check, golden dataset, documentation validation.

- [x] T052 Run `go test ./... -count=1` and verify all pre-existing tests pass with the new `Model=""` zero-value addition to `Request` and `RoutingSnapshot` (backward-compatibility smoke test)
- [x] T053 [P] Run `golangci-lint run ./...` and fix all lint violations introduced by new code (R6: no linter warnings in `sim/` packages)
- [x] T054 [P] Check if JSON output format changed (new `per_model` key); if golden dataset `testdata/goldendataset.json` is affected, regenerate and document the command (R12)
- [x] T055 Validate `quickstart.md` accuracy — run the example YAML through the new YAML parser and confirm no parse errors; spot-check that output contract matches actual JSON output

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup/Audit) → Phase 2 (Foundational) → Phases 3–7 (User Stories) → Phase 8 (Polish)
```

- **Phase 1** (T001–T002): No dependencies. Start immediately.
- **Phase 2** (T003–T010): Depends on Phase 1 completion. **Blocks all user stories.**
- **Phase 3** (T011–T018): Depends on Phase 2. US1 is a prerequisite for US2 (PlacementManager).
- **Phase 4** (T019–T027): Depends on Phase 3 (PlacementManager must exist).
- **Phase 5** (T028–T035): Depends on Phase 2. Independent of US1/US2 (node lifecycle is separate concern).
- **Phase 6** (T036–T044): Depends on Phase 4 (InstanceSimulator state machine built on top of placement).
- **Phase 7** (T045–T051): Depends on Phase 6 (routing filter uses `IsRoutable()`).
- **Phase 8** (T052–T055): Depends on all user story phases.

### User Story Dependencies

```
US1 (P1) → US2 (P1) → US4 (P2) → US5 (P2)
                  ↑
US3 (P2) ─────────┘  (US3 adds provisioning events; US4 adds instance startup events)
```

- **US1 (P1)**: Can start after Phase 2 completes.
- **US2 (P1)**: Depends on US1 (PlacementManager must be instantiated).
- **US3 (P2)**: Depends on Phase 2 only; can proceed in parallel with US1/US2.
- **US4 (P2)**: Depends on US2 (InstanceSimulator state machine extends placement).
- **US5 (P2)**: Depends on US4 (routing filter uses `IsRoutable()` from US4).

### Within Each User Story

1. Write ALL tests first — verify they fail (no implementation yet)
2. Implement in task order (types → logic → integration)
3. Verify tests pass before proceeding to next phase

### Parallel Opportunities

Within each phase, all `[P]`-marked tasks touch different files and can run concurrently:
- Phase 2: T005, T006, T007 can run in parallel (separate files)
- US1 tests (T011–T014) can all run in parallel
- US2 tests (T019–T022) can all run in parallel
- US4 tests (T036–T039) can all run in parallel
- US5 tests (T045–T047) can all run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests in parallel (all touch infra_node_test.go sections):
Agent A: T011 — two-pool initialization test
Agent B: T012 — VerifyConservation invariant test
Agent C: T013 — zero-node pool test
Agent D: T014 — cross-pool GPU uniqueness test

# Then implement (sequential after tests fail):
T015 → T016 → T017 → T018
```

## Parallel Example: User Story 4

```bash
# Launch all US4 tests in parallel (all touch instance_lifecycle_test.go sections):
Agent A: T036 — loading delay routing exclusion test
Agent B: T037 — warm-up TTFT penalty test
Agent C: T038 — WAIT drain policy test
Agent D: T039 — REDIRECT drain policy test

# Then implement sequentially (T040 → T041 → T042 → T043 → T044):
T040 → T041 → T042 → T043 → T044
```

---

## Implementation Strategy

### MVP First (P1 User Stories Only)

1. Complete Phase 1: Setup (T001–T002) — audit construction sites
2. Complete Phase 2: Foundational (T003–T010) — shared types and config
3. Complete Phase 3: US1 (T011–T018) — node materialization
4. Complete Phase 4: US2 (T019–T027) — bin-packing placement
5. **STOP and VALIDATE**: `go test ./sim/cluster/... -run TestNodePool -run TestPlacement` — all US1+US2 tests green; GPU conservation invariant holds

### Incremental Delivery

1. Setup + Foundational → types compile
2. US1 → node/GPU topology works independently
3. US2 → placement + pending state works
4. US3 → provisioning lifecycle works
5. US4 → instance startup and warm-up works
6. US5 → multi-model routing + per-model metrics works
7. Polish → full suite green, lint clean

### Solo Strategy (Sequential)

Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8
(TDD: write failing tests for each phase before implementing)

---

## Notes

- `[P]` tasks touch different files and have no intra-phase dependencies — safe to parallelize
- `[Story]` label maps each task to its user story for traceability
- **BDD/TDD is mandatory** per BLIS constitution: tests MUST be written and FAIL before implementation
- Commit after each completed phase (or after each test+impl pair)
- `go test ./... -count=1` after every phase before moving on
- Backward compatibility: all existing tests must remain green throughout (Model="" zero value is safe)
- GPU conservation invariant (INV-A) companion test required for every placement golden test (R7)
