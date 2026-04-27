# Tasks: Phase 1C Model Autoscaler

**Branch**: `006-model-autoscaler`  
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Data model**: [data-model.md](./data-model.md) | **Contracts**: [contracts/autoscaler-interfaces.md](./contracts/autoscaler-interfaces.md)  
**Issues**: [#905](https://github.com/inference-sim/inference-sim/issues/905) (1C-1b) · [#918](https://github.com/inference-sim/inference-sim/issues/918) (1C-1d) · [#954](https://github.com/inference-sim/inference-sim/issues/954) (QueueingModelAnalyzer, future)

**Format**: `[ID] [P?] [Story?] Description — file path`  
- **[P]**: parallelizable (touches a different file, no dependency on an incomplete task)  
- **[Story]**: US1–US5 maps to user stories in spec.md  
- Tests are **written first, must FAIL before implementation begins** (BDD/TDD, constitution Principle IV)

---

## Phase 1: Setup ✅ Complete (1C-1a, PR #934)

- [X] T001 Confirm existing test suite passes on branch `006-model-autoscaler`

---

## Phase 2: Foundational — Shared Types, Interfaces, Config ✅ Complete (1C-1a, PR #934)

- [X] T002 [P] Add three fields to `RoutingSnapshot` in `sim/router_state.go`
- [X] T003 [P] Add `CostPerHour float64` field to `NodePoolConfig` in `sim/cluster/infra_config.go`
- [X] T004 [P] Create `sim/cluster/autoscaler.go`; declare all shared value types
- [X] T005 [P] Add autoscaler config fields to `DeploymentConfig` in `sim/cluster/deployment.go`
- [X] T006 Append `Collector`, `Analyzer`, `Engine`, `Actuator` interface declarations
- [X] T007 Add `ScalingTickEvent` and `ScaleActuationEvent` to `sim/cluster/cluster_event.go`
- [X] T008 Update `buildRouterState()` to populate variant fields on `RoutingSnapshot`

---

## Phase 3: User Story 1 — Pipeline Wiring + Cooldown ✅ Complete (1C-1a, PR #934)

Includes US1 (pipeline wiring) and US5 (cooldown/flap prevention).

- [X] T009 [US1] `TestScalingTickScheduling` — interval=0, multi-tick, same-tick actuation, delayed actuation
- [X] T010 [US1] `TestNoOpPipelineDeterminism` — INV-6 byte-identical output
- [X] T011 [US1] `TestNilComponentGuard` — nil component → pipeline disables, no panic
- [X] T012 [US5] `TestStabilizationWindowFilter` — per-model stabilization window gate (scenarios a–e)
- [X] T013 [US1] `TestGPUInventory` — free slot computation, TP fallback, negative clamping
- [X] T014 [US1] Implement `autoscalerPipeline` struct + `tick()` + `actuate()` + `gpuInventory()`
- [X] T015 [US1] First `ScalingTickEvent` pushed in `Run()` when autoscaler enabled

**Checkpoint**: ✅ PR #934 merged. INV-6 determinism confirmed.

---

## Phase 4: User Story 2 — V2 Saturation-Based Scale Signal (1C-1b)

**Goal**: `V2SaturationAnalyzer` computes per-replica capacity in token units using `min(k1_memory, k2_compute)`, demand as `tokensInUse + queueLength * avgInputTokens`, and emits model-level RequiredCapacity/SpareCapacity signals matching WVA V2.

**Independent Test**: `go test ./sim/cluster/... -run TestV2SaturationAnalyzer` passes against all table cases including zero-replica, single-replica, and mixed-variant edge cases.

### Tests for User Story 2

> **Write these tests first — they must FAIL before T017–T018 are implemented**

- [x] T016 [US2] Write `TestV2SaturationAnalyzerAnalyze` (table-driven) in `sim/cluster/saturation_analyzer_test.go` with cases: (a) `Replicas=nil` → all-zero output, no panic; (b) all replicas where demand exceeds `effectiveCapacity * ScaleUpThreshold` → `RequiredCapacity > 0`, `SpareCapacity == 0`; (c) all replicas where supply greatly exceeds `demand / ScaleDownBoundary` and N-1 redistribution is safe → `SpareCapacity > 0`; (d) N=1 (single replica) → `SpareCapacity == 0` always; (e) mixed variants → `sum(vc.Supply)==TotalSupply` and `sum(vc.Demand)==TotalDemand`; (f) `RequiredCapacity > 0` implies `SpareCapacity == 0`; (g) k1 < k2 → effective capacity equals k1 (memory-bound bottleneck)

### Implementation for User Story 2

- [x] T017 [US2] Declare `V2SaturationAnalyzerConfig{KvCacheThreshold, ScaleUpThreshold, ScaleDownBoundary float64}` and `V2SaturationAnalyzer` struct with `NewV2SaturationAnalyzer` constructor (validate: all fields > 0, no NaN/Inf) in `sim/cluster/saturation_analyzer.go`. Token-based capacity signal design (new ReplicaMetrics fields vs deriving from existing fields) to be resolved during implementation.
- [x] T018 [US2] Implement `Name() string` returning `"v2-saturation"` and `Analyze(metrics ModelSignals) AnalyzerResult` in `sim/cluster/saturation_analyzer.go`: per-replica `k1 = totalKvCapacityTokens * KvCacheThreshold` (memory-bound), `k2` from compute-bound estimate or k1 fallback, `effectiveCapacity = min(k1, k2)`, `demand = tokensInUse + queueLength * avgInputTokens`; model-level `RequiredCapacity = max(0, (totalDemand / ScaleUpThreshold) - (totalReadySupply + pendingSupply))` where `pendingSupply = PendingTotalKvCapacityTokens * KvCacheThreshold` (pending supply suppresses premature scale-up; see #1109); `SpareCapacity = max(0, totalSupply - (totalDemand / ScaleDownBoundary))` with N-1 redistribution check — `TotalSupply`, `Utilization`, `SpareCapacity` use ready-replica supply only (pending supply excluded from scale-down); `VariantCapacities` grouped by VariantSpec (sort variant keys for determinism R2); guard all divisions by zero (R11)

**Checkpoint**: `TestV2SaturationAnalyzerAnalyze` all cases pass. Analyzer half of PR 1C-1b ready.

---

## Phase 5: User Story 3 — End-to-End WVA Loop: Collector, UnlimitedEngine, Actuator (1C-1b)

**Goal**: `DefaultCollector` produces correct `ModelSignals` from `RouterState`. `UnlimitedEngine` converts analyzer signals into scale decisions without GPU inventory constraints. `DirectActuator` calls `PlacementManager.PlaceInstance()` for scale-up and transitions instances to `Draining` for scale-down. Full pipeline integration test passes end-to-end.

**Independent Test**: `go test ./sim/cluster/... -run TestDefaultCollector` and `go test ./sim/cluster/... -run TestDirectActuator` and `go test ./sim/cluster/... -run TestUnlimitedEngine` and `go test ./sim/cluster/... -run TestFullPipelineEndToEnd` pass.

### Tests for User Story 3

> **Write these tests first — they must FAIL before T021–T024 are implemented**

- [x] T019 [P] [US3] Write `TestDefaultCollectorCollect` (table-driven) in `sim/cluster/default_collector_test.go`: (a) empty `RouterState` → empty `[]ModelSignals`; (b) 3 snapshots for model M1, 2 for M2 → two `ModelSignals` entries, correct `ReplicaMetrics` field mapping (`KVUtilization`, `QueueDepth`, `InFlightRequests → InFlightCount`, `CostPerHour`, `Variant` from `GPUType`/`TPDegree`); (c) `TTFT=0` and `DispatchRate=0` always (default for now); (d) `LoadingSnapshots` for model M1 → `PendingReplicaCount` and `PendingTotalKvCapacityTokens` aggregated correctly, loading-only model (no routable replicas) produces entry with `len(Replicas)==0`; (e) loading snapshot with empty Model or GPUType skipped — see `TestDefaultCollector_PendingReplicaCount_FromLoadingSnapshots` (#1109)
- [ ] T020 [P] [US3] Write `TestDirectActuatorApply` (table-driven) in `sim/cluster/actuator_test.go`: (a) `Delta=+1` → `PlaceInstance` called with correct model/gpuType/tpDegree; (b) `Delta=-1` → target instance transitions to `InstanceStateDraining`, no longer routable; (c) `Delta=-1` with pending placement for same model → pending placement cancelled before drain begins
- [ ] T025 [P] [US3] Write `TestUnlimitedEngineOptimize` (table-driven) in `sim/cluster/engine_test.go`: (a) cheapest variant has zero free slots (inventory exhausted) → `UnlimitedEngine` still selects it (no inventory check); (b) scale-down targeting most expensive variant; (c) no decision when neutral (RequiredCapacity=0, SpareCapacity=0)

### Implementation for User Story 3

- [ ] T021 [US3] Implement `DefaultCollector struct{}` + `Collect(state *RouterState) []ModelSignals` in `sim/cluster/default_collector.go`: group `state.Snapshots` by `Model`; for each group build `ReplicaMetrics{InstanceID: snap.ID, Variant: VariantSpec{GPUType: snap.GPUType, TPDegree: snap.TPDegree}, KVUtilization: snap.KVUtilization, QueueDepth: snap.QueueDepth, InFlightCount: snap.InFlightRequests, CostPerHour: snap.CostPerHour, TTFT: 0, DispatchRate: 0}`; return one `ModelSignals` per distinct model; sort model keys for determinism (R2)
- [ ] T022 [US3] Implement `DirectActuator{cluster *ClusterSimulator}` + `Apply(decisions []ScaleDecision)` in `sim/cluster/direct_actuator.go`: for `Delta > 0` call `cs.pm.PlaceInstance(newInstanceID, decision.ModelID, decision.Variant.GPUType, decision.Variant.TPDegree)` — on error, log to stderr (R1, not silent); for `Delta < 0`, select oldest active instance for that model+variant, cancel any `PendingPlacement` entries for that model, transition instance to Draining — use WaitDrain semantics (router already skips Draining instances; GPUs freed when InFlightCount reaches 0)
- [ ] T028 [US3] Implement `UnlimitedEngine struct{}` + `Optimize(results []AnalyzerResult, inventory GPUInventory) []ScaleDecision` in `sim/cluster/engine.go`: for each model with `RequiredCapacity > 0`, sort `VariantCapacities` by `CostPerReplica` asc, pick cheapest variant, emit `ScaleDecision{Delta:+1}`; for each model with `SpareCapacity > 0`, sort by `CostPerReplica` desc, pick most expensive with `ReplicaCount > 0`, emit `ScaleDecision{Delta:-1}`; at most one decision per model; `inventory` parameter accepted but not used
- [ ] T029 [US3] Write `TestFullPipelineEndToEnd` integration test in `sim/cluster/pipeline_integration_test.go`: wire `DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine → DirectActuator`; run a cluster under synthetic high load for two ticks; verify at least one `PlaceInstance` is called (scale-up propagated end-to-end)

**Checkpoint**: All US3 unit tests pass. Integration test passes. PR 1C-1b fully complete — minimal viable WVA loop operational.

---

## Phase 6: User Story 4 — GreedyEngine: Inventory-Aware Allocation (1C-1d)

**Goal**: `GreedyEngine` selects cheapest variant with available GPU slots for scale-up (falls back when cheapest is full), targets most expensive active variant for scale-down, prioritizes models with higher `RequiredCapacity` when inventory is scarce.

**Independent Test**: `go test ./sim/cluster/... -run TestGreedyEngine` passes.

### Tests for User Story 4

> **Write these tests first — they must FAIL before T026–T027 are implemented**

- [ ] T024 [US4] Write `TestGreedyEngineOptimize` (table-driven) in `sim/cluster/engine_test.go`: (a) single model with two variants, cheapest has free slots → cheapest selected; (b) cheapest variant full, second cheapest has slots → fallback to second; (c) scale-down signal → most expensive active variant targeted; (d) both `RequiredCapacity=0` and `SpareCapacity=0` → no decision emitted; (e) two models competing for scarce GPU slots → higher `RequiredCapacity` model wins; (f) same model never gets both scale-up and scale-down in one call

### Implementation for User Story 4

- [ ] T026 [US4] Declare `GreedyEngine struct{}` in `sim/cluster/engine.go`; add shared helper `sortVariantsByAscCost(variants []VariantCapacity) []VariantCapacity` and `sortVariantsByDescCost(variants []VariantCapacity) []VariantCapacity` (sort keys for R2 determinism)
- [ ] T027 [US4] Implement `GreedyEngine.Optimize(results []AnalyzerResult, inventory GPUInventory) []ScaleDecision` in `sim/cluster/engine.go`: (1) sort results by `RequiredCapacity` desc for cross-model priority; (2) for each model with `RequiredCapacity > 0`, sort its `VariantCapacities` by `CostPerReplica` asc, pick first variant where `inventory.FreeSlots(v) >= v.Variant.TPDegree`, emit `ScaleDecision{Delta:+1}`, decrement inventory; (3) for each model with `SpareCapacity > 0` and no scale-up pending, sort variants by `CostPerReplica` desc, pick first with `ReplicaCount > 0`, emit `ScaleDecision{Delta:-1}`; at most one decision per model

**Checkpoint**: `TestGreedyEngine` all cases pass. PR 1C-1d complete. GPU-inventory-aware allocation operational.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T036 [P] Run `go test ./... -count=1` — all tests must pass; run `golangci-lint run ./...` — zero lint violations
- [ ] T037 [P] Run INV-6 regression check: `./blis run --model qwen/qwen3-14b > out-autoscaler.txt` (with `ModelAutoscalerIntervalUs=0` in config), compare to `out-baseline.txt` captured in T001; must be byte-identical
- [ ] T038 Review `specs/006-model-autoscaler/quickstart.md` and update any config field names or YAML keys that changed during implementation

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 ✅ (Setup)
    └── Phase 2 ✅ (Foundational: types + interfaces + config)
            └── Phase 3 ✅ (US1+US5: pipeline wiring + cooldown)
                    ├── Phase 4 (US2: V2SaturationAnalyzer)        ← 1C-1b
                    │       └── Phase 5 (US3: Collector + UnlimitedEngine + Actuator + integration)  ← 1C-1b
                    └── Phase 6 (US4: GreedyEngine)                ← 1C-1d (after 1C-1b)
                            └── Phase 7 (Polish)
```

### User Story Dependencies

| Story | Depends on | PR |
|-------|-----------|-----|
| US1 (P1): Pipeline wiring | Phase 2 | 1C-1a ✅ #934 |
| US5 (P3): Cooldown | US1 | 1C-1a ✅ #934 |
| US2 (P2): V2SaturationAnalyzer | US1 | 1C-1b #905 |
| US3 (P2): Collector + UnlimitedEngine + Actuator | US2 | 1C-1b #905 |
| US4 (P3): GreedyEngine | US3 (full loop working) | 1C-1d #918 |

### Within Each Phase

1. Tests are written first and must FAIL before implementation begins (Principle IV)
2. Types before implementations (Phase 2 → Phase 3+)
3. Struct declaration before method implementations (within each story)
4. Analyzer before Collector+Actuator integration test (US2 before US3 integration)

---

## Implementation Strategy

### MVP (1C-1b) — WVA team validation target

1. ~~Phase 1: Setup~~ ✅
2. ~~Phase 2: Foundational types~~ ✅
3. ~~Phase 3: Pipeline wiring + cooldown~~ ✅ (PR #934)
4. Complete Phase 4: US2 — V2SaturationAnalyzer
5. Complete Phase 5: US3 — DefaultCollector + UnlimitedEngine + DirectActuator + integration test
6. **STOP and VALIDATE**: `DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine → DirectActuator` runs end-to-end

### Full Delivery

7. Complete Phase 6: US4 — GreedyEngine (PR 1C-1d)
8. Complete Phase 7: Polish
9. Future: QueueingModelAnalyzer (#954) — M/M/1/K-SD with EKF parameter learning

---

## Notes

- `[P]` = touches a different file; can be run in parallel with other `[P]` tasks at same phase
- Each `[Story]` label maps to a user story in `spec.md`; use for traceability in PR descriptions
- Tests that reference `testing.Short()` must add `if testing.Short() { t.Skip() }` for sub-tests > 1s
- `go test ./... -count=1` must complete in under 60s (Principle IV)
- Every `map[VariantSpec]int` iteration must sort keys before use (R2)
- Every new struct field added to an existing type requires grepping literal construction sites (R4)
- Commit after each phase checkpoint, not after every individual task
