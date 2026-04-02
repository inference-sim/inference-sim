# Tasks: Phase 1C Model Autoscaler

**Branch**: `006-model-autoscaler`  
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Data model**: [data-model.md](./data-model.md) | **Contracts**: [contracts/autoscaler-interfaces.md](./contracts/autoscaler-interfaces.md)  
**Issues**: [#692](https://github.com/inference-sim/inference-sim/issues/692) (1C-1a) · [#905](https://github.com/inference-sim/inference-sim/issues/905) (1C-1b) · [#906](https://github.com/inference-sim/inference-sim/issues/906) (1C-1c) · [#918](https://github.com/inference-sim/inference-sim/issues/918) (1C-1d)

**Format**: `[ID] [P?] [Story?] Description — file path`  
- **[P]**: parallelizable (touches a different file, no dependency on an incomplete task)  
- **[Story]**: US1–US6 maps to user stories in spec.md  
- Tests are **written first, must FAIL before implementation begins** (BDD/TDD, constitution Principle IV)

---

## Phase 1: Setup

**Purpose**: Confirm baseline is green. All subsequent work is additive; nothing should break existing tests.

- [X] T001 Confirm existing test suite passes on branch `006-model-autoscaler`: run `go test ./... -count=1` and `golangci-lint run ./...`; record any pre-existing failures to distinguish from regressions

---

## Phase 2: Foundational — Shared Types, Interfaces, Config (1C-1a types only)

**Purpose**: Declare all types and interfaces that every user story depends on. No behavior is wired here — all Execute() stubs return immediately. Deliverable: `go build ./...` passes.

**⚠️ CRITICAL**: No user story implementation can begin until this phase is complete.

- [X] T002 [P] Add three fields to `RoutingSnapshot` in `sim/router_state.go`: `GPUType string`, `TPDegree int`, `CostPerHour float64` — these are populated by `buildRouterState()` in the next task; adding them here makes the struct compile with zero values until then
- [X] T003 [P] Add `CostPerHour float64` field to `NodePoolConfig` in `sim/cluster/infra_config.go`; add validation: `CostPerHour < 0` or `math.IsNaN`/`math.IsInf` → return error; grep for all `NodePoolConfig{` literal construction sites in the codebase (R4) and update them (add `CostPerHour: 0` where no cost is specified)
- [X] T004 [P] Create `sim/cluster/autoscaler.go`; declare all shared value types: `VariantSpec{GPUType string, TPDegree int}`, `ReplicaMetrics{InstanceID, Variant, KVUtilization, QueueDepth, InFlightCount, CostPerHour, TTFT, DispatchRate}`, `ModelSignals{ModelID, Replicas []ReplicaMetrics}` (renamed from ModelMetrics to avoid collision with existing cluster.ModelMetrics output type), `VariantCapacity{Variant, Supply, Demand, ReplicaCount, CostPerReplica}`, `AnalyzerResult{ModelID, TotalSupply, TotalDemand, Utilization, RequiredCapacity, SpareCapacity, VariantCapacities []VariantCapacity}`, `ScaleDecision{ModelID, Variant, Delta int}`, `GPUInventory{ByVariant map[VariantSpec]int}`
- [X] T005 [P] Add `ModelAutoscalerIntervalUs float64`, `ActuationDelayUs DelaySpec`, `ScaleUpCooldownUs float64`, `ScaleDownCooldownUs float64` to `DeploymentConfig` in `sim/cluster/deployment.go`; grep for all `DeploymentConfig{` literal construction sites (R4) and confirm they still compile (new fields default to zero, which is correct: autoscaler disabled by default)
- [X] T006 Append `Collector`, `Analyzer`, `Engine`, `Actuator` interface declarations to `sim/cluster/autoscaler.go` (after T004 types are declared): `Collector.Collect(*RouterState) []ModelSignals`, `Analyzer.Name() string; Analyze(ModelSignals) AnalyzerResult`, `Engine.Optimize([]AnalyzerResult, GPUInventory) []ScaleDecision`, `Actuator.Apply([]ScaleDecision)`
- [X] T007 Add `ScalingTickEvent{At int64}` and `ScaleActuationEvent{At int64, Decisions []ScaleDecision}` to `sim/cluster/cluster_event.go`; implement `Timestamp() int64`, `Priority() int` (8 and 9 respectively), and stub `Execute(*ClusterSimulator)` methods that delegate to `cs.autoscaler.tick/actuate`
- [X] T008 Update `buildRouterState()` in `sim/cluster/cluster_event.go` to populate the three new `RoutingSnapshot` fields: read `GPUType`, `TPDegree`, `CostPerHour` from `InstanceSimulator` (fields set at placement time in cluster.go)

**Checkpoint**: `go build ./...` passes. No behavior change. All new types and interfaces are exported and visible.

---

## Phase 3: User Story 1 — Pipeline Wiring (Priority: P1) 🎯 MVP Foundation

**Goal**: The `ScalingTickEvent` fires at the configured interval, runs the four-stage pipeline (Collect → Analyze × models → Optimize → schedule actuation), and produces byte-identical output to a pre-autoscaler run when `ModelAutoscalerIntervalUs = 0` or when all pipeline fields are nil.

**Independent Test**: `go test ./sim/cluster/... -run TestScalingTick` passes; `TestNoOpPipelineDeterminism` confirms INV-6.

### Tests for User Story 1

> **Write these tests first — they must FAIL before T012–T016 are implemented**

- [X] T009 [US1] Write `TestScalingTickScheduling` (table-driven) in `sim/cluster/autoscaler_test.go`: (a) `ModelAutoscalerIntervalUs=0` → no `ScalingTickEvent` in queue after cluster init; (b) interval=T → verify tick events at `t=0`, `t=T`, `t=2T` by inspecting event queue; (c) `ActuationDelayUs={Mean:0}` → `ScaleActuationEvent.At == ScalingTickEvent.At`; (d) `ActuationDelayUs={Mean:30}` → `ScaleActuationEvent.At == ScalingTickEvent.At + 30e6`; add `testing.Short()` skip for any sub-test exceeding 1s
- [X] T010 [US1] Write `TestNoOpPipelineDeterminism` in `sim/cluster/autoscaler_test.go`: wire stub implementations of all four interfaces (stubs return `nil`/empty/zero), run simulation with `ModelAutoscalerIntervalUs=60e6`, capture stdout; run same simulation without autoscaler (`ModelAutoscalerIntervalUs=0`), capture stdout; assert bytes are identical (INV-6)

### Implementation for User Story 1

- [X] T011 [US1] Add autoscaler pipeline to `ClusterSimulator`: fields in `autoscalerPipeline` (collector, analyzer, engine, actuator, lastScaleUpAt/Down maps, rng), initialized in `NewClusterSimulator` when `ModelAutoscalerIntervalUs > 0`; `subsystemAutoscaler = "autoscaler"` constant added to infra_config.go
- [X] T012 [US1] Implement `gpuInventory() GPUInventory` on `*ClusterSimulator` in cluster.go: iterates placement.nodesByID (Ready nodes) for total GPUs per GPUType, subtracts GPUs from Loading/Active/WarmingUp/Draining instances; returns ByVariant keyed by observed VariantSpecs
- [X] T013 [US1] Implement full pipeline in `autoscalerPipeline.tick()` in autoscaler.go: Collect → Analyze × models → Optimize → cooldown filter → schedule ScaleActuationEvent → schedule next ScalingTickEvent; ScalingTickEvent.Execute() and ScaleActuationEvent.Execute() delegate to tick()/actuate()
- [X] T014 [US1] ScaleActuationEvent.Execute() calls cs.autoscaler.actuate() → actuator.Apply(decisions); already implemented as actuate() stub in autoscaler.go
- [X] T015 [US1] First ScalingTickEvent pushed in Run() after heap.Init, when autoscaler != nil and interval > 0

**Checkpoint**: `TestScalingTickScheduling` passes. `TestNoOpPipelineDeterminism` passes (INV-6 confirmed). PR 1C-1a ready.

---

## Phase 4: User Story 2 — Saturation-Based Scale Signal (Priority: P2)

**Goal**: `SaturationAnalyzer` correctly classifies replicas as saturated/idle, computes model-level supply/demand, emits `RequiredCapacity > 0` when spare KV or queue headroom is below threshold, and gates scale-down with an N-1 redistribution safety check.

**Independent Test**: `go test ./sim/cluster/... -run TestSaturationAnalyzer` passes against all table cases including zero-replica and single-replica edge cases.

### Tests for User Story 2

> **Write these tests first — they must FAIL before T017–T019 are implemented**

- [ ] T016 [US2] Write `TestSaturationAnalyzerAnalyze` (table-driven) in `sim/cluster/saturation_analyzer_test.go` with cases: (a) `Replicas=nil` → all-zero output, no panic; (b) all replicas above KV threshold → `RequiredCapacity > 0`, `SpareCapacity == 0`; (c) all replicas above queue threshold → `RequiredCapacity > 0`; (d) all replicas idle with N≥2 and N-1 redistribution leaves sufficient headroom → `SpareCapacity > 0`; (e) N=1 (single replica) → `SpareCapacity == 0` always; (f) mixed variants → `sum(vc.Supply)==TotalSupply` and `sum(vc.Demand)==TotalDemand`; (g) `RequiredCapacity > 0` implies `SpareCapacity == 0`

### Implementation for User Story 2

- [ ] T017 [US2] Declare `SaturationAnalyzerConfig{KVThreshold, QueueThreshold, MinSpareKV, MinSpareQueue, ScaleUpThreshold, ScaleDownBoundary float64}` and `SaturationAnalyzer{config SaturationAnalyzerConfig}` struct and `NewSaturationAnalyzer(cfg SaturationAnalyzerConfig) *SaturationAnalyzer` constructor (validate: all fields ≥0, no NaN/Inf, `ScaleUpThreshold` and `ScaleDownBoundary` > 0) in `sim/cluster/saturation_analyzer.go`
- [ ] T018 [US2] Implement `Name() string` returning `"saturation"` and `Analyze(metrics ModelMetrics) AnalyzerResult` in `sim/cluster/saturation_analyzer.go`: per-replica spare KV/queue, model-level TotalSupply/TotalDemand/Utilization, scale-up signal when `avg_spare_kv < MinSpareKV OR avg_spare_queue < MinSpareQueue`, N-1 redistribution check for scale-down, `VariantCapacities` grouped by VariantSpec (sort variant keys for determinism R2); guard all divisions by zero (R11); `Utilization = 0` when `TotalSupply == 0`

**Checkpoint**: `TestSaturationAnalyzerAnalyze` all cases pass. PR 1C-1b (analyzer half) ready.

---

## Phase 5: User Story 3 — Replica Count Changes Applied to Cluster (Priority: P2)

**Goal**: `DefaultCollector` produces correct `ModelMetrics` from `RouterState`. `DirectActuator` calls `PlacementManager.PlaceInstance()` for scale-up and transitions instances to `Draining` for scale-down with WaitDrain semantics.

**Independent Test**: `go test ./sim/cluster/... -run TestDefaultCollector` and `go test ./sim/cluster/... -run TestDirectActuator` pass. Integration test with stub engine passes.

### Tests for User Story 3

> **Write these tests first — they must FAIL before T022–T024 are implemented**

- [ ] T019 [P] [US3] Write `TestDefaultCollectorCollect` (table-driven) in `sim/cluster/saturation_analyzer_test.go`: (a) empty `RouterState` → empty `[]ModelMetrics`; (b) 3 snapshots for model M1, 2 for M2 → two `ModelMetrics` entries, correct `ReplicaMetrics` field mapping (`KVUtilization`, `QueueDepth`, `InFlightRequests → InFlightCount`, `CostPerHour`, `Variant` from `GPUType`/`TPDegree`); (c) `TTFT=0` and `DispatchRate=0` always (default for now)
- [ ] T020 [P] [US3] Write `TestDirectActuatorApply` (table-driven) in `sim/cluster/saturation_analyzer_test.go`: (a) `Delta=+1` → `PlaceInstance` called with correct model/gpuType/tpDegree; (b) `Delta=-1` → target instance transitions to `InstanceStateDraining`, no longer routable; (c) `Delta=-1` with pending placement for same model → pending placement cancelled before drain begins

### Implementation for User Story 3

- [ ] T021 [US3] Implement `DefaultCollector struct{}` + `Collect(state *RouterState) []ModelMetrics` in `sim/cluster/default_collector.go`: group `state.Snapshots` by `Model`; for each group build `ReplicaMetrics{InstanceID: snap.ID, Variant: VariantSpec{GPUType: snap.GPUType, TPDegree: snap.TPDegree}, KVUtilization: snap.KVUtilization, QueueDepth: snap.QueueDepth, InFlightCount: snap.InFlightRequests, CostPerHour: snap.CostPerHour, TTFT: 0, DispatchRate: 0}`; return one `ModelMetrics` per distinct model
- [ ] T022 [US3] Implement `DirectActuator{cluster *ClusterSimulator}` + `Apply(decisions []ScaleDecision)` in `sim/cluster/direct_actuator.go`: for `Delta > 0` call `cs.pm.PlaceInstance(newInstanceID, decision.ModelID, decision.Variant.GPUType, decision.Variant.TPDegree)` — on error, log to stderr (R1, not silent); for `Delta < 0`, select oldest active instance for that model+variant, cancel any `PendingPlacement` entries for that model, call `cs.instances[id].TransitionTo(InstanceStateDraining)` — use WaitDrain semantics (router already skips Draining instances; GPUs freed when InFlightCount reaches 0 via existing lifecycle events)
- [ ] T023 [US3] Write `TestPipelineWithSaturationAnalyzer` integration test in `sim/cluster/saturation_analyzer_test.go`: wire `DefaultCollector → SaturationAnalyzer → stub engine (returns empty []ScaleDecision) → DirectActuator`; run one tick; verify no panic, `DefaultCollector` received non-empty `RouterState`, `SaturationAnalyzer` returned result with correct model IDs

**Checkpoint**: All US3 unit tests pass. Integration test with stub engine passes. PR 1C-1b fully complete (analyzer + collector + actuator).

---

## Phase 6: User Story 4 — Variant-Aware Allocation (Priority: P3)

**Goal**: `GreedyEngine` selects cheapest variant with available GPU slots for scale-up (falls back when cheapest is full), targets most expensive active variant for scale-down, prioritizes models with higher `RequiredCapacity` when inventory is scarce. `UnlimitedEngine` uses same logic but skips inventory check.

**Independent Test**: `go test ./sim/cluster/... -run TestGreedyEngine` and `TestUnlimitedEngine` pass; full pipeline integration test passes.

### Tests for User Story 4

> **Write these tests first — they must FAIL before T027–T029 are implemented**

- [ ] T024 [P] [US4] Write `TestGreedyEngineOptimize` (table-driven) in `sim/cluster/engine_test.go`: (a) single model with two variants, cheapest has free slots → cheapest selected; (b) cheapest variant full, second cheapest has slots → fallback to second; (c) scale-down signal → most expensive active variant targeted; (d) both `RequiredCapacity=0` and `SpareCapacity=0` → no decision emitted; (e) two models competing for scarce GPU slots → higher `RequiredCapacity` model wins; (f) same model never gets both scale-up and scale-down in one call
- [ ] T025 [P] [US4] Write `TestUnlimitedEngineOptimize` (table-driven) in `sim/cluster/engine_test.go`: (a) cheapest variant has zero free slots (inventory exhausted) → `UnlimitedEngine` still selects it (no inventory check); (b) scale-down targeting; (c) no decision when neutral

### Implementation for User Story 4

- [ ] T026 [US4] Declare `GreedyEngine struct{}` and `UnlimitedEngine struct{}` in `sim/cluster/engine.go`; add shared helper `sortVariantsByAscCost(variants []VariantCapacity) []VariantCapacity` and `sortVariantsByDescCost(variants []VariantCapacity) []VariantCapacity` (sort keys for R2 determinism)
- [ ] T027 [US4] Implement `GreedyEngine.Optimize(results []AnalyzerResult, inventory GPUInventory) []ScaleDecision` in `sim/cluster/engine.go`: (1) sort results by `RequiredCapacity` desc for cross-model priority; (2) for each model with `RequiredCapacity > 0`, sort its `VariantCapacities` by `CostPerReplica` asc, pick first variant where `inventory.ByVariant[v] >= v.Variant.TPDegree`, emit `ScaleDecision{Delta:+1}`, decrement inventory; (3) for each model with `SpareCapacity > 0` and no scale-up pending, sort variants by `CostPerReplica` desc, pick first with `ReplicaCount > 0`, emit `ScaleDecision{Delta:-1}`; at most one decision per model
- [ ] T028 [US4] Implement `UnlimitedEngine.Optimize(results []AnalyzerResult, inventory GPUInventory) []ScaleDecision` in `sim/cluster/engine.go`: identical to `GreedyEngine` but omit the `inventory.ByVariant[v] >= v.Variant.TPDegree` check; `inventory` parameter accepted but not used
- [ ] T029 [US4] Write `TestFullPipelineEndToEnd` integration test in `sim/cluster/engine_test.go`: wire `DefaultCollector → SaturationAnalyzer → UnlimitedEngine → DirectActuator`; run a cluster under synthetic high-KV-load for two ticks; verify at least one `PlaceInstance` is called (scale-up propagated end-to-end)

**Checkpoint**: `TestGreedyEngine`, `TestUnlimitedEngine`, `TestFullPipelineEndToEnd` all pass. PR 1C-1d complete. Minimal viable pipeline (defaultcollector + saturationanalyzer + unlimitedengine + directactuator) fully operational.

---

## Phase 7: User Story 5 — Alternative Analyzer Baselines (Priority: P3)

**Goal**: `UtilizationAnalyzer` and `QueueAnalyzer` are drop-in replacements for `SaturationAnalyzer`. They implement the same `Analyzer` interface with simpler single-signal logic.

**Independent Test**: `go test ./sim/cluster/... -run TestUtilizationAnalyzer` and `TestQueueAnalyzer` pass, including hysteresis and edge cases.

### Tests for User Story 5

> **Write these tests first — they must FAIL before T032–T033 are implemented**

- [ ] T030 [P] [US5] Write `TestUtilizationAnalyzerAnalyze` (table-driven) in `sim/cluster/baseline_analyzers_test.go`: (a) utilization > `TargetUtilization` → `RequiredCapacity > 0`; (b) utilization < `TargetUtilization * ScaleDownFactor` → `SpareCapacity > 0`; (c) utilization between thresholds → both zero; (d) zero replicas → all-zero output, no panic; (e) `sum(vc.Supply) == TotalSupply`
- [ ] T031 [P] [US5] Write `TestQueueAnalyzerAnalyze` (table-driven) in `sim/cluster/baseline_analyzers_test.go`: (a) queue above threshold for 1 tick → `RequiredCapacity == 0` (spike suppressed); (b) queue above threshold for `ConsecutiveTicks` ticks → `RequiredCapacity > 0` on tick N; (c) queue drops below threshold mid-sequence → `consecutiveHigh` resets to 0; (d) queue below `ScaleDownThreshold` for `ConsecutiveTicks` ticks → `SpareCapacity > 0`; (e) `RequiredCapacity > 0` implies `SpareCapacity == 0`

### Implementation for User Story 5

- [ ] T032 [US5] Implement `UtilizationAnalyzerConfig{TargetUtilization, ScaleDownFactor float64}` + `UtilizationAnalyzer{config}` + `NewUtilizationAnalyzer` constructor + `Name() string` returning `"utilization"` + `Analyze(ModelMetrics) AnalyzerResult` in `sim/cluster/baseline_analyzers.go`; guard zero-replica case (R11); populate `VariantCapacities` proportionally (supply/demand split by replica count per variant); sort variant keys (R2)
- [ ] T033 [US5] Implement `QueueAnalyzerConfig{MaxQueuePerReplica, ScaleUpThreshold, ScaleDownThreshold float64; ConsecutiveTicks int}` + `QueueAnalyzer{config, consecutiveHigh, consecutiveLow int}` + `NewQueueAnalyzer` constructor + `Name() string` returning `"queue"` + `Analyze(ModelMetrics) AnalyzerResult` in `sim/cluster/baseline_analyzers.go`; increment `consecutiveHigh` when avg queue depth > `ScaleUpThreshold`, reset when not; emit `RequiredCapacity > 0` only when `consecutiveHigh >= ConsecutiveTicks`; mirror logic for `consecutiveLow` / scale-down

**Checkpoint**: Both baseline analyzers pass all table tests. PR 1C-1c complete. Any of the three analyzers (Saturation, Utilization, Queue) can now be plugged into the pipeline.

---

## Phase 8: User Story 6 — Cooldown and Flap Prevention (Priority: P3)

**Goal**: Scale-up decisions for a model are suppressed when fewer than `ScaleUpCooldownUs` microseconds have elapsed since the last forwarded scale-up for that model. Mirrored for scale-down. Cooldown `= 0` means no suppression.

**Independent Test**: `go test ./sim/cluster/... -run TestCooldown` passes; decision counts match expected suppression behavior.

### Tests for User Story 6

> **Write these tests first — they must FAIL before T035 is implemented**

- [ ] T034 [US6] Write `TestCooldown` (table-driven) in `sim/cluster/autoscaler_test.go`: (a) scale-up decision forwarded at `t=0`, second scale-up for same model at `t=30s` with `ScaleUpCooldownUs=60s` → second decision suppressed; (b) second scale-up at `t=90s` → forwarded; (c) `ScaleUpCooldownUs=0` → both decisions forwarded; (d) cooldown for scale-up does not suppress scale-down; (e) different models have independent cooldown timers

### Implementation for User Story 6

- [ ] T035 [US6] In `ScalingTickEvent.Execute()` in `sim/cluster/cluster.go`, add cooldown filter between `Engine.Optimize()` and scheduling `ScaleActuationEvent`: iterate decisions; for `Delta > 0`, if `now - cs.lastScaleUpAt[decision.ModelID] < cs.config.ScaleUpCooldownUs` then skip; else forward and set `cs.lastScaleUpAt[decision.ModelID] = now`; mirror for `Delta < 0` / `lastScaleDownAt` / `ScaleDownCooldownUs`; pass only surviving decisions to `ScaleActuationEvent.Decisions`

**Checkpoint**: `TestCooldown` passes. Autoscaler no longer oscillates under rapid consecutive signals. US6 complete.

---

## Phase 9: Polish & Cross-Cutting Concerns

- [ ] T036 [P] Run `go test ./... -count=1` — all tests must pass; run `golangci-lint run ./...` — zero lint violations
- [ ] T037 [P] Run INV-6 regression check: `./blis run --model qwen/qwen3-14b > out-autoscaler.txt` (with `ModelAutoscalerIntervalUs=0` in config), compare to `out-baseline.txt` captured in T001; must be byte-identical
- [ ] T038 Review `specs/006-model-autoscaler/quickstart.md` and update any config field names or YAML keys that changed during implementation

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
    └── Phase 2 (Foundational: types + interfaces + config) ← BLOCKS everything below
            ├── Phase 3 (US1: pipeline wiring)  ← BLOCKS US2, US3, US4, US5, US6
            │       ├── Phase 4 (US2: SaturationAnalyzer)
            │       │       └── Phase 5 (US3: Collector + Actuator + integration test)
            │       ├── Phase 6 (US4: GreedyEngine + UnlimitedEngine)  ← can run parallel with US2
            │       │       └── (US3 integration test T029 also needs US4)
            │       ├── Phase 7 (US5: baseline analyzers)              ← can run parallel with US2, US4
            │       └── Phase 8 (US6: cooldown)                        ← extends US1 wiring
            └── Phase 9 (Polish) ← after all desired stories complete
```

### User Story Dependencies

| Story | Depends on | Can run parallel with |
|-------|-----------|----------------------|
| US1 (P1): Pipeline wiring | Phase 2 | — |
| US2 (P2): SaturationAnalyzer | US1 | US4, US5 |
| US3 (P2): Collector + Actuator | US1; integration test also needs US2 + US4 | US4, US5 (unit tests only) |
| US4 (P3): GreedyEngine + UnlimitedEngine | US1 | US2, US5 |
| US5 (P3): Baseline analyzers | US1 | US2, US4 |
| US6 (P3): Cooldown | US1 (extends wiring) | US2, US4, US5 (after US1 complete) |

### Within Each Phase

1. Tests are written first and must FAIL before implementation begins (Principle IV)
2. Types before implementations (Phase 2 → Phase 3+)
3. Struct declaration before method implementations (within each story)
4. Analyzer before Collector+Actuator integration test (US2 before US3 integration)

---

## Parallel Execution Examples

### Phase 2 (Foundational) — all parallelizable

```
Parallel launch:
  Task T002: sim/router_state.go — RoutingSnapshot fields
  Task T003: sim/cluster/infra_config.go — NodePoolConfig.CostPerHour
  Task T004: sim/cluster/autoscaler.go — value types (new file)
  Task T005: sim/cluster/deployment.go — DeploymentConfig fields
```

### Phase 4 + Phase 6 + Phase 7 — run after Phase 3 completes

```
Developer A: Phase 4 (US2: SaturationAnalyzer)  sim/cluster/saturation_analyzer.go
Developer B: Phase 6 (US4: Engines)              sim/cluster/engine.go
Developer C: Phase 7 (US5: Baseline analyzers)   sim/cluster/baseline_analyzers.go
```

### Within Phase 4

```
Parallel launch (tests first, same file):
  Task T016: TestSaturationAnalyzerAnalyze — write test cases (must FAIL)
  → Then (sequential): T017 → T018 (implementation)
```

---

## Implementation Strategy

### MVP (US1 + US2 + US3) — WVA team validation target

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational types (CRITICAL — blocks everything)
3. Complete Phase 3: US1 — pipeline wiring + tick events (PR 1C-1a)
4. Complete Phase 4: US2 — SaturationAnalyzer (PR 1C-1b, first half)
5. Complete Phase 5: US3 — DefaultCollector + DirectActuator (PR 1C-1b, second half)
6. **STOP and VALIDATE**: `DefaultCollector → SaturationAnalyzer → UnlimitedEngine(stub) → DirectActuator` runs end-to-end; demo to WVA/llm-d team
7. Share with Lionel/team for feedback before proceeding to US4–US6

### Full Delivery

8. Complete Phase 6: US4 — Engines (PR 1C-1d); full MVP pipeline with UnlimitedEngine now working
9. Complete Phase 7: US5 — Baseline analyzers (PR 1C-1c)
10. Complete Phase 8: US6 — Cooldown
11. Complete Phase 9: Polish

---

## Notes

- `[P]` = touches a different file; can be run in parallel with other `[P]` tasks at same phase
- Each `[Story]` label maps to a user story in `spec.md`; use for traceability in PR descriptions
- Tests that reference `testing.Short()` must add `if testing.Short() { t.Skip() }` for sub-tests > 1s
- `go test ./... -count=1` must complete in under 60s (Principle IV)
- Every `map[VariantSpec]int` iteration must sort keys before use (R2)
- Every new struct field added to an existing type requires grepping literal construction sites (R4)
- Commit after each phase checkpoint, not after every individual task
