# Feature Specification: Phase 1C Model Autoscaler

**Feature Branch**: `006-model-autoscaler`  
**Created**: 2026-04-01  
**Status**: Draft  
**Tracking Issue**: [#696](https://github.com/inference-sim/inference-sim/issues/696)  
**Design Document**: `docs/plans/2026-04-01-phase1c-autoscaling-design.md`

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Pipeline Contracts and Wiring (Priority: P1)

A researcher configuring a BLIS simulation wants to enable dynamic replica management so that the simulator can scale model instances up or down during a run in response to load. Without this, all experiments use fixed replica counts and cannot model autoscaling behavior at all.

**Why this priority**: This is the foundational step. All other stories depend on the four interfaces (Collector, Analyzer, Engine, Actuator) and their wiring into the simulation event loop. Without the contracts defined and the tick event firing the pipeline, nothing else can be built or tested.

**Independent Test**: Enable autoscaling with a zero-interval tick and a no-op pipeline (all interfaces return empty/zero results). The simulation output must be byte-identical to a run without autoscaling enabled, confirming the pipeline is wired but not disruptive.

**Acceptance Scenarios**:

1. **Given** a simulation with `ModelAutoscalerIntervalUs = 0`, **When** the simulation runs, **Then** no scaling tick is ever scheduled and no autoscaler logic executes.
2. **Given** a simulation with `ModelAutoscalerIntervalUs = T`, **When** the simulation runs past time T, **Then** a scaling tick fires at `t=0`, `t=T`, `t=2T`.
3. **Given** a no-op pipeline where all interfaces return empty/zero results, **When** the simulation runs, **Then** the output is byte-identical to a run without the autoscaler (determinism preserved).
4. **Given** `ActuationDelayUs = constant(0)`, **When** a tick fires, **Then** the actuation event fires in the same simulation tick.
5. **Given** `ActuationDelayUs = constant(30s)`, **When** a tick fires at time T, **Then** the actuation event fires at time T+30s.

---

### User Story 2 - Saturation-Based Scale Signal (Priority: P2)

A researcher running a load spike experiment wants the simulator to detect when replicas are near saturation (high KV cache utilization or deep queues) and emit a scale-up signal, and to detect when replicas have excess headroom and emit a scale-down signal — matching how the llm-d WVA production autoscaler works.

**Why this priority**: The SaturationAnalyzer is the reference algorithm from WVA that the llm-d team can recognize and validate. Without it, there is no meaningful capacity signal to drive scaling decisions. It is the analytical core of the model autoscaler baseline.

**Independent Test**: Inject a ModelMetrics snapshot with all replicas above the KV saturation threshold and verify RequiredCapacity > 0. Inject a snapshot with all replicas well below threshold with enough headroom to remove one replica and verify SpareCapacity > 0.

**Acceptance Scenarios**:

1. **Given** replicas where average spare KV capacity falls below the minimum spare threshold, **When** Analyze is called, **Then** RequiredCapacity is positive and SpareCapacity is zero.
2. **Given** replicas where removing one replica and redistributing load still leaves adequate spare capacity, **When** Analyze is called, **Then** SpareCapacity is positive.
3. **Given** a single replica, **When** Analyze is called with that replica near saturation, **Then** SpareCapacity is zero (cannot scale below one replica).
4. **Given** a model with no active replicas, **When** Analyze is called, **Then** all output fields are zero and no division-by-zero occurs.
5. **Given** mixed variants serving the same model, **When** Analyze is called, **Then** the sum of per-variant supply values equals TotalSupply and the sum of per-variant demand values equals TotalDemand.

---

### User Story 3 - Replica Count Changes Applied to Cluster (Priority: P2)

A researcher running a scaling experiment wants the simulator to actually add or remove replicas when the autoscaler emits scale decisions, so that future requests are routed to the adjusted replica set.

**Why this priority**: Without the Collector (to read cluster state) and the Actuator (to apply decisions), the pipeline produces signals that go nowhere. These two thin components close the loop and make the autoscaler functional end-to-end.

**Independent Test**: Run the minimal viable pipeline (DefaultCollector → SaturationAnalyzer → UnlimitedEngine → DirectActuator) against a simulated cluster under load. Verify that a scale-up decision results in a new instance being placed, and a scale-down decision results in an existing instance entering drain state.

**Acceptance Scenarios**:

1. **Given** a RouterState with active model instances, **When** DefaultCollector.Collect is called, **Then** one ModelMetrics entry is produced per active model, with one ReplicaMetrics entry per active instance.
2. **Given** a ScaleDecision with Delta > 0, **When** DirectActuator.Apply is called, **Then** PlacementEngine attempts to place a new instance for the specified model and variant.
3. **Given** a ScaleDecision with Delta < 0, **When** DirectActuator.Apply is called, **Then** the selected instance enters Draining state, stops receiving new requests, and its GPUs are freed only after all in-flight requests complete.
4. **Given** a scale-down decision for a model that also has a pending scale-up placement, **When** DirectActuator.Apply is called, **Then** the pending placement is cancelled before drain begins.

---

### User Story 4 - Variant-Aware Allocation (Priority: P3)

A researcher running a multi-variant experiment (e.g., A100 and H100 nodes in the same pool) wants the autoscaler to make cost-aware allocation decisions: preferring the cheapest available variant for scale-up and targeting the most expensive active variant for scale-down.

**Why this priority**: This is the Engine's contribution. GreedyEngine adds GPU inventory awareness on top of UnlimitedEngine. Both engines must handle multi-model scenarios where GPU capacity may be scarce across multiple simultaneously-scaling models.

**Independent Test**: Configure a cluster with two variants (cheap/expensive) and trigger scale-up. Verify GreedyEngine selects the cheaper variant. Remove all capacity from the cheaper variant and verify it falls back to the more expensive one.

**Acceptance Scenarios**:

1. **Given** two variants with different costs and both have available GPU slots, **When** GreedyEngine processes a scale-up signal, **Then** it selects the variant with the lower cost per replica.
2. **Given** the cheapest variant has no available GPU slots, **When** GreedyEngine processes a scale-up signal, **Then** it falls back to the next cheapest variant that has available slots.
3. **Given** a scale-down signal for a model with replicas across two variants, **When** GreedyEngine processes the signal, **Then** it targets the most expensive active variant.
4. **Given** GPU inventory is insufficient to satisfy scale-up requests for multiple models simultaneously, **When** GreedyEngine processes all results, **Then** models with higher RequiredCapacity are served before models with lower RequiredCapacity.
5. **Given** a scale-up signal and sufficient GPU slots, **When** UnlimitedEngine processes the signal, **Then** it selects the cheapest variant without checking GPU inventory.

---

### User Story 5 - Alternative Analyzer Baselines (Priority: P3)

A researcher wanting to isolate the effect of a single signal (either KV utilization or queue depth alone) can substitute a simpler baseline analyzer instead of SaturationAnalyzer, making it easy to compare signal strategies without changing the rest of the pipeline.

**Why this priority**: UtilizationAnalyzer and QueueAnalyzer are the baselines that make the Analyzer interface a useful research hook. They enable controlled experiments where only one signal is varied at a time.

**Independent Test**: Configure the pipeline with UtilizationAnalyzer. Drive KV utilization above the target threshold. Verify RequiredCapacity > 0. Drop utilization below target × scale-down factor. Verify SpareCapacity > 0.

**Acceptance Scenarios**:

1. **Given** UtilizationAnalyzer configured with a target utilization of 70%, **When** aggregate KV utilization exceeds 70%, **Then** RequiredCapacity is positive.
2. **Given** UtilizationAnalyzer, **When** aggregate KV utilization is below target × scale-down factor, **Then** SpareCapacity is positive.
3. **Given** QueueAnalyzer configured with ConsecutiveTicks = 3, **When** queue depth exceeds the scale-up threshold for only 1 tick then drops, **Then** RequiredCapacity remains zero (single spike suppressed).
4. **Given** QueueAnalyzer, **When** queue depth exceeds threshold for exactly 3 consecutive ticks, **Then** RequiredCapacity becomes positive on tick 3.
5. **Given** QueueAnalyzer with consecutive count in progress, **When** queue depth drops below threshold mid-sequence, **Then** the consecutive counter resets to zero.

---

### User Story 6 - Cooldown and Flap Prevention (Priority: P3)

A researcher studying oscillation behavior wants to configure scale-up and scale-down cooldown windows so that the autoscaler does not immediately reverse a scaling decision, matching the stabilization window behavior in HPA/KEDA.

**Why this priority**: Without cooldown, the pipeline may oscillate rapidly. Cooldown is the primary mechanism for preventing flapping and is required for any meaningful scaling experiment.

**Independent Test**: Configure ScaleUpCooldownUs = 60s. Trigger two consecutive scale-up signals for the same model within 60s. Verify only the first results in a ScaleDecision forwarded to the Actuator.

**Acceptance Scenarios**:

1. **Given** a scale-up decision was applied for model M at time T, **When** the next tick fires at T + 30s with ScaleUpCooldownUs = 60s, **Then** the scale-up decision is suppressed even if RequiredCapacity > 0.
2. **Given** a scale-up decision was applied for model M at time T, **When** the next tick fires at T + 90s with ScaleUpCooldownUs = 60s, **Then** a new scale-up decision is forwarded if RequiredCapacity > 0.
3. **Given** ScaleUpCooldownUs = 0, **When** consecutive ticks produce scale-up signals, **Then** all decisions are forwarded without suppression.

---

### Edge Cases

- What happens when a model has zero active replicas? Collector produces empty Replicas list; Analyzer returns all-zero result; no scale-down is emitted.
- What happens when GPU inventory is fully exhausted? GreedyEngine emits no scale-up decisions for affected models; no panic or silent failure.
- What happens when both RequiredCapacity and SpareCapacity are non-zero for the same model? Neither should be non-zero simultaneously — Analyzer implementations must ensure scale-up and scale-down signals are mutually exclusive.
- What happens when ActuationDelayUs is sampled as zero? The actuation event fires in the same tick as the scaling tick; causality is preserved.
- What happens when a Draining instance's model receives another scale-down decision? The Draining instance is already excluded from routing; the decision targets a different active instance.
- What happens when a placement fails because capacity is unavailable? A PendingPlacement is queued; no ScaleDecision is silently dropped.

## Requirements *(mandatory)*

### Functional Requirements

**Pipeline Orchestration**

- **FR-001**: The simulator MUST fire the autoscaling pipeline at a configurable interval (`ModelAutoscalerIntervalUs`); when the interval is zero, no tick is ever scheduled.
- **FR-002**: The pipeline MUST execute in order: Collector → Analyzer (per model) → Engine → schedule actuation event after `ActuationDelayUs` → Actuator applies decisions.
- **FR-003**: The pipeline MUST support per-model scale-up and scale-down cooldown windows; a decision suppressed by cooldown is discarded silently (not forwarded to Actuator).
- **FR-004**: The actuation step MUST be separated from the decision step by a configurable delay (`ActuationDelayUs`), which defaults to zero for determinism compatibility.

**Collector**

- **FR-005**: The Collector MUST produce exactly one ModelMetrics snapshot per active model from the current cluster state, grouping all replicas for that model together.
- **FR-006**: The Collector MUST NOT filter, threshold, or modify the raw per-replica signals it collects.

**Analyzer**

- **FR-007**: Each Analyzer MUST produce model-level aggregate supply and demand from the per-replica snapshots it receives; it MUST NOT access cluster state directly.
- **FR-008**: Every Analyzer MUST handle the zero-replica case by returning all-zero output without error.
- **FR-009**: Every Analyzer result MUST satisfy: the sum of per-variant supply equals TotalSupply, and the sum of per-variant demand equals TotalDemand.
- **FR-010**: The SaturationAnalyzer MUST block scale-down when removing one replica and redistributing its load would leave average spare capacity below the minimum threshold for either KV or queue.
- **FR-011**: The QueueAnalyzer MUST require the queue depth threshold to be violated for a configurable number of consecutive ticks before emitting a scale-up signal; a single spike must not trigger a decision.
- **FR-012**: An Analyzer MUST NOT emit both a positive RequiredCapacity and a positive SpareCapacity for the same model in the same call.

**Engine**

- **FR-013**: The Engine MUST emit at most one ScaleDecision per model per call; it reassesses on the next tick.
- **FR-014**: The Engine MUST NOT emit both a scale-up and a scale-down decision for the same model in one call.
- **FR-015**: When scaling up, the Engine MUST select the variant with the lowest cost per replica that satisfies the capacity constraint.
- **FR-016**: When scaling down, the Engine MUST target the variant with the highest cost per replica that has active replicas.
- **FR-017**: The GreedyEngine MUST respect the GPU inventory: it may only target a variant for scale-up if free GPU slots are available for that variant's tensor-parallel degree.
- **FR-018**: When multiple models need scale-up and GPU inventory is insufficient for all, the Engine MUST prioritize models with higher RequiredCapacity.

**Actuator**

- **FR-019**: The Actuator MUST attempt placement for scale-up decisions and MUST NOT silently drop failures; failed placements produce a queued pending placement.
- **FR-020**: The Actuator MUST set a replica to Draining for scale-down decisions, ensuring it stops receiving new requests and its GPUs are freed only after all in-flight requests complete.
- **FR-021**: Before applying a scale-down decision, the Actuator MUST cancel any pending placements for the same model to prevent simultaneous scale-up and scale-down churn.
- **FR-022**: The Actuator MUST NOT block; all effects are scheduled as future simulation events.

### Key Entities

- **ModelMetrics**: A snapshot of all replica states for one model at the moment of collection. Contains one ReplicaMetrics per active replica.
- **ReplicaMetrics**: The state of a single replica at collection time: KV utilization, queue depth, in-flight count, time-to-first-token, dispatch rate, and the hardware variant it runs on.
- **AnalyzerResult**: The model-level capacity assessment output by an Analyzer: aggregate supply, aggregate demand, scale-up signal (RequiredCapacity), scale-down signal (SpareCapacity), and a per-variant breakdown.
- **VariantCapacity**: One variant's share of the model's total supply and demand, with replica count and cost per replica. Used by the Engine to make allocation decisions.
- **ScaleDecision**: A single instruction to change replica count for one model+variant combination. Positive delta = add replicas, negative delta = remove replicas.
- **GPUInventory**: A read-only snapshot of free GPU slots per variant at the time the Engine is called, accounting for running and loading instances but not pending placements.
- **VariantSpec**: The hardware configuration identifier: GPU type and tensor-parallel degree. Used as the key in GPUInventory and as the target in ScaleDecision.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A simulation with the minimal viable pipeline (DefaultCollector → SaturationAnalyzer → UnlimitedEngine → DirectActuator) runs to completion without error on any workload configuration that runs today without the autoscaler.
- **SC-002**: With `ActuationDelayUs = 0` and a no-op pipeline, simulation output is byte-identical to a run without the autoscaler enabled (zero regression on existing determinism).
- **SC-003**: The full pipeline (tick → collect → analyze × N models → optimize → actuate) completes within each scaling tick without delaying the simulation clock; autoscaler overhead does not appear in simulated time.
- **SC-004**: A controlled scale-up experiment — where load is driven above the saturation threshold — results in at least one new replica being placed within two scaling ticks of the threshold being exceeded.
- **SC-005**: A controlled scale-down experiment — where load drops and stays below the spare capacity threshold for the required hysteresis period — results in at least one replica entering drain state.
- **SC-006**: All four autoscaler sub-issues (1C-1a through 1C-1d) can be implemented and tested independently without requiring the others to be complete, confirming interface isolation.
- **SC-007**: The Analyzer and Engine interfaces are independently swappable: any Analyzer implementation works with any Engine implementation without modification to either.
- **SC-008**: Replacing the autoscaler configuration with a different Analyzer or Engine implementation requires changing only the configuration, not any simulation core code.

## Scope

### In Scope

- Four interfaces: Collector, Analyzer, Engine, Actuator
- All shared data types: ModelMetrics, ReplicaMetrics, AnalyzerResult, VariantCapacity, ScaleDecision, GPUInventory, VariantSpec
- Pipeline event types: ScalingTickEvent, ScaleActuationEvent
- Pipeline orchestration wiring: tick handler, actuation event handler, cooldown tracking
- Configuration fields: ModelAutoscalerIntervalUs, ActuationDelayUs, ScaleUpCooldownUs, ScaleDownCooldownUs
- Reference implementations: DefaultCollector, SaturationAnalyzer, UtilizationAnalyzer, QueueAnalyzer, GreedyEngine, UnlimitedEngine, DirectActuator
- Cross-cutting invariants: INV-A1 through INV-A7, INV-1 extension with drained_dropped terminal state
- Integration test: full pipeline end-to-end with a simulated cluster

### Out of Scope

- Cluster autoscaler (node provisioning): specs/008
- Coordinator and PendingPlacement queue: specs/009
- Full DrainPolicy interface (ImmediateDrain, WaitDrain, RedirectDrain): specs/010
- Observability and per-model autoscaler metrics: specs/011
- Scale-from-zero (no active replicas → first placement): deferred
- QueueingModelAnalyzer (M/M/1 token model): deferred
- MIP solver Engine: Phase 2 OpenEvolve target

## Assumptions

- GPU inventory is a committed-state snapshot: free slots = total − running − loading. Pending placements are not subtracted (no GPU committed yet). Draining instances are subtracted (they hold GPUs until drain completes).
- WaitDrain semantics are the default for DirectActuator scale-down: the instance stops receiving new requests immediately but GPU slots are freed only after all in-flight requests complete. Full DrainPolicy selection is deferred to specs/010.
- The Analyzer is stateless across ticks (except QueueAnalyzer which holds consecutive-tick counters). State is held in the struct, not across interface boundaries.
- Cooldown is tracked in the pipeline orchestrator (cluster.go tick handler), not inside any interface, so Engine and Analyzer remain stateless and independently testable.
- Each Analyzer is called once per model per tick. There is no batch call across models; the Engine is the cross-model layer.
- The minimal viable pipeline for WVA team validation is: DefaultCollector → SaturationAnalyzer → UnlimitedEngine → DirectActuator. GreedyEngine and baseline analyzers are additive.
- ActuationDelayUs defaults to zero. This preserves byte-identical output with existing tests (INV-6 determinism) and allows the delay to be introduced explicitly for oscillation research.
