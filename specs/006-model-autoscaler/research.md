# Research: Phase 1C Model Autoscaler

**Branch**: `006-model-autoscaler`  
**Phase**: 0 — Resolved decisions before implementation begins  
**Source**: Codebase exploration of `sim/cluster/`, `sim/router_state.go`, `sim/rng.go`

---

## Decision 1: PlacementManager.PlaceInstance vs "PlacementEngine.TryPlace"

**Decision**: `DirectActuator` calls `PlacementManager.PlaceInstance(id, model, gpuType, tpDegree)` directly.

**Rationale**: The design doc and issue text use the conceptual name "PlacementEngine.TryPlace()". The actual type in the codebase is `PlacementManager` (defined in `sim/cluster/infra_placement.go`) with method `PlaceInstance(id InstanceID, model, gpuType string, tpDegree int) (nodeID string, gpuIDs []string, err error)`. No interface extraction is needed for this scope. `DirectActuator` is a concrete struct in `sim/cluster/` and has direct access to `ClusterSimulator.pm` (the PlacementManager).

**Alternatives considered**:
- Extract a `Placer` interface: adds indirection with no immediate benefit. Deferred to a later PR if needed for testing isolation.
- Use `PlaceInstance` via a passed-in closure: unnecessary complexity for a concrete type.

---

## Decision 2: Variant info in RoutingSnapshot

**Decision**: Add `GPUType string`, `TPDegree int`, and `CostPerHour float64` to `RoutingSnapshot` (in `sim/router_state.go`). Populate in `buildRouterState()` from the instance's configuration.

**Rationale**: `DefaultCollector.Collect(*RouterState) []ModelSignals` receives only `RouterState`. The current `RoutingSnapshot` has no variant information (`GPUType`, `TPDegree`). To enable `DefaultCollector` to populate `ReplicaMetrics.Variant` and `ReplicaMetrics.CostPerHour` without broadening its interface, the variant fields must be present in `RoutingSnapshot`. This is a clean additive change: `sim/router_state.go` is in the `sim/` package which does not import `sim/cluster/`, so there is no import cycle. `buildRouterState()` in `cluster_event.go` already has access to instance config and can populate these fields.

**Alternatives considered**:
- Change `Collect` signature to `Collect(*RouterState, variantLookup func(id string) VariantSpec)`: breaks the declared interface contract and adds complexity to all Collector implementations.
- Access `ClusterSimulator.instances` directly from `DefaultCollector`: requires `DefaultCollector` to hold a `*ClusterSimulator` reference. This is fine for `DirectActuator` (which already needs the cluster reference) but unnecessarily couples `DefaultCollector` to the simulator for something that belongs in the snapshot.

---

## Decision 3: CostPerReplica propagation chain

**Decision**: Add `CostPerHour float64` to `NodePoolConfig` (in `infra_config.go`). Populate `RoutingSnapshot.CostPerHour` from the instance's node pool. `DefaultCollector` maps `RoutingSnapshot.CostPerHour` to a `CostPerHour float64` field added to `ReplicaMetrics`. `Analyzer` uses `replica.CostPerHour` when building `VariantCapacity.CostPerReplica` — taking the CostPerHour from any replica in the variant group (all replicas of the same variant have the same cost).

**Rationale**: The declared `ReplicaMetrics` type in the issue spec does not include `CostPerHour`. Adding this one field keeps the propagation chain clean without requiring the Analyzer to accept a cost map parameter (which would violate the `Analyze(ModelSignals) AnalyzerResult` interface contract). The Analyzer must remain pure: it must not access external state. The field is cheap to carry and semantically belongs on a replica snapshot.

**Alternatives considered**:
- Populate `VariantCapacity.CostPerReplica` in the orchestrator (cluster.go tick handler) after calling `Analyzer.Analyze()`: requires the orchestrator to re-look up cost per variant from NodePoolConfig. More indirection with no benefit. The Analyzer already groups by variant; doing the cost lookup in the Analyzer is simpler.
- Add a `CostMap map[VariantSpec]float64` parameter to `Analyze()`: changes the interface, violates the single-argument behavioral contract.

---

## Decision 4: HPAScrapeDelay uses existing DelaySpec

**Decision**: `DeploymentConfig.HPAScrapeDelay` is of type `DelaySpec` (existing, in `infra_config.go`), not a new `DistributionSpec` type.

**Rationale**: The design doc uses "DistributionSpec" as a conceptual name for a configurable delay distribution. The existing `DelaySpec{Mean, Stddev}` with Gaussian sampling already serves this purpose and is used for `NodePoolConfig.ProvisioningDelay` and `InstanceLifecycleConfig.LoadingDelay`. Introducing a new type would be duplicate code. `DelaySpec.IsZero()` directly implements the "zero = deterministic no-delay" semantic needed for INV-6 compatibility.

**Alternatives considered**:
- New `DistributionSpec` type with explicit `kind: constant/gaussian/uniform` enumeration: more expressive but unnecessary for current use cases. Deferred to a later PR if needed.

---

## Decision 5: Event priorities for autoscaler events

**Decision**: `ScalingTickEvent` priority = 8, `ScaleActuationEvent` priority = 9.

**Rationale**: Existing cluster events use priorities 0–7 (arrival=0 through PD events=4–7). Autoscaler events run after all request-path events at the same timestamp, which is correct: the scaler observes completed request state, not in-progress routing decisions. Both new event types are added to `cluster_event.go`.

**Alternatives considered**:
- Priority 10+: Unnecessarily large gap. 8 and 9 are the natural next values.
- Same priority as other events (interleaved): incorrect semantics — the scaler should see a stable snapshot.

---

## Decision 6: TTFT and DispatchRate are zero in DefaultCollector

**Decision**: `ReplicaMetrics.TTFT` and `ReplicaMetrics.DispatchRate` are populated as zero by `DefaultCollector` in this scope. `RoutingSnapshot` does not carry these signals.

**Rationale**: `TTFT` and `DispatchRate` are used only by the future `QueueingModelAnalyzer` (deferred). Neither `SaturationAnalyzer`, `UtilizationAnalyzer`, nor `QueueAnalyzer` reads these fields. Adding them to `RoutingSnapshot` requires metric collection infrastructure that does not yet exist. Zero values are safe defaults — `Analyzer.Analyze()` implementations must not divide by these fields without guarding.

**Alternatives considered**:
- Add TTFT tracking to `InstanceSimulator` and populate in `buildRouterState()`: premature; QueueingModelAnalyzer is out of scope for this feature.

---

## Decision 7: GPUInventory computation

**Decision**: A `gpuInventory()` helper on `ClusterSimulator` computes `GPUInventory.ByVariant` by iterating the instance list and counting: for each variant `v`, free slots = total GPU slots in all Ready nodes for variant `v` minus slots held by Running instances of variant `v` minus slots held by Loading instances of variant `v`. Pending placements are NOT subtracted. Draining instances ARE subtracted (they hold GPUs until drain completes).

**Rationale**: This matches the design doc's definition exactly. The `PlacementManager` already tracks node and instance state. The helper is a pure computation function over existing state — no new state needed.

---

## Decision 8: Cooldown tracking location

**Decision**: Cooldown is tracked in two maps on `ClusterSimulator`: `lastScaleUpAt map[string]int64` and `lastScaleDownAt map[string]int64` (keyed by ModelID). These are initialized to empty maps when the autoscaler is enabled. The tick handler in `cluster.go` suppresses a `ScaleDecision` if `now - lastScaleAt[modelID] < CooldownUs` before forwarding to Actuator.

**Rationale**: Cooldown is an orchestrator concern, not an Engine or Analyzer concern. Keeping it in the tick handler means Engine and Analyzer remain stateless and independently testable. This matches the design doc's explicit statement: "cooldown lives in the ScalingTickEvent handler in cluster.go, not inside any interface."

---

## Decision 9: DrainPolicy for DirectActuator

**Decision**: `DirectActuator` uses the existing `InstanceState` transition `Active → Draining`. It does NOT call `PlacementManager.DrainNode()` (which is for node-level drain). Instead, it transitions the instance state to `Draining` directly and schedules a watch loop that frees GPUs when `InFlightCount == 0`. This implements WaitDrain semantics using existing instance lifecycle machinery.

**Rationale**: The full `DrainPolicy` interface (specs/010) is deferred. The existing `InstanceStateDraining` state already prevents the router from sending new requests (routable check skips Draining instances). Implementing WaitDrain as an instance-level state transition requires no new types.

**Alternatives considered**:
- Implement ImmediateDrain now (drop in-flight): not the right default; WaitDrain is safer and matches K8s graceful termination.
- Use `PlacementManager.DrainNode()`: that's for node-level drain (all instances on a node), not single-instance scale-down.
