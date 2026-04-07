# Autoscaler Interface Contracts

**Branch**: `006-model-autoscaler`  
**File**: `sim/cluster/autoscaler.go`  
**Pattern**: Single-method interfaces for Collector, Engine, Actuator; Analyzer adds `Name()` for observability.

---

## Collector

```
Contract: Collector
  Signature:  Collect(state *RouterState) []ModelSignals
  Observes:   RouterState — per-instance signals, fully populated before each tick
  Produces:   One ModelSignals per active model, all replicas grouped by ModelID
  Must NOT:   Modify state; filter or threshold signals; return fewer models than are active
  Must NOT:   Access GPUInventory, AnalyzerResult, or ScaleDecision
  Invariant:  len(result) == number of distinct ModelIDs in active instances
  Zero state: Returns empty slice when no active instances exist
  Determinism: Pure function — same RouterState always produces same output
```

**Implementations**:
- `DefaultCollector{}` — maps RoutingSnapshot fields to ReplicaMetrics; sets TTFT=0, DispatchRate=0

---

## Analyzer

```
Contract: Analyzer
  Signature:  Name() string; Analyze(metrics ModelSignals) AnalyzerResult
  Observes:   ModelSignals for exactly one model (slice of ReplicaMetrics)
  Produces:   Model-level TotalSupply, TotalDemand, RequiredCapacity, SpareCapacity, VariantCapacities
  Must NOT:   Access RouterState, GPUInventory, or any external state
  Must NOT:   Emit ScaleDecision
  Must NOT:   Have RequiredCapacity > 0 AND SpareCapacity > 0 simultaneously
  Zero-replica: When metrics.Replicas is empty → return AnalyzerResult with all numeric fields = 0 (no panic)
  Invariant:  sum(vc.Supply for vc in VariantCapacities) == TotalSupply
  Invariant:  sum(vc.Demand for vc in VariantCapacities) == TotalDemand
  Invariant:  Utilization = TotalDemand / TotalSupply when TotalSupply > 0; else Utilization = 0
  Determinism: Must be a pure function of ModelSignals
```

**Implementations**:
- `V2SaturationAnalyzer{config V2SaturationAnalyzerConfig}` — WVA V2 token-based capacity model: `min(k1_memory, k2_compute)`, demand in tokens; N-1 redistribution check for scale-down

**Future**:
- `QueueingModelAnalyzer` — M/M/1/K-SD with online EKF parameter learning (WVA parity, #954)

---

## Engine

```
Contract: Engine
  Signature:  Optimize(results []AnalyzerResult, inventory GPUInventory) []ScaleDecision
  Observes:   All AnalyzerResults for this tick + current GPUInventory
  Produces:   At most one ScaleDecision per ModelID per call
  Must NOT:   Emit both scale-up and scale-down for the same ModelID in one call
  Must NOT:   Access RouterState, ModelSignals, or any external state
  Scale-up:   Target cheapest variant (CostPerReplica ascending) with available GPU slots
  Scale-down: Target most expensive variant (CostPerReplica descending) with active replicas
  Priority:   When inventory insufficient for all scale-ups, serve models in descending RequiredCapacity order
  Determinism: Must sort ByVariant keys and VariantCapacities before processing (R2)
```

**Implementations**:
- `GreedyEngine{}` — respects GPUInventory; falls back to next-cheapest when cheapest is full
- `UnlimitedEngine{}` — same scale-up/scale-down logic; skips GPU inventory check

---

## Actuator

```
Contract: Actuator
  Signature:  Apply(decisions []ScaleDecision) error
  Observes:   ScaleDecision slice (post-cooldown, already filtered by orchestrator)
  Effect:     Delta > 0 → calls PlacementManager.PlaceInstance(); failure is logged, not silently dropped
  Effect:     Delta < 0 → cancels pending placements for that model; transitions instance to Draining
  Must NOT:   Block; all effects are scheduled as future simulation events
  Must NOT:   Reorder or filter decisions — orchestrator already applied cooldown
  Drain semantics: WaitDrain — instance stops receiving new requests; GPUs freed after InFlightCount == 0
  Pending cancel: Before drain, cancel any PendingPlacement for the same ModelID
```

**Implementations**:
- `DirectActuator{cluster *ClusterSimulator}` — calls PlacementManager.PlaceInstance(); transitions instance state to Draining

---

## Orchestrator (Pipeline, not an interface)

The pipeline orchestrator is NOT an interface — the pipeline orchestration logic lives in `autoscalerPipeline.tick()` in `autoscaler.go`, called from `ScalingTickEvent.Execute()` in `cluster_event.go`. It:

1. Calls `Collector.Collect(routerState)` → `[]ModelSignals`
2. Calls `Analyzer.Analyze(m)` for each `m` in metrics → `[]AnalyzerResult`
3. Calls `Engine.Optimize(results, gpuInventory())` → `[]ScaleDecision`
4. Applies cooldown filter: suppresses decisions within cooldown window per model
5. Schedules `ScaleActuationEvent{At: now + ActuationDelay.Sample(rng), Decisions: filtered}`
6. Schedules next `ScalingTickEvent{At: now + ModelAutoscalerIntervalUs}`

The `ScaleActuationEvent.Execute()` method calls `Actuator.Apply(decisions)`.

**Cooldown state** (on ClusterSimulator):
- `lastScaleUpAt map[string]int64` — keyed by ModelID; updated when scale-up decision is forwarded
- `lastScaleDownAt map[string]int64` — keyed by ModelID; updated when scale-down decision is forwarded
- Suppression rule: `now - lastScaleAt[modelID] < CooldownUs → discard`
