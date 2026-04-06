# Data Model: Phase 1C Model Autoscaler

**Branch**: `006-model-autoscaler`  
**All types live in**: `sim/cluster/autoscaler.go` (new file, 1C-1a)  
**Extended types**: `sim/router_state.go` (RoutingSnapshot), `sim/cluster/infra_config.go` (NodePoolConfig)

---

## New Types

### VariantSpec

Identifies a specific hardware configuration. Used as key in GPUInventory and as allocation target in ScaleDecision.

```
VariantSpec
├── GPUType  string   // e.g. "A100-80GB", "H100-80GB"
└── TPDegree int      // tensor-parallel degree: 1, 2, 4, 8
```

**Validation**: GPUType must be non-empty; TPDegree must be ≥1.  
**Usage**: Key in `GPUInventory.ByVariant`. Carried in `ScaleDecision.Variant` and `VariantCapacity.Variant`.  
**Map key safety**: Used as a Go map key — both fields are comparable types (string, int). No pointer fields.

---

### ReplicaMetrics

Snapshot of one replica's observable state at collection time. Produced by Collector, consumed by Analyzer.

```
ReplicaMetrics
├── InstanceID    string     // matches RouterState.Snapshots[i].ID
├── Variant       VariantSpec
├── KVUtilization float64    // [0.0, 1.0] — from RoutingSnapshot.KVUtilization
├── QueueDepth    int        // from RoutingSnapshot.QueueDepth
├── InFlightCount int        // from RoutingSnapshot.InFlightRequests
├── CostPerHour   float64    // from RoutingSnapshot.CostPerHour (NodePool cost)
├── TTFT          float64    // μs — zero in DefaultCollector (future: QueueingModelAnalyzer)
└── DispatchRate  float64    // req/s — zero in DefaultCollector (future: QueueingModelAnalyzer)
```

**Note on CostPerHour**: This field extends the issue spec's 7-field ReplicaMetrics. See research.md Decision 3.  
**Invariant**: `KVUtilization ∈ [0.0, 1.0]`, `QueueDepth ≥ 0`, `InFlightCount ≥ 0`.

---

### ModelSignals

All replica snapshots for one model. Output of Collector. Input to Analyzer (one call per model).

Named `ModelSignals` (not `ModelMetrics`) to avoid collision with the existing `ModelMetrics` output-stats type in `metrics.go`.

```
ModelSignals
├── ModelID  string
└── Replicas []ReplicaMetrics   // may be empty (zero-replica model)
```

**Zero-replica invariant**: When `len(Replicas) == 0`, `Analyzer.Analyze()` must return all-zero `AnalyzerResult` without error.

---

### VariantCapacity

One variant's share of a model's total supply and demand. Used by Engine to select allocation target.

```
VariantCapacity
├── Variant         VariantSpec
├── Supply          float64     // this variant's contribution to TotalSupply
├── Demand          float64     // this variant's share of TotalDemand
├── ReplicaCount    int         // active replicas of this variant serving this model
└── CostPerReplica  float64     // from ReplicaMetrics.CostPerHour for replicas of this variant
```

**Aggregation invariant**: `sum(VariantCapacity.Supply over all variants) == AnalyzerResult.TotalSupply`.  
`sum(VariantCapacity.Demand over all variants) == AnalyzerResult.TotalDemand`.  
**Engine contract**: Engine reads CostPerReplica for scale-up/scale-down ordering. Never reads raw KVUtilization/QueueDepth.

---

### AnalyzerResult

Model-level capacity assessment. Output of Analyzer (one per model per tick). Input to Engine (all models at once).

```
AnalyzerResult
├── ModelID           string
├── TotalSupply       float64          // aggregate serving capacity (model-level)
├── TotalDemand       float64          // aggregate load (model-level)
├── Utilization       float64          // TotalDemand / TotalSupply (0 when TotalSupply==0)
├── RequiredCapacity  float64          // scale-up signal: capacity needed beyond current supply
├── SpareCapacity     float64          // scale-down signal: capacity safely removable
└── VariantCapacities []VariantCapacity
```

**Mutual exclusivity**: `RequiredCapacity > 0` implies `SpareCapacity == 0` and vice versa. Both may be zero (neutral state).  
**Utilization guard**: When `TotalSupply == 0`, `Utilization = 0` (no division).  
**VariantCapacities ordering**: sorted by `CostPerReplica` ascending for determinism (R2).

---

### ScaleDecision

Instruction to change replica count for one model+variant. Output of Engine, input to Actuator.

```
ScaleDecision
├── ModelID string
├── Variant  VariantSpec
└── Delta    int          // +N = scale up by N, -N = scale down by N
```

**Constraint**: `Delta != 0` always. Engine emits at most one ScaleDecision per ModelID per Optimize() call.  
**No up+down**: For a given ModelID, only scale-up (Delta > 0) or scale-down (Delta < 0) is emitted, never both.

---

### GPUInventory

Read-only view of available GPU capacity. Passed to Engine.

```
GPUInventory
└── ByVariant map[VariantSpec]int   // free GPU slots per variant
```

**Definition**: `ByVariant[v] = total GPU slots for v - slots held by Running instances of v - slots held by Loading instances of v`.  
Pending placements NOT subtracted. Draining instances ARE subtracted (hold GPUs until drain completes).  
**Map iteration**: callers must sort keys before iterating (R2).

---

## New Event Types

Added to `sim/cluster/cluster_event.go`.

### ScalingTickEvent

Fires the autoscaling pipeline at the configured interval. Endogenous (state-driven, self-scheduling).

```
ScalingTickEvent
└── At int64   // simulation timestamp (microseconds)
```

**Priority**: 8 (after all request-path events at same timestamp).  
**Self-scheduling**: On execution, schedules the next `ScalingTickEvent{At: now + ModelAutoscalerIntervalUs}`.  
**Zero-interval guard**: When `ModelAutoscalerIntervalUs == 0`, no tick is ever scheduled. No first event emitted at t=0.

---

### ScaleActuationEvent

Carries scale decisions to apply after the actuation delay elapses. Separates the "decide" step from the "act" step to model HPA/KEDA scrape lag.

```
ScaleActuationEvent
├── At        int64
└── Decisions []ScaleDecision
```

**Priority**: 9 (after ScalingTickEvent).  
**Delay**: Scheduled at `now + ActuationDelay.Sample(rng)`. With `ActuationDelay = {Mean:0, Stddev:0}` (default), fires in the same tick.

---

## Modified Types

### RoutingSnapshot (sim/router_state.go) — additive fields

```diff
 type RoutingSnapshot struct {
     ID               string
     QueueDepth       int
     BatchSize        int
     KVUtilization    float64
     FreeKVBlocks     int64
     CacheHitRate     float64
     InFlightRequests int
     Model            string
+    GPUType          string   // populated by buildRouterState() from instance config
+    TPDegree         int      // populated by buildRouterState() from instance config
+    CostPerHour      float64  // populated by buildRouterState() from NodePool.CostPerHour
 }
```

**Population**: `buildRouterState()` in `cluster_event.go` already iterates instances and can read these fields from instance config.

---

### NodePoolConfig (sim/cluster/infra_config.go) — additive field

```diff
 type NodePoolConfig struct {
     Name              string
     GPUType           string
     GPUsPerNode       int
     GPUMemoryGiB      float64
     InitialNodes      int
     MinNodes          int
     MaxNodes          int
     ProvisioningDelay DelaySpec
+    CostPerHour       float64  // $/hr per node; used for CostPerReplica in VariantCapacity
 }
```

**Validation**: `CostPerHour ≥ 0`. NaN/Inf must be rejected. Zero is valid (free tier).

---

### DeploymentConfig (sim/cluster/deployment.go) — additive fields

```diff
 // Autoscaler pipeline (Phase 1C)
+ModelAutoscalerIntervalUs float64   // tick interval in μs; 0 = autoscaler disabled
+ActuationDelay            DelaySpec // HPA/KEDA scrape lag; zero = same-tick actuation
+ScaleUpCooldownUs         float64   // min μs between scale-up decisions per model
+ScaleDownCooldownUs       float64   // min μs between scale-down decisions per model
```

**Wiring guards**: If `ModelAutoscalerIntervalUs == 0`, no ScalingTickEvent is ever scheduled. All four new fields go in `DeploymentConfig`, not `SimConfig` (R16).

---

## Relationships

```
NodePoolConfig.CostPerHour
        │
        ▼ buildRouterState()
RoutingSnapshot.{GPUType, TPDegree, CostPerHour}
        │
        ▼ DefaultCollector.Collect()
ReplicaMetrics.{Variant, CostPerHour}
        │ (grouped by ModelID)
        ▼
ModelSignals.Replicas[]
        │
        ▼ Analyzer.Analyze() — one per model
AnalyzerResult.{TotalSupply, TotalDemand, RequiredCapacity, SpareCapacity}
AnalyzerResult.VariantCapacities[].{Variant, Supply, Demand, ReplicaCount, CostPerReplica}
        │
        ▼ Engine.Optimize() — all models + GPUInventory
ScaleDecision[].{ModelID, Variant, Delta}
        │
        ▼ (ActuationDelay elapses — ScaleActuationEvent)
        ▼ Actuator.Apply()
PlacementManager.PlaceInstance() / instance state → Draining
```
