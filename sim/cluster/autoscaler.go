// autoscaler.go defines the Phase 1C model autoscaler types, interfaces, and implementations.
// Pipeline: Collector → Analyzer → Engine → Actuator, orchestrated by ScalingTickEvent.Execute().
// All types live here; events live in cluster_event.go.
package cluster

// VariantSpec identifies a specific hardware configuration for a replica.
// Used as a map key in GPUInventory.ByVariant and carried in ScaleDecision.Variant.
// Both fields are comparable types (string, int) — safe for use as a Go map key.
type VariantSpec struct {
	GPUType  string // e.g. "A100-80GB", "H100-80GB"
	TPDegree int    // tensor-parallel degree: 1, 2, 4, 8; must be ≥1
}

// ReplicaMetrics is a snapshot of one replica's observable state at collection time.
// Produced by Collector, consumed by Analyzer. All numeric invariants must hold:
// KVUtilization ∈ [0.0, 1.0], QueueDepth ≥ 0, InFlightCount ≥ 0.
type ReplicaMetrics struct {
	InstanceID    string
	Variant       VariantSpec
	KVUtilization float64 // [0.0, 1.0]
	QueueDepth    int
	InFlightCount int
	CostPerHour   float64 // $/hr from NodePool; used for CostPerReplica in VariantCapacity
	TTFT          float64 // μs — zero in DefaultCollector (future: QueueingModelAnalyzer)
	DispatchRate  float64 // req/s — zero in DefaultCollector (future: QueueingModelAnalyzer)
}

// ModelSignals aggregates all replica snapshots for one model.
// Output of Collector.Collect(). Input to Analyzer.Analyze() (one call per model).
// Replicas may be empty (zero-replica model); Analyzer must handle this without panic.
type ModelSignals struct {
	ModelID  string
	Replicas []ReplicaMetrics // may be empty
}

// VariantCapacity is one variant's share of a model's total supply and demand.
// Used by Engine to select the allocation target for scale-up and scale-down.
// Invariant: sum(VariantCapacity.Supply over all variants) == AnalyzerResult.TotalSupply.
type VariantCapacity struct {
	Variant        VariantSpec
	Supply         float64 // this variant's contribution to TotalSupply
	Demand         float64 // this variant's share of TotalDemand
	ReplicaCount   int     // active replicas of this variant serving this model
	CostPerReplica float64 // from ReplicaMetrics.CostPerHour for replicas of this variant
}

// AnalyzerResult is a model-level capacity assessment.
// Output of Analyzer.Analyze() (one per model per tick). Input to Engine.Optimize() (all models at once).
// Mutual exclusivity: RequiredCapacity > 0 implies SpareCapacity == 0 and vice versa.
// Utilization guard: when TotalSupply == 0, Utilization = 0 (no division).
type AnalyzerResult struct {
	ModelID           string
	TotalSupply       float64          // aggregate serving capacity (model-level)
	TotalDemand       float64          // aggregate load (model-level)
	Utilization       float64          // TotalDemand / TotalSupply; 0 when TotalSupply == 0
	RequiredCapacity  float64          // scale-up signal: capacity needed beyond current supply
	SpareCapacity     float64          // scale-down signal: capacity safely removable
	VariantCapacities []VariantCapacity // sorted by CostPerReplica ascending for determinism (R2)
}

// ScaleDecision instructs the Actuator to change replica count for one model+variant.
// Delta != 0 always. Engine emits at most one ScaleDecision per ModelID per Optimize() call.
// No up+down for the same ModelID in one call (Delta > 0 XOR Delta < 0).
type ScaleDecision struct {
	ModelID string
	Variant VariantSpec
	Delta   int // +N = scale up by N replicas, -N = scale down by N replicas; never 0
}

// GPUInventory is a read-only view of available GPU capacity, passed to Engine.Optimize().
// ByVariant[v] = total GPU slots for v - slots held by Running instances - slots held by Loading instances.
// Pending placements are NOT subtracted. Draining instances ARE subtracted (hold GPUs until drain completes).
// Callers must sort keys before iterating (R2: map iteration is non-deterministic).
type GPUInventory struct {
	ByVariant map[VariantSpec]int // free GPU slots per variant
}
