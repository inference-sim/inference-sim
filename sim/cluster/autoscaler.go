// autoscaler.go defines the Phase 1C model autoscaler types, interfaces, and implementations.
// Pipeline: Collector → Analyzer → Engine → Actuator, orchestrated by ScalingTickEvent.Execute().
// All types live here; events live in cluster_event.go.
package cluster

import (
	"container/heap"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// VariantSpec identifies a specific hardware configuration for a replica.
// Used as a map key in GPUInventory.byVariant and carried in ScaleDecision.Variant.
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
	TotalSupply       float64           // aggregate serving capacity (model-level)
	TotalDemand       float64           // aggregate load (model-level)
	Utilization       float64           // TotalDemand / TotalSupply; 0 when TotalSupply == 0
	RequiredCapacity  float64           // scale-up signal: capacity needed beyond current supply
	SpareCapacity     float64           // scale-down signal: capacity safely removable
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
// byVariant[v] = total GPU slots for v - slots held by Running instances - slots held by Loading instances.
// Pending placements are NOT subtracted. Draining instances ARE subtracted (hold GPUs until drain completes).
// Callers must use FreeSlots() and Variants() to read (R2: map iteration is non-deterministic).
type GPUInventory struct {
	byVariant map[VariantSpec]int // free GPU slots per variant; use FreeSlots() and Variants() to read
}

// FreeSlots returns the free GPU slots for the given variant (0 if variant not in inventory).
func (g GPUInventory) FreeSlots(v VariantSpec) int {
	return g.byVariant[v]
}

// Variants returns all variants in the inventory, sorted ascending by GPUType then TPDegree.
// Callers must use this instead of iterating byVariant directly (R2: deterministic iteration).
func (g GPUInventory) Variants() []VariantSpec {
	vs := make([]VariantSpec, 0, len(g.byVariant))
	for v := range g.byVariant {
		vs = append(vs, v)
	}
	sort.Slice(vs, func(i, j int) bool {
		if vs[i].GPUType != vs[j].GPUType {
			return vs[i].GPUType < vs[j].GPUType
		}
		return vs[i].TPDegree < vs[j].TPDegree
	})
	return vs
}

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

// Collector observes RouterState and produces one ModelSignals per active model,
// grouping replicas by ModelID. Must not modify state, filter models, or apply thresholds.
// Pure function: same RouterState always produces the same output (determinism).
type Collector interface {
	Collect(state *sim.RouterState) []ModelSignals
}

// Analyzer assesses capacity for one model. Name() returns a human-readable identifier.
// Analyze() is called once per model per tick. Must not access RouterState, GPUInventory,
// or any external state — only ModelSignals. Must not panic on empty Replicas slice.
// Invariants: sum(vc.Supply) == TotalSupply; sum(vc.Demand) == TotalDemand.
type Analyzer interface {
	Name() string
	Analyze(metrics ModelSignals) AnalyzerResult
}

// Engine optimizes replica allocation across all models given current GPU inventory.
// Produces at most one ScaleDecision per ModelID per call. Must not emit both scale-up and
// scale-down for the same ModelID. Must not access RouterState or ModelSignals directly.
// Scale-up targets cheapest variant (CostPerReplica ascending); scale-down targets most
// expensive variant (CostPerReplica descending). Map keys must be sorted before iteration (R2).
type Engine interface {
	Optimize(results []AnalyzerResult, inventory GPUInventory) []ScaleDecision
}

// Actuator applies scale decisions to the cluster. Delta > 0 calls PlacementManager.PlaceInstance();
// Delta < 0 transitions the instance to Draining (WaitDrain semantics). Must not block.
// Failure on PlaceInstance is logged, not silently dropped (INV-A2).
// Pending placements for the model must be cancelled before applying scale-down.
// Orchestrator has already applied cooldown filtering before calling Apply().
type Actuator interface {
	Apply(decisions []ScaleDecision) error
}

// ---------------------------------------------------------------------------
// autoscalerPipeline — internal orchestrator (not a public interface)
// ---------------------------------------------------------------------------

// autoscalerPipeline holds the four interface components and cooldown state.
// Owned by ClusterSimulator as an optional field (nil when autoscaler disabled).
// tick() runs the Collect → Analyze → Optimize pipeline and schedules a ScaleActuationEvent.
// actuate() calls Actuator.Apply() with the decisions from a ScaleActuationEvent.
type autoscalerPipeline struct {
	collector Collector
	analyzer  Analyzer
	engine    Engine
	actuator  Actuator

	// Cooldown state (Decision 8): keyed by ModelID.
	// Updated when a decision survives cooldown filtering and is forwarded to ScaleActuationEvent.
	lastScaleUpAt   map[string]int64
	lastScaleDownAt map[string]int64

	// rng is used to sample ActuationDelay (Decision 4).
	rng *rand.Rand
}

// tick executes the autoscaling pipeline for one tick at timestamp nowUs.
// Collect → Analyze → Optimize → cooldown filter → schedule ScaleActuationEvent → schedule next tick.
func (p *autoscalerPipeline) tick(cs *ClusterSimulator, nowUs int64) {
	// Guard: all four components must be wired (nil components skip this tick).
	if p.collector == nil || p.analyzer == nil || p.engine == nil || p.actuator == nil {
		logrus.Warnf("[autoscaler] tick at t=%d skipped: pipeline not fully wired (collector=%v analyzer=%v engine=%v actuator=%v)",
			nowUs, p.collector != nil, p.analyzer != nil, p.engine != nil, p.actuator != nil)
		p.scheduleNextTick(cs, nowUs)
		return
	}

	// Stage 1: Collect — build ModelSignals for each active model.
	routerState := buildRouterState(cs, nil)
	modelSignals := p.collector.Collect(routerState)

	// Stage 2: Analyze — run Analyzer once per model.
	results := make([]AnalyzerResult, 0, len(modelSignals))
	for _, ms := range modelSignals {
		results = append(results, p.analyzer.Analyze(ms))
	}

	// Stage 3: Optimize — ask Engine for scale decisions.
	inventory := cs.gpuInventory()
	decisions := p.engine.Optimize(results, inventory)

	// Stage 4: Cooldown filter — suppress decisions within cooldown window per model.
	filtered := make([]ScaleDecision, 0, len(decisions))
	for _, d := range decisions {
		if d.Delta == 0 {
			logrus.Warnf("[autoscaler] engine emitted ScaleDecision with Delta=0 for model %q — contract violation, skipping", d.ModelID)
			continue
		}
		if d.Delta > 0 {
			cooldown := cs.config.ScaleUpCooldownUs
			if cooldown > 0 && nowUs-p.lastScaleUpAt[d.ModelID] < int64(cooldown) {
				continue // suppressed by scale-up cooldown (INV-A7)
			}
			p.lastScaleUpAt[d.ModelID] = nowUs
		} else {
			cooldown := cs.config.ScaleDownCooldownUs
			if cooldown > 0 && nowUs-p.lastScaleDownAt[d.ModelID] < int64(cooldown) {
				continue // suppressed by scale-down cooldown (INV-A7)
			}
			p.lastScaleDownAt[d.ModelID] = nowUs
		}
		filtered = append(filtered, d)
	}

	// Stage 5: Schedule ScaleActuationEvent (even when filtered is empty, to preserve event semantics).
	// DelaySpec.Sample(rng) is safe with rng=nil when Stddev=0 (constant delay needs no RNG).
	// If Stddev>0 and rng=nil, Sample will panic — that is a configuration error, not a code bug.
	actuationAt := nowUs + cs.config.ActuationDelay.Sample(p.rng)
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &ScaleActuationEvent{At: actuationAt, Decisions: filtered},
		seqID: cs.nextSeqID(),
	})

	// Stage 6: Schedule next ScalingTickEvent.
	p.scheduleNextTick(cs, nowUs)
}

// scheduleNextTick pushes the next ScalingTickEvent to the cluster event queue.
func (p *autoscalerPipeline) scheduleNextTick(cs *ClusterSimulator, nowUs int64) {
	nextAt := nowUs + int64(cs.config.ModelAutoscalerIntervalUs)
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &ScalingTickEvent{At: nextAt},
		seqID: cs.nextSeqID(),
	})
}

// actuate calls Actuator.Apply() with decisions from a ScaleActuationEvent. No-ops when actuator is nil.
func (p *autoscalerPipeline) actuate(_ *ClusterSimulator, decisions []ScaleDecision) {
	if p.actuator == nil {
		if len(decisions) > 0 {
			logrus.Warnf("[autoscaler] actuate: %d decision(s) dropped — actuator not wired", len(decisions))
		}
		return
	}
	if err := p.actuator.Apply(decisions); err != nil {
		logrus.Errorf("[autoscaler] actuate: Apply returned error: %v", err)
	}
}
