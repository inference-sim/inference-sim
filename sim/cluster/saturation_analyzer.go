// saturation_analyzer.go implements the V2 token-based saturation analyzer,
// adapted from llm-d WVA's internal/engines/analyzers/saturation_v2.
// Capacity is measured in token units: k1 (memory-bound) and k2 (compute-bound).
// Effective capacity = min(k1, k2). Demand = tokensInUse + queueLength * avgInputTokens.
package cluster

import (
	"fmt"
	"math"
	"sort"

	"github.com/sirupsen/logrus"
)

// V2SaturationAnalyzerConfig configures the V2 token-based saturation analyzer.
type V2SaturationAnalyzerConfig struct {
	KvCacheThreshold  float64 // Fraction of KV capacity considered usable (0,1]; k1 = totalKvCapTokens * this
	ScaleUpThreshold  float64 // Utilization fraction triggering scale-up; RequiredCapacity = max(0, totalDemand/this - (totalReadySupply + pendingSupply)) where pendingSupply = PendingTotalKvCapacityTokens * KvCacheThreshold
	ScaleDownBoundary float64 // Utilization fraction below which scale-down is safe; SpareCapacity = max(0, totalSupply - totalDemand/this)
	AvgInputTokens    float64 // Average input tokens per request; used to convert queue depth to token demand
}

// V2SaturationAnalyzer implements the Analyzer interface using WVA V2's
// token-based capacity model. Per-replica capacity = min(k1, k2) where
// k1 = TotalKvCapacityTokens * KvCacheThreshold (memory-bound) and
// k2 is derived from batch parameters or falls back to k1.
type V2SaturationAnalyzer struct {
	config V2SaturationAnalyzerConfig
}

// NewV2SaturationAnalyzer constructs a V2SaturationAnalyzer. Panics on invalid config (R4).
func NewV2SaturationAnalyzer(cfg V2SaturationAnalyzerConfig) *V2SaturationAnalyzer {
	if cfg.KvCacheThreshold <= 0 || cfg.KvCacheThreshold > 1.0 || math.IsNaN(cfg.KvCacheThreshold) || math.IsInf(cfg.KvCacheThreshold, 0) {
		panic(fmt.Sprintf("NewV2SaturationAnalyzer: KvCacheThreshold must be in (0, 1.0], got %f", cfg.KvCacheThreshold))
	}
	if cfg.ScaleUpThreshold <= 0 || math.IsNaN(cfg.ScaleUpThreshold) || math.IsInf(cfg.ScaleUpThreshold, 0) {
		panic(fmt.Sprintf("NewV2SaturationAnalyzer: ScaleUpThreshold must be > 0, got %f", cfg.ScaleUpThreshold))
	}
	if cfg.ScaleDownBoundary <= 0 || math.IsNaN(cfg.ScaleDownBoundary) || math.IsInf(cfg.ScaleDownBoundary, 0) {
		panic(fmt.Sprintf("NewV2SaturationAnalyzer: ScaleDownBoundary must be > 0, got %f", cfg.ScaleDownBoundary))
	}
	if cfg.AvgInputTokens <= 0 || math.IsNaN(cfg.AvgInputTokens) || math.IsInf(cfg.AvgInputTokens, 0) {
		panic(fmt.Sprintf("NewV2SaturationAnalyzer: AvgInputTokens must be > 0, got %f", cfg.AvgInputTokens))
	}
	if cfg.ScaleDownBoundary >= cfg.ScaleUpThreshold {
		panic(fmt.Sprintf("NewV2SaturationAnalyzer: ScaleDownBoundary (%f) must be < ScaleUpThreshold (%f)",
			cfg.ScaleDownBoundary, cfg.ScaleUpThreshold))
	}
	return &V2SaturationAnalyzer{config: cfg}
}

// Name returns the analyzer name for observability.
func (a *V2SaturationAnalyzer) Name() string { return "v2-saturation" }

// Analyze computes model-level supply and demand in token units.
// Per-replica effective capacity = min(k1, k2) where:
//   - k1 (memory-bound) = TotalKvCapacityTokens * KvCacheThreshold
//   - k2 (compute-bound) = k1 fallback (initial implementation; future: derived from batch params)
//
// Demand per replica = KvTokensInUse + QueueDepth * AvgInputTokens.
// Model-level RequiredCapacity and SpareCapacity follow WVA V2 formulas.
func (a *V2SaturationAnalyzer) Analyze(metrics ModelSignals) AnalyzerResult {
	result := AnalyzerResult{ModelID: metrics.ModelID}

	if len(metrics.Replicas) == 0 {
		if metrics.PendingReplicaCount > 0 {
			logrus.Debugf("[analyzer] model %q: no routable replicas, %d pending — demand is zero (no ready replicas to measure from); pending supply not evaluated",
				metrics.ModelID, metrics.PendingReplicaCount)
		}
		return result
	}

	// Group replicas by variant for per-variant aggregation
	type variantAgg struct {
		supply       float64
		demand       float64
		replicaCount int
		costPerHour  float64
	}
	variants := make(map[VariantSpec]*variantAgg)

	for _, r := range metrics.Replicas {
		// Skip replicas with uninitialized KV cache. Including them would produce zero supply
		// with nonzero demand, triggering runaway scale-up. This should not occur for routable
		// replicas — Loading instances are excluded by buildRouterState before reaching here.
		if r.TotalKvCapacityTokens <= 0 {
			logrus.Warnf("[analyzer] model %q: routable replica %q has zero TotalKvCapacityTokens — excluded from supply/demand; check KV cache initialization",
				metrics.ModelID, r.InstanceID)
			continue
		}
		// k1: memory-bound capacity
		k1 := float64(r.TotalKvCapacityTokens) * a.config.KvCacheThreshold
		// k2: compute-bound capacity — falls back to k1 in initial implementation
		k2 := k1
		effectiveCapacity := math.Min(k1, k2)

		demand := float64(r.KvTokensInUse) + float64(r.QueueDepth)*a.config.AvgInputTokens

		agg, ok := variants[r.Variant]
		if !ok {
			agg = &variantAgg{costPerHour: r.CostPerHour}
			variants[r.Variant] = agg
		}
		agg.supply += effectiveCapacity
		agg.demand += demand
		agg.replicaCount++
	}

	// Build VariantCapacities sorted by CostPerReplica ascending (R2 determinism).
	// Float accumulation for TotalSupply/TotalDemand happens AFTER sorting to ensure
	// deterministic addition order (R2: IEEE 754 float addition is not associative).
	vcs := make([]VariantCapacity, 0, len(variants))
	for v, agg := range variants {
		vcs = append(vcs, VariantCapacity{
			Variant:        v,
			Supply:         agg.supply,
			Demand:         agg.demand,
			ReplicaCount:   agg.replicaCount,
			CostPerReplica: agg.costPerHour,
		})
	}
	sort.Slice(vcs, func(i, j int) bool {
		if vcs[i].CostPerReplica != vcs[j].CostPerReplica {
			return vcs[i].CostPerReplica < vcs[j].CostPerReplica
		}
		if vcs[i].Variant.GPUType != vcs[j].Variant.GPUType {
			return vcs[i].Variant.GPUType < vcs[j].Variant.GPUType
		}
		return vcs[i].Variant.TPDegree < vcs[j].Variant.TPDegree
	})
	result.VariantCapacities = vcs

	// Accumulate from sorted slice for deterministic float summation (R2, INV-6)
	for _, vc := range vcs {
		result.TotalSupply += vc.Supply
		result.TotalDemand += vc.Demand
	}

	// Utilization guard (R11)
	if result.TotalSupply > 0 {
		result.Utilization = result.TotalDemand / result.TotalSupply
	}

	// Scale-up signal: include pending (Loading instance) capacity alongside ready supply.
	// Pending supply covers anticipated capacity once Loading instances finish loading.
	// Only affects RequiredCapacity — TotalSupply, Utilization, and SpareCapacity are
	// based on ready replicas only (no premature scale-down risk from unstarted instances).
	// Matches WVA saturation_v2 anticipatedSupply semantics.
	var pendingSupply float64
	if metrics.PendingTotalKvCapacityTokens > 0 {
		pendingSupply = float64(metrics.PendingTotalKvCapacityTokens) * a.config.KvCacheThreshold
	}
	totalSupplyForScaleUp := result.TotalSupply + pendingSupply
	requiredCapacity := (result.TotalDemand / a.config.ScaleUpThreshold) - totalSupplyForScaleUp
	if requiredCapacity > 0 {
		result.RequiredCapacity = requiredCapacity
		// Mutual exclusivity: if scaling up, no spare capacity
		return result
	}

	// Scale-down signal with N-1 redistribution check:
	// Can only scale down if we have > 1 initialized replica AND redistributing load
	// across N-1 replicas still leaves supply above demand/ScaleDownBoundary.
	// Count only replicas that contributed to supply (TotalKvCapacityTokens > 0).
	initReplicas := 0
	for _, vc := range vcs {
		initReplicas += vc.ReplicaCount
	}
	if initReplicas <= 1 {
		return result
	}

	spareCapacity := result.TotalSupply - (result.TotalDemand / a.config.ScaleDownBoundary)
	if spareCapacity <= 0 {
		return result
	}

	// N-1 redistribution check: simulate removing the replica with the highest
	// per-replica capacity (conservative). This matches UnlimitedEngine's scale-down
	// behavior which targets the most expensive variant — typically the highest-capacity one.
	// Using max ensures the safety check holds regardless of which variant the Engine removes.
	maxPerReplicaSupply := 0.0
	for _, vc := range vcs {
		if vc.ReplicaCount > 0 {
			perReplica := vc.Supply / float64(vc.ReplicaCount)
			if perReplica > maxPerReplicaSupply {
				maxPerReplicaSupply = perReplica
			}
		}
	}

	supplyAfterRemoval := result.TotalSupply - maxPerReplicaSupply
	if supplyAfterRemoval > (result.TotalDemand / a.config.ScaleDownBoundary) {
		result.SpareCapacity = spareCapacity
	}

	return result
}
