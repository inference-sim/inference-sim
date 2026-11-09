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
	// MinReplicas is the fewest ready replicas a model keeps; scale-down never goes below it.
	// It is at least 1: scale-to-zero is a separate, opt-in drain (#909).
	MinReplicas int
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
	if cfg.MinReplicas < 1 {
		panic(fmt.Sprintf("NewV2SaturationAnalyzer: MinReplicas must be >= 1, got %d", cfg.MinReplicas))
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
		// replicas — Loading instances never appear in metrics.Replicas; they are collected into
		// RouterState.LoadingSnapshots and surfaced as PendingReplicaCount/PendingTotalKvCapacityTokens.
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
	if pendingSupply > 0 {
		logrus.Debugf("[analyzer] model %q: scale-up suppressed by pending supply (readySupply=%.0f pendingSupply=%.0f demand=%.0f)",
			metrics.ModelID, result.TotalSupply, pendingSupply, result.TotalDemand)
	}

	// Scale-down signal. The check proves how many replicas can go: it removes replicas one at
	// a time, the one with the highest per-replica capacity first (the conservative order, since
	// the Engine may remove any variant), and counts a removal as safe only while the supply left
	// stays above TotalDemand/ScaleDownBoundary and at least MinReplicas ready replicas remain.
	// SafeRemovals is that count; the Engine removes no more (#2015). At zero demand every
	// removal passes the supply test, so MinReplicas alone bounds the count.
	need := result.TotalDemand / a.config.ScaleDownBoundary
	spareCapacity := result.TotalSupply - need
	if spareCapacity <= 0 {
		return result
	}
	perReplica := make([]float64, 0, len(metrics.Replicas))
	for _, vc := range vcs {
		for i := 0; i < vc.ReplicaCount; i++ {
			perReplica = append(perReplica, vc.Supply/float64(vc.ReplicaCount))
		}
	}
	sort.SliceStable(perReplica, func(i, j int) bool { return perReplica[i] > perReplica[j] })
	remaining := result.TotalSupply
	safe := 0
	for _, c := range perReplica {
		if len(perReplica)-safe <= a.config.MinReplicas || remaining-c <= need {
			break
		}
		remaining -= c
		safe++
	}
	if safe > 0 {
		result.SpareCapacity = spareCapacity
		result.SafeRemovals = safe
	}

	return result
}
