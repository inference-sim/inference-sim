// engine.go implements Engine interface variants for the autoscaler pipeline.
// UnlimitedEngine (1C-1b): ignores GPU inventory, for fixed-node testing.
// GreedyEngine (1C-1d): respects GPU inventory, for production-like allocation.
package cluster

import (
	"math"
	"sort"
)

// UnlimitedEngine converts analyzer signals into scale decisions without checking
// GPU inventory. Scale-up targets cheapest variant; scale-down targets most expensive.
// At most one decision per model per call. Inventory parameter accepted but not used.
type UnlimitedEngine struct{}

// Optimize produces scale decisions for all models. Ignores GPU inventory.
func (e *UnlimitedEngine) Optimize(results []AnalyzerResult, _ GPUInventory) []ScaleDecision {
	var decisions []ScaleDecision

	for _, r := range results {
		if r.RequiredCapacity > 0 {
			// Scale up: pick cheapest variant, exact-N replicas
			vcs := sortedByAscCost(r.VariantCapacities)
			if len(vcs) > 0 {
				decisions = append(decisions, ScaleDecision{
					ModelID: r.ModelID,
					Variant: vcs[0].Variant,
					Delta:   scaleUpN(r.RequiredCapacity, vcs[0:1]),
				})
			}
			continue
		}
		if r.SpareCapacity > 0 {
			// Scale down: pick most expensive variant with active replicas, exact-N replicas.
			vcs := sortedByDescCost(r.VariantCapacities)
			for _, vc := range vcs {
				if vc.ReplicaCount > 0 {
					// A zero count means the check proved no removal safe: emit nothing (INV-19).
					if n := scaleDownN(r.SpareCapacity, vc, r.SafeRemovals); n > 0 {
						decisions = append(decisions, ScaleDecision{ModelID: r.ModelID, Variant: vc.Variant, Delta: -n})
					}
					break
				}
			}
		}
	}
	return decisions
}

// GreedyEngine converts analyzer signals into scale decisions while respecting GPU inventory.
// Scale-up targets cheapest variant with sufficient free slots; scale-down targets most expensive.
// At most one decision per model per call.
type GreedyEngine struct{}

// Optimize produces scale decisions for all models, respecting GPU inventory for scale-up.
func (e *GreedyEngine) Optimize(results []AnalyzerResult, inventory GPUInventory) []ScaleDecision {
	var decisions []ScaleDecision

	for _, r := range results {
		if r.RequiredCapacity > 0 {
			// Scale up: pick cheapest variant that has enough free GPU slots.
			// n is computed per-variant using that variant's own per-replica capacity
			// so that a more capable fallback variant does not get overscaled.
			vcs := sortedByAscCost(r.VariantCapacities)
			for _, vc := range vcs {
				n := scaleUpN(r.RequiredCapacity, []VariantCapacity{vc})
				needed := n * vc.Variant.TPDegree
				if needed < 1 {
					needed = 1
				}
				if inventory.FreeSlots(vc.Variant) >= needed {
					decisions = append(decisions, ScaleDecision{
						ModelID: r.ModelID,
						Variant: vc.Variant,
						Delta:   n,
					})
					break
				}
			}
			continue
		}
		if r.SpareCapacity > 0 {
			// Scale down: pick most expensive variant with active replicas, exact-N replicas
			vcs := sortedByDescCost(r.VariantCapacities)
			for _, vc := range vcs {
				if vc.ReplicaCount > 0 {
					// A zero count means the check proved no removal safe: emit nothing (INV-19).
					if n := scaleDownN(r.SpareCapacity, vc, r.SafeRemovals); n > 0 {
						decisions = append(decisions, ScaleDecision{ModelID: r.ModelID, Variant: vc.Variant, Delta: -n})
					}
					break
				}
			}
		}
	}
	return decisions
}

// scaleUpN returns exact replicas to add for the selected variant (vcs[0]).
// Uses Supply/ReplicaCount from the first active variant in vcs as perReplicaCapacity.
// Falls back to 1 when no active variant exists (perReplicaCapacity == 0).
func scaleUpN(requiredCapacity float64, vcs []VariantCapacity) int {
	prc := perReplicaCapacityForScaleUp(vcs)
	if prc <= 0 {
		return 1
	}
	n := int(math.Ceil(requiredCapacity / prc))
	if n < 1 {
		n = 1
	}
	return n
}

// scaleDownN returns how many replicas to remove from vc: floor(spareCapacity / per-replica
// capacity), at least 1, and at most both vc.ReplicaCount and safeRemovals, the count the
// Analyzer's check proved safe (AnalyzerResult.SafeRemovals). It returns 0 when vc has no active
// replicas or safeRemovals <= 0; the caller then emits no decision (INV-19). Without the
// safeRemovals bound an idle start, whose spare capacity is the whole supply, drained every
// replica at the first tick (#2015).
func scaleDownN(spareCapacity float64, vc VariantCapacity, safeRemovals int) int {
	if vc.ReplicaCount <= 0 || safeRemovals <= 0 {
		return 0
	}
	n := 1
	if vc.Supply > 0 {
		n = max(1, int(math.Floor(spareCapacity/(vc.Supply/float64(vc.ReplicaCount)))))
	}
	return min(n, vc.ReplicaCount, safeRemovals)
}

// perReplicaCapacityForScaleUp returns Supply/ReplicaCount from the first active variant in vcs.
// Returns 0 when no active variant exists.
func perReplicaCapacityForScaleUp(vcs []VariantCapacity) float64 {
	for _, vc := range vcs {
		if vc.ReplicaCount > 0 && vc.Supply > 0 {
			return vc.Supply / float64(vc.ReplicaCount)
		}
	}
	return 0
}

// sortedByAscCost returns a copy sorted by CostPerReplica ascending, then GPUType, then TPDegree (R2).
func sortedByAscCost(vcs []VariantCapacity) []VariantCapacity {
	out := make([]VariantCapacity, len(vcs))
	copy(out, vcs)
	sort.Slice(out, func(i, j int) bool {
		if out[i].CostPerReplica != out[j].CostPerReplica {
			return out[i].CostPerReplica < out[j].CostPerReplica
		}
		if out[i].Variant.GPUType != out[j].Variant.GPUType {
			return out[i].Variant.GPUType < out[j].Variant.GPUType
		}
		return out[i].Variant.TPDegree < out[j].Variant.TPDegree
	})
	return out
}

// sortedByDescCost returns a copy sorted by CostPerReplica descending, then GPUType, then TPDegree (R2).
func sortedByDescCost(vcs []VariantCapacity) []VariantCapacity {
	out := make([]VariantCapacity, len(vcs))
	copy(out, vcs)
	sort.Slice(out, func(i, j int) bool {
		if out[i].CostPerReplica != out[j].CostPerReplica {
			return out[i].CostPerReplica > out[j].CostPerReplica
		}
		if out[i].Variant.GPUType != out[j].Variant.GPUType {
			return out[i].Variant.GPUType < out[j].Variant.GPUType
		}
		return out[i].Variant.TPDegree < out[j].Variant.TPDegree
	})
	return out
}
