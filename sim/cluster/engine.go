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
					Delta:   scaleUpN(r.RequiredCapacity, vcs),
				})
			}
			continue
		}
		if r.SpareCapacity > 0 {
			// Scale down: pick most expensive variant with active replicas, exact-N replicas.
			// Falls back to first desc-cost variant (Delta=-1) when no active replicas exist.
			vcs := sortedByDescCost(r.VariantCapacities)
			chosen := -1
			for i, vc := range vcs {
				if vc.ReplicaCount > 0 {
					chosen = i
					break
				}
			}
			if chosen < 0 && len(vcs) > 0 {
				chosen = 0 // fallback: first (most expensive) variant
			}
			if chosen >= 0 {
				vc := vcs[chosen]
				decisions = append(decisions, ScaleDecision{
					ModelID: r.ModelID,
					Variant: vc.Variant,
					Delta:   -scaleDownN(r.SpareCapacity, vc),
				})
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
					decisions = append(decisions, ScaleDecision{
						ModelID: r.ModelID,
						Variant: vc.Variant,
						Delta:   -scaleDownN(r.SpareCapacity, vc),
					})
					break
				}
			}
		}
	}
	return decisions
}

// scaleUpN returns exact replicas to add.
// Uses cheapest active variant's Supply/ReplicaCount as perReplicaCapacity.
// Falls back to 1 when no active variant (perReplicaCapacity == 0).
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

// scaleDownN returns exact replicas to remove from vc.
// Uses vc.Supply/vc.ReplicaCount as perReplicaCapacity.
// Falls back to 1 when perReplicaCapacity == 0; clamps to [1, replicaCount].
func scaleDownN(spareCapacity float64, vc VariantCapacity) int {
	if vc.ReplicaCount <= 0 || vc.Supply <= 0 {
		return 1
	}
	prc := vc.Supply / float64(vc.ReplicaCount)
	if prc <= 0 {
		return 1
	}
	n := int(math.Floor(spareCapacity / prc))
	if n < 1 {
		n = 1
	}
	if n > vc.ReplicaCount {
		n = vc.ReplicaCount
	}
	return n
}

// perReplicaCapacityForScaleUp returns Supply/ReplicaCount from the cheapest active variant.
// vcs must be sorted ascending by cost. Returns 0 when no active variant exists.
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
