// engine.go implements Engine interface variants for the autoscaler pipeline.
// UnlimitedEngine (1C-1b): ignores GPU inventory, for fixed-node testing.
// GreedyEngine (1C-1d): respects GPU inventory, for production-like allocation.
package cluster

import "sort"

// UnlimitedEngine converts analyzer signals into scale decisions without checking
// GPU inventory. Scale-up targets cheapest variant; scale-down targets most expensive.
// At most one decision per model per call. Inventory parameter accepted but not used.
type UnlimitedEngine struct{}

// Optimize produces scale decisions for all models. Ignores GPU inventory.
func (e *UnlimitedEngine) Optimize(results []AnalyzerResult, _ GPUInventory) []ScaleDecision {
	var decisions []ScaleDecision

	for _, r := range results {
		if r.RequiredCapacity > 0 {
			// Scale up: pick cheapest variant
			vcs := sortedByAscCost(r.VariantCapacities)
			if len(vcs) > 0 {
				decisions = append(decisions, ScaleDecision{
					ModelID: r.ModelID,
					Variant: vcs[0].Variant,
					Delta:   1,
				})
			}
			continue
		}
		if r.SpareCapacity > 0 {
			// Scale down: pick most expensive variant with active replicas
			vcs := sortedByDescCost(r.VariantCapacities)
			for _, vc := range vcs {
				if vc.ReplicaCount > 0 {
					decisions = append(decisions, ScaleDecision{
						ModelID: r.ModelID,
						Variant: vc.Variant,
						Delta:   -1,
					})
					break
				}
			}
		}
	}
	return decisions
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
