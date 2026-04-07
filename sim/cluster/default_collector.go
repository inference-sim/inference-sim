// default_collector.go implements DefaultCollector — the standard Collector that maps
// RouterState snapshots into per-model ModelSignals for the autoscaler pipeline.
package cluster

import (
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// DefaultCollector maps RoutingSnapshot fields to ReplicaMetrics, grouping by model.
// Pure function: no filtering, no thresholding, no modification of signals.
// TTFT and DispatchRate are set to zero (future: QueueingModelAnalyzer).
type DefaultCollector struct{}

// Collect produces one ModelSignals per active model from the current RouterState.
// Models are sorted alphabetically for determinism (R2).
func (c *DefaultCollector) Collect(state *sim.RouterState) []ModelSignals {
	if state == nil || len(state.Snapshots) == 0 {
		return nil
	}

	// Group snapshots by model
	byModel := make(map[string][]ReplicaMetrics)
	for _, snap := range state.Snapshots {
		rm := ReplicaMetrics{
			InstanceID:            snap.ID,
			Variant:               VariantSpec{GPUType: snap.GPUType, TPDegree: max(snap.TPDegree, 1)},
			KVUtilization:         snap.KVUtilization,
			QueueDepth:            snap.QueueDepth,
			InFlightCount:         snap.InFlightRequests,
			CostPerHour:           snap.CostPerHour,
			TotalKvCapacityTokens: snap.TotalKvCapacityTokens,
			KvTokensInUse:         snap.KvTokensInUse,
		}
		byModel[snap.Model] = append(byModel[snap.Model], rm)
	}

	// Sort model keys for determinism (R2)
	models := make([]string, 0, len(byModel))
	for m := range byModel {
		models = append(models, m)
	}
	sort.Strings(models)

	result := make([]ModelSignals, 0, len(models))
	for _, m := range models {
		result = append(result, ModelSignals{
			ModelID:  m,
			Replicas: byModel[m],
		})
	}
	return result
}
