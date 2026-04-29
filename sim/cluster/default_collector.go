// default_collector.go implements DefaultCollector — the standard Collector that maps
// RouterState snapshots into per-model ModelSignals for the autoscaler pipeline.
package cluster

import (
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// DefaultCollector maps RoutingSnapshot fields to ReplicaMetrics, grouping by model.
// Filters structurally incomplete snapshots (empty Model or GPUType) with a Debugf log;
// all valid signals pass through unmodified — no thresholding, no business-logic suppression.
// TTFT and DispatchRate are set to zero (future: QueueingModelAnalyzer).
type DefaultCollector struct{}

// Collect produces one ModelSignals per model present in either routable or loading snapshots.
// Models are sorted alphabetically for determinism (R2).
func (c *DefaultCollector) Collect(state *sim.RouterState) []ModelSignals {
	if state == nil {
		return nil
	}

	// Group routable snapshots by model; skip instances with empty Model or GPUType.
	byModel := make(map[string][]ReplicaMetrics)
	for _, snap := range state.Snapshots {
		if snap.Model == "" {
			logrus.Debugf("[collector] skipping snapshot %q: empty Model field", snap.ID)
			continue
		}
		if snap.GPUType == "" {
			logrus.Debugf("[collector] skipping snapshot %q: empty GPUType field", snap.ID)
			continue
		}
		rm := ReplicaMetrics{
			InstanceID:            snap.ID,
			Variant:               NewVariantSpec(snap.GPUType, max(snap.TPDegree, 1)),
			KVUtilization:         snap.KVUtilization,
			QueueDepth:            snap.QueueDepth,
			InFlightCount:         snap.InFlightRequests,
			CostPerHour:           snap.CostPerHour,
			TotalKvCapacityTokens: snap.TotalKvCapacityTokens,
			KvTokensInUse:         snap.KvTokensInUse,
		}
		byModel[snap.Model] = append(byModel[snap.Model], rm)
	}

	// Group Loading instance capacity by model for pending supply estimation.
	type pendingAgg struct {
		count    int
		capacity int64
	}
	pendingByModel := make(map[string]pendingAgg)
	for _, snap := range state.LoadingSnapshots {
		if snap.Model == "" {
			logrus.Debugf("[collector] skipping loading snapshot %q: empty Model field", snap.ID)
			continue
		}
		if snap.GPUType == "" {
			logrus.Debugf("[collector] skipping loading snapshot %q: empty GPUType field", snap.ID)
			continue
		}
		if snap.TotalKvCapacityTokens <= 0 {
			logrus.Debugf("[collector] skipping loading snapshot %q model %q: TotalKvCapacityTokens <= 0",
				snap.ID, snap.Model)
			continue
		}
		p := pendingByModel[snap.Model]
		p.count++
		p.capacity += snap.TotalKvCapacityTokens
		pendingByModel[snap.Model] = p
	}

	// Collect all model keys from both routable and loading snapshots.
	allModels := make(map[string]struct{}, len(byModel)+len(pendingByModel))
	for m := range byModel {
		allModels[m] = struct{}{}
	}
	for m := range pendingByModel {
		allModels[m] = struct{}{}
	}

	if len(allModels) == 0 {
		return nil
	}

	// Sort model keys for determinism (R2).
	models := make([]string, 0, len(allModels))
	for m := range allModels {
		models = append(models, m)
	}
	sort.Strings(models)

	result := make([]ModelSignals, 0, len(models))
	for _, m := range models {
		ms := ModelSignals{
			ModelID:  m,
			Replicas: byModel[m],
		}
		if p, ok := pendingByModel[m]; ok {
			ms.PendingReplicaCount = p.count
			ms.PendingTotalKvCapacityTokens = p.capacity
		}
		result = append(result, ms)
	}
	return result
}
