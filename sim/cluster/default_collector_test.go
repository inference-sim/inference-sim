package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestDefaultCollectorCollect(t *testing.T) {
	collector := &DefaultCollector{}

	tests := []struct {
		name          string
		state         *sim.RouterState
		wantModels    int
		wantReplicas  map[string]int // model -> expected replica count
		checkFieldMap bool           // verify field mapping for first replica
	}{
		{
			name:       "nil state — empty result",
			state:      nil,
			wantModels: 0,
		},
		{
			name:       "empty snapshots — empty result",
			state:      &sim.RouterState{Snapshots: []sim.RoutingSnapshot{}},
			wantModels: 0,
		},
		{
			name: "two models with correct grouping",
			state: &sim.RouterState{
				Snapshots: []sim.RoutingSnapshot{
					{ID: "i1", Model: "modelA", GPUType: "A100", TPDegree: 1, KVUtilization: 0.5, QueueDepth: 3, InFlightRequests: 2, CostPerHour: 10.0, TotalKvCapacityTokens: 10000, KvTokensInUse: 5000},
					{ID: "i2", Model: "modelA", GPUType: "A100", TPDegree: 1, KVUtilization: 0.3, QueueDepth: 1, InFlightRequests: 1, CostPerHour: 10.0, TotalKvCapacityTokens: 10000, KvTokensInUse: 3000},
					{ID: "i3", Model: "modelB", GPUType: "H100", TPDegree: 2, KVUtilization: 0.7, QueueDepth: 5, InFlightRequests: 4, CostPerHour: 20.0, TotalKvCapacityTokens: 20000, KvTokensInUse: 14000},
				},
			},
			wantModels:    2,
			wantReplicas:  map[string]int{"modelA": 2, "modelB": 1},
			checkFieldMap: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := collector.Collect(tc.state)

			if len(result) != tc.wantModels {
				t.Fatalf("got %d ModelSignals, want %d", len(result), tc.wantModels)
			}

			if tc.wantReplicas != nil {
				for _, ms := range result {
					want, ok := tc.wantReplicas[ms.ModelID]
					if !ok {
						t.Errorf("unexpected model %q in results", ms.ModelID)
						continue
					}
					if len(ms.Replicas) != want {
						t.Errorf("model %q: got %d replicas, want %d", ms.ModelID, len(ms.Replicas), want)
					}
				}
			}

			if tc.checkFieldMap && len(result) > 0 {
				// Verify models are sorted (R2)
				for i := 1; i < len(result); i++ {
					if result[i].ModelID < result[i-1].ModelID {
						t.Errorf("models not sorted: %q before %q", result[i-1].ModelID, result[i].ModelID)
					}
				}

				// Check field mapping for modelB's replica
				for _, ms := range result {
					if ms.ModelID != "modelB" {
						continue
					}
					r := ms.Replicas[0]
					if r.InstanceID != "i3" {
						t.Errorf("InstanceID = %q, want i3", r.InstanceID)
					}
					if r.Variant.GPUType != "H100" || r.Variant.TPDegree != 2 {
						t.Errorf("Variant = %+v, want {H100, 2}", r.Variant)
					}
					if r.KVUtilization != 0.7 {
						t.Errorf("KVUtilization = %f, want 0.7", r.KVUtilization)
					}
					if r.QueueDepth != 5 {
						t.Errorf("QueueDepth = %d, want 5", r.QueueDepth)
					}
					if r.InFlightCount != 4 {
						t.Errorf("InFlightCount = %d, want 4", r.InFlightCount)
					}
					if r.CostPerHour != 20.0 {
						t.Errorf("CostPerHour = %f, want 20.0", r.CostPerHour)
					}
					if r.TotalKvCapacityTokens != 20000 {
						t.Errorf("TotalKvCapacityTokens = %d, want 20000", r.TotalKvCapacityTokens)
					}
					if r.KvTokensInUse != 14000 {
						t.Errorf("KvTokensInUse = %d, want 14000", r.KvTokensInUse)
					}
					if r.TTFT != 0 {
						t.Errorf("TTFT = %f, want 0 (future QueueingModelAnalyzer)", r.TTFT)
					}
					if r.DispatchRate != 0 {
						t.Errorf("DispatchRate = %f, want 0 (future QueueingModelAnalyzer)", r.DispatchRate)
					}
				}
			}
		})
	}
}

// TestDefaultCollector_PendingReplicaCount_FromLoadingSnapshots verifies that
// DefaultCollector populates PendingReplicaCount and PendingTotalKvCapacityTokens
// from RouterState.LoadingSnapshots.
func TestDefaultCollector_PendingReplicaCount_FromLoadingSnapshots(t *testing.T) {
	collector := &DefaultCollector{}

	tests := []struct {
		name                        string
		state                       *sim.RouterState
		wantModels                  int
		wantPendingCount            map[string]int
		wantPendingKvCapacityTokens map[string]int64
	}{
		{
			name:       "one loading replica alongside active replica",
			wantModels: 1,
			state: &sim.RouterState{
				Snapshots: []sim.RoutingSnapshot{
					{ID: "active-1", Model: "modelA", GPUType: "A100", TPDegree: 1,
						TotalKvCapacityTokens: 10000, CostPerHour: 10.0},
				},
				LoadingSnapshots: []sim.RoutingSnapshot{
					{ID: "loading-1", Model: "modelA", GPUType: "A100", TPDegree: 1,
						TotalKvCapacityTokens: 10000, CostPerHour: 10.0},
				},
			},
			wantPendingCount:            map[string]int{"modelA": 1},
			wantPendingKvCapacityTokens: map[string]int64{"modelA": 10000},
		},
		{
			name:       "two loading replicas accumulate capacity",
			wantModels: 1,
			state: &sim.RouterState{
				Snapshots: []sim.RoutingSnapshot{
					{ID: "active-1", Model: "modelA", GPUType: "A100", TPDegree: 1,
						TotalKvCapacityTokens: 10000, CostPerHour: 10.0},
				},
				LoadingSnapshots: []sim.RoutingSnapshot{
					{ID: "loading-1", Model: "modelA", GPUType: "A100", TPDegree: 1,
						TotalKvCapacityTokens: 10000, CostPerHour: 10.0},
					{ID: "loading-2", Model: "modelA", GPUType: "A100", TPDegree: 1,
						TotalKvCapacityTokens: 10000, CostPerHour: 10.0},
				},
			},
			wantPendingCount:            map[string]int{"modelA": 2},
			wantPendingKvCapacityTokens: map[string]int64{"modelA": 20000},
		},
		{
			name:       "loading replicas for different models bucketed separately",
			wantModels: 2,
			state: &sim.RouterState{
				Snapshots: []sim.RoutingSnapshot{
					{ID: "a1", Model: "modelA", GPUType: "A100", TPDegree: 1,
						TotalKvCapacityTokens: 10000, CostPerHour: 10.0},
					{ID: "b1", Model: "modelB", GPUType: "H100", TPDegree: 2,
						TotalKvCapacityTokens: 20000, CostPerHour: 20.0},
				},
				LoadingSnapshots: []sim.RoutingSnapshot{
					{ID: "loading-a", Model: "modelA", GPUType: "A100", TPDegree: 1,
						TotalKvCapacityTokens: 10000, CostPerHour: 10.0},
					{ID: "loading-b", Model: "modelB", GPUType: "H100", TPDegree: 2,
						TotalKvCapacityTokens: 20000, CostPerHour: 20.0},
				},
			},
			wantPendingCount:            map[string]int{"modelA": 1, "modelB": 1},
			wantPendingKvCapacityTokens: map[string]int64{"modelA": 10000, "modelB": 20000},
		},
		{
			name:       "nil LoadingSnapshots — pending fields are zero",
			wantModels: 1,
			state: &sim.RouterState{
				Snapshots: []sim.RoutingSnapshot{
					{ID: "a1", Model: "modelA", GPUType: "A100", TPDegree: 1,
						TotalKvCapacityTokens: 10000, CostPerHour: 10.0},
				},
				LoadingSnapshots: nil,
			},
			wantPendingCount:            map[string]int{"modelA": 0},
			wantPendingKvCapacityTokens: map[string]int64{"modelA": 0},
		},
		{
			// Loading-only model: no active snapshots, only loading snapshots.
			// Verifies the pendingByModel → allModels union produces a ModelSignals entry
			// for a model that has never had a routable replica.
			name:       "loading-only model — no active snapshots",
			wantModels: 1,
			state: &sim.RouterState{
				Snapshots: nil,
				LoadingSnapshots: []sim.RoutingSnapshot{
					{ID: "loading-1", Model: "modelA", GPUType: "A100", TPDegree: 1,
						TotalKvCapacityTokens: 10000, CostPerHour: 10.0},
				},
			},
			wantPendingCount:            map[string]int{"modelA": 1},
			wantPendingKvCapacityTokens: map[string]int64{"modelA": 10000},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := collector.Collect(tc.state)

			if len(result) != tc.wantModels {
				t.Fatalf("got %d ModelSignals, want %d", len(result), tc.wantModels)
			}

			for _, ms := range result {
				wantCount := tc.wantPendingCount[ms.ModelID]
				if ms.PendingReplicaCount != wantCount {
					t.Errorf("model %q: PendingReplicaCount = %d, want %d",
						ms.ModelID, ms.PendingReplicaCount, wantCount)
				}
				wantCap := tc.wantPendingKvCapacityTokens[ms.ModelID]
				if ms.PendingTotalKvCapacityTokens != wantCap {
					t.Errorf("model %q: PendingTotalKvCapacityTokens = %d, want %d",
						ms.ModelID, ms.PendingTotalKvCapacityTokens, wantCap)
				}
			}
		})
	}
}
