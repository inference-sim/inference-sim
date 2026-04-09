package cluster

import "testing"

func TestUnlimitedEngineOptimize(t *testing.T) {
	engine := &UnlimitedEngine{}

	tests := []struct {
		name       string
		results    []AnalyzerResult
		wantLen    int
		wantDelta  int    // expected delta for first decision (-1 or +1)
		wantModel  string // expected ModelID for first decision
		wantGPU    string // expected GPU type for first decision
	}{
		{
			name:    "no signal — no decisions",
			results: []AnalyzerResult{{ModelID: "m1"}},
			wantLen: 0,
		},
		{
			name: "scale up — cheapest variant selected ignoring inventory",
			results: []AnalyzerResult{
				{
					ModelID:          "m1",
					RequiredCapacity: 100,
					VariantCapacities: []VariantCapacity{
						{Variant: NewVariantSpec("H100", 2), CostPerReplica: 20.0, ReplicaCount: 1},
						{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10.0, ReplicaCount: 1},
					},
				},
			},
			wantLen:   1,
			wantDelta: 1,
			wantModel: "m1",
			wantGPU:   "A100", // cheapest
		},
		{
			name: "scale down — most expensive variant with replicas",
			results: []AnalyzerResult{
				{
					ModelID:       "m1",
					SpareCapacity: 50,
					VariantCapacities: []VariantCapacity{
						{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10.0, ReplicaCount: 2},
						{Variant: NewVariantSpec("H100", 2), CostPerReplica: 20.0, ReplicaCount: 1},
					},
				},
			},
			wantLen:   1,
			wantDelta: -1,
			wantModel: "m1",
			wantGPU:   "H100", // most expensive
		},
		{
			name: "scale down skips variant with zero replicas",
			results: []AnalyzerResult{
				{
					ModelID:       "m1",
					SpareCapacity: 50,
					VariantCapacities: []VariantCapacity{
						{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10.0, ReplicaCount: 2},
						{Variant: NewVariantSpec("H100", 2), CostPerReplica: 20.0, ReplicaCount: 0},
					},
				},
			},
			wantLen:   1,
			wantDelta: -1,
			wantModel: "m1",
			wantGPU:   "A100", // H100 has 0 replicas, falls back to A100
		},
		{
			name: "multiple models — one decision each",
			results: []AnalyzerResult{
				{
					ModelID:          "m1",
					RequiredCapacity: 100,
					VariantCapacities: []VariantCapacity{
						{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10.0, ReplicaCount: 1},
					},
				},
				{
					ModelID:       "m2",
					SpareCapacity: 50,
					VariantCapacities: []VariantCapacity{
						{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10.0, ReplicaCount: 2},
					},
				},
			},
			wantLen: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			decisions := engine.Optimize(tc.results, GPUInventory{})

			if len(decisions) != tc.wantLen {
				t.Fatalf("got %d decisions, want %d", len(decisions), tc.wantLen)
			}
			if tc.wantLen == 0 {
				return
			}

			d := decisions[0]
			if tc.wantModel != "" && d.ModelID != tc.wantModel {
				t.Errorf("ModelID = %q, want %q", d.ModelID, tc.wantModel)
			}
			if tc.wantDelta != 0 && d.Delta != tc.wantDelta {
				t.Errorf("Delta = %d, want %d", d.Delta, tc.wantDelta)
			}
			if tc.wantGPU != "" && d.Variant.GPUType != tc.wantGPU {
				t.Errorf("Variant.GPUType = %q, want %q", d.Variant.GPUType, tc.wantGPU)
			}
		})
	}
}
