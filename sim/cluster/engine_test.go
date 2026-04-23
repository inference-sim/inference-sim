package cluster

import "testing"

// newTestGPUInventory constructs a GPUInventory with the given free-slot counts.
// Used only in engine tests within the same package.
func newTestGPUInventory(slots map[VariantSpec]int) GPUInventory {
	return GPUInventory{byVariant: slots}
}

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

func TestUnlimitedEngineExactN(t *testing.T) {
	engine := &UnlimitedEngine{}
	tests := []struct {
		name      string
		results   []AnalyzerResult
		wantDelta int
	}{
		{
			name: "scale-up exact N=3 when perReplicaCapacity=10",
			results: []AnalyzerResult{{
				ModelID:          "m1",
				RequiredCapacity: 25, // ceil(25/10) = 3
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 10, ReplicaCount: 1},
				},
			}},
			wantDelta: 3,
		},
		{
			name: "scale-down exact N=2",
			results: []AnalyzerResult{{
				ModelID:       "m1",
				SpareCapacity: 25, // floor(25/10) = 2
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 30, ReplicaCount: 3},
				},
			}},
			wantDelta: -2,
		},
		{
			name: "scale-down clamped to replicaCount when floor would exceed it",
			results: []AnalyzerResult{{
				ModelID:       "m1",
				SpareCapacity: 100, // floor(100/10)=10, clamped to replicaCount=3
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 30, ReplicaCount: 3},
				},
			}},
			wantDelta: -3,
		},
		{
			name: "scale-up falls back to Delta=1 when no active replicas",
			results: []AnalyzerResult{{
				ModelID:          "m1",
				RequiredCapacity: 50,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 0, ReplicaCount: 0},
				},
			}},
			wantDelta: 1,
		},
		{
			name: "scale-down falls back to Delta=-1 when perReplicaCapacity=0",
			results: []AnalyzerResult{{
				ModelID:       "m1",
				SpareCapacity: 50,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 0, ReplicaCount: 0},
				},
			}},
			wantDelta: -1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decisions := engine.Optimize(tt.results, GPUInventory{})
			if len(decisions) != 1 {
				t.Fatalf("want 1 decision, got %d", len(decisions))
			}
			if decisions[0].Delta != tt.wantDelta {
				t.Errorf("Delta: want %d got %d", tt.wantDelta, decisions[0].Delta)
			}
		})
	}
}

func TestGreedyEngineOptimize(t *testing.T) {
	tests := []struct {
		name      string
		results   []AnalyzerResult
		inventory GPUInventory
		wantLen   int
		wantDelta int
		wantGPU   string
	}{
		{
			name: "scale up: cheapest variant with free slots",
			results: []AnalyzerResult{{
				ModelID:          "m1",
				RequiredCapacity: 10,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 10, ReplicaCount: 1},
					{Variant: NewVariantSpec("H100", 1), CostPerReplica: 20, Supply: 10, ReplicaCount: 1},
				},
			}},
			inventory: newTestGPUInventory(map[VariantSpec]int{
				NewVariantSpec("A100", 1): 4,
				NewVariantSpec("H100", 1): 4,
			}),
			wantLen:   1,
			wantDelta: 1,
			wantGPU:   "A100",
		},
		{
			name: "scale up: skip variant with insufficient free slots, use next cheapest",
			results: []AnalyzerResult{{
				ModelID:          "m1",
				RequiredCapacity: 40,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 10, ReplicaCount: 1},
					{Variant: NewVariantSpec("H100", 1), CostPerReplica: 20, Supply: 10, ReplicaCount: 1},
				},
			}},
			inventory: newTestGPUInventory(map[VariantSpec]int{
				NewVariantSpec("A100", 1): 2, // need 4 — skip
				NewVariantSpec("H100", 1): 5, // enough
			}),
			wantLen:   1,
			wantDelta: 4,
			wantGPU:   "H100",
		},
		{
			name: "scale up: no free slots anywhere → no decision",
			results: []AnalyzerResult{{
				ModelID:          "m1",
				RequiredCapacity: 10,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 10, ReplicaCount: 1},
				},
			}},
			inventory: newTestGPUInventory(map[VariantSpec]int{
				NewVariantSpec("A100", 1): 0,
			}),
			wantLen: 0,
		},
		{
			name: "scale down: exact N from most expensive active variant",
			results: []AnalyzerResult{{
				ModelID:       "m1",
				SpareCapacity: 20,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 30, ReplicaCount: 3},
					{Variant: NewVariantSpec("H100", 1), CostPerReplica: 20, Supply: 30, ReplicaCount: 3},
				},
			}},
			inventory: newTestGPUInventory(map[VariantSpec]int{
				NewVariantSpec("A100", 1): 0,
				NewVariantSpec("H100", 1): 0,
			}),
			wantLen:   1,
			wantDelta: -2, // floor(20/10)=2, using H100 perReplicaCapacity=30/3=10
			wantGPU:   "H100",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := &GreedyEngine{}
			decisions := engine.Optimize(tt.results, tt.inventory)
			if len(decisions) != tt.wantLen {
				t.Fatalf("want %d decisions, got %d: %+v", tt.wantLen, len(decisions), decisions)
			}
			if tt.wantLen == 0 {
				return
			}
			if decisions[0].Delta != tt.wantDelta {
				t.Errorf("Delta: want %d got %d", tt.wantDelta, decisions[0].Delta)
			}
			if decisions[0].Variant.GPUType != tt.wantGPU {
				t.Errorf("GPUType: want %q got %q", tt.wantGPU, decisions[0].Variant.GPUType)
			}
		})
	}
}
