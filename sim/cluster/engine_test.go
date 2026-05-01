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
		wantLen   int
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
			wantLen:   1,
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
			wantLen:   1,
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
			wantLen:   1,
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
			wantLen:   1,
			wantDelta: 1,
		},
		{
			name: "scale-down with no active replicas → no decision",
			results: []AnalyzerResult{{
				ModelID:       "m1",
				SpareCapacity: 50,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 0, ReplicaCount: 0},
				},
			}},
			wantLen: 0, // no active replicas; emitting Delta=-1 would trigger a no-op actuator warn
		},
		{
			name: "scale-up falls back to Delta=1 when cheapest variant is inactive but a more-expensive one is active",
			results: []AnalyzerResult{{
				ModelID:          "m1",
				RequiredCapacity: 50,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 5, Supply: 0, ReplicaCount: 0},    // cheapest, inactive
					{Variant: NewVariantSpec("H100", 1), CostPerReplica: 20, Supply: 30, ReplicaCount: 3}, // active
				},
			}},
			wantLen:   1,
			wantDelta: 1, // A100 selected; its prc=0 → fallback to 1; does not borrow H100's prc
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decisions := engine.Optimize(tt.results, GPUInventory{})
			if len(decisions) != tt.wantLen {
				t.Fatalf("want %d decisions, got %d", tt.wantLen, len(decisions))
			}
			if tt.wantLen == 0 {
				return
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
		{
			name: "scale down: no active replicas anywhere → no decision",
			results: []AnalyzerResult{{
				ModelID:       "m1",
				SpareCapacity: 20,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 0, ReplicaCount: 0},
					{Variant: NewVariantSpec("H100", 1), CostPerReplica: 20, Supply: 0, ReplicaCount: 0},
				},
			}},
			inventory: newTestGPUInventory(map[VariantSpec]int{}),
			wantLen:   0,
		},
		{
			name: "scale up: TPDegree=2 requires n*TPDegree free GPU slots",
			results: []AnalyzerResult{{
				ModelID:          "m1",
				RequiredCapacity: 20, // ceil(20/10)=2 replicas; needed = 2*2 = 4 slots
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 2), CostPerReplica: 10, Supply: 10, ReplicaCount: 1},
				},
			}},
			inventory: newTestGPUInventory(map[VariantSpec]int{
				NewVariantSpec("A100", 2): 4, // exactly 2 replicas * 2 GPUs each
			}),
			wantLen:   1,
			wantDelta: 2,
			wantGPU:   "A100",
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

// TestGreedyEngineScaleUpUsesChosenVariantCapacity verifies that when the cheapest
// variant has no free GPU slots and the engine falls back to a more expensive variant,
// it computes n using the chosen variant's own per-replica capacity — not the cheapest
// variant's capacity. Failing to do so overestimates n (scales too many replicas).
func TestGreedyEngineScaleUpUsesChosenVariantCapacity(t *testing.T) {
	engine := &GreedyEngine{}

	// Cheap (A100): 2 active replicas, 10 total supply → 5 RPS/replica.
	// Expensive (H100): 2 active replicas, 20 total supply → 10 RPS/replica.
	// Required = 20 RPS.
	// A100 has 0 free GPU slots → engine falls back to H100.
	// Correct n for H100 = ceil(20/10) = 2 (not ceil(20/5) = 4).
	results := []AnalyzerResult{{
		ModelID:          "m1",
		RequiredCapacity: 20,
		VariantCapacities: []VariantCapacity{
			{Variant: NewVariantSpec("A100", 1), CostPerReplica: 5, Supply: 10, ReplicaCount: 2},
			{Variant: NewVariantSpec("H100", 1), CostPerReplica: 20, Supply: 20, ReplicaCount: 2},
		},
	}}
	inventory := newTestGPUInventory(map[VariantSpec]int{
		NewVariantSpec("A100", 1): 0,  // full
		NewVariantSpec("H100", 1): 10, // plenty of room
	})

	decisions := engine.Optimize(results, inventory)
	if len(decisions) != 1 {
		t.Fatalf("want 1 decision, got %d: %+v", len(decisions), decisions)
	}
	d := decisions[0]
	if d.Variant.GPUType != "H100" {
		t.Errorf("GPUType: want H100, got %q", d.Variant.GPUType)
	}
	if d.Delta != 2 {
		t.Errorf("Delta: want 2 (ceil(20/10)), got %d — engine used cheapest variant's prc instead of chosen variant's", d.Delta)
	}
}

// TestScaleDownNFloorToZero verifies that when floor(spareCapacity/prc) == 0, scaleDownN
// clamps to 1 rather than emitting a Delta=0 no-op. Without the clamp a future refactor
// could silently produce a zero-replica removal decision.
func TestScaleDownNFloorToZero(t *testing.T) {
	// spareCapacity=1, Supply=100, ReplicaCount=10 → prc=10 → floor(1/10)=0 → clamped to 1
	vc := VariantCapacity{
		Variant:      NewVariantSpec("A100", 1),
		Supply:       100,
		ReplicaCount: 10,
	}
	n := scaleDownN(1, vc)
	if n != 1 {
		t.Errorf("scaleDownN: want 1 (clamped from floor=0), got %d", n)
	}
}
