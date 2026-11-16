package cluster

import "testing"

func TestUnlimitedEngineOptimize(t *testing.T) {
	engine := &UnlimitedEngine{}

	tests := []struct {
		name      string
		results   []AnalyzerResult
		wantLen   int
		wantDelta int    // expected delta for first decision (-1 or +1)
		wantModel string // expected ModelID for first decision
		wantGPU   string // expected GPU type for first decision
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
					SpareCapacity: 50, SafeRemovals: 10,
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
					SpareCapacity: 50, SafeRemovals: 10,
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
					SpareCapacity: 50, SafeRemovals: 10,
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
				SpareCapacity: 25, SafeRemovals: 10, // floor(25/10) = 2
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
				SpareCapacity: 100, SafeRemovals: 10, // floor(100/10)=10, clamped to replicaCount=3
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 30, ReplicaCount: 3},
				},
			}},
			wantLen:   1,
			wantDelta: -3,
		},
		{
			// #2015: an idle start. Zero demand makes the spare capacity the whole supply, so
			// floor(spare/prc) is every replica; the analyzer's check proved two removals safe.
			name: "scale-down bounded by the removals the analyzer proved safe",
			results: []AnalyzerResult{{
				ModelID:       "m1",
				SpareCapacity: 30, SafeRemovals: 2,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 30, ReplicaCount: 3},
				},
			}},
			wantLen:   1,
			wantDelta: -2,
		},
		{
			name: "scale-down with no removal proved safe → no decision",
			results: []AnalyzerResult{{
				ModelID:       "m1",
				SpareCapacity: 30,
				VariantCapacities: []VariantCapacity{
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 10, Supply: 30, ReplicaCount: 3},
				},
			}},
			wantLen: 0, // Delta=0 would break INV-19
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
				SpareCapacity: 50, SafeRemovals: 10,
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
					{Variant: NewVariantSpec("A100", 1), CostPerReplica: 5, Supply: 0, ReplicaCount: 0},   // cheapest, inactive
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
	n := scaleDownN(1, vc, vc.ReplicaCount)
	if n != 1 {
		t.Errorf("scaleDownN: want 1 (clamped from floor=0), got %d", n)
	}
}
