package cluster

import (
	"math"
	"testing"
)

// TestV2SaturationAnalyzerAnalyze verifies the V2 token-based saturation analyzer
// contract: capacity in tokens via min(k1_memory, k2_compute), demand as
// tokensInUse + queueLength*avgInputTokens, and model-level RequiredCapacity/SpareCapacity signals.
func TestV2SaturationAnalyzerAnalyze(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  0.8,  // k1 = totalKvCapTokens * 0.8
		ScaleUpThreshold:  0.8,  // RequiredCapacity when demand/supply > 1/0.8
		ScaleDownBoundary: 0.4,  // SpareCapacity when demand/supply < 0.4
		AvgInputTokens:    512,  // for queue-to-demand conversion
	}
	analyzer := NewV2SaturationAnalyzer(cfg)

	if analyzer.Name() != "v2-saturation" {
		t.Fatalf("Name() = %q, want %q", analyzer.Name(), "v2-saturation")
	}

	tests := []struct {
		name string
		input ModelSignals
		// Expected outputs — use -1 to skip check
		wantRequiredPositive bool
		wantSparePositive    bool
		wantTotalSupplyZero  bool
		wantTotalDemandZero  bool
		// Invariant checks
		checkAggregation bool // sum(vc.Supply)==TotalSupply, sum(vc.Demand)==TotalDemand
	}{
		{
			name:                "zero replicas — all-zero output",
			input:               ModelSignals{ModelID: "m1", Replicas: nil},
			wantRequiredPositive: false,
			wantSparePositive:    false,
			wantTotalSupplyZero:  true,
			wantTotalDemandZero:  true,
		},
		{
			name: "all replicas saturated — RequiredCapacity > 0",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         9000,  // 90% used
						QueueDepth:            10,     // additional demand: 10 * 512 = 5120 tokens
						CostPerHour:           10.0,
					},
					{
						InstanceID:            "i2",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         8500,
						QueueDepth:            8,
						CostPerHour:           10.0,
					},
				},
			},
			wantRequiredPositive: true,
			wantSparePositive:    false,
			checkAggregation:     true,
		},
		{
			name: "all replicas idle with headroom — SpareCapacity > 0",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         500,  // 5% used — very idle
						QueueDepth:            0,
						CostPerHour:           10.0,
					},
					{
						InstanceID:            "i2",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         600,
						QueueDepth:            0,
						CostPerHour:           10.0,
					},
					{
						InstanceID:            "i3",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         400,
						QueueDepth:            0,
						CostPerHour:           10.0,
					},
				},
			},
			wantRequiredPositive: false,
			wantSparePositive:    true,
			checkAggregation:     true,
		},
		{
			name: "single replica near saturation — SpareCapacity must be zero",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         7500,
						QueueDepth:            5,
						CostPerHour:           10.0,
					},
				},
			},
			wantRequiredPositive: false, // not necessarily saturated
			wantSparePositive:    false, // cannot scale below 1 replica
		},
		{
			name: "mixed variants — aggregation invariant holds",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         5000,
						QueueDepth:            2,
						CostPerHour:           10.0,
					},
					{
						InstanceID:            "i2",
						Variant:               NewVariantSpec("H100", 2),
						TotalKvCapacityTokens: 20000,
						KvTokensInUse:         8000,
						QueueDepth:            3,
						CostPerHour:           20.0,
					},
				},
			},
			checkAggregation: true,
		},
		{
			name: "mutual exclusivity — RequiredCapacity > 0 implies SpareCapacity == 0",
			input: ModelSignals{
				ModelID: "m1",
				Replicas: []ReplicaMetrics{
					{
						InstanceID:            "i1",
						Variant:               NewVariantSpec("A100", 1),
						TotalKvCapacityTokens: 10000,
						KvTokensInUse:         9500,
						QueueDepth:            20,
						CostPerHour:           10.0,
					},
				},
			},
			wantRequiredPositive: true,
			wantSparePositive:    false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := analyzer.Analyze(tc.input)

			// ModelID propagation
			if result.ModelID != tc.input.ModelID {
				t.Errorf("ModelID = %q, want %q", result.ModelID, tc.input.ModelID)
			}

			// RequiredCapacity / SpareCapacity checks
			if tc.wantRequiredPositive && result.RequiredCapacity <= 0 {
				t.Errorf("RequiredCapacity = %f, want > 0", result.RequiredCapacity)
			}
			if !tc.wantRequiredPositive && result.RequiredCapacity > 0 {
				// Only check if we explicitly expect it to be zero
				if tc.wantTotalSupplyZero || tc.wantSparePositive {
					t.Errorf("RequiredCapacity = %f, want 0", result.RequiredCapacity)
				}
			}
			if tc.wantSparePositive && result.SpareCapacity <= 0 {
				t.Errorf("SpareCapacity = %f, want > 0", result.SpareCapacity)
			}
			if !tc.wantSparePositive && result.SpareCapacity > 0 {
				t.Errorf("SpareCapacity = %f, want 0", result.SpareCapacity)
			}

			// Mutual exclusivity: never both positive
			if result.RequiredCapacity > 0 && result.SpareCapacity > 0 {
				t.Errorf("mutual exclusivity violated: RequiredCapacity=%f, SpareCapacity=%f",
					result.RequiredCapacity, result.SpareCapacity)
			}

			// Zero-supply guard
			if tc.wantTotalSupplyZero && result.TotalSupply != 0 {
				t.Errorf("TotalSupply = %f, want 0", result.TotalSupply)
			}
			if tc.wantTotalDemandZero && result.TotalDemand != 0 {
				t.Errorf("TotalDemand = %f, want 0", result.TotalDemand)
			}

			// Utilization guard: no NaN or Inf
			if math.IsNaN(result.Utilization) || math.IsInf(result.Utilization, 0) {
				t.Errorf("Utilization = %f, must not be NaN or Inf", result.Utilization)
			}

			// Aggregation invariant: sum(vc.Supply)==TotalSupply, sum(vc.Demand)==TotalDemand
			if tc.checkAggregation {
				var sumSupply, sumDemand float64
				for _, vc := range result.VariantCapacities {
					sumSupply += vc.Supply
					sumDemand += vc.Demand
				}
				if math.Abs(sumSupply-result.TotalSupply) > 1e-6 {
					t.Errorf("sum(vc.Supply)=%f != TotalSupply=%f", sumSupply, result.TotalSupply)
				}
				if math.Abs(sumDemand-result.TotalDemand) > 1e-6 {
					t.Errorf("sum(vc.Demand)=%f != TotalDemand=%f", sumDemand, result.TotalDemand)
				}
			}
		})
	}
}

// TestV2SaturationAnalyzerK1K2 verifies that effective capacity uses min(k1, k2).
func TestV2SaturationAnalyzerK1K2(t *testing.T) {
	// When k1 < k2 (memory is bottleneck), effective capacity should be k1.
	// We verify this indirectly: with low demand relative to k1, SpareCapacity should be
	// positive even if k2 would be higher.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  0.5,  // k1 = 10000 * 0.5 = 5000 tokens
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    512,
	}
	analyzer := NewV2SaturationAnalyzer(cfg)

	result := analyzer.Analyze(ModelSignals{
		ModelID: "m1",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "i1",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000, // k1 = 10000 * 0.5 = 5000
				KvTokensInUse:         100,   // very low demand
				QueueDepth:            0,
				CostPerHour:           10.0,
			},
			{
				InstanceID:            "i2",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000,
				KvTokensInUse:         100,
				QueueDepth:            0,
				CostPerHour:           10.0,
			},
			{
				InstanceID:            "i3",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000,
				KvTokensInUse:         100,
				QueueDepth:            0,
				CostPerHour:           10.0,
			},
		},
	})

	// Supply should be based on k1 (5000 tokens per replica × 3 replicas = 15000)
	// Demand = 300 tokens total. SpareCapacity should be positive.
	if result.TotalSupply <= 0 {
		t.Fatalf("TotalSupply = %f, want > 0", result.TotalSupply)
	}
	if result.SpareCapacity <= 0 {
		t.Errorf("SpareCapacity = %f, want > 0 (supply >> demand with 3 replicas)", result.SpareCapacity)
	}
	if result.RequiredCapacity > 0 {
		t.Errorf("RequiredCapacity = %f, want 0", result.RequiredCapacity)
	}
}

// TestV2SaturationAnalyzerN1MixedVariants verifies that the N-1 redistribution check
// uses the highest-capacity variant (conservative) to match Engine's most-expensive-first
// scale-down selection. Regression test for the min→max fix.
func TestV2SaturationAnalyzerN1MixedVariants(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  0.8,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.4,
		AvgInputTokens:    512,
	}
	analyzer := NewV2SaturationAnalyzer(cfg)

	// Scenario from review: A100 (cheap, low capacity) + H100 (expensive, high capacity).
	// Engine will remove H100 (most expensive). N-1 check must simulate removing the
	// highest-capacity replica (H100), not the lowest (A100).
	//
	// A100: 1 replica, totalKvCap=10000, k1=8000, demand=2000, supply=8000
	// H100: 1 replica, totalKvCap=25000, k1=20000, demand=2000, supply=20000
	// Total supply=28000, demand=4000
	// SpareCapacity = 28000 - 4000/0.4 = 28000 - 10000 = 18000 > 0
	//
	// N-1 with max (correct): remove H100's 20000 → supply=8000, 8000 > 10000? NO → no spare
	// N-1 with min (bug):     remove A100's 8000  → supply=20000, 20000 > 10000? YES → spare approved
	//
	// If the bug existed, Engine would remove H100, leaving only A100 (supply=8000)
	// which violates ScaleDownBoundary (need 10000).
	result := analyzer.Analyze(ModelSignals{
		ModelID: "m1",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "i1",
				Variant:               NewVariantSpec("A100", 1),
				TotalKvCapacityTokens: 10000,
				KvTokensInUse:         2000,
				QueueDepth:            0,
				CostPerHour:           10.0,
			},
			{
				InstanceID:            "i2",
				Variant:               NewVariantSpec("H100", 2),
				TotalKvCapacityTokens: 25000,
				KvTokensInUse:         2000,
				QueueDepth:            0,
				CostPerHour:           20.0,
			},
		},
	})

	// With conservative N-1 check (removing H100's capacity), scale-down should be blocked
	if result.SpareCapacity > 0 {
		t.Errorf("SpareCapacity = %f, want 0 (N-1 check should block: removing H100 leaves insufficient supply)", result.SpareCapacity)
	}
}

// TestV2SaturationAnalyzerConfigValidation verifies constructor rejects invalid configs.
func TestV2SaturationAnalyzerConfigValidation(t *testing.T) {
	tests := []struct {
		name string
		cfg  V2SaturationAnalyzerConfig
	}{
		{"zero KvCacheThreshold", V2SaturationAnalyzerConfig{KvCacheThreshold: 0, ScaleUpThreshold: 0.8, ScaleDownBoundary: 0.4, AvgInputTokens: 512}},
		{"KvCacheThreshold > 1.0", V2SaturationAnalyzerConfig{KvCacheThreshold: 1.5, ScaleUpThreshold: 0.8, ScaleDownBoundary: 0.4, AvgInputTokens: 512}},
		{"negative ScaleUpThreshold", V2SaturationAnalyzerConfig{KvCacheThreshold: 0.8, ScaleUpThreshold: -1, ScaleDownBoundary: 0.4, AvgInputTokens: 512}},
		{"NaN ScaleDownBoundary", V2SaturationAnalyzerConfig{KvCacheThreshold: 0.8, ScaleUpThreshold: 0.8, ScaleDownBoundary: math.NaN(), AvgInputTokens: 512}},
		{"zero AvgInputTokens", V2SaturationAnalyzerConfig{KvCacheThreshold: 0.8, ScaleUpThreshold: 0.8, ScaleDownBoundary: 0.4, AvgInputTokens: 0}},
		{"ScaleDownBoundary >= ScaleUpThreshold", V2SaturationAnalyzerConfig{KvCacheThreshold: 0.8, ScaleUpThreshold: 0.4, ScaleDownBoundary: 0.8, AvgInputTokens: 512}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic for invalid config, got none")
				}
			}()
			NewV2SaturationAnalyzer(tc.cfg)
		})
	}
}
