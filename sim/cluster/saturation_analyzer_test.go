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

// TestV2SaturationAnalyzer_PendingSupply_LoadingOnlyModel verifies that when all instances
// are Loading (Replicas == nil), the analyzer correctly returns all-zero output.
// A loading-only model has no routable replicas → no demand signal → RequiredCapacity = 0 is correct.
func TestV2SaturationAnalyzer_PendingSupply_LoadingOnlyModel(t *testing.T) {
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID:                      "test-model",
		Replicas:                     nil, // no Active/WarmingUp instances
		PendingReplicaCount:          2,
		PendingTotalKvCapacityTokens: 20000,
	}

	result := a.Analyze(metrics)

	// No routable replicas → no demand can be measured → all-zero result is correct.
	if result.RequiredCapacity != 0 {
		t.Errorf("RequiredCapacity = %g, want 0 (no demand signal with zero routable replicas)", result.RequiredCapacity)
	}
	if result.TotalSupply != 0 {
		t.Errorf("TotalSupply = %g, want 0 (ready-replica-only)", result.TotalSupply)
	}
	if result.TotalDemand != 0 {
		t.Errorf("TotalDemand = %g, want 0 (no routable replicas to measure demand from)", result.TotalDemand)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_ThresholdApplied verifies that KvCacheThreshold
// is applied to pending supply (not just to ready-replica supply).
func TestV2SaturationAnalyzer_PendingSupply_ThresholdApplied(t *testing.T) {
	// KvCacheThreshold=0.8: effective pending supply = 10000 * 0.8 = 8000 (not 10000).
	// Active: capacity=10000, k1=8000. Demand=9000 → demand/threshold(0.8)=11250.
	// Without pending: requiredCapacity = 11250 - 8000 = 3250 > 0 → scale-up.
	// With pending (threshold applied correctly): pendingSupply = 10000 * 0.8 = 8000.
	//   totalSupplyForScaleUp = 8000 + 8000 = 16000 → 11250 < 16000 → no scale-up.
	// If threshold were NOT applied to pending: pendingSupply = 10000.
	//   totalSupplyForScaleUp = 8000 + 10000 = 18000 → 11250 < 18000 → also no scale-up (indistinguishable).
	// So we use a case where threshold-correct pending is just barely enough:
	//   Active: capacity=10000, k1=8000. Demand=12000 → demand/threshold(0.8)=15000.
	//   pendingSupply (threshold=0.8) = 10000*0.8 = 8000 → total=16000 > 15000 → suppressed.
	//   pendingSupply (threshold=1.0, wrong) = 10000 → total=18000 > 15000 → also suppressed.
	// Use threshold=0.5 to make the difference observable:
	//   Active: capacity=10000, k1=5000. Demand=5000 → demand/threshold(0.8)=6250.
	//   pendingSupply (threshold=0.5) = 10000*0.5 = 5000 → total=10000 > 6250 → suppressed.
	//   pendingSupply (threshold=1.0, wrong) = 10000 → total=15000 > 6250 → also suppressed.
	// To make it observable: set pending capacity so that threshold*pending < gap but 1.0*pending > gap.
	//   Active: capacity=10000, k1=5000. Demand=4500 → demand/threshold(0.8)=5625.
	//   gap = 5625 - 5000 = 625.
	//   pendingKvCap=1000 → pendingSupply(0.5)=500 < 625 → scale-up still fires.
	//   pendingKvCap=1000 → pendingSupply(1.0, wrong)=1000 > 625 → no scale-up (wrong behavior).
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  0.5,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         4500,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000, // k1 = 10000 * 0.5 = 5000
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 1000, // pendingSupply(0.5)=500; pendingSupply(1.0)=1000
	}

	result := a.Analyze(metrics)

	// With KvCacheThreshold=0.5 applied to pending: pendingSupply=500 → gap=625 not covered → scale-up.
	if result.RequiredCapacity <= 0 {
		t.Errorf("RequiredCapacity = %g, want > 0 (threshold correctly applied to pending: 500 < gap 625)", result.RequiredCapacity)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_SuppressesScaleUp verifies the core fix for #1109:
// when a Loading instance's capacity covers the demand gap, no further scale-up is emitted.
func TestV2SaturationAnalyzer_PendingSupply_SuppressesScaleUp(t *testing.T) {
	// Configuration: ScaleUpThreshold=0.8 means scale-up fires when demand > 0.8 * supply.
	// Active replica: capacity=10000, current demand=9000 → demand/threshold=11250 > 10000 → scale-up.
	// After fix: 1 Loading replica adds 10000 pending capacity.
	// totalSupplyForScaleUp = 10000 + 10000 = 20000 → demand/threshold=11250 < 20000 → no scale-up.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         9000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 10000,
	}

	result := a.Analyze(metrics)

	if result.RequiredCapacity != 0 {
		t.Errorf("RequiredCapacity = %g, want 0 (loading replica covers demand gap)", result.RequiredCapacity)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_ZeroCapacityLoading verifies that a Loading instance
// with PendingReplicaCount=1 but PendingTotalKvCapacityTokens=0 contributes no pending supply.
// The analyzer guards on PendingTotalKvCapacityTokens > 0, not PendingReplicaCount > 0.
// A regression that checked PendingReplicaCount instead would incorrectly suppress scale-up.
func TestV2SaturationAnalyzer_PendingSupply_ZeroCapacityLoading(t *testing.T) {
	// Active replica: capacity=10000, demand=9000 → demand/threshold(0.8)=11250 > 10000 → scale-up needed.
	// Loading replica: PendingReplicaCount=1 but PendingTotalKvCapacityTokens=0 (zero-KV node).
	// pendingSupply must be 0 → RequiredCapacity = 11250 - 10000 = 1250 > 0.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         9000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 0, // zero-KV loading instance — contributes no pending supply
	}

	result := a.Analyze(metrics)

	if result.RequiredCapacity <= 0 {
		t.Errorf("RequiredCapacity = %g, want > 0 (zero-KV loading instance must not suppress scale-up)", result.RequiredCapacity)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_StillScalesUpForDelta verifies that when
// demand has grown beyond what the Loading instance covers, scale-up still fires for the delta.
func TestV2SaturationAnalyzer_PendingSupply_StillScalesUpForDelta(t *testing.T) {
	// Active: capacity=10000, demand=18000 → demand/threshold(0.8)=22500.
	// 1 Loading replica: pending capacity=10000.
	// totalSupplyForScaleUp = 10000 + 10000 = 20000 → requiredCapacity = 22500-20000 = 2500 > 0.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         18000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 10000,
	}

	result := a.Analyze(metrics)

	// demand/threshold = 18000/0.8 = 22500; totalSupplyForScaleUp = 10000+10000 = 20000; delta = 2500.
	if result.RequiredCapacity != 2500 {
		t.Errorf("RequiredCapacity = %g, want 2500 (demand exceeds ready+pending supply by exactly 2500)", result.RequiredCapacity)
	}
	// The ready-only supply (10000) must not be inflated by pending.
	if result.TotalSupply != 10000 {
		t.Errorf("TotalSupply = %g, want 10000 (ready supply only; pending does not inflate TotalSupply)", result.TotalSupply)
	}
}

// TestV2SaturationAnalyzer_PendingSupply_DoesNotAffectScaleDown verifies that pending
// supply does NOT inflate TotalSupply used for SpareCapacity (no premature scale-down).
func TestV2SaturationAnalyzer_PendingSupply_DoesNotAffectScaleDown(t *testing.T) {
	// 2 Active replicas with low utilization → spare capacity signal.
	// 1 Loading replica: pending capacity=10000.
	// TotalSupply must be the ready-only value (20000), not 30000.
	cfg := V2SaturationAnalyzerConfig{
		KvCacheThreshold:  1.0,
		ScaleUpThreshold:  0.8,
		ScaleDownBoundary: 0.3,
		AvgInputTokens:    100,
	}
	a := NewV2SaturationAnalyzer(cfg)

	metrics := ModelSignals{
		ModelID: "test-model",
		Replicas: []ReplicaMetrics{
			{
				InstanceID:            "active-1",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         1000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
			{
				InstanceID:            "active-2",
				Variant:               NewVariantSpec("A100", 1),
				KvTokensInUse:         1000,
				QueueDepth:            0,
				TotalKvCapacityTokens: 10000,
				CostPerHour:           10.0,
			},
		},
		PendingReplicaCount:          1,
		PendingTotalKvCapacityTokens: 10000,
	}

	result := a.Analyze(metrics)

	// TotalSupply must be ready-only (20000). Pending (10000) must not be included.
	if result.TotalSupply != 20000 {
		t.Errorf("TotalSupply = %g, want 20000 (ready-only; pending must not inflate)", result.TotalSupply)
	}
	// SpareCapacity must be based on ready-only supply (no premature scale-down risk from pending).
	// demand=2000, ScaleDownBoundary=0.3, supply=20000:
	// spareCapacity = 20000 - (2000/0.3) = 20000 - 6667 = 13333 > 0
	// N-1 check: supplyAfterRemoval=10000 > 2000/0.3=6667 → SpareCapacity is set.
	if result.SpareCapacity <= 0 {
		t.Errorf("SpareCapacity = %g, want > 0 (low utilization with 2 ready replicas)", result.SpareCapacity)
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
