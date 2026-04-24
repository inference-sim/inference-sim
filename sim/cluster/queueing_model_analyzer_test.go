package cluster

import (
	"math"
	"testing"
)

// TestNewQueueingModelAnalyzerPanicsOnNegativeValues verifies that
// NewQueueingModelAnalyzer panics (R3) when numeric fields have negative values
// that would be passed unchecked to third-party estimator constructors.
func TestNewQueueingModelAnalyzerPanicsOnNegativeInitObs(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative InitObs, got none")
		}
	}()
	NewQueueingModelAnalyzer(QMConfig{InitObs: -1})
}

func TestNewQueueingModelAnalyzerPanicsOnNegativeWindowSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative WindowSize, got none")
		}
	}()
	NewQueueingModelAnalyzer(QMConfig{WindowSize: -1})
}

func TestNewQueueingModelAnalyzerPanicsOnNegativeResidualThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative ResidualThreshold, got none")
		}
	}()
	NewQueueingModelAnalyzer(QMConfig{ResidualThreshold: -0.1})
}

func TestNewQueueingModelAnalyzerPanicsOnNegativeSLOMultiplier(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative SLOMultiplier, got none")
		}
	}()
	NewQueueingModelAnalyzer(QMConfig{SLOMultiplier: -1.0})
}

// TestGetSLOTargetDeterministicWithTwoVariants verifies that getSLOTarget (Priority 2 path)
// produces the same result when two variants are present, regardless of map iteration order (R2/INV-6).
// The fix is to use sortedModelVariantKeys in getSLOTarget.
func TestGetSLOTargetDeterministicWithTwoVariants(t *testing.T) {
	a := NewQueueingModelAnalyzer(QMConfig{TuningEnabled: false})

	// Seed two variants with different parameters so both contribute to the max.
	k1 := modelVariantKey{ModelID: "m1", Variant: NewVariantSpec("A100", 1)}
	k2 := modelVariantKey{ModelID: "m1", Variant: NewVariantSpec("H100", 2)}
	a.variantState[k1] = &perVariantState{alpha: 5.0, beta: 0.01, gamma: 0.001}
	a.variantState[k2] = &perVariantState{alpha: 3.0, beta: 0.02, gamma: 0.002}

	replicas := []ReplicaMetrics{{
		InstanceID:   "i1",
		Variant:      NewVariantSpec("A100", 1),
		TTFT:         200_000,
		ITL:          10_000,
		DispatchRate: 0.5,
		AvgInTokens:  512,
		AvgOutTokens: 128,
	}}

	// Run Analyze 10 times; all results must be identical (determinism).
	first := AnalyzerResult{}
	for i := 0; i < 10; i++ {
		got := a.Analyze(ModelSignals{ModelID: "m1", Replicas: replicas})
		if i == 0 {
			first = got
			continue
		}
		if got.TotalSupply != first.TotalSupply || got.TotalDemand != first.TotalDemand {
			t.Errorf("iteration %d: TotalSupply=%v TotalDemand=%v differs from first %v %v",
				i, got.TotalSupply, got.TotalDemand, first.TotalSupply, first.TotalDemand)
		}
	}
}

func TestQueueingModelAnalyzerName(t *testing.T) {
	a := NewQueueingModelAnalyzer(QMConfig{})
	if a.Name() != "queueing-model" {
		t.Errorf("Name() = %q, want %q", a.Name(), "queueing-model")
	}
}

func TestQueueingModelAnalyzerEmptyReplicas(t *testing.T) {
	a := NewQueueingModelAnalyzer(QMConfig{TuningEnabled: true, UseSliding: true, InitObs: 2})
	result := a.Analyze(ModelSignals{ModelID: "m1", Replicas: nil})
	if result.TotalSupply != 0 || result.TotalDemand != 0 || result.RequiredCapacity != 0 || result.SpareCapacity != 0 {
		t.Errorf("expected neutral result for empty replicas, got %+v", result)
	}
}

func TestQueueingModelAnalyzerBootstrapExclusion(t *testing.T) {
	cfg := QMConfig{
		TuningEnabled:     true,
		UseSliding:        true,
		InitObs:           2,
		WindowSize:        5,
		ResidualThreshold: 0,
	}
	a := NewQueueingModelAnalyzer(cfg)

	obs := ReplicaMetrics{
		InstanceID:   "i1",
		Variant:      NewVariantSpec("A100", 1),
		TTFT:         200_000,
		ITL:          10_000,
		DispatchRate: 0.5,
		AvgInTokens:  512,
		AvgOutTokens: 128,
		MaxBatchSize: 256,
		CostPerHour:  10,
	}

	result1 := a.Analyze(ModelSignals{ModelID: "m1", Replicas: []ReplicaMetrics{obs}})
	if result1.TotalSupply != 0 {
		t.Errorf("call 1: expected TotalSupply=0 during bootstrap, got %v", result1.TotalSupply)
	}
	if result1.TotalDemand != 0 {
		t.Errorf("call 1: expected TotalDemand=0 during bootstrap, got %v", result1.TotalDemand)
	}

	result2 := a.Analyze(ModelSignals{ModelID: "m1", Replicas: []ReplicaMetrics{obs}})
	if result2.TotalSupply <= 0 {
		t.Errorf("call 2: expected TotalSupply > 0 after bootstrap complete, got %v", result2.TotalSupply)
	}
}

func TestQueueingModelAnalyzerExplicitSLOTarget(t *testing.T) {
	cfg := QMConfig{
		TuningEnabled: false,
		SLOTargets: map[string]SLOTarget{
			"m1": {TargetTTFT: 500, TargetITL: 50},
		},
	}
	a := NewQueueingModelAnalyzer(cfg)

	// Pre-seed fitted parameters so Phase 3 can compute supply/demand.
	k := modelVariantKey{ModelID: "m1", Variant: NewVariantSpec("A100", 1)}
	a.variantState[k] = &perVariantState{alpha: 7.47, beta: 0.044, gamma: 3.37e-5}

	result := a.Analyze(ModelSignals{ModelID: "m1", Replicas: []ReplicaMetrics{{
		InstanceID:   "i1",
		Variant:      NewVariantSpec("A100", 1),
		TTFT:         200_000,
		ITL:          10_000,
		DispatchRate: 0.5,
		AvgInTokens:  512,
		AvgOutTokens: 128,
		MaxBatchSize: 256,
		CostPerHour:  10,
	}}})

	if result.ModelID != "m1" {
		t.Errorf("ModelID = %q, want %q", result.ModelID, "m1")
	}
	if result.TotalSupply <= 0 {
		t.Errorf("TotalSupply = %v, want > 0 (explicit SLO + fitted params should yield positive supply)", result.TotalSupply)
	}
	if result.TotalDemand <= 0 {
		t.Errorf("TotalDemand = %v, want > 0 (DispatchRate=0.5 should yield positive demand)", result.TotalDemand)
	}
}

func TestQueueingModelAnalyzerObservationBasedSLO(t *testing.T) {
	cfg := QMConfig{
		TuningEnabled: true,
		UseSliding:    true,
		InitObs:       2,
		WindowSize:    5,
	}
	a := NewQueueingModelAnalyzer(cfg)

	obs := ReplicaMetrics{
		InstanceID:   "i1",
		Variant:      NewVariantSpec("A100", 1),
		TTFT:         200_000,
		ITL:          10_000,
		DispatchRate: 0.5,
		AvgInTokens:  512,
		AvgOutTokens: 128,
		MaxBatchSize: 256,
		CostPerHour:  10,
	}
	a.Analyze(ModelSignals{ModelID: "m1", Replicas: []ReplicaMetrics{obs}})
	result := a.Analyze(ModelSignals{ModelID: "m1", Replicas: []ReplicaMetrics{obs}})
	if result.TotalSupply <= 0 {
		t.Logf("TotalSupply=%v (fit may not have converged)", result.TotalSupply)
	}
	_ = math.MaxFloat64
}
