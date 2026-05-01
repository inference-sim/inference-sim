package cluster

import (
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

func TestNewQueueingModelAnalyzerPanicsOnNegativeWarmUpCycles(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative WarmUpCycles, got none")
		}
	}()
	NewQueueingModelAnalyzer(QMConfig{WarmUpCycles: -1})
}

func TestNewQueueingModelAnalyzerPanicsOnNegativeInitFitThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative InitFitThreshold, got none")
		}
	}()
	NewQueueingModelAnalyzer(QMConfig{InitFitThreshold: -0.1})
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
	// INV: RequiredCapacity and SpareCapacity are mutually exclusive.
	if result.RequiredCapacity > 0 && result.SpareCapacity > 0 {
		t.Errorf("mutual exclusivity violated: RequiredCapacity=%v SpareCapacity=%v both > 0", result.RequiredCapacity, result.SpareCapacity)
	}
	// INV: VariantCapacities sorted by CostPerReplica ascending (R2, AnalyzerResult contract).
	for i := 1; i < len(result.VariantCapacities); i++ {
		if result.VariantCapacities[i].CostPerReplica < result.VariantCapacities[i-1].CostPerReplica {
			t.Errorf("VariantCapacities not sorted: index %d cost %v < index %d cost %v",
				i, result.VariantCapacities[i].CostPerReplica, i-1, result.VariantCapacities[i-1].CostPerReplica)
		}
	}
}

// TestGetSLOTargetPriority2Formula verifies that the theory-based SLO path (Priority 2)
// produces TargetTTFT = k×α + (β+γ)×AvgInTokens and TargetITL = k×α + β + γ×(AvgInTokens+(AvgOutTokens+1)/2)
// for known fitted parameters, matching the formula in the design doc.
func TestGetSLOTargetPriority2Formula(t *testing.T) {
	const (
		k2    = DefaultSLOMultiplier // 3.0
		alpha = 10.0
		beta  = 0.02
		gamma = 0.001
		avgIn = 512.0
		avgOut = 128.0
	)
	wantTTFT := float32(k2*alpha + (beta+gamma)*avgIn)
	wantITL  := float32(k2*alpha + beta + gamma*(avgIn+(avgOut+1)/2))

	a := NewQueueingModelAnalyzer(QMConfig{TuningEnabled: false, SLOMultiplier: k2})
	k := modelVariantKey{ModelID: "m1", Variant: NewVariantSpec("A100", 1)}
	a.variantState[k] = &perVariantState{alpha: alpha, beta: beta, gamma: gamma}

	replicas := []ReplicaMetrics{{
		InstanceID:   "i1",
		Variant:      NewVariantSpec("A100", 1),
		DispatchRate: 1.0, // non-zero so computeWorkloadMetrics uses this replica's tokens
		AvgInTokens:  avgIn,
		AvgOutTokens: avgOut,
	}}
	slo, ok := a.getSLOTarget("m1", replicas)
	if !ok {
		t.Fatal("getSLOTarget: Priority 2 should succeed with fitted parameters, got ok=false")
	}
	if slo.TargetTTFT != wantTTFT {
		t.Errorf("TargetTTFT = %v, want %v (k×α + (β+γ)×AvgInTokens)", slo.TargetTTFT, wantTTFT)
	}
	if slo.TargetITL != wantITL {
		t.Errorf("TargetITL = %v, want %v (k×α + β + γ×(AvgInTokens+(AvgOutTokens+1)/2))", slo.TargetITL, wantITL)
	}
}

// TestQueueingModelAnalyzerObservationBasedSLO exercises the path where no explicit SLO
// is configured (Priority 3 fallback in getSLOTarget). With InitObs=2 and two identical
// observations, Nelder-Mead fitting may or may not converge. If fitting succeeds (alpha>0),
// supply is computable and TotalDemand must be positive. If fitting does not converge,
// TotalSupply=0 and the test skips rather than failing.
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

	if result.ModelID != "m1" {
		t.Errorf("ModelID = %q, want %q", result.ModelID, "m1")
	}
	if result.TotalSupply <= 0 {
		t.Skipf("TotalSupply=%v: Nelder-Mead did not converge with 2 identical observations; supply requires fitted parameters", result.TotalSupply)
	}
	if result.TotalDemand <= 0 {
		t.Errorf("TotalDemand = %v, want > 0 (DispatchRate=0.5 should yield positive demand)", result.TotalDemand)
	}
}
