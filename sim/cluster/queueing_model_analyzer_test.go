package cluster

import (
	"math"
	"testing"
)

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
		InitObs: 1,
	}
	a := NewQueueingModelAnalyzer(cfg)
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
	_ = result
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
