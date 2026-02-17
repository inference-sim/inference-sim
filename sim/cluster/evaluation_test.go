package cluster

import (
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim/trace"
)

func TestNewEvaluationResult_AllFields_PopulatesCorrectly(t *testing.T) {
	// GIVEN all available result data
	metrics := &RawMetrics{RequestsPerSec: 50.0}
	fitness := &FitnessResult{Score: 0.75, Components: map[string]float64{"throughput": 0.75}}
	tr := trace.NewSimulationTrace(trace.TraceConfig{Level: trace.TraceLevelDecisions})
	summary := &trace.TraceSummary{TotalDecisions: 10}

	// WHEN constructing EvaluationResult
	result := NewEvaluationResult(metrics, fitness, tr, summary, 1000000, 5*time.Second)

	// THEN all fields are populated
	if result.Metrics != metrics {
		t.Error("metrics mismatch")
	}
	if result.Fitness != fitness {
		t.Error("fitness mismatch")
	}
	if result.Trace != tr {
		t.Error("trace mismatch")
	}
	if result.Summary != summary {
		t.Error("summary mismatch")
	}
	if result.SimDuration != 1000000 {
		t.Errorf("expected SimDuration 1000000, got %d", result.SimDuration)
	}
	if result.WallTime != 5*time.Second {
		t.Errorf("expected WallTime 5s, got %v", result.WallTime)
	}
}

func TestNewEvaluationResult_NilOptionals_Accepted(t *testing.T) {
	// GIVEN only required data (metrics), optionals nil
	metrics := &RawMetrics{RequestsPerSec: 10.0}

	// WHEN constructing with nils
	result := NewEvaluationResult(metrics, nil, nil, nil, 500000, time.Second)

	// THEN result is valid with nil optionals
	if result.Metrics == nil {
		t.Error("metrics should not be nil")
	}
	if result.Fitness != nil {
		t.Error("fitness should be nil")
	}
	if result.Trace != nil {
		t.Error("trace should be nil")
	}
	if result.Summary != nil {
		t.Error("summary should be nil")
	}
}
