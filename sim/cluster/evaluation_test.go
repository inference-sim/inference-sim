package cluster

import (
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

func TestNewEvaluationResult_WithTraceAndSummary_SummaryAccessible(t *testing.T) {
	// GIVEN a simulation with tracing enabled
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       5000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:    2,
		TraceLevel:      "decisions",
		CounterfactualK: 2,
	}
	workload := &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, NumRequests: 3,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")
	if err := cs.Run(); err != nil {
		t.Fatalf("cs.Run: %v", err)
	}

	rawMetrics := CollectRawMetrics(cs.AggregatedMetrics(), cs.PerInstanceMetrics(), cs.RejectedRequests(), "")
	traceSummary := trace.Summarize(cs.Trace())

	// WHEN constructing EvaluationResult from real simulation output
	result := NewEvaluationResult(rawMetrics, nil, cs.Trace(), traceSummary, cs.Clock(), 100*time.Millisecond)

	// THEN summary reflects actual simulation decisions
	if result.Summary == nil {
		t.Fatal("expected non-nil summary")
	}
	if result.Summary.TotalDecisions != 3 {
		t.Errorf("expected 3 total decisions, got %d", result.Summary.TotalDecisions)
	}
	if result.Summary.AdmittedCount != 3 {
		t.Errorf("expected 3 admitted (always-admit default), got %d", result.Summary.AdmittedCount)
	}
	if result.SimDuration <= 0 {
		t.Errorf("expected positive SimDuration, got %d", result.SimDuration)
	}
}
