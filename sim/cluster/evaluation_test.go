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
		NumInstances:       2,
		Horizon:            5000000,
		Seed:               42,
		TotalKVBlocks:      100,
		BlockSizeTokens:    16,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 50, 25},
		TraceLevel:         "decisions",
		CounterfactualK:    2,
	}
	workload := &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, MaxPrompts: 3,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

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
