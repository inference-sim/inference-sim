package cluster

import (
	"testing"
	"time"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim/trace"
)

func TestNewEvaluationResult_WithTraceAndSummary_SummaryAccessible(t *testing.T) {
	// GIVEN a simulation with tracing enabled
	config := func() DeploymentConfig {
		d := compatDeployment(cengine.SimConfig{
			Horizon:             5000000,
			Seed:                42,
			KVCacheConfig:       cengine.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         cengine.NewBatchConfig(10, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		}, 0)
		d.NumInstances = 2
		d.TraceLevel = "decisions"
		d.CounterfactualK = 2
		return d
	}()
	requests := testGenerateRequests(42, 5000000, 1.0/1e6, 3,
		0, 10, 0, 10, 10, 5, 0, 5, 5)
	cs := newTestCluster(config, NewSliceRequestSource(requests), nil)
	if err := cs.Run(); err != nil {
		t.Fatalf("cs.Run: %v", err)
	}

	rawMetrics := CollectRawMetrics(cs.AggregatedMetrics(), cs.PerInstanceMetrics(), cs.RejectedRequests(), "", 0, 0, nil)
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
