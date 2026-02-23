package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

func TestClusterSimulator_TraceLevelNone_NilTrace(t *testing.T) {
	// GIVEN trace level none (default)
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       1000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances: 2,
		TraceLevel:   "none",
	}
	workload := &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, NumRequests: 3,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")

	// WHEN run
	mustRun(t, cs)

	// THEN trace is nil (zero overhead)
	if cs.Trace() != nil {
		t.Error("expected nil trace for trace-level none")
	}
}

func TestClusterSimulator_TraceLevelDecisions_RecordsAllEvents(t *testing.T) {
	// GIVEN trace level decisions with 5 requests and 2 instances
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:    2,
		TraceLevel:      "decisions",
		CounterfactualK: 0,
	}
	workload := &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, NumRequests: 5,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")

	// WHEN run
	mustRun(t, cs)

	// THEN trace is non-nil with admission and routing records
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace for trace-level decisions")
	}
	if len(tr.Admissions) != 5 {
		t.Errorf("expected 5 admission records (one per request), got %d", len(tr.Admissions))
	}
	if len(tr.Routings) != 5 {
		t.Errorf("expected 5 routing records (all admitted with always-admit), got %d", len(tr.Routings))
	}

	// All admission records should be admitted (default always-admit)
	for i, a := range tr.Admissions {
		if !a.Admitted {
			t.Errorf("admission[%d]: expected admitted=true", i)
		}
	}
}

func TestClusterSimulator_TraceLevelDecisions_WithCounterfactual(t *testing.T) {
	// GIVEN trace with counterfactual k=2 and weighted scoring
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:         2,
		RoutingPolicy:        "weighted",
		RoutingScorerConfigs: sim.DefaultScorerConfigs(),
		TraceLevel:           "decisions",
		CounterfactualK:      2,
	}
	workload := &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, NumRequests: 3,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")

	// WHEN run
	mustRun(t, cs)

	// THEN routing records have candidates
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	for i, r := range tr.Routings {
		if len(r.Candidates) == 0 {
			t.Errorf("routing[%d]: expected candidates with k=2, got none", i)
		}
		if len(r.Candidates) > 2 {
			t.Errorf("routing[%d]: expected at most 2 candidates, got %d", i, len(r.Candidates))
		}
	}
}

func TestClusterSimulator_TraceWithTokenBucket_RecordsRejections(t *testing.T) {
	// GIVEN token bucket admission that rejects some requests
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       5000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:          1,
		AdmissionPolicy:       "token-bucket",
		TokenBucketCapacity:   2,        // very small: only 2 tokens
		TokenBucketRefillRate: 0.000001, // near-zero refill
		TraceLevel:            "decisions",
	}
	workload := &sim.GuideLLMConfig{
		Rate: 5.0 / 1e6, NumRequests: 10,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")

	// WHEN run
	mustRun(t, cs)

	// THEN some admissions are rejected
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	rejections := 0
	for _, a := range tr.Admissions {
		if !a.Admitted {
			rejections++
		}
	}
	if rejections == 0 {
		t.Error("expected some rejections with tiny token bucket, got 0")
	}
	// Routing records should be fewer than admissions (rejected requests don't route)
	if len(tr.Routings) >= len(tr.Admissions) {
		t.Errorf("expected fewer routings (%d) than admissions (%d) with rejections",
			len(tr.Routings), len(tr.Admissions))
	}

	// Summarize to verify rejection counts match
	summary := trace.Summarize(tr)
	if summary.RejectedCount != rejections {
		t.Errorf("summary rejected count %d != counted rejections %d", summary.RejectedCount, rejections)
	}
}
