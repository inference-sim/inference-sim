package evaluation

import (
	"math"
	"testing"
)

// run mirrors a published serving sweep: per-concurrency rows carrying the two
// latency metrics, throughput, KV utilization and cache hit rate.
func run() *Run {
	return &Run{
		Kind: "EvaluationRun", Name: "granite-230b-h200-tp8",
		Scenario: "granite-230b-h200-tp8", Harness: "aiperf",
		EngineVersion: "0.29.0", Date: "2026-09-20",
		Points: []Point{
			{Concurrency: 1, CompletedRequests: 96, OutputTokensPerSec: 157.28,
				InputTokensPerSec: 2542.98, TTFTms: 168.59, ITLms: 6.02,
				MeanISL: 8015, MeanOSL: 496, KVUtilization: 0.10},
			{Concurrency: 2, CompletedRequests: 180, OutputTokensPerSec: 287.58,
				InputTokensPerSec: 4761.51, TTFTms: 52.52, ITLms: 6.73,
				MeanISL: 8015, MeanOSL: 484, KVUtilization: 0.21,
				PrefixCacheHitRate: 0.538},
		},
	}
}

func TestValidRunPasses(t *testing.T) {
	if p := run().Validate(); !p.OK() {
		t.Fatalf("a valid run was rejected:\n%s", p.Error())
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Run)
	}{
		{"wrong kind", func(r *Run) { r.Kind = "Benchmark" }},
		{"no scenario", func(r *Run) { r.Scenario = "" }},
		{"no harness", func(r *Run) { r.Harness = "" }},
		{"no engine version", func(r *Run) { r.EngineVersion = "" }},
		{"no points", func(r *Run) { r.Points = nil }},
		{"zero concurrency", func(r *Run) { r.Points[0].Concurrency = 0 }},
		{"duplicate concurrency", func(r *Run) {
			r.Points[1].Concurrency = r.Points[0].Concurrency
		}},
		{"negative throughput", func(r *Run) { r.Points[0].OutputTokensPerSec = -1 }},
		{"negative ttft", func(r *Run) { r.Points[0].TTFTms = -1 }},
		{"utilization above one", func(r *Run) { r.Points[0].KVUtilization = 1.5 }},
		{"hit rate below zero", func(r *Run) { r.Points[0].PrefixCacheHitRate = -0.1 }},
		{"negative preemptions", func(r *Run) { r.Points[0].Preemptions = -1 }},
		{"NaN throughput", func(r *Run) { r.Points[0].OutputTokensPerSec = math.NaN() }},
		{"Inf ttft", func(r *Run) { r.Points[0].TTFTms = math.Inf(1) }},
		{"NaN utilization", func(r *Run) { r.Points[0].KVUtilization = math.NaN() }},
		{"NaN e2e", func(r *Run) { r.Points[0].EndToEndms = math.NaN() }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := run()
			tc.mutate(r)
			if p := r.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// TestThroughputWithoutLatencyWarns: such a point cannot be scored against a
// prediction, which is the record's purpose.
func TestThroughputWithoutLatencyWarns(t *testing.T) {
	r := run()
	r.Points[0].TTFTms = 0
	r.Points[0].ITLms = 0
	p := r.Validate()
	if !p.OK() {
		t.Errorf("should warn rather than fail:\n%s", p.Error())
	}
	if len(p.All()) == 0 {
		t.Error("expected a warning that the point cannot be scored")
	}
}
