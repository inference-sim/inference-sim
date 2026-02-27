package workload

import (
	"testing"
)

func TestSynthesizeFromDistribution_ProducesValidSpec(t *testing.T) {
	params := DistributionParams{
		Rate:               10.0,
		NumRequests:        100,
		PrefixTokens:       0,
		PromptTokensMean:   512,
		PromptTokensStdDev: 256,
		PromptTokensMin:    2,
		PromptTokensMax:    7000,
		OutputTokensMean:   512,
		OutputTokensStdDev: 256,
		OutputTokensMin:    2,
		OutputTokensMax:    7000,
	}

	spec := SynthesizeFromDistribution(params)

	if spec.Version != "2" {
		t.Errorf("version = %q, want %q", spec.Version, "2")
	}
	if spec.AggregateRate != 10.0 {
		t.Errorf("aggregate_rate = %f, want 10.0", spec.AggregateRate)
	}
	if spec.NumRequests != 100 {
		t.Errorf("num_requests = %d, want 100", spec.NumRequests)
	}
	if len(spec.Clients) != 1 {
		t.Fatalf("clients count = %d, want 1", len(spec.Clients))
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("synthesized spec fails validation: %v", err)
	}
}

func TestSynthesizeFromDistribution_ConstantArrival(t *testing.T) {
	params := DistributionParams{
		Rate:             1.0,
		NumRequests:      10,
		PromptTokensMean: 100, PromptTokensStdDev: 10, PromptTokensMin: 1, PromptTokensMax: 200,
		OutputTokensMean: 50, OutputTokensStdDev: 5, OutputTokensMin: 1, OutputTokensMax: 100,
	}

	spec := SynthesizeFromDistribution(params)

	if spec.Clients[0].Arrival.Process != "constant" {
		t.Errorf("arrival process = %q, want %q", spec.Clients[0].Arrival.Process, "constant")
	}
}

func TestSynthesizeFromDistribution_GaussianDistributions(t *testing.T) {
	params := DistributionParams{
		Rate:               10.0,
		NumRequests:        100,
		PromptTokensMean:   512,
		PromptTokensStdDev: 256,
		PromptTokensMin:    2,
		PromptTokensMax:    7000,
		OutputTokensMean:   128,
		OutputTokensStdDev: 64,
		OutputTokensMin:    1,
		OutputTokensMax:    1000,
	}

	spec := SynthesizeFromDistribution(params)

	c := spec.Clients[0]
	if c.InputDist.Type != "gaussian" {
		t.Errorf("input dist type = %q, want %q", c.InputDist.Type, "gaussian")
	}
	if c.InputDist.Params["mean"] != 512 {
		t.Errorf("input mean = %f, want 512", c.InputDist.Params["mean"])
	}
	if c.OutputDist.Type != "gaussian" {
		t.Errorf("output dist type = %q, want %q", c.OutputDist.Type, "gaussian")
	}
	if c.OutputDist.Params["mean"] != 128 {
		t.Errorf("output mean = %f, want 128", c.OutputDist.Params["mean"])
	}
}

func TestSynthesizeFromDistribution_WithPrefix(t *testing.T) {
	params := DistributionParams{
		Rate:             10.0,
		NumRequests:      100,
		PrefixTokens:     50,
		PromptTokensMean: 100, PromptTokensStdDev: 10, PromptTokensMin: 1, PromptTokensMax: 200,
		OutputTokensMean: 50, OutputTokensStdDev: 5, OutputTokensMin: 1, OutputTokensMax: 100,
	}

	spec := SynthesizeFromDistribution(params)

	if spec.Clients[0].PrefixGroup != "shared" {
		t.Errorf("prefix_group = %q, want %q", spec.Clients[0].PrefixGroup, "shared")
	}
	if spec.Clients[0].PrefixLength != 50 {
		t.Errorf("prefix_length = %d, want 50", spec.Clients[0].PrefixLength)
	}
}

func TestSynthesizeFromDistribution_GeneratesRequests(t *testing.T) {
	// Integration: synthesized spec -> GenerateRequests -> non-empty requests
	params := DistributionParams{
		Rate:             10.0,
		NumRequests:      20,
		PromptTokensMean: 100, PromptTokensStdDev: 10, PromptTokensMin: 1, PromptTokensMax: 200,
		OutputTokensMean: 50, OutputTokensStdDev: 5, OutputTokensMin: 1, OutputTokensMax: 100,
	}

	spec := SynthesizeFromDistribution(params)
	spec.Seed = 42

	requests, err := GenerateRequests(spec, 100_000_000, int64(params.NumRequests))
	if err != nil {
		t.Fatalf("GenerateRequests failed: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected non-empty request list from synthesized spec")
	}
	if int64(len(requests)) > int64(params.NumRequests) {
		t.Errorf("got %d requests, expected at most %d", len(requests), params.NumRequests)
	}
}
