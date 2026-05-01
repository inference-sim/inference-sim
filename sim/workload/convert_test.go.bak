package workload

import (
	"testing"
)

func TestConvertPreset_ValidParams_ProducesV2Spec(t *testing.T) {
	// GIVEN a valid preset config
	preset := PresetConfig{
		PrefixTokens:      0,
		PromptTokensMean:  512,
		PromptTokensStdev: 256,
		PromptTokensMin:   2,
		PromptTokensMax:   7000,
		OutputTokensMean:  512,
		OutputTokensStdev: 256,
		OutputTokensMin:   2,
		OutputTokensMax:   7000,
	}

	// WHEN converting
	spec, err := ConvertPreset("chatbot", 10.0, 100, preset)

	// THEN a valid v2 spec is produced
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
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
	if spec.Clients[0].InputDist.Type != "gaussian" {
		t.Errorf("input dist type = %q, want %q", spec.Clients[0].InputDist.Type, "gaussian")
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("converted spec fails validation: %v", err)
	}
}

func TestConvertPreset_InvalidRate_ReturnsError(t *testing.T) {
	preset := PresetConfig{}
	_, err := ConvertPreset("test", 0, 100, preset)
	if err == nil {
		t.Fatal("expected error for zero rate")
	}
}

func TestComposeSpecs_TwoSpecs_MergesClients(t *testing.T) {
	// GIVEN two specs with one client each
	specA := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 10.0,
		Clients: []ClientSpec{
			{ID: "a", RateFraction: 1.0, Arrival: ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}}},
		},
	}
	specB := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 5.0,
		Clients: []ClientSpec{
			{ID: "b", RateFraction: 1.0, Arrival: ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 200, "std_dev": 20, "min": 1, "max": 400}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}}},
		},
	}

	// WHEN composing
	merged, err := ComposeSpecs([]*WorkloadSpec{specA, specB})

	// THEN merged spec has both clients, summed rate, renormalized fractions
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(merged.Clients) != 2 {
		t.Fatalf("clients count = %d, want 2", len(merged.Clients))
	}
	if merged.AggregateRate != 15.0 {
		t.Errorf("aggregate_rate = %f, want 15.0", merged.AggregateRate)
	}
	// Rate-weighted fractions: A (10/15 ≈ 0.6667), B (5/15 ≈ 0.3333)
	const eps = 1e-9
	expectedA := 10.0 / 15.0
	expectedB := 5.0 / 15.0
	if diff := merged.Clients[0].RateFraction - expectedA; diff > eps || diff < -eps {
		t.Errorf("client %s rate_fraction = %f, want %f", merged.Clients[0].ID, merged.Clients[0].RateFraction, expectedA)
	}
	if diff := merged.Clients[1].RateFraction - expectedB; diff > eps || diff < -eps {
		t.Errorf("client %s rate_fraction = %f, want %f", merged.Clients[1].ID, merged.Clients[1].RateFraction, expectedB)
	}
}

func TestComposeSpecs_AllConcurrency_MergesClients(t *testing.T) {
	dist := DistSpec{Type: "constant", Params: map[string]float64{"value": 100}}
	spec1 := &WorkloadSpec{
		Version:  "2",
		Category: "language",
		Clients: []ClientSpec{
			{ID: "c1", Concurrency: 5, Arrival: ArrivalSpec{Process: "constant"}, InputDist: dist, OutputDist: dist},
		},
	}
	spec2 := &WorkloadSpec{
		Version:  "2",
		Category: "language",
		Clients: []ClientSpec{
			{ID: "c2", Concurrency: 10, Arrival: ArrivalSpec{Process: "constant"}, InputDist: dist, OutputDist: dist},
		},
	}
	merged, err := ComposeSpecs([]*WorkloadSpec{spec1, spec2})
	if err != nil {
		t.Fatalf("ComposeSpecs failed for all-concurrency specs: %v", err)
	}
	if len(merged.Clients) != 2 {
		t.Errorf("expected 2 merged clients, got %d", len(merged.Clients))
	}
	if merged.AggregateRate != 0 {
		t.Errorf("expected AggregateRate=0 for all-concurrency, got %f", merged.AggregateRate)
	}
	if err := merged.Validate(); err != nil {
		t.Errorf("merged spec fails validation: %v", err)
	}
}

func TestComposeSpecs_NegativeTotalRate_ReturnsError(t *testing.T) {
	dist := DistSpec{Type: "constant", Params: map[string]float64{"value": 100}}
	spec := &WorkloadSpec{
		Version:       "2",
		Category:      "language",
		AggregateRate: -5.0,
		Clients: []ClientSpec{
			{ID: "c1", RateFraction: 1.0, Arrival: ArrivalSpec{Process: "poisson"}, InputDist: dist, OutputDist: dist},
		},
	}
	_, err := ComposeSpecs([]*WorkloadSpec{spec})
	if err == nil {
		t.Error("expected error for negative aggregate rate")
	}
}

func TestComposeSpecs_EmptyList_ReturnsError(t *testing.T) {
	_, err := ComposeSpecs(nil)
	if err == nil {
		t.Fatal("expected error for empty spec list")
	}
}

func TestConvertServeGen_EmptyPath_ReturnsError(t *testing.T) {
	_, err := ConvertServeGen("", "")
	if err == nil {
		t.Fatal("expected error for empty path")
	}
}

func TestConvertInferencePerf_EmptyPath_ReturnsError(t *testing.T) {
	_, err := ConvertInferencePerf("")
	if err == nil {
		t.Fatal("expected error for empty path")
	}
}
