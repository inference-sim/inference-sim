package workload

import (
	"os"
	"path/filepath"
	"testing"
)

func TestConvertCSVTrace_ValidFile_ProducesV2Spec(t *testing.T) {
	// GIVEN a valid CSV trace file
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "trace.csv")
	content := "arrival_time,col1,col2,prefill_tokens,decode_tokens\n" +
		"0.0,a,b,\"[1,2,3]\",\"[4,5]\"\n" +
		"1.0,a,b,\"[6,7,8,9]\",\"[10,11,12]\"\n" +
		"2.0,a,b,\"[13,14]\",\"[15]\"\n"
	if err := os.WriteFile(csvPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN converting to v2 spec
	spec, err := ConvertCSVTrace(csvPath, 0)

	// THEN a valid spec is returned with correct aggregate properties
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.Version != "2" {
		t.Errorf("version = %q, want %q", spec.Version, "2")
	}
	if spec.NumRequests != 3 {
		t.Errorf("num_requests = %d, want 3", spec.NumRequests)
	}
	if len(spec.Clients) != 1 {
		t.Fatalf("clients count = %d, want 1", len(spec.Clients))
	}
	if spec.Clients[0].Arrival.Process != "constant" {
		t.Errorf("arrival process = %q, want %q", spec.Clients[0].Arrival.Process, "constant")
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("converted spec fails validation: %v", err)
	}
}

func TestConvertCSVTrace_EmptyFile_ReturnsError(t *testing.T) {
	// GIVEN a CSV file with only a header
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "empty.csv")
	content := "arrival_time,col1,col2,prefill_tokens,decode_tokens\n"
	if err := os.WriteFile(csvPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN converting
	_, err := ConvertCSVTrace(csvPath, 0)

	// THEN an error is returned
	if err == nil {
		t.Fatal("expected error for empty CSV")
	}
}

func TestConvertCSVTrace_MalformedRow_ReturnsErrorWithLine(t *testing.T) {
	// GIVEN a CSV file with a malformed row
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "bad.csv")
	content := "arrival_time,col1,col2,prefill_tokens,decode_tokens\n" +
		"0.0,a,b,\"[1,2]\",\"[3]\"\n" +
		"notanumber,a,b,\"[1]\",\"[2]\"\n"
	if err := os.WriteFile(csvPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN converting
	_, err := ConvertCSVTrace(csvPath, 0)

	// THEN an error is returned with row context
	if err == nil {
		t.Fatal("expected error for malformed row")
	}
}

func TestConvertCSVTrace_HorizonTruncation(t *testing.T) {
	// GIVEN a CSV with requests spanning 3 seconds
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "trace.csv")
	content := "arrival_time,col1,col2,prefill_tokens,decode_tokens\n" +
		"0.0,a,b,\"[1]\",\"[2]\"\n" +
		"1.0,a,b,\"[3]\",\"[4]\"\n" +
		"2.0,a,b,\"[5]\",\"[6]\"\n"
	if err := os.WriteFile(csvPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN converting with horizon = 1.5s (1_500_000 µs)
	spec, err := ConvertCSVTrace(csvPath, 1_500_000)

	// THEN only 2 requests are included (arrival 0.0 and 1.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.NumRequests != 2 {
		t.Errorf("num_requests = %d, want 2 (horizon truncation)", spec.NumRequests)
	}
}

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
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "stdev": 10, "min": 1, "max": 200}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "stdev": 5, "min": 1, "max": 100}}},
		},
	}
	specB := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 5.0,
		Clients: []ClientSpec{
			{ID: "b", RateFraction: 1.0, Arrival: ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 200, "stdev": 20, "min": 1, "max": 400}},
				OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "stdev": 10, "min": 1, "max": 200}}},
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
	// Each had fraction 1.0, so each should be renormalized to 0.5
	for _, c := range merged.Clients {
		if c.RateFraction != 0.5 {
			t.Errorf("client %s rate_fraction = %f, want 0.5", c.ID, c.RateFraction)
		}
	}
}

func TestComposeSpecs_EmptyList_ReturnsError(t *testing.T) {
	_, err := ComposeSpecs(nil)
	if err == nil {
		t.Fatal("expected error for empty spec list")
	}
}

func TestConvertServeGen_EmptyPath_ReturnsError(t *testing.T) {
	_, err := ConvertServeGen("")
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
