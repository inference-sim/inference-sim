package workload

import (
	"os"
	"testing"

	"gopkg.in/yaml.v3"
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

func TestConvertServeGen_EmptyPath_ReturnsError(t *testing.T) {
	_, err := ConvertServeGen("", 600, 180)
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

// #1915 (DEV-7): a converted multi-stage inference-perf spec offers each stage its own rate,
// as the native expansion does. The old conversion gave every stage the summed rate.
func TestConvertInferencePerf_KeepsEachStagesRate(t *testing.T) {
	path := t.TempDir() + "/ip.yaml"
	src := `version: "1"
seed: 42
inference_perf:
  stages:
    - rate: 8.0
      duration: 20
    - rate: 20.0
      duration: 20
  shared_prefix:
    num_unique_system_prompts: 3
    num_users_per_system_prompt: 2
    system_prompt_len: 10
    question_len: 50
    output_len: 20
`
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	native, err := LoadWorkloadSpec(path)
	if err != nil {
		t.Fatal(err)
	}
	nativeSpec, err := ExpandInferencePerfSpec(native.InferencePerf, native.Seed)
	if err != nil {
		t.Fatal(err)
	}
	conv, err := ConvertInferencePerf(path)
	if err != nil {
		t.Fatal(err)
	}
	// The converted spec survives serialisation: no in-memory sampler is needed.
	data, err := yaml.Marshal(conv)
	if err != nil {
		t.Fatal(err)
	}
	var round WorkloadSpec
	if err := yaml.Unmarshal(data, &round); err != nil {
		t.Fatal(err)
	}
	perStage := func(s *WorkloadSpec) [2]int {
		reqs, err := generateRequests(s, 40_000_000, 0)
		if err != nil {
			t.Fatal(err)
		}
		var n [2]int
		for _, r := range reqs {
			if r.ArrivalTime < 20_000_000 {
				n[0]++
			} else {
				n[1]++
			}
		}
		return n
	}
	want, got := perStage(nativeSpec), perStage(&round)
	for i := range want {
		if d := float64(got[i]-want[i]) / float64(want[i]); d > 0.1 || d < -0.1 {
			t.Errorf("stage %d: converted spec offers %d requests, native %d (rates 8 and 20 req/s over 20 s)", i, got[i], want[i])
		}
	}
	if want[0] < 100 || want[1] < 300 {
		t.Fatalf("non-vacuity: native stage counts %v", want)
	}
}

// Composing keeps what the inputs bound.
