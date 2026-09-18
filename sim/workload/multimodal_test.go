package workload

import (
	"math/rand"
	"testing"
)

func TestGenerateMultimodalTokens_TokenAccounting(t *testing.T) {
	// BC-8: text + image + audio + video = total input
	rng := rand.New(rand.NewSource(42))
	mm := &MultimodalSpec{
		TextDist:       DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 50, "max": 200}},
		ImageDist:      DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 200, "std_dev": 20, "min": 100, "max": 400}},
		ImageCountDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 2, "std_dev": 1, "min": 1, "max": 5}},
		AudioDist:      DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 10, "min": 20, "max": 100}},
		AudioCountDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 1, "std_dev": 0, "min": 1, "max": 1}},
	}

	tokens, textLen, imageLen, audioLen, videoLen, err := GenerateMultimodalTokens(rng, mm)
	if err != nil {
		t.Fatal(err)
	}

	totalModality := textLen + imageLen + audioLen + videoLen
	if len(tokens) != totalModality {
		t.Errorf("len(tokens) = %d, modality sum = %d (text=%d image=%d audio=%d video=%d)",
			len(tokens), totalModality, textLen, imageLen, audioLen, videoLen)
	}
	if textLen == 0 {
		t.Error("expected non-zero text tokens")
	}
	if imageLen == 0 {
		t.Error("expected non-zero image tokens")
	}
}

func TestGenerateMultimodalTokens_ZeroModalities(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	mm := &MultimodalSpec{
		TextDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 50, "max": 200}},
	}

	tokens, textLen, imageLen, audioLen, videoLen, err := GenerateMultimodalTokens(rng, mm)
	if err != nil {
		t.Fatal(err)
	}
	if imageLen != 0 || audioLen != 0 || videoLen != 0 {
		t.Errorf("unconfigured modalities should be 0: image=%d audio=%d video=%d", imageLen, audioLen, videoLen)
	}
	if textLen == 0 {
		t.Error("text should be non-zero")
	}
	if len(tokens) != textLen {
		t.Errorf("tokens len = %d, want %d (text only)", len(tokens), textLen)
	}
}

func TestGenerateMultimodalTokens_EmptySpec_MinimumOneToken(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	mm := &MultimodalSpec{}

	tokens, _, _, _, _, err := GenerateMultimodalTokens(rng, mm)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) < 1 {
		t.Errorf("expected at least 1 token, got %d", len(tokens))
	}
}

func TestGenerateMultimodalTokens_InvalidDistType_ReturnsError(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	mm := &MultimodalSpec{
		TextDist: DistSpec{Type: "invalid_type"},
	}
	_, _, _, _, _, err := GenerateMultimodalTokens(rng, mm)
	if err == nil {
		t.Fatal("expected error for invalid distribution type")
	}
}
