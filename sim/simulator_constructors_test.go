package sim

import "testing"

// TestNewGuideLLMConfig_Fields verifies BC-4: constructor maps all parameters to fields.
func TestNewGuideLLMConfig_Fields(t *testing.T) {
	// GIVEN all field values
	// All values MUST be distinct to catch any parameter transposition in the
	// 11-parameter positional constructor (10 ints share the same type).
	cfg := NewGuideLLMConfig(
		0.001, // rate
		100,   // numRequests
		50,    // prefixTokens
		200,   // promptTokens
		20,    // promptTokensStdDev
		110,   // promptTokensMin  (distinct from numRequests=100)
		400,   // promptTokensMax
		150,   // outputTokens
		15,    // outputTokensStdDev
		55,    // outputTokensMin  (distinct from prefixTokens=50)
		300,   // outputTokensMax
	)

	// THEN every field matches the corresponding parameter
	if cfg.Rate != 0.001 {
		t.Errorf("Rate = %f, want 0.001", cfg.Rate)
	}
	if cfg.NumRequests != 100 {
		t.Errorf("NumRequests = %d, want 100", cfg.NumRequests)
	}
	if cfg.PrefixTokens != 50 {
		t.Errorf("PrefixTokens = %d, want 50", cfg.PrefixTokens)
	}
	if cfg.PromptTokens != 200 {
		t.Errorf("PromptTokens = %d, want 200", cfg.PromptTokens)
	}
	if cfg.PromptTokensStdDev != 20 {
		t.Errorf("PromptTokensStdDev = %d, want 20", cfg.PromptTokensStdDev)
	}
	if cfg.PromptTokensMin != 110 {
		t.Errorf("PromptTokensMin = %d, want 110", cfg.PromptTokensMin)
	}
	if cfg.PromptTokensMax != 400 {
		t.Errorf("PromptTokensMax = %d, want 400", cfg.PromptTokensMax)
	}
	if cfg.OutputTokens != 150 {
		t.Errorf("OutputTokens = %d, want 150", cfg.OutputTokens)
	}
	if cfg.OutputTokensStdDev != 15 {
		t.Errorf("OutputTokensStdDev = %d, want 15", cfg.OutputTokensStdDev)
	}
	if cfg.OutputTokensMin != 55 {
		t.Errorf("OutputTokensMin = %d, want 55", cfg.OutputTokensMin)
	}
	if cfg.OutputTokensMax != 300 {
		t.Errorf("OutputTokensMax = %d, want 300", cfg.OutputTokensMax)
	}
}

// TestNewGuideLLMConfig_RateOnlyPlaceholder verifies the workload-spec path
// where only Rate is meaningful and other fields are zero.
func TestNewGuideLLMConfig_RateOnlyPlaceholder(t *testing.T) {
	// GIVEN only rate is set, all others zero
	cfg := NewGuideLLMConfig(0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

	// THEN rate is set and all token fields are zero
	if cfg.Rate != 0.001 {
		t.Errorf("Rate = %f, want 0.001", cfg.Rate)
	}
	if cfg.NumRequests != 0 {
		t.Errorf("NumRequests = %d, want 0", cfg.NumRequests)
	}
	if cfg.PromptTokens != 0 {
		t.Errorf("PromptTokens = %d, want 0", cfg.PromptTokens)
	}
}
