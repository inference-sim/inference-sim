package workload

import (
	"strings"
	"testing"
)

func TestValidateInferencePerfSpec_ValidSpec_NoError(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
			{Rate: 20.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  9,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	if err := validateInferencePerfSpec(spec); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidateInferencePerfSpec_ZeroDuration_ReturnsError(t *testing.T) {
	// BC-10: zero-duration stages rejected
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 0},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for zero duration")
	}
	if !strings.Contains(err.Error(), "duration must be positive") {
		t.Errorf("error should mention duration: %v", err)
	}
}

func TestValidateInferencePerfSpec_ZeroPrompts_ReturnsError(t *testing.T) {
	// BC-11: zero system prompts rejected
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  0,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for zero system prompts")
	}
}

func TestValidateInferencePerfSpec_NegativeLength_ReturnsError(t *testing.T) {
	// BC-12: negative lengths rejected
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         -1,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for negative system_prompt_len")
	}
}

func TestValidateInferencePerfSpec_NegativeRate_ReturnsError(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: -1.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for negative rate")
	}
}

func TestValidateInferencePerfSpec_NoStages_ReturnsError(t *testing.T) {
	spec := &InferencePerfSpec{
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for no stages")
	}
}

func TestValidateInferencePerfSpec_NoSharedPrefix_ReturnsError(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for nil shared_prefix")
	}
}
