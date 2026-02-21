package workload

import (
	"fmt"
	"math"
)

// InferencePerfSpec defines an inference-perf style workload using a compact
// format. It is expanded into a standard WorkloadSpec via ExpandInferencePerfSpec().
//
// Stage-based rates: sequential rate/duration pairs that produce lifecycle windows.
// Shared prefix: auto-generates N*M clients with prefix groups.
// Multi-turn: maps to BLIS reasoning.multi_turn with context accumulation.
type InferencePerfSpec struct {
	Stages       []StageSpec       `yaml:"stages"`
	SharedPrefix *SharedPrefixSpec `yaml:"shared_prefix"`
}

// StageSpec defines a single rate/duration stage for stage-based load patterns.
type StageSpec struct {
	Rate     float64 `yaml:"rate"`     // requests per second
	Duration int64   `yaml:"duration"` // seconds
}

// SharedPrefixSpec defines shared prefix expansion parameters.
type SharedPrefixSpec struct {
	NumUniqueSystemPrompts  int  `yaml:"num_unique_system_prompts"`
	NumUsersPerSystemPrompt int  `yaml:"num_users_per_system_prompt"`
	SystemPromptLen         int  `yaml:"system_prompt_len"`
	QuestionLen             int  `yaml:"question_len"`
	OutputLen               int  `yaml:"output_len"`
	EnableMultiTurnChat     bool `yaml:"enable_multi_turn_chat"`
}

// validateInferencePerfSpec validates all fields of an InferencePerfSpec.
// Returns error describing the first invalid field found.
func validateInferencePerfSpec(spec *InferencePerfSpec) error {
	if spec == nil {
		return fmt.Errorf("inference_perf spec is nil")
	}
	if len(spec.Stages) == 0 {
		return fmt.Errorf("inference_perf: at least one stage required")
	}
	for i, stage := range spec.Stages {
		if stage.Duration <= 0 {
			return fmt.Errorf("inference_perf.stages[%d]: duration must be positive, got %d", i, stage.Duration)
		}
		if stage.Rate <= 0 || math.IsNaN(stage.Rate) || math.IsInf(stage.Rate, 0) {
			return fmt.Errorf("inference_perf.stages[%d]: rate must be a finite positive number, got %f", i, stage.Rate)
		}
	}
	if spec.SharedPrefix == nil {
		return fmt.Errorf("inference_perf: shared_prefix is required")
	}
	sp := spec.SharedPrefix
	if sp.NumUniqueSystemPrompts <= 0 {
		return fmt.Errorf("inference_perf.shared_prefix: num_unique_system_prompts must be positive, got %d", sp.NumUniqueSystemPrompts)
	}
	if sp.NumUsersPerSystemPrompt <= 0 {
		return fmt.Errorf("inference_perf.shared_prefix: num_users_per_system_prompt must be positive, got %d", sp.NumUsersPerSystemPrompt)
	}
	if sp.SystemPromptLen < 0 {
		return fmt.Errorf("inference_perf.shared_prefix: system_prompt_len must be non-negative, got %d", sp.SystemPromptLen)
	}
	if sp.QuestionLen < 0 {
		return fmt.Errorf("inference_perf.shared_prefix: question_len must be non-negative, got %d", sp.QuestionLen)
	}
	if sp.OutputLen < 0 {
		return fmt.Errorf("inference_perf.shared_prefix: output_len must be non-negative, got %d", sp.OutputLen)
	}
	return nil
}
