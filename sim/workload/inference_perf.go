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

// ExpandInferencePerfSpec converts an InferencePerfSpec into a standard WorkloadSpec.
// The seed is passed through to the resulting WorkloadSpec.
// Returns error if the spec is invalid.
func ExpandInferencePerfSpec(spec *InferencePerfSpec, seed int64) (*WorkloadSpec, error) {
	if err := validateInferencePerfSpec(spec); err != nil {
		return nil, fmt.Errorf("validating inference-perf spec: %w", err)
	}

	sp := spec.SharedPrefix
	numClients := sp.NumUniqueSystemPrompts * sp.NumUsersPerSystemPrompt

	// Compute aggregate rate as time-weighted average across stages.
	// Division is safe: validation guarantees at least one stage with positive duration.
	var totalDuration int64
	var weightedRateSum float64
	for _, stage := range spec.Stages {
		totalDuration += stage.Duration
		weightedRateSum += stage.Rate * float64(stage.Duration)
	}
	aggregateRate := weightedRateSum / float64(totalDuration)

	// Compute lifecycle windows from stages (nil for single stage)
	windows := stagesToWindows(spec.Stages)

	// Build constant distributions for fixed lengths
	inputDist := constantDist(float64(sp.QuestionLen))
	outputDist := constantDist(float64(sp.OutputLen))

	// Build optional reasoning spec for multi-turn
	var reasoning *ReasoningSpec
	if sp.EnableMultiTurnChat {
		reasoning = &ReasoningSpec{
			ReasonRatioDist: DistSpec{
				Type:   "constant",
				Params: map[string]float64{"value": 0},
			},
			MultiTurn: &MultiTurnSpec{
				MaxRounds:     5,
				ThinkTimeUs:   500000, // 500ms
				ContextGrowth: "accumulate",
			},
		}
	}

	category := "language"
	if sp.EnableMultiTurnChat {
		category = "reasoning"
	}

	// Generate N*M clients
	clients := make([]ClientSpec, 0, numClients)
	rateFraction := 1.0 / float64(numClients)

	for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
		prefixGroup := fmt.Sprintf("prompt-%d", p)
		for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
			clientID := fmt.Sprintf("prompt-%d-user-%d", p, u)
			client := ClientSpec{
				ID:           clientID,
				TenantID:     prefixGroup,
				SLOClass:     "batch",
				RateFraction: rateFraction,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    inputDist,
				OutputDist:   outputDist,
				PrefixGroup:  prefixGroup,
				PrefixLength: sp.SystemPromptLen,
				Reasoning:    reasoning,
			}
			if len(windows) > 0 {
				client.Lifecycle = &LifecycleSpec{Windows: windows}
			}
			clients = append(clients, client)
		}
	}

	return &WorkloadSpec{
		Version:       "1",
		Seed:          seed,
		Category:      category,
		AggregateRate: aggregateRate,
		Clients:       clients,
	}, nil
}

// stagesToWindows converts stage specs into lifecycle ActiveWindows.
// Returns nil for single-stage specs (always active, no windows needed).
// Duration is in seconds, converted to microseconds for BLIS.
func stagesToWindows(stages []StageSpec) []ActiveWindow {
	if len(stages) <= 1 {
		return nil
	}
	windows := make([]ActiveWindow, len(stages))
	var offsetUs int64
	for i, stage := range stages {
		durationUs := stage.Duration * 1_000_000 // seconds to microseconds
		windows[i] = ActiveWindow{
			StartUs: offsetUs,
			EndUs:   offsetUs + durationUs,
		}
		offsetUs += durationUs
	}
	return windows
}

// constantDist creates a DistSpec for a constant (zero-variance) distribution.
func constantDist(value float64) DistSpec {
	return DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": value},
	}
}
