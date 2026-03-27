package workload

import (
	"fmt"
	"math"
	"math/rand"
)

// InferencePerfSpec defines an inference-perf style workload using a compact
// format. It is expanded into a standard WorkloadSpec via ExpandInferencePerfSpec().
//
// Stage-based rates: sequential rate/duration pairs that produce lifecycle windows.
// Shared prefix: auto-generates N*M clients with prefix groups.
// Multi-turn: maps to BLIS reasoning.multi_turn with fixed-length inputs (no context accumulation).
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
//
// Single-stage: N*M clients with no lifecycle windows, aggregateRate = stage rate.
// Multi-stage: N*M clients per stage, each active only during its stage's window.
// aggregateRate = sum of stage rates; each client's rateFraction = stageRate / (N*M).
// This ensures each stage emits at its configured rate during its time window.
//
// Returns error if the spec is invalid.
func ExpandInferencePerfSpec(spec *InferencePerfSpec, seed int64) (*WorkloadSpec, error) {
	if err := validateInferencePerfSpec(spec); err != nil {
		return nil, fmt.Errorf("validating inference-perf spec: %w", err)
	}

	sp := spec.SharedPrefix
	numClientsPerStage := sp.NumUniqueSystemPrompts * sp.NumUsersPerSystemPrompt

	// Build constant distributions for fixed lengths
	inputDist := constantDist(float64(sp.QuestionLen))
	outputDist := constantDist(float64(sp.OutputLen))

	category := "language"
	if sp.EnableMultiTurnChat {
		category = "reasoning"
	}

	var clients []ClientSpec
	var aggregateRate float64

	if len(spec.Stages) == 1 {
		// Single stage: no lifecycle windows needed.
		// Use NormalizedExponentialSampler for inference-perf parity:
		// each client pre-generates N intervals that sum exactly to stage duration.
		stage := spec.Stages[0]
		aggregateRate = stage.Rate
		rateFraction := 1.0 / float64(numClientsPerStage)
		clients = make([]ClientSpec, 0, numClientsPerStage)

		var reasoning *ReasoningSpec
		if sp.EnableMultiTurnChat {
			reasoning = computeReasoningSpec(stage.Rate, stage.Duration, numClientsPerStage)
		}

		// For multi-turn (reasoning) workloads, use Poisson arrival for session start time.
		// The rounds within each session are spaced by ThinkTimeUs (not sampled IATs).
		// NormalizedExponentialSampler only applies to non-reasoning (language) workloads
		// where each request is independent (not part of a multi-round session).
		if sp.EnableMultiTurnChat {
			// Multi-turn: use Poisson, rounds controlled by MaxRounds + ThinkTimeUs
			for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
				prefixGroup := fmt.Sprintf("prompt-%d", p)
				for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
					clientID := fmt.Sprintf("prompt-%d-user-%d", p, u)
					clients = append(clients, ClientSpec{
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
					})
				}
			}
		} else {
			// Single-request (language) workload: use NormalizedExponentialSampler
			// Each client gets exactly ceil(stageRate * duration / numClients) requests
			// over the stage duration, using normalized exponential distribution.
			// NOTE: math.Ceil means total requests may exceed stage.Rate * stage.Duration
			// by up to numClients-1. This matches inference-perf behavior.
			requestsPerClient := int64(math.Ceil(stage.Rate * float64(stage.Duration) / float64(numClientsPerStage)))
			durationUs := stage.Duration * 1_000_000 // seconds to microseconds

			// Validate sampler parameters before construction (prevent panic on user input)
			if requestsPerClient <= 0 {
				return nil, fmt.Errorf("inference_perf: requestsPerClient must be positive, got %d", requestsPerClient)
			}
			if requestsPerClient > 10_000_000 {
				return nil, fmt.Errorf("inference_perf: requestsPerClient %d exceeds safety limit (10M); reduce rate, duration, or increase clients", requestsPerClient)
			}
			if durationUs < requestsPerClient {
				return nil, fmt.Errorf("inference_perf: durationUs (%d) < requestsPerClient (%d) produces degenerate distribution", durationUs, requestsPerClient)
			}

			// Defensive: prevent integer overflow in seed calculation (cast before multiply)
			totalClients := int64(sp.NumUniqueSystemPrompts) * int64(sp.NumUsersPerSystemPrompt)
			if totalClients > 1_000_000 {
				return nil, fmt.Errorf("total client count %d exceeds safety limit (1M)", totalClients)
			}

			// Create factory closure that captures requestsPerClient and durationUs.
			// Each GenerateRequests call will invoke this factory with a sub-RNG,
			// producing a fresh sampler instance (workload reusability).
			factory := func(rng *rand.Rand) ArrivalSampler {
				return NewNormalizedExponentialSampler(rng, requestsPerClient, durationUs)
			}

			for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
				prefixGroup := fmt.Sprintf("prompt-%d", p)
				for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
					clientID := fmt.Sprintf("prompt-%d-user-%d", p, u)

					clients = append(clients, ClientSpec{
						ID:                   clientID,
						TenantID:             prefixGroup,
						SLOClass:             "batch",
						RateFraction:         rateFraction,
						Arrival:              ArrivalSpec{Process: "poisson"}, // Fallback for diagnostics/serialization
						CustomSamplerFactory: factory,
						InputDist:            inputDist,
						OutputDist:           outputDist,
						PrefixGroup:          prefixGroup,
						PrefixLength:         sp.SystemPromptLen,
						Reasoning:            reasoning,
					})
				}
			}
		}
	} else {
		// Multi-stage: create per-stage client cohorts with lifecycle windows.
		// Each stage's N*M clients are active only during that stage's window
		// and emit at that stage's rate.
		//
		// Math: aggregateRate = sum(stageRates), rateFraction = stageRate/numClientsPerStage.
		// After normalization, each client's rate = stageRate/numClientsPerStage.
		// During a stage, N*M clients are active → total rate = stageRate.
		//
		// NOTE: Multi-stage workloads use Poisson (not NormalizedExponentialSampler).
		// Rationale: NormalizedExponentialSampler pre-generates intervals spanning the
		// full workload duration, but per-stage clients are only active during their
		// stage's window (a subset of the full duration). This mismatch would waste
		// intervals or require complex windowing. Poisson generates incrementally
		// during the active window, matching the per-stage lifecycle architecture.
		// See BC-4 in plan for full details.
		windows := stagesToWindows(spec.Stages)

		for _, stage := range spec.Stages {
			aggregateRate += stage.Rate
		}

		clients = make([]ClientSpec, 0, numClientsPerStage*len(spec.Stages))
		for s, stage := range spec.Stages {
			rateFraction := stage.Rate / float64(numClientsPerStage)
			stageLifecycle := &LifecycleSpec{
				Windows: []ActiveWindow{windows[s]},
			}

			var reasoning *ReasoningSpec
			if sp.EnableMultiTurnChat {
				reasoning = computeReasoningSpec(stage.Rate, stage.Duration, numClientsPerStage)
			}

			for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
				prefixGroup := fmt.Sprintf("prompt-%d", p)
				for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
					clientID := fmt.Sprintf("stage-%d-prompt-%d-user-%d", s, p, u)
					clients = append(clients, ClientSpec{
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
						Lifecycle:    stageLifecycle,
					})
				}
			}
		}
	}

	return &WorkloadSpec{
		Version:       "2",
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

// computeReasoningSpec builds a ReasoningSpec for inference-perf multi-turn mode.
// It derives MaxRounds and ThinkTimeUs from stage parameters to match inference-perf's
// round-robin cycling behavior: N sessions cycle at rate R over duration D seconds.
// MaxRounds = ceil(R * D / N): total requests per session
// ThinkTimeUs = floor((N / R) * 1e6): inter-round delay in microseconds
//
// ContextGrowth is intentionally empty (fixed-length inputs per round) because
// real inference-perf sends constant input tokens per request — the chat template
// is applied but context is NOT accumulated across turns (H30 finding).
//
// Note: ThinkTimeUs does not account for the 1µs/token output completion heuristic
// in GenerateReasoningRequests. This is negligible for typical parameterizations
// (e.g., OutputLen=248 adds 248µs to a ThinkTimeUs of 600,000µs = 0.04% error).
func computeReasoningSpec(stageRate float64, stageDurationSec int64, numSessions int) *ReasoningSpec {
	maxRounds := int(math.Ceil(stageRate * float64(stageDurationSec) / float64(numSessions)))
	thinkTimeUs := int64(float64(numSessions) / stageRate * 1e6)
	return &ReasoningSpec{
		ReasonRatioDist: DistSpec{
			Type:   "constant",
			Params: map[string]float64{"value": 0},
		},
		MultiTurn: &MultiTurnSpec{
			MaxRounds:     maxRounds,
			ThinkTimeUs:   thinkTimeUs,
			ContextGrowth: "", // fixed-length: matches real inference-perf behavior
			SingleSession: true,
		},
	}
}

// constantDist creates a DistSpec for a constant (zero-variance) distribution.
func constantDist(value float64) DistSpec {
	return DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": value},
	}
}
