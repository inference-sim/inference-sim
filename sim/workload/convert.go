package workload

import (
	"fmt"
	"math"
)

// ConvertServeGen converts a ServeGen data directory (containing chunk-*-trace.csv
// and dataset.json files) into a v2 WorkloadSpec with multi-period CohortSpec entries.
// windowDurationSecs controls how long each period runs (default 600s = 10 minutes).
// drainTimeoutSecs controls the gap between periods (default 180s = 3 minutes).
// Returns error if the directory is empty or contains invalid data (R6: never Fatalf).
func ConvertServeGen(path string, windowDurationSecs, drainTimeoutSecs int) (*WorkloadSpec, error) {
	if path == "" {
		return nil, fmt.Errorf("ServeGen path must not be empty")
	}
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 0, // Absolute rate mode (trace_rate per cohort)
		Seed:          42, // Default seed for deterministic RNG (BC-7)
		ServeGenData: &ServeGenDataSpec{
			Path:               path,
			WindowDurationSecs: windowDurationSecs,
			DrainTimeoutSecs:   drainTimeoutSecs,
		},
	}
	if err := loadServeGenData(spec); err != nil {
		return nil, fmt.Errorf("loading ServeGen data from %s: %w", path, err)
	}
	spec.ServeGenData = nil // clear after loading; cohorts are now populated
	return spec, nil
}

// PresetConfig holds the token distribution parameters for a workload preset.
// Exported so cmd/ can pass loaded presets to the synthesis layer.
type PresetConfig struct {
	PrefixTokens      int
	PromptTokensMean  int
	PromptTokensStdev int
	PromptTokensMin   int
	PromptTokensMax   int
	OutputTokensMean  int
	OutputTokensStdev int
	OutputTokensMin   int
	OutputTokensMax   int
}

// ConvertPreset converts a named workload preset (e.g., "chatbot") into a v2 WorkloadSpec.
// rate is in requests/second. Returns error for unknown preset names.
func ConvertPreset(name string, rate float64, numRequests int, preset PresetConfig) (*WorkloadSpec, error) {
	if math.IsNaN(rate) || math.IsInf(rate, 0) || rate <= 0 {
		return nil, fmt.Errorf("rate must be a finite positive number, got %f", rate)
	}
	if numRequests <= 0 {
		return nil, fmt.Errorf("num_requests must be positive, got %d", numRequests)
	}

	spec := &WorkloadSpec{
		Version:       "2",
		Category:      "language",
		AggregateRate: rate,
		NumRequests:   int64(numRequests),
		Clients: []ClientSpec{
			{
				ID:           fmt.Sprintf("preset-%s", name),
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{
					Type: "gaussian",
					Params: map[string]float64{
						"mean":  float64(preset.PromptTokensMean),
						"std_dev": float64(preset.PromptTokensStdev),
						"min":   float64(preset.PromptTokensMin),
						"max":   float64(preset.PromptTokensMax),
					},
				},
				OutputDist: DistSpec{
					Type: "gaussian",
					Params: map[string]float64{
						"mean":  float64(preset.OutputTokensMean),
						"std_dev": float64(preset.OutputTokensStdev),
						"min":   float64(preset.OutputTokensMin),
						"max":   float64(preset.OutputTokensMax),
					},
				},
				PrefixLength: preset.PrefixTokens,
			},
		},
	}

	if preset.PrefixTokens > 0 {
		spec.Clients[0].PrefixGroup = "shared"
	}

	return spec, nil
}

// ConvertInferencePerf converts an inference-perf YAML spec file into a v2 WorkloadSpec.
// Wraps existing ExpandInferencePerfSpec with file loading.
func ConvertInferencePerf(path string) (*WorkloadSpec, error) {
	if path == "" {
		return nil, fmt.Errorf("inference-perf spec path must not be empty")
	}
	spec, err := LoadWorkloadSpec(path)
	if err != nil {
		return nil, fmt.Errorf("loading inference-perf spec from %s: %w", path, err)
	}
	if spec.InferencePerf == nil {
		return nil, fmt.Errorf("file %s does not contain an inference_perf section", path)
	}
	expanded, err := ExpandInferencePerfSpec(spec.InferencePerf, spec.Seed)
	if err != nil {
		return nil, fmt.Errorf("expanding inference-perf spec: %w", err)
	}
	expanded.Version = "2"
	return expanded, nil
}

// ComposeSpecs merges multiple WorkloadSpecs into a single spec.
// Client lists are concatenated, aggregate rates summed, rate fractions renormalized.
// Each client's fraction is weighted by its original spec's rate contribution
// to preserve absolute rates: a 10 req/s spec + 5 req/s spec → 15 req/s total,
// with the first spec's clients receiving 10/15 of the merged rate.
func ComposeSpecs(specs []*WorkloadSpec) (*WorkloadSpec, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("at least one spec file required")
	}

	merged := &WorkloadSpec{
		Version:  "2",
		Category: specs[0].Category,
	}

	var totalRate float64
	for _, s := range specs {
		totalRate += s.AggregateRate
	}
	if math.IsNaN(totalRate) || math.IsInf(totalRate, 0) {
		return nil, fmt.Errorf("compose: total aggregate rate is not finite: %f", totalRate)
	}
	merged.AggregateRate = totalRate

	if totalRate == 0 {
		// All specs are concurrency-only (Validate() ensures any spec with a rate-based
		// client requires AggregateRate > 0, so totalRate==0 means all-concurrency).
		// No rate fraction scaling is needed — just concatenate clients.
		for _, s := range specs {
			merged.Clients = append(merged.Clients, s.Clients...)
		}
		return merged, nil
	}

	if totalRate < 0 {
		return nil, fmt.Errorf("compose: total aggregate rate must be non-negative, got %f", totalRate)
	}

	// Rate-based or mixed: renormalize each client's RateFraction by its
	// spec's proportional share of the combined rate.
	for _, s := range specs {
		weight := s.AggregateRate / totalRate
		for _, c := range s.Clients {
			c.RateFraction *= weight
			merged.Clients = append(merged.Clients, c)
		}
	}

	return merged, nil
}
