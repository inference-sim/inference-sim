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
		AggregateRate: 0,  // Absolute rate mode (trace_rate per cohort)
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
						"mean":    float64(preset.PromptTokensMean),
						"std_dev": float64(preset.PromptTokensStdev),
						"min":     float64(preset.PromptTokensMin),
						"max":     float64(preset.PromptTokensMax),
					},
				},
				OutputDist: DistSpec{
					Type: "gaussian",
					Params: map[string]float64{
						"mean":    float64(preset.OutputTokensMean),
						"std_dev": float64(preset.OutputTokensStdev),
						"min":     float64(preset.OutputTokensMin),
						"max":     float64(preset.OutputTokensMax),
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
	serialiseStageRates(expanded, spec.InferencePerf)
	return expanded, nil
}

// serialiseStageRates replaces the in-memory per-stage samplers of a multi-stage expansion
// with rates a spec file can carry (#1915, DEV-7). The expansion gives each stage's clients a
// Poisson sampler at the stage's own per-client rate; a file cannot hold that sampler, and
// summed aggregate_rate with equal fractions would offer every stage the total of all stages.
// Absolute-rate mode (aggregate_rate 0) states each client's rate in its stage window instead.
func serialiseStageRates(w *WorkloadSpec, ip *InferencePerfSpec) {
	if len(ip.Stages) <= 1 {
		return
	}
	sp := ip.SharedPrefix
	perStage := float64(sp.NumUniqueSystemPrompts * sp.NumUsersPerSystemPrompt)
	for i := range w.Clients {
		c := &w.Clients[i]
		if c.CustomSamplerFactory == nil || c.Lifecycle == nil || len(c.Lifecycle.Windows) != 1 {
			continue
		}
		stage := i / int(perStage)
		rate := ip.Stages[stage].Rate / perStage
		c.CustomSamplerFactory = nil
		c.Lifecycle.Windows[0].TraceRate = &rate
	}
	w.AggregateRate = 0
}
