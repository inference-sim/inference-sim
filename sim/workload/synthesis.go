package workload

import (
	"fmt"

	"github.com/sirupsen/logrus"
)

// DistributionParams holds the legacy CLI parameters for distribution-based
// workload generation. Maps directly to the old GuideLLMConfig fields.
type DistributionParams struct {
	Rate               float64 // req/s
	NumRequests        int
	PrefixTokens       int
	PromptTokensMean   int
	PromptTokensStdDev int
	PromptTokensMin    int
	PromptTokensMax    int
	OutputTokensMean   int
	OutputTokensStdDev int
	OutputTokensMin    int
	OutputTokensMax    int
}

// SynthesizeFromDistribution creates a v2 WorkloadSpec from legacy distribution
// parameters (the old --workload distribution CLI path).
// The synthesized spec uses constant arrival (matching the old Poisson-like
// fixed-interval behavior) and gaussian token distributions.
func SynthesizeFromDistribution(params DistributionParams) *WorkloadSpec {
	spec := &WorkloadSpec{
		Version:       "2",
		Category:      "language",
		AggregateRate: params.Rate,
		NumRequests:   int64(params.NumRequests),
		Clients: []ClientSpec{
			{
				ID:           "distribution",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "constant"},
				InputDist: DistSpec{
					Type: "gaussian",
					Params: map[string]float64{
						"mean":    float64(params.PromptTokensMean),
						"std_dev": float64(params.PromptTokensStdDev),
						"min":     float64(params.PromptTokensMin),
						"max":     float64(params.PromptTokensMax),
					},
				},
				OutputDist: DistSpec{
					Type: "gaussian",
					Params: map[string]float64{
						"mean":    float64(params.OutputTokensMean),
						"std_dev": float64(params.OutputTokensStdDev),
						"min":     float64(params.OutputTokensMin),
						"max":     float64(params.OutputTokensMax),
					},
				},
			},
		},
	}

	if params.PrefixTokens > 0 {
		spec.Clients[0].PrefixGroup = "shared"
		spec.Clients[0].PrefixLength = params.PrefixTokens
	}

	return spec
}

// SynthesizeFromPreset creates a v2 WorkloadSpec from a named preset configuration.
// This is equivalent to SynthesizeFromDistribution with parameters from defaults.yaml.
func SynthesizeFromPreset(presetName string, preset PresetConfig, rate float64, numRequests int) *WorkloadSpec {
	return SynthesizeFromDistribution(DistributionParams{
		Rate:               rate,
		NumRequests:        numRequests,
		PrefixTokens:       preset.PrefixTokens,
		PromptTokensMean:   preset.PromptTokensMean,
		PromptTokensStdDev: preset.PromptTokensStdev,
		PromptTokensMin:    preset.PromptTokensMin,
		PromptTokensMax:    preset.PromptTokensMax,
		OutputTokensMean:   preset.OutputTokensMean,
		OutputTokensStdDev: preset.OutputTokensStdev,
		OutputTokensMin:    preset.OutputTokensMin,
		OutputTokensMax:    preset.OutputTokensMax,
	})
}

// SynthesizeFromCSVTrace creates a v2 WorkloadSpec from a CSV trace file path.
// WARNING: this is a lossy conversion — per-request arrival times and token
// lengths are replaced by aggregate statistics (mean lengths, constant rate).
// For faithful trace replay preserving per-request fidelity, use --workload-spec
// with a trace v2 YAML file (LoadTraceV2 + ReplayTraceV2Requests).
func SynthesizeFromCSVTrace(path string, horizon int64) (*WorkloadSpec, error) {
	logrus.Warn("--workload traces uses lossy CSV conversion (averaged token lengths, constant arrival). " +
		"For faithful trace replay, use --workload-spec with a trace v2 YAML file instead.")
	spec, err := ConvertCSVTrace(path, horizon)
	if err != nil {
		return nil, fmt.Errorf("synthesizing from CSV trace: %w", err)
	}
	return spec, nil
}
