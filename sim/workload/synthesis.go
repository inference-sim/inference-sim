package workload

// DistributionParams holds the legacy CLI parameters for distribution-based
// workload generation. Maps directly to the old GuideLLMConfig fields.
type DistributionParams struct {
	Rate               float64 // req/s (rate mode; mutually exclusive with Concurrency)
	Concurrency        int     // number of concurrent sessions (concurrency mode; mutually exclusive with Rate)
	ThinkTimeMs        int     // inter-round think time in milliseconds (concurrency mode only)
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
	GIEPriority        int // GIE integer priority (0 = default/unset)
}

// SynthesizeFromDistribution creates a v2 WorkloadSpec from legacy distribution
// parameters (the old --workload distribution CLI path).
// The synthesized spec uses constant arrival (matching the old Poisson-like
// fixed-interval behavior) and gaussian token distributions.
//
// Two modes are supported:
//   - Rate mode (Concurrency == 0): sets AggregateRate and RateFraction=1.0
//   - Concurrency mode (Concurrency > 0): sets client Concurrency and ThinkTimeUs;
//     AggregateRate and RateFraction are both zero.
func SynthesizeFromDistribution(params DistributionParams) *WorkloadSpec {
	client := ClientSpec{
		ID:          "distribution",
		GIEPriority: params.GIEPriority,
		Arrival:     ArrivalSpec{Process: "constant"},
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
	}

	var aggregateRate float64
	if params.Concurrency > 0 {
		// Concurrency mode: closed-loop sessions drive arrival.
		client.Concurrency = params.Concurrency
		client.ThinkTimeUs = int64(params.ThinkTimeMs) * 1000
		// RateFraction and AggregateRate remain zero.
	} else {
		// Rate mode: open-loop Poisson/constant arrival.
		client.RateFraction = 1.0
		aggregateRate = params.Rate
	}

	if params.PrefixTokens > 0 {
		client.PrefixGroup = "shared"
		client.PrefixLength = params.PrefixTokens
	}

	return &WorkloadSpec{
		Version:       "2",
		Category:      "language",
		AggregateRate: aggregateRate,
		NumRequests:   int64(params.NumRequests),
		Clients:       []ClientSpec{client},
	}
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
