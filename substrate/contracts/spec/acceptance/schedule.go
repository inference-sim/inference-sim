package acceptance

import (
	"fmt"
	"math"
)

// RatesSumTolerance is how far stated per-position rates may sum from AL-1. The engine logs AL
// to two decimals and each rate to three, so a faithful transcription at K=8 can be off by
// up to 0.005 + 8*0.0005 = 0.009.
const RatesSumTolerance = 0.01

// MinimumVarianceRates derives per-position rates from an acceptance length the way the engine
// the engine does for synthetic rejection sampling (SpeculativeConfig.
// _acceptance_length_to_rates, config/speculative.py:557-564):
//
//	rates = [1.0]*floor(AL-1) + [frac(AL-1)] + [0.0]*rest, truncated to K
//
// It is how the engine, and InferenceX's forced agentic runs, realise a published AL, so a
// simulator using it reproduces those runs. Natural acceptance spreads more than this
// schedule, so per-request tails derived from it are optimistic.
//
// It refuses a K below 1 and an AL outside [1, K+1], as the engine's
// _resolve_synthetic_acceptance_rates does.
func MinimumVarianceRates(al float64, k int) ([]float64, error) {
	if err := CheckLength(al, k); err != nil {
		return nil, err
	}
	drafts := al - 1
	full := int(drafts)
	rates := make([]float64, k)
	for i := range rates {
		switch {
		case i < full:
			rates[i] = 1
		case i == full:
			rates[i] = drafts - float64(full)
		}
	}
	return rates, nil
}

// CheckLength reports whether al is a finite acceptance length in [1, k+1] for k >= 1.
func CheckLength(al float64, k int) error {
	if k < 1 {
		return fmt.Errorf("num_spec_tokens must be at least 1, got %d", k)
	}
	if math.IsNaN(al) || math.IsInf(al, 0) {
		return fmt.Errorf("acceptance length must be finite, got %v", al)
	}
	if al < 1 || al > float64(k+1) {
		return fmt.Errorf("acceptance length %v lies outside [1, %d] for num_spec_tokens %d",
			al, k+1, k)
	}
	return nil
}

// CheckRates reports whether rates are valid per-position rates for k: k entries, each
// finite and in [0, 1], non-increasing. The engine's _resolve_synthetic_acceptance_rates makes
// the same checks.
func CheckRates(rates []float64, k int) error {
	if len(rates) != k {
		return fmt.Errorf("per_position_rates has %d entries; num_spec_tokens %d needs %d",
			len(rates), k, k)
	}
	for i, r := range rates {
		if math.IsNaN(r) || math.IsInf(r, 0) || r < 0 || r > 1 {
			return fmt.Errorf("per_position_rates[%d] = %v lies outside [0, 1]", i, r)
		}
		if i > 0 && r > rates[i-1] {
			return fmt.Errorf("per_position_rates[%d] = %v exceeds per_position_rates[%d] = %v; "+
				"per_position_rates[i] is the probability the first i+1 drafts are all accepted, so it cannot rise",
				i, r, i-1, rates[i-1])
		}
	}
	return nil
}

// RateFromLength is the acceptance rate, the accepted fraction of drafts: (AL-1)/K.
func RateFromLength(al float64, k int) float64 { return (al - 1) / float64(k) }

// LengthFromRate is the acceptance length an acceptance rate states at K: 1 + rate*K.
func LengthFromRate(rate float64, k int) float64 { return 1 + rate*float64(k) }

func sum(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s
}
