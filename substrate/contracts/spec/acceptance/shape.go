package acceptance

import (
	"fmt"
	"math"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// Realisation is how a run realised its acceptance, which decides how an acceptance length
// that states no per-position rates is spread over the draft positions (design D8,
// inference-sim/blis-substrate#103).
type Realisation string

const (
	// RealisationNatural: the engine verified the drafts for real, so acceptance decays with
	// the position, as every measured curve does.
	RealisationNatural Realisation = "natural"
	// RealisationSynthetic: the engine forced the acceptance length with its synthetic
	// rejection sampler, which spreads it with MinimumVarianceRates.
	RealisationSynthetic Realisation = "synthetic"
)

var realisations = map[Realisation]bool{RealisationNatural: true, RealisationSynthetic: true}

// Valid reports whether r is a recognized realisation.
func (r Realisation) Valid() bool { return realisations[r] }

// AllRealisations returns every realisation, sorted, for error text.
func AllRealisations() []string { return vocab.Enumerate(realisations) }

// Schedule names how a resolution's per-position rates were obtained.
type Schedule string

const (
	// ScheduleStated: the value or profile states its rates.
	ScheduleStated Schedule = "stated"
	// ScheduleMinimumVariance: derived with MinimumVarianceRates, the engine's synthetic
	// schedule.
	ScheduleMinimumVariance Schedule = "minimum_variance"
	// ScheduleNaturalDecay: derived with NaturalRates from a profile set's measured shape.
	ScheduleNaturalDecay Schedule = "natural_decay"
)

// Shape is how natural acceptance spreads over the draft positions, fitted to a set's
// measured curves. Under it the probability that the first i drafts are all accepted is
//
//	p_i = q^(i^DecayExponent)
//
// with q chosen so the rates sum to AL-1. An exponent of 1 is a constant per-token
// acceptance probability (p_i = q^i); a larger one decays faster down the positions.
type Shape struct {
	// SpeculativeMethod is the method whose curves the shape was fitted to, or "" for one
	// fitted to every method's, which a lookup takes when its method has none.
	SpeculativeMethod string `yaml:"speculative_method,omitempty"`
	// DecayExponent is the fitted exponent, in [0.5, 2].
	DecayExponent float64 `yaml:"decay_exponent"`
	// FittedFrom lists the source of every curve the shape was fitted to.
	FittedFrom []string `yaml:"fitted_from"`
	// Rule states how the exponent was fitted.
	Rule string `yaml:"rule"`
}

// DecayExponent bounds: outside them the family no longer looks like any measured curve.
const (
	MinDecayExponent = 0.5
	MaxDecayExponent = 2.0
)

func (s Shape) validate(p *validate.Problems, at string) {
	if p.FiniteField(at+".decay_exponent", s.DecayExponent) &&
		(s.DecayExponent < MinDecayExponent || s.DecayExponent > MaxDecayExponent) {
		p.Field(at+".decay_exponent", "must lie in [%v, %v], got %v",
			MinDecayExponent, MaxDecayExponent, s.DecayExponent)
	}
	if len(s.FittedFrom) == 0 {
		p.Field(at+".fitted_from", "required: the sources of the curves it was fitted to")
	}
	for i, f := range s.FittedFrom {
		if f == "" {
			p.Field(fmt.Sprintf("%s.fitted_from[%d]", at, i), "must not be empty")
		}
	}
	if s.Rule == "" {
		p.Field(at+".rule", "required: how the exponent was fitted")
	}
}

// NaturalRates spreads an acceptance length over k positions with the shape's decay:
// p_i = q^(i^exponent), with q in [0, 1] solved so the rates sum to al-1. The rates are
// non-increasing, so they are a valid schedule; al = 1 gives all zeros and al = k+1 all ones.
func NaturalRates(al float64, k int, exponent float64) ([]float64, error) {
	if err := CheckLength(al, k); err != nil {
		return nil, err
	}
	if math.IsNaN(exponent) || exponent < MinDecayExponent || exponent > MaxDecayExponent {
		return nil, fmt.Errorf("decay exponent %v lies outside [%v, %v]", exponent, MinDecayExponent, MaxDecayExponent)
	}
	rates := func(q float64) []float64 {
		r := make([]float64, k)
		for i := range r {
			r[i] = math.Pow(q, math.Pow(float64(i+1), exponent))
		}
		return r
	}
	want := al - 1
	switch {
	case want <= 0:
		return make([]float64, k), nil
	case want >= float64(k):
		return rates(1), nil
	}
	// The sum rises strictly with q on (0, 1), so bisection finds the one q that gives al.
	lo, hi := 0.0, 1.0
	for range 200 {
		q := (lo + hi) / 2
		if sum(rates(q)) < want {
			lo = q
		} else {
			hi = q
		}
	}
	return rates((lo + hi) / 2), nil
}

// ShapeFor returns the shape a natural value of method takes from sets: a set's shape fitted
// to the method, else a set's pooled shape, searching the sets in order, and the name of the
// set that holds it. It returns nil when no set states one.
func ShapeFor(sets []*ProfileSet, method string) (*Shape, string) {
	for _, want := range []string{method, ""} {
		for _, s := range sets {
			if s == nil {
				continue
			}
			for i := range s.Shapes {
				if s.Shapes[i].SpeculativeMethod == want {
					return &s.Shapes[i], s.Name
				}
			}
		}
	}
	return nil, ""
}

func validateShapes(p *validate.Problems, shapes []Shape) {
	seen := map[string]int{}
	for i, s := range shapes {
		at := fmt.Sprintf("shapes[%d]", i)
		s.validate(p, at)
		if first, dup := seen[s.SpeculativeMethod]; dup {
			p.Field(at+".speculative_method", "repeats the method %q of shapes[%d]", s.SpeculativeMethod, first)
		}
		seen[s.SpeculativeMethod] = i
	}
}
