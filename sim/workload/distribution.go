package workload

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// LengthSampler generates token count samples.
type LengthSampler interface {
	// Sample returns a positive token count (>= 1).
	Sample(rng *rand.Rand) int
}

// GaussianSampler produces clamped Gaussian token lengths.
type GaussianSampler struct {
	mean, stdDev float64
	min, max     int
}

func (s *GaussianSampler) Sample(rng *rand.Rand) int {
	if s.min == s.max {
		return s.min
	}
	val := rng.NormFloat64()*s.stdDev + s.mean
	clamped := math.Min(float64(s.max), math.Max(float64(s.min), val))
	result := int(math.Round(clamped))
	if result < 1 {
		return 1
	}
	return result
}

// ExponentialSampler produces exponentially-distributed token lengths.
type ExponentialSampler struct {
	mean float64
}

func (s *ExponentialSampler) Sample(rng *rand.Rand) int {
	val := rng.ExpFloat64() * s.mean
	result := int(math.Round(val))
	if result < 1 {
		return 1
	}
	return result
}

// ParetoLogNormalSampler is a mixture of Pareto and LogNormal distributions.
// With probability mixWeight, draw from Pareto(alpha, xm); otherwise LogNormal(mu, sigma).
type ParetoLogNormalSampler struct {
	alpha     float64 // Pareto shape
	xm        float64 // Pareto scale (minimum)
	mu        float64 // LogNormal mean of ln(X)
	sigma     float64 // LogNormal std dev of ln(X)
	mixWeight float64 // Probability of drawing from Pareto
}

func (s *ParetoLogNormalSampler) Sample(rng *rand.Rand) int {
	var val float64
	if rng.Float64() < s.mixWeight {
		// Pareto: X = xm / U^(1/alpha)
		u := rng.Float64()
		val = s.xm / math.Pow(u, 1.0/s.alpha)
	} else {
		// LogNormal: X = exp(mu + sigma * Z)
		z := rng.NormFloat64()
		val = math.Exp(s.mu + s.sigma*z)
	}
	result := int(math.Round(val))
	if result < 1 {
		return 1
	}
	return result
}

// EmpiricalPDFSampler samples from an empirical probability distribution
// using inverse CDF via binary search. Primary mode for ServeGen-faithful generation.
type EmpiricalPDFSampler struct {
	values []int     // Sorted token count values
	cdf    []float64 // Cumulative probabilities (same length as values)
}

// NewEmpiricalPDFSampler creates a sampler from a PDF map (token_count → probability).
// Automatically normalizes probabilities if they don't sum to 1.0.
func NewEmpiricalPDFSampler(pdf map[int]float64) *EmpiricalPDFSampler {
	// Sort keys
	keys := make([]int, 0, len(pdf))
	for k := range pdf {
		keys = append(keys, k)
	}
	sort.Ints(keys)

	// Compute CDF with normalization
	totalProb := 0.0
	for _, k := range keys {
		totalProb += pdf[k]
	}

	values := make([]int, 0, len(keys))
	cdf := make([]float64, 0, len(keys))
	cumulative := 0.0
	for _, k := range keys {
		p := pdf[k]
		if p <= 0 {
			continue // skip zero or negative probabilities
		}
		cumulative += p / totalProb
		values = append(values, k)
		cdf = append(cdf, cumulative)
	}
	// Ensure last CDF entry is exactly 1.0
	if len(cdf) > 0 {
		cdf[len(cdf)-1] = 1.0
	}

	return &EmpiricalPDFSampler{values: values, cdf: cdf}
}

func (s *EmpiricalPDFSampler) Sample(rng *rand.Rand) int {
	if len(s.values) == 0 {
		return 1
	}
	if len(s.values) == 1 {
		return s.values[0]
	}
	u := rng.Float64()
	idx := sort.SearchFloat64s(s.cdf, u)
	if idx >= len(s.values) {
		idx = len(s.values) - 1
	}
	return s.values[idx]
}

// NewLengthSampler creates a LengthSampler from a DistSpec.
func NewLengthSampler(spec DistSpec) (LengthSampler, error) {
	switch spec.Type {
	case "gaussian":
		return &GaussianSampler{
			mean:   spec.Params["mean"],
			stdDev: spec.Params["std_dev"],
			min:    int(spec.Params["min"]),
			max:    int(spec.Params["max"]),
		}, nil

	case "exponential":
		return &ExponentialSampler{
			mean: spec.Params["mean"],
		}, nil

	case "pareto_lognormal":
		return &ParetoLogNormalSampler{
			alpha:     spec.Params["alpha"],
			xm:        spec.Params["xm"],
			mu:        spec.Params["mu"],
			sigma:     spec.Params["sigma"],
			mixWeight: spec.Params["mix_weight"],
		}, nil

	case "empirical":
		if spec.File == "" && len(spec.Params) == 0 {
			return nil, fmt.Errorf("empirical distribution requires a file path or inline params")
		}
		// Inline params used as PDF (token_count → probability)
		pdf := make(map[int]float64, len(spec.Params))
		for k, v := range spec.Params {
			// Parse string key as int
			var tokenCount int
			if _, err := fmt.Sscanf(k, "%d", &tokenCount); err != nil {
				return nil, fmt.Errorf("empirical PDF key %q is not an integer: %w", k, err)
			}
			pdf[tokenCount] = v
		}
		if len(pdf) == 0 {
			return nil, fmt.Errorf("empirical distribution has no valid bins")
		}
		return NewEmpiricalPDFSampler(pdf), nil

	default:
		return nil, fmt.Errorf("unknown distribution type %q", spec.Type)
	}
}
