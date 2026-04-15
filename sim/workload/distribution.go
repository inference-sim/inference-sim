package workload

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
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
		if u == 0 {
			u = math.SmallestNonzeroFloat64 // prevent division by zero → +Inf
		}
		val = s.xm / math.Pow(u, 1.0/s.alpha)
	} else {
		// LogNormal: X = exp(mu + sigma * Z)
		z := rng.NormFloat64()
		val = math.Exp(s.mu + s.sigma*z)
	}
	// Guard against +Inf from extreme u or sigma values
	if math.IsInf(val, 0) || math.IsNaN(val) {
		return 1
	}
	result := int(math.Round(val))
	if result < 1 {
		return 1
	}
	return result
}

// LognormalSampler produces lognormally-distributed samples.
// X = exp(mu + sigma*Z), where Z ~ N(0,1).
// mu and sigma are the mean and std dev of ln(X) — the caller selects the
// natural unit (tokens for length distributions; µs for think-time distributions,
// with mu adjusted by ln(1e6) to shift from seconds to µs).
// Optional min/max clamp (0 = no bound). Output is always >= 1.
type LognormalSampler struct {
	mu, sigma float64
	min, max  int // 0 = no bound; applied after rounding
}

func (s *LognormalSampler) Sample(rng *rand.Rand) int {
	z := rng.NormFloat64()
	val := math.Exp(s.mu + s.sigma*z)
	if math.IsInf(val, 0) || math.IsNaN(val) {
		if s.min > 0 {
			return s.min
		}
		return 1
	}
	result := int(math.Round(val))
	if s.min > 0 && result < s.min {
		result = s.min
	}
	if s.max > 0 && result > s.max {
		result = s.max
	}
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

// ConstantSampler always returns the same fixed value.
// Used for inference-perf fixed-length token specifications (zero variance).
type ConstantSampler struct {
	value int
}

func (s *ConstantSampler) Sample(_ *rand.Rand) int {
	if s.value < 1 {
		return 1
	}
	return s.value
}

// SequenceSampler replays a pre-recorded sequence of values in order.
// Used for trace replay where token counts are known per-round.
// Wraps to the beginning when the sequence is exhausted.
// Safe for empty sequences: returns 1 (minimum token count).
type SequenceSampler struct {
	values []int
	index  int
}

func (s *SequenceSampler) Sample(_ *rand.Rand) int {
	if len(s.values) == 0 {
		return 1 // defensive: empty sequence returns minimum token count
	}
	v := s.values[s.index%len(s.values)]
	s.index++
	return v
}

// requireParam checks that all required keys exist in a params map.
func requireParam(params map[string]float64, keys ...string) error {
	for _, k := range keys {
		if _, ok := params[k]; !ok {
			return fmt.Errorf("distribution requires parameter %q", k)
		}
	}
	return nil
}

// NewLengthSampler creates a LengthSampler from a DistSpec.
func NewLengthSampler(spec DistSpec) (LengthSampler, error) {
	switch spec.Type {
	case "gaussian":
		if err := requireParam(spec.Params, "mean", "std_dev", "min", "max"); err != nil {
			return nil, err
		}
		return &GaussianSampler{
			mean:   spec.Params["mean"],
			stdDev: spec.Params["std_dev"],
			min:    int(spec.Params["min"]),
			max:    int(spec.Params["max"]),
		}, nil

	case "exponential":
		if err := requireParam(spec.Params, "mean"); err != nil {
			return nil, err
		}
		return &ExponentialSampler{
			mean: spec.Params["mean"],
		}, nil

	case "pareto_lognormal":
		if err := requireParam(spec.Params, "alpha", "xm", "mu", "sigma", "mix_weight"); err != nil {
			return nil, err
		}
		return &ParetoLogNormalSampler{
			alpha:     spec.Params["alpha"],
			xm:        spec.Params["xm"],
			mu:        spec.Params["mu"],
			sigma:     spec.Params["sigma"],
			mixWeight: spec.Params["mix_weight"],
		}, nil

	case "lognormal":
		if err := requireParam(spec.Params, "mu", "sigma"); err != nil {
			return nil, err
		}
		s := &LognormalSampler{
			mu:    spec.Params["mu"],
			sigma: spec.Params["sigma"],
		}
		if v, ok := spec.Params["min"]; ok {
			s.min = int(v)
		}
		if v, ok := spec.Params["max"]; ok {
			s.max = int(v)
		}
		return s, nil

	case "constant":
		if err := requireParam(spec.Params, "value"); err != nil {
			return nil, err
		}
		val := int(spec.Params["value"])
		return &ConstantSampler{value: val}, nil

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

// ParseThinkTimeDist parses a think-time distribution spec string into a LengthSampler
// that produces values in microseconds.
//
// Supported formats:
//
//	lognormal:mu=2.0,sigma=0.6
//	lognormal:mu=2.0,sigma=0.6,min=3s,max=30s
//	constant:value=500ms
//
// For lognormal, mu and sigma parameterize ln(X) where X is in seconds.
// The sampler internally adjusts mu by ln(1e6) so output is in microseconds.
// min/max accept time suffixes s, ms, us; bare numbers are treated as milliseconds.
// For constant, value accepts the same time suffixes.
func ParseThinkTimeDist(spec string) (LengthSampler, error) {
	colon := strings.IndexByte(spec, ':')
	if colon < 0 {
		return nil, fmt.Errorf("think-time-dist must have format type:key=value,...; got %q", spec)
	}
	distType := spec[:colon]
	params, err := parseKVParams(spec[colon+1:])
	if err != nil {
		return nil, fmt.Errorf("think-time-dist %q: %w", distType, err)
	}

	switch distType {
	case "lognormal":
		muStr, ok := params["mu"]
		if !ok {
			return nil, fmt.Errorf("think-time-dist lognormal requires parameter \"mu\"")
		}
		sigmaStr, ok := params["sigma"]
		if !ok {
			return nil, fmt.Errorf("think-time-dist lognormal requires parameter \"sigma\"")
		}
		mu, err := strconv.ParseFloat(muStr, 64)
		if err != nil {
			return nil, fmt.Errorf("think-time-dist lognormal: invalid mu %q: %w", muStr, err)
		}
		sigma, err := strconv.ParseFloat(sigmaStr, 64)
		if err != nil {
			return nil, fmt.Errorf("think-time-dist lognormal: invalid sigma %q: %w", sigmaStr, err)
		}
		// mu/sigma are in log-space of seconds; shift to µs: mu_us = mu_s + ln(1e6)
		muUs := mu + math.Log(1e6)
		s := &LognormalSampler{mu: muUs, sigma: sigma}
		if v, ok := params["min"]; ok {
			minUs, err := parseTimeToMicros(v)
			if err != nil {
				return nil, fmt.Errorf("think-time-dist lognormal: invalid min %q: %w", v, err)
			}
			s.min = int(minUs)
		}
		if v, ok := params["max"]; ok {
			maxUs, err := parseTimeToMicros(v)
			if err != nil {
				return nil, fmt.Errorf("think-time-dist lognormal: invalid max %q: %w", v, err)
			}
			s.max = int(maxUs)
		}
		return s, nil

	case "constant":
		valStr, ok := params["value"]
		if !ok {
			return nil, fmt.Errorf("think-time-dist constant requires parameter \"value\"")
		}
		valueUs, err := parseTimeToMicros(valStr)
		if err != nil {
			return nil, fmt.Errorf("think-time-dist constant: invalid value %q: %w", valStr, err)
		}
		return &ConstantSampler{value: int(valueUs)}, nil

	default:
		return nil, fmt.Errorf("unsupported think-time-dist type %q; supported: lognormal, constant", distType)
	}
}

// parseKVParams splits "k=v,k=v,..." into a map. Empty string returns empty map.
func parseKVParams(s string) (map[string]string, error) {
	result := make(map[string]string)
	if strings.TrimSpace(s) == "" {
		return result, nil
	}
	for _, kv := range strings.Split(s, ",") {
		kv = strings.TrimSpace(kv)
		if kv == "" {
			continue
		}
		eq := strings.IndexByte(kv, '=')
		if eq < 0 {
			return nil, fmt.Errorf("invalid parameter %q: expected key=value", kv)
		}
		result[kv[:eq]] = kv[eq+1:]
	}
	return result, nil
}

// parseTimeToMicros converts a time string to microseconds.
// Supported suffixes: s (seconds), ms (milliseconds), us (microseconds).
// Bare numbers (no suffix) are treated as milliseconds.
func parseTimeToMicros(s string) (int64, error) {
	switch {
	case strings.HasSuffix(s, "us"):
		v, err := strconv.ParseFloat(strings.TrimSuffix(s, "us"), 64)
		if err != nil {
			return 0, err
		}
		return int64(v), nil
	case strings.HasSuffix(s, "ms"):
		v, err := strconv.ParseFloat(strings.TrimSuffix(s, "ms"), 64)
		if err != nil {
			return 0, err
		}
		return int64(v * 1_000), nil
	case strings.HasSuffix(s, "s"):
		v, err := strconv.ParseFloat(strings.TrimSuffix(s, "s"), 64)
		if err != nil {
			return 0, err
		}
		return int64(v * 1_000_000), nil
	default:
		// bare number → milliseconds (consistent with --think-time-ms convention)
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return 0, err
		}
		return int64(v * 1_000), nil
	}
}
