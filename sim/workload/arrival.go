package workload

import (
	"math"
	"math/rand"

	"github.com/sirupsen/logrus"
)

// ArrivalSampler generates inter-arrival times for a client.
type ArrivalSampler interface {
	// SampleIAT returns the next inter-arrival time in microseconds.
	// Always returns a positive value (>= 1).
	SampleIAT(rng *rand.Rand) int64
}

// PoissonSampler generates exponentially-distributed inter-arrival times (CV=1).
type PoissonSampler struct {
	rateMicros float64 // requests per microsecond
}

func (s *PoissonSampler) SampleIAT(rng *rand.Rand) int64 {
	iat := int64(rng.ExpFloat64() / s.rateMicros)
	if iat < 1 {
		return 1
	}
	return iat
}

// GammaSampler generates Gamma-distributed inter-arrival times.
// CV > 1 produces bursty arrivals (ServeGen Finding 1: best fit for M-large).
// Implemented using Marsaglia-Tsang's method for shape >= 1,
// with transformation for shape < 1.
type GammaSampler struct {
	shape float64 // 1/CV² (alpha parameter)
	scale float64 // CV²/rate in microseconds (beta parameter)
}

func (s *GammaSampler) SampleIAT(rng *rand.Rand) int64 {
	sample := gammaRand(rng, s.shape, s.scale)
	iat := int64(sample)
	if iat < 1 {
		return 1
	}
	return iat
}

// gammaRand samples from Gamma(shape, scale) using Marsaglia-Tsang's method.
// For shape >= 1: direct method.
// For shape < 1: Gamma(shape) = Gamma(shape+1) * U^(1/shape).
func gammaRand(rng *rand.Rand, shape, scale float64) float64 {
	if shape < 1.0 {
		// Ahrens-Dieter: Gamma(a) = Gamma(a+1) * U^(1/a)
		u := rng.Float64()
		return gammaRand(rng, shape+1.0, scale) * math.Pow(u, 1.0/shape)
	}

	// Marsaglia-Tsang for shape >= 1
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)

	for {
		var x, v float64
		for {
			x = rng.NormFloat64()
			v = 1.0 + c*x
			if v > 0 {
				break
			}
		}
		v = v * v * v
		u := rng.Float64()

		// Squeeze test
		if u < 1.0-0.0331*(x*x)*(x*x) {
			return d * v * scale
		}
		if math.Log(u) < 0.5*x*x+d*(1.0-v+math.Log(v)) {
			return d * v * scale
		}
	}
}

// WeibullSampler generates Weibull-distributed inter-arrival times.
// ServeGen Finding 1: best fit for M-mid models.
type WeibullSampler struct {
	shape float64 // Weibull k parameter
	scale float64 // Weibull λ parameter (in microseconds)
}

func (s *WeibullSampler) SampleIAT(rng *rand.Rand) int64 {
	// Inverse CDF: scale * (-ln(U))^(1/shape)
	u := rng.Float64()
	if u == 0 {
		u = math.SmallestNonzeroFloat64 // prevent -ln(0) = +Inf
	}
	sample := s.scale * math.Pow(-math.Log(u), 1.0/s.shape)
	iat := int64(sample)
	if iat < 1 {
		return 1
	}
	return iat
}

// NewArrivalSampler creates an ArrivalSampler from a spec and rate.
// ratePerMicrosecond is the client's request rate in requests/microsecond.
func NewArrivalSampler(spec ArrivalSpec, ratePerMicrosecond float64) ArrivalSampler {
	// Defensive floor: avoid division by zero or numerical instability
	if ratePerMicrosecond < 1e-15 {
		ratePerMicrosecond = 1e-15
	}
	switch spec.Process {
	case "poisson":
		return &PoissonSampler{rateMicros: ratePerMicrosecond}

	case "gamma":
		cv := 1.0
		if spec.CV != nil {
			cv = *spec.CV
		}
		if cv <= 0 {
			cv = 1.0
		}
		// shape = 1/CV², scale = mean * CV² = (1/rate) * CV²
		shape := 1.0 / (cv * cv)
		mean := 1.0 / ratePerMicrosecond
		scale := mean * cv * cv
		if shape < 0.01 {
			logrus.Warnf("Gamma shape %.4f (CV=%.1f) is very small; falling back to Poisson", shape, cv)
			return &PoissonSampler{rateMicros: ratePerMicrosecond}
		}
		return &GammaSampler{shape: shape, scale: scale}

	case "weibull":
		cv := 1.0
		if spec.CV != nil {
			cv = *spec.CV
		}
		if cv <= 0 {
			cv = 1.0
		}
		mean := 1.0 / ratePerMicrosecond
		k := weibullShapeFromCV(cv)
		// scale = mean / Γ(1 + 1/k)
		scale := mean / math.Gamma(1.0+1.0/k)
		return &WeibullSampler{shape: k, scale: scale}

	default:
		// Validated before reaching here; defensive fallback
		return &PoissonSampler{rateMicros: ratePerMicrosecond}
	}
}

// weibullShapeFromCV finds Weibull shape parameter k such that
// CV² = Γ(1+2/k)/Γ(1+1/k)² - 1, using bisection.
// Range: k ∈ [0.1, 100], tolerance: |CV_computed - CV_target| < 0.001.
// Max 100 iterations; logs warning if convergence fails.
func weibullShapeFromCV(targetCV float64) float64 {
	lo, hi := 0.1, 100.0
	for i := 0; i < 100; i++ {
		mid := (lo + hi) / 2.0
		cv := weibullCV(mid)
		if math.Abs(cv-targetCV) < 0.001 {
			return mid
		}
		// CV is monotonically decreasing in k
		if cv > targetCV {
			lo = mid
		} else {
			hi = mid
		}
	}
	logrus.Warnf("weibullShapeFromCV: bisection did not converge for CV=%.3f after 100 iterations; using k=%.3f", targetCV, (lo+hi)/2.0)
	return (lo + hi) / 2.0
}

// weibullCV computes the coefficient of variation for Weibull(k).
func weibullCV(k float64) float64 {
	g1 := math.Gamma(1.0 + 1.0/k)
	g2 := math.Gamma(1.0 + 2.0/k)
	return math.Sqrt(g2/(g1*g1) - 1.0)
}
