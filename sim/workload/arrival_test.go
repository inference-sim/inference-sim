package workload

import (
	"math"
	"math/rand"
	"testing"
)

func TestPoissonSampler_MeanIAT_MatchesRate(t *testing.T) {
	// GIVEN a Poisson sampler at 10 req/sec (0.00001 req/µs)
	rng := rand.New(rand.NewSource(42))
	sampler := NewArrivalSampler(ArrivalSpec{Process: "poisson"}, 10.0/1e6)

	// WHEN 10000 IATs are sampled
	n := 10000
	sum := int64(0)
	for i := 0; i < n; i++ {
		sum += sampler.SampleIAT(rng)
	}
	meanIAT := float64(sum) / float64(n)

	// THEN mean IAT ≈ 1/rate = 100000 µs (within 5%)
	expected := 1e6 / 10.0
	if math.Abs(meanIAT-expected)/expected > 0.05 {
		t.Errorf("mean IAT = %.0f µs, want ≈ %.0f µs (within 5%%)", meanIAT, expected)
	}
}

func TestGammaSampler_HighCV_ProducesBurstierArrivals(t *testing.T) {
	// GIVEN a Gamma sampler with CV=3.5 and a Poisson sampler at same rate
	rng1 := rand.New(rand.NewSource(42))
	rng2 := rand.New(rand.NewSource(42))
	cv := 3.5
	rate := 10.0 / 1e6 // 10 req/sec
	gamma := NewArrivalSampler(ArrivalSpec{Process: "gamma", CV: &cv}, rate)
	poisson := NewArrivalSampler(ArrivalSpec{Process: "poisson"}, rate)

	// WHEN 10000 IATs sampled from each
	n := 10000
	gammaIATs := make([]float64, n)
	poissonIATs := make([]float64, n)
	for i := 0; i < n; i++ {
		gammaIATs[i] = float64(gamma.SampleIAT(rng1))
		poissonIATs[i] = float64(poisson.SampleIAT(rng2))
	}

	// THEN Gamma CV > 2.0 and Poisson CV ≈ 1.0
	gammaCV := coefficientOfVariation(gammaIATs)
	poissonCV := coefficientOfVariation(poissonIATs)
	if gammaCV < 2.0 {
		t.Errorf("gamma CV = %.2f, want > 2.0", gammaCV)
	}
	if poissonCV < 0.8 || poissonCV > 1.2 {
		t.Errorf("poisson CV = %.2f, want ≈ 1.0", poissonCV)
	}
}

func TestGammaSampler_MeanAndVariance_MatchTheoretical(t *testing.T) {
	// Tighter test: verify both mean and variance
	rng := rand.New(rand.NewSource(42))
	cv := 2.0
	rate := 10.0 / 1e6 // 10 req/sec
	sampler := NewArrivalSampler(ArrivalSpec{Process: "gamma", CV: &cv}, rate)

	n := 50000
	vals := make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = float64(sampler.SampleIAT(rng))
	}
	// Theoretical: mean = 1/rate = 100000 µs, variance = mean² * CV² = 100000² * 4
	mean, variance := meanAndVariance(vals)
	expectedMean := 1e6 / 10.0
	expectedVar := expectedMean * expectedMean * cv * cv
	if math.Abs(mean-expectedMean)/expectedMean > 0.05 {
		t.Errorf("gamma mean = %.0f, want ≈ %.0f (within 5%%)", mean, expectedMean)
	}
	if math.Abs(variance-expectedVar)/expectedVar > 0.15 {
		t.Errorf("gamma variance = %.0f, want ≈ %.0f (within 15%%)", variance, expectedVar)
	}
}

func TestWeibullSampler_MeanIAT_MatchesRate(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cv := 1.5
	rate := 10.0 / 1e6
	sampler := NewArrivalSampler(ArrivalSpec{Process: "weibull", CV: &cv}, rate)

	n := 10000
	sum := int64(0)
	for i := 0; i < n; i++ {
		sum += sampler.SampleIAT(rng)
	}
	meanIAT := float64(sum) / float64(n)
	expected := 1e6 / 10.0
	// Weibull mean should match target within 10%
	if math.Abs(meanIAT-expected)/expected > 0.10 {
		t.Errorf("weibull mean IAT = %.0f µs, want ≈ %.0f µs (within 10%%)", meanIAT, expected)
	}
}

func TestPoissonSampler_AllPositive(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	sampler := NewArrivalSampler(ArrivalSpec{Process: "poisson"}, 10.0/1e6)
	for i := 0; i < 10000; i++ {
		if iat := sampler.SampleIAT(rng); iat <= 0 {
			t.Fatalf("IAT must be positive, got %d at iteration %d", iat, i)
		}
	}
}

// coefficientOfVariation computes std_dev / mean.
func coefficientOfVariation(vals []float64) float64 {
	mean, variance := meanAndVariance(vals)
	return math.Sqrt(variance) / mean
}

func meanAndVariance(vals []float64) (float64, float64) {
	n := float64(len(vals))
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	mean := sum / n
	sumSq := 0.0
	for _, v := range vals {
		d := v - mean
		sumSq += d * d
	}
	return mean, sumSq / n
}
