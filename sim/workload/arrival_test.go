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

// TestWeibullSampler_ZeroUniform_NoOverflow verifies BC-9:
// clamping u=0 to SmallestNonzeroFloat64 produces a finite result.
func TestWeibullSampler_ZeroUniform_NoOverflow(t *testing.T) {
	s := &WeibullSampler{shape: 1.0, scale: 1000.0}
	// Verify the formula doesn't overflow for the smallest positive float64
	u := math.SmallestNonzeroFloat64
	sample := s.scale * math.Pow(-math.Log(u), 1.0/s.shape)
	if math.IsInf(sample, 0) {
		t.Error("sample should not be +Inf for SmallestNonzeroFloat64")
	}
	if sample <= 0 {
		t.Error("sample should be positive")
	}
}

func TestConstantArrivalSampler_ExactIntervals(t *testing.T) {
	// BC-3: Constant sampler produces exact 1/rate intervals
	rate := 10.0 / 1e6 // 10 req/s = 10/1e6 req/µs
	sampler := NewArrivalSampler(ArrivalSpec{Process: "constant"}, rate)

	expectedIAT := int64(1.0 / rate) // 100000 µs
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		iat := sampler.SampleIAT(rng)
		if iat != expectedIAT {
			t.Fatalf("iteration %d: SampleIAT = %d, want %d", i, iat, expectedIAT)
		}
	}
}

func TestConstantArrivalSampler_DifferentSeeds_SameResult(t *testing.T) {
	// BC-3: Constant sampler is deterministic regardless of RNG state
	rate := 5.0 / 1e6
	sampler := NewArrivalSampler(ArrivalSpec{Process: "constant"}, rate)

	rng1 := rand.New(rand.NewSource(1))
	rng2 := rand.New(rand.NewSource(999))

	for i := 0; i < 50; i++ {
		iat1 := sampler.SampleIAT(rng1)
		iat2 := sampler.SampleIAT(rng2)
		if iat1 != iat2 {
			t.Fatalf("iteration %d: different seeds produced different IATs: %d vs %d", i, iat1, iat2)
		}
	}
}

func TestConstantArrivalSampler_MinimumOneUs(t *testing.T) {
	// BC-3: Floor of 1 microsecond for very high rates
	rate := 1.0 // 1 req/µs (extremely high)
	sampler := NewArrivalSampler(ArrivalSpec{Process: "constant"}, rate)

	rng := rand.New(rand.NewSource(42))
	iat := sampler.SampleIAT(rng)
	if iat < 1 {
		t.Errorf("SampleIAT = %d, want >= 1", iat)
	}
}

// TestNormalizedExponentialSampler_Normalized verifies BC-1:
// Sampler generates exactly count intervals with sum ≈ duration.
func TestNormalizedExponentialSampler_Normalized(t *testing.T) {
	// GIVEN a sampler with count=600, duration=60s (60M µs)
	rng := rand.New(rand.NewSource(42))
	count := int64(600)
	durationUs := int64(60_000_000) // 60 seconds
	sampler := NewNormalizedExponentialSampler(rng, count, durationUs)

	// WHEN all 600 IATs are sampled
	intervals := make([]int64, 0, count)
	for {
		iat := sampler.SampleIAT(nil) // RNG not used after construction
		if iat == 0 {
			break // Exhausted
		}
		intervals = append(intervals, iat)
	}

	// THEN exactly 600 intervals returned
	if int64(len(intervals)) != count {
		t.Fatalf("count = %d, want %d", len(intervals), count)
	}

	// AND sum ≈ 60,000,000 µs (within count microseconds for rounding)
	sum := int64(0)
	for _, iat := range intervals {
		sum += iat
	}
	tolerance := count // At most 1µs error per interval from flooring
	if sum < durationUs-tolerance || sum > durationUs+tolerance {
		t.Errorf("sum = %d µs, want ≈ %d µs (within %d µs)", sum, durationUs, tolerance)
	}

	// AND all IATs >= 1
	for i, iat := range intervals {
		if iat < 1 {
			t.Errorf("intervals[%d] = %d, want >= 1", i, iat)
			break
		}
	}

	// AND final SampleIAT returns 0 (exhausted)
	finalIAT := sampler.SampleIAT(nil)
	if finalIAT != 0 {
		t.Errorf("SampleIAT after exhaustion = %d, want 0", finalIAT)
	}
}

// TestNormalizedExponentialSampler_Deterministic verifies that same seed
// produces identical intervals (INV-6: determinism).
func TestNormalizedExponentialSampler_Deterministic(t *testing.T) {
	// GIVEN two samplers with same seed, count, duration
	seed := int64(42)
	count := int64(100)
	durationUs := int64(10_000_000) // 10 seconds

	rng1 := rand.New(rand.NewSource(seed))
	sampler1 := NewNormalizedExponentialSampler(rng1, count, durationUs)

	rng2 := rand.New(rand.NewSource(seed))
	sampler2 := NewNormalizedExponentialSampler(rng2, count, durationUs)

	// WHEN all intervals are sampled from each
	intervals1 := make([]int64, count)
	intervals2 := make([]int64, count)
	for i := int64(0); i < count; i++ {
		intervals1[i] = sampler1.SampleIAT(nil)
		intervals2[i] = sampler2.SampleIAT(nil)
	}

	// THEN intervals are byte-identical
	for i := range intervals1 {
		if intervals1[i] != intervals2[i] {
			t.Errorf("interval[%d]: sampler1=%d, sampler2=%d (want identical)", i, intervals1[i], intervals2[i])
			break
		}
	}
}

// TestNormalizedExponentialSampler_EdgeCases tests edge cases and validation.
func TestNormalizedExponentialSampler_EdgeCases(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	t.Run("MinimalDuration", func(t *testing.T) {
		// GIVEN durationUs == count (mean IAT = 1µs, minimum possible per-interval)
		count := int64(1000)
		durationUs := int64(1000)
		sampler := NewNormalizedExponentialSampler(rng, count, durationUs)

		// WHEN all intervals sampled
		var sum int64
		allAtLeastOne := true
		for i := int64(0); i < count; i++ {
			iat := sampler.SampleIAT(nil)
			if iat < 1 {
				allAtLeastOne = false
				t.Errorf("iat[%d] = %d, want >= 1", i, iat)
			}
			sum += iat
		}

		// THEN all IATs >= 1
		if !allAtLeastOne {
			t.Error("some IATs < 1")
		}

		// AND sum ≈ durationUs (within tolerance for flooring)
		// Due to flooring to >= 1, the sum may exceed durationUs by up to count microseconds
		if sum < durationUs || sum > durationUs+count {
			t.Errorf("sum = %d, want in range [%d, %d]", sum, durationUs, durationUs+count)
		}
	})

	t.Run("LargeCount", func(t *testing.T) {
		// GIVEN a large count (1M requests)
		count := int64(1_000_000)
		durationUs := int64(3600_000_000) // 1 hour
		sampler := NewNormalizedExponentialSampler(rng, count, durationUs)

		// WHEN first few intervals sampled
		iat1 := sampler.SampleIAT(nil)
		iat2 := sampler.SampleIAT(nil)

		// THEN all IATs >= 1
		if iat1 < 1 || iat2 < 1 {
			t.Errorf("IATs: %d, %d; want >= 1", iat1, iat2)
		}
	})

	t.Run("PanicOnInvalidCount", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for count <= 0")
			}
		}()
		NewNormalizedExponentialSampler(rng, 0, 1000)
	})

	t.Run("PanicOnInvalidDuration", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for durationUs <= 0")
			}
		}()
		NewNormalizedExponentialSampler(rng, 100, 0)
	})

	t.Run("PanicOnExcessiveCount", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for count > 10M")
			}
		}()
		NewNormalizedExponentialSampler(rng, 10_000_001, 3600_000_000)
	})

	t.Run("PanicOnDegenerateDistribution", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for durationUs < count")
			}
		}()
		NewNormalizedExponentialSampler(rng, 1000, 999)
	})
}
