package workload

import (
	"math"
	"math/rand"
	"testing"
)

func TestGaussianSampler_MeanMatchesParam(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	s, err := NewLengthSampler(DistSpec{
		Type:   "gaussian",
		Params: map[string]float64{"mean": 512, "std_dev": 128, "min": 10, "max": 4096},
	})
	if err != nil {
		t.Fatal(err)
	}
	n := 10000
	sum := 0
	for i := 0; i < n; i++ {
		sum += s.Sample(rng)
	}
	mean := float64(sum) / float64(n)
	if math.Abs(mean-512)/512 > 0.05 {
		t.Errorf("gaussian mean = %.1f, want ≈ 512 (within 5%%)", mean)
	}
}

func TestGaussianSampler_ClampedToRange(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	s, err := NewLengthSampler(DistSpec{
		Type:   "gaussian",
		Params: map[string]float64{"mean": 512, "std_dev": 1000, "min": 100, "max": 900},
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10000; i++ {
		v := s.Sample(rng)
		if v < 100 || v > 900 {
			t.Errorf("sample %d: %d outside [100, 900]", i, v)
			break
		}
	}
}

func TestExponentialSampler_MeanMatchesParam(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	s, err := NewLengthSampler(DistSpec{
		Type:   "exponential",
		Params: map[string]float64{"mean": 256},
	})
	if err != nil {
		t.Fatal(err)
	}
	n := 10000
	sum := 0
	for i := 0; i < n; i++ {
		sum += s.Sample(rng)
	}
	mean := float64(sum) / float64(n)
	if math.Abs(mean-256)/256 > 0.05 {
		t.Errorf("exponential mean = %.1f, want ≈ 256 (within 5%%)", mean)
	}
}

func TestExponentialSampler_AlwaysPositive(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	s, err := NewLengthSampler(DistSpec{
		Type:   "exponential",
		Params: map[string]float64{"mean": 10},
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10000; i++ {
		if v := s.Sample(rng); v < 1 {
			t.Errorf("sample %d: got %d, want >= 1", i, v)
			break
		}
	}
}

func TestParetoLogNormalSampler_ProducesPositiveValues(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	s, err := NewLengthSampler(DistSpec{
		Type: "pareto_lognormal",
		Params: map[string]float64{
			"alpha": 1.5, "xm": 50, "mu": 5.5, "sigma": 1.2, "mix_weight": 0.3,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 1000; i++ {
		v := s.Sample(rng)
		if v < 1 {
			t.Errorf("sample %d: got %d, want >= 1", i, v)
			break
		}
	}
}

func TestParetoLogNormalSampler_MixWeightChangesDistribution(t *testing.T) {
	// GIVEN two samplers with different mix_weights but same RNG seed
	rng1 := rand.New(rand.NewSource(42))
	s1, _ := NewLengthSampler(DistSpec{
		Type: "pareto_lognormal",
		Params: map[string]float64{
			"alpha": 1.5, "xm": 50, "mu": 5.0, "sigma": 0.5, "mix_weight": 0.9,
		},
	})
	rng2 := rand.New(rand.NewSource(42))
	s2, _ := NewLengthSampler(DistSpec{
		Type: "pareto_lognormal",
		Params: map[string]float64{
			"alpha": 1.5, "xm": 50, "mu": 5.0, "sigma": 0.5, "mix_weight": 0.1,
		},
	})
	// WHEN samples are drawn
	n := 10000
	sum1, sum2 := 0, 0
	for i := 0; i < n; i++ {
		sum1 += s1.Sample(rng1)
		sum2 += s2.Sample(rng2)
	}
	// THEN different mix weights produce different means (behavioral: distribution changes)
	mean1 := float64(sum1) / float64(n)
	mean2 := float64(sum2) / float64(n)
	if mean1 == mean2 {
		t.Errorf("different mix weights should produce different means, both = %.0f", mean1)
	}
}

func TestEmpiricalPDFSampler_ReproducesDistribution(t *testing.T) {
	// GIVEN a simple empirical PDF: {10: 0.5, 20: 0.5}
	rng := rand.New(rand.NewSource(42))
	pdf := map[int]float64{10: 0.5, 20: 0.5}
	s := NewEmpiricalPDFSampler(pdf)

	// WHEN 10000 samples drawn
	n := 10000
	counts := make(map[int]int)
	for i := 0; i < n; i++ {
		v := s.Sample(rng)
		counts[v]++
	}

	// THEN each value appears ~50% of the time (within 5%)
	frac10 := float64(counts[10]) / float64(n)
	if math.Abs(frac10-0.5) > 0.05 {
		t.Errorf("P(10) = %.3f, want ≈ 0.5", frac10)
	}
}

func TestEmpiricalPDFSampler_SingleBin_AlwaysReturnsThatValue(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	pdf := map[int]float64{42: 1.0}
	s := NewEmpiricalPDFSampler(pdf)
	for i := 0; i < 100; i++ {
		if v := s.Sample(rng); v != 42 {
			t.Errorf("sample %d: got %d, want 42", i, v)
		}
	}
}

func TestEmpiricalPDFSampler_NonNormalized_NormalizesAutomatically(t *testing.T) {
	// GIVEN probabilities that sum to 2.0 (not 1.0)
	rng := rand.New(rand.NewSource(42))
	pdf := map[int]float64{10: 1.0, 20: 1.0}
	s := NewEmpiricalPDFSampler(pdf)
	counts := make(map[int]int)
	n := 10000
	for i := 0; i < n; i++ {
		counts[s.Sample(rng)]++
	}
	frac := float64(counts[10]) / float64(n)
	if frac < 0.45 || frac > 0.55 {
		t.Errorf("P(10) = %.3f, want ≈ 0.5 (non-normalized input should auto-normalize)", frac)
	}
}

func TestNewLengthSampler_EmptyEmpiricalPDF_ReturnsError(t *testing.T) {
	_, err := NewLengthSampler(DistSpec{Type: "empirical"})
	if err == nil {
		t.Fatal("expected error for empty empirical PDF")
	}
}

func TestNewLengthSampler_InvalidType_ReturnsError(t *testing.T) {
	_, err := NewLengthSampler(DistSpec{Type: "unknown"})
	if err == nil {
		t.Fatal("expected error for unknown distribution type")
	}
}
