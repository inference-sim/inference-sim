package workload

import (
	"math"
	"math/rand"
	"sort"
	"strings"
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

func TestConstantSampler_AlwaysReturnsExactValue(t *testing.T) {
	// BC-6: constant distribution always returns exact value
	sampler, err := NewLengthSampler(DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 447},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		got := sampler.Sample(rng)
		if got != 447 {
			t.Fatalf("iteration %d: got %d, want 447", i, got)
		}
	}
}

func TestConstantSampler_ValueOne_ReturnsOne(t *testing.T) {
	// Edge: minimum valid constant
	sampler, err := NewLengthSampler(DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 1},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rng := rand.New(rand.NewSource(99))
	if got := sampler.Sample(rng); got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}

func TestConstantSampler_ZeroValue_ReturnsOne(t *testing.T) {
	// Edge: zero value clamped to minimum of 1
	sampler, err := NewLengthSampler(DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 0},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rng := rand.New(rand.NewSource(99))
	if got := sampler.Sample(rng); got != 1 {
		t.Errorf("got %d, want 1 (clamped)", got)
	}
}

// TestParetoLogNormalSampler_ZeroUniform_NoOverflow verifies BC-9:
// extreme u values produce valid (finite, positive) samples.
func TestParetoLogNormalSampler_ZeroUniform_NoOverflow(t *testing.T) {
	// The Pareto formula xm/u^(1/alpha) can produce +Inf for very small u.
	// The sampler guards against this by returning 1 for Inf/NaN results.
	s := &ParetoLogNormalSampler{alpha: 1.0, xm: 100.0, mu: 0, sigma: 1, mixWeight: 1.0}
	rng := rand.New(rand.NewSource(42))
	// Run many samples; none should panic or return non-positive
	for i := 0; i < 1000; i++ {
		result := s.Sample(rng)
		if result < 1 {
			t.Errorf("sample %d returned %d, want >= 1", i, result)
		}
	}
}

func TestNewLengthSampler_InvalidType_ReturnsError(t *testing.T) {
	_, err := NewLengthSampler(DistSpec{Type: "unknown"})
	if err == nil {
		t.Fatal("expected error for unknown distribution type")
	}
}

// TestNewLengthSampler_MissingRequiredParams_ReturnsError verifies BC-10.
// TestSequenceSampler_ReplayInOrder verifies BC-1:
// values are returned in order on successive calls.
func TestSequenceSampler_ReplayInOrder(t *testing.T) {
	s := &SequenceSampler{values: []int{100, 200, 300}}
	for i, want := range []int{100, 200, 300} {
		got := s.Sample(nil)
		if got != want {
			t.Errorf("call %d: got %d, want %d", i, got, want)
		}
	}
}

// TestSequenceSampler_WrapsOnExhaustion verifies BC-2:
// after exhaustion the sequence restarts from the beginning.
func TestSequenceSampler_WrapsOnExhaustion(t *testing.T) {
	s := &SequenceSampler{values: []int{10, 20}}
	_ = s.Sample(nil) // 10
	_ = s.Sample(nil) // 20
	got := s.Sample(nil)
	if got != 10 {
		t.Errorf("wrap: got %d, want 10", got)
	}
}

// TestSequenceSampler_SingleValue verifies that a single-element sampler
// always returns the same value regardless of call count.
func TestSequenceSampler_SingleValue(t *testing.T) {
	s := &SequenceSampler{values: []int{42}}
	for i := 0; i < 5; i++ {
		got := s.Sample(nil)
		if got != 42 {
			t.Errorf("call %d: got %d, want 42", i, got)
		}
	}
}

// TestSequenceSampler_EmptyValues verifies that an empty SequenceSampler
// returns 1 (minimum token count) without panicking.
func TestSequenceSampler_EmptyValues(t *testing.T) {
	s := &SequenceSampler{}
	got := s.Sample(nil)
	if got != 1 {
		t.Errorf("empty sampler: got %d, want 1", got)
	}
}

func TestNewLengthSampler_MissingRequiredParams_ReturnsError(t *testing.T) {
	tests := []struct {
		name    string
		spec    DistSpec
		wantErr string
	}{
		{
			name:    "gaussian missing mean",
			spec:    DistSpec{Type: "gaussian", Params: map[string]float64{"std_dev": 1, "min": 1, "max": 10}},
			wantErr: "mean",
		},
		{
			name:    "exponential missing mean",
			spec:    DistSpec{Type: "exponential", Params: map[string]float64{}},
			wantErr: "mean",
		},
		{
			name:    "pareto_lognormal missing alpha",
			spec:    DistSpec{Type: "pareto_lognormal", Params: map[string]float64{"xm": 1, "mu": 0, "sigma": 1, "mix_weight": 0.5}},
			wantErr: "alpha",
		},
		{
			name:    "constant missing value",
			spec:    DistSpec{Type: "constant", Params: map[string]float64{}},
			wantErr: "value",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewLengthSampler(tt.spec)
			if err == nil {
				t.Fatal("expected error for missing required param")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error %q should mention %q", err.Error(), tt.wantErr)
			}
		})
	}
}

// --- LognormalThinkTimeSampler tests ---

// TestLognormalThinkTimeSampler_ClampMin verifies that samples below minUs are
// clamped to minUs (very negative NormFloat64 → tiny lognormal value).
func TestLognormalThinkTimeSampler_ClampMin(t *testing.T) {
	const minUs = 3_000_000
	s := NewLognormalThinkTimeSampler(2.0, 0.6, minUs, 30_000_000)

	// Feed a very low NormFloat64 value by using a fixed source that produces -10σ.
	// We do this indirectly: sample enough times that at least one must hit the floor.
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		v := s.Sample(rng)
		if v < minUs {
			t.Errorf("sample %d below minUs %d on iteration %d", v, minUs, i)
		}
	}
}

// TestLognormalThinkTimeSampler_ClampMax verifies that no sample exceeds maxUs.
func TestLognormalThinkTimeSampler_ClampMax(t *testing.T) {
	const maxUs = 30_000_000
	s := NewLognormalThinkTimeSampler(2.0, 0.6, 3_000_000, maxUs)
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < 1000; i++ {
		v := s.Sample(rng)
		if v > maxUs {
			t.Errorf("sample %d above maxUs %d on iteration %d", v, maxUs, i)
		}
	}
}

// TestLognormalThinkTimeSampler_AllSamplesInRange verifies all samples lie in [minUs, maxUs].
func TestLognormalThinkTimeSampler_AllSamplesInRange(t *testing.T) {
	const minUs, maxUs = 3_000_000, 30_000_000
	s := NewLognormalThinkTimeSampler(2.0, 0.6, minUs, maxUs)
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 10_000; i++ {
		v := s.Sample(rng)
		if v < minUs || v > maxUs {
			t.Errorf("sample %d out of range [%d, %d] on iteration %d", v, minUs, maxUs, i)
		}
	}
}

// TestLognormalThinkTimeSampler_MedianNearExpected verifies the median of a large
// sample set is near e^mu seconds (in µs), confirming the lognormal parameterisation.
// With mu=2.0 the median in seconds is e^2 ≈ 7.389 → 7_389_000 µs.
func TestLognormalThinkTimeSampler_MedianNearExpected(t *testing.T) {
	const mu, sigma = 2.0, 0.6
	const minUs, maxUs = 1, 1<<62 // wide clamps so clamping doesn't skew the median
	s := NewLognormalThinkTimeSampler(mu, sigma, minUs, maxUs)
	rng := rand.New(rand.NewSource(1234))

	const n = 50_000
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = float64(s.Sample(rng))
	}
	// Median via sort
	sort.Float64s(samples)
	median := samples[n/2]

	expectedMedianUs := math.Exp(mu) * 1e6 // e^2 * 1_000_000
	tol := expectedMedianUs * 0.05           // 5% tolerance
	if math.Abs(median-expectedMedianUs) > tol {
		t.Errorf("median %.0f µs, want %.0f ± %.0f µs", median, expectedMedianUs, tol)
	}
}

// TestLognormalThinkTimeSampler_Deterministic verifies identical seeds produce
// identical sample sequences (INV-6 determinism).
func TestLognormalThinkTimeSampler_Deterministic(t *testing.T) {
	s1 := NewLognormalThinkTimeSampler(2.0, 0.6, 3_000_000, 30_000_000)
	s2 := NewLognormalThinkTimeSampler(2.0, 0.6, 3_000_000, 30_000_000)
	rng1 := rand.New(rand.NewSource(555))
	rng2 := rand.New(rand.NewSource(555))
	for i := 0; i < 100; i++ {
		v1, v2 := s1.Sample(rng1), s2.Sample(rng2)
		if v1 != v2 {
			t.Errorf("iteration %d: s1=%d, s2=%d — not deterministic", i, v1, v2)
		}
	}
}
