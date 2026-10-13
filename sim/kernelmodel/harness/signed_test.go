package harness

import (
	"math"
	"testing"
)

// A model that over- and under-predicts in equal measure has a large absolute error but
// near-zero bias. This is the distinction the signed report exists to make: MAPE2 alone
// cannot tell this case apart from a model that is uniformly 10% high, and the two call for
// different fixes.
func TestSymmetricScatterHasNoBiasButRealMagnitude(t *testing.T) {
	scatter := []float64{+10, -10, +10, -10}
	uniform := []float64{+10, +10, +10, +10}

	s, u := Signed(scatter), Signed(uniform)

	if math.Abs(s.MeanAbs-u.MeanAbs) > 1e-9 {
		t.Fatalf("magnitudes should be indistinguishable: scatter %.4f, uniform %.4f",
			s.MeanAbs, u.MeanAbs)
	}
	if math.Abs(s.Mean) > 1e-9 {
		t.Errorf("symmetric scatter should show no bias, got mean %+.4f", s.Mean)
	}
	if math.Abs(u.Mean-10) > 1e-9 {
		t.Errorf("uniform over-prediction should show its bias, got mean %+.4f", u.Mean)
	}
	if math.Abs(s.FractionOver-0.5) > 1e-9 {
		t.Errorf("symmetric scatter should be half over, got %.4f", s.FractionOver)
	}
	if math.Abs(u.FractionOver-1.0) > 1e-9 {
		t.Errorf("uniform over-prediction should be all over, got %.4f", u.FractionOver)
	}
}

// Under-prediction must be reported as negative. The concurrency diagnosis turns on this:
// a model that under-predicts the rise in latency with batch size is missing contention,
// whereas one that over-predicts is over-serialising. Reporting both as a positive
// percentage makes the two indistinguishable.
func TestUnderPredictionReportsNegative(t *testing.T) {
	s := Signed([]float64{-20, -30, -25})
	if s.Mean >= 0 {
		t.Fatalf("under-prediction must carry a negative mean, got %+.4f", s.Mean)
	}
	if s.Median >= 0 {
		t.Errorf("under-prediction must carry a negative median, got %+.4f", s.Median)
	}
	if s.FractionOver != 0 {
		t.Errorf("no point over-predicts, want FractionOver 0, got %.4f", s.FractionOver)
	}
	if s.MeanAbs <= 0 {
		t.Errorf("magnitude must stay positive, got %.4f", s.MeanAbs)
	}
}

// MeanAbs must agree with MAPE2 over the same points, so adding the signed report cannot
// silently move any previously published absolute figure.
func TestMeanAbsAgreesWithMAPE2(t *testing.T) {
	for _, errs := range [][]float64{
		{+10, -10, +10, -10},
		{-20, -30, -25},
		{0, +5, -5, +100, -1},
		{+3.7},
	} {
		if got, want := Signed(errs).MeanAbs, MAPE2(Abs(errs)); math.Abs(got-want) > 1e-9 {
			t.Errorf("MeanAbs %.6f disagrees with MAPE2 %.6f on %v", got, want, errs)
		}
	}
}

// Asymmetric tails must survive aggregation. A model that is mostly right but occasionally
// very wrong in ONE direction is the signature of a missing mechanism rather than a
// miscalibrated coefficient. The mean is dragged toward the long tail while the median is
// not, so the gap between them is what exposes the asymmetry.
func TestAsymmetricTailsAreVisible(t *testing.T) {
	// Eleven points: ten clustered near zero, one far positive.
	errs := []float64{-2, -1, 0, +1, +1, +2, +2, +3, +3, +4, +90}
	s := Signed(errs)

	if s.Mean <= s.Median {
		t.Fatalf("a long positive tail must pull the mean above the median, "+
			"got mean %+.2f median %+.2f", s.Mean, s.Median)
	}
	// The median is robust to the single outlier; the mean is not. That difference is the
	// diagnostic, and it is lost if either is computed on absolute values.
	if s.Median > 5 {
		t.Errorf("median should resist the outlier, got %+.2f", s.Median)
	}
	if s.Mean < 5 {
		t.Errorf("mean should absorb the outlier, got %+.2f", s.Mean)
	}
	if s.P90 <= s.P10 {
		t.Errorf("percentiles must order P10 <= P90, got %+.2f %+.2f", s.P10, s.P90)
	}
}

// Mirroring every error must mirror every signed statistic and leave magnitudes untouched.
// This is the invariant that guarantees no direction is privileged by the arithmetic.
func TestMirroringErrorsMirrorsTheReport(t *testing.T) {
	errs := []float64{-2, -1, +1, +2, +3, +90} // no zeroes: zero is neither over nor under
	neg := make([]float64, len(errs))
	for i, e := range errs {
		neg[i] = -e
	}
	a, b := Signed(errs), Signed(neg)

	if math.Abs(a.Mean+b.Mean) > 1e-9 {
		t.Errorf("means should mirror: %+.6f vs %+.6f", a.Mean, b.Mean)
	}
	if math.Abs(a.Median+b.Median) > 1e-9 {
		t.Errorf("medians should mirror: %+.6f vs %+.6f", a.Median, b.Median)
	}
	if math.Abs(a.MeanAbs-b.MeanAbs) > 1e-9 {
		t.Errorf("magnitude must be direction-blind: %.6f vs %.6f", a.MeanAbs, b.MeanAbs)
	}
	if math.Abs(a.FractionOver+b.FractionOver-1) > 1e-9 {
		t.Errorf("FractionOver should complement (no zeroes here): %.4f vs %.4f",
			a.FractionOver, b.FractionOver)
	}
	if math.Abs(a.P90+b.P10) > 1e-9 {
		t.Errorf("P90 should mirror P10: %+.4f vs %+.4f", a.P90, b.P10)
	}
}

func TestSignedOfNothingIsNotANumber(t *testing.T) {
	s := Signed(nil)
	if s.N != 0 {
		t.Errorf("N should be 0, got %d", s.N)
	}
	for name, v := range map[string]float64{"Mean": s.Mean, "MeanAbs": s.MeanAbs,
		"Median": s.Median, "FractionOver": s.FractionOver} {
		if !math.IsNaN(v) {
			t.Errorf("%s of an empty slice should be NaN, got %v", name, v)
		}
	}
}

// Abs must not mutate its input: the caller keeps the signed slice for the signed report
// and passes a copy to the absolute aggregate.
func TestAbsLeavesTheCallersSliceSigned(t *testing.T) {
	errs := []float64{-5, +5}
	_ = Abs(errs)
	if errs[0] != -5 {
		t.Errorf("Abs mutated its input: %v", errs)
	}
}

// Two-sided tails must mirror under p -> 100-p at EVERY sample size, including the small
// buckets the per-concurrency report uses. Nearest-rank rounding fails this at n=6: both
// percentiles round up a rank and the report gains a direction absent from the data.
func TestPercentilePairsMirrorAtEverySampleSize(t *testing.T) {
	for n := 1; n <= 40; n++ {
		xs := make([]float64, n)
		neg := make([]float64, n)
		for i := range xs {
			xs[i] = float64(i) - float64(n-1)/2 // symmetric about zero
			neg[i] = -xs[i]
		}
		lo, hi := Percentile(xs, 10), Percentile(xs, 90)
		if math.Abs(lo+hi) > 1e-9 {
			t.Errorf("n=%d: P10 %+.4f and P90 %+.4f should mirror about zero", n, lo, hi)
		}
		if got, want := Percentile(neg, 10), -Percentile(xs, 90); math.Abs(got-want) > 1e-9 {
			t.Errorf("n=%d: negating data should swap tails, got %+.4f want %+.4f", n, got, want)
		}
	}
}

// The percentile of a single point is that point, at any p: there is nothing to interpolate.
func TestPercentileOfOnePoint(t *testing.T) {
	for _, p := range []float64{0, 10, 50, 90, 100} {
		if got := Percentile([]float64{7.5}, p); got != 7.5 {
			t.Errorf("p%v of one point should be that point, got %v", p, got)
		}
	}
}
