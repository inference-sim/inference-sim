package harness

import (
	"math"
	"testing"
)

// A pure bias is fully removable: scatter is zero and the residual floor is zero. This is the
// case where chasing a coefficient is the right move, and the statistics must say so.
func TestPureBiasHasNoScatterAndNoResidual(t *testing.T) {
	// Every point over-predicts by exactly 10%.
	s := Logs([]float64{10, 10, 10, 10})
	if math.Abs(s.GeoMeanPct-10) > 1e-9 {
		t.Errorf("bias should read +10%%, got %+.6f%%", s.GeoMeanPct)
	}
	if s.SDLog > 1e-12 {
		t.Errorf("a pure bias has no scatter, got sd %.9f", s.SDLog)
	}
	if s.ResidualFloor > 1e-9 {
		t.Errorf("a pure bias is fully removable, got floor %.9f%%", s.ResidualFloor)
	}
}

// Pure scatter is NOT removable: the bias is zero yet the residual floor equals the error
// already present. A model in this state cannot be improved by rescaling, which is the whole
// reason to report the two separately.
func TestPureScatterIsNotRemovable(t *testing.T) {
	// Symmetric in log space: x1.25 and x0.8 are equal and opposite.
	s := Logs([]float64{25, -20, 25, -20})
	if math.Abs(s.GeoMeanPct) > 1e-9 {
		t.Errorf("symmetric log scatter has no bias, got %+.6f%%", s.GeoMeanPct)
	}
	if s.SDLog < 0.1 {
		t.Errorf("scatter should be substantial, got sd %.6f", s.SDLog)
	}
	before := Signed([]float64{25, -20, 25, -20}).MeanAbs
	if s.ResidualFloor < before*0.95 {
		t.Errorf("pure scatter should survive bias removal: floor %.4f%% vs error %.4f%%",
			s.ResidualFloor, before)
	}
}

// For errors of the magnitude this comparison actually produces, removing the bias must not
// make the model worse. The guarantee is NOT universal -- de-biasing scales every ratio by one
// constant, which in percentage space can cost more than it saves for a point near a predicted
// value of zero -- so the bound is asserted over the regime the kernel occupies (p10 about
// -11%, p90 about +17%) rather than claimed in general.
func TestResidualFloorDoesNotExceedTheErrorInThisRegime(t *testing.T) {
	for _, errs := range [][]float64{
		{10, 10, 10},
		{25, -20, 25, -20},
		{50, 5, -5, -30, 80, 2},
		{-40, -41, -39},
		{17, -11, 3, 8, -6},
	} {
		floor := Logs(errs).ResidualFloor
		have := Signed(errs).MeanAbs
		if floor > have*1.001+1e-9 {
			t.Errorf("floor %.4f%% exceeds the error %.4f%% on %v", floor, have, errs)
		}
	}
}

// The documented failure mode must be real and must stay documented: a point near a predicted
// value of zero makes the floor exceed the error it decomposes. If this ever stops holding, the
// caveat on LogStats is stale and should be removed rather than left to mislead.
func TestResidualFloorIsNotABoundNearTotalUnderPrediction(t *testing.T) {
	errs := []float64{-99.97, 41.6}
	floor := Logs(errs).ResidualFloor
	have := Signed(errs).MeanAbs
	if floor <= have {
		t.Errorf("expected the documented blow-up (floor above error), got floor %.2f%% "+
			"vs error %.2f%% -- update the caveat on LogStats", floor, have)
	}
}

// Mean-of-log and the arithmetic mean of ratios are different functionals, and the arithmetic
// one is the larger (Jensen). A reader comparing a signed mean against a log mean must not be
// able to conclude the two disagree -- this asserts the relationship they actually obey.
func TestArithmeticMeanIsNeverBelowTheGeometric(t *testing.T) {
	for _, errs := range [][]float64{
		{10, 10, 10},
		{25, -20, 25, -20},
		{50, 5, -5, -30, 80, 2},
		{-10, -20, -30},
	} {
		arith := Signed(errs).Mean
		geo := Logs(errs).GeoMeanPct
		if arith < geo-1e-9 {
			t.Errorf("arithmetic %+.6f%% below geometric %+.6f%% on %v -- impossible for one "+
				"residual set", arith, geo, errs)
		}
	}
}

// A point implying a non-positive prediction has no logarithm and must be dropped, with N
// reporting the reduction rather than the statistic silently becoming NaN.
func TestImpossiblePointsAreExcludedAndCounted(t *testing.T) {
	s := Logs([]float64{10, -100, -150, 20})
	if s.N != 2 {
		t.Errorf("want 2 usable points, got %d", s.N)
	}
	if math.IsNaN(s.MeanLog) {
		t.Error("usable points remain; the mean should not be NaN")
	}
}

func TestLogsOfNothingIsNotANumber(t *testing.T) {
	s := Logs(nil)
	if s.N != 0 || !math.IsNaN(s.MeanLog) || !math.IsNaN(s.ResidualFloor) {
		t.Errorf("empty input should give N=0 and NaN statistics, got %+v", s)
	}
}
