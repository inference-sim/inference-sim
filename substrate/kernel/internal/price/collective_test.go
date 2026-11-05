package price

import (
	"math"
	"testing"
)

// The three-parameter form must beat the two-parameter one on the data it was fitted to,
// and must do so in the message range a forward pass actually produces. These are the
// measured H200 8-rank fp16 all-reduce points from AISimulate's NCCL 2.29.2 sweep,
// transcribed so the claim is checkable without the parquet file.
var h200AllReduce8Rank = []struct {
	bytes   float64
	latency float64 // microseconds, measured
}{
	{256, 16.01}, {1024, 15.98}, {2048, 16.04}, {4096, 16.67}, {8192, 18.30},
	{16384, 18.38}, {32768, 19.89}, {65536, 21.45}, {131072, 20.80}, {262144, 20.46},
	{524288, 26.14}, {1048576, 37.46}, {2097152, 51.37}, {4194304, 78.30},
	{8388608, 121.74}, {16777216, 197.34}, {33554432, 319.63}, {67108864, 564.52},
	{134217728, 1052.63}, {268435456, 2040.31},
}

const (
	// The committed coefficients for this configuration.
	h200ARFloorUs     = 15.98
	h200ARPeakBPerUs  = 131566.0
	h200ARTransBPerUs = 92500.0
)

// geometricError returns the geometric mean of the absolute ratio between prediction and
// measurement, which weighs an under-prediction and an over equally.
func geometricError(predict func(bytes float64) float64) float64 {
	var total float64
	for _, p := range h200AllReduce8Rank {
		total += math.Abs(math.Log(predict(p.bytes) / (p.latency * 1e-6)))
	}
	return math.Exp(total / float64(len(h200AllReduce8Rank)))
}

func TestThreeParameterFormBeatsTwoOnMeasuredData(t *testing.T) {
	floor := h200ARFloorUs * 1e-6
	peak := h200ARPeakBPerUs * 1e6
	transition := h200ARTransBPerUs * 1e6

	two := geometricError(func(b float64) float64 {
		return FloorAndRate(b, peak, floor)
	})
	three := geometricError(func(b float64) float64 {
		return CollectiveTime(b, transition, peak, floor)
	})
	t.Logf("geometric error: two-parameter %.3fx, three-parameter %.3fx", two, three)
	if three >= two {
		t.Errorf("the third parameter did not improve the fit: %.3fx against %.3fx",
			three, two)
	}
	// The aggregate gain claimed in the registry's rationale for this configuration.
	if three > 1.25 {
		t.Errorf("three-parameter error %.3fx is worse than the 1.25x this "+
			"configuration should reach", three)
	}
}

// The two-parameter form's failure is specifically an UNDER-prediction in the transition
// region, which is the direction that matters: a cost model that predicts a collective is
// cheaper than it is will recommend a layout that does not perform.
func TestTwoParameterFormUnderpredictsTheTransitionRegion(t *testing.T) {
	floor := h200ARFloorUs * 1e-6
	peak := h200ARPeakBPerUs * 1e6
	transition := h200ARTransBPerUs * 1e6

	var worstTwo, worstThree float64 = 1, 1
	for _, p := range h200AllReduce8Rank {
		if p.bytes < 64<<10 || p.bytes > 8<<20 {
			continue
		}
		measured := p.latency * 1e-6
		if r := measured / FloorAndRate(p.bytes, peak, floor); r > worstTwo {
			worstTwo = r
		}
		if r := measured / CollectiveTime(p.bytes, transition, peak, floor); r > worstThree {
			worstThree = r
		}
	}
	t.Logf("worst under-prediction in 64KiB..8MiB: two-parameter %.2fx, "+
		"three-parameter %.2fx", worstTwo, worstThree)
	if worstTwo < 2.0 {
		t.Errorf("the two-parameter form should understate by over 2x somewhere in "+
			"this range; worst was %.2fx, so this test is no longer measuring what "+
			"it claims", worstTwo)
	}
	if worstThree > 1.5 {
		t.Errorf("the three-parameter form understates by %.2fx, worse than the 1.5x "+
			"it should hold to", worstThree)
	}
}

// A collective never prices below its asymptote: the peak is a ceiling on achievable
// bandwidth, so the additive term cannot make a large transfer look faster than the
// hardware can move it.
func TestPeakRateIsACeilingOnAchievableBandwidth(t *testing.T) {
	floor, peak := 10e-6, 200e9
	// A transition rate absurdly above the peak would, without the ceiling, predict a
	// 1 GiB transfer far faster than the link allows.
	got := CollectiveTime(1<<30, 10*peak, peak, floor)
	atPeak := float64(1<<30) / peak
	if got < atPeak {
		t.Errorf("priced %v, faster than the asymptote's %v", got, atPeak)
	}
}

// Below the floor, size stops mattering. That is what a floor means, and the additive
// term must not reintroduce a size dependence that dominates there.
func TestSmallTransfersAreFloorDominated(t *testing.T) {
	floor := 15.98e-6
	peak, transition := 131566.0*1e6, 92500.0*1e6
	small := CollectiveTime(256, transition, peak, floor)
	if ratio := small / floor; ratio > 1.01 {
		t.Errorf("a 256-byte transfer priced %.3fx the floor; it should be within 1%%",
			ratio)
	}
}

// A missing transition rate falls back to the two-parameter form rather than pricing at
// zero or infinity, so a coefficient set predating the third parameter still works.
func TestAbsentTransitionRateFallsBackToTwoParameters(t *testing.T) {
	floor, peak := 10e-6, 200e9
	for _, bytes := range []float64{1024, 1 << 20, 1 << 30} {
		got := CollectiveTime(bytes, 0, peak, floor)
		want := FloorAndRate(bytes, peak, floor)
		if got != want {
			t.Errorf("at %v bytes: fallback gave %v, two-parameter form gives %v",
				bytes, got, want)
		}
	}
}
