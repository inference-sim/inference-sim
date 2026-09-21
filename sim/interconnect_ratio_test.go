package sim

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestInterconnectBwRatio_UncalibratedIsNeutral verifies BC-4/BC-8: every
// hardware calib that does not declare BOTH interconnect bandwidths — including
// every calib that existed before #1530 — yields a neutral ratio of exactly 1.0,
// so cross-node traffic is priced identically to intra-node traffic (INV-6).
func TestInterconnectBwRatio_UncalibratedIsNeutral(t *testing.T) {
	tests := []struct {
		name  string
		calib HardwareCalib
	}{
		{"zero value (pre-#1530 calib)", HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35}},
		{"intra only", HardwareCalib{IntraNodeBwGBps: 450}},
		{"inter only", HardwareCalib{InterNodeBwGBps: 50}},
		{"negative intra", HardwareCalib{IntraNodeBwGBps: -450, InterNodeBwGBps: 50}},
		{"negative inter", HardwareCalib{IntraNodeBwGBps: 450, InterNodeBwGBps: -50}},
		{"NaN intra", HardwareCalib{IntraNodeBwGBps: math.NaN(), InterNodeBwGBps: 50}},
		{"Inf intra", HardwareCalib{IntraNodeBwGBps: math.Inf(1), InterNodeBwGBps: 50}},
		{"NaN inter", HardwareCalib{IntraNodeBwGBps: 450, InterNodeBwGBps: math.NaN()}},
		{"Inf inter", HardwareCalib{IntraNodeBwGBps: 450, InterNodeBwGBps: math.Inf(1)}},
		{"fabric faster than on-node link (clamped)", HardwareCalib{IntraNodeBwGBps: 50, InterNodeBwGBps: 450}},
		{"fabric equal to on-node link", HardwareCalib{IntraNodeBwGBps: 450, InterNodeBwGBps: 450}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, 1.0, tc.calib.InterconnectBwRatio(),
				"an uncalibrated or sub-unit interconnect must price cross-node exactly like intra-node")
		})
	}
}

// TestInterconnectBwRatio_CalibratedValue verifies the ratio itself: a fabric N
// times slower than the on-node link yields N.
func TestInterconnectBwRatio_CalibratedValue(t *testing.T) {
	calib := HardwareCalib{IntraNodeBwGBps: 450, InterNodeBwGBps: 50}
	assert.InDelta(t, 9.0, calib.InterconnectBwRatio(), 1e-12)
}

// TestInterconnectBwRatio_MonotoneInFabricSpeed verifies BC-3 at the source: a
// slower fabric never yields a smaller ratio, and strictly increases it over a
// realistic range. This is the law the whole cross-node cost inherits.
func TestInterconnectBwRatio_MonotoneInFabricSpeed(t *testing.T) {
	prev := 0.0
	for _, interBw := range []float64{450, 200, 100, 50, 25, 12.5, 5} {
		got := HardwareCalib{IntraNodeBwGBps: 450, InterNodeBwGBps: interBw}.InterconnectBwRatio()
		assert.GreaterOrEqual(t, got, prev,
			"ratio must not decrease as the fabric gets slower (inter=%v)", interBw)
		prev = got
	}
	assert.Greater(t, prev, 1.0, "the slowest fabric tested must yield a ratio above the neutral 1.0")
}

// TestInterconnectBwRatio_ScaleInvariant verifies that only the ratio matters:
// doubling both bandwidths leaves the charged ratio unchanged. This documents why
// the absolute GB/s values need not be precise, only their proportion.
func TestInterconnectBwRatio_ScaleInvariant(t *testing.T) {
	base := HardwareCalib{IntraNodeBwGBps: 450, InterNodeBwGBps: 50}.InterconnectBwRatio()
	doubled := HardwareCalib{IntraNodeBwGBps: 900, InterNodeBwGBps: 100}.InterconnectBwRatio()
	assert.InDelta(t, base, doubled, 1e-12)
}

// TestInterconnectBwRatio_AlwaysFinite verifies the accessor's contract that every
// consumer relies on: the returned ratio is always a finite number. Two finite,
// positive bandwidths can still have a quotient that overflows float64, and an
// infinite ratio would otherwise flow into the cost model.
func TestInterconnectBwRatio_AlwaysFinite(t *testing.T) {
	overflowing := HardwareCalib{IntraNodeBwGBps: math.MaxFloat64, InterNodeBwGBps: 5e-324}
	got := overflowing.InterconnectBwRatio()
	assert.False(t, math.IsInf(got, 0), "the ratio must never be infinite")
	assert.False(t, math.IsNaN(got), "the ratio must never be NaN")
	assert.Equal(t, 1.0, got, "an unusable (overflowing) ratio must degrade to the neutral baseline")

	// A large-but-representable ratio is honored, so the guard is not over-broad.
	large := HardwareCalib{IntraNodeBwGBps: 1e6, InterNodeBwGBps: 1e-3}
	assert.InDelta(t, 1e9, large.InterconnectBwRatio(), 1.0)
}
