package workload

import (
	"math"
	"testing"
)

func TestComputeHitRateComparison(t *testing.T) {
	cases := []struct {
		name       string
		real, sim  float64
		tolPP      float64
		wantErrPP  float64
		wantWithin bool
	}{
		{"exact match", 0.70, 0.70, 5, 0, true},
		{"within band", 0.70, 0.73, 5, 3, true},
		{"at boundary", 0.70, 0.75, 5, 5, true},
		{"exceeds band", 0.70, 0.80, 5, 10, false},
		{"sim under real still within", 0.70, 0.66, 5, 4, true},
		{"zero real", 0.0, 0.03, 5, 3, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ComputeHitRateComparison(c.real, c.sim, c.tolPP, "tiered")
			if math.Abs(got.AbsErrorPP-c.wantErrPP) > 1e-9 {
				t.Errorf("abs_error_pp = %v, want %v", got.AbsErrorPP, c.wantErrPP)
			}
			if got.Within != c.wantWithin {
				t.Errorf("within = %v, want %v (err=%v tol=%v)", got.Within, c.wantWithin, got.AbsErrorPP, c.tolPP)
			}
			if got.Source != "tiered" {
				t.Errorf("source = %q, want tiered", got.Source)
			}
			if got.RealHitRate != c.real || got.SimHitRate != c.sim {
				t.Errorf("real/sim not echoed: got %v/%v", got.RealHitRate, got.SimHitRate)
			}
		})
	}
}
