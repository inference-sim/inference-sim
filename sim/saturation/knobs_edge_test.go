package saturation

import (
	"math"
	"testing"
)

// The Score/Level coupling must survive ANY SlopeK a caller can set, including
// adversarial magnitudes. This is the observable form of the obligation: whenever
// the detector says OVERLOADED, Score must have reached its 1.0 cap.
//
// The multiplier is used both to place the band boundary and as the score
// denominator, so a value that drives their PRODUCT with the noise floor to zero
// (or to a non-finite value) decouples them -- which is why this asserts on the
// verdicts rather than on the multiplier in isolation. An earlier version of this
// test checked only that the multiplier itself was finite and positive, and that
// version passed while a subnormal slope_k silently produced OVERLOADED with
// Score 0.
func TestBacklogDrift_ScoreLevelCouplingSurvivesAnySlopeK(t *testing.T) {
	events := risingBacklogStream(40)
	for _, in := range []float64{
		0, -1, -1e300, math.NaN(), math.Inf(1), math.Inf(-1),
		math.SmallestNonzeroFloat64, 1e-320, 1e-300, 1e-6, 0.5, 3.0, 1e300, math.MaxFloat64,
	} {
		cfg := slopeKConfig(in)
		d := NewBacklogDriftDetectorWithConfig(cfg)
		d.Reset()
		for _, e := range events {
			d.Observe(e)
			r := d.Detect()
			if math.IsNaN(r.Score) || math.IsInf(r.Score, 0) {
				t.Fatalf("SlopeK=%v produced a non-finite Score %v", in, r.Score)
			}
			if r.Level == Overloaded && r.Score != 1.0 {
				t.Errorf("SlopeK=%v: OVERLOADED with Score %v, want the 1.0 cap; Level and Score are decoupled", in, r.Score)
			}
			// The key is present only when the knob was explicitly configured
			// (an unset SlopeK keeps a default report byte-identical). When it IS
			// present it must describe a usable multiplier.
			if sk, present := r.Signals["slope_k"]; present {
				if sk <= 0 || math.IsNaN(sk) || math.IsInf(sk, 0) {
					t.Errorf("SlopeK=%v: reported slope_k %v is not finite and positive", in, sk)
				}
			}
		}
	}
}

// The same obligation for composite, stated on the observable verdict: whatever
// sensitivity a caller passes, the reported floor must stay finite and STRICTLY
// POSITIVE, and the verdict must remain consistent with it. A floor that
// underflows to zero would make every event non-STABLE (since score >= 0), which
// is the "reports saturated on everything" failure R20 exists to prevent.
func TestCompositeSensitivity_NeverDegenerate(t *testing.T) {
	events := makeCompositeStream(30)
	for _, in := range []float64{
		0, -1, math.NaN(), math.Inf(1), math.Inf(-1),
		math.SmallestNonzeroFloat64, 1e-320, 1e-300, 1e-6, 1.0, 1e300, math.MaxFloat64,
	} {
		d := NewCompositeDetectorWithSensitivity(in)
		d.Reset()
		for _, e := range events {
			d.Observe(e)
			r := d.Detect()
			nf := r.Signals["noise_floor"]
			if math.IsNaN(nf) || math.IsInf(nf, 0) {
				t.Fatalf("sensitivity=%v produced a non-finite noise_floor %v", in, nf)
			}
			if nf <= 0 {
				t.Fatalf("sensitivity=%v produced a non-positive noise_floor %v; every event would be non-STABLE", in, nf)
			}
			if math.IsNaN(r.Score) || math.IsInf(r.Score, 0) {
				t.Fatalf("sensitivity=%v produced a non-finite Score %v", in, r.Score)
			}
			// The verdict must stay consistent with the reported floor.
			if r.Level == Stable && r.Score >= nf {
				t.Errorf("sensitivity=%v: STABLE but Score %v >= floor %v", in, r.Score, nf)
			}
		}
	}
}
