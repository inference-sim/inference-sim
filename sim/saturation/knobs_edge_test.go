package saturation

import (
	"math"
	"testing"
)

// Self-audit D1: effectiveSlopeK must never return a value that makes the banding
// degenerate. Its output is used as a MULTIPLIER on the noise floor and as a
// DIVISOR for the score, so a zero, negative, NaN or Inf return would break both.
func TestEffectiveSlopeK_NeverDegenerate(t *testing.T) {
	for _, in := range []float64{
		0, -1, -1e300, math.NaN(), math.Inf(1), math.Inf(-1),
		math.SmallestNonzeroFloat64, 1e-300, 0.5, 3.0, 1e300, math.MaxFloat64,
	} {
		cfg := BacklogDriftConfig{SlopeK: in}
		got := cfg.effectiveSlopeK()
		if got <= 0 || math.IsNaN(got) || math.IsInf(got, 0) {
			t.Errorf("effectiveSlopeK() with SlopeK=%v returned %v; must be finite and > 0 (it multiplies the floor and divides the score)", in, got)
		}
	}
}

// The same obligation for composite: whatever the caller passes, the applied
// sensitivity must keep the noise floor finite and positive.
func TestCompositeSensitivity_NeverDegenerate(t *testing.T) {
	events := makeCompositeStream(30)
	for _, in := range []float64{
		0, -1, math.NaN(), math.Inf(1), math.Inf(-1),
		math.SmallestNonzeroFloat64, 1e-300, 1.0, 1e300, math.MaxFloat64,
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
			if math.IsNaN(r.Score) || math.IsInf(r.Score, 0) {
				t.Fatalf("sensitivity=%v produced a non-finite Score %v", in, r.Score)
			}
		}
	}
}
