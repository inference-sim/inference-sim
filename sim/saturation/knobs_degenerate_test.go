package saturation

import "testing"

// R20 / self-audit D6: zero events and single-event streams must yield STABLE
// with a usable Signals map, never a panic -- at ANY knob setting including the
// extremes a user could plausibly type.
func TestKnobs_DegenerateInputNeverPanics(t *testing.T) {
	extremes := []float64{1e-12, 0.5, 1.0, 2.0, 1e12}

	for _, s := range extremes {
		d := NewCompositeDetectorWithSensitivity(s)
		if got := d.Detect().Level; got != Stable {
			t.Errorf("composite sensitivity=%v: zero events gave %v, want STABLE", s, got)
		}
		if d.Detect().Signals == nil {
			t.Errorf("composite sensitivity=%v: nil Signals map on zero events", s)
		}
		// One arrival, no completions.
		d.Observe(Event{Timestamp: 1, Type: Arrival, RequestID: "r"})
		_ = d.Detect()
		// Reset must restore the zero-event verdict.
		d.Reset()
		if got := d.Detect().Level; got != Stable {
			t.Errorf("composite sensitivity=%v: after Reset gave %v, want STABLE", s, got)
		}
	}

	for _, k := range extremes {
		cfg := DefaultBacklogDriftConfig()
		cfg.SlopeK = k
		d := NewBacklogDriftDetectorWithConfig(cfg)
		if got := d.Detect().Level; got != Stable {
			t.Errorf("backlog-drift slope_k=%v: zero events gave %v, want STABLE", k, got)
		}
		d.Observe(Event{Timestamp: 1, Type: Arrival, RequestID: "r"})
		r := d.Detect()
		if r.Signals == nil {
			t.Errorf("backlog-drift slope_k=%v: nil Signals map", k)
		}
		d.Reset()
		if got := d.Detect().Level; got != Stable {
			t.Errorf("backlog-drift slope_k=%v: after Reset gave %v, want STABLE", k, got)
		}
	}
}

// All events sharing one timestamp is a real degenerate shape (a burst injected at
// t=0) and must not divide by a zero span.
func TestKnobs_AllEventsSameTimestampNoPanic(t *testing.T) {
	d := NewCompositeDetectorWithSensitivity(2.0)
	bd := NewBacklogDriftDetectorWithConfig(DefaultBacklogDriftConfig())
	for i := 0; i < 40; i++ {
		e := Event{Timestamp: 0, Type: Arrival, RequestID: "r"}
		d.Observe(e)
		bd.Observe(e)
		c := Event{Timestamp: 0, Type: Completion, RequestID: "r", LatencyMs: 5}
		d.Observe(c)
		bd.Observe(c)
	}
	_ = d.Detect()
	_ = bd.Detect()
}
