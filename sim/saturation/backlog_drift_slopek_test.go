package saturation

import (
	"testing"
	"time"
)

// risingBacklogStream produces a steadily growing in-flight count, so the online
// OLS slope is positive and the band classifier has something to band.
// Arrivals outpace completions 2:1.
func risingBacklogStream(n int) []Event {
	events := make([]Event, 0, 3*n)
	var ts int64
	for i := 0; i < n; i++ {
		ts += 1_000_000
		events = append(events,
			Event{Timestamp: ts, Type: Arrival, RequestID: "a"},
			Event{Timestamp: ts + 100_000, Type: Arrival, RequestID: "b"},
			Event{Timestamp: ts + 200_000, Type: Completion, RequestID: "a", LatencyMs: 10},
		)
	}
	return events
}

// slopeKConfig builds a BacklogDriftConfig with a small window (so a handful of
// events span enough buckets to drive the slope) and an explicit SlopeK.
func slopeKConfig(slopeK float64) BacklogDriftConfig {
	c := NewBacklogDriftConfig(
		1*time.Second, 5, 2.0, 0.2, 0.95, 2, 1, 0.95, 0.98,
	)
	c.SlopeK = slopeK
	return c
}

// countLevels tallies how many events produced each level.
func countLevels(d Detector, events []Event) map[Level]int {
	d.Reset()
	out := map[Level]int{}
	for _, e := range events {
		d.Observe(e)
		out[d.Detect().Level]++
	}
	return out
}

// BC-2: slope_k is the severity boundary. A stream that is BACKLOGGED at a large
// K must escalate to OVERLOADED at a small K -- asserted over observable Levels,
// not by reproducing the K*noiseFloor formula.
func TestBacklogDrift_SlopeKMovesSeverityBoundary(t *testing.T) {
	events := risingBacklogStream(40)

	loose := countLevels(NewBacklogDriftDetectorWithConfig(slopeKConfig(50.0)), events)
	tight := countLevels(NewBacklogDriftDetectorWithConfig(slopeKConfig(0.01)), events)

	// Vacuous-pass guard: the stream must reach a saturated level at all.
	if loose[Backlogged]+loose[Overloaded] == 0 {
		t.Fatalf("fixture never leaves STABLE; the escalation assertion would be vacuous (levels=%v)", loose)
	}

	if tight[Overloaded] <= loose[Overloaded] {
		t.Errorf("tightening slope_k did not escalate severity: OVERLOADED count %d at K=0.01 vs %d at K=50",
			tight[Overloaded], loose[Overloaded])
	}
	if loose[Backlogged] <= tight[Backlogged] {
		t.Errorf("loosening slope_k did not retain BACKLOGGED instead of escalating: %d at K=50 vs %d at K=0.01",
			loose[Backlogged], tight[Backlogged])
	}
}

// BC-5: the zero-fill trap. NewBacklogDriftConfig returns a struct literal, so
// any config built through it -- or via a bare literal -- leaves SlopeK at zero.
// An unguarded zero would make `slope <= 0*noiseFloor` false for every positive
// slope, banding everything OVERLOADED. Every construction path must behave as
// the documented default instead.
func TestBacklogDrift_ZeroSlopeKBehavesAsDefault(t *testing.T) {
	events := risingBacklogStream(40)
	explicit := countLevels(NewBacklogDriftDetectorWithConfig(slopeKConfig(backlogDriftSlopeK)), events)

	t.Run("nine-argument constructor zero-fills SlopeK", func(t *testing.T) {
		c := NewBacklogDriftConfig(1*time.Second, 5, 2.0, 0.2, 0.95, 2, 1, 0.95, 0.98)
		if c.SlopeK != 0 {
			t.Fatalf("precondition: expected the constructor to leave SlopeK zero, got %v", c.SlopeK)
		}
		got := countLevels(NewBacklogDriftDetectorWithConfig(c), events)
		if got[Overloaded] != explicit[Overloaded] || got[Backlogged] != explicit[Backlogged] {
			t.Errorf("zero SlopeK diverged from the default: got %v, want %v", got, explicit)
		}
	})

	t.Run("bare struct literal", func(t *testing.T) {
		c := BacklogDriftConfig{WindowSize: 1 * time.Second, MinWindows: 5}
		got := countLevels(NewBacklogDriftDetectorWithConfig(c), events)
		if got[Overloaded] != explicit[Overloaded] || got[Backlogged] != explicit[Backlogged] {
			t.Errorf("bare literal diverged from the default: got %v, want %v", got, explicit)
		}
	})

	t.Run("DefaultBacklogDriftConfig matches an explicit default", func(t *testing.T) {
		got := countLevels(NewBacklogDriftDetectorWithConfig(DefaultBacklogDriftConfig()), events)
		want := countLevels(NewBacklogDriftDetector(), events)
		if got[Overloaded] != want[Overloaded] || got[Backlogged] != want[Backlogged] {
			t.Errorf("DefaultBacklogDriftConfig diverged from the default constructor: got %v, want %v", got, want)
		}
	})
}

// BC-6: Level and Score must derive from the SAME multiplier. backlog_drift.go
// reads the band multiplier in two places (the band switch and the score
// denominator); patching only one makes Score==1.0 stop coinciding with the
// OVERLOADED boundary. This test fails if either read site is missed.
func TestBacklogDrift_ScoreAndLevelAgreeAtCustomSlopeK(t *testing.T) {
	events := risingBacklogStream(40)

	for _, k := range []float64{0.5, 1.0, 3.0, 10.0} {
		d := NewBacklogDriftDetectorWithConfig(slopeKConfig(k))
		d.Reset()
		sawSaturated := false
		for _, e := range events {
			d.Observe(e)
			r := d.Detect()
			// The contract: Score reaches its 1.0 cap exactly at the OVERLOADED
			// band edge. So a Score strictly below 1.0 must never be OVERLOADED,
			// and OVERLOADED must always carry the capped Score.
			if r.Level == Overloaded {
				sawSaturated = true
				if r.Score != 1.0 {
					t.Errorf("slope_k=%v: OVERLOADED with Score %v, want the 1.0 cap (Level and Score read different multipliers)", k, r.Score)
				}
			}
			if r.Score < 1.0 && r.Level == Overloaded {
				t.Errorf("slope_k=%v: Score %v < 1.0 but Level is OVERLOADED", k, r.Score)
			}
		}
		if k <= 1.0 && !sawSaturated {
			t.Errorf("slope_k=%v: expected the rising stream to reach OVERLOADED at a tight K; assertion was vacuous", k)
		}
	}
}

// The trace must explain each verdict, so the ACTIVE band multiplier is reported
// in Signals -- matching the convention threshold already follows (it reports
// "threshold") and what the extension guide asks for. This pins the key as an
// intentional part of the report rather than an incidental addition.
//
// Verdicts are unaffected: the report gains a diagnostic, and stdout (the INV-6
// surface) is unchanged.
func TestBacklogDrift_ReportsActiveSlopeKInSignals(t *testing.T) {
	events := risingBacklogStream(20)
	for _, k := range []float64{1.0, 3.0, 7.5} {
		d := NewBacklogDriftDetectorWithConfig(slopeKConfig(k))
		d.Reset()
		for _, e := range events {
			d.Observe(e)
		}
		got, ok := d.Detect().Signals["slope_k"]
		if !ok {
			t.Fatalf("slope_k=%v: Signals is missing the slope_k key; the trace cannot explain the band boundary", k)
		}
		if got != k {
			t.Errorf("Signals[\"slope_k\"] = %v, want the active multiplier %v", got, k)
		}
	}

	// An UNCONFIGURED detector must report a POSITIVE multiplier, never a zero: a
	// zero would misdescribe the banding, and would mean the detector bands every
	// rising trace OVERLOADED. Asserted as the property that matters rather than
	// against the constant, so the test survives a change of default.
	//
	// The config below leaves slope_k unset but keeps the small window, so the
	// comparison isolates the multiplier (the 60s production window of
	// NewBacklogDriftDetector would complete no bucket on this fixture).
	unset := NewBacklogDriftConfig(1*time.Second, 5, 2.0, 0.2, 0.95, 2, 1, 0.95, 0.98)
	d := NewBacklogDriftDetectorWithConfig(unset)
	d.Reset()
	for _, e := range events {
		d.Observe(e)
	}
	reported := d.Detect().Signals["slope_k"]
	if reported <= 0 {
		t.Fatalf("an unset slope_k was reported as %v; must be > 0 or the banding is inverted", reported)
	}
	// And the reported value must be the one actually in force: configuring it
	// explicitly must reproduce the same verdicts.
	explicit := countLevels(NewBacklogDriftDetectorWithConfig(slopeKConfig(reported)), events)
	implied := countLevels(NewBacklogDriftDetectorWithConfig(unset), events)
	if explicit[Overloaded] != implied[Overloaded] || explicit[Backlogged] != implied[Backlogged] {
		t.Errorf("the reported slope_k %v does not reproduce the unconfigured detector's verdicts: %v vs %v", reported, explicit, implied)
	}
}
