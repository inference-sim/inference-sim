package saturation

import (
	"testing"
)

// makeCompositeStream builds a deterministic event stream with a rising latency
// profile and a completion deficit, so composite has a non-trivial score to band.
// Both legs of every comparison below consume this identical stream.
func makeCompositeStream(n int) []Event {
	events := make([]Event, 0, 2*n)
	for i := 0; i < n; i++ {
		ts := int64(i+1) * 1_000_000
		events = append(events, Event{Timestamp: ts, Type: Arrival, RequestID: string(rune('a' + i%26))})
		// Latency climbs monotonically so the quartile filter validates the trend.
		events = append(events, Event{
			Timestamp: ts + 500_000, Type: Completion,
			RequestID: string(rune('a' + i%26)),
			LatencyMs: float64(100 + i*20),
		})
	}
	return events
}

// levelSequence streams one detector over events and records the level after each.
func levelSequence(d Detector, events []Event) []Level {
	d.Reset()
	out := make([]Level, 0, len(events))
	for _, e := range events {
		d.Observe(e)
		out = append(out, d.Detect().Level)
	}
	return out
}

// resultSequence records the FULL Result after each event, so byte-identity
// covers Score, Confidence and every Signals entry -- not just Level.
func resultSequence(d Detector, events []Event) []Result {
	d.Reset()
	out := make([]Result, 0, len(events))
	for _, e := range events {
		d.Observe(e)
		out = append(out, d.Detect())
	}
	return out
}

func resultsEqual(t *testing.T, want, got []Result, ctx string) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: length %d != %d", ctx, len(got), len(want))
	}
	for i := range want {
		if want[i].Level != got[i].Level {
			t.Errorf("%s: event %d Level %v != %v", ctx, i, got[i].Level, want[i].Level)
		}
		if want[i].Score != got[i].Score {
			t.Errorf("%s: event %d Score %v != %v", ctx, i, got[i].Score, want[i].Score)
		}
		if want[i].Confidence != got[i].Confidence {
			t.Errorf("%s: event %d Confidence %v != %v", ctx, i, got[i].Confidence, want[i].Confidence)
		}
		if len(want[i].Signals) != len(got[i].Signals) {
			t.Errorf("%s: event %d signal count %d != %d", ctx, i, len(got[i].Signals), len(want[i].Signals))
			continue
		}
		for k, wv := range want[i].Signals {
			gv, ok := got[i].Signals[k]
			if !ok {
				t.Errorf("%s: event %d missing signal %q", ctx, i, k)
				continue
			}
			if wv != gv {
				t.Errorf("%s: event %d signal %q = %v, want %v", ctx, i, k, gv, wv)
			}
		}
	}
}

// BC-1: raising sensitivity raises the noise floor, so the detector reports a
// saturated level on no event where a lower sensitivity reported STABLE.
// Stated over observable Levels (not the internal floor), so the test survives a
// reimplementation that preserves behaviour.
func TestComposite_SensitivityIsMonotone(t *testing.T) {
	events := makeCompositeStream(60)
	ladder := []float64{0.25, 0.5, 1.0, 2.0, 4.0, 8.0}

	// Vacuous-pass guard: the most sensitive setting MUST reach a non-STABLE
	// level on this fixture, else "fires nowhere is a subset of fires nowhere"
	// would pass with the feature entirely broken.
	base := levelSequence(NewCompositeDetectorWithSensitivity(ladder[0]), events)
	sawSaturated := false
	for _, l := range base {
		if l != Stable {
			sawSaturated = true
			break
		}
	}
	if !sawSaturated {
		t.Fatalf("fixture never leaves STABLE at sensitivity %v; the monotonicity assertion below would be vacuous", ladder[0])
	}

	prev := base
	for _, s := range ladder[1:] {
		cur := levelSequence(NewCompositeDetectorWithSensitivity(s), events)
		if len(cur) != len(prev) {
			t.Fatalf("sensitivity %v: sequence length changed", s)
		}
		for i := range cur {
			if cur[i] != Stable && prev[i] == Stable {
				t.Errorf("sensitivity %v: event %d became %v while the lower sensitivity was STABLE (monotonicity violated)",
					s, i, cur[i])
			}
		}
		prev = cur
	}
}

// BC-3 + BC-4: an absent composite: block and an explicit sensitivity of 1.0 must
// both reproduce the historical behaviour field-for-field, including every entry
// of the Signals map (which is serialized into the trace report, so a drift here
// would break byte-identity, INV-6).
func TestComposite_DefaultSensitivityIsByteIdentical(t *testing.T) {
	events := makeCompositeStream(60)

	// Reference: the historical computation, with the noise floor unscaled.
	reference := resultSequence(NewCompositeDetector(), events)

	t.Run("explicit 1.0 equals the default constructor", func(t *testing.T) {
		got := resultSequence(NewCompositeDetectorWithSensitivity(1.0), events)
		resultsEqual(t, reference, got, "sensitivity=1.0")
	})

	t.Run("absent block resolves to the default constructor", func(t *testing.T) {
		d, err := BuildDetector("composite", SaturationConfig{})
		if err != nil {
			t.Fatalf("BuildDetector: %v", err)
		}
		got := resultSequence(d, events)
		resultsEqual(t, reference, got, "absent composite block")
	})

	t.Run("explicit 1.0 via config equals absent", func(t *testing.T) {
		one := 1.0
		d, err := BuildDetector("composite", SaturationConfig{Composite: &CompositeBlock{Sensitivity: &one}})
		if err != nil {
			t.Fatalf("BuildDetector: %v", err)
		}
		got := resultSequence(d, events)
		resultsEqual(t, reference, got, "sensitivity: 1.0 via config")
	})
}

// The reported noise_floor must describe the floor the banding ACTUALLY used,
// otherwise the trace explains a verdict the detector did not reach.
//
// Stated as an observable property rather than by recomputing the implementation's
// formula: the reported floor must be a threshold CONSISTENT with the verdict.
// Whenever the detector says STABLE its score sits below the reported floor, and
// whenever it says OVERLOADED its latency trend sits above it. A detector that
// reported an unscaled floor while banding on a scaled one violates this on the
// events between the two floors -- and it holds for ANY implementation that reports
// the floor it used, including one that scales the boundary differently.
func TestComposite_ReportedFloorIsConsistentWithTheVerdict(t *testing.T) {
	events := makeCompositeStream(60)

	for _, s := range []float64{0.25, 1.0, 4.0, 16.0} {
		d := NewCompositeDetectorWithSensitivity(s)
		d.Reset()
		checked := 0
		for _, e := range events {
			d.Observe(e)
			r := d.Detect()
			floor := r.Signals["noise_floor"]
			score := r.Score
			lt := r.Signals["latency_trend"]

			switch r.Level {
			case Stable:
				if score >= floor {
					t.Errorf("sensitivity=%v: STABLE but score %v >= reported floor %v; the reported floor is not the one used", s, score, floor)
				}
				checked++
			case Overloaded:
				if lt <= floor {
					t.Errorf("sensitivity=%v: OVERLOADED but latency_trend %v <= reported floor %v", s, lt, floor)
				}
				checked++
			case Backlogged:
				if score < floor {
					t.Errorf("sensitivity=%v: BACKLOGGED but score %v < reported floor %v (should have been STABLE)", s, score, floor)
				}
				checked++
			}
		}
		if checked == 0 {
			t.Fatalf("sensitivity=%v: no verdicts examined; the consistency assertion was vacuous", s)
		}
	}
}

// Raising sensitivity must raise the reported floor. This is the observable
// direction of the knob -- it says nothing about HOW the floor is computed, only
// that the dial moves the bar the way its documentation promises.
func TestComposite_HigherSensitivityRaisesTheReportedFloor(t *testing.T) {
	events := makeCompositeStream(60)
	ladder := []float64{0.5, 1.0, 2.0, 8.0}

	var prev []float64
	for _, s := range ladder {
		d := NewCompositeDetectorWithSensitivity(s)
		d.Reset()
		var floors []float64
		for _, e := range events {
			d.Observe(e)
			floors = append(floors, d.Detect().Signals["noise_floor"])
		}
		if prev != nil {
			rose := 0
			for i := range floors {
				if floors[i] < prev[i] {
					t.Errorf("sensitivity=%v: event %d floor %v is BELOW the lower sensitivity's %v", s, i, floors[i], prev[i])
				}
				if floors[i] > prev[i] {
					rose++
				}
			}
			if rose == 0 {
				t.Errorf("sensitivity=%v: raising sensitivity moved no floor; the knob is inert", s)
			}
		}
		prev = floors
	}
}

// Severity must never INCREASE as sensitivity increases. This is the strong form
// of the calibration contract: an operator raising sensitivity to suppress false
// alarms must never be handed a MORE severe verdict on any event.
//
// (The weaker "never becomes non-STABLE" form is blind to a BACKLOGGED -> OVERLOADED
// escalation, which would be just as wrong for a knob sold as a suppressor.)
func TestComposite_SeverityNeverEscalatesWithSensitivity(t *testing.T) {
	events := makeCompositeStream(60)
	severity := map[Level]int{Stable: 0, Backlogged: 1, Overloaded: 2}

	var prev []Level
	for _, s := range []float64{0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0} {
		cur := levelSequence(NewCompositeDetectorWithSensitivity(s), events)
		if prev != nil {
			for i := range cur {
				if severity[cur[i]] > severity[prev[i]] {
					t.Errorf("sensitivity=%v: event %d escalated %v -> %v as sensitivity ROSE", s, i, prev[i], cur[i])
				}
			}
		}
		prev = cur
	}

	// Positive control: the ladder must actually span a change in verdicts, or the
	// no-escalation assertion above proves nothing.
	lo := levelSequence(NewCompositeDetectorWithSensitivity(0.25), events)
	hi := levelSequence(NewCompositeDetectorWithSensitivity(16.0), events)
	differs := false
	for i := range lo {
		if lo[i] != hi[i] {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("the sensitivity ladder changed no verdict; the monotonicity assertions were vacuous")
	}
}
