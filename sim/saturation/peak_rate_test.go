package saturation

import (
	"math"
	"strconv"
	"testing"
)

// --- Fixtures -------------------------------------------------------------
//
// These build event streams with a KNOWN qualitative regime, so the assertions
// below are about the detector's behaviour on healthy vs overloaded traffic rather
// than about any internal quantity.

// saturatedStream models a server falling behind: arrivals outpace completions, so
// the backlog high-water mark keeps climbing for the whole run. Peak grows roughly
// linearly in elapsed time, which is the positive-drift regime.
func saturatedStream(rounds int) []Event {
	events := make([]Event, 0, 3*rounds)
	var ts int64
	for i := 0; i < rounds; i++ {
		ts += 100_000 // 100ms per round
		events = append(events,
			Event{Timestamp: ts, Type: Arrival, RequestID: "a"},
			Event{Timestamp: ts + 10_000, Type: Arrival, RequestID: "b"},
			Event{Timestamp: ts + 20_000, Type: Completion, RequestID: "a", LatencyMs: 50})
	}
	return events
}

// healthyStream models a server keeping up: every arrival completes, so the backlog
// oscillates in a bounded range while elapsed time grows without bound. The peak
// levels off almost immediately, which is the negative-drift regime.
func healthyStream(rounds int) []Event {
	events := make([]Event, 0, 2*rounds)
	var ts int64
	for i := 0; i < rounds; i++ {
		ts += 100_000
		events = append(events,
			Event{Timestamp: ts, Type: Arrival, RequestID: "a"},
			Event{Timestamp: ts + 50_000, Type: Completion, RequestID: "a", LatencyMs: 50})
	}
	return events
}

// peakRateWith returns a detector whose parameters differ from the defaults only as
// named, so each test varies one thing.
func peakRateWith(threshold float64, minObs, consecK int) Detector {
	return newPeakRateDetector(peakRateConfig{
		Threshold:        threshold,
		MinObservations:  minObs,
		ConsecutiveK:     consecK,
		OverloadMultiple: defaultPeakRateOverloadMultiple,
	})
}

// streamLevels drives a detector over events and returns the verdict after each.
func streamLevels(d Detector, events []Event) []Level {
	d.Reset()
	out := make([]Level, 0, len(events))
	for _, e := range events {
		d.Observe(e)
		out = append(out, d.Detect().Level)
	}
	return out
}

// firedAnywhere reports whether any verdict left STABLE.
func firedAnywhere(levels []Level) bool {
	for _, l := range levels {
		if l != Stable {
			return true
		}
	}
	return false
}

// --- The core discrimination contract ------------------------------------

// A server whose backlog high-water mark keeps growing must be reported saturated;
// one whose backlog levels off must settle to STABLE and STAY there. This is the
// whole claim of the detector, stated over observable verdicts on traffic of a known
// regime.
//
// The healthy assertion is about the SETTLED verdict, not the opening one: R_t is
// large early by construction (small elapsed time), so every run begins in a
// transient. The detector's own contract is that a bounded backlog drives R_t down
// and keeps it down -- see TestPeakRate_SettlesAndStaysSettledOnHealthyTraffic for
// the transient itself.
func TestPeakRate_DiscriminatesGrowingFromLevellingBacklog(t *testing.T) {
	d := peakRateWith(defaultPeakRateThreshold, 20, defaultPeakRateConsecutiveK)

	saturated := streamLevels(d, saturatedStream(200))
	if !firedAnywhere(saturated) {
		t.Error("a backlog growing for the whole run was never reported saturated")
	}
	// And it must STAY saturated: a growing backlog does not become healthy.
	if tail := saturated[len(saturated)/2:]; firedFraction(tail) < 1.0 {
		t.Errorf("a growing backlog did not stay saturated over the run's second half (fired on %.0f%% of it)", 100*firedFraction(tail))
	}

	healthy := streamLevels(d, healthyStream(400))
	if tail := healthy[len(healthy)/2:]; firedAnywhere(tail) {
		t.Errorf("a bounded backlog was still reported saturated in the run's second half: %v", firstNonStable(tail))
	}
}

// On healthy traffic the verdict must settle to STABLE and never return to a
// saturated level. That is the substantive property: the opening transient is
// expected (R_t is large when elapsed time is small), but a bounded backlog must not
// keep re-firing afterwards.
//
// This is also the contract that documents why min_observations is part of the
// algorithm rather than input validation: it exists to hold the verdict during
// exactly this transient.
func TestPeakRate_SettlesAndStaysSettledOnHealthyTraffic(t *testing.T) {
	levels := streamLevels(peakRateWith(defaultPeakRateThreshold, 20, defaultPeakRateConsecutiveK), healthyStream(400))

	settledAt := -1
	for i, l := range levels {
		if l == Stable {
			if settledAt < 0 {
				settledAt = i
			}
		} else {
			settledAt = -1 // a saturated verdict restarts the search
		}
	}
	if settledAt < 0 {
		t.Fatal("healthy traffic never settled to a lasting STABLE verdict")
	}
	// Once settled it must stay settled for the rest of the run.
	for i := settledAt; i < len(levels); i++ {
		if levels[i] != Stable {
			t.Fatalf("healthy traffic re-fired at event %d after settling at %d", i, settledAt)
		}
	}
	// The transient must be bounded, not most of the run -- otherwise "settles"
	// would be trivially satisfiable by settling on the final event.
	if settledAt > len(levels)/2 {
		t.Errorf("healthy traffic took %d of %d events to settle; the transient should be a small fraction of the run", settledAt, len(levels))
	}
}

// The detector must reach the more severe level on more severe traffic, so the two
// saturated levels carry information rather than being interchangeable.
func TestPeakRate_ReachesOverloadedOnlyBeyondTheOverloadBoundary(t *testing.T) {
	events := saturatedStream(200)

	// A low threshold puts the OVERLOADED boundary within reach of this stream.
	low := streamLevels(peakRateWith(0.05, 20, 3), events)
	// A threshold just under the observed statistic fires, but its OVERLOADED
	// boundary (overload_multiple x threshold) stays out of reach.
	high := streamLevels(peakRateWith(6.0, 20, 3), events)

	if !containsLevel(low, Overloaded) {
		t.Error("a low threshold never reached OVERLOADED on a growing backlog")
	}
	if containsLevel(high, Overloaded) {
		t.Error("a high threshold reached OVERLOADED even though the statistic never crossed overload_multiple x threshold")
	}
	if !containsLevel(high, Backlogged) {
		t.Error("a high threshold that fires should report BACKLOGGED; it reported neither saturated level")
	}
}

// --- The knob contracts (sensitivity vs false alarms) --------------------

// threshold is the primary false-alarm dial: raising it must never produce a MORE
// severe verdict on any event of a fixed stream. Without this, the knob cannot be
// solved for a target false-alarm rate.
func TestPeakRate_ThresholdIsMonotone(t *testing.T) {
	events := saturatedStream(200)
	severity := map[Level]int{Stable: 0, Backlogged: 1, Overloaded: 2}
	ladder := []float64{0.05, 0.2, 0.5, 2.0, 8.0, 32.0}

	base := streamLevels(peakRateWith(ladder[0], 20, 3), events)
	if !firedAnywhere(base) {
		t.Fatalf("fixture never fires at threshold %v; the monotonicity assertion would be vacuous", ladder[0])
	}

	prev := base
	for _, thr := range ladder[1:] {
		cur := streamLevels(peakRateWith(thr, 20, 3), events)
		for i := range cur {
			if severity[cur[i]] > severity[prev[i]] {
				t.Errorf("threshold=%v: event %d escalated %v -> %v as the threshold ROSE", thr, i, prev[i], cur[i])
			}
		}
		prev = cur
	}

	// The ladder must span a real change, or nesting proves nothing.
	if firedAnywhere(prev) {
		t.Errorf("the highest threshold %v still fires; the ladder does not span the knob's range", ladder[len(ladder)-1])
	}
}

// min_observations is the second sensitivity dial: raising it defers the verdict, so
// it can only suppress firing on a fixed stream, never induce it.
func TestPeakRate_MinObservationsOnlySuppresses(t *testing.T) {
	events := saturatedStream(200)
	severity := map[Level]int{Stable: 0, Backlogged: 1, Overloaded: 2}

	prev := streamLevels(peakRateWith(defaultPeakRateThreshold, 10, 3), events)
	if !firedAnywhere(prev) {
		t.Fatal("fixture never fires at min_observations=10; the assertion would be vacuous")
	}
	for _, m := range []int{50, 200, 400} {
		cur := streamLevels(peakRateWith(defaultPeakRateThreshold, m, 3), events)
		for i := range cur {
			if severity[cur[i]] > severity[prev[i]] {
				t.Errorf("min_observations=%d: event %d escalated %v -> %v as the gate ROSE", m, i, prev[i], cur[i])
			}
		}
		prev = cur
	}
}

// consecutive_k debounces: raising it can only delay or suppress firing.
func TestPeakRate_ConsecutiveKOnlyDelaysFiring(t *testing.T) {
	events := saturatedStream(200)

	firstFire := func(k int) int {
		levels := streamLevels(peakRateWith(defaultPeakRateThreshold, 20, k), events)
		for i, l := range levels {
			if l != Stable {
				return i
			}
		}
		return len(levels) // never fired
	}

	prev := firstFire(1)
	if prev == len(events) {
		t.Fatal("fixture never fires at consecutive_k=1; the assertion would be vacuous")
	}
	for _, k := range []int{2, 5, 20} {
		cur := firstFire(k)
		if cur < prev {
			t.Errorf("consecutive_k=%d fired EARLIER (event %d) than a smaller k (event %d)", k, cur, prev)
		}
		prev = cur
	}
}

// The two dials must be independently usable: tightening either one alone must be
// able to silence a stream that fires at the defaults. This is what lets an operator
// trade sensitivity against false alarms without changing the other knob.
func TestPeakRate_EitherDialAloneCanSilenceAStream(t *testing.T) {
	events := saturatedStream(200)

	if !firedAnywhere(streamLevels(peakRateWith(defaultPeakRateThreshold, 20, 3), events)) {
		t.Fatal("fixture does not fire at the defaults; nothing to silence")
	}
	if firedAnywhere(streamLevels(peakRateWith(1e6, 20, 3), events)) {
		t.Error("raising threshold alone did not silence the stream")
	}
	if firedAnywhere(streamLevels(peakRateWith(defaultPeakRateThreshold, len(events)+1, 3), events)) {
		t.Error("raising min_observations alone did not silence the stream")
	}
}

// --- Invariance and purity ----------------------------------------------

// The statistic is backlog per unit time, so a server that is uniformly c-times
// faster (all timestamps divided by c) has its statistic multiplied by c. What must
// NOT change is the ORDERING: whichever of two streams has the larger statistic must
// keep that relation under any common rescaling, so a calibrated threshold stays
// meaningful when the whole timeline is stretched.
func TestPeakRate_RescalingTimePreservesOrdering(t *testing.T) {
	scale := func(events []Event, c float64) []Event {
		out := make([]Event, len(events))
		for i, e := range events {
			e.Timestamp = int64(float64(e.Timestamp) * c)
			out[i] = e
		}
		return out
	}
	statOf := func(events []Event) float64 {
		d := peakRateWith(defaultPeakRateThreshold, 20, 3)
		d.Reset()
		var last float64
		for _, e := range events {
			d.Observe(e)
			last = d.Detect().Signals["peak_rate"]
		}
		return last
	}

	sat, healthy := saturatedStream(200), healthyStream(200)
	for _, c := range []float64{0.5, 1, 2, 10, 1000} {
		s, h := statOf(scale(sat, c)), statOf(scale(healthy, c))
		if !(s > h) {
			t.Errorf("time scale x%v: the growing-backlog statistic %v is not above the bounded one %v; the ordering did not survive rescaling", c, s, h)
		}
	}
}

// Detect is a query: repeated calls without an intervening Observe must agree, and
// interleaving extra Detect calls must not change the verdict sequence. The second
// half is the part that matters -- it fails for any implementation that advances
// state inside Detect.
func TestPeakRate_DetectIsAPureQuery(t *testing.T) {
	events := saturatedStream(100)

	d := peakRateWith(defaultPeakRateThreshold, 20, 3)
	d.Reset()
	sparse := make([]Level, 0, len(events))
	for _, e := range events {
		d.Observe(e)
		sparse = append(sparse, d.Detect().Level)
	}

	// Same stream, but Detect is called several times per event.
	d.Reset()
	chatty := make([]Level, 0, len(events))
	for _, e := range events {
		d.Observe(e)
		first := d.Detect()
		for i := 0; i < 4; i++ {
			if got := d.Detect(); got.Level != first.Level || got.Score != first.Score {
				t.Fatalf("repeated Detect disagreed: %v then %v", first, got)
			}
		}
		chatty = append(chatty, first.Level)
	}

	for i := range sparse {
		if sparse[i] != chatty[i] {
			t.Fatalf("event %d: verdict depends on how often Detect was called (%v vs %v)", i, sparse[i], chatty[i])
		}
	}
}

// Reset must restore the initial state, including the breach streak, so a detector
// reused across replay legs cannot inherit a verdict from the previous leg. Both
// drivers call Reset before each leg.
func TestPeakRate_ResetClearsAccumulatedState(t *testing.T) {
	d := peakRateWith(defaultPeakRateThreshold, 20, 3)

	if !firedAnywhere(streamLevels(d, saturatedStream(200))) {
		t.Fatal("fixture never fires; the reset assertion would be vacuous")
	}
	// After Reset, EVERY leg must behave exactly as it does on a fresh detector --
	// the previous leg must not leak in through the peak, the timestamps, or the
	// breach streak. Comparing against a fresh detector (rather than asserting
	// "never fires") keeps the opening transient out of the assertion.
	//
	// The saturated leg is the one that exercises the breach streak: it leaves the
	// counter high, so a Reset that forgets to zero it would fire the next leg
	// early. A healthy-only comparison cannot see that, because the streak is
	// already zero there.
	for _, leg := range []struct {
		name   string
		events []Event
	}{
		{"saturated after saturated", saturatedStream(200)},
		{"healthy after saturated", healthyStream(400)},
		{"saturated after healthy", saturatedStream(200)},
	} {
		reused := streamLevels(d, leg.events)
		fresh := streamLevels(peakRateWith(defaultPeakRateThreshold, 20, 3), leg.events)
		for i := range fresh {
			if reused[i] != fresh[i] {
				t.Fatalf("%s: event %d: a reused detector gave %v where a fresh one gave %v; state leaked across Reset",
					leg.name, i, reused[i], fresh[i])
			}
		}
	}
	// And the configuration must survive Reset: the same saturated stream must still
	// fire on a third leg.
	if !firedAnywhere(streamLevels(d, saturatedStream(200))) {
		t.Error("the detector stopped firing on a growing backlog after Reset; configuration was lost")
	}
}

// --- Degenerate input (R20) ---------------------------------------------

// Degenerate streams must yield STABLE with a populated Signals map and no panic.
func TestPeakRate_DegenerateInputIsStable(t *testing.T) {
	for _, tc := range []struct {
		name   string
		events []Event
	}{
		{"no events", nil},
		{"one arrival", []Event{{Timestamp: 1_000, Type: Arrival, RequestID: "a"}}},
		// NOTE: an arrivals-only stream is NOT degenerate-healthy -- it is an
		// unboundedly growing backlog, so reporting saturation is correct. It is
		// covered by TestPeakRate_ArrivalsWithoutCompletionsIsSaturation instead.
		{"completions only", func() []Event {
			var out []Event
			for i := 0; i < 100; i++ {
				out = append(out, Event{Timestamp: int64(i) * 1000, Type: Completion, RequestID: "a", LatencyMs: 1})
			}
			return out
		}()},
		{"all events share one timestamp", func() []Event {
			var out []Event
			for i := 0; i < 100; i++ {
				out = append(out,
					Event{Timestamp: 7, Type: Arrival, RequestID: "a"},
					Event{Timestamp: 7, Type: Completion, RequestID: "a", LatencyMs: 1})
			}
			return out
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := NewPeakRateDetector()
			d.Reset()
			for _, e := range tc.events {
				d.Observe(e)
			}
			r := d.Detect()
			if r.Level != Stable {
				t.Errorf("expected STABLE, got %v", r.Level)
			}
			if r.Signals == nil {
				t.Error("Signals map is nil; the trace cannot explain the verdict")
			}
			if math.IsNaN(r.Score) || math.IsInf(r.Score, 0) {
				t.Errorf("non-finite Score %v", r.Score)
			}
		})
	}
}

// Whatever parameters a caller supplies -- including adversarial magnitudes that
// bypass YAML validation -- the emitted Result must stay finite and its Level must
// stay consistent with its Score. A parameter that drove the score denominator to
// zero would decouple the two.
func TestPeakRate_ResultStaysCoherentForAnyParameters(t *testing.T) {
	events := saturatedStream(200)
	for _, thr := range []float64{
		0, -1, math.NaN(), math.Inf(1), math.Inf(-1),
		math.SmallestNonzeroFloat64, 1e-300, 0.5, 1e300, math.MaxFloat64,
	} {
		for _, om := range []float64{0, -1, math.NaN(), math.Inf(1), 1, 3, 1e300} {
			d := newPeakRateDetector(peakRateConfig{
				Threshold: thr, MinObservations: 20, ConsecutiveK: 3, OverloadMultiple: om,
			})
			d.Reset()
			for _, e := range events {
				d.Observe(e)
				r := d.Detect()
				if math.IsNaN(r.Score) || math.IsInf(r.Score, 0) {
					t.Fatalf("threshold=%v overload_multiple=%v: non-finite Score %v", thr, om, r.Score)
				}
				if r.Score < 0 || r.Score > 1 {
					t.Fatalf("threshold=%v overload_multiple=%v: Score %v outside [0,1]", thr, om, r.Score)
				}
				if r.Level == Overloaded && r.Score != 1.0 {
					t.Fatalf("threshold=%v overload_multiple=%v: OVERLOADED with Score %v, want the 1.0 cap", thr, om, r.Score)
				}
			}
		}
	}
}

// --- helpers ------------------------------------------------------------

func containsLevel(levels []Level, want Level) bool {
	for _, l := range levels {
		if l == want {
			return true
		}
	}
	return false
}

func firstNonStable(levels []Level) string {
	for i, l := range levels {
		if l != Stable {
			return l.String() + " at event " + strconv.Itoa(i)
		}
	}
	return "none"
}

// A stream of arrivals with no completions is the most extreme saturation there is:
// nothing ever finishes. The detector must say so rather than treating the missing
// completions as an absence of evidence (R20 -- the degenerate case is often the one
// that matters most).
func TestPeakRate_ArrivalsWithoutCompletionsIsSaturation(t *testing.T) {
	var events []Event
	for i := 0; i < 200; i++ {
		events = append(events, Event{Timestamp: int64(i) * 100_000, Type: Arrival, RequestID: "a"})
	}
	levels := streamLevels(NewPeakRateDetector(), events)
	if !containsLevel(levels, Overloaded) {
		t.Errorf("a stream where nothing ever completes was not reported OVERLOADED; got %v", firstNonStable(levels))
	}
}

// firedFraction is the fraction of verdicts that are not STABLE.
func firedFraction(levels []Level) float64 {
	if len(levels) == 0 {
		return 0
	}
	n := 0
	for _, l := range levels {
		if l != Stable {
			n++
		}
	}
	return float64(n) / float64(len(levels))
}
