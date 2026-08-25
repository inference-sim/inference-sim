package saturation

import (
	"math"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
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

// spikeThenDrainStream is the fixture that distinguishes a HIGH-WATER MARK from the
// instantaneous backlog: a large burst drives the backlog up, it then drains
// completely, and a long healthy tail follows.
//
// The two implementations diverge sharply here. Tracking the running maximum keeps
// R_t elevated for the whole tail (the peak is remembered), while tracking the
// current backlog would let it collapse the moment the queue drains. Without a
// fixture where the peak LAGS the current backlog, no assertion can tell the two
// apart -- and the running max is the detector's defining feature.
func spikeThenDrainStream(spike, tail int) []Event {
	events := make([]Event, 0, 2*(spike+tail))
	var ts int64

	// Burst: `spike` arrivals in quick succession, so the backlog climbs to `spike`.
	for i := 0; i < spike; i++ {
		ts += 1_000
		events = append(events, Event{Timestamp: ts, Type: Arrival, RequestID: "s"})
	}
	// Drain: every one of them completes, so the backlog returns to zero.
	for i := 0; i < spike; i++ {
		ts += 1_000
		events = append(events, Event{Timestamp: ts, Type: Completion, RequestID: "s", LatencyMs: 10})
	}
	// Healthy tail: one request at a time, arriving and completing, for a long while.
	for i := 0; i < tail; i++ {
		ts += 100_000
		events = append(events, Event{Timestamp: ts, Type: Arrival, RequestID: "h"})
		ts += 50_000
		events = append(events, Event{Timestamp: ts, Type: Completion, RequestID: "h", LatencyMs: 50})
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
				// The converse: a capped Score means the magnitude cleared the
				// OVERLOADED boundary, so the level must be OVERLOADED unless the
				// debounce streak is simply not satisfied yet. Level and Score must
				// never disagree about the MAGNITUDE.
				if r.Score == 1.0 && r.Level == Backlogged {
					t.Fatalf("threshold=%v overload_multiple=%v: Score reached the 1.0 cap but Level is BACKLOGGED; the band and the score denominator disagree",
						thr, om)
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

// saturatedRequests builds RequestMetrics whose reconstructed event stream is a
// growing backlog: requests arrive faster than they complete, and each takes longer
// than the last. Used for the Bank.Run path, which consumes RequestMetrics.
func saturatedRequests(n int) []sim.RequestMetrics {
	out := make([]sim.RequestMetrics, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, sim.RequestMetrics{
			ID:        "request_" + strconv.Itoa(i),
			ArrivedAt: float64(i) * 0.01,   // 100/s arrivals
			E2E:       float64(500 + i*30), // ms, growing: completions fall behind
		})
	}
	return out
}

// Under the bank, tuning peak-rate must not perturb any peer detector: selection
// and configuration filter WHICH detectors run and how they band, never HOW any of
// them sees traffic (INV-6).
//
// The vacuity guard at the end is essential -- without it a threshold that happened
// to change nothing would let this pass while proving nothing.
func TestPeakRate_TuningDoesNotPerturbPeerDetectors(t *testing.T) {
	reqs := saturatedRequests(300)
	huge := 1e6

	collect := func(cfg SaturationConfig) map[string][]Level {
		sink := NewInMemoryCollector()
		bank, err := NewBank(AllDetectorNames(), cfg, sink)
		if err != nil {
			t.Fatalf("NewBank: %v", err)
		}
		if err := bank.Run(reqs); err != nil {
			t.Fatalf("Run: %v", err)
		}
		out := map[string][]Level{}
		for _, r := range sink.Records() {
			out[r.Detector] = append(out[r.Detector], r.Result.Level)
		}
		return out
	}

	untuned := collect(SaturationConfig{})
	tuned := collect(SaturationConfig{PeakRate: &PeakRateBlock{Threshold: &huge}})

	for _, peer := range []string{"composite", "threshold", "backlog-drift"} {
		a, b := untuned[peer], tuned[peer]
		if len(a) == 0 {
			t.Fatalf("%s produced no records; the isolation assertion would be vacuous", peer)
		}
		if len(a) != len(b) {
			t.Fatalf("%s: record count changed when peak-rate was tuned (%d vs %d)", peer, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Errorf("%s: event %d changed %v -> %v when a PEER detector was tuned", peer, i, a[i], b[i])
			}
		}
	}

	changed := false
	for i := range untuned["peak-rate"] {
		if untuned["peak-rate"][i] != tuned["peak-rate"][i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("tuning peak-rate changed none of its own verdicts; the isolation assertion is vacuous")
	}
}

// The documented recovery property: because the numerator is an ALL-TIME high-water
// mark, a detector that fired during a transient keeps firing until enough further
// work completes to pull R_t back under the threshold. That is a real limitation, so
// it is pinned as a contract rather than left implicit -- and it must be BOUNDED: a
// healthy tail must eventually clear the verdict.
//
// Stated behaviorally: after a burst, feeding healthy traffic must return the verdict
// to STABLE, and raising the threshold must clear it SOONER (the knob controls
// recovery time as well as sensitivity).
func TestPeakRate_RecoversAfterATransientAndTheThresholdControlsHowFast(t *testing.T) {
	// A burst that drives the peak up, then a long healthy tail.
	burst := saturatedStream(60)
	lastTs := burst[len(burst)-1].Timestamp
	tail := healthyStream(1200)
	for i := range tail {
		tail[i].Timestamp += lastTs
	}
	stream := append(append([]Event{}, burst...), tail...)

	clearedAt := func(threshold float64) int {
		levels := streamLevels(peakRateWith(threshold, 20, 3), stream)
		// The last index at which the verdict was still saturated; everything after
		// it is STABLE.
		last := -1
		for i, l := range levels {
			if l != Stable {
				last = i
			}
		}
		if last == len(levels)-1 {
			return -1 // never recovered
		}
		return last + 1
	}

	low := clearedAt(0.5)
	high := clearedAt(4.0)

	if low < 0 {
		t.Fatal("the verdict never returned to STABLE after the burst; recovery is unbounded")
	}
	if high < 0 {
		t.Fatal("the verdict never returned to STABLE even at a high threshold")
	}
	if high >= low {
		t.Errorf("raising the threshold did not shorten recovery: cleared at event %d (threshold 4.0) vs %d (threshold 0.5)", high, low)
	}
}

// The detector must track the backlog's HIGH-WATER MARK, not its instantaneous
// value. After a large burst has fully drained, the statistic must still reflect the
// peak that was reached -- that is what makes R_t a saturation detector rather than
// a backlog gauge, and it is the property the whole asymptotic argument rests on.
//
// Stated behaviorally: on a stream whose backlog spikes and then drains to zero, the
// verdict must remain saturated well past the point where the queue is empty, and
// the reported statistic must stay far above what the (zero) current backlog implies.
func TestPeakRate_TracksTheHighWaterMarkNotTheCurrentBacklog(t *testing.T) {
	events := spikeThenDrainStream(200, 100)

	d := peakRateWith(defaultPeakRateThreshold, 20, 3)
	d.Reset()

	type sample struct {
		level    Level
		stat     float64
		inFlight float64
		peak     float64
	}
	var samples []sample
	for _, e := range events {
		d.Observe(e)
		r := d.Detect()
		samples = append(samples, sample{r.Level, r.Signals["peak_rate"], r.Signals["in_flight"], r.Signals["peak_backlog"]})
	}

	// Look at the healthy tail, well after the burst has drained.
	tail := samples[len(samples)*3/4:]

	drained := 0
	for _, s := range tail {
		if s.inFlight <= 1 {
			drained++
		}
	}
	if drained == 0 {
		t.Fatal("the backlog never drained in the tail; this fixture cannot distinguish a high-water mark from the current backlog")
	}

	// The remembered peak must exceed the drained backlog by a wide margin --
	// otherwise the detector is tracking the instantaneous value.
	for i, s := range tail {
		if s.peak <= s.inFlight+1 {
			t.Fatalf("tail sample %d: reported peak %v is not above the current backlog %v; the high-water mark is not being tracked",
				i, s.peak, s.inFlight)
		}
	}

	// And the verdict must still be saturated: the run DID saturate, even though the
	// queue is now empty.
	tailLevels := make([]Level, len(tail))
	for i, s := range tail {
		tailLevels[i] = s.level
	}
	if !firedAnywhere(tailLevels) {
		t.Error("after a large burst drained, the verdict returned to STABLE immediately; the peak was forgotten")
	}
}

// An out-of-order timestamp must not silence the detector. Observe is a public
// interface method, so a caller need not deliver events in time order; tracking
// "the first timestamp" rather than the minimum would pin the elapsed span at zero
// and report STABLE on an unboundedly growing backlog forever -- a silent failure,
// which is strictly worse than a wrong answer because it looks like a healthy run.
func TestPeakRate_OutOfOrderTimestampsDoNotSilenceIt(t *testing.T) {
	d := NewPeakRateDetector()
	d.Reset()

	// The largest timestamp arrives FIRST, then 200 arrivals at earlier times.
	d.Observe(Event{Timestamp: 10_000_000, Type: Arrival, RequestID: "late"})
	for i := 0; i < 200; i++ {
		d.Observe(Event{Timestamp: int64(i) * 1_000, Type: Arrival, RequestID: "early"})
	}

	r := d.Detect()
	if r.Signals["elapsed_sec"] <= 0 {
		t.Fatalf("elapsed span is %v after events spanning 10s; the detector cannot form a verdict", r.Signals["elapsed_sec"])
	}
	if r.Level == Stable {
		t.Errorf("201 arrivals with no completions was reported STABLE (statistic %v); an unbounded backlog must not read as healthy",
			r.Signals["peak_rate"])
	}
}

// An unrecognized event type must not change the verdict at all. Advancing the
// elapsed span for an event that does not move the backlog would shrink the
// statistic while leaving the breach streak frozen, so the reported level would
// contradict the reported statistic.
func TestPeakRate_UnknownEventTypeIsFullyIgnored(t *testing.T) {
	d := peakRateWith(defaultPeakRateThreshold, 20, 3)
	d.Reset()
	for _, e := range saturatedStream(100) {
		d.Observe(e)
	}
	before := d.Detect()

	ts := int64(100_000_000)
	for i := 0; i < 1000; i++ {
		ts += 1_000_000
		d.Observe(Event{Timestamp: ts, Type: EventType(99), RequestID: "junk"})
	}
	after := d.Detect()

	if after.Level != before.Level {
		t.Errorf("1000 unknown-type events changed the level %v -> %v", before.Level, after.Level)
	}
	if after.Signals["peak_rate"] != before.Signals["peak_rate"] {
		t.Errorf("1000 unknown-type events changed the statistic %v -> %v", before.Signals["peak_rate"], after.Signals["peak_rate"])
	}
	if after.Signals["elapsed_sec"] != before.Signals["elapsed_sec"] {
		t.Errorf("1000 unknown-type events advanced the elapsed span %v -> %v", before.Signals["elapsed_sec"], after.Signals["elapsed_sec"])
	}
}

// Whenever the detector reports a saturated level, its reported statistic must be
// above the reported threshold -- the verdict and the evidence for it must agree.
// This is the property that a stale streak (a verdict outliving the statistic that
// justified it) violates.
func TestPeakRate_SaturatedVerdictsAlwaysExceedTheReportedThreshold(t *testing.T) {
	for _, events := range [][]Event{
		saturatedStream(200),
		healthyStream(400),
		spikeThenDrainStream(200, 100),
	} {
		d := peakRateWith(defaultPeakRateThreshold, 20, 3)
		d.Reset()
		checked := 0
		for _, e := range events {
			d.Observe(e)
			r := d.Detect()
			if r.Level == Stable {
				continue
			}
			checked++
			if r.Signals["peak_rate"] <= r.Signals["threshold"] {
				t.Fatalf("reported %v while the statistic %v is not above the threshold %v; the verdict outlived its evidence",
					r.Level, r.Signals["peak_rate"], r.Signals["threshold"])
			}
		}
		if checked == 0 {
			t.Error("no saturated verdict was examined; the assertion was vacuous for this stream")
		}
	}
}

// The separation between healthy and overloaded traffic must WIDEN with the
// observation horizon. This is the asymptotic property the whole design rests on
// (the statistic is defined by a limit), and it is why min_observations is part of
// the algorithm rather than input validation: at a short horizon the two regimes are
// genuinely hard to tell apart, because the healthy server's peak has not levelled
// off yet.
//
// Asserted as a trend rather than against the measured figures in the doc comment,
// so the test states the law instead of pinning one apparatus's numbers.
func TestPeakRate_SeparationWidensWithTheHorizon(t *testing.T) {
	statAtEnd := func(events []Event) float64 {
		d := peakRateWith(defaultPeakRateThreshold, 20, 3)
		d.Reset()
		var last float64
		for _, e := range events {
			d.Observe(e)
			last = d.Detect().Signals["peak_rate"]
		}
		return last
	}

	var prev float64
	for _, rounds := range []int{50, 100, 200, 400, 800} {
		sat := statAtEnd(saturatedStream(rounds))
		healthy := statAtEnd(healthyStream(rounds))
		if healthy <= 0 {
			t.Fatalf("rounds=%d: healthy statistic is %v; cannot form a ratio", rounds, healthy)
		}
		sep := sat / healthy
		if sep <= 1 {
			t.Fatalf("rounds=%d: separation %v does not favour the saturated stream at all", rounds, sep)
		}
		if prev > 0 && sep < prev {
			t.Errorf("rounds=%d: separation NARROWED to %.2fx from %.2fx at the shorter horizon; the statistic should discriminate better with more observation, not worse",
				rounds, sep, prev)
		}
		prev = sep
	}
}

// The knobs' exact semantics, pinned at their boundaries. Ordering tests (a bigger
// knob fires no more) cannot see an off-by-one, because both sides shift together --
// so the absolute meaning of each knob needs its own assertion.
//
// Constructed so the statistic lands EXACTLY on the threshold: a single arrival
// gives peak=1, and choosing the elapsed span makes peak/elapsed exactly 1.0.
func TestPeakRate_KnobBoundarySemantics(t *testing.T) {
	// A stream whose statistic is exactly 1.0 backlog/second and stays there:
	// one arrival at t=0, then arrivals and completions that hold peak at 2 while
	// elapsed grows to 2s -> R_t = 1.0.
	exactly := func(n int) []Event {
		out := []Event{{Timestamp: 0, Type: Arrival, RequestID: "a"}}
		var ts int64
		for i := 0; i < n; i++ {
			ts += 1_000_000 // 1s steps
			out = append(out,
				Event{Timestamp: ts, Type: Arrival, RequestID: "b"},
				Event{Timestamp: ts, Type: Completion, RequestID: "b", LatencyMs: 1})
		}
		return out
	}

	t.Run("threshold is strict: a statistic EQUAL to the threshold does not fire", func(t *testing.T) {
		events := exactly(4)
		// Find the exact statistic at the end, then use it AS the threshold.
		probe := peakRateWith(1e-6, 1, 1)
		probe.Reset()
		var stat float64
		for _, e := range events {
			probe.Observe(e)
			stat = probe.Detect().Signals["peak_rate"]
		}
		if stat <= 0 {
			t.Fatalf("probe statistic is %v; cannot set an exact-boundary threshold", stat)
		}

		atBoundary := streamLevels(peakRateWith(stat, 1, 1), events)
		if atBoundary[len(atBoundary)-1] != Stable {
			t.Errorf("a statistic exactly equal to the threshold fired (%v); the comparison must be strict so `threshold` means 'fire ABOVE this'",
				atBoundary[len(atBoundary)-1])
		}

		// And a hair below the boundary must fire, proving the test is not passing
		// because nothing ever fires.
		justUnder := streamLevels(peakRateWith(stat*0.99, 1, 1), events)
		if justUnder[len(justUnder)-1] == Stable {
			t.Fatalf("a threshold 1%% below the statistic did not fire; the boundary assertion above is vacuous")
		}
	})

	t.Run("min_observations is inclusive: exactly that many events is enough", func(t *testing.T) {
		events := exactly(20)
		// With min_observations == the event count, the final event must be eligible
		// to produce a verdict (an exclusive gate would need one more).
		levels := streamLevels(peakRateWith(1e-6, len(events), 1), events)
		if levels[len(levels)-1] == Stable {
			t.Errorf("with min_observations equal to the event count the detector never became eligible; the gate must be inclusive (>=)")
		}
		// One more than the count must NOT be eligible.
		levels = streamLevels(peakRateWith(1e-6, len(events)+1, 1), events)
		if levels[len(levels)-1] != Stable {
			t.Errorf("with min_observations ABOVE the event count the detector formed a verdict; the gate is not being applied")
		}
	})

	t.Run("consecutive_k is inclusive: exactly k breaches fires", func(t *testing.T) {
		events := exactly(40)
		// A tiny threshold makes every eligible event a breach, so the first fire
		// index tells us how many breaches k actually requires.
		firstFire := func(k int) int {
			for i, l := range streamLevels(peakRateWith(1e-6, 1, k), events) {
				if l != Stable {
					return i
				}
			}
			return -1
		}
		f1, f2 := firstFire(1), firstFire(2)
		if f1 < 0 || f2 < 0 {
			t.Fatalf("did not fire at k=1 (%d) or k=2 (%d); the assertion would be vacuous", f1, f2)
		}
		// Requiring one more breach must delay firing by exactly one eligible event.
		if f2 != f1+1 {
			t.Errorf("k=2 first fired at event %d and k=1 at %d; requiring one more breach should delay by exactly one event, so consecutive_k counts breaches inclusively", f2, f1)
		}
	})
}

// The OVERLOADED boundary is strict too, and the elapsed-span guard is load-bearing.
// Both are boundary conditions that ordering tests cannot see.
func TestPeakRate_OverloadBoundaryAndElapsedGuard(t *testing.T) {
	t.Run("OVERLOADED is strict: a statistic exactly at the boundary stays BACKLOGGED", func(t *testing.T) {
		// Hold peak at 1 with elapsed growing, so the statistic is a clean value.
		events := []Event{{Timestamp: 0, Type: Arrival, RequestID: "a"}}
		var ts int64
		for i := 0; i < 8; i++ {
			ts += 1_000_000
			events = append(events,
				Event{Timestamp: ts, Type: Arrival, RequestID: "b"},
				Event{Timestamp: ts, Type: Completion, RequestID: "b", LatencyMs: 1})
		}

		probe := peakRateWith(1e-6, 1, 1)
		probe.Reset()
		var stat float64
		for _, e := range events {
			probe.Observe(e)
			stat = probe.Detect().Signals["peak_rate"]
		}

		// overload_multiple x threshold == stat exactly.
		const om = 4.0
		d := newPeakRateDetector(peakRateConfig{
			Threshold: stat / om, MinObservations: 1, ConsecutiveK: 1, OverloadMultiple: om,
		})
		levels := streamLevels(d, events)
		last := levels[len(levels)-1]
		if last == Stable {
			t.Fatalf("the detector did not fire at all; the boundary assertion would be vacuous")
		}
		if last == Overloaded {
			t.Errorf("a statistic exactly AT overload_multiple x threshold reported OVERLOADED; the comparison must be strict so the knob means 'OVERLOADED above this'")
		}
	})

	t.Run("a zero elapsed span cannot produce a verdict", func(t *testing.T) {
		// Every event at the same instant: the backlog is huge but no time has
		// passed, so peak/elapsed is undefined. The detector must decline to judge
		// rather than treat the undefined ratio as evidence.
		var events []Event
		for i := 0; i < 500; i++ {
			events = append(events, Event{Timestamp: 42, Type: Arrival, RequestID: "a"})
		}
		d := peakRateWith(1e-9, 1, 1) // maximally sensitive
		levels := streamLevels(d, events)
		for i, l := range levels {
			if l != Stable {
				t.Fatalf("event %d reported %v with a zero elapsed span; the statistic is undefined there and must not drive a verdict", i, l)
			}
		}
		// Once time advances, the same detector must be willing to fire -- proving
		// the assertion above is about the zero span, not about a dead detector.
		d.Observe(Event{Timestamp: 43, Type: Arrival, RequestID: "a"})
		if d.Detect().Level == Stable {
			t.Error("after time advanced the detector still would not fire on a 501-deep backlog; the guard is suppressing more than the zero-span case")
		}
	})
}
