package harness

import (
	"math"
	"testing"
)

const absolutesPath = "/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/inferencex_absolutes.json"

// The measured absolute curves must reproduce the artifact's own measured RELATIVES. This is the
// check that the InferenceX rows loaded here are the same rows NVIDIA scored AISimulate against.
// Without it, a mape computed from these absolutes would be against a different measurement, and
// the comparison with AISimulate would silently stop being apples to apples.
func TestAbsolutesReproduceTheArtifactsOwnRelatives(t *testing.T) {
	set, err := LoadAbsolutes(absolutesPath)
	if err != nil {
		t.Skipf("measured absolutes unavailable: %v", err)
	}
	c, err := LoadCorpus(
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/aisimulate_e2e.json")
	if err != nil {
		t.Skip(err)
	}
	checked, missing := 0, 0
	for i := range c.Sweeps {
		sw := c.Sweeps[i]
		a := set.For(sw)
		if a == nil {
			missing++
			continue
		}
		for _, m := range AllMetrics {
			anchorConc := sw.Points[0].Concurrency
			anchor, ok := a.At(m, anchorConc)
			if !ok {
				t.Errorf("%s %s: no measured %s at the anchor concurrency %d",
					sw.Scenario, sw.Label, m, anchorConc)
				continue
			}
			for _, p := range sw.Points {
				got, ok := a.At(m, p.Concurrency)
				if !ok {
					t.Errorf("%s %s: no measured %s at concurrency %d",
						sw.Scenario, sw.Label, m, p.Concurrency)
					continue
				}
				want, _, _ := p.MetricOf(m)
				if want <= 0 {
					continue
				}
				// The corpus's relative and the absolutes' own ratio must agree.
				if rel := got / anchor; math.Abs(rel-want)/want > 0.005 {
					t.Errorf("%s %s %s c=%d: absolute ratio %.4f vs the artifact's relative "+
						"%.4f -- these are not the same measurement",
						sw.Scenario, sw.Label, m, p.Concurrency, rel, want)
				}
				checked++
			}
		}
	}
	if missing > 0 {
		t.Logf("%d sweep(s) have no measured absolutes and are excluded from mape", missing)
	}
	if checked == 0 {
		t.Fatal("no points checked")
	}
	t.Logf("%d point-metric pairs reproduce the artifact's relatives", checked)
}

// An absolute must be a plausible latency in microseconds, not a seconds value left unconverted.
// The dump stores seconds; a missed conversion would make every mape wrong by 1e6 while leaving
// every ratio-based figure untouched, so no shape test would catch it.
func TestAbsolutesAreMicrosecondsNotSeconds(t *testing.T) {
	set, err := LoadAbsolutes(absolutesPath)
	if err != nil {
		t.Skipf("measured absolutes unavailable: %v", err)
	}
	for i := range set.Anchors {
		a := &set.Anchors[i]
		for conc, v := range a.TPOTUs {
			// Time per output token: hundreds of microseconds to tens of milliseconds. One
			// microsecond per token would be faster than any memory system; ten seconds is not a
			// serving deployment.
			if v < 100 || v > 10_000_000 {
				t.Errorf("%s %s: TPOT at c=%s is %.3f us, outside any plausible range",
					a.Scenario, a.Label, conc, v)
			}
		}
		for conc, v := range a.TTFTUs {
			if v < 100 || v > 600_000_000 {
				t.Errorf("%s %s: TTFT at c=%s is %.3f us, outside any plausible range",
					a.Scenario, a.Label, conc, v)
			}
		}
	}
}

// Every loaded sweep must record which InferenceX run it came from, so a figure traces to a row.
func TestAbsolutesCarryTheirProvenance(t *testing.T) {
	set, err := LoadAbsolutes(absolutesPath)
	if err != nil {
		t.Skipf("measured absolutes unavailable: %v", err)
	}
	if set.SourceURL == "" || set.ReleaseTag == "" {
		t.Errorf("the set names no source URL or release tag: %q / %q",
			set.SourceURL, set.ReleaseTag)
	}
	for i := range set.Anchors {
		a := &set.Anchors[i]
		if a.RunDate == "" || a.ConfigID == 0 {
			t.Errorf("%s %s: no run date or config id", a.Scenario, a.Label)
		}
		if a.Agreement > 0.005 {
			t.Errorf("%s %s: selected run disagrees with the artifact by %.3f%%",
				a.Scenario, a.Label, a.Agreement*100)
		}
	}
}
