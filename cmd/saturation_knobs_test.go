package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/saturation"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// writeSatConfig writes a saturation config file and returns its path.
func writeSatConfig(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "sat.yaml")
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return path
}

// risingRequests produces enough completions, with a rising latency profile, for
// composite's quartile filter to validate a trend -- so the detector reaches a
// non-STABLE verdict and the knob has something to suppress.
func risingRequests(n int) []sim.RequestMetrics {
	out := make([]sim.RequestMetrics, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, sim.RequestMetrics{
			ID:        "request_" + string(rune('A'+i%26)) + string(rune('a'+i/26)),
			ArrivedAt: float64(i) * 0.01,
			E2E:       float64(100 + i*40),
		})
	}
	return out
}

// labelsFor drives the whole CLI-facing path -- flag globals -> resolveSaturation
// -> saturationTracer.run -> ReduceAll -- and returns the per-detector final
// labels, exactly as a `blis run` would emit them on stdout.
func labelsFor(t *testing.T, detectors, configPath string, reqs []sim.RequestMetrics) map[string]saturation.Level {
	t.Helper()
	resetSaturationGlobals()
	detectorName = detectors
	saturationConfigPath = configPath

	tracer, err := resolveSaturation()
	if err != nil {
		t.Fatalf("resolveSaturation(%q, %q): %v", detectors, configPath, err)
	}
	if tracer == nil {
		t.Fatalf("resolveSaturation(%q) returned a nil tracer", detectors)
	}
	final, err := tracer.run(reqs)
	if err != nil {
		t.Fatalf("tracer.run: %v", err)
	}
	return final
}

// The knob must actually change the verdict a user sees on stdout when supplied
// via --saturation-config. This is the end-to-end contract: a package-level test
// proves the detector honours a sensitivity, but only this proves the value
// survives YAML parsing, block-ownership, the tracer, and the reducer.
func TestSaturationKnobs_CompositeSensitivityChangesTheReportedLabel(t *testing.T) {
	defer resetSaturationGlobals()
	reqs := risingRequests(80)

	untuned := labelsFor(t, "composite", "", reqs)
	if untuned["composite"] == saturation.Stable {
		t.Fatalf("fixture is already STABLE untuned (%v); the suppression assertion below would be vacuous", untuned)
	}

	// A large sensitivity raises the bar far enough to suppress the alarm.
	tuned := labelsFor(t, "composite", writeSatConfig(t, "composite:\n  sensitivity: 1000.0\n"), reqs)
	if tuned["composite"] != saturation.Stable {
		t.Errorf("a large sensitivity did not suppress the label: untuned=%v tuned=%v", untuned, tuned)
	}
}

// A run WITHOUT --saturation-config must produce exactly the labels it produced
// before the knobs existed. This is the CLI-level face of INV-6: the feature is
// inert unless opted into.
func TestSaturationKnobs_AbsentConfigMatchesExplicitDefaults(t *testing.T) {
	defer resetSaturationGlobals()
	reqs := risingRequests(80)

	for _, selection := range []string{"composite", "backlog-drift", "all"} {
		absent := labelsFor(t, selection, "", reqs)

		// Ownership is enforced over the SELECTED set, so a single-detector
		// selection may carry only its own block.
		body := "composite:\n  sensitivity: 1.0\nbacklog_drift:\n  slope_k: 3.0\n"
		switch selection {
		case "composite":
			body = "composite:\n  sensitivity: 1.0\n"
		case "backlog-drift":
			body = "backlog_drift:\n  slope_k: 3.0\n"
		}
		explicit := labelsFor(t, selection, writeSatConfig(t, body), reqs)

		for name, want := range absent {
			if got := explicit[name]; got != want {
				t.Errorf("--detectors %s: %s label with explicit defaults = %v, want %v (absent config)",
					selection, name, got, want)
			}
		}
		if len(explicit) != len(absent) {
			t.Errorf("--detectors %s: label count changed (%d vs %d)", selection, len(explicit), len(absent))
		}
	}
}

// Supplying a knob for a detector that is not selected must be a hard error at the
// CLI boundary, not a silently ignored file (R1). This is the user-visible face of
// the ownership contract, including for composite -- which owns a block for the
// first time.
func TestSaturationKnobs_ForeignBlockIsRejectedAtTheCLI(t *testing.T) {
	defer resetSaturationGlobals()

	for _, tc := range []struct{ detectors, config, wantIn string }{
		{"threshold", "composite:\n  sensitivity: 2.0\n", "composite"},
		{"composite", "backlog_drift:\n  slope_k: 2.0\n", "backlog_drift"},
		{"composite,threshold", "backlog_drift:\n  slope_k: 2.0\n", "backlog_drift"},
	} {
		resetSaturationGlobals()
		detectorName = tc.detectors
		saturationConfigPath = writeSatConfig(t, tc.config)

		_, err := resolveSaturation()
		if err == nil {
			t.Errorf("--detectors %s with a %s block: expected an error, got none (the block would be silently dropped)",
				tc.detectors, tc.wantIn)
			continue
		}
		if !strings.Contains(err.Error(), tc.wantIn) {
			t.Errorf("--detectors %s: error should name the offending block %q, got: %v", tc.detectors, tc.wantIn, err)
		}
	}
}

// An out-of-range knob must fail the run with a message naming the field, rather
// than being clamped to something that silently produces different verdicts.
func TestSaturationKnobs_InvalidValueFailsNamingTheField(t *testing.T) {
	defer resetSaturationGlobals()

	for _, tc := range []struct{ detectors, config, wantField string }{
		{"composite", "composite:\n  sensitivity: 0\n", "composite.sensitivity"},
		{"composite", "composite:\n  sensitivity: -1\n", "composite.sensitivity"},
		{"composite", "composite:\n  sensitivity: 5e-324\n", "composite.sensitivity"},
		{"backlog-drift", "backlog_drift:\n  slope_k: 0\n", "backlog_drift.slope_k"},
		{"backlog-drift", "backlog_drift:\n  slope_k: 5e-324\n", "backlog_drift.slope_k"},
		{"all", "composite:\n  sensitivity: .nan\n", "composite.sensitivity"},
	} {
		resetSaturationGlobals()
		detectorName = tc.detectors
		saturationConfigPath = writeSatConfig(t, tc.config)

		_, err := resolveSaturation()
		if err == nil {
			t.Errorf("--detectors %s with %q: expected an error, got none", tc.detectors, tc.config)
			continue
		}
		if !strings.Contains(err.Error(), tc.wantField) {
			t.Errorf("error should name %q, got: %v", tc.wantField, err)
		}
	}
}

// A misspelled knob must not be silently ignored -- strict YAML parsing is what
// stops a typo from reading as "use the default" (R10).
func TestSaturationKnobs_MisspelledKnobIsRejected(t *testing.T) {
	defer resetSaturationGlobals()

	for _, cfg := range []string{
		"composite:\n  sensitivty: 2.0\n",
		"backlog_drift:\n  slope_kk: 3.0\n",
	} {
		resetSaturationGlobals()
		detectorName = "all"
		saturationConfigPath = writeSatConfig(t, cfg)
		if _, err := resolveSaturation(); err == nil {
			t.Errorf("misspelled knob %q was accepted; a typo would silently read as the default", cfg)
		}
	}
}

// peak-rate must be selectable and tunable through the real flag path, and its
// threshold must move the headline label a user reads on stdout. This is the
// end-to-end face of the detector's calibration contract.
func TestSaturationKnobs_PeakRateThresholdChangesTheReportedLabel(t *testing.T) {
	defer resetSaturationGlobals()
	reqs := risingRequests(120)

	loose := labelsFor(t, "peak-rate", writeSatConfig(t, "peak_rate:\n  threshold: 0.001\n"), reqs)
	if loose["peak-rate"] == saturation.Stable {
		t.Fatalf("a tiny threshold did not fire on a rising-backlog fixture (%v); the suppression assertion would be vacuous", loose)
	}

	tight := labelsFor(t, "peak-rate", writeSatConfig(t, "peak_rate:\n  threshold: 1e9\n"), reqs)
	if tight["peak-rate"] != saturation.Stable {
		t.Errorf("a huge threshold did not suppress the label: loose=%v tight=%v", loose, tight)
	}
}

// Every peak_rate knob must be rejected out of range, with the error naming the
// field so an operator can find it.
func TestSaturationKnobs_PeakRateInvalidValuesFailNamingTheField(t *testing.T) {
	defer resetSaturationGlobals()

	for _, tc := range []struct{ config, wantField string }{
		{"peak_rate:\n  threshold: 0\n", "peak_rate.threshold"},
		{"peak_rate:\n  threshold: -1\n", "peak_rate.threshold"},
		{"peak_rate:\n  threshold: 5e-324\n", "peak_rate.threshold"},
		{"peak_rate:\n  threshold: .nan\n", "peak_rate.threshold"},
		{"peak_rate:\n  min_observations: 0\n", "peak_rate.min_observations"},
		{"peak_rate:\n  min_observations: -5\n", "peak_rate.min_observations"},
		{"peak_rate:\n  consecutive_k: 0\n", "peak_rate.consecutive_k"},
		{"peak_rate:\n  warmup_us: -1\n", "peak_rate.warmup_us"},
		{"peak_rate:\n  overload_multiple: 0\n", "peak_rate.overload_multiple"},
		{"peak_rate:\n  overload_multiple: -1\n", "peak_rate.overload_multiple"},
	} {
		resetSaturationGlobals()
		detectorName = "peak-rate"
		saturationConfigPath = writeSatConfig(t, tc.config)
		_, err := resolveSaturation()
		if err == nil {
			t.Errorf("%q: expected an error, got none", tc.config)
			continue
		}
		if !strings.Contains(err.Error(), tc.wantField) {
			t.Errorf("error should name %q, got: %v", tc.wantField, err)
		}
	}
}

// A misspelled peak_rate knob must be rejected rather than silently read as the
// default (R10).
func TestSaturationKnobs_PeakRateMisspelledKnobIsRejected(t *testing.T) {
	defer resetSaturationGlobals()
	for _, cfg := range []string{
		"peak_rate:\n  threshhold: 1.0\n",
		"peak_rate:\n  min_observation: 100\n",
		"peak_rate:\n  overload_multipler: 3.0\n",
		"peak_rate:\n  warmup_u: 1000\n",
	} {
		resetSaturationGlobals()
		detectorName = "peak-rate"
		saturationConfigPath = writeSatConfig(t, cfg)
		if _, err := resolveSaturation(); err == nil {
			t.Errorf("misspelled knob %q was accepted; a typo would silently read as the default", cfg)
		}
	}
}

// INV-13 at the CLI boundary: `blis run` and `blis replay` must write byte-identical
// saturation reports for the same workload, with the knob configured. The acceptance
// criterion in #1614 asks for this per detector, so it is asserted over the whole
// roster -- a newly added detector cannot quietly escape the guarantee.
//
// The two legs differ exactly as the real commands differ: run resolves its requests
// from the simulator's own metrics, while replay resolves them from an exported
// TraceV2 round trip. Everything downstream -- the resolver, the tracer, the event
// sorter, the reducer and the JSON writer -- is shared, so this pins the property the
// sharing is supposed to give.
func TestSaturationKnobs_RunReplayReportsAreByteIdentical(t *testing.T) {
	defer resetSaturationGlobals()

	// A workload with a growing backlog, so every detector has a non-trivial verdict
	// sequence rather than a uniform STABLE one.
	reqs := risingRequests(120)

	// The replay leg's requests, reconstructed the way `blis replay` reconstructs
	// them: through the TraceV2 record round trip.
	simReqs := make([]*sim.Request, 0, len(reqs))
	for _, r := range reqs {
		simReqs = append(simReqs, &sim.Request{
			ID:             r.ID,
			ArrivalTime:    int64(r.ArrivedAt * 1e6),
			TTFTSet:        true,
			FirstTokenTime: int64(r.E2E * 1e3),
			ITL:            []int64{},
			State:          sim.StateCompleted,
		})
	}
	replayReqs := workload.TraceRecordsToRequestMetrics(workload.RequestsToTraceRecords(simReqs))

	// Each detector's own knob block. Ownership is enforced over the SELECTED set, so
	// a single-detector selection may carry only its own block; `all` carries them
	// all. Parity is therefore asserted for the CONFIGURED path, not just the default.
	blocks := map[string]string{
		"composite":     "composite:\n  sensitivity: 2.0\n",
		"threshold":     "threshold:\n  threshold_ms: 250\n",
		"backlog-drift": "backlog_drift:\n  slope_k: 4.0\n",
		"peak-rate":     "peak_rate:\n  threshold: 0.25\n  min_observations: 10\n  warmup_us: 1000\n",
	}
	var everyBlock string
	for _, name := range saturation.AllDetectorNames() {
		everyBlock += blocks[name]
	}

	reportFor := func(selection string, requests []sim.RequestMetrics) []byte {
		body := blocks[selection]
		if selection == "all" {
			body = everyBlock
		}
		resetSaturationGlobals()
		detectorName = selection
		saturationConfigPath = writeSatConfig(t, body)
		saturationReport = filepath.Join(t.TempDir(), "sat.json")

		tracer, err := resolveSaturation()
		if err != nil {
			t.Fatalf("resolveSaturation(%q): %v", selection, err)
		}
		if _, err := tracer.run(requests); err != nil {
			t.Fatalf("tracer.run(%q): %v", selection, err)
		}
		data, err := os.ReadFile(saturationReport)
		if err != nil {
			t.Fatalf("read report: %v", err)
		}
		return data
	}

	for _, selection := range append(saturation.AllDetectorNames(), "all") {
		runLeg := reportFor(selection, reqs)
		replayLeg := reportFor(selection, replayReqs)

		if len(runLeg) == 0 {
			t.Fatalf("--detectors %s: run leg wrote an empty report; the comparison would be vacuous", selection)
		}
		if string(runLeg) != string(replayLeg) {
			t.Errorf("--detectors %s: run and replay reports differ (INV-13)\n--- run ---\n%s\n--- replay ---\n%s",
				selection, runLeg, replayLeg)
		}
	}
}
