package cmd

import (
	"fmt"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/saturation"
)

// Saturation CLI flags (#1516, extended in #1519, #1517), shared across run,
// replay, and observe. --saturation-report means "per-event verdict trace file".
var (
	detectorName          string // --detectors: "" = off; one name = single detector (#1516); "all" or comma-list = bank (#1519)
	saturationConfigPath  string // --saturation-config: strict-YAML tuning file
	saturationReport      string // --saturation-report: per-event verdict trace file (shared across run/replay/observe)
	saturationFinalWindow string // --saturation-final-window: Go duration for the last-window final-label vote (#1517)
)

// defaultFinalWindowUs is the trailing-window default (30s in µs) for the final
// last-window plurality vote when neither --saturation-final-window nor a
// backlog_drift.window_size_sec config value is supplied (#1517).
const defaultFinalWindowUs int64 = 30_000_000

// registerDetectorFlags registers the saturation flags on cmd. Called by run,
// replay, and observe so all three share one flag surface (#1516, #1519, #1517).
func registerDetectorFlags(cmd *cobra.Command) {
	// The valid-name list is derived from the roster so it cannot desync as
	// detectors are added (#1614).
	validNames := strings.Join(saturation.AllDetectorNames(), ", ")
	cmd.Flags().StringVar(&detectorName, "detectors", "",
		"Post-hoc saturation detector(s) to trace: one of "+validNames+"; a comma-list of those; or \"all\". Empty = off.")
	cmd.Flags().StringVar(&saturationConfigPath, "saturation-config", "",
		"Path to a strict-YAML saturation tuning file. Every detector has a calibration knob: composite: {sensitivity}, threshold: {threshold_ms}, backlog_drift: {slope_k, ...}, peak_rate: {threshold, min_observations, warmup_us, consecutive_k, overload_multiple}.")
	cmd.Flags().StringVar(&saturationReport, "saturation-report", "",
		"File to write the selected detector(s)' per-event verdict trace as {\"final\":{...},\"trace\":[...]} JSON. Requires --detectors.")
	cmd.Flags().StringVar(&saturationFinalWindow, "saturation-final-window", "",
		"Go duration for the trailing window of the stdout final-label plurality vote (e.g. 30s). Default: backlog_drift.window_size_sec if set, else 30s. Requires --detectors.")
}

// resolveFinalWindowUs computes the trailing-window size (µs) for the final
// last-window plurality vote, the SAME value for every detector in the run
// (#1517). Resolution order:
//  1. --saturation-final-window (Go duration) if set → d.Microseconds();
//  2. else --saturation-config backlog_drift.window_size_sec (whole seconds) → ×1e6;
//  3. else defaultFinalWindowUs (30s).
//
// A non-empty --saturation-final-window that fails to parse, or a non-positive
// duration, is a returned error (never silently defaulted — R1/R3).
func resolveFinalWindowUs(finalWindowFlag string, cfg saturation.SaturationConfig) (int64, error) {
	if finalWindowFlag != "" {
		d, err := time.ParseDuration(finalWindowFlag)
		if err != nil {
			return 0, fmt.Errorf("--saturation-final-window %q is not a valid Go duration: %w", finalWindowFlag, err)
		}
		if d <= 0 {
			return 0, fmt.Errorf("--saturation-final-window must be > 0, got %s", finalWindowFlag)
		}
		return d.Microseconds(), nil
	}
	if cfg.BacklogDrift != nil && cfg.BacklogDrift.WindowSizeSec != nil {
		// buildDetector → resolveBacklogDriftConfig has already rejected
		// window_size_sec <= 0 by the time resolveSaturation calls this (the bank /
		// single detector is constructed first), so the value here is > 0.
		return int64(*cfg.BacklogDrift.WindowSizeSec) * 1_000_000, nil
	}
	return defaultFinalWindowUs, nil
}

// saturationTracer holds a resolved saturation selection and drives it over
// completed request metrics. It hides whether the selection is #1516's single
// streaming detector or #1519's multi-detector bank, so run/replay/observe share
// ONE call site (R23) — they build a tracer with resolveSaturation and, if it is
// non-nil, call run(requests). Exactly one of detector/bank is set.
//
// reportPath, selection, and windowUs are CAPTURED at construction from the flag
// globals rather than re-read in run(): once resolveSaturation returns, the tracer
// is self-contained and does not depend on the flag globals still holding their
// values (they otherwise couple construction to a later call across the cobra
// lifecycle — fragile if a future refactor clears them between the two).
type saturationTracer struct {
	collector  *saturation.InMemoryCollector
	detector   saturation.Detector // single-detector path (#1516); nil when the bank is used
	bank       *saturation.Bank    // multi-detector path (#1519); nil when a single detector is used
	reportPath string              // --saturation-report captured at construction ("" ⇒ no trace file, stdout label only)
	selection  string              // --detectors value captured at construction (for warning messages)
	windowUs   int64               // resolved trailing-window size (µs) for the final label vote (#1517)
}

// run streams the resolved selection over requests, computes the per-detector
// final label via the detector-agnostic reducer (saturation.ReduceAll), and
// writes the {"final":...,"trace":...} report to the captured report path (only
// when one was given). It returns the final label map (detector→level), which the
// caller splices onto stdout (#1517). The []sim.RequestMetrics input is the only
// per-command difference (sim-derived for run/replay, server-derived for observe);
// the pipeline is identical.
//
// Unlike #1516's trace(), run() does NOT early-return when reportPath is empty:
// the collected records feed the stdout label even when no trace file is
// requested. Only the file write is conditional.
func (t *saturationTracer) run(requests []sim.RequestMetrics) (map[string]saturation.Level, error) {
	// Enforce the exactly-one-path invariant loudly rather than relying on the
	// branch below to nil-deref. resolveSaturation always sets exactly one of
	// bank/detector; a violation here means a construction bug, not user input.
	if (t.bank == nil) == (t.detector == nil) {
		return nil, fmt.Errorf("saturation tracer: exactly one of bank/detector must be set (bank=%v, detector=%v) — construction bug", t.bank != nil, t.detector != nil)
	}
	// Zero completed requests yields an empty trace and an all-STABLE degenerate
	// final label. Warn so the empty result isn't mistaken for a detector bug —
	// consistent across run, replay, and observe (the input source differs, this
	// signal does not).
	if len(requests) == 0 {
		logrus.Warnf("--detectors %q: 0 completed requests; saturation trace will be empty and final labels default to STABLE", t.selection)
	}
	if t.bank != nil {
		// The bank was constructed with the collector as its sink. Run replays once
		// and fans out to every detector, then Close flushes the sink.
		if err := t.bank.Run(requests); err != nil {
			return nil, err
		}
		t.bank.Close()
	} else {
		saturation.ReplayOneDetector(t.detector, requests, t.collector)
	}

	// Reduce the collected per-event trace to one final label per detector by the
	// uniform last-window plurality rule (#1517). ReduceAll groups by detector name,
	// so a single-detector selection yields a one-key map and the bank a full map.
	final := saturation.ReduceAll(t.collector.Records(), t.windowUs)

	// Write the report only when a path was given; the final map is embedded so the
	// on-disk shape is {"final":{...},"trace":[...]} (#1517).
	if t.reportPath != "" {
		if err := saturation.WriteCombinedReport(t.reportPath, t.collector, final); err != nil {
			return nil, err
		}
	}
	return final, nil
}

// resolveSaturation turns the three saturation flags into a saturationTracer, or
// nil when saturation is off. It is the ONE shared helper (R23) run, replay, and
// observe route through so the pipeline is identical across commands.
//
// Selection grammar for --detectors (isBankSelection routes on the presence of a
// comma, so ANY comma — even a trailing one — takes the bank path):
//   - ""                              → off (nil tracer)
//   - a single bare name             → #1516 single streaming detector (byte-identical continuity)
//   - "all"                          → #1519 bank over the full roster
//   - any value containing a comma    → #1519 bank over the named subset (empty entries trimmed)
//
// Errors (returned, not fatal — the caller decides how to surface):
//   - --saturation-config / --saturation-report / --saturation-final-window given
//     without --detectors.
//   - unknown detector name (lists the valid names, R1).
//   - an empty selection after trimming (e.g. "," or ", ,").
//   - bad config file (unknown YAML key, out-of-range param), including a config
//     block that does not belong to the single selected detector (#1516 ownership).
//   - an invalid or non-positive --saturation-final-window (#1517).
//   - unwritable --saturation-report path (checked up front so the run fails fast).
func resolveSaturation() (*saturationTracer, error) {
	// Off: no detector selected. Config/report/final-window without a detector is an
	// error — they would otherwise be silently ignored.
	if detectorName == "" {
		if saturationConfigPath != "" {
			return nil, fmt.Errorf("--saturation-config requires --detectors")
		}
		if saturationReport != "" {
			return nil, fmt.Errorf("--saturation-report requires --detectors")
		}
		if saturationFinalWindow != "" {
			return nil, fmt.Errorf("--saturation-final-window requires --detectors")
		}
		return nil, nil
	}

	cfg, err := saturation.LoadSaturationConfig(saturationConfigPath)
	if err != nil {
		return nil, err
	}

	collector := saturation.NewInMemoryCollector()
	// Capture the flag values now so the tracer is self-contained and does not
	// re-read the globals in run() (see the type doc).
	tracer := &saturationTracer{collector: collector, reportPath: saturationReport, selection: detectorName}

	if isBankSelection(detectorName) {
		names, err := parseDetectorSelection(detectorName)
		if err != nil {
			return nil, err
		}
		// NewBank validates names, de-dups, canonicalizes order, and builds each
		// detector; a config block whose owning detector is not in the selection is
		// a hard error (R1 — checkBlockOwnershipSet), as is a value error in a
		// selected detector's own block (R6).
		bank, err := saturation.NewBank(names, cfg, collector)
		if err != nil {
			return nil, err
		}
		tracer.bank = bank
	} else {
		// Single detector (#1516): BuildDetector enforces block↔detector ownership,
		// so a config block for a different detector is a hard error here.
		detector, err := saturation.BuildDetector(detectorName, cfg)
		if err != nil {
			return nil, err
		}
		tracer.detector = detector
	}

	// Resolve the final-label trailing window (#1517), the same value for every
	// detector. Done AFTER detector/bank construction so any out-of-range
	// window_size_sec has already been rejected by resolveBacklogDriftConfig.
	windowUs, err := resolveFinalWindowUs(saturationFinalWindow, cfg)
	if err != nil {
		return nil, err
	}
	tracer.windowUs = windowUs

	// A detector was selected but no report path given: the per-event trace file is
	// suppressed, but the stdout final label is still produced (#1517), so this is a
	// valid, useful configuration. Note it at debug level rather than warning.
	if saturationReport == "" {
		logrus.Debugf("--detectors %q selected without --saturation-report; emitting the stdout final label only (no per-event trace file)", detectorName)
	}

	// Validate the report path up front so an unwritable destination fails before
	// the (expensive) simulation runs rather than after. An empty path is a no-op.
	if err := saturation.ValidateReportPath(saturationReport); err != nil {
		return nil, err
	}

	return tracer, nil
}

// isBankSelection reports whether --detectors selects the multi-detector bank
// (#1519): the literal "all", or any value containing a comma. A bare single
// name routes to #1516's single-detector path for byte-identical continuity.
func isBankSelection(sel string) bool {
	return sel == "all" || strings.Contains(sel, ",")
}

// parseDetectorSelection expands a bank selection into an explicit name list.
// "all" becomes the full roster; a comma-list is split, trimmed, and stripped of
// empty entries (so a trailing comma or stray whitespace is tolerated). An
// all-empty list (e.g. "," or ", ,") is a hard error — the bank must drive at
// least one detector. "all" mixed into a comma-list (e.g. "composite,all") is a
// hard error naming the conflict: "all" is a roster keyword, not a detector name,
// so combining it with individual names is ambiguous — caught here rather than
// letting NewBank reject the literal "all" with a misleading "unknown detector"
// message. Other unknown names are left for NewBank to reject so the valid list
// is reported from one place (R1).
func parseDetectorSelection(sel string) ([]string, error) {
	if sel == "all" {
		return saturation.AllDetectorNames(), nil
	}
	parts := strings.Split(sel, ",")
	names := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "all" {
			return nil, fmt.Errorf("--detectors %q: \"all\" cannot be combined with individual detector names; use \"--detectors all\" alone to select the full roster", sel)
		}
		if p != "" {
			names = append(names, p)
		}
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("--detectors %q selects no detectors; pass one of: %s, a comma-list of those, or \"all\"",
			sel, strings.Join(saturation.AllDetectorNames(), ", "))
	}
	return names, nil
}
