package cmd

import (
	"fmt"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/saturation"
)

// Saturation CLI flags (#1516, extended in #1519), shared across run, replay,
// and observe. --saturation-report means "per-event verdict trace file".
var (
	detectorName         string // --detectors: "" = off; one name = single detector (#1516); "all" or comma-list = bank (#1519)
	saturationConfigPath string // --saturation-config: strict-YAML tuning file
	saturationReport     string // --saturation-report: per-event verdict trace file (shared across run/replay/observe)
)

// registerDetectorFlags registers the three saturation flags on cmd. Called by
// run, replay, and observe so all three share one flag surface (#1516, #1519).
func registerDetectorFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&detectorName, "detectors", "",
		"Post-hoc saturation detector(s) to trace: one of composite, threshold, backlog-drift; a comma-list of those; or \"all\". Empty = off.")
	cmd.Flags().StringVar(&saturationConfigPath, "saturation-config", "",
		"Path to a strict-YAML saturation tuning file (optional threshold: and backlog_drift: blocks). composite has no tunable params.")
	cmd.Flags().StringVar(&saturationReport, "saturation-report", "",
		"File to write the selected detector(s)' per-event verdict trace as {\"trace\":[...]} JSON. Requires --detectors.")
}

// saturationTracer holds a resolved saturation selection and drives it over
// completed request metrics. It hides whether the selection is #1516's single
// streaming detector or #1519's multi-detector bank, so run/replay/observe share
// ONE call site (R23) — they build a tracer with resolveSaturation and, if it is
// non-nil, call trace(requests). Exactly one of detector/bank is set.
type saturationTracer struct {
	collector *saturation.InMemoryCollector
	detector  saturation.Detector // single-detector path (#1516); nil when the bank is used
	bank      *saturation.Bank    // multi-detector path (#1519); nil when a single detector is used
}

// trace streams the resolved selection over requests and writes the per-event
// trace to --saturation-report. It is a no-op when no report path was given (the
// trace would be discarded anyway), mirroring #1516. The []sim.RequestMetrics
// input is the only per-command difference (sim-derived for run/replay,
// server-derived for observe); the pipeline is identical.
func (t *saturationTracer) trace(requests []sim.RequestMetrics) error {
	if saturationReport == "" {
		return nil
	}
	// Enforce the exactly-one-path invariant loudly rather than relying on the
	// branch below to nil-deref. resolveSaturation always sets exactly one of
	// bank/detector; a violation here means a construction bug, not user input.
	if (t.bank == nil) == (t.detector == nil) {
		return fmt.Errorf("saturation tracer: exactly one of bank/detector must be set (bank=%v, detector=%v) — construction bug", t.bank != nil, t.detector != nil)
	}
	// Zero completed requests writes a valid but empty {"trace":[]}. Warn so the
	// empty file isn't mistaken for a detector bug — consistent across run,
	// replay, and observe (the input source differs, this signal does not).
	if len(requests) == 0 {
		logrus.Warnf("--detectors %q: 0 completed requests; saturation trace will be empty", detectorName)
	}
	if t.bank != nil {
		// The bank was constructed with the collector as its sink. Classify replays
		// once and fans out to every detector, then Close flushes the sink.
		//
		// totalArrivals is IGNORED by the streaming replay in this PR (the detectors
		// derive rate from the arrival events they observe). We pass len(requests) as
		// a placeholder. NOTE: this is the COMPLETED count, not the injected total
		// (completed + timed-out + dropped) that sim.BatchClassifier documents for
		// totalArrivals — #1517 must supply the true arrival total before it wires
		// this into the stdout label.
		t.bank.Classify(requests, len(requests))
		t.bank.Close()
	} else {
		saturation.ReplayOneDetector(t.detector, requests, t.collector)
	}
	return saturation.WriteCombinedReport(saturationReport, t.collector)
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
//   - --saturation-config / --saturation-report given without --detectors.
//   - unknown detector name (lists the valid names, R1).
//   - an empty selection after trimming (e.g. "," or ", ,").
//   - bad config file (unknown YAML key, out-of-range param), including a config
//     block that does not belong to the single selected detector (#1516 ownership).
//   - unwritable --saturation-report path (checked up front so the run fails fast).
func resolveSaturation() (*saturationTracer, error) {
	// Off: no detector selected. Config/report without a detector is an error —
	// they would otherwise be silently ignored.
	if detectorName == "" {
		if saturationConfigPath != "" {
			return nil, fmt.Errorf("--saturation-config requires --detectors")
		}
		if saturationReport != "" {
			return nil, fmt.Errorf("--saturation-report requires --detectors")
		}
		return nil, nil
	}

	cfg, err := saturation.LoadSaturationConfig(saturationConfigPath)
	if err != nil {
		return nil, err
	}

	collector := saturation.NewInMemoryCollector()
	tracer := &saturationTracer{collector: collector}

	if isBankSelection(detectorName) {
		names, err := parseDetectorSelection(detectorName)
		if err != nil {
			return nil, err
		}
		// NewBank validates names, de-dups, canonicalizes order, and builds each
		// detector; a foreign config block (for an unselected detector) is ignored,
		// while a value error in a selected detector's own block surfaces (R1/R6).
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

	// A detector was selected but no report path given: the per-event trace is the
	// only output in this PR (the stdout final label lands in #1517), so this
	// combination produces nothing. Warn rather than silently discard the work.
	if saturationReport == "" {
		logrus.Warnf("--detectors %q selected but --saturation-report not set; no saturation output will be produced (the trace file is the only output until #1517 adds the stdout label)", detectorName)
	}

	// Validate the report path up front so an unwritable destination fails before
	// the (expensive) simulation runs rather than after.
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
// least one detector. Unknown names are left for NewBank to reject so the valid
// list is reported from one place (R1).
func parseDetectorSelection(sel string) ([]string, error) {
	if sel == "all" {
		return saturation.AllDetectorNames(), nil
	}
	parts := strings.Split(sel, ",")
	names := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
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
