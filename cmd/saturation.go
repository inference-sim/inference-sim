package cmd

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/saturation"
)

// Saturation CLI flags (#1516), shared across run, replay, and observe. These
// replace the 13 legacy flags (registerSaturationFlags + --post-hoc-detector +
// --saturation-threshold-ms). --saturation-report keeps its name but now means
// "per-event verdict trace file" rather than the old BacklogDriftReport.
var (
	detectorName         string // --detectors <name>: exactly ONE of composite, threshold, backlog-drift ("" = off)
	saturationConfigPath string // --saturation-config: strict-YAML tuning file
	saturationReport     string // --saturation-report: per-event verdict trace file (shared across run/replay/observe)
)

// registerDetectorFlags registers the three saturation flags on cmd. Called by
// run, replay, and observe so all three share one flag surface (#1516).
func registerDetectorFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&detectorName, "detectors", "",
		"Post-hoc saturation detector to trace (exactly one of: composite, threshold, backlog-drift). Empty = off. Multiple detectors / \"all\" require the bank (#1519).")
	cmd.Flags().StringVar(&saturationConfigPath, "saturation-config", "",
		"Path to a strict-YAML saturation tuning file (optional threshold: and backlog_drift: blocks). composite has no tunable params.")
	cmd.Flags().StringVar(&saturationReport, "saturation-report", "",
		"File to write the selected detector's per-event verdict trace as {\"trace\":[...]} JSON. Requires --detectors.")
}

// resolveSaturation turns the three saturation flags into a configured single
// detector and an in-memory collector, or (nil, nil) when saturation is off.
// It is the ONE shared helper the issue calls for — run, replay, and observe all
// route through it so the pipeline is identical across commands.
//
// Errors (returned, not fatal — the caller decides how to surface):
//   - --detectors "all" or a comma-list: needs the bank (#1519).
//   - unknown detector name: lists the valid single names.
//   - --saturation-config / --saturation-report given without --detectors.
//   - bad config file (unknown YAML key, out-of-range param).
//   - unwritable --saturation-report path (checked up front so the run fails fast).
func resolveSaturation() (saturation.Detector, *saturation.InMemoryCollector, error) {
	// Off: no detector selected. Config/report without a detector is an error —
	// they would otherwise be silently ignored.
	if detectorName == "" {
		if saturationConfigPath != "" {
			return nil, nil, fmt.Errorf("--saturation-config requires --detectors")
		}
		if saturationReport != "" {
			return nil, nil, fmt.Errorf("--saturation-report requires --detectors")
		}
		return nil, nil, nil
	}

	// Reject the bank / multi-detector syntax that belongs to #1519. "all" and any
	// comma-list must error here rather than silently pick one.
	if detectorName == "all" || strings.Contains(detectorName, ",") {
		return nil, nil, fmt.Errorf("--detectors %q selects multiple detectors, which needs the bank (#1519); pass exactly one of: composite, threshold, backlog-drift", detectorName)
	}

	cfg, err := saturation.LoadSaturationConfig(saturationConfigPath)
	if err != nil {
		return nil, nil, err
	}

	detector, err := saturation.BuildDetector(detectorName, cfg)
	if err != nil {
		return nil, nil, err
	}

	// Validate the report path up front so an unwritable destination fails before
	// the (expensive) simulation runs rather than after.
	if err := saturation.ValidateReportPath(saturationReport); err != nil {
		return nil, nil, err
	}

	return detector, saturation.NewInMemoryCollector(), nil
}

// runSaturationTrace streams the resolved detector over completed request
// metrics and writes its per-event trace to --saturation-report, if a report
// path was given. It is a no-op when detector is nil (saturation off) or when no
// report path was set (the trace would be discarded anyway). This is the shared
// tail used by all three commands; the only per-command difference is the
// []sim.RequestMetrics input (sim-derived for run/replay, server-derived for
// observe).
func runSaturationTrace(detector saturation.Detector, collector *saturation.InMemoryCollector, requests []sim.RequestMetrics) error {
	if detector == nil || saturationReport == "" {
		return nil
	}
	saturation.ReplayOneDetector(detector, requests, collector)
	return saturation.WriteCombinedReport(saturationReport, collector)
}
