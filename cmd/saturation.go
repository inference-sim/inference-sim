package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/saturation"
)

// Saturation CLI flags, shared by run and replay.
var (
	detectorName     string // --detectors: "" = off; "settle" = the DEV-10 steady-state detector
	saturationReport string // --saturation-report: file for the detector's verdict JSON
)

// settleDetector is the one detector name (DEV-10, inference-sim#1618): the roster it replaced
// (composite, threshold, backlog-drift, peak-rate) is gone.
const settleDetector = "settle"

// registerDetectorFlags registers the saturation flags on cmd. Called by run and replay so both
// share one flag surface.
func registerDetectorFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&detectorName, "detectors", "",
		"Steady-state detector: \"settle\" classifies the measured window SETTLED, OVERLOADED or INCOMPLETE (DEV-10). Empty = off.")
	cmd.Flags().StringVar(&saturationReport, "saturation-report", "",
		"File to write the detector's verdict (state, reason and the numbers behind it) to, as JSON. Requires --detectors.")
}

// saturationTracer holds the resolved saturation selection.
type saturationTracer struct {
	reportPath string
}

// resolveSaturation turns the saturation flags into a tracer, or nil when saturation is off.
func resolveSaturation() (*saturationTracer, error) {
	if detectorName == "" {
		if saturationReport != "" {
			return nil, fmt.Errorf("--saturation-report requires --detectors")
		}
		return nil, nil
	}
	if detectorName != settleDetector {
		return nil, fmt.Errorf("--detectors %q: the only detector is %q; the composite, threshold, backlog-drift and peak-rate detectors were replaced by it (inference-sim#1618)", detectorName, settleDetector)
	}
	if err := saturation.ValidateReportPath(saturationReport); err != nil {
		return nil, err
	}
	return &saturationTracer{reportPath: saturationReport}, nil
}

// saturationJobs are the requests that were in the system at some time: from arrival to the
// clock their fate was written at, or, for a request still in the system when the run ended, to
// the end of the run. A request dropped at arrival or never started was never in the system.
func saturationJobs(cs *cluster.ClusterSimulator, m *sim.Metrics, endUs int64) []saturation.Job {
	facts := cs.RequestFacts()
	e2e := map[string]float64{}
	for _, rm := range m.RequestRows() {
		e2e[rm.ID] = rm.E2E
	}
	var jobs []saturation.Job
	for _, rec := range cs.Fates().Records() {
		arr := facts[rec.ID].ArrivalUs
		switch rec.Fate {
		case sim.FateCompleted:
			jobs = append(jobs, saturation.Job{ArrivalUs: arr, DepartUs: rec.At, LatencyMs: e2e[rec.ID]})
		case sim.FateTimedOut:
			jobs = append(jobs, saturation.Job{ArrivalUs: arr, DepartUs: rec.At})
		case sim.FateInFlightAtHorizon, sim.FateQueuedAtHorizon:
			jobs = append(jobs, saturation.Job{ArrivalUs: arr, DepartUs: endUs, Open: true})
		}
	}
	return jobs
}

// run classifies the measured window [startUs, endUs) and writes the verdict file when one was
// asked for. The Result's saturation block holds the verdict.
func (t *saturationTracer) run(jobs []saturation.Job, startUs, endUs int64) (saturation.Verdict, error) {
	v := saturation.Settle(jobs, startUs, endUs, saturation.Options{})
	if t.reportPath != "" {
		b, err := json.MarshalIndent(v, "", "  ")
		if err != nil {
			return v, err
		}
		if err := os.WriteFile(t.reportPath, append(b, '\n'), 0o644); err != nil {
			return v, fmt.Errorf("--saturation-report: %w", err)
		}
	}
	return v, nil
}
