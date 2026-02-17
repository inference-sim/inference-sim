package cluster

import (
	"time"

	"github.com/inference-sim/inference-sim/sim/trace"
)

// EvaluationResult bundles all outputs from a cluster simulation run.
// Used by downstream consumers (GEPA, OpenEvolve adapters) for unified access
// to metrics, fitness scores, decision traces, and trace summaries.
type EvaluationResult struct {
	Metrics  *RawMetrics
	Fitness  *FitnessResult         // nil if no fitness weights provided
	Trace    *trace.SimulationTrace // nil if trace-level is "none"
	Summary  *trace.TraceSummary    // nil if --summarize-trace not set

	SimDuration int64         // simulation clock at end (ticks)
	WallTime    time.Duration // wall-clock duration of Run()
}

// NewEvaluationResult constructs an EvaluationResult.
// metrics is required; fitness, tr, and summary may be nil.
func NewEvaluationResult(metrics *RawMetrics, fitness *FitnessResult, tr *trace.SimulationTrace, summary *trace.TraceSummary, simDuration int64, wallTime time.Duration) *EvaluationResult {
	return &EvaluationResult{
		Metrics:     metrics,
		Fitness:     fitness,
		Trace:       tr,
		Summary:     summary,
		SimDuration: simDuration,
		WallTime:    wallTime,
	}
}
