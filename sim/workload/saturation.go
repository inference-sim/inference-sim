package workload

import (
	"fmt"
	"math"
	"time"

	sim "github.com/inference-sim/inference-sim/sim"
)

// ComputeSimEndUs computes the simulation end time from request completion times.
// Returns max(all completion times, horizon) where horizon is used as a floor if > 0.
// This is the canonical simEndUs calculation used by run, replay, and calibrate commands.
func ComputeSimEndUs(requests []*sim.Request, horizon int64) int64 {
	simEndUs := int64(0)
	for _, req := range requests {
		completionUs := req.ArrivalTime
		if req.TTFTSet {
			completionUs += req.FirstTokenTime
			// Only sum ITL if request has valid TTFT (prevents malformed data from inflating simEndUs)
			for _, itl := range req.ITL {
				completionUs += itl
			}
		}
		if completionUs > simEndUs {
			simEndUs = completionUs
		}
	}
	// Use horizon as floor if explicitly set and larger
	if horizon > 0 && horizon < math.MaxInt64 && horizon > simEndUs {
		simEndUs = horizon
	}
	return simEndUs
}

// BacklogDriftConfig configures the backlog-drift saturation analyzer.
type BacklogDriftConfig struct {
	WindowSize      time.Duration // Window width for sampling and per-window metrics (BC-1)
	MinWindows      int           // Minimum complete windows required for classification (BC-7)
	PeakRatio       float64       // Peak-to-mean threshold for TRANSIENT_BACKLOG detection (BC-6)
	PeakRatioBand   float64       // Confidence band around PeakRatio (±band creates borderline zone)
	ConfidenceCI    float64       // Confidence level for slope significance test (BC-3)

	// Drain-ratio knobs (#1392), retained as user-facing --saturation-config
	// backlog_drift: YAML fields. Validation in NewBacklogDriftConfig enforces the
	// relation SaturatedDrainRatio <= TransientDrainRatio so the (formerly
	// classifier-defined) regions don't overlap.
	WarmupWindows       int     // Inject windows skipped at the start (engine ramp-up)
	TailWindows         int     // Inject windows skipped at the end (rate ramp-down boundary)
	SaturatedDrainRatio float64 // Mean DrainRatio < this → PERSISTENTLY_SATURATED
	TransientDrainRatio float64 // Mean DrainRatio < this → TRANSIENT_BACKLOG
}

// NewBacklogDriftConfig creates a BacklogDriftConfig with validation (BC-10, BC-14, R3).
// Panics if any parameter is invalid (NaN, Inf, out of range).
//
// warmupWindows must be >= 0. saturatedDrainRatio and transientDrainRatio must each be
// in (0, 1]; saturatedDrainRatio <= transientDrainRatio so PERSISTENTLY_SATURATED and
// TRANSIENT_BACKLOG regions form a contiguous partition of [0, 1].
func NewBacklogDriftConfig(
	windowSize time.Duration,
	minWindows int,
	peakRatio, peakRatioBand, confidenceCI float64,
	warmupWindows, tailWindows int,
	saturatedDrainRatio, transientDrainRatio float64,
) BacklogDriftConfig {
	if windowSize <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: WindowSize must be > 0, got %v", windowSize))
	}
	if minWindows <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: MinWindows must be > 0, got %d", minWindows))
	}
	if peakRatio <= 0 || math.IsNaN(peakRatio) || math.IsInf(peakRatio, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: PeakRatio must be a finite value > 0, got %f", peakRatio))
	}
	if peakRatioBand < 0 || math.IsNaN(peakRatioBand) || math.IsInf(peakRatioBand, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: PeakRatioBand must be >= 0, got %f", peakRatioBand))
	}
	if confidenceCI <= 0 || confidenceCI >= 1 || math.IsNaN(confidenceCI) || math.IsInf(confidenceCI, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: ConfidenceCI must be in (0, 1), got %f", confidenceCI))
	}
	if warmupWindows < 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: WarmupWindows must be >= 0, got %d", warmupWindows))
	}
	if tailWindows < 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: TailWindows must be >= 0, got %d", tailWindows))
	}
	if saturatedDrainRatio <= 0 || saturatedDrainRatio > 1 || math.IsNaN(saturatedDrainRatio) || math.IsInf(saturatedDrainRatio, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: SaturatedDrainRatio must be in (0, 1], got %f", saturatedDrainRatio))
	}
	if transientDrainRatio <= 0 || transientDrainRatio > 1 || math.IsNaN(transientDrainRatio) || math.IsInf(transientDrainRatio, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: TransientDrainRatio must be in (0, 1], got %f", transientDrainRatio))
	}
	if saturatedDrainRatio > transientDrainRatio {
		panic(fmt.Sprintf("BacklogDriftConfig: SaturatedDrainRatio (%f) must be <= TransientDrainRatio (%f); regions would overlap", saturatedDrainRatio, transientDrainRatio))
	}
	return BacklogDriftConfig{
		WindowSize:          windowSize,
		MinWindows:          minWindows,
		PeakRatio:           peakRatio,
		PeakRatioBand:       peakRatioBand,
		ConfidenceCI:        confidenceCI,
		WarmupWindows:       warmupWindows,
		TailWindows:         tailWindows,
		SaturatedDrainRatio: saturatedDrainRatio,
		TransientDrainRatio: transientDrainRatio,
	}
}

// DefaultBacklogDriftConfig returns the default configuration per issues #1298, #1392.
//
// WarmupWindows=2 and TailWindows=1 are empirical defaults from a Llama-3.1-70B
// reference experiment (rate=80, num_requests=6000):
//   - Window 1 was a clear engine ramp-up (DrainRatio ≈ 0.6) before steady state.
//   - The window where inject ends mid-bucket has artificially low NumEntered
//     and full NumLeft (engine continues draining backlog), pushing DrainRatio > 1
//     and biasing the steady-state mean upward toward "unsaturated".
//
// Routes through NewBacklogDriftConfig so the defaults are self-validating; if a
// future change introduces an inter-field invariant, this function will panic at
// init time rather than silently producing an inconsistent default config.
func DefaultBacklogDriftConfig() BacklogDriftConfig {
	return NewBacklogDriftConfig(
		60*time.Second, // WindowSize
		5,              // MinWindows
		2.0,            // PeakRatio
		0.2,            // PeakRatioBand (absolute, ≈ 10% of PeakRatio)
		0.95,           // ConfidenceCI
		2,              // WarmupWindows
		1,              // TailWindows
		0.95,           // SaturatedDrainRatio: mean DrainRatio < this → PERSISTENTLY_SATURATED
		0.98,           // TransientDrainRatio: mean DrainRatio < this → TRANSIENT_BACKLOG
	)
}
