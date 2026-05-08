package workload

import (
	"fmt"
	"math"
	"time"
)

// RequestInterval represents a request's active period [ArrivalUs, CompletionUs).
// Used as input to backlog-drift saturation analysis (BC-2).
type RequestInterval struct {
	ArrivalUs    int64
	CompletionUs int64
}

// BacklogDriftConfig configures the backlog-drift saturation analyzer.
type BacklogDriftConfig struct {
	WindowSize   time.Duration // Window width for sampling and per-window metrics (BC-1)
	MinWindows   int           // Minimum complete windows required for classification (BC-7)
	PeakRatio    float64       // Peak-to-mean threshold for TRANSIENT_BACKLOG detection (BC-6)
	ConfidenceCI float64       // Confidence level for slope significance test (BC-3)
}

// NewBacklogDriftConfig creates a BacklogDriftConfig with validation (BC-10, BC-14, R3).
// Panics if any parameter is invalid (NaN, Inf, out of range).
func NewBacklogDriftConfig(windowSize time.Duration, minWindows int, peakRatio, confidenceCI float64) BacklogDriftConfig {
	if windowSize <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: WindowSize must be > 0, got %v", windowSize))
	}
	if minWindows <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: MinWindows must be > 0, got %d", minWindows))
	}
	if peakRatio <= 0 || math.IsNaN(peakRatio) || math.IsInf(peakRatio, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: PeakRatio must be a finite value > 0, got %f", peakRatio))
	}
	if confidenceCI <= 0 || confidenceCI >= 1 || math.IsNaN(confidenceCI) || math.IsInf(confidenceCI, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: ConfidenceCI must be in (0, 1), got %f", confidenceCI))
	}
	return BacklogDriftConfig{
		WindowSize:   windowSize,
		MinWindows:   minWindows,
		PeakRatio:    peakRatio,
		ConfidenceCI: confidenceCI,
	}
}

// DefaultBacklogDriftConfig returns the default configuration per issue #1298.
func DefaultBacklogDriftConfig() BacklogDriftConfig {
	return BacklogDriftConfig{
		WindowSize:   60 * time.Second,
		MinWindows:   5,
		PeakRatio:    2.0,
		ConfidenceCI: 0.95,
	}
}

// WindowMetrics captures per-window saturation metrics (BC-1).
type WindowMetrics struct {
	StartUs      int64   // Window start timestamp (µs)
	EndUs        int64   // Window end timestamp (µs)
	NumEntered   int     // Requests with arrival in [start, end)
	NumLeft      int     // Requests with completion in [start, end)
	ActiveStart  int     // Active requests at window start
	ActiveEnd    int     // Active requests at window end
	DeltaBacklog int     // ActiveEnd - ActiveStart (must equal NumEntered - NumLeft, BC-1)
	DrainRatio   float64 // NumLeft / NumEntered (NaN if NumEntered==0)
}

// BacklogDriftReport contains saturation classification results (BC-4, BC-5, BC-6, BC-7).
type BacklogDriftReport struct {
	Classification string          `json:"classification"` // "UNSATURATED", "TRANSIENT_BACKLOG", "PERSISTENTLY_SATURATED"
	Slope          float64         `json:"slope"`          // Linear regression slope (req/µs)
	SlopeLower     float64         `json:"slope_lower"`    // Slope CI lower bound
	SlopeUpper     float64         `json:"slope_upper"`    // Slope CI upper bound
	InitialBacklog int             `json:"initial_backlog"`
	FinalBacklog   int             `json:"final_backlog"`
	PeakInFlight   int             `json:"peak_in_flight"`
	MeanInFlight   float64         `json:"mean_in_flight"`
	Windows        []WindowMetrics `json:"windows"`
	Note           string          `json:"note,omitempty"`           // Explanation (e.g., "observation too short")
	Recommendation string          `json:"recommendation,omitempty"` // User-facing guidance
}
