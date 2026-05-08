package workload

import (
	"fmt"

	sim "github.com/inference-sim/inference-sim/sim"
)

// SaturationAnalyzer defines the interface for saturation detection algorithms.
// Different implementations can use different metrics and thresholds to classify
// system saturation state.
//
// Extension recipe: To add a new saturation algorithm:
//  1. Create a new type implementing this interface (e.g., UtilizationBasedAnalyzer)
//  2. Define algorithm-specific configuration struct
//  3. Implement the Analyze method returning a SaturationReport
//  4. Add a factory function (e.g., NewUtilizationBasedAnalyzer)
//  5. Update CLI flags to support the new algorithm
//
// Example algorithms:
//  - BacklogDriftAnalyzer: Linear regression on active request counts (current)
//  - UtilizationBasedAnalyzer: GPU/KV cache utilization thresholds
//  - QueueDepthAnalyzer: Queue depth percentile analysis
//  - LatencyBasedAnalyzer: Latency degradation detection
type SaturationAnalyzer interface {
	// Analyze examines request timing and system behavior to classify saturation state.
	// Returns a SaturationReport with classification, metrics, and human-readable notes.
	//
	// Parameters:
	//   requests: Slice of completed/in-flight requests with timing data
	//   simEndUs: End time of observation window in microseconds
	//
	// The analyzer must handle edge cases:
	//   - Empty request slice
	//   - Requests with incomplete timing data
	//   - Very short observation windows
	Analyze(requests []*sim.Request, simEndUs int64) SaturationReport
}

// SaturationReport contains the classification result and supporting metrics.
// This is the common output format for all saturation analyzers.
type SaturationReport struct {
	// Classification is one of: "UNSATURATED", "TRANSIENT_BACKLOG", "PERSISTENTLY_SATURATED"
	Classification string `json:"classification"`

	// Algorithm identifies which analyzer produced this report (e.g., "backlog-drift", "utilization-based")
	Algorithm string `json:"algorithm"`

	// Note is a human-readable explanation of the classification decision
	Note string `json:"note"`

	// Recommendation suggests actions based on the classification
	Recommendation string `json:"recommendation"`

	// AlgorithmData contains algorithm-specific metrics (e.g., BacklogDriftData for backlog-drift)
	// Serialized as nested JSON object
	AlgorithmData interface{} `json:"algorithm_data,omitempty"`
}

// BacklogDriftData contains algorithm-specific metrics for the backlog-drift analyzer.
// Embedded in SaturationReport.AlgorithmData when using BacklogDriftAnalyzer.
type BacklogDriftData struct {
	Slope          float64         `json:"slope"`           // Linear regression slope (req/µs)
	SlopeLower     float64         `json:"slope_lower"`     // Slope CI lower bound
	SlopeUpper     float64         `json:"slope_upper"`     // Slope CI upper bound
	InitialBacklog int             `json:"initial_backlog"` // Active requests at observation start
	FinalBacklog   int             `json:"final_backlog"`   // Active requests at observation end
	PeakInFlight   int             `json:"peak_in_flight"`  // Maximum active requests
	MeanInFlight   float64         `json:"mean_in_flight"`  // Mean active requests across windows
	Windows        []WindowMetrics `json:"windows"`         // Per-window metrics
}

// NewSaturationAnalyzer creates a saturation analyzer based on algorithm name.
// Supported algorithms: "backlog-drift"
// Returns error if algorithm is unknown or config is invalid.
func NewSaturationAnalyzer(algorithm string, config interface{}) (SaturationAnalyzer, error) {
	switch algorithm {
	case "backlog-drift":
		cfg, ok := config.(BacklogDriftConfig)
		if !ok {
			return nil, fmt.Errorf("backlog-drift analyzer requires BacklogDriftConfig, got %T", config)
		}
		return NewBacklogDriftAnalyzer(cfg), nil
	default:
		return nil, fmt.Errorf("unknown saturation algorithm: %s (supported: backlog-drift)", algorithm)
	}
}
