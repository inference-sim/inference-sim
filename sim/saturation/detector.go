// Package saturation provides post-hoc saturation detection for completed traces.
//
// NAMING NOTE (I6): This package shares the "saturation" name with sim.SaturationDetector
// (real-time flow control used by gateway queue). They serve different purposes:
//   - sim.SaturationDetector: real-time signal (float64) for admission control
//   - saturation.Detector: post-hoc classification (Result) for trace analysis
//
// Import as "github.com/inference-sim/inference-sim/sim/saturation" to disambiguate.
package saturation

import "github.com/inference-sim/inference-sim/sim"

type EventType int

const (
	Arrival    EventType = iota
	Completion
)

type Event struct {
	Timestamp    int64
	Type         EventType
	RequestID    string
	LatencyMs    float64
	InputTokens  int
	OutputTokens int
}

type Level int

const (
	Stable     Level = iota
	Backlogged
	Overloaded
)

func (l Level) String() string {
	switch l {
	case Stable:
		return "STABLE"
	case Backlogged:
		return "BACKLOGGED"
	case Overloaded:
		return "OVERLOADED"
	default:
		return "UNKNOWN"
	}
}

// MarshalJSON implements json.Marshaler for Level enum.
func (l Level) MarshalJSON() ([]byte, error) {
	return []byte(`"` + l.String() + `"`), nil
}

// UnmarshalJSON implements json.Unmarshaler for Level enum (I3).
func (l *Level) UnmarshalJSON(data []byte) error {
	// Remove quotes
	s := string(data)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		s = s[1 : len(s)-1]
	}

	switch s {
	case "STABLE":
		*l = Stable
	case "BACKLOGGED":
		*l = Backlogged
	case "OVERLOADED":
		*l = Overloaded
	default:
		*l = Stable // Default to stable for unknown values
	}
	return nil
}

type Result struct {
	Level      Level              `json:"level"`
	Score      float64            `json:"score"`
	Confidence float64            `json:"confidence"`
	Signals    map[string]float64 `json:"signals"`
}

type Detector interface {
	Name() string
	Observe(event Event)
	Detect() Result
	// Classify performs batch classification. Returns interface{} to implement sim.BatchClassifier.
	// Actual return type is Result.
	Classify(requests []sim.RequestMetrics) interface{}
	Reset()
}
