// Package saturation provides post-hoc saturation detection for completed traces.
// Distinct from sim.SaturationDetector (real-time flow control).
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
	Classify(requests []sim.RequestMetrics) Result
	Reset()
}
