// sim/saturation/timeline.go
package saturation

// TimelineLabel is the per-interval saturation verdict. Unlike Level (a severity
// axis: Stable < Backlogged < Overloaded), the timeline collapses severity to a
// binary saturated/unsaturated decision and adds an ORTHOGONAL Unsure state for
// "the detector lacks enough information to decide" — Unsure is NOT a middle
// severity between Unsaturated and Saturated.
type TimelineLabel int

const (
	Unsaturated TimelineLabel = iota
	Saturated
	Unsure
)

func (l TimelineLabel) String() string {
	switch l {
	case Unsaturated:
		return "UNSATURATED"
	case Saturated:
		return "SATURATED"
	case Unsure:
		return "UNSURE"
	default:
		return "UNKNOWN"
	}
}

// MarshalJSON serializes the label as its uppercase string, mirroring Level.
func (l TimelineLabel) MarshalJSON() ([]byte, error) {
	return []byte(`"` + l.String() + `"`), nil
}

// UnmarshalJSON parses the uppercase string form. Unknown values decode to
// Unsure (the conservative "no decision" state), mirroring Level's tolerant decode.
func (l *TimelineLabel) UnmarshalJSON(data []byte) error {
	s := string(data)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		s = s[1 : len(s)-1]
	}
	switch s {
	case "UNSATURATED":
		*l = Unsaturated
	case "SATURATED":
		*l = Saturated
	default:
		*l = Unsure
	}
	return nil
}

// TimelineConfig parameterizes timeline construction.
//
// IntervalUs is the simulation-clock spacing (microseconds) between successive
// timeline points. MinRequests and MinConfidence gate the Unsure label: a point
// is Unsure when it has seen fewer than MinRequests arrivals OR the detector's
// own Confidence for that cumulative prefix is below MinConfidence. Both default
// to 0 in the zero value, which disables Unsure entirely.
type TimelineConfig struct {
	IntervalUs    int64
	MinRequests   int
	MinConfidence float64
}

// TimelinePoint is one entry in the saturation timeline: the cumulative verdict
// for all data from t=0 to ClockUs.
type TimelinePoint struct {
	ClockUs     int64         `json:"clock_us"`
	Label       TimelineLabel `json:"label"`
	Level       Level         `json:"level"`      // underlying detector severity (diagnostic)
	Score       float64       `json:"score"`      // detector score (diagnostic)
	Confidence  float64       `json:"confidence"` // detector confidence (diagnostic)
	Arrivals    int           `json:"arrivals"`
	Completions int           `json:"completions"`
}

// LabelFromResult maps a detector Result to a TimelineLabel under cfg.
//
// The Unsure gate is checked FIRST and takes precedence over the severity mapping:
// insufficient data (arrivals < MinRequests) or low detector confidence
// (Confidence < MinConfidence) yields Unsure regardless of Level. Otherwise:
// Overloaded → Saturated; Stable and Backlogged → Unsaturated (only sustained
// overload counts as saturation — transient backlog is treated as healthy).
func LabelFromResult(r Result, arrivals int, cfg TimelineConfig) TimelineLabel {
	if arrivals < cfg.MinRequests || r.Confidence < cfg.MinConfidence {
		return Unsure
	}
	if r.Level == Overloaded {
		return Saturated
	}
	return Unsaturated
}
