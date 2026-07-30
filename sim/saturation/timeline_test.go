// sim/saturation/timeline_test.go
package saturation

import (
	"testing"
)

// --- LabelFromResult contracts ---

func TestLabelFromResult_Mapping(t *testing.T) {
	// cfg with no Unsure gate (thresholds 0) so mapping is exercised directly.
	cfg := TimelineConfig{IntervalUs: 1, MinRequests: 0, MinConfidence: 0}
	cases := []struct {
		name  string
		level Level
		want  TimelineLabel
	}{
		// BC: Overloaded → SATURATED
		{"overloaded", Overloaded, Saturated},
		// BC: Stable → UNSATURATED
		{"stable", Stable, Unsaturated},
		// BC: Backlogged → UNSATURATED (transient backlog is treated as healthy)
		{"backlogged", Backlogged, Unsaturated},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := LabelFromResult(Result{Level: tc.level, Confidence: 1.0}, 100, cfg)
			if got != tc.want {
				t.Errorf("level %v: got %v, want %v", tc.level, got, tc.want)
			}
		})
	}
}

func TestLabelFromResult_UnsureOnFewRequests(t *testing.T) {
	// BC: arrivals < MinRequests → UNSURE regardless of Level (orthogonality).
	cfg := TimelineConfig{IntervalUs: 1, MinRequests: 20, MinConfidence: 0}
	got := LabelFromResult(Result{Level: Overloaded, Confidence: 1.0}, 5, cfg)
	if got != Unsure {
		t.Errorf("few requests: got %v, want UNSURE", got)
	}
}

func TestLabelFromResult_UnsureOnLowConfidence(t *testing.T) {
	// BC: Confidence < MinConfidence → UNSURE regardless of Level.
	cfg := TimelineConfig{IntervalUs: 1, MinRequests: 0, MinConfidence: 0.5}
	got := LabelFromResult(Result{Level: Overloaded, Confidence: 0.3}, 100, cfg)
	if got != Unsure {
		t.Errorf("low confidence: got %v, want UNSURE", got)
	}
}

func TestLabelFromResult_UnsurePrecedence(t *testing.T) {
	// BC: the Unsure gate is checked before the severity mapping — an Overloaded
	// result with insufficient data is UNSURE, not SATURATED.
	cfg := TimelineConfig{IntervalUs: 1, MinRequests: 20, MinConfidence: 0.5}
	got := LabelFromResult(Result{Level: Overloaded, Confidence: 0.1}, 3, cfg)
	if got != Unsure {
		t.Errorf("unsure precedence: got %v, want UNSURE", got)
	}
}

func TestLabelFromResult_NeverUnsureWhenThresholdsZero(t *testing.T) {
	// BC: with MinRequests=0 and MinConfidence=0, a point is never UNSURE.
	cfg := TimelineConfig{IntervalUs: 1, MinRequests: 0, MinConfidence: 0}
	if got := LabelFromResult(Result{Level: Stable, Confidence: 0}, 0, cfg); got == Unsure {
		t.Errorf("zero thresholds should never yield UNSURE, got %v", got)
	}
}

func TestTimelineLabel_JSONRoundTrip(t *testing.T) {
	for _, l := range []TimelineLabel{Unsaturated, Saturated, Unsure} {
		b, err := l.MarshalJSON()
		if err != nil {
			t.Fatalf("marshal %v: %v", l, err)
		}
		var got TimelineLabel
		if err := got.UnmarshalJSON(b); err != nil {
			t.Fatalf("unmarshal %s: %v", b, err)
		}
		if got != l {
			t.Errorf("round trip: %v → %s → %v", l, b, got)
		}
	}
}
