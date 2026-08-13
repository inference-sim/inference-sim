// sim/saturation/reduce_test.go
package saturation

import (
	"reflect"
	"testing"
)

// rec is a terse constructor for a TraceRecord with just the fields the reducer
// reads (Timestamp, Detector, Result.Level). Score/Confidence/Signals are
// irrelevant to the last-window plurality rule and are left zero.
func rec(ts int64, detector string, level Level) TraceRecord {
	return TraceRecord{
		Timestamp: ts,
		Detector:  detector,
		Result:    Result{Level: level},
	}
}

// TestReduceOne_Contracts is the table-driven contract suite for the last-window
// plurality rule (#1517). Each case is one detector's records; the reducer keeps
// records within windowUs of the max timestamp, takes the plurality level, and
// breaks count-ties toward the more severe level.
func TestReduceOne_Contracts(t *testing.T) {
	tests := []struct {
		name     string
		records  []TraceRecord
		windowUs int64
		want     Level
	}{
		{
			// Plurality: all three kept (window covers them), OVERLOADED wins 2-1.
			name: "plurality_overloaded_wins",
			records: []TraceRecord{
				rec(0, "composite", Stable),
				rec(10, "composite", Overloaded),
				rec(20, "composite", Overloaded),
			},
			windowUs: 100,
			want:     Overloaded,
		},
		{
			// Windowing excludes old records: lastT=20, keep Timestamp>=15, so only
			// OVERLOADED@20 survives; STABLE@0 is out of window.
			name: "windowing_excludes_old",
			records: []TraceRecord{
				rec(0, "composite", Stable),
				rec(10, "composite", Overloaded),
				rec(20, "composite", Overloaded),
			},
			windowUs: 5,
			want:     Overloaded,
		},
		{
			// Empty group → STABLE degenerate default (R20).
			name:     "empty_defaults_stable",
			records:  nil,
			windowUs: 100,
			want:     Stable,
		},
		{
			// Count tie {STABLE:1, OVERLOADED:1} → more severe OVERLOADED.
			name: "tie_stable_overloaded_picks_overloaded",
			records: []TraceRecord{
				rec(0, "composite", Stable),
				rec(0, "composite", Overloaded),
			},
			windowUs: 100,
			want:     Overloaded,
		},
		{
			// Count tie {STABLE:1, BACKLOGGED:1} → more severe BACKLOGGED.
			name: "tie_stable_backlogged_picks_backlogged",
			records: []TraceRecord{
				rec(0, "composite", Stable),
				rec(0, "composite", Backlogged),
			},
			windowUs: 100,
			want:     Backlogged,
		},
		{
			// Three-way tie {STABLE:1, BACKLOGGED:1, OVERLOADED:1} → most severe.
			name: "three_way_tie_picks_overloaded",
			records: []TraceRecord{
				rec(0, "composite", Stable),
				rec(1, "composite", Backlogged),
				rec(2, "composite", Overloaded),
			},
			windowUs: 100,
			want:     Overloaded,
		},
		{
			name: "degenerate_all_stable",
			records: []TraceRecord{
				rec(0, "composite", Stable),
				rec(10, "composite", Stable),
			},
			windowUs: 100,
			want:     Stable,
		},
		{
			name: "degenerate_all_overloaded",
			records: []TraceRecord{
				rec(0, "composite", Overloaded),
				rec(10, "composite", Overloaded),
			},
			windowUs: 100,
			want:     Overloaded,
		},
		{
			// Order-independence: same multiset, shuffled, must yield the same label
			// (INV-6). Plurality BACKLOGGED (2) over STABLE (1) / OVERLOADED (1).
			name: "order_independent",
			records: []TraceRecord{
				rec(20, "composite", Backlogged),
				rec(0, "composite", Stable),
				rec(30, "composite", Overloaded),
				rec(10, "composite", Backlogged),
			},
			windowUs: 100,
			want:     Backlogged,
		},
		{
			// Single record → that record's level.
			name:     "single_record",
			records:  []TraceRecord{rec(5, "composite", Backlogged)},
			windowUs: 100,
			want:     Backlogged,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ReduceOne(tt.records, tt.windowUs); got != tt.want {
				t.Errorf("ReduceOne() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestReduceOne_ExactWindowBoundaryInclusive verifies a record exactly on the
// lastT-windowUs boundary is KEPT (the rule is Timestamp >= lastT - windowUs).
func TestReduceOne_ExactWindowBoundaryInclusive(t *testing.T) {
	// lastT=100, windowUs=50 → boundary=50. STABLE@50 is kept, so {STABLE:1,
	// OVERLOADED:1} ties → OVERLOADED. If the boundary were exclusive, only
	// OVERLOADED@100 would remain (still OVERLOADED) — so to actually observe the
	// inclusion we make STABLE outvote: two STABLE at/after boundary vs one OVER.
	records := []TraceRecord{
		rec(50, "composite", Stable),
		rec(75, "composite", Stable),
		rec(100, "composite", Overloaded),
	}
	if got := ReduceOne(records, 50); got != Stable {
		t.Errorf("boundary-inclusive ReduceOne() = %v, want STABLE (record at lastT-windowUs must be kept)", got)
	}
}

// TestReduceOne_AllLevelsOutOfRange_DefaultsToStable is the defensive-guard
// regression (PR #1546 review): if every in-window record carries a Level outside
// [Stable, Overloaded], the bounds check drops them all, leaving zero counts. The
// reducer must then fall back to the "empty group → STABLE" default rather than
// returning the most-severe level with a zero count. Unreachable with real
// detectors (they only emit valid levels), but locks the guard.
func TestReduceOne_AllLevelsOutOfRange_DefaultsToStable(t *testing.T) {
	const bogus Level = 99 // outside [Stable=0, Overloaded=2]
	records := []TraceRecord{
		rec(0, "composite", bogus),
		rec(10, "composite", bogus),
	}
	if got := ReduceOne(records, 100); got != Stable {
		t.Errorf("ReduceOne() with only out-of-range levels = %v, want STABLE (all filtered ⇒ degenerate default)", got)
	}
}

// TestReduceAll_GroupsByDetector verifies ReduceAll splits a flat slice by the
// Detector field and reduces each group independently.
func TestReduceAll_GroupsByDetector(t *testing.T) {
	records := []TraceRecord{
		rec(0, "composite", Stable),
		rec(10, "composite", Stable),
		rec(0, "threshold", Overloaded),
		rec(10, "threshold", Overloaded),
	}
	got := ReduceAll(records, 100)
	want := map[string]Level{
		"composite": Stable,
		"threshold": Overloaded,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ReduceAll() = %v, want %v", got, want)
	}
}

// TestReduceAll_SingleDetectorStillMap verifies a one-detector selection yields a
// one-key map (never a bare level) — the uniform stdout shape (#1517).
func TestReduceAll_SingleDetectorStillMap(t *testing.T) {
	records := []TraceRecord{
		rec(0, "composite", Overloaded),
	}
	got := ReduceAll(records, 100)
	want := map[string]Level{"composite": Overloaded}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ReduceAll() = %v, want %v", got, want)
	}
}

// TestReduceAll_EmptyInputEmptyMap verifies zero records yield an empty (non-nil
// is not required) map so cmd's len()>0 guard drops the stdout field (BC-8).
func TestReduceAll_EmptyInputEmptyMap(t *testing.T) {
	got := ReduceAll(nil, 100)
	if len(got) != 0 {
		t.Errorf("ReduceAll(nil) = %v, want empty map", got)
	}
}

// TestReduceAll_PerGroupWindowing verifies each group's window is anchored to
// that group's own max timestamp, not a global max — a detector that stopped
// early is not truncated by a later detector's records.
func TestReduceAll_PerGroupWindowing(t *testing.T) {
	records := []TraceRecord{
		// composite ends at 20; its own window keeps both records.
		rec(0, "composite", Overloaded),
		rec(20, "composite", Overloaded),
		// threshold ends at 1000, far later; must not pull composite's window.
		rec(1000, "threshold", Stable),
	}
	got := ReduceAll(records, 100)
	if got["composite"] != Overloaded {
		t.Errorf("composite = %v, want OVERLOADED (its window must anchor to its own max ts)", got["composite"])
	}
	if got["threshold"] != Stable {
		t.Errorf("threshold = %v, want STABLE", got["threshold"])
	}
}
