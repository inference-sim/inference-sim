package price

import "testing"

// A request under the tracked layout holds its working state, the engine's tracking slots and the
// speculative blocks: SGLang's extra_buffer holds 3 states with the overlap schedule and 2 without
// it; the page-aligned layout holds 2 whichever the schedule, so it is one short with overlap.
func TestTrackedRecurrentStateIsWorkingPlusTrackingSlots(t *testing.T) {
	const page = 1000.0
	for _, tc := range []struct {
		tracking, spec int
		want           float64
	}{{2, 0, 3 * page}, {1, 0, 2 * page}, {2, 3, 6 * page}} {
		got := RecurrentStateBytes(page, RecurrentCacheTracked, 4096, 16, tc.spec, 0, tc.tracking)
		if got != tc.want {
			t.Errorf("%d tracking slots, %d speculative blocks: %v bytes, want %v", tc.tracking, tc.spec, got, tc.want)
		}
	}
	if a := RecurrentStateBytes(page, RecurrentCacheAlign, 4096, 16, 0, 0, 2); a != 2*page {
		t.Errorf("the page-aligned layout ignores tracking slots: %v, want %v", a, 2*page)
	}
	if RecurrentCacheTracked.ProportionalToContext() {
		t.Error("the tracked layout does not grow with the context")
	}
}
