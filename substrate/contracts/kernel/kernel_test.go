package kernel

import (
	"math"
	"testing"
)

// The width accessors are the only behaviour Resolution has. Each reads a width a
// resolver stated and applies a floor of one, because a consumer multiplies by these —
// a scheduler scales its sequence cap, token budget and KV budget by the data-parallel
// width — and a zero that means "unset" would silently make all three zero.
//
// Adding the fields broke no existing construction precisely because they default to
// zero, so every reader must survive a zero-valued Resolution. That is the case these
// tests exist for.
var accessors = []struct {
	name string
	set  func(*Resolution, int)
	get  func(Resolution) int
}{
	{"TensorParallel", func(r *Resolution, w int) { r.TensorParallelWidth = w }, Resolution.TensorParallel},
	{"DataParallel", func(r *Resolution, w int) { r.DataParallelWidth = w }, Resolution.DataParallel},
	{"ExpertParallel", func(r *Resolution, w int) { r.ExpertParallelWidth = w }, Resolution.ExpertParallel},
}

// TestZeroResolutionReadsAsOneRankPerAxis is the case the floor exists for: a Resolution
// whose implementation did not fill the widths must read as an unsharded layout, not as
// a layout with no ranks.
func TestZeroResolutionReadsAsOneRankPerAxis(t *testing.T) {
	var r Resolution
	for _, a := range accessors {
		if got := a.get(r); got != 1 {
			t.Errorf("zero Resolution: %s() = %d, want 1", a.name, got)
		}
	}
}

// TestWidthAccessorsReturnStatedWidthOrOne states the behaviour as a property over the
// whole int range rather than a few chosen values: a stated width of at least one is
// returned unchanged, and anything below one reads as one. The edges are included
// explicitly because they are where a floor is most easily written wrong (w == 0 rather
// than w < 1, or an off-by-one at 1).
func TestWidthAccessorsReturnStatedWidthOrOne(t *testing.T) {
	widths := []int{math.MinInt, -1 << 40, -2, -1, 0, 1, 2, 3, 8, 72, 1 << 20, math.MaxInt}
	for w := -64; w <= 64; w++ {
		widths = append(widths, w)
	}
	for _, a := range accessors {
		for _, w := range widths {
			var r Resolution
			a.set(&r, w)
			want := w
			if w < 1 {
				want = 1
			}
			if got := a.get(r); got != want {
				t.Errorf("%s() with width %d = %d, want %d", a.name, w, got, want)
			}
		}
	}
}

// TestEachAccessorReadsItsOwnWidth is metamorphic: changing one width must move exactly
// one accessor. Three widths with distinct values catch an accessor wired to the wrong
// field, which a test setting every width to the same number cannot.
func TestEachAccessorReadsItsOwnWidth(t *testing.T) {
	for i, changed := range accessors {
		base := Resolution{TensorParallelWidth: 2, DataParallelWidth: 3, ExpertParallelWidth: 5}
		after := base
		changed.set(&after, 97)
		for j, a := range accessors {
			before, now := a.get(base), a.get(after)
			if i == j && now != 97 {
				t.Errorf("setting %s's width to 97: %s() = %d", changed.name, a.name, now)
			}
			if i != j && now != before {
				t.Errorf("setting %s's width moved %s() from %d to %d", changed.name, a.name, before, now)
			}
		}
	}
}
