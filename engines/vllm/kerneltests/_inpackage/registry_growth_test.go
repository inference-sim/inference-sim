// Moved from blis-latency-kernel/registry_growth_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// These tests are about THIS REPOSITORY'S LOGIC under a changing registry, not about what
// the registry happens to contain today.
//
// The distinction matters because the registry is expected to grow: more widths, more
// parts, coefficients fitted where there were none, and figures refitted as better data
// arrives. A test that pins a value the registry supplies breaks on every such change
// while saying nothing about whether this kernel still composes correctly — and worse, it
// trains a reader to "fix" the test by editing the number.
//
// So each case below builds a SYNTHETIC coefficient set, states what the kernel should do
// with it, and never consults the committed registry. They would all still pass if every
// figure upstream were replaced tomorrow.
//
// What they pin is the resolution POLICY: exact widths resolve, narrower clamps up,
// unmeasured-but-in-the-search-space refuses, past-the-space clamps down, and — the
// forward-compatibility property — a width this file has never heard of is discovered
// from the registry rather than ignored.

// TestAPartGainingItsOwnFitStopsBorrowing is the growth path for the rack-scale case
// f7281fb fixed: a part swept only to 4 ranks refuses an 8-rank group, and must resolve it
// the moment that part is swept at 8.
//
// Stated against synthetic sets for both states, so it asserts the transition rather than
// either registry snapshot.
func TestAPartGainingItsOwnFitStopsBorrowing(t *testing.T) {
	build := func(widths ...int) *resolve.Coefficients {
		t.Helper()
		var entries []coefficient.Entry
		for _, w := range widths {
			entries = append(entries, tripleFor("all_reduce", "fp16", "gb300", w, 6.0)...)
		}
		return coeffs(t, entries)
	}
	at8 := func(c *resolve.Coefficients) (int, error) {
		k := &Kernel{chip: hardware.Chip{Name: "gb300"}}
		k.layout.TP = 8
		return k.groupWidth(c, collKey{Op: model.OpAllReduce, Group: price.GroupTP}, "all_reduce", "fp16", "gb300")
	}

	// Before: swept at 2 and 4 only. 8 is in the search space, so it must refuse rather
	// than borrow the 4-rank floor.
	if _, err := at8(build(2, 4)); err == nil {
		t.Error("a part swept only to 4 ranks must refuse an 8-rank group, not borrow")
	}
	// After: the sweep lands. No code change, and the figure is now used.
	got, err := at8(build(2, 4, 8))
	if err != nil {
		t.Fatalf("once the part is swept at 8 ranks the group must resolve: %v", err)
	}
	if got != 8 {
		t.Errorf("resolved to %d, want 8", got)
	}
}
