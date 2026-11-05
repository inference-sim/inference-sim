package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
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

// TestANewlyFittedWidthIsUsedWithoutACodeChange is the property the growth question turns
// on: when the registry gains a width, the kernel must USE it.
//
// Previously measuredWidths probed a hardcoded list, so a sweep at any other width was
// invisible — committed upstream, never looked for, and the deployment needing it either
// clamped to a narrower figure or was refused. Discovering widths from the coefficient
// names means the registry is the single source of truth for what exists.
func TestANewlyFittedWidthIsUsedWithoutACodeChange(t *testing.T) {
	// A width deliberately ABSENT from candidateRankWidths, standing for whatever width
	// upstream fits next. The point is that this file does not need to know it.
	const novel = 32

	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8, novel} {
		entries = append(entries, tripleFor("alltoall", "fp16", "h200", w, 6.4)...)
	}
	c := coeffs(t, entries)

	if got := measuredWidths(c, "alltoall", "fp16", "h200"); len(got) != 4 ||
		got[3] != novel {
		t.Fatalf("measured widths = %v; a %d-rank sweep in the registry must be found "+
			"even though candidateRankWidths does not mention it", got, novel)
	}

	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	k.layout.ExpertWidth = novel
	got, err := k.groupWidth(c, collKey{Op: model.OpAll2All, Group: price.GroupExpert}, "alltoall", "fp16", "h200")
	if err != nil {
		t.Fatalf("a %d-rank group must resolve once the width is fitted: %v", novel, err)
	}
	if got != novel {
		t.Errorf("resolved to %d, want the exact %d-rank figure the registry now carries",
			got, novel)
	}
}

// TestRefittingAValueChangesNothingAboutResolution separates the two things a registry
// update can do. Changing a VALUE must not change WHICH coefficient is selected.
//
// Without this, a test suite cannot tell "the registry was refitted" from "the resolver
// regressed", and the usual response to the resulting failure is to edit the expected
// number — which silently accepts whatever the resolver now does.
func TestRefittingAValueChangesNothingAboutResolution(t *testing.T) {
	widths := []int{2, 4, 8}
	resolveAt := func(floor float64) int {
		t.Helper()
		var entries []coefficient.Entry
		for _, w := range widths {
			entries = append(entries, tripleFor("alltoall", "fp16", "h200", w, floor)...)
		}
		k := &Kernel{chip: hardware.Chip{Name: "h200"}}
		k.layout.ExpertWidth = 4
		got, err := k.groupWidth(coeffs(t, entries),
			collKey{Op: model.OpAll2All, Group: price.GroupExpert},
			"alltoall", "fp16", "h200")
		if err != nil {
			t.Fatalf("floor %v: %v", floor, err)
		}
		return got
	}
	if a, b := resolveAt(6.4), resolveAt(99.9); a != b {
		t.Errorf("a refit moved the selected width from %d to %d; resolution depends on "+
			"WHICH widths exist, never on their values", a, b)
	}
}

// TestResolutionPolicyAcrossTheWidthSpace pins the whole policy as a table, in terms of a
// synthetic part swept at 2, 4 and 8.
//
// One table rather than four tests because the cases are a single decision and reading
// them together is how a reader checks the boundaries line up. Expressed relative to the
// search space's own bounds, so extending candidateRankWidths upstream does not invalidate
// the expectations.
func TestResolutionPolicyAcrossTheWidthSpace(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("alltoall", "fp16", "h200", w, 6.4)...)
	}
	c := coeffs(t, entries)
	spaceMax := candidateRankWidths[len(candidateRankWidths)-1]

	for _, tc := range []struct {
		what   string
		width  int
		want   int
		wantOK bool
	}{
		{"narrower than anything measured clamps up to the narrowest", 1, 2, true},
		{"an exactly measured width resolves to itself", 4, 4, true},
		{"the widest measured width resolves to itself", 8, 8, true},
		{"inside the search space but unmeasured refuses", spaceMax, 0, false},
		{"past the search space clamps to the widest measured", spaceMax * 2, 8, true},
	} {
		t.Run(tc.what, func(t *testing.T) {
			k := &Kernel{chip: hardware.Chip{Name: "h200"}}
			k.layout.ExpertWidth = tc.width
			got, err := k.groupWidth(c, collKey{Op: model.OpAll2All, Group: price.GroupExpert}, "alltoall", "fp16", "h200")
			if tc.wantOK {
				if err != nil {
					t.Fatalf("width %d: %v", tc.width, err)
				}
				if got != tc.want {
					t.Errorf("width %d resolved to %d, want %d", tc.width, got, tc.want)
				}
				return
			}
			if err == nil {
				t.Fatalf("width %d resolved to %d; it must refuse", tc.width, got)
			}
		})
	}
}

// TestAnIncompleteTripleIsNotAMeasuredWidth guards the completeness rule against a
// half-landed registry change — a floor committed before its rates, say.
//
// A width with only some of floor/peak/transition cannot price a collective, so treating
// it as measured would select it and then fail a layer down with a confusing message. That
// is the accidental-guard failure mode f7281fb's own commit message describes.
func TestAnIncompleteTripleIsNotAMeasuredWidth(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4} {
		entries = append(entries, tripleFor("alltoall", "fp16", "h200", w, 6.4)...)
	}
	// An 8-rank floor with no rates beside it.
	entries = append(entries, coefficient.Entry{
		Name: "collective_floor_alltoall_fp16_8rank_h200", Value: 7.4,
	})
	c := coeffs(t, entries)

	got := measuredWidths(c, "alltoall", "fp16", "h200")
	for _, w := range got {
		if w == 8 {
			t.Fatalf("measured widths = %v; 8 carries a floor but no rates, so it cannot "+
				"price a collective and is not a measured width", got)
		}
	}
}

// TestWidthDiscoveryIgnoresOtherPartsAndOperations is the scoping property.
//
// Names are parsed out of one flat namespace, so the parse has to be exact: a 16-rank
// figure for another chip, another operation or another dtype must not make this part look
// swept at 16. Getting this wrong would resolve a deployment against a coefficient fitted
// for different hardware, which is worse than refusing it.
func TestWidthDiscoveryIgnoresOtherPartsAndOperations(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("alltoall", "fp16", "h200", w, 6.4)...)
	}
	// All at 16 ranks, none of them this (op, dtype, chip).
	entries = append(entries, tripleFor("alltoall", "fp16", "b200", 16, 6.4)...)
	entries = append(entries, tripleFor("all_reduce", "fp16", "h200", 16, 6.0)...)
	entries = append(entries, tripleFor("alltoall", "int8", "h200", 16, 6.4)...)
	c := coeffs(t, entries)

	if got := measuredWidths(c, "alltoall", "fp16", "h200"); len(got) != 3 {
		t.Errorf("measured widths = %v, want [2 4 8]: a 16-rank figure for another part, "+
			"operation or dtype says nothing about this one", got)
	}
}

// TestMalformedWidthNamesAreIgnored keeps name parsing from admitting nonsense, since the
// width now comes from a string in the registry rather than from a constant here.
func TestMalformedWidthNamesAreIgnored(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4} {
		entries = append(entries, tripleFor("alltoall", "fp16", "h200", w, 6.4)...)
	}
	for _, bad := range []string{
		"collective_floor_alltoall_fp16_rank_h200",     // no digits
		"collective_floor_alltoall_fp16_xrank_h200",    // not a number
		"collective_floor_alltoall_fp16_0rank_h200",    // zero ranks
		"collective_floor_alltoall_fp16_-4rank_h200",   // negative
		"collective_floor_alltoall_fp16_8rank_h200_x2", // chip does not match
	} {
		entries = append(entries, coefficient.Entry{Name: bad, Value: 1})
	}
	c := coeffs(t, entries)

	if got := measuredWidths(c, "alltoall", "fp16", "h200"); len(got) != 2 {
		t.Errorf("measured widths = %v, want [2 4]: a malformed name is not a width", got)
	}
}
