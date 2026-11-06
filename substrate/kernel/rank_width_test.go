package latencykernel

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/artifacttest"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// The three coefficients liftCollectiveFloors needs for one (op, dtype, width). A
// width holding fewer than all three cannot price a collective, so measuredWidths
// does not count it.
func tripleFor(op, dtype, chip string, ranks int, floor float64) []coefficient.Entry {
	stem := op + "_" + dtype + "_" + itoa(ranks) + "rank_" + chip
	return []coefficient.Entry{
		{Name: "collective_floor_" + stem, Value: floor},
		{Name: "collective_peak_rate_" + stem, Value: 200},
		{Name: "collective_transition_rate_" + stem, Value: 150},
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// coeffs builds a resolved lookup over entries with no scope, so every entry applies.
func coeffs(t *testing.T, entries []coefficient.Entry) *resolve.Coefficients {
	t.Helper()
	set := &coefficient.Set{Name: "test-set", Coefficients: entries}
	c, err := resolve.Load([]*coefficient.Set{set}, resolve.Scope{})
	if err != nil {
		t.Fatalf("resolve.Load: %v", err)
	}
	return c
}

// A part swept at 2 and 4 ranks only — which is what a Grace-Blackwell tray is, four
// GPUs behind one NVSwitch domain — must REFUSE an 8-rank group rather than price it
// at the 4-rank floor. The measured 8/4 floor ratio across the seven parts that do
// carry both is 1.53x to 1.91x for all_reduce, so the substitution is not a rounding
// error; it is a systematic under-price of the term, and silently making it is what
// liftCollectiveFloors' doc comment promises not to do.
func TestGroupWidthRefusesAnUnmeasuredWidthInsideTheRange(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "gb300", w, 9.37)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "gb300"}}
	k.layout.TP = 8

	_, err := k.groupWidth(c, collKey{Op: model.OpAllReduce, Group: price.GroupTP}, "all_reduce", "fp16", "gb300")
	if err == nil {
		t.Fatal("an 8-rank group on a part swept at [2 4] must be an error, not a clamp")
	}
	for _, want := range []string{"8-rank", "all_reduce", "gb300", "[2 4]"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error must name %q so a reader can act on it; got: %v", want, err)
		}
	}
}

// A group PAST THE WHOLE SEARCH SPACE clamps: there is no sweep at any such width for
// any part, so refusing would decline to price a deployment the data merely does not
// reach. The clamp understates the floor, which is why it is recorded in provenance
// rather than treated as equivalent to a measurement.
func TestGroupWidthClampsPastTheWholeSearchSpace(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "h200", w, 6.01)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	// 32 is wider than any width any comm sweep in this project carries.
	k.layout.TP = 32

	got, err := k.groupWidth(c, collKey{Op: model.OpAllReduce, Group: price.GroupTP}, "all_reduce", "fp16", "h200")
	if err != nil {
		t.Fatalf("a width past the search space must clamp: %v", err)
	}
	if got != 8 {
		t.Errorf("want clamp to 8, got %d", got)
	}
}

// 16 ranks IS in the search space — gb300's vLLM all-reduce sweep carries it, and so
// does gb200's nccl 2.27.7 collection. So a part swept only to 8 must refuse a
// 16-rank group rather than clamp: the data for that width exists somewhere, and the
// honest answer is to fit it for this part, not to reuse a narrower floor.
func TestGroupWidthRefusesSixteenRanksWhenOnlyEightIsMeasured(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "h200", w, 6.01)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	k.layout.TP = 16

	if _, err := k.groupWidth(c, collKey{Op: model.OpAllReduce, Group: price.GroupTP}, "all_reduce", "fp16", "h200"); err == nil {
		t.Fatal("16 ranks is a swept width elsewhere; an unmeasured part must error")
	}
}

// An exact hit resolves to itself. This is the ordinary path for every part the
// registry sweeps at the deployment's width, and it must not be disturbed by the
// refusal logic above.
func TestGroupWidthResolvesAnExactlyMeasuredWidth(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "h200", w, 6.01)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	k.layout.TP = 4

	got, err := k.groupWidth(c, collKey{Op: model.OpAllReduce, Group: price.GroupTP}, "all_reduce", "fp16", "h200")
	if err != nil {
		t.Fatalf("a measured width must resolve: %v", err)
	}
	if got != 4 {
		t.Errorf("want 4, got %d", got)
	}
}

// An all-to-all spans the EXPERT width, not the tensor-parallel one. A test that only
// drove TP would pass with the two confused, which would misprice every MoE
// deployment whose expert width differs from its TP width.
func TestGroupWidthUsesExpertWidthForAllToAll(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("alltoall", "fp16", "h200", w, 6.4)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	k.layout.TP = 2
	k.layout.ExpertWidth = 8

	got, err := k.groupWidth(c, collKey{Op: model.OpAll2All, Group: price.GroupExpert}, "alltoall", "fp16", "h200")
	if err != nil {
		t.Fatalf("alltoall at expert width 8: %v", err)
	}
	if got != 8 {
		t.Errorf("alltoall must use ExpertWidth (8), got %d", got)
	}
}

// A width carrying only part of the triple is not a measured width. Counting it would
// let groupWidth return a width whose floor or transition rate is then missing, which
// surfaces as a confusing "no measured floor" error one layer down instead of the
// specific refusal this returns.
func TestMeasuredWidthsRequiresTheCompleteTriple(t *testing.T) {
	entries := tripleFor("all_reduce", "fp16", "h200", 2, 6.01)
	// 4 ranks: floor and peak present, transition rate absent.
	entries = append(entries,
		coefficient.Entry{Name: "collective_floor_all_reduce_fp16_4rank_h200", Value: 9.44},
		coefficient.Entry{Name: "collective_peak_rate_all_reduce_fp16_4rank_h200", Value: 200},
	)
	c := coeffs(t, entries)

	got := measuredWidths(c, "all_reduce", "fp16", "h200")
	if len(got) != 1 || got[0] != 2 {
		t.Errorf("an incomplete width must not count as measured; want [2], got %v", got)
	}
}

// The memory families keyed on rank count -- communicator bytes and activation multiples --
// come from NVIDIA's descriptor and memory model rather than a sweep, so they SNAP to a
// declared width instead of refusing, and must not consult the collective widths: a part
// can declare an 8-rank communicator while its collectives were swept only at 2 and 4.
//
// The property, over every width a deployment can state: the snapped width is one the
// family declares; it is the widest declared at or below the request; and below every
// declared width it is the narrowest. That is NVIDIA's own indexing at min(tp, 8) for every
// width its tables carry.
func TestMemoryFamiliesSnapToTheirOwnDeclaredWidths(t *testing.T) {
	for _, fam := range []struct {
		format   string
		declared []int
	}{
		{"nccl_communicator_bytes_%drank", []int{2, 4, 8}},
		{"activation_buffer_count_moe_%drank", []int{1, 2, 4, 8}},
	} {
		var entries []coefficient.Entry
		for _, w := range fam.declared {
			entries = append(entries, coefficient.Entry{Name: fmt.Sprintf(fam.format, w), Value: 1})
		}
		// Collectives swept at 2 and 4 only, as on a rack-scale tray.
		for _, w := range []int{2, 4} {
			entries = append(entries, tripleFor("all_reduce", "fp16", "gb300", w, 9.37)...)
		}
		c := coeffs(t, entries)
		for width := 1; width <= 20; width++ {
			got := snapDeclared(c, fam.format, width)
			want := fam.declared[0]
			for _, w := range fam.declared {
				if w <= width {
					want = w
				}
			}
			if got != want {
				t.Errorf("%s at width %d: snapped to %d, want %d", fam.format, width, got, want)
			}
		}
	}
}

// END TO END, against the committed registry rather than literals.
//
// States the PROPERTY and not the registry's inventory. An earlier version of this test
// asserted that gb200-nvl72 at tp=8 refuses, because that part was swept at 2 and 4 ranks
// only -- true when written, and false as soon as the registry fitted its 8- and 16-rank
// groups from a collection that had carried them all along. A test that pins which widths
// a part happens to have fails on exactly the change it should welcome, and says nothing
// about the behaviour worth protecting.
//
// The behaviour worth protecting: whatever the registry contains, a width it does NOT
// contain must be refused rather than silently served by a narrower one. Borrowing a
// narrower width's floor understates an 8-rank all-reduce by 1.53x to 1.91x across the
// parts measured at both widths, and doing that quietly is what liftCollectiveFloors'
// doc comment promises not to do.
func TestAnUnmeasuredWidthIsRefusedAgainstTheCommittedRegistry(t *testing.T) {
	names := []string{"cost-model-primitives", "cost-model-collectives"}
	var sets []*coefficient.Set
	for _, n := range names {
		path := filepath.Join(registryRoot, "coefficients", n+".yaml")
		s, err := contracts.LoadCoefficientSet(path)
		artifacttest.RequireArtifact(t, registryRoot, path, "registry", err)
		sets = append(sets, s)
	}

	const chip = "gb200-nvl72"
	key := strings.ReplaceAll(chip, "-", "_")
	c, err := resolve.Load(sets, resolve.Scope{Hardware: chip})
	if err != nil {
		t.Fatalf("resolve.Load: %v", err)
	}

	// Read what this part actually carries, so the assertions below follow the registry
	// instead of a remembered snapshot of it.
	measured := measuredWidths(c, "all_reduce", "fp16", key)
	if len(measured) == 0 {
		t.Skip("the registry carries no all_reduce widths for this part")
	}
	t.Logf("%s carries %v-rank all_reduce", chip, measured)

	// Every measured width must resolve to itself. A refusal here would mean the probe
	// and liftCollectiveFloors disagree about what is present.
	for _, w := range measured {
		k := &Kernel{chip: hardware.Chip{Name: chip}}
		k.layout.TP = w
		if got, err := k.groupWidth(c, collKey{Op: model.OpAllReduce, Group: price.GroupTP}, "all_reduce", "fp16", key); err != nil || got != w {
			t.Errorf("%s at tp=%d must resolve to %d; got %d, err %v", chip, w, w, got, err)
		}
	}

	// A width inside the search space that this part does NOT carry must refuse, and the
	// message must name the widths it does, so a reader can act on it. Found rather than
	// hardcoded: if the registry ever covers every candidate width, there is nothing to
	// assert and the test says so.
	var absent int
	for _, w := range candidateRankWidths {
		if !slices.Contains(measured, w) {
			absent = w
			break
		}
	}
	if absent == 0 {
		t.Logf("every candidate width %v is measured for %s; the refusal path has no "+
			"case to exercise on this part", candidateRankWidths, chip)
		return
	}
	k := &Kernel{chip: hardware.Chip{Name: chip}}
	k.layout.TP = absent
	_, err = k.groupWidth(c, collKey{Op: model.OpAllReduce, Group: price.GroupTP}, "all_reduce", "fp16", key)
	if err == nil {
		t.Fatalf("%s at tp=%d must refuse: that width is not in %v", chip, absent, measured)
	}
	if !strings.Contains(err.Error(), fmt.Sprint(measured)) {
		t.Errorf("the error should report the measured widths %v so a reader can act on "+
			"it; got: %v", measured, err)
	}
	t.Logf("refusal at tp=%d: %v", absent, err)
}
