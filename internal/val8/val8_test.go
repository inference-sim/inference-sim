// Package val8 holds the simulator-side cross-check of design VAL-8: every coefficient registry
// entry that states a kernel implementation is selected by a lookup on a run, or its kernel hash
// appears in the kernel_impls.json of an engine release this build pins. The runs are the byte
// gate's goldens (testdata/bytegate/golden), whose Results carry selected_entries; the tables are
// the engine-neutral copies in substrate/registry/inputs, which each engine's own test holds to
// the table it embeds.
//
// Entries that state no kernel implementation have no hash to appear in a table, so only a
// selection accounts for them. The byte gate's few runs select none of most of them, and the
// conformance scenarios that would are not defined yet, so that half of VAL-8 is not asserted.
package val8

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/internal/substratedata"
	"github.com/inference-sim/inference-sim/substrate/contracts"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/kernelimpls"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"github.com/inference-sim/inference-sim/substrate/validate/dataset"
	"github.com/inference-sim/inference-sim/substrate/validate/registry"
)

type inputs struct {
	sets    map[string]*coefficient.Set
	reports []result.SelectedEntries
	tables  []*kernelimpls.Table
}

func load(t *testing.T) inputs {
	t.Helper()
	root, err := substratedata.Registry()
	if err != nil {
		t.Fatal(err)
	}
	paths, err := registry.SetPaths(root)
	if err != nil {
		t.Fatal(err)
	}
	in := inputs{sets: map[string]*coefficient.Set{}}
	for _, p := range paths {
		s, err := contracts.LoadCoefficientSet(p)
		if err != nil {
			t.Fatal(err)
		}
		in.sets[dataset.Stem(p)] = s
	}
	goldens, err := filepath.Glob("../../testdata/bytegate/golden/*.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range goldens {
		r, err := contracts.LoadResult(p)
		if err != nil {
			t.Fatal(err)
		}
		in.reports = append(in.reports, r.SelectedEntries)
	}
	copies, err := filepath.Glob(filepath.Join(root, "inputs", "kernel_impls.*.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range copies {
		tab, err := contracts.LoadKernelImplTable(p)
		if err != nil {
			t.Fatal(err)
		}
		in.tables = append(in.tables, tab)
	}
	if len(in.sets) == 0 || len(in.reports) == 0 || len(in.tables) == 0 {
		t.Fatalf("%d coefficient sets, %d selection reports, %d kernel tables; with an input missing every entry would pass or fail for the wrong reason (X22)",
			len(in.sets), len(in.reports), len(in.tables))
	}
	return in
}

func unpublished(t *testing.T, in inputs, reports []result.SelectedEntries, tables []*kernelimpls.Table) []dataset.Unselected {
	t.Helper()
	found, err := dataset.FindUnselected(in.sets, reports, tables)
	if err != nil {
		t.Fatal(err)
	}
	var out []dataset.Unselected
	for _, u := range found {
		if len(u.Unpublished) > 0 {
			out = append(out, u)
		}
	}
	return out
}

// Every entry that states a kernel implementation is accounted for. One listed here is read by no
// run and keyed to a kernel no engine release this build pins publishes.
func TestEveryKeyedEntryIsSelectedOrPublished(t *testing.T) {
	in := load(t)
	for _, u := range unpublished(t, in, in.reports, in.tables) {
		t.Error(u)
	}
}

// The check can fail: with the tables withheld, entries that state a kernel implementation and
// that no byte-gate run selected are listed, so the test above passes because of the tables.
func TestTheCheckListsKeyedEntriesWhenTheTablesAreWithheld(t *testing.T) {
	in := load(t)
	if got := unpublished(t, in, in.reports, nil); len(got) == 0 {
		t.Error("no keyed entry depends on a published table: the check above cannot fail")
	}
}
