package dataset

import (
	"fmt"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/kernelimpls"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// Unselected is a registry entry that meets neither condition of design VAL-8: no lookup on the
// conformance scenarios selected it, and none of the kernel implementations it is valid for is
// a row of the kernel-implementation table of an engine release a supported simulator release
// pins. Nothing the project runs reads such an entry, so it is removed in the next substrate
// release.
type Unselected struct {
	// Set is the coefficient set's name, its file stem.
	Set   string
	Entry coefficient.Entry
	// Unpublished are the entry's kernel implementations, none of which a table lists; empty for
	// an entry that states none.
	Unpublished []coefficient.KernelImpl
}

func (u Unselected) String() string {
	why := "no lookup selected it and it states no kernel implementation"
	if len(u.Unpublished) > 0 {
		why = "no lookup selected it and no engine release's kernel_impls.json lists " +
			coefficient.DescribeImpls(u.Unpublished)
	}
	return fmt.Sprintf("coefficients/%s.yaml: %s [%s] scope {%s}: %s (VAL-8)", u.Set, u.Entry.Name,
		u.Entry.Method, u.Entry.Scope.Key(), why)
}

// FindUnselected lists the entries of sets that meet neither condition of VAL-8, sorted by set
// and then in each set's order. selected are the selected-entry reports of the conformance
// scenarios' runs (a Result's selected_entries); an entry is selected when one of them names it
// by set, name, scope, method and kernel implementations, the identity design §3.3 gives an
// entry. tables are the kernel-implementation tables of the engine releases that some supported
// simulator release pins; an entry's kernel hash appears in one when a table lists one of the
// implementations the entry is valid for, by library, variant and code hash together. An entry
// that states none has no hash to appear, so only a selection accounts for it.
func FindUnselected(sets map[string]*coefficient.Set, selected []result.SelectedEntries, tables []*kernelimpls.Table) []Unselected {
	picked := map[string]bool{}
	for _, s := range selected {
		for _, c := range s.Coefficients {
			e := coefficient.Entry{Name: c.Name, Scope: c.Scope, Method: c.Method, KernelImpls: c.KernelImpls}
			picked[c.Set+"\x00"+e.Identity()] = true
		}
	}
	listed := func(k coefficient.KernelImpl) bool {
		for _, t := range tables {
			if t.Lists(k) {
				return true
			}
		}
		return false
	}
	stems := make([]string, 0, len(sets))
	for stem := range sets {
		stems = append(stems, stem)
	}
	sort.Strings(stems)
	var out []Unselected
	for _, stem := range stems {
		if sets[stem] == nil {
			continue
		}
		for _, e := range sets[stem].Coefficients {
			if picked[stem+"\x00"+e.Identity()] {
				continue
			}
			published := false
			for _, k := range e.KernelImpls {
				published = published || listed(k)
			}
			if published {
				continue
			}
			out = append(out, Unselected{Set: stem, Entry: e, Unpublished: append([]coefficient.KernelImpl(nil), e.KernelImpls...)})
		}
	}
	return out
}
