package load

import (
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// The selected-entry report (design §3.3, VAL-6) says which registry entries lookups
// returned during a run, and which registry files no reader loaded. Without it a candidate
// that adds a family no released code reads would be ignored and look like a no-op.
//
// Every reader behind the Environment records into the Data it read from: the cost model
// records each coefficient its lookups returned (kernel.Record, called by kernel.Open), and
// the caller that resolves speculative acceptance and routing imbalance records each
// resolution (RecordAcceptance, RecordRouting). The simulator writes Report() into the
// Result's selected_entries, converting it to the Result's own type in contracts/spec/result.

// Kind is the kind of registry entry a lookup selected.
type Kind string

const (
	KindCoefficient Kind = "coefficient"
	KindAcceptance  Kind = "acceptance"
	KindRouting     Kind = "routing"
)

// Selected is one registry entry a lookup returned.
type Selected struct {
	Kind Kind
	// File is the registry file the entry is in, relative to the registry directory.
	File string
	// Set is the set's name.
	Set string
	// Entry identifies the entry within its set: a coefficient's name, or a profile's key
	// as its Key().String() renders it.
	Entry string
	// Scope is a coefficient's scope; the zero value for a profile.
	Scope coefficient.Scope
	// Method is the entry's method.
	Method string
}

// Report is the selected-entry report.
type Report struct {
	// Selected is every entry a lookup returned, sorted by kind, set, entry, method and
	// scope: the order the Result's validator requires (the file order differs when one set's
	// name is a prefix of another's).
	Selected []Selected
	// Unloaded is every registry data file no reader loaded, sorted by path.
	Unloaded []string
}

// RecordCoefficient records a coefficient a lookup returned, from the named set.
func (d *Data) RecordCoefficient(set, name string, scope coefficient.Scope, method string) {
	d.record(Selected{Kind: KindCoefficient, File: d.setFileOf(set), Set: set, Entry: name,
		Scope: scope, Method: method})
}

// RecordAcceptance records the profile an acceptance resolution used, if it used one.
func (d *Data) RecordAcceptance(r *acceptance.Resolution) {
	if r == nil || r.Profile == nil || r.Set == "" {
		return
	}
	d.record(Selected{Kind: KindAcceptance, File: "acceptance/" + r.Set + ".yaml", Set: r.Set,
		Entry: r.Profile.Key().String(), Method: string(r.Profile.Method)})
}

// RecordRouting records the profiles a routing resolution used, if it used any.
func (d *Data) RecordRouting(r *routing.Resolution) {
	if r == nil || r.Set == "" {
		return
	}
	for _, p := range []*routing.Profile{r.Decode, r.Prefill} {
		if p != nil {
			d.record(Selected{Kind: KindRouting, File: "routing/" + r.Set + ".yaml", Set: r.Set,
				Entry: p.Key().String(), Method: p.Method})
		}
	}
}

func (d *Data) setFileOf(set string) string {
	if f, ok := d.setFile[set]; ok {
		return f
	}
	return "coefficients/" + set + ".yaml"
}

func (d *Data) record(s Selected) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.loaded[s.File] = true
	d.selected[string(s.Kind)+"\x00"+s.Set+"\x00"+s.Entry+"\x00"+s.Method+"\x00"+s.Scope.Key()] = s
}

// Report returns the selected-entry report as it stands.
func (d *Data) Report() Report {
	d.mu.Lock()
	defer d.mu.Unlock()
	keys := sortedKeys(d.selected)
	r := Report{Selected: make([]Selected, 0, len(keys)), Unloaded: []string{}}
	for _, k := range keys {
		r.Selected = append(r.Selected, d.selected[k])
	}
	for _, f := range d.registryFileList() {
		if !d.loaded[f] {
			r.Unloaded = append(r.Unloaded, f)
		}
	}
	return r
}

// registryFileList is every file of the combined registry.
func (d *Data) registryFileList() []string {
	var out []string
	for stem := range d.registry.Sets {
		out = append(out, "coefficients/"+stem+".yaml")
	}
	for stem := range d.registry.Acceptance {
		out = append(out, "acceptance/"+stem+".yaml")
	}
	for stem := range d.registry.Routing {
		out = append(out, "routing/"+stem+".yaml")
	}
	sort.Strings(out)
	return out
}

// Result converts the report to the Result's selected_entries (contracts/spec/result). It is
// the one place the conversion is written, so every consumer records the same thing.
func (r Report) Result() result.SelectedEntries {
	var out result.SelectedEntries
	for _, s := range r.Selected {
		switch s.Kind {
		case KindCoefficient:
			out.Coefficients = append(out.Coefficients, result.SelectedCoefficient{
				Set: s.Set, Name: s.Entry, Method: vocab.Method(s.Method), Scope: s.Scope, File: s.File})
		case KindAcceptance:
			out.AcceptanceProfiles = append(out.AcceptanceProfiles, result.SelectedProfile{Set: s.Set, Key: s.Entry, File: s.File})
		case KindRouting:
			out.RoutingProfiles = append(out.RoutingProfiles, result.SelectedProfile{Set: s.Set, Key: s.Entry, File: s.File})
		}
	}
	out.UnreadFiles = append([]string(nil), r.Unloaded...)
	return out
}

// Result converts one provenance record to the Result's form.
func (p Provenance) Result() result.DataSource {
	out := result.DataSource{Source: result.Source(p.Source), Hash: p.Hash, CombinedHash: p.CombinedHash}
	for _, o := range p.Overlays {
		out.Overlays = append(out.Overlays, result.Overlay{Hash: o.Hash})
	}
	return out
}
