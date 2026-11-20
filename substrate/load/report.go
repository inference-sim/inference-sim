package load

import (
	"reflect"
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
	// KernelImpls, FitRange, Extrapolated and Unidentified are a coefficient's, as the
	// Result's SelectedCoefficient carries them; zero for a profile.
	KernelImpls []coefficient.KernelImpl
	// Priced are the implementations of KernelImpls that run the operations that read the entry
	// in the pool it priced (the primitive's operations, narrowed to the entry's kind): the
	// lowered implementation, with its code hash. Match says how the entry met them (exact, or
	// unkeyed when it lists none). MeasuredOn are the implementations a transferred value was
	// measured on. A sibling implementation the entry is also valid for is in KernelImpls and
	// not in Priced.
	Priced     []coefficient.KernelImpl
	Match      coefficient.KernelMatch
	MeasuredOn []coefficient.KernelImpl
	// Primitive is the lowered primitive the entry's name prices (coefficient.PrimitiveOf), or
	// "" for a name that prices no lowered kernel.
	Primitive    string
	FitRange     []coefficient.Bound
	Extrapolated bool
	Unidentified bool
	// CI95 is the entry's 95% interval, when it states one.
	CI95 *coefficient.Interval
	// Whole is true when the entry was recorded with RecordEntry, which carries the four
	// facts above, and false when only its identity was (RecordCoefficient).
	Whole bool
}

// Report is the selected-entry report.
type Report struct {
	// Implementations are the kernel implementations the recorded cost models ran, by lowered
	// primitive, each once and sorted: what the priced implementation of each coefficient is
	// one of, for its primitive.
	Implementations map[string][]coefficient.KernelImpl
	// Selected is every entry a lookup returned, sorted by kind, set, entry, method and
	// scope: the order the Result's validator requires (the file order differs when one set's
	// name is a prefix of another's).
	Selected []Selected
	// Unloaded is every registry data file no reader loaded, sorted by path.
	Unloaded []string
}

// RecordCoefficient records a coefficient a lookup returned, from the named set, when the
// caller holds only its identity. The report then carries none of the entry's kernel
// implementations or fit range; RecordEntry records them.
func (d *Data) RecordCoefficient(set, name string, scope coefficient.Scope, method string) {
	d.record(Selected{Kind: KindCoefficient, File: d.setFileOf(set), Set: set, Entry: name,
		Scope: scope, Method: method})
}

// RecordEntry records a coefficient entry a lookup returned, from the named set, with the
// facts the Result reports of it (design §11.3): the kernel implementations and fit range the
// entry states, its interval, whether its data leave it unidentified
// (coefficient.Entry.Unidentified), and extrapolated, which the caller knows from the steps it
// priced. Records of one entry accumulate: several cost models may record into one Data (an
// engine instance, or a pool, each has its own), and a flag any record raised stays raised.
// One Data serves one run: nothing is cleared, so a Data reused for a second run carries the
// first run's flags into its report.
//
// primitive is the lowered primitive the entry prices and priced the implementations of the
// operations that read it, which a keyed entry matched whole (resolve.Resolved.Primitive and
// Running);
// match says how it met them (design VAL-8): a keyed entry is exact and names the build that
// ran, an unkeyed one applies to every build and names none. Two entries that differ only in the implementations they are valid for are two
// records.
func (d *Data) RecordEntry(set string, e coefficient.Entry, primitive string, priced []coefficient.KernelImpl, match coefficient.KernelMatch, extrapolated bool) {
	d.record(Selected{Kind: KindCoefficient, File: d.setFileOf(set), Set: set, Entry: e.Name,
		Scope: e.Scope, Method: string(e.Method),
		Primitive: primitive, Priced: append([]coefficient.KernelImpl(nil), priced...), Match: match,
		MeasuredOn:   append([]coefficient.KernelImpl(nil), e.MeasuredOn...),
		KernelImpls:  append([]coefficient.KernelImpl(nil), e.KernelImpls...),
		FitRange:     append([]coefficient.Bound(nil), e.FitRange...),
		CI95:         copyInterval(e.CI95),
		Extrapolated: extrapolated, Unidentified: e.Unidentified(), Whole: true})
}

func copyInterval(iv *coefficient.Interval) *coefficient.Interval {
	if iv == nil {
		return nil
	}
	c := *iv
	return &c
}

// ExtrapolationObserver says, when a report is read, whether a step priced so far read a
// coefficient outside its fit range. The cost model is one: it learns which entries a step
// left only while the run prices steps, after OpenPool has recorded the entries.
type ExtrapolationObserver interface {
	Extrapolated(set, name string, scope coefficient.Scope, method string, impls []coefficient.KernelImpl) bool
}

// Observe asks o, at every later Report, whether each recorded coefficient was extrapolated.
// An observer already registered is not added again (one that is not a comparable value is
// always added). Report reads the answer then, so the Result describes the steps priced up to
// the moment it is written and no second record is needed at the end of a run. A nil observer
// is ignored. The Data keeps every observer for its life.
func (d *Data) Observe(o ExtrapolationObserver) {
	if o == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if reflect.TypeOf(o).Comparable() {
		for _, x := range d.observers {
			if reflect.TypeOf(x).Comparable() && x == o {
				return
			}
		}
	}
	d.observers = append(d.observers, o)
}

// RecordImplementations records the kernel implementations a cost model ran, by the lowered
// primitive each executes (the pool's plan.PoolPlan.ImplementationsByPrimitive, design VAL-8), so
// the report states them once. Several cost models accumulate; an implementation is kept once.
func (d *Data) RecordImplementations(impls map[string][]coefficient.KernelImpl) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.running == nil {
		d.running = map[string]map[coefficient.KernelImpl]bool{}
	}
	for prim, l := range impls {
		if d.running[prim] == nil {
			d.running[prim] = map[coefficient.KernelImpl]bool{}
		}
		for _, k := range l {
			d.running[prim][k] = true
		}
	}
}

// unionImpls returns the implementations of a and b, each once, sorted.
func unionImpls(a, b []coefficient.KernelImpl) []coefficient.KernelImpl {
	seen := map[coefficient.KernelImpl]bool{}
	var out []coefficient.KernelImpl
	for _, l := range [][]coefficient.KernelImpl{a, b} {
		for _, k := range l {
			if !seen[k] {
				seen[k] = true
				out = append(out, k)
			}
		}
	}
	sortImpls(out)
	return out
}

func sortImpls(l []coefficient.KernelImpl) {
	sort.Slice(l, func(i, j int) bool {
		a, b := l[i], l[j]
		if a.Library != b.Library {
			return a.Library < b.Library
		}
		if a.Variant != b.Variant {
			return a.Variant < b.Variant
		}
		return a.CodeHash < b.CodeHash
	})
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
	key := string(s.Kind) + "\x00" + s.Set + "\x00" + s.Entry + "\x00" + s.Method + "\x00" + s.Scope.Key() +
		"\x00" + coefficient.DescribeImpls(s.KernelImpls)
	if old, ok := d.selected[key]; ok && s.Kind == KindCoefficient {
		s = mergeCoefficient(old, s)
	}
	d.selected[key] = s
}

// mergeCoefficient combines two records of one coefficient, so that several cost models
// recording into one Data (one per engine instance, or per pool) never lose a flag: a flag that
// either record raised stays raised. A coefficient is identified by its set, name, method and
// scope, and the registry admits one entry for that key, so every record of it carries the same
// implementations, fit range and interval. Records that disagree cannot come from one registry;
// the merge then states none of the three and withholds the report's marker, rather than guess
// which was right. The entry is whole only if both records were.
func mergeCoefficient(a, b Selected) Selected {
	out := b
	out.Extrapolated = a.Extrapolated || b.Extrapolated
	out.Unidentified = a.Unidentified || b.Unidentified
	out.Whole = a.Whole && b.Whole
	out.Priced = unionImpls(a.Priced, b.Priced)
	agree := reflect.DeepEqual(a.KernelImpls, b.KernelImpls) && a.Match == b.Match &&
		reflect.DeepEqual(a.MeasuredOn, b.MeasuredOn) && reflect.DeepEqual(a.FitRange, b.FitRange) &&
		reflect.DeepEqual(a.CI95, b.CI95)
	if a.Whole && b.Whole && !agree {
		out.KernelImpls, out.Priced, out.MeasuredOn, out.Match = nil, nil, nil, ""
		out.FitRange, out.CI95, out.Whole = nil, nil, false
	}
	return out
}

// Report returns the selected-entry report as it stands.
func (d *Data) Report() Report {
	d.mu.Lock()
	defer d.mu.Unlock()
	keys := sortedKeys(d.selected)
	r := Report{Selected: make([]Selected, 0, len(keys)), Unloaded: []string{}}
	if len(d.running) > 0 {
		r.Implementations = map[string][]coefficient.KernelImpl{}
	}
	for prim, set := range d.running {
		for k := range set {
			r.Implementations[prim] = append(r.Implementations[prim], k)
		}
		sortImpls(r.Implementations[prim])
	}
	for _, k := range keys {
		s := d.selected[k]
		if s.Kind == KindCoefficient && !s.Extrapolated {
			for _, o := range d.observers {
				if o.Extrapolated(s.Set, s.Entry, s.Scope, s.Method, s.KernelImpls) {
					s.Extrapolated = true
					break
				}
			}
		}
		r.Selected = append(r.Selected, s)
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
	// The report carries the closeness data exactly when it holds a coefficient and every one
	// was recorded whole. One recorded by identity alone has none, and a reader could not tell
	// its absent data from data the entry did not state; the report then carries no closeness
	// data for any coefficient, so that it validates. A report with no coefficient says
	// nothing about the run's pricing and is not marked.
	whole, n := true, 0
	for _, s := range r.Selected {
		if s.Kind == KindCoefficient {
			n++
			whole = whole && s.Whole
		}
	}
	if whole && n > 0 {
		out.ClosenessVersion = result.ClosenessVersion
		if len(r.Implementations) > 0 {
			out.Implementations = map[string][]coefficient.KernelImpl{}
			for prim, l := range r.Implementations {
				out.Implementations[prim] = append([]coefficient.KernelImpl(nil), l...)
			}
		}
	}
	for _, s := range r.Selected {
		switch s.Kind {
		case KindCoefficient:
			c := result.SelectedCoefficient{
				Set: s.Set, Name: s.Entry, Method: vocab.Method(s.Method), Scope: s.Scope, File: s.File}
			if out.ClosenessVersion != 0 {
				c.KernelImpls, c.FitRange, c.CI95 = s.KernelImpls, s.FitRange, s.CI95
				c.Primitive, c.Priced, c.KernelMatch, c.MeasuredOn = s.Primitive, s.Priced, s.Match, s.MeasuredOn
				c.Extrapolated, c.Unidentified = s.Extrapolated, s.Unidentified
			}
			out.Coefficients = append(out.Coefficients, c)
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
