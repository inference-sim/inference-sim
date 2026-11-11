// Package resolve turns a scenario, the catalog facts it names and the registry sets it
// lists into one frozen configuration a kernel prices against.
//
// Resolution happens once, in the constructor, so that every method afterwards is a pure
// function of its arguments. Two things are resolved rather than read: a coefficient's
// applicable value, which depends on scope, and a requested engine setting, which the
// layout may override.
package resolve

import (
	"fmt"
	"maps"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// Coefficients is a resolved lookup over the sets a scenario named.
//
// Sets are applied in the order the scenario lists them, and a later set's entry
// overrides an earlier one's. That ordering is the scenario's to choose: a deployment
// that wants a measured primitive set plus an assumed host set lists both, and one that
// wants to override a single value lists a narrow set last.
type Coefficients struct {
	// byName holds the winning entry per name, with the set it came from.
	byName map[string]Resolved
	// byNameFabric holds the winning FABRIC-scoped entry per name, by the same precedence
	// among fabric-scoped entries alone (EntryScopedToFabric).
	byNameFabric map[string]Resolved
	// order records the set names in the order they were applied, for provenance.
	order []string
	// selected holds every entry a lookup returned, keyed by set, name and scope, for the
	// selected-entry report (design §3.3, VAL-6). A lookup is Value, ValueOr, Entry or
	// EntryScopedToFabric finding the entry; Has and Names do not select.
	selected map[string]Resolved
	// refused holds the entries the deployment's scope admitted and the running kernel
	// implementations did not (VAL-8), for the lookup's error and the registry's report.
	refused []ImplRefusal
	// running is the kernel implementations the lookup was keyed by, by lowered primitive.
	running map[string][]coefficient.KernelImpl
	// operations are the lowered operations the lookup was keyed by, when the caller gave them.
	operations []Operation
}

// ImplRefusal is an entry that covers the deployment and is not valid for the kernel
// implementations that run: it lists implementations and none of them is running. Running holds
// the implementations the plan supplies for the primitive the entry prices, which are the ones a
// reader expects it to have matched.
type ImplRefusal struct {
	Set     string
	Entry   coefficient.Entry
	Running []coefficient.KernelImpl
}

// Resolved is one coefficient with the set that supplied it, and the other entries that
// matched the same lookup and lost to it (VAL-4): a price's provenance names them.
type Resolved struct {
	Entry     coefficient.Entry
	Set       string
	Displaced []coefficient.Candidate
	// Match says how the entry met the running kernel implementations (VAL-8): exact, or
	// unkeyed when it states none and so applies to every implementation without having been
	// shown to match the one that runs. Never refused: a lookup does not return such an entry.
	Match coefficient.KernelMatch
	// Primitive is the lowered primitive the entry's name prices (coefficient.PrimitiveOf), and
	// Running the implementations of the operations that consume it, which the entry was
	// matched against: the primitive's operations narrowed to the entry's class where the name
	// states one (coefficient.ClassOf). Running is empty when the caller supplied no operation
	// that consumes it.
	Primitive string
	Running   []coefficient.KernelImpl
}

// Load builds a lookup from sets in scenario order. An entry whose scope excludes the
// deployment is skipped rather than applied: a constant fitted on one part does not
// describe another, and silently using it is the error this filtering prevents.
//
// An entry that lists kernel implementations is admitted only when every implementation that
// scope.Implementations supplies for the primitive the entry prices is one of them, by library,
// variant and code hash together (VAL-8): a different hash of the same kernel, or a primitive that
// also runs a kernel the entry does not list, is refused, never matched as a sibling. An entry
// that lists none applies to every implementation and is returned as unkeyed. For a primitive
// with no implementation supplied, only unkeyed entries are admitted.
//
// Among the admitted entries of one name, the lookup uses the one design VAL-4 chooses: the
// smaller method rank (design §9.2) first, then the scope precedence over every dimension.
// The order of the sets decides nothing. When no entry is chosen over all the others the
// lookup is refused, naming the entries the precedence does not order; the registry
// validator refuses the same pairs at load (dataset.CheckOrdered), from the same code.
func Load(sets []*coefficient.Set, scope Scope) (*Coefficients, error) {
	if len(scope.Operations) == 0 {
		return load(sets, scope, func(string) bool { return false }, true)
	}
	// Which operations read a part-wide term depends on which classes the registry carries a
	// pair for, and whether it carries one depends on the operations that read that class's
	// entry, which depends on the classes before it in their chain. The dependency runs one way
	// along each chain, so repeating the lookup with what the last one selected settles in at
	// most the length of the longest chain.
	have := map[string]bool{}
	for i := 0; i < 8; i++ {
		c, _ := load(sets, scope, func(a string) bool { return have[a] }, false)
		next := map[string]bool{}
		if c != nil {
			for name := range c.byName {
				next[nameKey(name)] = true
			}
			carried := func(n string) bool { _, ok := c.byName[n]; return ok }
			exact := func(n string) bool { return c.byName[n].Match == coefficient.KernelExact }
			for _, op := range scope.Operations {
				for _, class := range op.Chain {
					if coefficient.ClassUsable(op.Primitive, class, carried, exact) {
						next[classKey(op.Primitive, class)] = true
					}
				}
			}
		}
		if maps.Equal(next, have) {
			break
		}
		have = next
	}
	return load(sets, scope, func(a string) bool { return have[a] }, true)
}

// load is Load for one assumption about which class anchors the registry carries. A lookup that
// is not strict skips a name it cannot choose, and the absence of any coefficient, instead of
// failing: it only finds out what is selected.
func load(sets []*coefficient.Set, scope Scope, have present, strict bool) (*Coefficients, error) {
	c := &Coefficients{byName: map[string]Resolved{}, byNameFabric: map[string]Resolved{},
		selected: map[string]Resolved{}}
	all := map[string][]coefficient.Candidate{}
	fabric := map[string][]coefficient.Candidate{}
	var names []string
	for _, s := range sets {
		if s == nil {
			continue
		}
		c.order = append(c.order, s.Name)
		for _, e := range s.Coefficients {
			if !scope.Admits(e.Scope) {
				continue
			}
			if e.MatchImpls(scope.consuming(e.Name, have)) == coefficient.KernelRefused {
				c.refused = append(c.refused, ImplRefusal{Set: s.Name, Entry: e,
					Running: scope.consuming(e.Name, have)})
				continue
			}
			if _, seen := all[e.Name]; !seen {
				names = append(names, e.Name)
			}
			cand := coefficient.Candidate{Set: s.Name, Entry: e}
			all[e.Name] = append(all[e.Name], cand)
			if len(e.Scope.Fabric) > 0 {
				fabric[e.Name] = append(fabric[e.Name], cand)
			}
		}
	}
	sort.Strings(names)
	choose := func(into map[string]Resolved, from map[string][]coefficient.Candidate) error {
		for _, name := range names {
			cands := from[name]
			if len(cands) == 0 {
				continue
			}
			w, displaced, err := coefficient.Select(cands)
			if err != nil {
				if !strict {
					continue
				}
				return err
			}
			into[name] = Resolved{Entry: w.Entry, Set: w.Set, Displaced: displaced,
				Match: w.Entry.MatchImpls(scope.consuming(w.Entry.Name, have)), Primitive: coefficient.PrimitiveOf(w.Entry.Name),
				Running: scope.consuming(w.Entry.Name, have)}
		}
		return nil
	}
	if err := choose(c.byName, all); err != nil {
		return nil, err
	}
	// Fabric-scoped entries are chosen among themselves (EntryScopedToFabric).
	if err := choose(c.byNameFabric, fabric); err != nil {
		return nil, err
	}
	c.running = scope.PooledImplementations()
	c.operations = append([]Operation(nil), scope.Operations...)
	if len(c.byName) == 0 && strict {
		return nil, fmt.Errorf("no coefficient in %v applies to %s%s", c.order, scope, c.refusedNote(""))
	}
	return c, nil
}

// Refused returns the entries the scope admitted and the running implementations refused, in
// the order the sets list them. A reader that asks why a measured entry did not price a term
// finds it here.
func (c *Coefficients) Refused() []ImplRefusal { return append([]ImplRefusal(nil), c.refused...) }

// Operations returns the lowered operations the lookup was keyed by, nil when the caller gave
// only the kernels by primitive.
func (c *Coefficients) Operations() []Operation { return append([]Operation(nil), c.operations...) }

// Implementations returns the kernel implementations the lookup was keyed by, by primitive.
func (c *Coefficients) Implementations() map[string][]coefficient.KernelImpl {
	out := map[string][]coefficient.KernelImpl{}
	for p, impls := range c.running {
		out[p] = append([]coefficient.KernelImpl(nil), impls...)
	}
	return out
}

// refusedNote renders the refused entries of a name ("" for all) for an error: which entries
// would have priced the term and which implementation each was keyed to.
func (c *Coefficients) refusedNote(name string) string {
	var lines []string
	for _, r := range c.refused {
		if name != "" && r.Entry.Name != name {
			continue
		}
		prim := coefficient.PrimitiveOf(r.Entry.Name)
		ran := "no implementation was stated for primitive " + prim
		switch {
		case prim == "":
			ran = "its name prices no lowered primitive"
		case len(r.Running) > 0:
			ran = "primitive " + prim + " runs " + coefficient.DescribeImpls(r.Running)
		}
		lines = append(lines, fmt.Sprintf("%s: %s [%s] is valid for %s (%s)", r.Set, r.Entry.Name,
			r.Entry.Method, coefficient.DescribeImpls(r.Entry.KernelImpls), ran))
	}
	if len(lines) == 0 {
		return ""
	}
	return "; refused for the kernel implementations that run (VAL-8):\n  " + strings.Join(lines, "\n  ")
}

// Scope is the deployment dimensions a coefficient's scope is matched against.
type Scope struct {
	Hardware     string
	Model        string
	TP           int
	EP           int
	NodesSpanned int
	// Fabric is the cluster's inter-node network, its catalog identity, or "" for a
	// single-node cluster. An entry scoped to a fabric applies only over that fabric.
	Fabric string
	// Implementations are the kernel implementations that run in the priced pool, by the lowered
	// primitive each executes, every operation of the primitive pooled (VAL-8). It is what a
	// caller gives that cannot say which operation runs which kernel. Operations, when given,
	// replaces it.
	Implementations map[string][]coefficient.KernelImpl
	// Operations are the pool's lowered operations, each with the implementation that executes
	// it and the classes it belongs to (VAL-8, 6.2). An entry that lists implementations is
	// admitted only when every implementation among the operations that consume it is one it
	// lists: the operations of the primitive its name prices, narrowed to its class where the
	// name states one (coefficient.ClassOf). An operation that does not consume the name does
	// not count. With no operation that consumes it, only entries that list none are admitted.
	Operations []Operation
}

// Operation is one lowered operation as the cost model sees it: the primitive, the kernel that
// executes it, and which of the primitive's classes of coefficient it reads (VAL-8, 6.2).
//
// Chain lists, in the order the kernel prefers them, the classes whose terms the operation is
// priced from when the registry carries them (an attention layer of sliding-window kind: swa; a
// BF16 GEMM: bf16; an MXFP8 GEMM on a part that serves it weight-only: w8a16_mxfp8, then bf16).
// The operation reads the part-wide terms of its primitive only for what no class of its chain
// supplies. An operation with an empty chain reads the part-wide terms alone. Wild marks an
// operation whose class could not be told (no format stated for a GEMM): it is taken to read
// every class of its primitive, which is the conservative reading.
type Operation struct {
	Primitive string
	Impl      coefficient.KernelImpl
	Chain     []string
	Wild      bool
	// Draft is true for an operation of the draft model, false for the target's.
	Draft bool
	// Scorer is true for the block-index scorer node of a layer, false for its primary.
	Scorer bool
	// NonLatentPrefill marks the prefill of a non-latent layer's primary attention, the operations
	// the trtllm-gen lane may price, and CacheDtype the element type of the cache it reads, as the
	// plan states it ("" when it does not).
	NonLatentPrefill bool
	CacheDtype       string
}

// present reports whether a class has a usable family (classKey), or a coefficient is carried
// (nameKey), for the fallback rule.
type present func(key string) bool

func classKey(primitive, class string) string { return "class:" + primitive + "|" + class }
func nameKey(name string) string              { return "name:" + name }

// reads reports whether the operation reads a coefficient of this name, given which class
// anchors the registry carries. A name with a class is read by the operations that list the
// class and for which no earlier class of the chain is carried. A part-wide name is read by
// the operations for which no class of the chain is carried; the part-wide decode floor is also
// read where the carried class has a rate and no floor of its own.
func (op Operation) reads(name, class string, have present) bool {
	if op.Primitive != coefficient.PrimitiveOf(name) {
		return false
	}
	if class != "" {
		if op.Wild {
			return true
		}
		for _, c := range op.Chain {
			if c == class {
				return true
			}
			if have(classKey(op.Primitive, c)) {
				return false
			}
		}
		return false
	}
	if op.Wild {
		return true
	}
	for _, c := range op.Chain {
		if have(classKey(op.Primitive, c)) {
			return coefficient.FloorInheritsFromClass(name) && !have(nameKey(name+"_"+c))
		}
	}
	return true
}

// consuming returns the distinct implementations of the operations that read the named
// coefficient, sorted. With no Operations it is the pooled list of the primitive the name
// prices.
func (s Scope) consuming(name string, have present) []coefficient.KernelImpl {
	prim := coefficient.PrimitiveOf(name)
	if len(s.Operations) == 0 {
		return s.Implementations[prim]
	}
	if prim == "" {
		return nil
	}
	class := coefficient.ClassOf(name)
	seen := map[coefficient.KernelImpl]bool{}
	var out []coefficient.KernelImpl
	for _, op := range s.Operations {
		if op.reads(name, class, have) && !seen[op.Impl] {
			seen[op.Impl] = true
			out = append(out, op.Impl)
		}
	}
	sortImpls(out)
	return out
}

// PooledImplementations returns the implementations by primitive, every operation of a
// primitive pooled: Implementations as given, or the projection of Operations.
func (s Scope) PooledImplementations() map[string][]coefficient.KernelImpl {
	out := map[string][]coefficient.KernelImpl{}
	if len(s.Operations) == 0 {
		for p, impls := range s.Implementations {
			out[p] = append([]coefficient.KernelImpl(nil), impls...)
		}
		return out
	}
	seen := map[string]map[coefficient.KernelImpl]bool{}
	for _, op := range s.Operations {
		if seen[op.Primitive] == nil {
			seen[op.Primitive] = map[coefficient.KernelImpl]bool{}
		}
		if !seen[op.Primitive][op.Impl] {
			seen[op.Primitive][op.Impl] = true
			out[op.Primitive] = append(out[op.Primitive], op.Impl)
		}
	}
	for _, impls := range out {
		sortImpls(impls)
	}
	return out
}

func sortImpls(impls []coefficient.KernelImpl) {
	sort.Slice(impls, func(i, j int) bool {
		a, b := impls[i], impls[j]
		if a.Library != b.Library {
			return a.Library < b.Library
		}
		if a.Variant != b.Variant {
			return a.Variant < b.Variant
		}
		return a.CodeHash < b.CodeHash
	})
}

func (s Scope) String() string {
	return fmt.Sprintf("hardware=%s model=%s tp=%d ep=%d nodes_spanned=%d fabric=%s implementations=[%s]",
		s.Hardware, s.Model, s.TP, s.EP, s.NodesSpanned, s.Fabric, describeRunning(s.PooledImplementations()))
}

// Admits reports whether an entry's scope covers this deployment (coefficient.Scope.Admits).
func (s Scope) Admits(e coefficient.Scope) bool {
	return e.Admits(coefficient.Point{Hardware: s.Hardware, Model: s.Model, TP: s.TP, EP: s.EP,
		NodesSpanned: s.NodesSpanned, Fabric: s.Fabric})
}

// Value returns a coefficient by name. A missing coefficient is an error rather than a
// zero: pricing a term at zero because its constant was absent produces a step time that
// looks plausible and is wrong by whatever that term contributes.
func (c *Coefficients) Value(name string) (float64, error) {
	r, ok := c.byName[name]
	if !ok {
		return 0, &MissingError{Name: name, Available: c.Names(), Refused: c.refusedNote(name)}
	}
	c.select_(r)
	return r.Entry.Value, nil
}

// ValueOr returns a coefficient, or a fallback when it is absent. The fallback is NOT
// recorded anywhere: Provenance lists only resolved entries, so a caller whose fallback
// matters to a prediction's footing must say so itself.
func (c *Coefficients) ValueOr(name string, fallback float64) float64 {
	if r, ok := c.byName[name]; ok {
		c.select_(r)
		return r.Entry.Value
	}
	return fallback
}

// Has reports whether a coefficient was resolved. It does not count as a lookup for the
// selected-entry report: a caller that uses the value reads it with Value or ValueOr.
func (c *Coefficients) Has(name string) bool {
	_, ok := c.byName[name]
	return ok
}

// Entry returns the full resolved entry, for a caller that needs its evidence.
func (c *Coefficients) Entry(name string) (Resolved, bool) {
	r, ok := c.byName[name]
	if ok {
		c.select_(r)
	}
	return r, ok
}

// EntryScopedToFabric returns the winning entry of a name among the admitted entries that
// state a fabric scope, by Load's precedence applied to those alone. A term that only a
// fabric's own measurement may set reads this rather than Entry: an entry of the same name
// scoped by hardware or model alone, wherever it sits in the scenario's sets, would win
// Entry and hide the fabric's.
func (c *Coefficients) EntryScopedToFabric(name string) (Resolved, bool) {
	r, ok := c.byNameFabric[name]
	if ok {
		c.select_(r)
	}
	return r, ok
}

// Admitted returns the winning entry of a name without recording a lookup. It is for
// reporting what the sets resolved to (Provenance); pricing code uses Entry, so that the
// selected-entry report lists only what pricing read.
func (c *Coefficients) Admitted(name string) (Resolved, bool) {
	r, ok := c.byName[name]
	return r, ok
}

// Selected returns every entry a lookup has returned, sorted by set, name and scope.
func (c *Coefficients) Selected() []Resolved {
	keys := make([]string, 0, len(c.selected))
	for k := range c.selected {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]Resolved, 0, len(keys))
	for _, k := range keys {
		out = append(out, c.selected[k])
	}
	return out
}

// select_ records r as returned by a lookup.
func (c *Coefficients) select_(r Resolved) {
	c.selected[r.Set+"\x00"+r.Entry.Identity()] = r
}

// Names returns every resolved coefficient name, sorted.
func (c *Coefficients) Names() []string {
	out := make([]string, 0, len(c.byName))
	for k := range c.byName {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Sets returns the set names in application order.
func (c *Coefficients) Sets() []string { return append([]string{}, c.order...) }

// Evidence summarizes how much of the resolved set rests on measurement. A caller
// reporting a prediction uses it to state the prediction's footing rather than leaving a
// reader to inspect every entry.
func (c *Coefficients) Evidence() Evidence {
	var ev Evidence
	for _, r := range c.byName {
		ev.Total++
		if r.Entry.Method.Evidenced() {
			ev.Evidenced++
		}
		if r.Entry.Method == vocab.MethodAssumed {
			ev.Assumed = append(ev.Assumed, r.Entry.Name)
		}
	}
	sort.Strings(ev.Assumed)
	return ev
}

// Evidence counts a resolved set's footing.
type Evidence struct {
	Total     int
	Evidenced int
	// Assumed names the entries that rest on judgement rather than observation, so a
	// prediction can say which of its terms are estimates.
	Assumed []string
}

// MissingError reports a coefficient a term needed and the set did not supply.
type MissingError struct {
	Name      string
	Available []string
	// Refused describes the entries of this name that the running kernel implementations
	// refused, or is empty.
	Refused string
}

func (e *MissingError) Error() string {
	return fmt.Sprintf("coefficient %q is not in the resolved set; available: %s%s",
		e.Name, strings.Join(e.Available, ", "), e.Refused)
}

func containsString(list []string, want string) bool {
	for _, s := range list {
		if strings.EqualFold(s, want) {
			return true
		}
	}
	return false
}

func containsInt(list []int, want int) bool {
	for _, v := range list {
		if v == want {
			return true
		}
	}
	return false
}

// describeRunning renders implementations by primitive, in a fixed order.
func describeRunning(m map[string][]coefficient.KernelImpl) string {
	prims := make([]string, 0, len(m))
	for p := range m {
		prims = append(prims, p)
	}
	sort.Strings(prims)
	parts := make([]string, len(prims))
	for i, p := range prims {
		parts[i] = p + ": " + coefficient.DescribeImpls(m[p])
	}
	return strings.Join(parts, "; ")
}
