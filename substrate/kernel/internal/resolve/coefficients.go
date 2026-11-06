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
}

// Resolved is one coefficient with the set that supplied it.
type Resolved struct {
	Entry coefficient.Entry
	Set   string
}

// Load builds a lookup from sets in scenario order. An entry whose scope excludes the
// deployment is skipped rather than applied: a constant fitted on one part does not
// describe another, and silently using it is the error this filtering prevents.
//
// An entry scoped to the deployment's model is never displaced by one that does not state
// a model, wherever the two sit: a value fitted on one model is a narrower claim than the
// part-wide one it sits beside, and letting file or set order decide between them would
// make a model's own constant silently lose to the generic one placed after it. Among
// entries equally specific in that sense, the later one wins as before.
func Load(sets []*coefficient.Set, scope Scope) (*Coefficients, error) {
	c := &Coefficients{byName: map[string]Resolved{}, byNameFabric: map[string]Resolved{},
		selected: map[string]Resolved{}}
	var ambiguous error
	put := func(into map[string]Resolved, e coefficient.Entry, set string) {
		if prev, ok := into[e.Name]; ok {
			if len(prev.Entry.Scope.Model) > 0 && len(e.Scope.Model) == 0 {
				return
			}
			// Two admitted entries of one name and one scope that differ only in method are
			// two coefficients (design §3.3) which only method rank can order, and rank
			// selection is M2. Letting set order pick between them would make the data mean
			// something its files do not say, so the lookup is refused instead.
			if ambiguous == nil && prev.Entry.Scope.Key() == e.Scope.Key() &&
				prev.Entry.Method != e.Method {
				ambiguous = fmt.Errorf("coefficient %q at scope (%s) is supplied with method %q by set %q "+
					"and with method %q by set %q; selecting between methods needs rank selection, "+
					"which is not implemented, so the lookup is refused",
					e.Name, e.Scope.Describe(), prev.Entry.Method, prev.Set, e.Method, set)
			}
		}
		into[e.Name] = Resolved{Entry: e, Set: set}
	}
	for _, s := range sets {
		if s == nil {
			continue
		}
		c.order = append(c.order, s.Name)
		for _, e := range s.Coefficients {
			if !scope.Admits(e.Scope) {
				continue
			}
			put(c.byName, e, s.Name)
			if len(e.Scope.Fabric) > 0 {
				put(c.byNameFabric, e, s.Name)
			}
		}
	}
	if ambiguous != nil {
		return nil, ambiguous
	}
	if len(c.byName) == 0 {
		return nil, fmt.Errorf("no coefficient in %v applies to %s",
			c.order, scope)
	}
	return c, nil
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
}

func (s Scope) String() string {
	return fmt.Sprintf("hardware=%s model=%s tp=%d ep=%d nodes_spanned=%d fabric=%s",
		s.Hardware, s.Model, s.TP, s.EP, s.NodesSpanned, s.Fabric)
}

// Admits reports whether an entry's scope covers this deployment.
//
// An unstated dimension in the entry's scope means the entry does not constrain it, so
// it admits any value. A stated dimension must list the deployment's value. That
// asymmetry is deliberate: a coefficient scoped to hardware alone applies at every
// parallelism width, where one scoped to tp: [8] does not apply at tp 4.
func (s Scope) Admits(e coefficient.Scope) bool {
	if len(e.Hardware) > 0 && !containsString(e.Hardware, s.Hardware) {
		return false
	}
	if len(e.Model) > 0 && !containsString(e.Model, s.Model) {
		return false
	}
	if len(e.TP) > 0 && !containsInt(e.TP, s.TP) {
		return false
	}
	if len(e.EP) > 0 && !containsInt(e.EP, s.EP) {
		return false
	}
	if len(e.NodesSpanned) > 0 && !containsInt(e.NodesSpanned, s.NodesSpanned) {
		return false
	}
	// A fabric-scoped entry was measured over one interconnect, so a deployment on another
	// fabric, or on none, is outside it.
	if len(e.Fabric) > 0 && (s.Fabric == "" || !containsString(e.Fabric, s.Fabric)) {
		return false
	}
	return true
}

// Value returns a coefficient by name. A missing coefficient is an error rather than a
// zero: pricing a term at zero because its constant was absent produces a step time that
// looks plausible and is wrong by whatever that term contributes.
func (c *Coefficients) Value(name string) (float64, error) {
	r, ok := c.byName[name]
	if !ok {
		return 0, &MissingError{Name: name, Available: c.Names()}
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
	c.selected[r.Set+"\x00"+r.Entry.Name+"\x00"+r.Entry.Scope.Key()+"\x00"+string(r.Entry.Method)] = r
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
}

func (e *MissingError) Error() string {
	return fmt.Sprintf("coefficient %q is not in the resolved set; available: %s",
		e.Name, strings.Join(e.Available, ", "))
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
