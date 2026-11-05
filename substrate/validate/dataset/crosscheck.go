package dataset

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
)

// CheckFits checks a registry against the catalog it runs with (design §3.3): every chip,
// model and fabric a coefficient's scope names, and every model and chip an acceptance or
// routing profile names, is an entry of that catalog. A candidate registry that does not
// fit the catalog in use is refused, rather than leaving its entries to match nothing.
func CheckFits(c *Catalog, r *Registry) []Problem {
	var out []Problem
	miss := func(file, what, name, have string) {
		out = append(out, Problem{file, fmt.Sprintf("%s %q is not in the catalog in use; it has %s",
			what, name, have)})
	}
	chips, models, fabrics := keys(c.Chips), keys(c.Models), keys(c.Fabrics)
	for _, stem := range sortedKeys(r.Sets) {
		file := "coefficients/" + stem + ".yaml"
		for _, e := range r.Sets[stem].Coefficients {
			for _, h := range e.Scope.Hardware {
				if c.Chips[h] == nil {
					miss(file, "entry "+e.Name+" is scoped to chip", h, chips)
				}
			}
			for _, m := range e.Scope.Model {
				if c.Models[m] == nil {
					miss(file, "entry "+e.Name+" is scoped to model", m, models)
				}
			}
			for _, f := range e.Scope.Fabric {
				if c.Fabrics[f] == nil {
					miss(file, "entry "+e.Name+" is scoped to fabric", f, fabrics)
				}
			}
		}
	}
	for _, stem := range sortedKeys(r.Acceptance) {
		file := "acceptance/" + stem + ".yaml"
		for _, p := range r.Acceptance[stem].Profiles {
			if c.Models[p.Model] == nil {
				miss(file, "profile "+p.Key().String()+" names model", p.Model, models)
			}
			if p.Provenance != nil && p.Provenance.Hardware != "" && c.Chips[p.Provenance.Hardware] == nil {
				miss(file, "profile "+p.Key().String()+" was measured on chip", p.Provenance.Hardware, chips)
			}
		}
	}
	for _, stem := range sortedKeys(r.Routing) {
		file := "routing/" + stem + ".yaml"
		for _, p := range r.Routing[stem].Profiles {
			if c.Models[p.Model] == nil {
				miss(file, "profile "+p.Key().String()+" names model", p.Model, models)
			}
			if p.Provenance != nil && p.Provenance.Hardware != "" && c.Chips[p.Provenance.Hardware] == nil {
				miss(file, "profile "+p.Key().String()+" was measured on chip", p.Provenance.Hardware, chips)
			}
		}
	}
	sortProblems(out)
	return out
}

// CheckOrdered refuses two entries of one name, in any sets, that one deployment could
// match and that the cost model's precedence does not order (VAL-4).
//
// Two entries both match some deployment when, on every scope dimension, either leaves the
// dimension unstated or the two share a value. The written precedence today is the cost
// model's (kernel/internal/resolve.Load): an entry scoped to a model is chosen over one that
// states no model. Any other matching pair would be decided by the order the sets are
// listed in, so the data would mean something its files do not say; such a pair is
// refused. The rank order of design §9.2 joins this precedence with the coefficient-method
// migration (M2).
func CheckOrdered(sets map[string]*coefficient.Set) []Problem {
	type at struct {
		set string
		i   int
		e   coefficient.Entry
	}
	byName := map[string][]at{}
	for _, stem := range sortedKeys(sets) {
		for i, e := range sets[stem].Coefficients {
			byName[e.Name] = append(byName[e.Name], at{stem, i, e})
		}
	}
	var out []Problem
	for _, name := range sortedKeys(byName) {
		es := byName[name]
		for i := range es {
			for j := i + 1; j < len(es); j++ {
				a, b := es[i], es[j]
				if (len(a.e.Scope.Model) > 0) != (len(b.e.Scope.Model) > 0) || !overlap(a.e.Scope, b.e.Scope) {
					continue
				}
				out = append(out, Problem{"coefficients/" + b.set + ".yaml", fmt.Sprintf(
					"entry %s (%s, %s) and coefficients/%s.yaml's entry %s (%s, %s) can match one "+
						"deployment and no scope precedence orders them, so the order the sets are "+
						"listed in would choose (VAL-4); narrow one scope or remove one entry",
					name, b.e.Scope.Describe(), b.e.Method, a.set, name, a.e.Scope.Describe(), a.e.Method)})
			}
		}
	}
	return out
}

// overlap reports whether some deployment lies inside both scopes.
func overlap(a, b coefficient.Scope) bool {
	return shareS(a.Hardware, b.Hardware) && shareS(a.Model, b.Model) && shareI(a.TP, b.TP) &&
		shareI(a.EP, b.EP) && shareI(a.NodesSpanned, b.NodesSpanned) && shareS(a.Fabric, b.Fabric)
}

// shareS reports whether two stated value lists admit a common value: an unstated list
// admits any. The cost model matches names without regard to case, so this does too.
func shareS(a, b []string) bool {
	if len(a) == 0 || len(b) == 0 {
		return true
	}
	for _, x := range a {
		for _, y := range b {
			if strings.EqualFold(x, y) {
				return true
			}
		}
	}
	return false
}

func shareI(a, b []int) bool {
	if len(a) == 0 || len(b) == 0 {
		return true
	}
	for _, x := range a {
		for _, y := range b {
			if x == y {
				return true
			}
		}
	}
	return false
}

func keys[V any](m map[string]V) string {
	ks := sortedKeys(m)
	if len(ks) > 12 {
		return fmt.Sprintf("%d entries, among them %v", len(ks), ks[:12])
	}
	return fmt.Sprint(ks)
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortProblems(ps []Problem) {
	sort.SliceStable(ps, func(i, j int) bool {
		if ps[i].File != ps[j].File {
			return ps[i].File < ps[j].File
		}
		return ps[i].Message < ps[j].Message
	})
}
