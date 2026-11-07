package resolve

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/validate/dataset"
)

// VAL-4 and X15: the registry validator and the cost model's lookup agree on precedence.
//
// The validator (dataset.CheckOrdered) accepts a pair of same-name entries only if the
// written precedence orders them or no deployment matches both. For every such accepted
// pair, and every deployment that matches both entries, the lookup must return the same
// value whichever order the two sets are loaded in. If the lookup ever depended on load
// order for a pair the validator accepts, the data would mean something its files do not say.
//
// The scopes enumerated cover each scope dimension both stated and unstated, with two
// values on the hardware dimension, so every pair of scope kinds is present.
func TestLookupAgreesWithTheValidatorOnPrecedenceInBothLoadOrders(t *testing.T) {
	var scopes []coefficient.Scope
	for _, hw := range [][]string{nil, {"a"}, {"b"}} {
		for _, md := range [][]string{nil, {"m"}} {
			for _, tp := range [][]int{nil, {1}} {
				for _, ep := range [][]int{nil, {1}} {
					for _, ns := range [][]int{nil, {1}} {
						for _, fb := range [][]string{nil, {"f"}} {
							scopes = append(scopes, coefficient.Scope{Hardware: hw, Model: md,
								TP: tp, EP: ep, NodesSpanned: ns, Fabric: fb})
						}
					}
				}
			}
		}
	}
	var deployments []Scope
	for _, hw := range []string{"a", "b"} {
		for _, md := range []string{"m", "other"} {
			for _, tp := range []int{1, 2} {
				for _, ep := range []int{1, 2} {
					for _, ns := range []int{1, 2} {
						for _, fb := range []string{"", "f"} {
							deployments = append(deployments, Scope{Hardware: hw, Model: md,
								TP: tp, EP: ep, NodesSpanned: ns, Fabric: fb})
						}
					}
				}
			}
		}
	}
	set := func(name string, s coefficient.Scope, v float64) *coefficient.Set {
		return &coefficient.Set{Kind: "CoefficientSet", Name: name, Coefficients: []coefficient.Entry{
			{Name: "x", Value: v, Units: vocab.UnitDimensionless, Method: vocab.MethodMeasured, Scope: s}}}
	}
	var accepted, orderedBoth, refusedAndOrderDependent int
	for i, a := range scopes {
		for _, b := range scopes[i:] {
			sa, sb := set("a", a, 1), set("b", b, 2)
			flagged := len(dataset.CheckOrdered(map[string]*coefficient.Set{"a": sa, "b": sb})) > 0
			for _, d := range deployments {
				if !d.Admits(a) || !d.Admits(b) {
					continue
				}
				v1 := lookup(t, []*coefficient.Set{sa, sb}, d)
				v2 := lookup(t, []*coefficient.Set{sb, sa}, d)
				switch {
				case !flagged:
					accepted++
					if v1 != v2 {
						t.Fatalf("the validator accepts scopes {%s} and {%s} but the lookup returns %v or %v "+
							"for %s by load order", a.Describe(), b.Describe(), v1, v2, d)
					}
					if len(a.Model) > 0 != (len(b.Model) > 0) {
						orderedBoth++
					}
				case v1 != v2:
					refusedAndOrderDependent++
				}
			}
		}
	}
	// The test must exercise both sides: pairs the precedence orders (model over no model),
	// and pairs the validator refuses because load order would decide.
	if orderedBoth == 0 || refusedAndOrderDependent == 0 {
		t.Fatalf("not exercised: %d ordered matches, %d refused order-dependent matches", orderedBoth, refusedAndOrderDependent)
	}
	t.Logf("%d accepted matching (pair, deployment) cases, %d of them ordered by model; "+
		"%d refused cases where load order changes the value", accepted, orderedBoth, refusedAndOrderDependent)
}

func lookup(t *testing.T, sets []*coefficient.Set, d Scope) float64 {
	t.Helper()
	c, err := Load(sets, d)
	if err != nil {
		t.Fatalf("Load(%s): %v", d, err)
	}
	v, err := c.Value("x")
	if err != nil {
		t.Fatalf("Value: %v", err)
	}
	return v
}
