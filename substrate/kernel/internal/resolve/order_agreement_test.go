package resolve

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/validate/dataset"
)

// VAL-4 and X15: the registry validator and the cost model's lookup agree on precedence.
//
// The validator (dataset.CheckOrdered) flags a pair of same-name entries exactly when the
// written precedence does not order them. For every pair and every deployment that matches
// both entries, the lookup must refuse when the validator flags the pair, in both load
// orders, and otherwise return the same value in both. If the lookup ever depended on load
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
	var accepted, orderedByModel, refused int
	for i, a := range scopes {
		for _, b := range scopes[i:] {
			sa, sb := set("a", a, 1), set("b", b, 2)
			flagged := len(dataset.CheckOrdered(map[string]*coefficient.Set{"a": sa, "b": sb})) > 0
			for _, d := range deployments {
				if !d.Admits(a) || !d.Admits(b) {
					continue
				}
				v1, e1 := lookup(t, []*coefficient.Set{sa, sb}, d)
				v2, e2 := lookup(t, []*coefficient.Set{sb, sa}, d)
				// The lookup refuses a deployment exactly when the validator refuses the pair,
				// in both load orders; and where it does not refuse, the value is the same in
				// both orders.
				if (e1 != nil) != flagged || (e2 != nil) != flagged {
					t.Fatalf("scopes {%s} and {%s} at %s: the validator flags the pair = %v, but the "+
						"lookup errors were %v and %v", a.Describe(), b.Describe(), d, flagged, e1, e2)
				}
				if flagged {
					refused++
					continue
				}
				accepted++
				if v1 != v2 {
					t.Fatalf("the validator accepts scopes {%s} and {%s} but the lookup returns %v or %v "+
						"for %s by load order", a.Describe(), b.Describe(), v1, v2, d)
				}
				if len(a.Model) > 0 != (len(b.Model) > 0) {
					orderedByModel++
				}
			}
		}
	}
	// The test must exercise both sides: pairs the precedence orders, and pairs it refuses.
	if orderedByModel == 0 || refused == 0 {
		t.Fatalf("not exercised: %d matches ordered by model, %d refused", orderedByModel, refused)
	}
	t.Logf("%d accepted matching (pair, deployment) cases, %d of them ordered by model; "+
		"%d refused by both the validator and the lookup", accepted, orderedByModel, refused)
}

func lookup(t *testing.T, sets []*coefficient.Set, d Scope) (float64, error) {
	t.Helper()
	c, err := Load(sets, d)
	if err != nil {
		return 0, err
	}
	v, err := c.Value("x")
	if err != nil {
		t.Fatalf("Value: %v", err)
	}
	return v, nil
}
