package resolve

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// A fabric-scoped entry applies over its own fabric only: not over another, and not to a
// single-node deployment that crosses none. An entry that states no fabric applies over
// any, as every other unstated dimension does.
func TestAFabricScopedEntryAppliesOverItsFabricOnly(t *testing.T) {
	roce := coefficient.Scope{Fabric: []string{"roce-200g"}}
	for _, c := range []struct {
		fabric string
		want   bool
	}{
		{"roce-200g", true},
		{"ROCE-200G", true},
		{"ib-400g", false},
		{"", false},
	} {
		if got := (Scope{Hardware: "h100", Fabric: c.fabric}).Admits(roce); got != c.want {
			t.Errorf("fabric %q: Admits = %v, want %v", c.fabric, got, c.want)
		}
	}
	if !(Scope{Hardware: "h100"}).Admits(coefficient.Scope{Hardware: []string{"h100"}}) {
		t.Error("an entry stating no fabric was refused on a deployment with none")
	}
}

// The selected-entry report (VAL-6) lists exactly the entries a lookup returned. Has and
// Names list what exists without selecting it; Admitted, which provenance reads, does not
// select either.
func TestSelectedListsOnlyWhatLookupsReturned(t *testing.T) {
	sets := []*coefficient.Set{{Kind: "CoefficientSet", Name: "s", Coefficients: []coefficient.Entry{
		{Name: "a", Value: 1, Method: vocab.MethodAssumed, Scope: coefficient.Scope{Hardware: []string{"h100"}}},
		{Name: "b", Value: 2, Method: vocab.MethodAssumed, Scope: coefficient.Scope{Hardware: []string{"h100"}}},
		{Name: "c", Value: 3, Method: vocab.MethodAssumed, Scope: coefficient.Scope{Hardware: []string{"h100"}}},
	}}}
	c, err := Load(sets, Scope{Hardware: "h100"})
	if err != nil {
		t.Fatal(err)
	}
	c.Has("a")
	c.Names()
	c.Admitted("a")
	if got := c.Selected(); len(got) != 0 {
		t.Fatalf("Has, Names and Admitted selected %v", got)
	}
	if _, err := c.Value("b"); err != nil {
		t.Fatal(err)
	}
	c.ValueOr("c", 0)
	c.ValueOr("missing", 0)
	got := c.Selected()
	if len(got) != 2 || got[0].Entry.Name != "b" || got[1].Entry.Name != "c" || got[0].Set != "s" {
		t.Errorf("selected %+v, want b and c from set s", got)
	}
}

// Two admitted entries of one name and scope that differ in method can only be ordered by
// method rank (M2). Until then the lookup is refused, naming both methods, rather than
// left to set order.
func TestTwoMethodsAtOneScopeAreRefusedNotOrderedBySetOrder(t *testing.T) {
	scope := coefficient.Scope{Fabric: []string{"ethernet-100gbe"}}
	mk := func(name string, m vocab.Method, v float64) *coefficient.Set {
		return &coefficient.Set{Kind: "CoefficientSet", Name: name, Coefficients: []coefficient.Entry{
			{Name: "fabric_bandwidth_efficiency", Value: v, Method: m, Scope: scope}}}
	}
	for _, order := range [][]*coefficient.Set{
		{mk("base", vocab.MethodAssumed, 0.8), mk("over", vocab.MethodLiterature, 0.9)},
		{mk("over", vocab.MethodLiterature, 0.9), mk("base", vocab.MethodAssumed, 0.8)},
	} {
		_, err := Load(order, Scope{Hardware: "h100", Fabric: "ethernet-100gbe"})
		if err == nil || !strings.Contains(err.Error(), `"assumed"`) || !strings.Contains(err.Error(), `"literature"`) {
			t.Errorf("order %s,%s: err = %v, want a refusal naming both methods",
				order[0].Name, order[1].Name, err)
		}
	}
	// The same method at the same scope in two sets is still decided by set order (the
	// later wins), as before.
	c, err := Load([]*coefficient.Set{mk("a", vocab.MethodAssumed, 0.8), mk("b", vocab.MethodAssumed, 0.9)},
		Scope{Hardware: "h100", Fabric: "ethernet-100gbe"})
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := c.Value("fabric_bandwidth_efficiency"); v != 0.9 {
		t.Errorf("later same-method entry did not win: %v", v)
	}
}
