package resolve

import (
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
