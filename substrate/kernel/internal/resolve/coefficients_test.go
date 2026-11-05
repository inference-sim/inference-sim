package resolve

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
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
