package rules

import (
	"reflect"
	"testing"
)

// executed is the release blis-vllm executes: blisvllm.Upstream.Version in the root module
// (provenance.go), which this data-only module must not import.
const executed = "0.32.0"

// Original to blis-vllm.

// The registry holds a pack for every release a scenario file may declare, and one for the
// release the engine executes: a scenario declaring blis-vllm's own Upstream must find rules.
func TestEveryReleaseHasItsPack(t *testing.T) {
	if got, want := Versions(), []string{"0.29.0", "0.31.0", "0.32.0"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Versions() = %v, want %v", got, want)
	}
	if Lookup(executed) == nil {
		t.Errorf("no pack for %s, the release the engine executes", executed)
	}
	for _, v := range Versions() {
		p := Lookup(v)
		if p == nil || p.Version != v || len(p.Rules) == 0 {
			t.Errorf("Lookup(%q) = %v: not that release's pack, or it has no rules", v, p)
		}
	}
	if Lookup("0.30.0") != nil {
		t.Error("a release with no pack must look up as nil, not as a neighbour's")
	}
}
