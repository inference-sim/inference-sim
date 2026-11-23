package rulepack

import (
	"github.com/inference-sim/inference-sim/engines/vllm/internal/upstream"
	"reflect"
	"testing"
)

// executed is the release blis-vllm executes: blisvllm.Upstream.Version in the root module
// (provenance.go), the engine release the pin states (internal/upstream).
const executed = upstream.Version

// Original to blis-vllm.

// The registry holds the pack of the one release the engine executes: a scenario declaring
// blis-vllm's own Upstream must find rules.
func TestEveryReleaseHasItsPack(t *testing.T) {
	if got, want := Versions(), []string{upstream.Version}; !reflect.DeepEqual(got, want) {
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
