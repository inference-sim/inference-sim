package engine

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// A handoff states where its KV sits when the pool's ranks span nodes; the endpoint cannot be
// negative, and no endpoint means the pool's first node, as before.
func TestAHandoffMayStateWhereItsKVSits(t *testing.T) {
	if err := (Handoff{ID: "r"}).Validate(); err != nil {
		t.Errorf("a handoff with no endpoint: %v", err)
	}
	at := &plan.Endpoint{Node: 3, Rack: 1}
	if err := (Handoff{ID: "r", At: at}).Validate(); err != nil {
		t.Errorf("a handoff held on node 3: %v", err)
	}
	if err := (Handoff{ID: "r", At: &plan.Endpoint{Node: -1}}).Validate(); err == nil {
		t.Error("a handoff held on node -1 was accepted")
	}
}
