package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"

	"github.com/inference-sim/inference-sim/sim"
)

type fakeHolder struct{ at map[string]*plan.Endpoint }

func (f fakeHolder) HandoffAt(r *sim.Request) *plan.Endpoint { return f.at[r.ID] }

// The decode engine pulls a request's KV from the node its prefill engine says holds it
// (engine.Handoff.At: the rank of a data-parallel group that computed it), not from the first node
// of the prefill group, which is where a group spanning nodes would otherwise be priced from
// (blis-vllm#267). An engine that states nothing keeps the prefill instance's own endpoint.
func TestADecodeRequestPullsFromTheNodeThatOwnsTheHandoff(t *testing.T) {
	own := plan.Endpoint{Node: 0, Rack: 0}
	owner := plan.Endpoint{Node: 3, Rack: 1}
	holder := fakeHolder{at: map[string]*plan.Endpoint{"far": &owner, "near": nil}}
	for _, c := range []struct {
		name string
		pre  handoffAter
		sub  *sim.Request
		want plan.Endpoint
	}{
		{"the engine states the owning node", holder, &sim.Request{ID: "far"}, owner},
		{"the engine states nothing", holder, &sim.Request{ID: "near"}, own},
		{"a request the engine holds no handoff for", holder, &sim.Request{ID: "unknown"}, own},
		{"no prefill sub-request", holder, nil, own},
		{"no prefill instance", nil, &sim.Request{ID: "far"}, own},
	} {
		if got := handoffPeer(c.pre, c.sub, own); got != c.want {
			t.Errorf("%s: pulls from %+v, want %+v", c.name, got, c.want)
		}
	}
}
