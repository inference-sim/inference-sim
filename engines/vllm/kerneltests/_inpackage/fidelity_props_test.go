// Moved from blis-latency-kernel/fidelity_props_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
)

// (#29) On a rack that is one NVLink domain, a same-rack cross-node handoff avoids the
// NICs: strictly faster than a cross-rack one, sharing none of its links.
func TestRackDomainHandoffAvoidsTheNICs(t *testing.T) {
	prefill, _ := pdKernels(t, "minimax-m25-gb300-tp4.yaml", "ib-400g")
	at := func(node, rack int) kernel.Placement { return kernel.Placement{Node: node, Rack: rack} }
	inRack := prefill.PDTransferDemand(1<<16, at(0, 0), at(1, 0))
	acrossRack := prefill.PDTransferDemand(1<<16, at(0, 0), at(1, 1))
	if inRack.AloneTime() >= acrossRack.AloneTime() {
		t.Errorf("an in-rack handoff takes %v, not faster than a cross-rack one's %v",
			inRack.AloneTime(), acrossRack.AloneTime())
	}
	for _, a := range inRack.Links {
		for _, b := range acrossRack.Links {
			if a.Link == b.Link {
				t.Errorf("in-rack and cross-rack handoffs share %s", a.Link)
			}
		}
	}
}

// (#29) Two handoffs into one receiver share its ingress.
func TestOneReceiverSharesItsIngress(t *testing.T) {
	prefill, _ := pdKernels(t, "glm5-h200-tp8.yaml", "ib-400g")
	at := func(node int) kernel.Placement { return kernel.Placement{Node: node} }
	a := prefill.PDTransferDemand(4096, at(0), at(2))
	b := prefill.PDTransferDemand(4096, at(1), at(2))
	for _, x := range a.Links {
		for _, y := range b.Links {
			if x.Link == y.Link {
				return
			}
		}
	}
	t.Errorf("two handoffs into node 2 share no link: %v and %v", a.Links, b.Links)
}
