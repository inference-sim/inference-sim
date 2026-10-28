// Moved from blis-latency-kernel/transfer_demand_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/kernel/kerneltest"
	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// The P/D handoff's separable demand (blis-schemas kernel.TransferDemand): its contract with
// PDTransferTime, checked by blis-schemas' own law test, and the laws a simulator
// arbitrating shared links relies on.

// pdKernels builds the prefill and decode kernels of one disaggregated deployment of
// scenario: its pool split into a prefill pool and a decode pool on a node each, joined by
// fabric.
func pdKernels(t *testing.T, scenario, fabric string) (prefill, decode *Kernel) {
	t.Helper()
	in := fixtureInputs(t, scenario)
	pool := in.Deployment.Pools[0]
	pool.Nodes = 1
	p, d := clonePool(pool), clonePool(pool)
	p.Role, d.Role = deployment.RolePrefill, deployment.RoleDecode
	in.Deployment.Pools = []deployment.Pool{p, d}
	in.Deployment.PDTransfer = &deployment.PDTransfer{Connector: "NixlConnector"}
	in.Scenario.Cluster.Nodes = 2
	if fabric != "" {
		in.Scenario.Cluster.Fabric = fabric
		f := fixtureInputsFabric(t, fabric)
		in.Fabric = f
	}
	in.PoolIndex = 0
	prefill = build(t, in)
	in.PoolIndex = 1
	decode = build(t, in)
	return prefill, decode
}

func TestPDTransferDemandKeepsTheContract(t *testing.T) {
	for _, c := range []struct{ scenario, fabric string }{
		{"glm5-h200-tp8.yaml", "ib-400g"},
		{"deepseek-v41-flash-h200-tp8.yaml", "ib-400g"},
		{"minimax-m25-gb300-tp4.yaml", "ib-400g"},
	} {
		t.Run(c.scenario, func(t *testing.T) {
			prefill, decode := pdKernels(t, c.scenario, c.fabric)
			// Alone, the demand costs exactly PDTransferTime, for every case.
			kerneltest.TestPDTransfer(t, prefill, kerneltest.PDCases())
			kerneltest.TestPDTransfer(t, decode, kerneltest.PDCases())
			// The two pools' kernels describe one handoff with the same links, so a
			// simulator can tell a transfer one priced shares a link with one the other did.
			for _, pc := range kerneltest.PDCases() {
				if err := kerneltest.CheckLinksAgree(prefill, decode, pc); err != nil {
					t.Error(err)
				}
			}
		})
	}
}

func TestPDTransferDemandLaws(t *testing.T) {
	prefill, _ := pdKernels(t, "glm5-h200-tp8.yaml", "ib-400g")
	const tokens = 4096
	at := func(node, rack int) kernel.Placement { return kernel.Placement{Node: node, Rack: rack} }
	ids := func(d kernel.TransferDemand) map[kernel.LinkID]bool {
		m := map[kernel.LinkID]bool{}
		for _, l := range d.Links {
			m[l.Link] = true
		}
		return m
	}
	shares := func(a, b kernel.TransferDemand) bool {
		ia := ids(a)
		for id := range ids(b) {
			if ia[id] {
				return true
			}
		}
		return false
	}

	same := prefill.PDTransferDemand(tokens, at(0, 0), at(0, 0))
	cross := prefill.PDTransferDemand(tokens, at(0, 0), at(1, 0))
	if same.Floor != cross.Floor {
		t.Errorf("the floor moved with placement: %v on one node, %v across nodes",
			same.Floor, cross.Floor)
	}
	if shares(same, cross) {
		t.Errorf("an on-node and a cross-node transfer share a link: %v and %v",
			same.Links, cross.Links)
	}
	// Different senders to different receivers cross disjoint links.
	if shares(prefill.PDTransferDemand(tokens, at(0, 0), at(1, 0)),
		prefill.PDTransferDemand(tokens, at(2, 1), at(3, 1))) {
		t.Error("two transfers between disjoint node pairs share a link")
	}
	// Two transfers out of one sender share its egress.
	if !shares(prefill.PDTransferDemand(tokens, at(0, 0), at(1, 0)),
		prefill.PDTransferDemand(tokens, at(0, 0), at(2, 0))) {
		t.Error("two transfers from one node share no link; its egress is common to both")
	}

	// A slower inter-node fabric never raises a cross-node link's rate.
	slowPrefill, _ := pdKernels(t, "glm5-h200-tp8.yaml", "ethernet-100gbe")
	slow := slowPrefill.PDTransferDemand(tokens, at(0, 0), at(1, 0))
	minRate := func(d kernel.TransferDemand) float64 {
		r := d.Links[0].BytesPerSecond
		for _, l := range d.Links {
			r = min(r, l.BytesPerSecond)
		}
		return r
	}
	if minRate(slow) > minRate(cross) {
		t.Errorf("a 100 GbE fabric moves a handoff at %v B/s, faster than 400 Gb IB's %v",
			minRate(slow), minRate(cross))
	}
}

// In a rack that is one NVLink domain, the fabric is non-blocking: two in-rack handoffs
// between disjoint trays share no link, and two out of one tray share its egress.
func TestRackHandoffsShareOnlyTheirTrays(t *testing.T) {
	prefill, _ := pdKernels(t, "minimax-m25-gb300-tp4.yaml", "ib-400g")
	at := func(node int) kernel.Placement { return kernel.Placement{Node: node, Rack: 0} }
	ids := func(d kernel.TransferDemand) map[kernel.LinkID]bool {
		m := map[kernel.LinkID]bool{}
		for _, l := range d.Links {
			m[l.Link] = true
		}
		return m
	}
	share := func(a, b kernel.TransferDemand) bool {
		ia := ids(a)
		for id := range ids(b) {
			if ia[id] {
				return true
			}
		}
		return false
	}
	if share(prefill.PDTransferDemand(4096, at(0), at(1)), prefill.PDTransferDemand(4096, at(2), at(3))) {
		t.Error("two in-rack handoffs between disjoint trays share a link")
	}
	if !share(prefill.PDTransferDemand(4096, at(0), at(1)), prefill.PDTransferDemand(4096, at(0), at(2))) {
		t.Error("two in-rack handoffs out of one tray share no link")
	}
}
