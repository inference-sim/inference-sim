package latencykernel

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// A KV move that every node of a prefill pool makes at once (plan.Transfer.Nodes) is the sum of
// each node's pull over its own links. Two nodes of one switch group pulling to the other group
// load the group's uplink twice, so the move takes twice as long as one node's pull, while two
// nodes inside one group, each on its own NIC, take as long as one.
func TestAMoveEveryNodeMakesLoadsTheSharedUplinkOnce(t *testing.T) {
	k := pdKernel(t, fourByThree)
	one := k.PDTransferDemand(tokens, at(0), at(4))
	across := kernel.Concurrent(
		k.PDTransferDemand(tokens, at(0), at(4)),
		k.PDTransferDemand(tokens, at(1), at(5)),
	)
	if err := across.Check(); err != nil {
		t.Fatalf("the combined demand is malformed: %v", err)
	}
	if ratio := float64(across.AloneTime()) / float64(one.AloneTime()); math.Abs(ratio-2) > 0.001 {
		t.Errorf("two nodes pulling across the uplink take %v, one takes %v (x%.3f); want x2",
			across.AloneTime(), one.AloneTime(), ratio)
	}
	inside := kernel.Concurrent(
		k.PDTransferDemand(tokens, at(0), at(2)),
		k.PDTransferDemand(tokens, at(1), at(3)),
	)
	single := k.PDTransferDemand(tokens, at(0), at(2))
	if inside.AloneTime() != single.AloneTime() {
		t.Errorf("two nodes pulling inside one group take %v, one takes %v; each has a NIC of its own",
			inside.AloneTime(), single.AloneTime())
	}
	// Reporting one endpoint for the two-node pool prices the cross-group move at one node's pull.
	if one.AloneTime() >= across.AloneTime() {
		t.Errorf("one endpoint prices the move at %v, the two nodes at %v: the single endpoint is too low",
			one.AloneTime(), across.AloneTime())
	}
}
