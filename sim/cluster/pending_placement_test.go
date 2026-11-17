package cluster

import (
	"testing"
)

// pendingFixture is a cluster whose only pool has one node of 2 GPUs, both free, no instances, and
// room to provision a second node.
func pendingFixture(t *testing.T) *ClusterSimulator {
	t.Helper()
	d := newTestDeploymentConfig(1)
	d.Model = "test-model"
	d.NodePools = []NodePoolConfig{{Name: "p", GPUType: "A100-80", GPUsPerNode: 2, InitialNodes: 1, MaxNodes: 2, GPUMemoryGiB: 80}}
	cs := newTestCluster(d, NewSliceRequestSource(nil), nil)
	for _, inst := range cs.instances { // the startup instance gives its GPU back
		if err := cs.placement.ReleaseInstance(inst.ID()); err != nil {
			t.Fatal(err)
		}
	}
	cs.instances = nil
	return cs
}

// A scale-up with no room is queued, not dropped (#741): it appears in the queue with its model,
// variant and the time it was requested, a repeat of the same decision does not queue it twice, and
// the queue is in arrival order.
func TestPendingPlacement_ScaleUpWithoutRoomIsQueuedOnce(t *testing.T) {
	cs := pendingFixture(t)
	cs.clock = 1234
	a := NewDirectActuator(cs)
	want := ScaleDecision{ModelID: "test-model", Variant: NewVariantSpec("A100-80", 2), Delta: 2}
	// One 2-GPU replica fits; the second does not.
	if err := a.Apply([]ScaleDecision{want}); err != nil {
		t.Fatal(err)
	}
	if len(cs.instances) != 1 {
		t.Fatalf("%d instance(s) placed, want the one that fits", len(cs.instances))
	}
	q := cs.placement.Pending()
	if len(q) != 1 || q[0].ModelID != "test-model" || q[0].Variant != want.Variant || q[0].RequestedAt != 1234 {
		t.Fatalf("pending queue = %+v, want one entry for test-model %v requested at 1234", q, want.Variant)
	}
	cs.clock = 5678
	// The next tick sees one more replica, so it still wants the one that is waiting.
	again := ScaleDecision{ModelID: "test-model", Variant: want.Variant, Delta: 1}
	if err := a.Apply([]ScaleDecision{again}); err != nil {
		t.Fatal(err)
	}
	if q := cs.placement.Pending(); len(q) != 1 || q[0].RequestedAt != 1234 {
		t.Errorf("a repeated decision changed the queue to %+v; the waiting placement already covers it", q)
	}
}

// When a node becomes Ready the queue is retried in arrival order, the placed replicas leave it,
// and the instance serves the model that asked for it (the integration path: autoscaler asks, no
// room, node provisioned, placement retried).
func TestPendingPlacement_NodeReadyPlacesInArrivalOrder(t *testing.T) {
	cs := pendingFixture(t)
	a := NewDirectActuator(cs)
	two := NewVariantSpec("A100-80", 2)
	if err := a.Apply([]ScaleDecision{{ModelID: "m-first", Variant: two, Delta: 1}}); err != nil { // fills the node
		t.Fatal(err)
	}
	cs.clock = 10
	_ = a.Apply([]ScaleDecision{{ModelID: "m-second", Variant: two, Delta: 1}})
	cs.clock = 20
	_ = a.Apply([]ScaleDecision{{ModelID: "m-third", Variant: two, Delta: 1}})
	q := cs.placement.Pending()
	if len(q) != 2 || q[0].ModelID != "m-second" || q[1].ModelID != "m-third" {
		t.Fatalf("queue = %+v, want m-second then m-third", q)
	}

	// One new node has room for one more replica: the earlier request is served, the later waits.
	node, _ := cs.placement.ProvisionNode("p", cs.clock)
	(&NodeReadyEvent{timestamp: cs.clock, nodeID: node.ID}).Execute(cs)
	q = cs.placement.Pending()
	if len(q) != 1 || q[0].ModelID != "m-third" {
		t.Errorf("queue after the node came up = %+v, want only m-third", q)
	}
	served := map[string]bool{}
	for _, inst := range cs.instances {
		served[inst.Model] = true
	}
	if !served["m-second"] || served["m-third"] {
		t.Errorf("models served = %v, want m-second placed on the new node and m-third still waiting", served)
	}
	if err := cs.placement.VerifyConservation(); err != nil {
		t.Errorf("GPU conservation after the retry: %v", err)
	}
}

// A request no pool can ever hold is an error at once, not a placement that waits forever.
func TestPendingPlacement_UnsatisfiableIsNotQueued(t *testing.T) {
	cs := pendingFixture(t)
	a := NewDirectActuator(cs)
	err := a.Apply([]ScaleDecision{{ModelID: "test-model", Variant: NewVariantSpec("A100-80", 3), Delta: 1}})
	if err == nil {
		t.Error("TP 3 on 2-GPU nodes is unsatisfiable in any pool and must be reported")
	}
	if n := len(cs.placement.Pending()); n != 0 {
		t.Errorf("%d placement(s) queued for a request that can never be placed", n)
	}
}
