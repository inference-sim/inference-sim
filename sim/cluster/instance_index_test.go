package cluster

import "testing"

// instanceByID is a lookup by the ID index, not a scan (DEV-5). The index follows the
// instances the cluster builds at startup and the ones added while it runs, and it recovers when
// the slice is replaced under it.
func TestInstanceByID_UsesTheIndexAndFollowsTheInstances(t *testing.T) {
	cs := newTestCluster(newTestDeploymentConfig(3), NewSliceRequestSource(nil), nil)
	if len(cs.byID) != 3 {
		t.Fatalf("the index holds %d instances after construction, want 3", len(cs.byID))
	}
	for _, inst := range cs.instances {
		if got := cs.instanceByID(inst.ID()); got != inst {
			t.Errorf("instanceByID(%s) = %p, want %p", inst.ID(), got, inst)
		}
	}
	if cs.instanceByID("no-such-instance") != nil {
		t.Error("an unknown ID found an instance")
	}
	// A lookup is answered from the index: an index entry with no instance behind it in the slice
	// is still returned only while the index and the slice agree in size, so remove one of each.
	gone := cs.instances[2]
	cs.instances = cs.instances[:2]
	if cs.instanceByID(gone.ID()) != nil {
		t.Error("an instance removed from the slice is still found")
	}
	if cs.instanceByID(cs.instances[0].ID()) != cs.instances[0] {
		t.Error("after the rebuild the remaining instances are not found")
	}
	cs.addInstance(gone)
	if cs.instanceByID(gone.ID()) != gone {
		t.Error("an added instance is not found")
	}
}
