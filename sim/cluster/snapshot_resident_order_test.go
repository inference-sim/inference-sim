package cluster

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// residentOrderFixture drives one LoRA instance (capacity 4) until "b" has run to
// completion and "a" is mid-decode: both resident, b loaded first and unpinned, a
// pinned by its in-flight request. It asserts that premise before returning.
func residentOrderFixture(t *testing.T, interval int64) (*ClusterSimulator, *InstanceSimulator) {
	t.Helper()
	cs := newLoRAInstanceUnrun(t, 1, 4, interval)
	inst := cs.Instances()[0]
	reqs := newTestRequests(2)
	short, long := reqs[0], reqs[1]
	short.Adapter, short.ArrivalTime = "b", 0
	short.OutputTokens = short.OutputTokens[:1]
	long.Adapter = "a"

	inst.InjectRequest(short)
	for inst.HasPendingEvents() {
		inst.ProcessNextEvent()
	}
	require.Equal(t, []string{"b"}, inst.ResidentAdapterIDs(), "premise: b resident after its run")

	long.ArrivalTime = inst.Clock()
	inst.InjectRequest(long)
	for inst.HasPendingEvents() && (inst.BatchSize() == 0 || len(inst.ResidentAdapterIDs()) != 2) {
		inst.ProcessNextEvent()
	}
	require.Equal(t, []string{"b", "a"}, inst.ResidentAdapterIDs(), "premise: a loaded after b")
	require.Equal(t, []string{"b"}, inst.UnpinnedResidentAdapterIDs(), "premise: a pinned, b not")
	return cs, inst
}

func assertResidentOrder(t *testing.T, cs *ClusterSimulator, inst *InstanceSimulator, clock int64) {
	t.Helper()
	snap := cs.snapshotProvider.Snapshot(inst.ID(), clock)
	assert.Equal(t, map[string]bool{"a": true, "b": true}, snap.ResidentAdapters)
	assert.Equal(t, []string{"b", "a"}, snap.ResidentOrder, "true LRU→MRU order")
	assert.Equal(t, map[string]bool{"a": true}, snap.ResidentPinned, "only the in-flight adapter is pinned")
}

// Immediate (no refresh interval): Snapshot publishes the residency order and pinned
// subset live.
func TestSnapshot_ResidentOrder_Immediate(t *testing.T) {
	cs, inst := residentOrderFixture(t, 0)
	assertResidentOrder(t, cs, inst, inst.Clock())
}

// RefreshAll (the other fill path) publishes the same three fields.
func TestSnapshot_ResidentOrder_RefreshAll(t *testing.T) {
	cs, inst := residentOrderFixture(t, 1<<60) // Periodic, never due on its own
	cs.snapshotProvider.RefreshAll(inst.Clock())
	assertResidentOrder(t, cs, inst, inst.Clock())
}

// The published order does not alias the set: mutating it cannot reorder the instance.
func TestSnapshot_ResidentOrder_IsACopy(t *testing.T) {
	cs, inst := residentOrderFixture(t, 0)
	snap := cs.snapshotProvider.Snapshot(inst.ID(), inst.Clock())
	snap.ResidentOrder[0] = "zzz"
	assert.Equal(t, []string{"b", "a"}, inst.ResidentAdapterIDs())
}

// A load in progress is published as LoadingAdapter, not yet as resident.
func TestSnapshot_LoadingAdapter(t *testing.T) {
	cs := newLoRAInstanceUnrun(t, 1, 4, 0)
	inst := cs.Instances()[0]
	req := newTestRequests(1)[0]
	req.Adapter, req.ArrivalTime = "c", 0
	inst.InjectRequest(req)
	for inst.HasPendingEvents() && inst.LoadingAdapter() == "" {
		inst.ProcessNextEvent()
	}
	require.Equal(t, "c", inst.LoadingAdapter(), "premise: c is loading")
	snap := cs.snapshotProvider.Snapshot(inst.ID(), inst.Clock())
	assert.Equal(t, "c", snap.LoadingAdapter)
	assert.Nil(t, snap.ResidentOrder, "c joins the resident set only when loaded")
	cs.snapshotProvider.RefreshAll(inst.Clock())
	assert.Equal(t, "c", cs.snapshotProvider.Snapshot(inst.ID(), inst.Clock()).LoadingAdapter)

	for inst.HasPendingEvents() && inst.LoadingAdapter() != "" {
		inst.ProcessNextEvent()
	}
	snap = cs.snapshotProvider.Snapshot(inst.ID(), inst.Clock())
	assert.Equal(t, "", snap.LoadingAdapter)
	assert.Equal(t, []string{"c"}, snap.ResidentOrder)
}

// Nothing resident or loading ⇒ the order, pinned set and loading adapter are empty.
func TestSnapshot_ResidentOrder_NilWhenEmpty(t *testing.T) {
	cs := newLoRAInstanceUnrun(t, 1, 4, 0)
	inst := cs.Instances()[0]
	snap := cs.snapshotProvider.Snapshot(inst.ID(), 0)
	assert.Nil(t, snap.ResidentOrder)
	assert.Nil(t, snap.ResidentPinned)
	assert.Equal(t, "", snap.LoadingAdapter)
}
