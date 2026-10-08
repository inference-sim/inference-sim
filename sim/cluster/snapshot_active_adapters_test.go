package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests cover the router-observable LoRA snapshot fields ActiveAdapters and
// MaxLoras (the BLIS analogue of vLLM's vllm:lora_requests_info as llm-d reads it).
// Unlike ResidentAdapters (LRU ground truth), ActiveAdapters is derived from the
// instance's wait queue + running batch only.

// newLoRAInstanceUnrun builds (but does not run) a cluster of n instances with
// LoRA capacity capVal and returns it. capVal 0 leaves the LoRA subsystem inert.
func newLoRAInstanceUnrun(t *testing.T, n, capVal int, interval int64) *ClusterSimulator {
	t.Helper()
	config := newTestDeploymentConfig(n)
	config.RoutingPolicy = "round-robin"
	config.SnapshotRefreshInterval = interval
	if capVal > 0 {
		c := capVal
		base, bw, fp := 1000.0, 2.0e6, 2.0e6
		config.LoRAConfig = sim.LoRAConfig{
			AdapterCapacity:       &c,
			LoadBaseLatencyUs:     &base,
			LoadBandwidthBytesUs:  &bw,
			FootprintBytesPerRank: &fp,
			Adapters: []sim.AdapterSpec{
				{ID: "a", Rank: 8}, {ID: "b", Rank: 8}, {ID: "c", Rank: 8},
			},
		}
	}
	cs := NewClusterSimulator(config, NewSliceRequestSource(nil), nil)
	require.Len(t, cs.Instances(), n)
	return cs
}

func adapterReq(id, adapter string) *sim.Request {
	return &sim.Request{ID: id, Adapter: adapter}
}

// The accessor counts queued AND running requests per adapter, skips base-model
// requests, and returns nil when nothing adapter-bearing is present.
func TestInstanceActiveAdapterCounts_QueueAndBatch(t *testing.T) {
	cs := newLoRAInstanceUnrun(t, 1, 4, 0)
	inst := cs.Instances()[0]

	require.Nil(t, inst.ActiveAdapterCounts(), "empty instance ⇒ nil")

	inst.sim.WaitQ.Enqueue(adapterReq("q1", "a"))
	inst.sim.WaitQ.Enqueue(adapterReq("q2", "a"))
	inst.sim.WaitQ.Enqueue(adapterReq("q3", "")) // base model: not counted
	inst.sim.RunningBatch = &sim.Batch{Requests: []*sim.Request{
		adapterReq("r1", "a"), adapterReq("r2", "b"), adapterReq("r3", ""),
	}}

	assert.Equal(t, map[string]int{"a": 3, "b": 1}, inst.ActiveAdapterCounts())

	// Queue-only and batch-only are each sufficient for membership.
	inst.sim.RunningBatch = nil
	assert.Equal(t, map[string]int{"a": 2}, inst.ActiveAdapterCounts(), "queue-only")
	inst.sim.DrainWaitQueue()
	inst.sim.RunningBatch = &sim.Batch{Requests: []*sim.Request{adapterReq("r2", "b")}}
	assert.Equal(t, map[string]int{"b": 1}, inst.ActiveAdapterCounts(), "batch-only")
}

// Immediate (default): the snapshot reflects live queue state and the capacity.
// ActiveAdapters is independent of residency: nothing is resident here, yet the
// queued adapter is reported (a real router sees the queued request, not the LRU).
func TestSnapshot_ActiveAdapters_ImmediateLive(t *testing.T) {
	cs := newLoRAInstanceUnrun(t, 1, 4, 0)
	inst := cs.Instances()[0]
	require.Empty(t, inst.ResidentAdapterIDs(), "precondition: nothing resident")

	inst.sim.WaitQ.Enqueue(adapterReq("q1", "c"))
	snap := cs.snapshotProvider.Snapshot(inst.ID(), 0)
	assert.Equal(t, map[string]int{"c": 1}, snap.ActiveAdapters)
	assert.Equal(t, 4, snap.MaxLoras)
	assert.Nil(t, snap.ResidentAdapters, "ActiveAdapters must not be derived from residency")
}

// Periodic (--snapshot-refresh-interval > 0): the fields go stale like a scrape —
// refreshed only once the interval has elapsed since the last refresh.
func TestSnapshot_ActiveAdapters_PeriodicStaleness(t *testing.T) {
	const interval = 1000
	cs := newLoRAInstanceUnrun(t, 1, 4, interval)
	inst := cs.Instances()[0]
	p := cs.snapshotProvider

	inst.sim.WaitQ.Enqueue(adapterReq("q1", "a"))
	s0 := p.Snapshot(inst.ID(), 0) // fresh provider, lastRefresh 0: not yet due
	assert.Nil(t, s0.ActiveAdapters, "clock 0 < interval ⇒ stale zero value")
	assert.Equal(t, 0, s0.MaxLoras, "MaxLoras rides the same scrape ⇒ stale zero")

	s1 := p.Snapshot(inst.ID(), interval)
	assert.Equal(t, map[string]int{"a": 1}, s1.ActiveAdapters, "interval elapsed ⇒ refreshed")
	assert.Equal(t, 4, s1.MaxLoras)

	inst.sim.WaitQ.Enqueue(adapterReq("q2", "b"))
	s2 := p.Snapshot(inst.ID(), interval+interval/2)
	assert.Equal(t, map[string]int{"a": 1}, s2.ActiveAdapters, "mid-interval ⇒ frozen view")

	s3 := p.Snapshot(inst.ID(), 2*interval)
	assert.Equal(t, map[string]int{"a": 1, "b": 1}, s3.ActiveAdapters, "next interval ⇒ refreshed")

	// RefreshAll refreshes regardless of mode.
	inst.sim.WaitQ.Enqueue(adapterReq("q3", "c"))
	p.RefreshAll(2*interval + 1)
	s4 := p.Snapshot(inst.ID(), 2*interval+2)
	assert.Equal(t, map[string]int{"a": 1, "b": 1, "c": 1}, s4.ActiveAdapters)
}

// route-to-holder pins ResidentAdapters to Immediate (D7) but must NOT pin the
// router-observable ActiveAdapters: it stays at the global scrape interval.
func TestPinResidentAdaptersImmediate_LeavesActiveAdaptersPeriodic(t *testing.T) {
	cfg := newObservabilityConfig(hugeInterval, 0)
	require.Equal(t, Periodic, cfg.ActiveAdapters.Mode)
	cfg.PinResidentAdaptersImmediate()
	assert.Equal(t, Periodic, cfg.ActiveAdapters.Mode)
	assert.Equal(t, hugeInterval, cfg.ActiveAdapters.Interval)
}

// INV-6 inert default: with no adapter capacity (LoRA off) the fields stay at their
// zero values even when requests carry an adapter id — vLLM without LoRA publishes
// no lora_requests_info metric.
func TestSnapshot_ActiveAdapters_InertWhenNoCapacity(t *testing.T) {
	cs := newLoRAInstanceUnrun(t, 1, 0, 0)
	inst := cs.Instances()[0]
	inst.sim.WaitQ.Enqueue(adapterReq("q1", "a"))
	require.Equal(t, map[string]int{"a": 1}, inst.ActiveAdapterCounts(),
		"precondition: the accessor itself does see the request")

	snap := cs.snapshotProvider.Snapshot(inst.ID(), 0)
	assert.Nil(t, snap.ActiveAdapters)
	assert.Equal(t, 0, snap.MaxLoras)

	cs.snapshotProvider.RefreshAll(1)
	snap = cs.snapshotProvider.Snapshot(inst.ID(), 1)
	assert.Nil(t, snap.ActiveAdapters)
	assert.Equal(t, 0, snap.MaxLoras)
}

// End to end through buildRouterState after a real run: everything drained, so no
// adapter is active even though one is resident (the two signals differ).
func TestBuildRouterState_ActiveAdaptersEmptyAfterDrain(t *testing.T) {
	cs, inst := runLoRAClusterWithResident(t, "round-robin", 0)
	require.Contains(t, inst.ResidentAdapterIDs(), "adapter_x")
	state := buildRouterState(cs, nil)
	require.Len(t, state.Snapshots, 1)
	assert.Nil(t, state.Snapshots[0].ActiveAdapters, "drained ⇒ nothing queued/running")
	assert.Equal(t, 8, state.Snapshots[0].MaxLoras)
	assert.True(t, state.Snapshots[0].ResidentAdapters["adapter_x"])
}
