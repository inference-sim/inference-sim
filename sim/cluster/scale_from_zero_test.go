package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// coldStartRun scales a two-replica model to zero after 3 s idle, then sends one more request at
// 20 s. holdUs is the scale-from-zero hold (0: off). It returns the cluster and that late request.
func coldStartRun(t *testing.T, holdUs float64) (*ClusterSimulator, *sim.Request) {
	t.Helper()
	reqs := newTestRequests(4)
	for i, r := range reqs[:3] {
		r.ArrivalTime = int64(i) * 100_000
	}
	late := reqs[3]
	late.ArrivalTime = 20_000_000
	d := newTestDeploymentConfig(2)
	d.Model = "test-model"
	d.ModelAutoscalerIntervalUs = 1_000_000
	d.ScaleToZeroIdleUs = 3_000_000
	d.ScaleFromZeroHoldUs = holdUs
	d.InstanceLifecycle = InstanceLifecycleConfig{LoadingDelay: DelaySpec{Mean: 2}, WarmStartInitialInstances: true}
	d.NodePools = []NodePoolConfig{{Name: "p", GPUType: "A100", GPUsPerNode: 2, InitialNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80}}
	cs := newTestCluster(d, NewSliceRequestSource(reqs), nil)
	cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, &nopEngine{}, NewDirectActuator(cs))
	mustRun(t, cs)
	return cs, late
}

// A request that finds its model scaled to zero starts a replica and is served after the cold
// start, its wait counted in its latency (#908); with the hold off it is rejected, as before.
func TestScaleFromZero_ARequestStartsAReplicaAndWaitsForIt(t *testing.T) {
	cs, late := coldStartRun(t, 60_000_000)
	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 4 {
		t.Fatalf("%d of 4 requests completed; the late one must be served after the cold start", m.CompletedRequests)
	}
	rm, ok := m.Requests[late.ID]
	if !ok {
		t.Fatalf("no metrics for the late request %s", late.ID)
	}
	if e2e := m.RequestE2Es[late.ID]; e2e < 2_000_000 {
		t.Errorf("the late request's end-to-end time is %.0f µs, want at least the 2 s loading delay it waited out (%+v)", e2e, rm)
	}
	if held := cs.heldRequests(); len(held) != 0 {
		t.Errorf("requests still held at the end: %v", held)
	}
	if err := cs.placement.VerifyConservation(); err != nil {
		t.Errorf("GPU conservation: %v", err)
	}

	off, _ := coldStartRun(t, 0)
	if got := off.AggregatedMetrics().CompletedRequests; got != 3 {
		t.Errorf("with the hold off %d requests completed, want the 3 that arrived before the model went to zero", got)
	}
}

// A request is held only as long as the hold allows; then it goes to the gateway, which rejects it
// if nothing serves (the hold is shorter than the cold start here).
func TestScaleFromZero_HoldIsBounded(t *testing.T) {
	cs, _ := coldStartRun(t, 500_000)
	if got := cs.AggregatedMetrics().CompletedRequests; got != 3 {
		t.Errorf("%d completed; a 0.5 s hold is shorter than the 2 s cold start, so the late request is rejected", got)
	}
	if held := cs.heldRequests(); len(held) != 0 {
		t.Errorf("requests still held at the end: %v", held)
	}
}

// A closed-loop session whose think time exceeds the idle period loses its replicas between
// rounds, and the next round starts one and is served: no session is abandoned (INV-11).
func TestScaleFromZero_ASessionSurvivesScaleToZero(t *testing.T) {
	reqs := newTestRequests(2)
	first, next := reqs[0], reqs[1]
	first.ArrivalTime = 0
	next.ArrivalTime = 10_000_000 // the second round, 10 s of think time later
	followed := false
	onDone := func(r *sim.Request, _ int64) []*sim.Request {
		if r.ID == first.ID && !followed {
			followed = true
			return []*sim.Request{next}
		}
		return nil
	}
	d := newTestDeploymentConfig(1)
	d.Model = "test-model"
	d.ModelAutoscalerIntervalUs = 1_000_000
	d.ScaleToZeroIdleUs = 3_000_000
	d.ScaleFromZeroHoldUs = 60_000_000
	d.InstanceLifecycle = InstanceLifecycleConfig{LoadingDelay: DelaySpec{Mean: 2}, WarmStartInitialInstances: true}
	d.NodePools = []NodePoolConfig{{Name: "p", GPUType: "A100", GPUsPerNode: 1, InitialNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80}}
	cs := newTestCluster(d, NewSliceRequestSource([]*sim.Request{first}), onDone)
	cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, &nopEngine{}, NewDirectActuator(cs))
	mustRun(t, cs)
	if got := cs.AggregatedMetrics().CompletedRequests; got != 2 {
		t.Errorf("%d of the session's 2 rounds completed; the second must be served after the model came back from zero", got)
	}
}
