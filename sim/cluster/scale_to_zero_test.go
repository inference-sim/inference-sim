package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// scaleToZeroRun runs a one-pool cluster of two instances serving "test-model" with the model
// autoscaler ticking every second (its analyzer decides nothing), on the given requests, and
// returns the cluster and the clock at which the last instance left service (0 if none did).
func scaleToZeroRun(t *testing.T, idleUs float64, reqs []*sim.Request) *ClusterSimulator {
	t.Helper()
	d := newTestDeploymentConfig(2)
	d.Model = "test-model"
	d.ModelAutoscalerIntervalUs = 1_000_000
	d.ScaleToZeroIdleUs = idleUs
	d.NodePools = []NodePoolConfig{{Name: "p", GPUType: "A100", GPUsPerNode: 2, InitialNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80}}
	cs := newTestCluster(d, NewSliceRequestSource(reqs), nil)
	cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, &nopEngine{}, &nopActuator{})
	mustRun(t, cs)
	return cs
}

func servingReplicas(cs *ClusterSimulator) int {
	n := 0
	for _, inst := range cs.instances {
		if inst.State == sim.InstanceStateActive || inst.State == sim.InstanceStateWarmingUp {
			n++
		}
	}
	return n
}

// After the idle period with no request, all of a model's replicas are drained and the node they
// held is left empty and flagged; every request that arrived was served first (INV-1).
func TestScaleToZero_DrainsAnIdleModel(t *testing.T) {
	reqs := newTestRequests(3) // all arrive within the first moments
	cs := scaleToZeroRun(t, 3_000_000, reqs)
	if n := servingReplicas(cs); n != 0 {
		t.Errorf("%d replica(s) still serving after the idle period, want none", n)
	}
	if got := cs.AggregatedMetrics().CompletedRequests; got != len(reqs) {
		t.Errorf("%d of %d requests completed; the drain must not take work", got, len(reqs))
	}
	if idle := cs.IdleNodes(); len(idle) != 1 {
		t.Errorf("idle nodes = %v, want the one node the replicas left empty", idle)
	}
	if err := cs.placement.VerifyConservation(); err != nil {
		t.Errorf("GPU conservation: %v", err)
	}
}

// With scale to zero off (0), or while requests keep coming inside the idle period, nothing drains.
func TestScaleToZero_DoesNotDrainABusyOrDisabledModel(t *testing.T) {
	if n := servingReplicas(scaleToZeroRun(t, 0, newTestRequests(3))); n != 2 {
		t.Errorf("scale to zero off: %d replicas serving, want both", n)
	}
	// A request every 2 s over 10 s with an idle period of 3 s: the timer resets at each arrival, so
	// nothing drains while they arrive. After the last, the model drains.
	reqs := newTestRequests(5)
	for i, r := range reqs {
		r.ArrivalTime = int64(i) * 2_000_000
	}
	cs := scaleToZeroRun(t, 3_000_000, reqs)
	if got := cs.AggregatedMetrics().CompletedRequests; got != len(reqs) {
		t.Errorf("%d of %d requests completed: replicas were drained while requests were still arriving", got, len(reqs))
	}
	if n := servingReplicas(cs); n != 0 {
		t.Errorf("%d replica(s) serving after the last request and the idle period, want none", n)
	}
}
