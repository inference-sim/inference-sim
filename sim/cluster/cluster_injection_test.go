package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestClusterArrivalEvent_IncrementsInjectedByClass verifies BC-5: every cluster
// arrival increments injectedByClass keyed by SLOClass before any admission/route
// decision.
func TestClusterArrivalEvent_IncrementsInjectedByClass(t *testing.T) {
	cs := &ClusterSimulator{
		injectedByClass:  make(map[string]int64),
		clusterEvents:    make(ClusterEventQueue, 0),
		admissionLatency: 0,
		pendingArrivals:  3,
	}
	req1 := &sim.Request{ID: "r1", SLOClass: "critical"}
	req2 := &sim.Request{ID: "r2", SLOClass: "critical"}
	req3 := &sim.Request{ID: "r3", SLOClass: ""} // empty class — counted under ""

	(&ClusterArrivalEvent{time: 0, request: req1}).Execute(cs)
	(&ClusterArrivalEvent{time: 1, request: req2}).Execute(cs)
	(&ClusterArrivalEvent{time: 2, request: req3}).Execute(cs)

	if cs.injectedByClass["critical"] != 2 {
		t.Errorf(`injectedByClass["critical"] = %d, want 2`, cs.injectedByClass["critical"])
	}
	if cs.injectedByClass[""] != 1 {
		t.Errorf(`injectedByClass[""] = %d, want 1`, cs.injectedByClass[""])
	}
}

// TestRawMetrics_InjectedByClass_DefensiveCopy verifies BC-6: CollectRawMetrics
// produces a defensive copy of the injection counter.
func TestRawMetrics_InjectedByClass_DefensiveCopy(t *testing.T) {
	src := map[string]int64{"critical": 10, "batch": 3}
	raw := CollectRawMetrics(sim.NewMetrics(), nil, 0, "fcfs", 0, 0, src)

	if raw.InjectedByClass["critical"] != 10 || raw.InjectedByClass["batch"] != 3 {
		t.Errorf("InjectedByClass mismatch: got %v, want critical=10 batch=3", raw.InjectedByClass)
	}
	src["critical"] = 999
	if raw.InjectedByClass["critical"] != 10 {
		t.Errorf("RawMetrics.InjectedByClass aliased the source map; got critical=%d, want 10", raw.InjectedByClass["critical"])
	}
}

// TestRawMetrics_InjectedByClass_NilSourceLeavesFieldEmpty verifies that nil
// (the test-callsite default) does not allocate an empty map.
func TestRawMetrics_InjectedByClass_NilSourceLeavesFieldEmpty(t *testing.T) {
	raw := CollectRawMetrics(sim.NewMetrics(), nil, 0, "fcfs", 0, 0, nil)
	if raw.InjectedByClass != nil {
		t.Errorf("InjectedByClass = %v, want nil for nil source", raw.InjectedByClass)
	}
}
