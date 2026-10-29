package cluster

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Two runs with identical seed, requests and the default gateway produce identical
// aggregated metrics (INV-6 regression guard).
func TestCluster_IdenticalRunsProduceIdenticalMetrics(t *testing.T) {
	requests1 := newTestRequests(50)
	requests2 := newTestRequests(50)

	cfg1 := newTestDeploymentConfig(2)
	cs1 := newTestCluster(cfg1, NewSliceRequestSource(requests1), nil)
	mustRun(t, cs1)

	cfg2 := newTestDeploymentConfig(2)
	cs2 := newTestCluster(cfg2, NewSliceRequestSource(requests2), nil)
	mustRun(t, cs2)

	m1, err1 := json.Marshal(cs1.AggregatedMetrics())
	m2, err2 := json.Marshal(cs2.AggregatedMetrics())
	if err1 != nil || err2 != nil {
		t.Fatalf("json marshal error: %v / %v", err1, err2)
	}
	if !bytes.Equal(m1, m2) {
		t.Error("two identical runs produced different aggregated metrics (INV-6 violated)")
	}
}

// Every gateway admission rejection is counted against the request's SLO class in
// ShedByTier, whatever the class.
func TestShedByTier_AdmissionRejections_PopulateTierCounts(t *testing.T) {
	requests := []*sim.Request{
		{ID: "r1", ArrivalTime: 0, SLOClass: "critical", InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5), State: sim.StateQueued},
		{ID: "r2", ArrivalTime: 10, SLOClass: "standard", InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5), State: sim.StateQueued},
		{ID: "r3", ArrivalTime: 20, SLOClass: "batch", InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5), State: sim.StateQueued},
	}
	cs := newTestCluster(newTestDeploymentConfig(2), NewSliceRequestSource(requests), nil)
	cs.SetGateway(&fakeGateway{reject: func(*sim.Request) bool { return true }})
	mustRun(t, cs)

	if cs.RejectedRequests() != 3 {
		t.Fatalf("expected 3 rejections, got %d", cs.RejectedRequests())
	}
	shed := cs.ShedByTier()
	for _, class := range []string{"critical", "standard", "batch"} {
		if shed[class] != 1 {
			t.Errorf("ShedByTier[%q] = %d, want 1", class, shed[class])
		}
	}
}
