package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// ClusterSimulator.Arrivals is the Result's generated count (#1871): every request that
// reaches the cluster, once. It is observed apart from INV-1's buckets, so these laws are what
// keep the ledger's generated == injected + rejected clause a check rather than a restatement.

// Every request of the workload is one arrival.
func TestArrivals_CountsEachRequestOfTheWorkloadOnce(t *testing.T) {
	const n = 8
	cs := newTestCluster(newTestDeploymentConfig(2), NewSliceRequestSource(newTestRequests(n)), nil)
	if err := cs.Run(); err != nil {
		t.Fatal(err)
	}
	if got := cs.Arrivals(); got != n {
		t.Errorf("Arrivals() = %d, want the %d requests the workload produced", got, n)
	}
	assertClusterINV1Conservation(t, cs, cs.Arrivals(), cs.RejectedRequests(), "arrivals")
}

// An arrival the client's gate refuses (an agentic driver holding a send) is not a request:
// it is not an arrival.
func TestArrivals_AGateRefusedArrivalIsNotCounted(t *testing.T) {
	reqs := newTestRequests(4)
	cs := newTestCluster(newTestDeploymentConfig(1), NewSliceRequestSource(reqs), nil)
	refused := reqs[1].ID
	cs.SetArrivalGate(func(req *sim.Request, _ int64) (bool, []*sim.Request) { return req.ID != refused, nil })
	if err := cs.Run(); err != nil {
		t.Fatal(err)
	}
	if got := cs.Arrivals(); got != 3 {
		t.Errorf("Arrivals() = %d, want 3 (one of 4 refused at the gate)", got)
	}
}
