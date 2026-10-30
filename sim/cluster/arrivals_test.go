package cluster

import (
	"container/heap"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

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

// A drain redirect re-arrives a request at the cluster; it is the same request, not a new one.
func TestArrivals_ADrainRedirectIsNotANewArrival(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	cfg.InstanceLifecycle = InstanceLifecycleConfig{DrainPolicy: string(DrainPolicyRedirect)}
	editEngine(&cfg, func(c *simConfig) { c.BatchConfig = cengine.NewBatchConfig(1, 2048, 0) }) // one running request: the rest queue
	const n = 12
	reqs := newTestRequests(n)
	for _, r := range reqs {
		r.ArrivalTime = 0 // all at once, so instance 0 holds a queue to redirect
	}
	cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
	heap.Push(&cs.clusterEvents, clusterEventEntry{event: &drainAt{at: 0, inst: 0}, seqID: cs.nextSeqID()})
	if err := cs.Run(); err != nil {
		t.Fatal(err)
	}
	redirected := 0
	for _, r := range reqs {
		if r.Redirected {
			redirected++
		}
	}
	if redirected == 0 {
		t.Fatal("non-vacuity: the drain redirected no request")
	}
	if got := cs.Arrivals(); got != n {
		t.Errorf("Arrivals() = %d after %d redirects, want the %d requests", got, redirected, n)
	}
	assertClusterINV1Conservation(t, cs, n, cs.RejectedRequests(), "drain redirect")
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
