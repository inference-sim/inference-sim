package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// admitToLedger records req as created and arrived, for a fixture that hands a request to an
// instance directly instead of through the cluster's arrival.
func admitToLedger(t *testing.T, cs *ClusterSimulator, req *sim.Request) {
	t.Helper()
	cs.createRequest(req)
	cs.arriveRequest(req)
	if cs.fateErr != nil {
		t.Fatal(cs.fateErr)
	}
}
