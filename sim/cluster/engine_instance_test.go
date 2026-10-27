package cluster

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine/enginetest"

	"github.com/inference-sim/inference-sim/sim"
)

// A cluster whose engines are built through the engine.Engine contract (DeploymentConfig.NewEngine)
// runs every request to completion through the cluster's own loop and gateway: the instances
// are engine-driven, and the report is the host's.
func TestCluster_EngineDrivenInstancesCompleteEveryRequest(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	built := 0
	cfg.NewEngine = func(int, sim.SimConfig) (*sim.EngineDriver, error) {
		built++
		eng, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 8, TokenBudget: 256, StepBaseUs: 1000, StepPerTokenUs: 2})
		if err != nil {
			return nil, err
		}
		return sim.NewEngineDriver(eng, sim.EngineDriverConfig{Horizon: cfg.Horizon}), nil
	}
	reqs := newTestRequests(20)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	if built != 2 {
		t.Fatalf("engines built = %d, want 2", built)
	}
	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 20 {
		t.Fatalf("completed = %d, want 20 (dropped %d, timed out %d, still queued %d, running %d)",
			m.CompletedRequests, m.DroppedUnservable, m.TimedOutRequests, m.StillQueued, m.StillRunning)
	}
	for _, r := range reqs {
		if r.FirstTokenTime <= 0 {
			t.Errorf("request %s has no first-token time", r.ID)
		}
		if r.State != sim.StateCompleted {
			t.Errorf("request %s ended %v, want completed", r.ID, r.State)
		}
	}
	for _, inst := range cs.Instances() {
		if inst.drv == nil {
			t.Errorf("instance %s was built on the compatibility engine", inst.ID())
		}
	}
}
