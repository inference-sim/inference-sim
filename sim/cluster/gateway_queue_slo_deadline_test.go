package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func newFlowControlSLODeadlineConfig(numInstances int, sloTargets map[string]int64) DeploymentConfig {
	cfg := newTestDeploymentConfig(numInstances)
	cfg.FlowControlEnabled = true
	cfg.FlowControlDetector = "never"
	cfg.FlowControlDispatchOrder = "slo-deadline"
	cfg.FlowControlSLOTargets = sloTargets
	return cfg
}

func TestSLODeadlineOrdering_Integration(t *testing.T) {
	cfg := newFlowControlSLODeadlineConfig(1, map[string]int64{
		"critical": 100_000,
		"batch":    5_000_000,
	})
	cfg.Horizon = 200_000
	reqs := []*sim.Request{
		{ID: "batch1", ArrivalTime: 0, SLOClass: "batch",
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: sim.StateQueued},
		{ID: "crit1", ArrivalTime: 100, SLOClass: "critical",
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: sim.StateQueued},
	}
	cs := NewClusterSimulator(cfg, reqs, nil)
	mustRun(t, cs)

	// With never-saturated detector, both dispatch immediately.
	// SLO-deadline: critical (tighter target) should dispatch before batch.
	// Verify: both processed without panic, gateway queue empty.
	if cs.GatewayQueueDepth() != 0 {
		t.Errorf("expected gateway queue empty, got depth=%d", cs.GatewayQueueDepth())
	}
}

func TestSLODeadlineOrdering_NoTargets_FCFS(t *testing.T) {
	cfg := newFlowControlSLODeadlineConfig(1, nil)
	cfg.Horizon = 200_000
	reqs := []*sim.Request{
		{ID: "r1", ArrivalTime: 0, SLOClass: "standard",
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: sim.StateQueued},
		{ID: "r2", ArrivalTime: 100, SLOClass: "standard",
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: sim.StateQueued},
	}
	cs := NewClusterSimulator(cfg, reqs, nil)
	mustRun(t, cs)

	// No SLO targets → degenerates to FCFS (BC-4). Both should process fine.
	if cs.GatewayQueueDepth() != 0 {
		t.Errorf("expected gateway queue empty, got depth=%d", cs.GatewayQueueDepth())
	}
}
