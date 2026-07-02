package cluster

import (
	"testing"

	"blis/sim"
)

func newFlowControlTTLConfig(numInstances int, ttlUs int64, detector string) DeploymentConfig {
	cfg := newTestDeploymentConfig(numInstances)
	cfg.FlowControlEnabled = true
	cfg.FlowControlRequestTTL = ttlUs
	cfg.FlowControlDetector = detector
	cfg.FlowControlDispatchOrder = "fifo"
	if detector == "concurrency" {
		cfg.FlowControlMaxConcurrency = 1
	}
	return cfg
}

func TestGatewayQueueTTL_ExpiresQueuedRequest(t *testing.T) {
	cfg := newFlowControlTTLConfig(1, 5000, "concurrency")
	cfg.FlowControlMaxConcurrency = 1
	cfg.Horizon = 100_000
	reqs := []*sim.Request{
		{ID: "r1", ArrivalTime: 0, SLOClass: "standard",
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: sim.StateQueued},
		{ID: "r2", ArrivalTime: 100, SLOClass: "batch",
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: sim.StateQueued},
	}
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	if cs.GatewayExpired() != 1 {
		t.Fatalf("expected gatewayExpired=1, got %d", cs.GatewayExpired())
	}
	shed := cs.ShedByTier()
	if shed["batch"] != 1 {
		t.Fatalf("expected shedByTier[batch]=1, got %d", shed["batch"])
	}
}

func TestGatewayQueueTTL_NoOpWhenDispatched(t *testing.T) {
	cfg := newFlowControlTTLConfig(1, 5000, "never")
	cfg.Horizon = 100_000
	reqs := newTestRequests(3)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	if cs.GatewayExpired() != 0 {
		t.Fatalf("expected gatewayExpired=0 (all dispatched before TTL), got %d", cs.GatewayExpired())
	}
}

func TestGatewayQueueTTL_DisabledByDefault(t *testing.T) {
	cfg := newFlowControlTTLConfig(1, 0, "never")
	cfg.Horizon = 100_000
	reqs := newTestRequests(5)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	if cs.GatewayExpired() != 0 {
		t.Fatalf("expected gatewayExpired=0 (TTL disabled), got %d", cs.GatewayExpired())
	}
}
