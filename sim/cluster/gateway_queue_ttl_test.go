package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
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
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5), State: sim.StateQueued},
		{ID: "r2", ArrivalTime: 100, SLOClass: "batch",
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5), State: sim.StateQueued},
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

// TestINV1_Conservation_GatewayTTLExpiry asserts INV-1 on the one path no other
// conservation test exercises: a request removed from the gateway queue by TTL
// expiry. Gateway TTL is off by default, so gateway_expired is zero in every other
// fixture — which is why cluster-level assertions could omit the bucket entirely
// and still pass (issue #1720).
func TestINV1_Conservation_GatewayTTLExpiry(t *testing.T) {
	cfg := newFlowControlTTLConfig(1, 5000, "concurrency")
	cfg.Horizon = 100_000
	reqs := []*sim.Request{
		{ID: "r1", ArrivalTime: 0, SLOClass: "standard",
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5), State: sim.StateQueued},
		{ID: "r2", ArrivalTime: 100, SLOClass: "batch",
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5), State: sim.StateQueued},
	}
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	// Non-vacuity gate: without a real expiry this asserts nothing the other
	// conservation tests do not already cover.
	if cs.GatewayExpired() == 0 {
		t.Fatal("no request expired — fixture no longer exercises the TTL path, so the assertion below is vacuous")
	}

	assertClusterINV1Conservation(t, cs, len(reqs), cs.RejectedRequests(), "gateway TTL expiry")
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
