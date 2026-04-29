package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// ─── Warm-up TTFT penalty ────────────────────────────────────────────────────

func TestInstanceLifecycle_WarmUpTTFTPenalty(t *testing.T) {
	// GIVEN an instance with WarmUpTTFTFactor=2.0 and WarmUpRequestCount=2
	cfg := newTestDeploymentConfig(1)
	cfg.InstanceLifecycle = InstanceLifecycleConfig{
		WarmUpTTFTFactor:   2.0,
		WarmUpRequestCount: 2,
	}

	// Create 3 requests with identical input/output lengths
	requests := []*sim.Request{
		{
			ID:           "req1",
			InputTokens:  make([]int, 10),
			OutputTokens: make([]int, 5),
			ArrivalTime:  0,
		},
		{
			ID:           "req2",
			InputTokens:  make([]int, 10),
			OutputTokens: make([]int, 5),
			ArrivalTime:  1000,
		},
		{
			ID:           "req3",
			InputTokens:  make([]int, 10),
			OutputTokens: make([]int, 5),
			ArrivalTime:  2000,
		},
	}

	cs := NewClusterSimulator(cfg, requests, nil)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	metrics := cs.AggregatedMetrics()

	// THEN first 2 requests have TTFT penalty applied, 3rd does not
	ttft1, ok1 := metrics.RequestTTFTs["req1"]
	ttft2, ok2 := metrics.RequestTTFTs["req2"]
	ttft3, ok3 := metrics.RequestTTFTs["req3"]

	if !ok1 || !ok2 || !ok3 {
		t.Fatalf("missing TTFT data: req1=%v req2=%v req3=%v", ok1, ok2, ok3)
	}

	// Warm-up requests should have factor applied to their baseline TTFT.
	// Since requests arrive at different times, they have different baseline TTFTs
	// due to queueing effects. We verify the factor was applied by checking that
	// warm-up requests have higher TTFT than the non-warm-up request, and that
	// the ratio is reasonable (between 1.5x and 2.5x to account for queueing variance).
	if ttft1 <= ttft3 {
		t.Errorf("req1 TTFT = %.0f should be > req3 TTFT (%.0f) due to warm-up factor", ttft1, ttft3)
	}
	if ttft2 <= ttft3 {
		t.Errorf("req2 TTFT = %.0f should be > req3 TTFT (%.0f) due to warm-up factor", ttft2, ttft3)
	}
	// Verify factor was applied (ratio should be between 1.3x and 2.5x).
	// The lower bound allows for queueing variance: the warm-up 2.0x multiplier
	// is applied to the baseline TTFT, but queueing/scheduling delays are
	// additive and independent of the multiplier, which compresses the observed
	// ratio below 2.0x. The exact compression depends on the latency model
	// backend (roofline vs trained-physics produce different baseline timing).
	ratio1 := ttft1 / ttft3
	ratio2 := ttft2 / ttft3
	if ratio1 < 1.3 || ratio1 > 2.5 {
		t.Errorf("req1/req3 ratio = %.2f, expected between 1.3 and 2.5", ratio1)
	}
	if ratio2 < 1.3 || ratio2 > 2.5 {
		t.Errorf("req2/req3 ratio = %.2f, expected between 1.3 and 2.5", ratio2)
	}
}

// ─── Drain policies ──────────────────────────────────────────────────────────

func TestInstanceLifecycle_WaitDrainExcludesRouting(t *testing.T) {
	// GIVEN an instance in Draining state with WAIT policy
	inst := &InstanceSimulator{id: "wait-inst"}
	inst.TransitionTo(sim.InstanceStateActive)

	policy := &drainWait{}
	policy.Drain(inst, nil)

	// THEN instance is not routable
	if inst.IsRoutable() {
		t.Error("Draining instance with WAIT policy should not be routable")
	}

	// AND instance is in Draining state
	if inst.State != sim.InstanceStateDraining {
		t.Errorf("instance state = %q, want Draining", inst.State)
	}
}

func TestInstanceLifecycle_ImmediateDrain(t *testing.T) {
	// GIVEN an Active instance
	inst := &InstanceSimulator{id: "imm-inst"}
	inst.TransitionTo(sim.InstanceStateActive)

	// drainImmediate needs releaseInstanceGPUs which needs a cluster — use a mock cs
	// that has nil placement (no-op release)
	cs := &ClusterSimulator{placement: nil}
	policy := &drainImmediate{}
	policy.Drain(inst, cs)

	t.Run("instance terminates immediately", func(t *testing.T) {
		if inst.State != sim.InstanceStateTerminated {
			t.Errorf("instance state = %q, want Terminated after IMMEDIATE drain", inst.State)
		}
	})

	t.Run("instance not routable after IMMEDIATE drain", func(t *testing.T) {
		if inst.IsRoutable() {
			t.Error("Terminated instance should not be routable")
		}
	})
}

// TestInstanceLifecycle_RedirectDrainPreservesConservation verifies that DrainRedirect
// policy preserves INV-1 (request conservation) when requests are actually in the
// source WaitQ at drain time.
//
// The previous version of this test was a no-op: it relied on a manual event loop that
// never ran (clusterEvents is empty before Run()), so DrainWaitQueue returned [] and
// no redirection occurred. This version seeds requests directly into inst0's WaitQ
// using InjectRequestOnline — the same path RoutingDecisionEvent uses — so the drain
// genuinely redirects work.
func TestInstanceLifecycle_RedirectDrainPreservesConservation(t *testing.T) {
	// GIVEN a 2-instance cluster with no workload (empty request list — we seed manually)
	cfg := newTestDeploymentConfig(2)
	cfg.InstanceLifecycle = InstanceLifecycleConfig{
		DrainPolicy: string(DrainPolicyRedirect),
	}

	// Build 3 requests using the shared helper so they have valid token IDs.
	// Pass an empty workload to the cluster so Run() does not push duplicate
	// ClusterArrivalEvents for these requests (we seed them directly into WaitQ).
	const numSeeded = 3
	seeded := newTestRequests(numSeeded)

	cs := NewClusterSimulator(cfg, []*sim.Request{}, nil)
	inst0 := cs.instances[0]

	// Seed requests directly into inst0's WaitQ (bypassing admission/routing),
	// mirroring what RoutingDecisionEvent does, so they are present at drain time.
	// EnqueueRequest puts the request in WaitQ immediately (unlike InjectArrivalAt
	// which schedules an ArrivalEvent that would only fire during Run()).
	for _, req := range seeded {
		inst0.sim.EnqueueRequest(req)
		cs.inFlightRequests[string(inst0.ID())]++
	}

	// Precondition: inst0 must have work to redirect.
	if inst0.QueueDepth() == 0 {
		t.Fatal("precondition failed: expected seeded requests in inst0 WaitQ before drain")
	}

	// WHEN drain with REDIRECT fires
	NewDrainPolicy(DrainPolicyRedirect).Drain(inst0, cs)

	// THEN inst0 WaitQ is empty and inFlightRequests decremented (C1 fix)
	if inst0.QueueDepth() != 0 {
		t.Errorf("expected inst0 WaitQ empty after redirect drain, got %d", inst0.QueueDepth())
	}
	if got := cs.inFlightRequests[string(inst0.ID())]; got != 0 {
		t.Errorf("C1: inFlightRequests[inst0] = %d after redirect drain, want 0", got)
	}
	// AND the redirected requests are now pending as ClusterArrivalEvents
	if len(cs.clusterEvents) == 0 {
		t.Error("expected redirected requests in clusterEvents after drain")
	}

	// Run to completion (processes the redirected ClusterArrivalEvents via inst1)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	// INV-1: injected = completed + queued + running + dropped + timed_out
	m := cs.AggregatedMetrics()
	injected := numSeeded
	total := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable + m.TimedOutRequests
	if total != injected {
		t.Errorf("INV-1 violated: injected=%d total=%d (completed=%d queued=%d running=%d dropped=%d timedOut=%d)",
			injected, total, m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable, m.TimedOutRequests)
	}
	if m.CompletedRequests != injected {
		t.Errorf("expected all %d redirected requests to complete, got %d", injected, m.CompletedRequests)
	}
}

// ─── Instance state machine ──────────────────────────────────────────────────

func TestInstanceStateMachine_ValidTransitions(t *testing.T) {
	cases := []struct {
		name   string
		from   sim.InstanceState
		to     sim.InstanceState
		wantOK bool
	}{
		{"Scheduling→Loading", sim.InstanceStateScheduling, sim.InstanceStateLoading, true},
		{"Loading→WarmingUp", sim.InstanceStateLoading, sim.InstanceStateWarmingUp, true},
		{"Loading→Active", sim.InstanceStateLoading, sim.InstanceStateActive, true},
		{"WarmingUp→Active", sim.InstanceStateWarmingUp, sim.InstanceStateActive, true},
		{"Active→Draining", sim.InstanceStateActive, sim.InstanceStateDraining, true},
		{"Draining→Terminated", sim.InstanceStateDraining, sim.InstanceStateTerminated, true},
		{"Active→Loading (invalid)", sim.InstanceStateActive, sim.InstanceStateLoading, false},
		{"Terminated→Active (invalid)", sim.InstanceStateTerminated, sim.InstanceStateActive, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			inst := &InstanceSimulator{id: "sm-test"}
			inst.State = tc.from

			defer func() {
				r := recover()
				if tc.wantOK && r != nil {
					t.Errorf("TransitionTo panicked unexpectedly: %v", r)
				}
				if !tc.wantOK && r == nil {
					t.Errorf("TransitionTo should have panicked for invalid transition %s→%s", tc.from, tc.to)
				}
			}()
			inst.TransitionTo(tc.to)
		})
	}
}

// ─── sim.InstanceState enums ─────────────────────────────────────────────────────

func TestInstanceState_IsValidInstanceState(t *testing.T) {
	valid := []string{"Scheduling", "Loading", "WarmingUp", "Active", "Draining", "Terminated"}
	for _, s := range valid {
		if !sim.IsValidInstanceState(s) {
			t.Errorf("sim.IsValidInstanceState(%q) = false, want true", s)
		}
	}
	if sim.IsValidInstanceState("unknown") {
		t.Error("sim.IsValidInstanceState(unknown) = true, want false")
	}
}

func TestNodeState_IsValidNodeState(t *testing.T) {
	valid := []string{"Provisioning", "Ready", "Draining", "Terminated"}
	for _, s := range valid {
		if !IsValidNodeState(s) {
			t.Errorf("IsValidNodeState(%q) = false, want true", s)
		}
	}
	if IsValidNodeState("unknown") {
		t.Error("IsValidNodeState(unknown) = true, want false")
	}
}

// ─── State transition monotonicity invariant ─────────────────────────────────

// TestInstanceStateMachine_NoBackwardTransitions verifies the monotonicity law:
// instance lifecycle states advance strictly forward and never regress.
// This is a system-law test (R7 companion to all lifecycle golden tests).
func TestInstanceStateMachine_NoBackwardTransitions(t *testing.T) {
	// Define the forward order — each state can only transition to a higher-index state.
	forwardOrder := []sim.InstanceState{
		sim.InstanceStateScheduling,
		sim.InstanceStateLoading,
		sim.InstanceStateWarmingUp,
		sim.InstanceStateActive,
		sim.InstanceStateDraining,
		sim.InstanceStateTerminated,
	}
	indexOf := make(map[sim.InstanceState]int, len(forwardOrder))
	for i, s := range forwardOrder {
		indexOf[s] = i
	}

	// For every valid transition, verify the target index >= source index (monotone).
	for src, targets := range validInstanceTransitions {
		srcIdx, ok := indexOf[src]
		if !ok {
			continue
		}
		for tgt := range targets {
			tgtIdx, ok2 := indexOf[tgt]
			if !ok2 {
				continue
			}
			if tgtIdx <= srcIdx {
				t.Errorf("backward transition allowed: %s (idx=%d) → %s (idx=%d) — violates lifecycle monotonicity law",
					src, srcIdx, tgt, tgtIdx)
			}
		}
	}
}

// ─── InstanceLifecycleConfig validation ─────────────────────────────────────

func TestInstanceLifecycleConfig_Validation(t *testing.T) {
	cases := []struct {
		name    string
		cfg     InstanceLifecycleConfig
		wantErr bool
	}{
		{"zero value is valid", InstanceLifecycleConfig{}, false},
		{"valid warm-up factor", InstanceLifecycleConfig{WarmUpTTFTFactor: 2.0, WarmUpRequestCount: 5}, false},
		{"factor < 1.0 invalid", InstanceLifecycleConfig{WarmUpTTFTFactor: 0.5}, true},
		{"negative warm-up count", InstanceLifecycleConfig{WarmUpRequestCount: -1}, true},
		{"valid REDIRECT policy", InstanceLifecycleConfig{DrainPolicy: "REDIRECT"}, false},
		{"invalid drain policy", InstanceLifecycleConfig{DrainPolicy: "BOGUS"}, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.IsValid()
			if (err != nil) != tc.wantErr {
				t.Errorf("IsValid() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}