package cluster

import (
	"fmt"
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

	// Warm-up requests should have ~2× TTFT of normal request
	// Allow 10% tolerance for latency model variance
	if ttft1 < ttft3*1.8 || ttft1 > ttft3*2.2 {
		t.Errorf("req1 TTFT = %.0f, expected ~2× req3 TTFT (%.0f)", ttft1, ttft3)
	}
	if ttft2 < ttft3*1.8 || ttft2 > ttft3*2.2 {
		t.Errorf("req2 TTFT = %.0f, expected ~2× req3 TTFT (%.0f)", ttft2, ttft3)
	}
}

// ─── Drain policies ──────────────────────────────────────────────────────────

func TestInstanceLifecycle_WaitDrainExcludesRouting(t *testing.T) {
	// GIVEN an instance in Draining state with WAIT policy
	inst := &InstanceSimulator{id: "wait-inst"}
	inst.TransitionTo(InstanceStateActive)

	policy := &drainWait{}
	policy.Drain(inst, nil)

	// THEN instance is not routable
	if inst.IsRoutable() {
		t.Error("Draining instance with WAIT policy should not be routable")
	}

	// AND instance is in Draining state
	if inst.State != InstanceStateDraining {
		t.Errorf("instance state = %q, want Draining", inst.State)
	}
}

func TestInstanceLifecycle_ImmediateDrain(t *testing.T) {
	// GIVEN an Active instance
	inst := &InstanceSimulator{id: "imm-inst"}
	inst.TransitionTo(InstanceStateActive)

	// drainImmediate needs releaseInstanceGPUs which needs a cluster — use a mock cs
	// that has nil placement (no-op release)
	cs := &ClusterSimulator{placement: nil}
	policy := &drainImmediate{}
	policy.Drain(inst, cs)

	t.Run("instance terminates immediately", func(t *testing.T) {
		if inst.State != InstanceStateTerminated {
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
// policy does not double-count requests in CompletedRequests (INV-1 conservation).
// This is a regression test for the drain redirect double-counting bug identified in PR #697.
func TestInstanceLifecycle_RedirectDrainPreservesConservation(t *testing.T) {
	// GIVEN a 2-instance cluster with REDIRECT drain policy
	cfg := newTestDeploymentConfig(2)
	cfg.InstanceLifecycle = InstanceLifecycleConfig{
		DrainPolicy: string(DrainPolicyRedirect),
	}

	// Create 10 requests that will be routed round-robin to both instances
	requests := make([]*sim.Request, 10)
	for i := range requests {
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("req-%d", i),
			InputTokens:  make([]int, 10),
			OutputTokens: make([]int, 5),
			ArrivalTime:  int64(i * 100),
		}
	}

	cs := NewClusterSimulator(cfg, requests, nil)

	// Drain instance 0 after 500 ticks (after ~5 requests have been routed)
	// This will redirect queued requests from instance 0 to instance 1
	inst0 := cs.instances[0]
	drainPolicy := NewDrainPolicy(DrainPolicyRedirect)
	
	// Run simulation partway to let some requests queue
	for cs.clock < 500 && len(cs.clusterEvents) > 0 {
		entry := cs.clusterEvents[0]
		cs.clock = entry.event.Timestamp()
		if cs.clock > 500 {
			break
		}
		cs.clusterEvents = cs.clusterEvents[1:]
		entry.event.Execute(cs)
	}

	// Drain instance 0 (redirects queued requests)
	drainPolicy.Drain(inst0, cs)

	// Continue simulation to completion
	if err := cs.Run(); err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	metrics := cs.AggregatedMetrics()

	// THEN INV-1 conservation holds: injected = completed + queued + running + dropped + timed_out
	injected := len(requests)
	completed := metrics.CompletedRequests
	stillQueued := metrics.StillQueued
	stillRunning := metrics.StillRunning
	dropped := metrics.DroppedUnservable
	timedOut := metrics.TimedOutRequests

	total := completed + stillQueued + stillRunning + dropped + timedOut
	if total != injected {
		t.Errorf("INV-1 conservation violated: injected=%d, completed=%d, queued=%d, running=%d, dropped=%d, timedOut=%d (total=%d)",
			injected, completed, stillQueued, stillRunning, dropped, timedOut, total)
	}

	// AND all requests should complete (no horizon cutoff)
	if completed != injected {
		t.Errorf("expected all %d requests to complete, got %d", injected, completed)
	}
}

// ─── Instance state machine ──────────────────────────────────────────────────

func TestInstanceStateMachine_ValidTransitions(t *testing.T) {
	cases := []struct {
		name   string
		from   InstanceState
		to     InstanceState
		wantOK bool
	}{
		{"Scheduling→Loading", InstanceStateScheduling, InstanceStateLoading, true},
		{"Loading→WarmingUp", InstanceStateLoading, InstanceStateWarmingUp, true},
		{"Loading→Active", InstanceStateLoading, InstanceStateActive, true},
		{"WarmingUp→Active", InstanceStateWarmingUp, InstanceStateActive, true},
		{"Active→Draining", InstanceStateActive, InstanceStateDraining, true},
		{"Draining→Terminated", InstanceStateDraining, InstanceStateTerminated, true},
		{"Active→Loading (invalid)", InstanceStateActive, InstanceStateLoading, false},
		{"Terminated→Active (invalid)", InstanceStateTerminated, InstanceStateActive, false},
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

// ─── InstanceState enums ─────────────────────────────────────────────────────

func TestInstanceState_IsValidInstanceState(t *testing.T) {
	valid := []string{"Scheduling", "Loading", "WarmingUp", "Active", "Draining", "Terminated"}
	for _, s := range valid {
		if !IsValidInstanceState(s) {
			t.Errorf("IsValidInstanceState(%q) = false, want true", s)
		}
	}
	if IsValidInstanceState("unknown") {
		t.Error("IsValidInstanceState(unknown) = true, want false")
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
	forwardOrder := []InstanceState{
		InstanceStateScheduling,
		InstanceStateLoading,
		InstanceStateWarmingUp,
		InstanceStateActive,
		InstanceStateDraining,
		InstanceStateTerminated,
	}
	indexOf := make(map[InstanceState]int, len(forwardOrder))
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