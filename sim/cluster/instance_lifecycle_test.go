// instance_lifecycle_test.go — BDD/TDD tests for Phase 1A instance lifecycle (US4).
package cluster

import (
	"testing"
)

// ─── US4: Instance Startup Phases Including Warm-Up ─────────────────────────

// T036: instance with loading delay is excluded from routing until InstanceLoadedEvent fires.
func TestInstanceLifecycle_LoadingExcludedFromRouting(t *testing.T) {
	inst := &InstanceSimulator{id: "test-inst"}
	inst.TransitionTo(InstanceStateLoading)

	t.Run("Loading instance is not routable", func(t *testing.T) {
		if inst.IsRoutable() {
			t.Error("Loading instance should not be routable")
		}
	})

	inst.TransitionTo(InstanceStateWarmingUp)
	t.Run("WarmingUp instance IS routable", func(t *testing.T) {
		if !inst.IsRoutable() {
			t.Error("WarmingUp instance should be routable")
		}
	})

	inst.TransitionTo(InstanceStateActive)
	t.Run("Active instance IS routable", func(t *testing.T) {
		if !inst.IsRoutable() {
			t.Error("Active instance should be routable")
		}
	})
}

// T037: warm-up TTFT factor applied to first N requests, not N+1.
func TestInstanceLifecycle_WarmUpTTFTPenalty(t *testing.T) {
	inst := &InstanceSimulator{id: "warm-inst"}
	inst.TransitionTo(InstanceStateWarmingUp)
	inst.warmUpRemaining = 3

	t.Run("first 3 requests are warming up", func(t *testing.T) {
		for i := 0; i < 3; i++ {
			if !inst.IsWarmingUp() {
				t.Errorf("request %d: IsWarmingUp() = false, want true", i)
			}
			inst.RecordWarmUpRequest("req-" + string(rune('0'+i)))
			inst.ConsumeWarmUpRequest()
		}
	})

	t.Run("4th request is no longer warming up", func(t *testing.T) {
		if inst.IsWarmingUp() {
			t.Error("after consuming all warm-up requests, IsWarmingUp() should be false")
		}
	})

	t.Run("instance transitions to Active after warm-up", func(t *testing.T) {
		if inst.State != InstanceStateActive {
			t.Errorf("instance state = %q, want Active after warm-up completion", inst.State)
		}
	})

	t.Run("WarmUpRequestIDs recorded exactly N entries", func(t *testing.T) {
		ids := inst.WarmUpRequestIDs()
		if len(ids) != 3 {
			t.Errorf("WarmUpRequestIDs() len = %d, want 3", len(ids))
		}
	})
}

// T038: WAIT drain policy excludes instance from routing, in-flight requests complete.
func TestInstanceLifecycle_WaitDrainExcludesRouting(t *testing.T) {
	inst := &InstanceSimulator{id: "drain-inst"}
	inst.TransitionTo(InstanceStateActive)

	policy := &drainWait{}
	// Call Drain with a nil ClusterSimulator (drainWait only transitions state)
	policy.Drain(inst, nil)

	t.Run("Draining instance excluded from routing", func(t *testing.T) {
		if inst.IsRoutable() {
			t.Error("Draining instance should not be routable")
		}
	})

	t.Run("instance state is Draining", func(t *testing.T) {
		if inst.State != InstanceStateDraining {
			t.Errorf("instance state = %q, want Draining", inst.State)
		}
	})
}

// T039: IMMEDIATE drain terminates instance immediately.
func TestInstanceLifecycle_ImmediateDrain(t *testing.T) {
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
