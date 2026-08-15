package cluster

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Tests for the periodic-trigger event's inertness law and interface shape. B-7 (#1495)
// wrote these against an unconditionally-inert scaffold (INV-PS3); Spec 3 activated the
// tick, so they now pin INV-PS3' — the CONDITIONAL law: a set interval is inert exactly
// when the effective creation policy cannot act on a tick. The complement, that a tick
// policy does change the run, lives in lora_periodic_test.go.

// TestPeriodicInterval_ByteIdenticalToUnset is the INV-PS3' inertness law: a positive
// LoRAPeriodicIntervalUs produces aggregated metrics byte-identical to 0 WHEN no tick can
// fire. It runs BOTH ways that can be true, because they are separate branches of
// newLoRAPeriodicPipeline and a test of one says nothing about the other:
//
//   - "lora-inactive" — no adapters declared at all. This is B-7's original case and
//     covers every pre-LoRA leaf.
//   - "gate-only-policy" — LoRA fully active, but the effective creation policy does not
//     implement sim.PeriodicCreationPolicy. This is the actual statement of INV-PS3' and
//     covers every pre-Spec-3 LoRA leaf, all of which run the on-demand default.
//
// The complement — that a TICK policy DOES change the run — is
// TestPeriodicTick_NotInertUnderKeepWarm. Both are required: this one alone cannot
// distinguish "correctly inert" from "never wired".
func TestPeriodicInterval_ByteIdenticalToUnset(t *testing.T) {
	cases := []struct {
		name string
		// build returns a config with the interval unset; the test sets it per run.
		build func() DeploymentConfig
		reqs  func() []*sim.Request
		// premise asserts that this case reaches the branch of newLoRAPeriodicPipeline it
		// is named for. Each case asserts the OTHER's negation, so the two can never
		// silently converge on the same branch and quietly become duplicates.
		premise func(*testing.T, DeploymentConfig)
	}{
		{
			name:  "lora-inactive",
			build: func() DeploymentConfig { return newTestDeploymentConfig(2) },
			reqs:  func() []*sim.Request { return newTestRequests(50) },
			premise: func(t *testing.T, cfg DeploymentConfig) {
				t.Helper()
				if cfg.HasAdapters() || cfg.AdapterCapacity != nil {
					t.Fatalf("premise broken: the lora-inactive case must declare NO adapters and NO capacity so it exercises the subsystem-inactive branch, got HasAdapters=%v AdapterCapacity=%v", cfg.HasAdapters(), cfg.AdapterCapacity)
				}
			},
		},
		{
			name: "gate-only-policy",
			// LoRA fully active — adapters declared, capacity set, adapter-carrying
			// requests — so cold loads and evictions genuinely happen. Only the creation
			// policy's lack of a tick method makes the interval inert here.
			build: func() DeploymentConfig {
				return loraAffinityTestConfig(2, 2, periodicTestAdapters)
			},
			reqs: func() []*sim.Request { return zipfianAdapterRequests(50, periodicTestAdapters) },
			premise: func(t *testing.T, cfg DeploymentConfig) {
				t.Helper()
				// THIS is the property whose absence let an earlier version of this test
				// pass while the gate-only check was broken: with LoRA inactive,
				// newLoRAPeriodicPipeline returns nil at the subsystem branch and never
				// reaches the sim.PeriodicCreationPolicy type assertion that this case
				// exists to exercise, so the subtest silently degrades into a duplicate of
				// lora-inactive while keeping its name and staying green.
				//
				// The risk is live, not hypothetical: loraAffinityTestConfig is a shared
				// helper owned by lora_affinity_routing_e2e_test.go, so an edit there could
				// drop Adapters or AdapterCapacity without anyone touching this file.
				if !cfg.HasAdapters() || cfg.AdapterCapacity == nil {
					t.Fatalf("premise broken: the gate-only-policy case must be LoRA-ACTIVE so newLoRAPeriodicPipeline reaches the PeriodicCreationPolicy type assertion, got HasAdapters=%v AdapterCapacity=%v", cfg.HasAdapters(), cfg.AdapterCapacity)
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfgUnset := tc.build()
			cfgUnset.LoRAPeriodicIntervalUs = 0
			csUnset := NewClusterSimulator(cfgUnset, NewSliceRequestSource(tc.reqs()), nil)
			mustRun(t, csUnset)

			cfgSet := tc.build()
			cfgSet.LoRAPeriodicIntervalUs = 1_000_000 // 1s — inert when no tick can fire
			// Assert this case reaches the branch it is named for, before relying on the
			// outcome below.
			tc.premise(t, cfgSet)
			// Assert the creation-policy premise too, so this cannot pass for the wrong reason: an
			// empty CreationPolicy is the gate-only on-demand default (both NewSimulator
			// and newLoRAPeriodicPipeline resolve ""→on-demand). If a future default
			// became a tick policy, byte-identity would be the WRONG expectation and this
			// test would be silently asserting that the tick is broken.
			if cfgSet.CreationPolicy != "" {
				t.Fatalf("premise broken: this is INV-PS3''s no-tick case, but CreationPolicy = %q; a tick-capable policy must be tested by TestPeriodicTick_NotInertUnderKeepWarm instead", cfgSet.CreationPolicy)
			}
			csSet := NewClusterSimulator(cfgSet, NewSliceRequestSource(tc.reqs()), nil)
			mustRun(t, csSet)
			if csSet.loraPeriodic != nil {
				t.Error("a pipeline was constructed even though no tick can fire; INV-PS3' must hold structurally (no pipeline, no tick), not by a runtime check")
			}

			mUnset, err1 := json.Marshal(csUnset.AggregatedMetrics())
			mSet, err2 := json.Marshal(csSet.AggregatedMetrics())
			if err1 != nil || err2 != nil {
				t.Fatalf("json marshal error: %v / %v", err1, err2)
			}
			if !bytes.Equal(mUnset, mSet) {
				t.Errorf("LoRAPeriodicIntervalUs is not inert (INV-PS3' violated):\n unset=%s\n   set=%s", mUnset, mSet)
			}
		})
	}
}

// TestPeriodicTriggerEvent_AccessorsAndInterface pins the event's ClusterEvent shape
// (Timestamp/Priority/Execute) so the wiring point stays type-correct.
func TestPeriodicTriggerEvent_AccessorsAndInterface(t *testing.T) {
	var ev ClusterEvent = &LoRAPeriodicTriggerEvent{At: 12345}
	if got := ev.Timestamp(); got != 12345 {
		t.Errorf("Timestamp() = %d, want 12345", got)
	}
	if got := ev.Priority(); got != 8 {
		t.Errorf("Priority() = %d, want 8 (ScalingTickEvent band)", got)
	}
	// Execute is no longer a no-op scaffold — it dereferences the cluster to reach the
	// pipeline. The contract for a nil cluster is warned-and-dropped, not a panic, which
	// the nil guard provides; that is what this call asserts.
	ev.Execute(nil)
}
