package cluster

import (
	"bytes"
	"encoding/json"
	"testing"
)

// B-7 (#1495) tests for the inert periodic-trigger scaffold (D5, INV-PS3). The
// LoRAPeriodicTriggerEvent exists so a follow-up PR can wire periodic LoRA-seam
// re-resolution without reshaping the event model, but this round NewClusterSimulator
// NEVER enqueues it — so a set interval must be byte-identical to an unset one.

// TestPeriodicInterval_ByteIdenticalToUnset is the authoritative inertness law
// (INV-PS3): a positive LoRAPeriodicIntervalUs produces aggregated metrics that are
// byte-identical to LoRAPeriodicIntervalUs=0. If a future change accidentally
// schedules the trigger (or any observable side effect leaks from the field), the two
// JSON blobs diverge and this fails.
func TestPeriodicInterval_ByteIdenticalToUnset(t *testing.T) {
	reqsUnset := newTestRequests(50)
	reqsSet := newTestRequests(50)

	cfgUnset := newTestDeploymentConfig(2)
	cfgUnset.LoRAPeriodicIntervalUs = 0
	csUnset := NewClusterSimulator(cfgUnset, NewSliceRequestSource(reqsUnset), nil)
	mustRun(t, csUnset)

	cfgSet := newTestDeploymentConfig(2)
	cfgSet.LoRAPeriodicIntervalUs = 1_000_000 // 1s — must remain inert this round
	csSet := NewClusterSimulator(cfgSet, NewSliceRequestSource(reqsSet), nil)
	mustRun(t, csSet)

	mUnset, err1 := json.Marshal(csUnset.AggregatedMetrics())
	mSet, err2 := json.Marshal(csSet.AggregatedMetrics())
	if err1 != nil || err2 != nil {
		t.Fatalf("json marshal error: %v / %v", err1, err2)
	}
	if !bytes.Equal(mUnset, mSet) {
		t.Errorf("LoRAPeriodicIntervalUs is not inert (INV-PS3 violated):\n unset=%s\n   set=%s", mUnset, mSet)
	}
}

// TestPeriodicTriggerEvent_AccessorsAndInterface pins the scaffold's ClusterEvent
// shape (Timestamp/Priority/Execute) so the reserved wiring point stays type-correct.
// Execute is a documented no-op that must not panic when invoked directly.
func TestPeriodicTriggerEvent_AccessorsAndInterface(t *testing.T) {
	var ev ClusterEvent = &LoRAPeriodicTriggerEvent{At: 12345}
	if got := ev.Timestamp(); got != 12345 {
		t.Errorf("Timestamp() = %d, want 12345", got)
	}
	if got := ev.Priority(); got != 8 {
		t.Errorf("Priority() = %d, want 8 (ScalingTickEvent band)", got)
	}
	// No-op Execute must be safe to call (scaffold contract): it neither panics nor
	// mutates the simulator. A nil receiver-arg is sufficient since the body is empty.
	ev.Execute(nil)
}
