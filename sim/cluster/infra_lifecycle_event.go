// infra_lifecycle_event.go defines the DES events for instance lifecycle transitions.
// Phase 1A. The node events and the drain policies, which no run schedules or calls, are in
// node_events_and_drain_test.go (design LIF-7).
package cluster

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// Event priority constants for lifecycle events.
// Lifecycle events process before request events (Arrival=0, Admission=1, Routing=2)
// at the same timestamp to ensure infrastructure state updates complete first (I6).
const (
	priorityNodeLifecycle     = -2 // Node provisioning/draining events (the test-only node events)
	priorityInstanceLifecycle = -1 // Instance loading/warm-up events
)

// ─── Instance lifecycle events ──────────────────────────────────────────────

// InstanceLoadedEvent fires when an instance finishes loading model weights.
// Priority -1: instance lifecycle after node events (-2) but before request events (0+) (I6).
type InstanceLoadedEvent struct {
	timestamp  int64
	instanceID InstanceID
}

func (e *InstanceLoadedEvent) Timestamp() int64 { return e.timestamp }
func (e *InstanceLoadedEvent) Priority() int    { return priorityInstanceLifecycle }

// Execute transitions the instance Loading → WarmingUp (or Active if no warm-up configured).
func (e *InstanceLoadedEvent) Execute(cs *ClusterSimulator) {
	if inst := cs.instanceByID(e.instanceID); inst != nil {
		warmUpCount := cs.config.InstanceLifecycle.WarmUpRequestCount
		if warmUpCount <= 0 {
			// Skip WarmingUp phase entirely — go directly to Active
			inst.TransitionTo(sim.InstanceStateActive)
		} else {
			inst.TransitionTo(sim.InstanceStateWarmingUp)
		}
		return
	}
	// R1: log when target instance not found (should not happen unless instance was removed)
	logrus.Warnf("[cluster] InstanceLoadedEvent: instance %s not found — event dropped", e.instanceID)
}

// ─── scheduleInstanceLoadedEvent ────────────────────────────────────────────

// scheduleInstanceLoadedEvent schedules an InstanceLoadedEvent for inst based on
// the configured loading delay. If the delay is zero, transitions immediately to
// Loading → WarmingUp (or Active).
func (cs *ClusterSimulator) scheduleInstanceLoadedEvent(inst *InstanceSimulator) {
	delay := cs.placement.SampleLoadingDelay(&cs.config.InstanceLifecycle)
	readyTime := cs.clock + delay

	if delay == 0 {
		// I7 (deliberate optimization): No delay — transition inline instead of scheduling
		// an event at the same timestamp. Avoids unnecessary heap push/pop overhead.
		warmUpCount := cs.config.InstanceLifecycle.WarmUpRequestCount
		if warmUpCount <= 0 {
			inst.TransitionTo(sim.InstanceStateActive)
		} else {
			inst.TransitionTo(sim.InstanceStateWarmingUp)
		}
		return
	}

	cs.schedule(&InstanceLoadedEvent{timestamp: readyTime, instanceID: inst.ID()}, &cs.lifecycleSender)
}

// ─── Helper: releaseInstanceGPUs ────────────────────────────────────────────

// releaseInstanceGPUs releases the GPU allocations for a terminated instance.
// Logs a warning if the release fails (instance was not placed — expected for Scheduling state).
func (cs *ClusterSimulator) releaseInstanceGPUs(inst *InstanceSimulator) {
	if cs.placement == nil {
		return
	}
	if err := cs.placement.ReleaseInstance(inst.ID()); err != nil {
		// Instance may never have been placed (still in Scheduling state) — not a bug.
		// Log at debug level so placement bugs surface during investigation.
		logrus.Debugf("[cluster] releaseInstanceGPUs %s: %v", inst.ID(), err)
	}
}
