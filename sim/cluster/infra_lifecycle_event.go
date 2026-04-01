// infra_lifecycle_event.go defines DES events and drain policies for node and
// instance lifecycle transitions. Phase 1A.
package cluster

import (
	"container/heap"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// Event priority constants for lifecycle events.
// Lifecycle events process before request events (Arrival=0, Admission=1, Routing=2)
// at the same timestamp to ensure infrastructure state updates complete first (I6).
const (
	priorityNodeLifecycle     = -2 // Node provisioning/draining events
	priorityInstanceLifecycle = -1 // Instance loading/warm-up events
)

// ─── Node lifecycle events ──────────────────────────────────────────────────

// NodeReadyEvent fires when a provisioning delay elapses and a node becomes Ready.
type NodeReadyEvent struct {
	timestamp int64
	nodeID    string
}

func (e *NodeReadyEvent) Timestamp() int64 { return e.timestamp }
func (e *NodeReadyEvent) Priority() int    { return priorityNodeLifecycle }

// Execute transitions the node Provisioning → Ready and retries any pending instances.
// Deferred construction: pending instances have no InstanceSimulator yet — construct them
// here using the pool's GPU type (SC-003: pool-authoritative, not CLI flag).
func (e *NodeReadyEvent) Execute(cs *ClusterSimulator) {
	if cs.placement == nil {
		return
	}
	if err := cs.placement.MarkNodeReady(e.nodeID); err != nil {
		// Node may have been terminated before becoming ready — not a fatal error
		return
	}

	// Retry pending instances that may now fit on the newly-ready node.
	placed := cs.placement.RetryPendingInstances()
	for idx := range placed {
		p := &placed[idx]
		// Deferred construction: set pool's GPU type (authoritative per SC-003).
		// See cluster.go construction loop comment re: HWConfig limitation for roofline backends.
		p.simCfg.GPU = p.gpuType
		inst := NewInstanceSimulator(p.id, p.simCfg)
		inst.Model = cs.config.Model
		inst.nodeID = p.nodeID
		inst.allocatedGPUIDs = p.gpuIDs
		inst.warmUpRemaining = cs.config.InstanceLifecycle.WarmUpRequestCount
		inst.TransitionTo(InstanceStateLoading)

		// Register with snapshot provider for routing (deferred instances were not
		// in the initial instanceMap passed to NewCachedSnapshotProvider).
		// Must happen BEFORE scheduleInstanceLoadedEvent so the instance is routable
		// when the load event fires.
		csp, ok := cs.snapshotProvider.(*CachedSnapshotProvider)
		if !ok {
			// snapshotProvider is nil or not *CachedSnapshotProvider — release GPUs and skip.
			// R1: no silent data loss; this can occur in unit tests that bypass NewClusterSimulator.
			// Release GPUs so they are not held by an instance that can never be routed to.
			logrus.Warnf("[cluster] NodeReadyEvent: snapshotProvider is not *CachedSnapshotProvider for instance %s — releasing GPUs and skipping", p.id)
			cs.releaseInstanceGPUs(inst)
			continue
		}
		csp.AddInstance(p.id, inst)

		cs.scheduleInstanceLoadedEvent(inst)
		cs.instances = append(cs.instances, inst)
		cs.inFlightRequests[string(p.id)] = 0

		// Register with cacheQueryFn for precise prefix scoring (deferred instances).
		if cs.cacheQueryFn != nil {
			inst := inst // capture for closure
			cs.cacheQueryFn[string(p.id)] = func(tokens []int) int {
				return inst.GetCachedBlockCount(tokens)
			}
		}

		// Wire OnRequestDone callback — mirror startup path in NewClusterSimulator (R4).
		onRequestDone := cs.sessionCallback
		if onRequestDone != nil || cs.tenantTracker != nil {
			inst.sim.OnRequestDone = func(req *sim.Request, tick int64) []*sim.Request {
				if cs.tenantTracker != nil {
					cs.tenantTracker.OnComplete(req.TenantID)
				}
				if onRequestDone == nil {
					return nil
				}
				nextReqs := onRequestDone(req, tick)
				for _, next := range nextReqs {
					heap.Push(&cs.clusterEvents, clusterEventEntry{
						event: &ClusterArrivalEvent{time: next.ArrivalTime, request: next},
						seqID: cs.nextSeqID(),
					})
				}
				return nil // don't inject locally — route through cluster pipeline
			}
		}
	}
}

// NodeDrainedEvent fires when a draining node has no more allocated instances.
// Priority -2: lifecycle events process before request events (I6).
type NodeDrainedEvent struct {
	timestamp int64
	nodeID    string
}

func (e *NodeDrainedEvent) Timestamp() int64 { return e.timestamp }
func (e *NodeDrainedEvent) Priority() int    { return priorityNodeLifecycle }

// Execute transitions the node Draining → Terminated and releases GPU inventory.
func (e *NodeDrainedEvent) Execute(cs *ClusterSimulator) {
	if cs.placement == nil {
		return
	}
	if err := cs.placement.MarkNodeTerminated(e.nodeID); err != nil {
		// R1: log rather than silently discard — aids debugging placement issues
		logrus.Debugf("[cluster] NodeDrainedEvent: MarkNodeTerminated(%s): %v", e.nodeID, err)
	}
}

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
	for _, inst := range cs.instances {
		if inst.ID() == e.instanceID {
			warmUpCount := cs.config.InstanceLifecycle.WarmUpRequestCount
			if warmUpCount <= 0 {
				// Skip WarmingUp phase entirely — go directly to Active
				inst.TransitionTo(InstanceStateActive)
			} else {
				inst.TransitionTo(InstanceStateWarmingUp)
			}
			return
		}
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
			inst.TransitionTo(InstanceStateActive)
		} else {
			inst.TransitionTo(InstanceStateWarmingUp)
		}
		return
	}

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &InstanceLoadedEvent{
			timestamp:  readyTime,
			instanceID: inst.ID(),
		},
		seqID: cs.nextSeqID(),
	})
}

// ─── DrainPolicy interface and implementations ──────────────────────────────

// DrainPolicy defines the behavior when an instance is drained. (R13: ≥2 implementations)
//
// Phase 1A Note: DrainPolicy implementations are infrastructure for Phase 1C (autoscaler).
// They are tested in instance_lifecycle_test.go but not yet wired into the simulation event
// loop. The autoscaler (Phase 1C) will call these policies when scaling down instances.
//
// Extension recipe — adding a new drain policy:
//  1. Add a new DrainPolicyName constant in infra_config.go (e.g. DrainPolicyGraceful).
//  2. Add it to the validDrainPolicies map in infra_config.go.
//  3. Implement the DrainPolicy interface (this file) as a new unexported struct.
//  4. Add a case to NewDrainPolicy() below.
//  5. Add a test in instance_lifecycle_test.go.
type DrainPolicy interface {
	// Drain initiates drain on the given instance within the cluster simulation.
	Drain(inst *InstanceSimulator, cs *ClusterSimulator)
}

// NewDrainPolicy returns the DrainPolicy implementation for the named policy.
// Panics on unknown policy name (constructor invariant per Principle V).
func NewDrainPolicy(name DrainPolicyName) DrainPolicy {
	switch name {
	case DrainPolicyImmediate:
		return &drainImmediate{}
	case DrainPolicyWait:
		return &drainWait{}
	case DrainPolicyRedirect:
		return &drainRedirect{}
	default:
		panic("NewDrainPolicy: unknown policy " + string(name))
	}
}

// drainImmediate terminates the instance immediately.
// In-flight requests receive no further steps; they complete with whatever progress
// they have made. New requests are no longer routed to the instance.
type drainImmediate struct{}

func (d *drainImmediate) Drain(inst *InstanceSimulator, cs *ClusterSimulator) {
	inst.TransitionTo(InstanceStateDraining)

	// C1/I2: Decrement inFlightRequests for all requests that will never complete —
	// both queued (WaitQ) and in-flight (running batch). Drain the WaitQ so those
	// requests don't continue processing on the terminated instance.
	if cs != nil && inst.HasSim() {
		instID := string(inst.ID())
		abandoned := inst.QueueDepth() + inst.BatchSize()
		_ = inst.DrainWaitQueue() // discard queued requests; they won't be re-routed
		if abandoned > 0 {
			cs.inFlightRequests[instID] -= abandoned
			if cs.inFlightRequests[instID] < 0 {
				cs.inFlightRequests[instID] = 0
			}
		}
	}

	inst.TransitionTo(InstanceStateTerminated)
	if cs != nil {
		cs.releaseInstanceGPUs(inst)
	}
}

// drainWait stops routing new requests to the instance and waits for in-flight
// requests to complete before transitioning to Terminated.
type drainWait struct{}

func (d *drainWait) Drain(inst *InstanceSimulator, cs *ClusterSimulator) {
	inst.TransitionTo(InstanceStateDraining)
	// Routing exclusion handled by buildRouterState() via inst.IsRoutable().

	// I3: If the instance is already idle (no queued or running requests), transition
	// immediately to Terminated and release GPUs. Otherwise the drain completion check
	// in the main event loop (cluster.go:264, T042 marker) handles the transition when
	// work finishes. This ensures GPUs are always released (INV-4 conservation).
	if cs != nil && inst.HasSim() && inst.QueueDepth() == 0 && inst.BatchSize() == 0 {
		inst.TransitionTo(InstanceStateTerminated)
		cs.releaseInstanceGPUs(inst)
	}
}

// drainRedirect stops routing new requests and re-injects all queued (not yet
// scheduled) requests as new ClusterArrivalEvents so they can be routed elsewhere.
type drainRedirect struct{}

func (d *drainRedirect) Drain(inst *InstanceSimulator, cs *ClusterSimulator) {
	inst.TransitionTo(InstanceStateDraining)

	// Extract queued requests from the instance WaitQ and re-inject into the cluster.
	// Simulation simplification: re-injected at current clock (cs.clock), not original ArrivalTime.
	// The request's ArrivalTime field is preserved, so E2E latency correctly accounts for the
	// full wait time including the pre-drain queue time. INV-5 (arrival ≤ enqueue ≤ schedule ≤
	// completion) is preserved because the new ClusterArrivalEvent fires at cs.clock ≥ ArrivalTime.
	queued := inst.DrainWaitQueue()

	// C1 (INV-1 conservation): Decrement inFlightRequests for re-injected requests.
	// These requests are leaving this instance and will get fresh inFlight counts
	// when re-routed through the cluster pipeline.
	instID := string(inst.ID())
	if len(queued) > 0 {
		cs.inFlightRequests[instID] -= len(queued)
		if cs.inFlightRequests[instID] < 0 {
			cs.inFlightRequests[instID] = 0
		}
	}

	// I10 (intentional): Re-injected requests go through admission again.
	// This is deliberate — re-validates capacity after the drain event rather than
	// assuming the cluster can absorb the redirected load.
	for _, req := range queued {
		// Remove from source Metrics.Requests so aggregateMetrics() does not log
		// a false-alarm "duplicate request ID" warning. The destination re-registers
		// the entry when the request is enqueued there.
		if inst.sim != nil {
			delete(inst.sim.Metrics.Requests, req.ID)
		}
		req.Redirected = true
		heap.Push(&cs.clusterEvents, clusterEventEntry{
			event: &ClusterArrivalEvent{time: cs.clock, request: req},
			seqID: cs.nextSeqID(),
		})
	}

	// I5 (known edge case): Arrival events already in the instance event queue for
	// previously-routed requests may still enqueue into this draining instance. The
	// drain completion check (QueueDepth==0 && BatchSize==0) in the main event loop
	// will eventually transition the instance to Terminated once those requests complete.
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


