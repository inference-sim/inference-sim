package cluster

import (
	"container/heap"
	"fmt"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// ClusterEvent defines the interface for cluster-level events.
// These are separate from sim.Event and processed by ClusterSimulator's control plane.
// Priority is the event's class: its place among the events of one instant (eventKey).
type ClusterEvent interface {
	Timestamp() int64
	Priority() int // the class; see "Classes of cluster events" below
	Execute(*ClusterSimulator)
}

// Classes of cluster events, in the order they run at one instant (DEV-2, design §8.2). The
// class is the event's Priority(); a lower class runs first. Each position has a reason:
//
//	-2  node lifecycle (NodeReadyEvent, NodeDrainedEvent): a node's state changes before any
//	    instance on it does.
//	-1  instance lifecycle (InstanceLoadedEvent): an instance that finishes loading at t is
//	    routable for a request that arrives at t.
//	 0  arrival (ClusterArrivalEvent): external input enters before anything that reacts to it.
//	 2  router request path (CallbackEvent at PriorityGatewayRequest): the router decides on
//	    the requests that arrived at t, where the cluster's own routing decision always ran.
//	 4  prefill routing (PrefillRoutingEvent): a disaggregated request reaches its prefill
//	    engine after the router chose it.
//	 5  KV transfer start (KVTransferStartedEvent): a prefill that ended at t starts its
//	    handoff after every routing decision of t.
//	 6  KV transfer completion, link wake and prefill lease expiry (KVTransferCompletedEvent,
//	    PDLinkWakeEvent, PrefillKVLeaseExpiredEvent): a transfer that starts and ends at t
//	    completes after it starts.
//	 8  scaling tick (ScalingTickEvent): the autoscaler observes every request-path change of t.
//	 9  scale actuation (ScaleActuationEvent): a decision is applied after the tick that made it.
//	10+ router background work (metrics collection, KV events, timers, housekeeping): after
//	    every cluster event of t, as in the router's own process.
//
// Every cluster event of an instant runs before every instance event of that instant (the
// cluster loop's "cluster first" rule, BC-4), so an engine sees the instant's arrivals and
// routing before it steps. Instance events of one instant run in instance order (the
// instance is their sender). A change to this order is a change to the model (design §8.9).

// senderID names a component that sends cluster events: the sender in eventKey. Its value
// orders two events of the same time and class from different senders, so each value's place
// below is part of the model, and two components never share one.
type senderID int

const (
	// senderLifecycle is the cluster's placement and lifecycle controller.
	senderLifecycle senderID = iota
	// senderWorkload is the workload generator: the request source, closed-loop session
	// follow-ups and the agentic arrival gate. It comes before every instance, so an arrival
	// the workload planned at t is routed before a drained instance's redirect at t.
	senderWorkload
	// senderRouter is the embedded router (every gateway callback) and the prefill routing it
	// orders.
	senderRouter
	// senderKVConnector is the KV connector between prefill and decode pools: transfer
	// starts, completions, link wakes and prefill-side lease expiries. Those of one class run
	// in the order the connector sent them.
	senderKVConnector
	// senderAutoscaler is the autoscaler: its ticks and actuations.
	senderAutoscaler
	// senderFirstInstance is the sender of the instance at index 0; instance i sends as
	// senderFirstInstance + i, in the order the cluster loop breaks instance ties.
	senderFirstInstance senderID = 1 << 20
)

// eventSender is one component's outbox: its sender identity and its own count of the events
// it has sent. The count, not a counter shared by the whole simulator, orders two events from
// one sender at the same time and class, so the order of an instant depends only on what each
// sender has done (DEV-2).
type eventSender struct {
	id   senderID
	sent int64
}

// next returns the count of the event being sent and advances it.
func (s *eventSender) next() int64 {
	n := s.sent
	s.sent++
	return n
}

// eventKey orders the cluster's events: (time, class, sender, sender's own count), the key of
// design §8.2. No global counter enters it. Two queued events with the same key are a defect:
// ClusterSimulator.schedule panics on one.
type eventKey struct {
	time   int64
	class  int
	sender senderID
	count  int64
}

// less orders two keys.
func (a eventKey) less(b eventKey) bool {
	if a.time != b.time {
		return a.time < b.time
	}
	if a.class != b.class {
		return a.class < b.class
	}
	if a.sender != b.sender {
		return a.sender < b.sender
	}
	return a.count < b.count
}

// clusterEventEntry is a queued event with its key.
type clusterEventEntry struct {
	event ClusterEvent
	key   eventKey
	// salt is consulted only between two equal keys, which schedule refuses, so it never
	// decides an order in a run. It is zero except under the permutation test, which sets it
	// at random to stand for an arbitrary scheduling order: a key that stopped being total
	// would then let the scheduling order decide, and the test would see the Result move.
	salt int64
}

// ClusterEventQueue is a min-heap ordered by eventKey. Implements heap.Interface.
type ClusterEventQueue []clusterEventEntry

func (q ClusterEventQueue) Len() int { return len(q) }

func (q ClusterEventQueue) Less(i, j int) bool {
	if q[i].key != q[j].key {
		return q[i].key.less(q[j].key)
	}
	return q[i].salt < q[j].salt
}

func (q ClusterEventQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }

func (q *ClusterEventQueue) Push(x any) {
	*q = append(*q, x.(clusterEventEntry))
}

func (q *ClusterEventQueue) Pop() any {
	old := *q
	n := len(old)
	item := old[n-1]
	*q = old[:n-1]
	return item
}

// schedule queues ev, sent by from, under its key. It panics when another queued event already
// has that key: two components sharing a sender identity, or a sender reusing a count (DEV-2).
func (cs *ClusterSimulator) schedule(ev ClusterEvent, from *eventSender) {
	k := eventKey{time: ev.Timestamp(), class: ev.Priority(), sender: from.id, count: from.next()}
	if cs.queuedKeys == nil {
		cs.queuedKeys = make(map[eventKey]struct{})
	}
	if _, dup := cs.queuedKeys[k]; dup {
		panic(fmt.Sprintf("ClusterSimulator.schedule: two queued events share the key (time %d, class %d, sender %d, count %d); "+
			"DEV-2 requires every key to be unique (%T)", k.time, k.class, k.sender, k.count, ev))
	}
	cs.queuedKeys[k] = struct{}{}
	var salt int64
	if cs.permuteSchedule != nil {
		salt = cs.permuteSchedule.Int63()
	}
	heap.Push(&cs.clusterEvents, clusterEventEntry{event: ev, key: k, salt: salt})
}

// popEvent removes and returns the earliest queued event.
func (cs *ClusterSimulator) popEvent() clusterEventEntry {
	entry := heap.Pop(&cs.clusterEvents).(clusterEventEntry)
	delete(cs.queuedKeys, entry.key)
	if cs.observePop != nil {
		cs.observePop(entry.key)
	}
	return entry
}

// instanceSender is inst's outbox for the cluster events it sends: the requests a drain
// redirects from it. Its sender identity is its index.
func (cs *ClusterSimulator) instanceSender(inst *InstanceSimulator) *eventSender {
	if s, ok := cs.instanceSenders[inst.ID()]; ok {
		return s
	}
	idx := -1
	for i, in := range cs.instances {
		if in == inst {
			idx = i
			break
		}
	}
	if idx < 0 {
		panic(fmt.Sprintf("ClusterSimulator.instanceSender: instance %s is not in the cluster", inst.ID()))
	}
	if cs.instanceSenders == nil {
		cs.instanceSenders = make(map[InstanceID]*eventSender)
	}
	s := &eventSender{id: senderFirstInstance + senderID(idx)}
	cs.instanceSenders[inst.ID()] = s
	return s
}

// buildClusterState constructs the autoscaler's view of the cluster from live instance state.
// Filters to instances that are routable (IsRoutable()) and, when req is non-nil and
// has a non-empty Model, further filters to instances serving that model (T044, T048).
// Complexity: O(N) per call where N = number of instances.
func buildClusterState(cs *ClusterSimulator, req *sim.Request) *sim.ClusterState {
	snapshots := make([]sim.InstanceStatus, 0, len(cs.instances))
	for _, inst := range cs.instances {
		// Filter by lifecycle state (T044): exclude non-routable instances
		if !inst.IsRoutable() {
			continue
		}
		// Filter by model (T048): when request has a model, only include matching instances.
		// When req.Model is empty, include all (single-model backward-compat).
		if req != nil && req.Model != "" && inst.Model != req.Model {
			continue
		}
		snap := liveSnapshot(inst)
		snap.InFlightRequests = cs.inFlightRequests[string(inst.ID())]
		snap.Model = inst.Model
		snap.GPUType = inst.GPU()
		snap.TPDegree = inst.TPDegree
		snap.CostPerHour = inst.CostPerHour
		snap.MaxBatchSize = float64(inst.MaxBatchSize()) // float64: QueueingModelAnalyzer uses it in float arithmetic
		snapshots = append(snapshots, snap)
	}
	// Collect Loading instances as pending supply information for the autoscaler.
	// TotalKvCapacityTokens is
	// provisioned at NewInstanceSimulator and does not change over the instance lifetime.
	loadingSnapshots := make([]sim.InstanceStatus, 0, len(cs.instances))
	for _, inst := range cs.instances {
		if inst.State != sim.InstanceStateLoading {
			continue
		}
		if inst.Model == "" || inst.GPU() == "" {
			logrus.Debugf("[cluster] buildClusterState: loading instance %q has empty Model=%q or GPU=%q — excluded from LoadingSnapshots",
				inst.ID(), inst.Model, inst.GPU())
			continue
		}
		snap := sim.NewInstanceStatus(string(inst.ID()))
		snap.Model = inst.Model
		snap.GPUType = inst.GPU()
		snap.TPDegree = inst.TPDegree
		snap.CostPerHour = inst.CostPerHour
		snap.TotalKvCapacityTokens = inst.TotalKvCapacityTokens()
		// QueueDepth, BatchSize, KVUtilization, FreeKVBlocks, CacheHitRate, InFlightRequests, KvTokensInUse remain zero.
		loadingSnapshots = append(loadingSnapshots, snap)
	}
	return &sim.ClusterState{
		Snapshots:        snapshots,
		LoadingSnapshots: loadingSnapshots,
		Clock:            cs.clock,
	}
}

// liveSnapshot reads an instance's current state.
func liveSnapshot(inst *InstanceSimulator) sim.InstanceStatus {
	snap := sim.NewInstanceStatus(string(inst.ID()))
	snap.PreemptionCount = inst.PreemptionCount()
	snap.QueueDepth = inst.QueueDepth()
	snap.BatchSize = inst.BatchSize()
	snap.KVUtilization = inst.KVUtilization()
	snap.FreeKVBlocks = inst.FreeKVBlocks()
	snap.CacheHitRate = inst.CacheHitRate()
	snap.TotalKvCapacityTokens = inst.TotalKvCapacityTokens()
	snap.KvTokensInUse = inst.KvTokensInUse()
	return snap
}

// ClusterArrivalEvent represents a request arriving at the cluster.
// Priority 0 (highest): processed before every other cluster event at the same timestamp.
type ClusterArrivalEvent struct {
	time    int64
	request *sim.Request
}

func (e *ClusterArrivalEvent) Timestamp() int64 { return e.time }
func (e *ClusterArrivalEvent) Priority() int    { return 0 }

// Execute hands the request to the gateway.
// Records the request as injected on its SLO class BEFORE any admission/routing
// decision so that drops and timeouts count against goodput (issue #1409, BC-5).
func (e *ClusterArrivalEvent) Execute(cs *ClusterSimulator) {
	cs.pendingArrivals--
	if !cs.gateArrival(e.request, e.time) {
		return
	}
	cs.injectedByClass[e.request.SLOClass]++
	if !e.request.Redirected {
		cs.arrivals++
	}
	cs.arriveRequest(e.request)
	logrus.Debugf("[cluster] req %s arrived at tick %d", e.request.ID, e.time)
	// Fire the arrival hook (issue #1440): trace exporters see each fresh
	// arrival exactly once, in clock-monotonic order (INV-3). REDIRECT
	// re-injections are filtered inside fireArrivalHook.
	cs.fireArrivalHook(e.request, e.time)
	cs.gateway.OnArrival(e.request)
}

// ---------------------------------------------------------------------------
// Phase 1C: Autoscaler events
// ---------------------------------------------------------------------------

// ScalingTickEvent fires the autoscaling pipeline at the configured interval.
// Priority 8: after all request-path events (0–7) at the same timestamp, so the
// scaler observes a stable snapshot of completed request state.
// Self-scheduling: Execute() schedules the next ScalingTickEvent.
// Zero-interval guard: when ModelAutoscalerIntervalUs == 0, no tick is ever scheduled.
type ScalingTickEvent struct {
	At int64 // simulation timestamp in microseconds
}

func (e *ScalingTickEvent) Timestamp() int64 { return e.At }
func (e *ScalingTickEvent) Priority() int    { return 8 }

// Execute runs the autoscaling pipeline: Collect → Analyze → Optimize → stabilization window gate
// → schedule ScaleActuationEvent → schedule next ScalingTickEvent.
// Full orchestrator logic is wired in US1 (T009–T015).
func (e *ScalingTickEvent) Execute(cs *ClusterSimulator) {
	if cs.autoscaler == nil {
		logrus.Warnf("[autoscaler] ScalingTickEvent at t=%d fired but cs.autoscaler is nil — event dropped", e.At)
		return
	}
	cs.autoscaler.tick(cs, e.At)
}

// ScaleActuationEvent carries scale decisions to apply after the actuation delay elapses.
// Separates the "decide" step from the "act" step to model HPA/KEDA scrape lag.
// Priority 9: after ScalingTickEvent at the same timestamp.
type ScaleActuationEvent struct {
	At        int64
	Decisions []ScaleDecision
}

func (e *ScaleActuationEvent) Timestamp() int64 { return e.At }
func (e *ScaleActuationEvent) Priority() int    { return 9 }

// Execute calls Actuator.Apply(decisions).
// Full actuator logic is wired in US3 (T019–T023).
func (e *ScaleActuationEvent) Execute(cs *ClusterSimulator) {
	if cs.autoscaler == nil {
		if len(e.Decisions) > 0 {
			logrus.Warnf("[autoscaler] ScaleActuationEvent at t=%d: %d decision(s) dropped — cs.autoscaler is nil", e.At, len(e.Decisions))
		}
		return
	}
	cs.autoscaler.actuate(cs, e.Decisions)
}
