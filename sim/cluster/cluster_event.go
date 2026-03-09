package cluster

import (
	"container/heap"
	"fmt"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// ClusterEvent defines the interface for cluster-level events.
// These are separate from sim.Event and processed by ClusterSimulator's control plane.
type ClusterEvent interface {
	Timestamp() int64
	Priority() int // 0=Arrival, 1=Admission, 2=Routing, 3=Handoff, 4=DecodeRouting
	Execute(*ClusterSimulator)
}

// clusterEventEntry wraps a ClusterEvent with a sequence ID for deterministic FIFO
// tie-breaking when timestamp and priority are equal.
type clusterEventEntry struct {
	event ClusterEvent
	seqID int64
}

// ClusterEventQueue is a min-heap ordered by (Timestamp, Priority, seqID).
// Implements heap.Interface.
type ClusterEventQueue []clusterEventEntry

func (q ClusterEventQueue) Len() int { return len(q) }

func (q ClusterEventQueue) Less(i, j int) bool {
	if q[i].event.Timestamp() != q[j].event.Timestamp() {
		return q[i].event.Timestamp() < q[j].event.Timestamp()
	}
	if q[i].event.Priority() != q[j].event.Priority() {
		return q[i].event.Priority() < q[j].event.Priority()
	}
	return q[i].seqID < q[j].seqID
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

// buildRouterState constructs a RouterState from current cluster state.
// Collects snapshots from all instances via SnapshotProvider and bundles with clock.
// Used by both AdmissionDecisionEvent and RoutingDecisionEvent (BC-8).
func buildRouterState(cs *ClusterSimulator) *sim.RouterState {
	snapshots := make([]sim.RoutingSnapshot, len(cs.instances))
	for i, inst := range cs.instances {
		snap := cs.snapshotProvider.Snapshot(inst.ID(), cs.clock)
		snap.InFlightRequests = cs.inFlightRequests[string(inst.ID())]
		snapshots[i] = snap
	}
	return &sim.RouterState{
		Snapshots: snapshots,
		Clock:     cs.clock,
	}
}

// ClusterArrivalEvent represents a request arriving at the cluster control plane.
// Priority 0 (highest): processed before admission and routing at the same timestamp.
type ClusterArrivalEvent struct {
	time    int64
	request *sim.Request
}

func (e *ClusterArrivalEvent) Timestamp() int64 { return e.time }
func (e *ClusterArrivalEvent) Priority() int     { return 0 }

// Execute schedules an AdmissionDecisionEvent with the configured admission latency.
func (e *ClusterArrivalEvent) Execute(cs *ClusterSimulator) {
	logrus.Debugf("[cluster] req %s arrived at tick %d", e.request.ID, e.time)
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &AdmissionDecisionEvent{
			time:    e.time + cs.admissionLatency,
			request: e.request,
		},
		seqID: cs.nextSeqID(),
	})
}

// AdmissionDecisionEvent represents the admission decision point for a request.
// Priority 1: processed after arrivals but before routing at the same timestamp.
type AdmissionDecisionEvent struct {
	time    int64
	request *sim.Request
}

func (e *AdmissionDecisionEvent) Timestamp() int64 { return e.time }
func (e *AdmissionDecisionEvent) Priority() int     { return 1 }

// Execute checks admission policy with full RouterState (BC-8: includes snapshots).
// If admitted, schedules a RoutingDecisionEvent.
// If rejected, increments cs.rejectedRequests counter (EC-2).
func (e *AdmissionDecisionEvent) Execute(cs *ClusterSimulator) {
	state := buildRouterState(cs)
	admitted, reason := cs.admissionPolicy.Admit(e.request, state)
	logrus.Debugf("[cluster] req %s: admitted=%v reason=%q", e.request.ID, admitted, reason)

	// Record admission decision if tracing is enabled (BC-2)
	if cs.trace != nil {
		cs.trace.RecordAdmission(trace.AdmissionRecord{
			RequestID: e.request.ID,
			Clock:     cs.clock,
			Admitted:  admitted,
			Reason:    reason,
		})
	}

	if !admitted {
		cs.rejectedRequests++
		return
	}
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &RoutingDecisionEvent{
			time:    e.time + cs.routingLatency,
			request: e.request,
		},
		seqID: cs.nextSeqID(),
	})
}

// RoutingDecisionEvent represents the routing decision point for a request.
// Priority 2 (lowest): processed after arrivals and admissions at the same timestamp.
type RoutingDecisionEvent struct {
	time    int64
	request *sim.Request
}

func (e *RoutingDecisionEvent) Timestamp() int64 { return e.time }
func (e *RoutingDecisionEvent) Priority() int     { return 2 }

// Execute routes the request using the configured routing policy and injects it.
// In disaggregated mode, routes to the prefill pool only.
func (e *RoutingDecisionEvent) Execute(cs *ClusterSimulator) {
	// In disaggregated mode, route to prefill pool; otherwise all instances
	var state *sim.RouterState
	if cs.IsDisaggregated() {
		state = buildRouterStateForPool(cs, cs.prefillInstances, cs.prefillSnapshotProvider)
	} else {
		state = buildRouterState(cs)
	}
	decision := cs.routingPolicy.Route(e.request, state)
	logrus.Debugf("[cluster] req %s → instance %s (reason=%s)", e.request.ID, decision.TargetInstance, decision.Reason)

	// BC-9: Apply cluster-level priority hint if set by routing policy
	if decision.Priority != 0 {
		e.request.Priority = decision.Priority
	}

	// #181: Stamp request with assigned instance for per-request metrics
	e.request.AssignedInstance = decision.TargetInstance

	// Record routing decision if tracing is enabled (BC-3, BC-4, BC-5, BC-6)
	// Placed after priority assignment to minimize diff; recording reads decision, not request.Priority
	if cs.trace != nil {
		record := trace.RoutingRecord{
			RequestID:      e.request.ID,
			Clock:          cs.clock,
			ChosenInstance: decision.TargetInstance,
			Reason:         decision.Reason,
			Scores:         copyScores(decision.Scores),
		}
		if cs.trace.Config.CounterfactualK > 0 {
			record.Candidates, record.Regret = computeCounterfactual(
				decision.TargetInstance, decision.Scores,
				state.Snapshots, cs.trace.Config.CounterfactualK,
			)
		}
		cs.trace.RecordRouting(record)
	}

	// Find target instance, increment in-flight count, and inject request
	pool := cs.instances
	if cs.IsDisaggregated() {
		pool = cs.prefillInstances
	}
	for _, inst := range pool {
		if string(inst.ID()) == decision.TargetInstance {
			cs.inFlightRequests[decision.TargetInstance]++
			inst.InjectRequestOnline(e.request, e.time)
			return
		}
	}

	// Should never reach here (policy contract ensures valid target)
	panic(fmt.Sprintf("RoutingDecisionEvent: invalid TargetInstance %q", decision.TargetInstance))
}

// buildRouterStateForPool constructs a RouterState from a specific instance pool.
// Used in disaggregated mode for per-pool routing.
func buildRouterStateForPool(cs *ClusterSimulator, pool []*InstanceSimulator, provider SnapshotProvider) *sim.RouterState {
	snapshots := make([]sim.RoutingSnapshot, len(pool))
	for i, inst := range pool {
		snap := provider.Snapshot(inst.ID(), cs.clock)
		snap.InFlightRequests = cs.inFlightRequests[string(inst.ID())]
		snapshots[i] = snap
	}
	return &sim.RouterState{
		Snapshots: snapshots,
		Clock:     cs.clock,
	}
}

// HandoffEvent models KV cache transfer from prefill to decode instance.
// Priority 3: processed after routing decisions.
type HandoffEvent struct {
	time    int64
	request *sim.Request
}

func (e *HandoffEvent) Timestamp() int64 { return e.time }
func (e *HandoffEvent) Priority() int    { return 3 }

// Execute sets HandoffTime and schedules DecodeRoutingDecisionEvent.
func (e *HandoffEvent) Execute(cs *ClusterSimulator) {
	e.request.HandoffTime = e.time
	logrus.Debugf("[cluster] handoff req %s at tick %d", e.request.ID, e.time)

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &DecodeRoutingDecisionEvent{
			time:    e.time,
			request: e.request,
		},
		seqID: cs.nextSeqID(),
	})
}

// DecodeRoutingDecisionEvent routes a prefill-completed request to a decode instance.
// Priority 4: processed after handoffs.
type DecodeRoutingDecisionEvent struct {
	time    int64
	request *sim.Request
}

func (e *DecodeRoutingDecisionEvent) Timestamp() int64 { return e.time }
func (e *DecodeRoutingDecisionEvent) Priority() int    { return 4 }

// Execute routes the request to a decode instance and injects it for decode processing.
func (e *DecodeRoutingDecisionEvent) Execute(cs *ClusterSimulator) {
	state := buildRouterStateForPool(cs, cs.decodeInstances, cs.decodeSnapshotProvider)
	decision := cs.decodeRoutingPolicy.Route(e.request, state)
	logrus.Debugf("[cluster] decode-route req %s → instance %s", e.request.ID, decision.TargetInstance)

	// Apply cluster-level priority hint if set by routing policy (I5 fix, R23 parity)
	if decision.Priority != 0 {
		e.request.Priority = decision.Priority
	}

	// Update assigned instance to decode instance
	e.request.AssignedInstance = decision.TargetInstance

	// Record decode routing decision if tracing is enabled (C5 fix, R1 no silent data loss)
	if cs.trace != nil {
		record := trace.RoutingRecord{
			RequestID:      e.request.ID,
			Clock:          cs.clock,
			ChosenInstance: decision.TargetInstance,
			Reason:         "decode-routing:" + decision.Reason,
			Scores:         copyScores(decision.Scores),
		}
		if cs.trace.Config.CounterfactualK > 0 {
			record.Candidates, record.Regret = computeCounterfactual(
				decision.TargetInstance, decision.Scores,
				state.Snapshots, cs.trace.Config.CounterfactualK,
			)
		}
		cs.trace.RecordRouting(record)
	}

	for _, inst := range cs.decodeInstances {
		if string(inst.ID()) == decision.TargetInstance {
			cs.inFlightRequests[decision.TargetInstance]++
			// C2 convergence fix: InjectForDecode may drop the request via the R19
			// unservable guard. That DroppedUnservable increment happens outside
			// ProcessNextEvent, so the cluster event loop's delta check never sees it.
			// Detect the drop here and correct inFlightRequests immediately.
			droppedBefore := inst.Metrics().DroppedUnservable
			inst.InjectForDecode(e.request, e.time)
			if droppedAfter := inst.Metrics().DroppedUnservable; droppedAfter > droppedBefore {
				cs.inFlightRequests[decision.TargetInstance] -= droppedAfter - droppedBefore
			}
			return
		}
	}

	panic(fmt.Sprintf("DecodeRoutingDecisionEvent: invalid TargetInstance %q", decision.TargetInstance))
}
