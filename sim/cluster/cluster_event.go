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
	Priority() int // 0=Arrival, 1=Admission, 2=Routing, 3=Disaggregation, 4=PrefillRouting, 5=KVTransferStarted, 6=KVTransferCompleted, 7=DecodeRouting
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

	// BC-PD-4: When pools are configured, schedule DisaggregationDecisionEvent
	// between admission and routing. When not configured, go directly to routing.
	if cs.poolsConfigured() {
		heap.Push(&cs.clusterEvents, clusterEventEntry{
			event: &DisaggregationDecisionEvent{
				time:    e.time,
				request: e.request,
			},
			seqID: cs.nextSeqID(),
		})
	} else {
		heap.Push(&cs.clusterEvents, clusterEventEntry{
			event: &RoutingDecisionEvent{
				time:    e.time + cs.routingLatency,
				request: e.request,
			},
			seqID: cs.nextSeqID(),
		})
	}
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
func (e *RoutingDecisionEvent) Execute(cs *ClusterSimulator) {
	state := buildRouterState(cs)
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
	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			// Increment in-flight AFTER target validation — gives next routing decision
			// visibility into this routing decision (#170)
			cs.inFlightRequests[decision.TargetInstance]++
			inst.InjectRequestOnline(e.request, e.time)
			// BC-PD-28: Notify observer after routing so decider can learn prefix (R17, INV-7)
			cs.notifyDisaggregationObserver(e.request, decision.TargetInstance)
			return
		}
	}

	// Should never reach here (policy contract ensures valid target)
	panic(fmt.Sprintf("RoutingDecisionEvent: invalid TargetInstance %q", decision.TargetInstance))
}

// DisaggregationDecisionEvent represents the PD disaggregation decision point for a request.
// Priority 3: processed after routing events at the same timestamp.
// Bifurcates: disaggregate=true → PrefillRoutingEvent, disaggregate=false → RoutingDecisionEvent.
// For a single request, this event always precedes the event it schedules, since that event is
// pushed at T+routingLatency from within Execute() and will fire at a later timestamp.
type DisaggregationDecisionEvent struct {
	time    int64
	request *sim.Request
}

func (e *DisaggregationDecisionEvent) Timestamp() int64 { return e.time }
func (e *DisaggregationDecisionEvent) Priority() int     { return 3 }

// Execute calls the disaggregation decider and bifurcates the request flow.
// disaggregate=true: splits request into prefill sub-request, schedules PrefillRoutingEvent.
// disaggregate=false: schedules standard RoutingDecisionEvent (unchanged path).
func (e *DisaggregationDecisionEvent) Execute(cs *ClusterSimulator) {
	decision := cs.disaggregationDecider.Decide(e.request)
	logrus.Debugf("[cluster] req %s: disaggregate=%v", e.request.ID, decision.Disaggregate)

	// Record disaggregation decision if tracing is enabled (BC-PD-17)
	if cs.trace != nil {
		cs.trace.RecordDisaggregation(trace.DisaggregationRecord{
			RequestID:    e.request.ID,
			Clock:        cs.clock,
			Disaggregate: decision.Disaggregate,
		})
	}

	if !decision.Disaggregate {
		// Local path: standard routing (unchanged)
		heap.Push(&cs.clusterEvents, clusterEventEntry{
			event: &RoutingDecisionEvent{
				time:    e.time + cs.routingLatency,
				request: e.request,
			},
			seqID: cs.nextSeqID(),
		})
		return
	}

	// Disaggregated path: split request and route to prefill pool
	parent := NewParentRequest(e.request, cs.config.BlockSizeTokens)
	cs.parentRequests[parent.ID] = parent

	// Create prefill sub-request: same input, no output (completes after prefill)
	prefillSubReq := &sim.Request{
		ID:          parent.PrefillSubReqID,
		InputTokens: e.request.InputTokens,
		// No output tokens assigned: zero-output sub-request completes at prefill end
		State:       sim.StateQueued,
		ArrivalTime: e.request.ArrivalTime,
		TenantID:    e.request.TenantID,
		SLOClass:    e.request.SLOClass,
		Model:       e.request.Model,
	}

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &PrefillRoutingEvent{
			time:      e.time + cs.routingLatency,
			request:   prefillSubReq,
			parentReq: parent,
		},
		seqID: cs.nextSeqID(),
	})
}
