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
	Priority() int // 0=Arrival, 1=Admission, 2=Routing, 4-6=PD, 8=ScalingTick, 9=ScaleActuation
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
// Filters to instances that are routable (IsRoutable()) and, when req is non-nil and
// has a non-empty Model, further filters to instances serving that model (T044, T048).
// Used by both AdmissionDecisionEvent and RoutingDecisionEvent (BC-8).
// Complexity: O(N) per call where N = number of instances. At current BLIS scale (≤32 instances)
// this is negligible. A model→instances index could pre-partition if instance counts exceed 100.
func buildRouterState(cs *ClusterSimulator, req *sim.Request) *sim.RouterState {
	// Refresh stale cache snapshots if interval has elapsed (#919, #1060).
	// No-op when CacheBlocks.Mode != Periodic (oracle mode).
	cs.snapshotProvider.RefreshCacheIfNeeded(cs.clock)

	snapshots := make([]sim.RoutingSnapshot, 0, len(cs.instances))
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
		snap := cs.snapshotProvider.Snapshot(inst.ID(), cs.clock)
		snap.InFlightRequests = cs.inFlightRequests[string(inst.ID())]
		snap.Model = inst.Model
		snap.GPUType = inst.GPU()
		snap.TPDegree = inst.TPDegree
		snap.CostPerHour = inst.CostPerHour
		// Populate latency stats from completed-request metrics (zero until first completion).
		if inst.HasSim() {
			ls := inst.LatencyStats()
			snap.TTFT = ls.TTFT
			snap.ITL = ls.ITL
			snap.DispatchRate = ls.DispatchRate
			snap.AvgInTokens = ls.AvgInTokens
			snap.AvgOutTokens = ls.AvgOutTokens
		}
		snap.MaxBatchSize = float64(inst.MaxBatchSize()) // float64: QueueingModelAnalyzer uses it in float arithmetic
		snapshots = append(snapshots, snap)
	}
	// Collect Loading instances as pending supply information for the autoscaler.
	// Uses inst accessors directly (not snapshotProvider) — TotalKvCapacityTokens is
	// provisioned at NewInstanceSimulator and does not change over the instance lifetime.
	loadingSnapshots := make([]sim.RoutingSnapshot, 0, len(cs.instances))
	for _, inst := range cs.instances {
		if inst.State != sim.InstanceStateLoading {
			continue
		}
		if inst.Model == "" || inst.GPU() == "" {
			logrus.Debugf("[cluster] buildRouterState: loading instance %q has empty Model=%q or GPU=%q — excluded from LoadingSnapshots",
				inst.ID(), inst.Model, inst.GPU())
			continue
		}
		snap := sim.NewRoutingSnapshot(string(inst.ID()))
		snap.Model = inst.Model
		snap.GPUType = inst.GPU()
		snap.TPDegree = inst.TPDegree
		snap.CostPerHour = inst.CostPerHour
		snap.TotalKvCapacityTokens = inst.TotalKvCapacityTokens()
		// QueueDepth, BatchSize, KVUtilization, FreeKVBlocks, CacheHitRate, InFlightRequests, KvTokensInUse remain zero.
		loadingSnapshots = append(loadingSnapshots, snap)
	}
	return &sim.RouterState{
		Snapshots:        snapshots,
		LoadingSnapshots: loadingSnapshots,
		Clock:            cs.clock,
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
	cs.pendingArrivals--
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

// Execute processes the admission decision for an incoming request.
// Checks admission policy with full RouterState (BC-8: includes snapshots).
// If admitted, schedules a RoutingDecisionEvent.
// If rejected, increments cs.rejectedRequests counter (EC-2).
func (e *AdmissionDecisionEvent) Execute(cs *ClusterSimulator) {
	state := buildRouterState(cs, e.request)
	admitted, reason := cs.admissionPolicy.Admit(e.request, state)
	logrus.Debugf("[cluster] req %s: admitted=%v reason=%q", e.request.ID, admitted, reason)

	if !admitted {
		// Record rejection from admission policy before returning (BC-2).
		if cs.trace != nil {
			cs.trace.RecordAdmission(trace.AdmissionRecord{
				RequestID: e.request.ID,
				Clock:     cs.clock,
				Admitted:  false,
				Reason:    reason,
			})
		}
		cs.rejectedRequests++
		// Populate per-tier shed counter for every admission rejection, regardless of policy.
		tier := e.request.SLOClass
		if tier == "" {
			tier = "standard" // normalize empty → standard (matches SLOPriorityMap default)
		}
		cs.shedByTier[tier]++
		return
	}

	// Record admission (BC-2): tenant budget enforcement is in the TenantBudgetAdmission decorator.
	if cs.trace != nil {
		cs.trace.RecordAdmission(trace.AdmissionRecord{
			RequestID: e.request.ID,
			Clock:     cs.clock,
			Admitted:  true,
			Reason:    reason,
		})
	}

	// Flow control: FlowControlAdmission.Admit() already processed the request (enqueue or rejection).
	// Handle queue-level outcomes (shed victim accounting, dispatch).
	// When flow control is disabled (default), cs.flowControlAdmission is nil (BC-1).
	//
	// Trace gap: RecordAdmission above already wrote Admitted:true for any request
	// that reaches the gateway queue. No trace.RoutingRecord is emitted for requests
	// subsequently rejected or shed from the queue. Trace consumers must not assume
	// all Admitted:true requests have routing records when flow control is enabled.
	if cs.flowControlAdmission != nil {
		switch cs.flowControlAdmission.LastOutcome() {
		case Rejected:
			// Queue-rejected. NOT an admission rejection (Admit returned true).
			// Accounted via gatewayQueue.RejectedCount() (INV-1: gateway_queue_rejected).
			logrus.Debugf("[cluster] req %s: flow-control queue rejected", e.request.ID)
			return
		case ShedVictim:
			victim := cs.flowControlAdmission.LastShedVictim()
			if victim == nil {
				panic(fmt.Sprintf("FlowControlAdmission: ShedVictim outcome but nil victim for req %s", e.request.ID))
			}
			tier := victim.SLOClass
			if tier == "" {
				tier = "standard"
			}
			cs.shedByTier[tier]++
		case Enqueued:
			// nothing extra
		default:
			panic(fmt.Sprintf("unhandled FlowControlAdmission outcome: %v", cs.flowControlAdmission.LastOutcome()))
		}
		cs.tryDispatchFromGatewayQueue()
		return
	}

	// Schedule routing. RoutingDecisionEvent.Execute branches internally on
	// cs.poolsConfigured(): disaggregated routing for PD topology, standard routing
	// otherwise. This mirrors llm-d-inference-scheduler's disagg-profile-handler,
	// which handles both paths inside a single scheduling plugin entry point.
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
// Dispatches to executeDisaggregatedRouting when pool topology is configured (PD
// disaggregation active), otherwise to executeStandardRouting. Both paths live on
// ClusterSimulator so the fork happens inside one event type rather than at the
// two pre-existing call sites in AdmissionDecisionEvent and tryDispatchFromGatewayQueue.
func (e *RoutingDecisionEvent) Execute(cs *ClusterSimulator) {
	if cs.poolsConfigured() {
		cs.executeDisaggregatedRouting(e.request, e.time)
	} else {
		cs.executeStandardRouting(e.request, e.time)
	}
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
func (e *ScalingTickEvent) Priority() int     { return 8 }

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
func (e *ScaleActuationEvent) Priority() int     { return 9 }

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
