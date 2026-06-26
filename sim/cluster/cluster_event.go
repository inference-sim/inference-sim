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
// Records the request as injected on its SLO class BEFORE any admission/routing
// decision so that drops and timeouts count against goodput (issue #1409, BC-5).
func (e *ClusterArrivalEvent) Execute(cs *ClusterSimulator) {
	cs.pendingArrivals--
	cs.injectedByClass[e.request.SLOClass]++
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
		// Schedule TTL event for the incoming request if TTL is enabled and
		// the request was accepted into the queue (Enqueued or ShedVictim).
		if cs.requestTTL > 0 && cs.flowControlAdmission.LastOutcome() != Rejected {
			heap.Push(&cs.clusterEvents, clusterEventEntry{
				event: &GatewayQueueTTLEvent{
					time:      cs.clock + cs.requestTTL,
					requestID: e.request.ID,
				},
				seqID: cs.nextSeqID(),
			})
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

// GatewayEvictionEvent terminates a dispatched sheddable request on its instance.
// Gateway-level eviction (distinct from instance-level KV preemption).
// The request is terminated — not requeued. This is a terminal state.
// Priority 5: processed after routing decisions at the same timestamp.
type GatewayEvictionEvent struct {
	time           int64
	request        *sim.Request
	targetInstance string
}

func (e *GatewayEvictionEvent) Timestamp() int64 { return e.time }
func (e *GatewayEvictionEvent) Priority() int     { return 5 }

func (e *GatewayEvictionEvent) Execute(cs *ClusterSimulator) {
	logrus.Debugf("[cluster] gateway eviction: req %s evicted from instance %s at tick %d",
		e.request.ID, e.targetInstance, e.time)

	var evicted bool
	for _, inst := range cs.instances {
		if string(inst.ID()) == e.targetInstance {
			evicted = inst.EvictRequest(e.request)
			break
		}
	}
	if !evicted {
		return
	}

	cs.inFlightRequests[e.targetInstance]--
	if cs.inFlightRequests[e.targetInstance] < 0 {
		cs.inFlightRequests[e.targetInstance] = 0
	}

	if cs.tenantTracker != nil {
		cs.tenantTracker.OnComplete(e.request.TenantID)
	}

	cs.gatewayEvicted++
	tier := e.request.SLOClass
	if tier == "" {
		tier = "standard"
	}
	cs.shedByTier[tier]++

}

// GatewayQueueTTLEvent fires when a request's TTL expires while queued.
// If the request is still in the gateway queue, it is removed and counted as expired.
// If the request was already dispatched or shed, this is a no-op.
// Priority 6: after eviction (5), before scaling (8).
type GatewayQueueTTLEvent struct {
	time      int64
	requestID string
}

func (e *GatewayQueueTTLEvent) Timestamp() int64 { return e.time }
func (e *GatewayQueueTTLEvent) Priority() int     { return 6 }

func (e *GatewayQueueTTLEvent) Execute(cs *ClusterSimulator) {
	req := cs.gatewayQueue.RemoveByRequestID(e.requestID)
	if req == nil {
		return
	}
	logrus.Debugf("[cluster] gateway TTL expiry: req %s expired at tick %d", e.requestID, e.time)
	cs.gatewayExpired++
	tier := req.SLOClass
	if tier == "" {
		tier = "standard"
	}
	cs.shedByTier[tier]++
}

// GatewayDispatchTickEvent is a periodic dispatch trigger for the gateway queue.
// DES equivalent of llm-d's 1ms dispatchTicker (processor.go:179).
// Demand-driven: only active when flow control enabled AND queue non-empty.
// Self-scheduling: reschedules while queue has requests, stops when drained.
// Priority 7: after TTL expiry (6), before scaling (8).
type GatewayDispatchTickEvent struct {
	At       int64
	Interval int64
}

func (e *GatewayDispatchTickEvent) Timestamp() int64 { return e.At }
func (e *GatewayDispatchTickEvent) Priority() int     { return 7 }

func (e *GatewayDispatchTickEvent) Execute(cs *ClusterSimulator) {
	if cs.gatewayQueue == nil {
		cs.dispatchTickPending = false
		return
	}

	if cs.gatewayQueue.Len() > 0 {
		cs.tryDispatchFromGatewayQueue()
	}

	if cs.gatewayQueue.Len() > 0 {
		heap.Push(&cs.clusterEvents, clusterEventEntry{
			event: &GatewayDispatchTickEvent{At: e.At + e.Interval, Interval: e.Interval},
			seqID: cs.nextSeqID(),
		})
	} else {
		cs.dispatchTickPending = false
	}
}

// DisaggregationDecisionEvent represents the PD disaggregation decision point for a request.
// Priority 3: scheduled by AdmissionEvent in place of RoutingDecisionEvent (2) when pool
// topology is configured; fires after admission but before per-pool routing events (4+).
// Bifurcates: disaggregate=true → PrefillRoutingEvent, disaggregate=false → RoutingDecisionEvent.
type DisaggregationDecisionEvent struct {
	time    int64
	request *sim.Request
}

func (e *DisaggregationDecisionEvent) Timestamp() int64 { return e.time }
func (e *DisaggregationDecisionEvent) Priority() int     { return 3 }

// Execute implements the llm-d decode-first routing order:
// 1. Select a decode pod first (via decode pool routing).
// 2. Decide disaggregation with the decode pod known.
// 3a. disaggregate=false: inject directly to the selected decode pod.
// 3b. disaggregate=true: store decode pod in ParentRequest upfront, route prefill,
//
//	KV-transfer, then inject directly to the pre-selected decode pod (no second
//	routing decision).
func (e *DisaggregationDecisionEvent) Execute(cs *ClusterSimulator) {
	// Step 1: route to decode pool first (llm-d parity: decode pod always selected first).
	filteredSnapshots := cs.buildPoolFilteredSnapshots(PoolRoleDecode)
	if len(filteredSnapshots) == 0 {
		logrus.Warnf("[cluster] req %s: no routable instances in decode pool — request rejected at routing", e.request.ID)
		cs.routingRejections++
		return
	}
	state := &sim.RouterState{Snapshots: filteredSnapshots, Clock: cs.clock}
	policy := cs.decodeRoutingPolicy
	if policy == nil {
		policy = cs.routingPolicy
	}
	decodeDecision := policy.Route(e.request, state)
	logrus.Debugf("[cluster] req %s: decode pod pre-selected → %s", e.request.ID, decodeDecision.TargetInstance)

	// Step 2: disaggregation decision with decode pod known.
	disaggDecision := cs.disaggregationDecider.Decide(e.request, state)
	logrus.Debugf("[cluster] req %s: disaggregate=%v", e.request.ID, disaggDecision.Disaggregate)

	// Record disaggregation decision if tracing is enabled (BC-PD-17).
	if cs.trace != nil {
		cs.trace.RecordDisaggregation(trace.DisaggregationRecord{
			RequestID:    e.request.ID,
			Clock:        cs.clock,
			Disaggregate: disaggDecision.Disaggregate,
		})
	}

	// Find the target decode instance object (used in both paths below).
	var decodeInst *InstanceSimulator
	for _, inst := range cs.instances {
		if string(inst.ID()) == decodeDecision.TargetInstance {
			decodeInst = inst
			break
		}
	}
	if decodeInst == nil {
		// R6: routing policy must return a valid target from the provided snapshot set.
		panic(fmt.Sprintf("DisaggregationDecisionEvent: invalid decode TargetInstance %q returned by routing policy", decodeDecision.TargetInstance))
	}

	if !disaggDecision.Disaggregate {
		// Step 3a: local path — inject directly to the selected decode pod.
		// This fixes P3: non-disaggregated requests are now routed exclusively to the decode
		// pool, not to all instances via buildRouterState().
		e.request.AssignedInstance = decodeDecision.TargetInstance

		// Record standard routing trace for BC-TRACE-COMPAT: consumers (e.g.
		// TestPDTrace_NeverDecider_WithPools) expect len(tr.Routings) == numRequests.
		if cs.trace != nil {
			record := trace.RoutingRecord{
				RequestID:      e.request.ID,
				Clock:          cs.clock,
				ChosenInstance: decodeDecision.TargetInstance,
				Reason:         decodeDecision.Reason,
				Scores:         copyScores(decodeDecision.Scores),
			}
			if cs.trace.Config.CounterfactualK > 0 {
				record.Candidates, record.Regret = computeCounterfactual(
					decodeDecision.TargetInstance, decodeDecision.Scores,
					filteredSnapshots, cs.trace.Config.CounterfactualK,
				)
			}
			cs.trace.RecordRouting(record)
		}

		cs.inFlightRequests[decodeDecision.TargetInstance]++
		if cs.tenantTracker != nil {
			cs.tenantTracker.OnStart(e.request.TenantID)
		}
		warmUpCount := cs.config.InstanceLifecycle.WarmUpRequestCount
		if warmUpCount > 0 && len(decodeInst.WarmUpRequestIDs()) < warmUpCount {
			decodeInst.RecordWarmUpRequest(e.request.ID)
		}
		decodeInst.InjectRequestOnline(e.request, e.time+cs.routingLatency)
		if cs.evictionTracker != nil {
			cs.evictionTracker.Track(e.request, decodeDecision.TargetInstance, cs.priorityMap)
		}
		return
	}

	// Step 3b: disaggregated path — decode pod pre-selected, route prefill next.
	// This fixes P1: decode pod is stored upfront in ParentRequest before prefill routing.
	parent := NewParentRequest(e.request, cs.config.BlockSizeTokens)
	parent.DecodeInstanceID = InstanceID(decodeDecision.TargetInstance)
	cs.parentRequests[parent.ID] = parent

	// Create prefill sub-request: same input, no output (completes after prefill).
	// Output is intentionally nil: zero-output request completes at prefill end.
	// InputTokens is a slice-header alias of e.request.InputTokens (#1445) — the
	// sub-request views the same underlying token buffer, no flatten. If
	// Request.InputTokens ever becomes lazy/chained, this site must update.
	prefillSubReq := &sim.Request{
		ID:           parent.PrefillSubReqID,
		InputTokens:  e.request.InputTokens,
		MaxOutputLen: e.request.MaxOutputLen,
		Deadline:     e.request.Deadline,
		PrefixGroup:  e.request.PrefixGroup,
		State:        sim.StateQueued,
		ArrivalTime:  e.request.ArrivalTime,
		TenantID:     e.request.TenantID,
		SLOClass:     e.request.SLOClass,
		Model:        e.request.Model,
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
