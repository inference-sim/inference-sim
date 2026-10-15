package cluster

import (
	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// ClusterEvent defines the interface for cluster-level events.
// These are separate from sim.Event and processed by ClusterSimulator's control plane.
type ClusterEvent interface {
	Timestamp() int64
	Priority() int // 0=Arrival, 4-6=PD, 8=ScalingTick, 9=ScaleActuation, 10+=gateway callbacks
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

// buildRouterState constructs the autoscaler's view of the cluster from live instance state.
// Filters to instances that are routable (IsRoutable()) and, when req is non-nil and
// has a non-empty Model, further filters to instances serving that model (T044, T048).
// Complexity: O(N) per call where N = number of instances.
func buildRouterState(cs *ClusterSimulator, req *sim.Request) *sim.RouterState {
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

// liveSnapshot reads an instance's current state.
func liveSnapshot(inst *InstanceSimulator) sim.RoutingSnapshot {
	snap := sim.NewRoutingSnapshot(string(inst.ID()))
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
	cs.injectedByClass[e.request.SLOClass]++
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
