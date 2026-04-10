package cluster

import (
	"container/heap"
	"fmt"
	"math"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// DecodeRoutingEvent is the first step in the decode-first PD pipeline.
// Selects the decode instance, queries its cache state, and schedules
// DisaggregationDecisionEvent with the instance context.
// Priority 3: after admission (1) and standard routing (2), before disaggregation decision (4).
type DecodeRoutingEvent struct {
	time    int64
	request *sim.Request
}

func (e *DecodeRoutingEvent) Timestamp() int64 { return e.time }
func (e *DecodeRoutingEvent) Priority() int     { return 3 }

// Execute selects the decode instance, builds DecodeContext, and schedules
// DisaggregationDecisionEvent. Routing rejection increments routingRejections.
func (e *DecodeRoutingEvent) Execute(cs *ClusterSimulator) {
	filteredSnapshots := cs.buildPoolFilteredSnapshots(PoolRoleDecode)
	if len(filteredSnapshots) == 0 {
		logrus.Warnf("[cluster] req %s: no routable instances in decode pool — request rejected at decode routing", e.request.ID)
		cs.routingRejections++
		return
	}
	state := &sim.RouterState{Snapshots: filteredSnapshots, Clock: cs.clock}

	policy := cs.decodeRoutingPolicy
	if policy == nil {
		policy = cs.routingPolicy
	}
	decision := policy.Route(e.request, state)

	logrus.Debugf("[cluster] decode-first req %s → decode instance %s", e.request.ID, decision.TargetInstance)

	// Create parent request tracking record.
	parent := NewParentRequest(e.request, cs.config.BlockSizeTokens)
	parent.DecodeRouteDecisionTime = e.time
	parent.DecodeInstanceID = InstanceID(decision.TargetInstance)
	cs.parentRequests[parent.ID] = parent

	// Query cache state on selected decode instance (INV-PD-8).
	// Defaults to 0 (no cache hit) if the instance is not in cacheQueryFn (defensive).
	var cachedBlockCount int
	if queryFn, ok := cs.cacheQueryFn[decision.TargetInstance]; ok {
		cachedBlockCount = queryFn(e.request.InputTokens)
	}

	decodeCtx := sim.DecodeContext{
		InstanceID:       decision.TargetInstance,
		CachedBlockCount: cachedBlockCount,
	}

	// Record decode routing decision in trace (BC-PD-19).
	// Recorded here (decode-first) so the record is always present regardless of
	// the disaggregation outcome. Skip-path and disagg-path both produce a record.
	if cs.trace != nil {
		decodeRecord := trace.DecodeRoutingRecord{
			ParentRequestID: parent.ID,
			Clock:           cs.clock,
			ChosenInstance:  decision.TargetInstance,
			Scores:          copyScores(decision.Scores),
		}
		if cs.trace.Config.CounterfactualK > 0 {
			decodeRecord.Candidates, decodeRecord.Regret = computeCounterfactual(
				decision.TargetInstance, decision.Scores,
				filteredSnapshots, cs.trace.Config.CounterfactualK,
			)
		}
		cs.trace.RecordDecodeRouting(decodeRecord)
	}

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &DisaggregationDecisionEvent{
			time:          e.time,
			request:       e.request,
			parentReq:     parent,
			decodeContext: decodeCtx,
		},
		seqID: cs.nextSeqID(),
	})
}

// DisaggregationDecisionEvent decides whether a request should be disaggregated
// (sent to a dedicated prefill pool) or served locally on the already-selected decode instance.
// Priority 4: after DecodeRoutingEvent (3), before PrefillRoutingEvent/DecodeEnqueueEvent (5).
//
// In the decode-first flow, the decode instance has already been chosen by DecodeRoutingEvent
// before this event fires. decodeContext carries the instance's identity and cache state so
// the disaggregation decider can make a per-instance decision (INV-PD-8).
type DisaggregationDecisionEvent struct {
	time          int64
	request       *sim.Request
	parentReq     *ParentRequest
	decodeContext sim.DecodeContext
}

func (e *DisaggregationDecisionEvent) Timestamp() int64 { return e.time }
func (e *DisaggregationDecisionEvent) Priority() int     { return 4 }

// Execute calls the disaggregation decider with decode context and bifurcates:
//   - skip path (Disaggregate=false): DecodeEnqueueEvent at priority 5 (serves locally)
//   - disagg path (Disaggregate=true): PrefillRoutingEvent at priority 5
func (e *DisaggregationDecisionEvent) Execute(cs *ClusterSimulator) {
	decision := cs.disaggregationDecider.Decide(e.request, e.decodeContext)
	logrus.Debugf("[cluster] req %s: disaggregate=%v (decodeInstance=%s, cachedBlocks=%d)",
		e.request.ID, decision.Disaggregate, e.decodeContext.InstanceID, e.decodeContext.CachedBlockCount)

	e.parentReq.DisaggDecisionTime = e.time

	// Record disaggregation decision in trace (BC-PD-17).
	if cs.trace != nil {
		cs.trace.RecordDisaggregation(trace.DisaggregationRecord{
			RequestID:        e.request.ID,
			Clock:            cs.clock,
			Disaggregate:     decision.Disaggregate,
			DecodeInstanceID: e.decodeContext.InstanceID,
			CachedBlockCount: e.decodeContext.CachedBlockCount,
		})
	}

	if !decision.Disaggregate {
		// Skip path: serve directly on the already-selected decode instance.
		// InjectRequestOnline handles full prefill+decode on the instance.
		e.parentReq.SkippedDisaggregation = true
		heap.Push(&cs.clusterEvents, clusterEventEntry{
			event: &DecodeEnqueueEvent{
				time:      e.time,
				priority:  5,
				parentReq: e.parentReq,
				skipPath:  true,
			},
			seqID: cs.nextSeqID(),
		})
		return
	}

	// Disagg path: create prefill sub-request and route to prefill pool.
	// Output is intentionally nil: zero-output request completes at prefill end.
	prefillSubReq := &sim.Request{
		ID:           e.parentReq.PrefillSubReqID,
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
			parentReq: e.parentReq,
		},
		seqID: cs.nextSeqID(),
	})
}

// DecodeEnqueueEvent injects the request (or decode sub-request) onto the already-selected
// decode instance. Used in both the skip path (priority 5) and the disagg path (priority 8).
//
//   - skipPath=true (priority 5): injects original request via InjectRequestOnline;
//     instance handles full prefill+decode locally. The request is NOT tracked in
//     pendingDecodeCompletions — the instance's normal completion machinery handles it.
//   - skipPath=false (priority 8): creates decode sub-request, allocates transferred KV,
//     and injects via InjectDecodeOnline after a successful KV transfer.
type DecodeEnqueueEvent struct {
	time      int64
	priority  int // 5 for skip path, 8 for disagg path
	parentReq *ParentRequest
	skipPath  bool
}

func (e *DecodeEnqueueEvent) Timestamp() int64 { return e.time }
func (e *DecodeEnqueueEvent) Priority() int     { return e.priority }

// Execute injects the request onto the selected decode instance.
func (e *DecodeEnqueueEvent) Execute(cs *ClusterSimulator) {
	instID := string(e.parentReq.DecodeInstanceID)

	// Find the decode instance by ID (set by DecodeRoutingEvent).
	var targetInst *InstanceSimulator
	for _, inst := range cs.instances {
		if string(inst.ID()) == instID {
			targetInst = inst
			break
		}
	}
	if targetInst == nil {
		// Should never happen: DecodeRoutingEvent validates the target.
		panic(fmt.Sprintf("DecodeEnqueueEvent: decode instance %q not found (programming error)", instID))
	}

	if e.skipPath {
		e.executeSkipPath(cs, targetInst, instID)
	} else {
		e.executeDisaggPath(cs, targetInst, instID)
	}
}

// executeSkipPath injects the original request directly onto the decode instance.
// The instance processes prefill+decode locally (no KV transfer needed).
func (e *DecodeEnqueueEvent) executeSkipPath(cs *ClusterSimulator, inst *InstanceSimulator, instID string) {
	orig := e.parentReq.OriginalRequest
	orig.AssignedInstance = instID
	e.parentReq.DecodeEnqueueTime = e.time

	cs.inFlightRequests[instID]++
	if cs.tenantTracker != nil {
		cs.tenantTracker.OnStart(orig.TenantID)
	}

	logrus.Debugf("[cluster] skip-path req %s → decode instance %s", orig.ID, instID)
	inst.InjectRequestOnline(orig, e.time)
}

// executeDisaggPath creates the decode sub-request, allocates transferred KV, and injects.
func (e *DecodeEnqueueEvent) executeDisaggPath(cs *ClusterSimulator, inst *InstanceSimulator, instID string) {
	orig := e.parentReq.OriginalRequest
	decodeSubReq := &sim.Request{
		ID:                 e.parentReq.DecodeSubReqID,
		InputTokens:        orig.InputTokens,
		OutputTokens:       orig.OutputTokens,
		MaxOutputLen:       orig.MaxOutputLen,
		Deadline:           orig.Deadline,
		PrefixGroup:        orig.PrefixGroup,
		State:              sim.StateQueued,
		ArrivalTime:        orig.ArrivalTime,
		TenantID:           orig.TenantID,
		SLOClass:           orig.SLOClass,
		Model:              orig.Model,
		IsDecodeSubRequest: true,
	}

	// Pre-allocate KV blocks for the transferred input.
	if ok := inst.AllocateTransferredKV(decodeSubReq); !ok {
		logrus.Warnf("[cluster] decode instance %s: insufficient KV capacity for %s (%d input tokens)",
			instID, decodeSubReq.ID, len(decodeSubReq.InputTokens))
		// R1/INV-1: count the drop so aggregated DroppedUnservable remains accurate.
		cs.droppedAtDecodeKV++
		// Mark parent CompletionTime so ParentRequests() doesn't contain records in limbo.
		e.parentReq.CompletionTime = e.time
		return
	}

	// Set state after successful allocation (R5: no partial state on failure path).
	decodeSubReq.AssignedInstance = instID
	e.parentReq.DecodeEnqueueTime = e.time

	// INV-PD-1 structural guarantee: DecodeEnqueueTime >= TransferCompleteTime.
	// KVTransferCompletedEvent (priority 7) schedules DecodeEnqueueEvent (priority 8)
	// at the same timestamp, so both fields are equal by construction.

	// Record KV transfer after successful allocation (BC-PD-17).
	// Placement after AllocateTransferredKV ensures no KVTransferRecord is written for drops.
	if cs.trace != nil {
		// INV-PD-4: transfer_start ≤ transfer_complete by timestamp sequencing.
		// Defensive clamp: warn and record 0 if violated.
		transferDuration := e.parentReq.TransferCompleteTime - e.parentReq.TransferStartTime
		if transferDuration < 0 {
			logrus.Warnf("[cluster] INV-PD-4 violated: TransferCompleteTime (%d) < TransferStartTime (%d) for req %s; recording 0",
				e.parentReq.TransferCompleteTime, e.parentReq.TransferStartTime, e.parentReq.ID)
			transferDuration = 0
		}
		cs.trace.RecordKVTransfer(trace.KVTransferRecord{
			ParentRequestID:   e.parentReq.ID,
			TransferStartTime: e.parentReq.TransferStartTime,
			TransferDuration:  transferDuration,
			NumKVBlocks:       e.parentReq.NumKVBlocks,
			PrefillInstanceID: string(e.parentReq.PrefillInstanceID),
			DecodeInstanceID:  instID,
		})
	}

	cs.inFlightRequests[instID]++
	// Phase 1B-2a: track decode slot for fair-share (see PrefillRoutingEvent comment
	// for PD slot-doubling semantics). OnStart placement here (after AllocateTransferredKV)
	// ensures balance: a failed KV allocation returns early above without calling OnStart,
	// matching the zero OnComplete calls for the dropped decode sub-request.
	if cs.tenantTracker != nil {
		cs.tenantTracker.OnStart(decodeSubReq.TenantID)
	}
	// Register decode sub-request so detectDecodeCompletions can stamp ParentRequest.CompletionTime
	// and read DecodeSubReq.State/ProgressIndex for timeout detection and context accumulation.
	e.parentReq.DecodeSubReq = decodeSubReq
	cs.pendingDecodeCompletions[decodeSubReq.ID] = e.parentReq.ID
	logrus.Debugf("[cluster] disagg-path decode req %s → instance %s", decodeSubReq.ID, instID)
	inst.InjectDecodeOnline(decodeSubReq, e.time)
}

// PrefillRoutingEvent routes a prefill sub-request to a prefill pool instance.
// Priority 5: after DisaggregationDecisionEvent (4), before KV transfer events.
type PrefillRoutingEvent struct {
	time      int64
	request   *sim.Request   // Prefill sub-request
	parentReq *ParentRequest
}

func (e *PrefillRoutingEvent) Timestamp() int64 { return e.time }
func (e *PrefillRoutingEvent) Priority() int     { return 5 }

// Execute routes the prefill sub-request to a prefill pool instance using pool-filtered snapshots.
func (e *PrefillRoutingEvent) Execute(cs *ClusterSimulator) {
	filteredSnapshots := cs.buildPoolFilteredSnapshots(PoolRolePrefill)
	if len(filteredSnapshots) == 0 {
		logrus.Warnf("[cluster] prefill req %s: no routable instances in prefill pool — request rejected at routing", e.request.ID)
		cs.routingRejections++
		return
	}
	state := &sim.RouterState{Snapshots: filteredSnapshots, Clock: cs.clock}

	policy := cs.prefillRoutingPolicy
	if policy == nil {
		policy = cs.routingPolicy
	}
	decision := policy.Route(e.request, state)

	logrus.Debugf("[cluster] prefill req %s → instance %s", e.request.ID, decision.TargetInstance)

	e.request.AssignedInstance = decision.TargetInstance
	e.parentReq.PrefillInstanceID = InstanceID(decision.TargetInstance)
	e.parentReq.PrefillEnqueueTime = e.time

	// Record prefill routing decision if tracing is enabled (BC-PD-17, BC-PD-19)
	if cs.trace != nil {
		record := trace.PrefillRoutingRecord{
			ParentRequestID: e.parentReq.ID,
			Clock:           cs.clock,
			ChosenInstance:  decision.TargetInstance,
			Scores:          copyScores(decision.Scores),
		}
		if cs.trace.Config.CounterfactualK > 0 {
			record.Candidates, record.Regret = computeCounterfactual(
				decision.TargetInstance, decision.Scores,
				filteredSnapshots, cs.trace.Config.CounterfactualK,
			)
		}
		cs.trace.RecordPrefillRouting(record)
	}

	// Register as pending prefill completion for detection in event loop
	cs.pendingPrefillCompletions[e.request.ID] = e.parentReq.ID

	// Find target instance and inject
	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			cs.inFlightRequests[decision.TargetInstance]++
			// Phase 1B-2a: track tenant in-flight count for fair-share enforcement in PD mode.
			// PD disaggregation semantics: OnStart is called once here (prefill) and once in
			// DecodeEnqueueEvent (disagg path), so one parent request consumes 2 capacity slots —
			// one on the prefill pool, one on the decode pool. IsOverBudget reflects actual
			// resource occupancy across both pools. OnComplete fires symmetrically via
			// OnRequestDone when each sub-request finishes.
			if cs.tenantTracker != nil {
				cs.tenantTracker.OnStart(e.request.TenantID)
			}
			inst.InjectRequestOnline(e.request, e.time)
			return
		}
	}
	// R6: library panic on programming error — policy must return valid target
	panic(fmt.Sprintf("PrefillRoutingEvent: invalid TargetInstance %q returned by routing policy", decision.TargetInstance))
}

// KVTransferStartedEvent fires when a prefill sub-request completes.
// Records transfer initiation, computes duration, schedules completion.
// Priority 6: after prefill routing.
type KVTransferStartedEvent struct {
	time      int64
	parentReq *ParentRequest
}

func (e *KVTransferStartedEvent) Timestamp() int64 { return e.time }
func (e *KVTransferStartedEvent) Priority() int     { return 6 }

// Execute computes transfer duration and schedules KVTransferCompletedEvent.
func (e *KVTransferStartedEvent) Execute(cs *ClusterSimulator) {
	cs.transfersInitiated++
	e.parentReq.TransferStartTime = e.time

	// Contention tracking (INV-P2-2): increment before duration calculation so the
	// divisor reflects the active count including this transfer.
	if cs.config.PDTransferContention {
		cs.activeTransfers++
		if cs.activeTransfers > cs.peakConcurrentTransfers {
			cs.peakConcurrentTransfers = cs.activeTransfers
		}
		cs.transferDepthSum += int64(cs.activeTransfers)
		cs.transferStartCount++
	}

	// Transfer duration: base_latency_us + (numBlocks * blockSizeTokens * kvBytesPerToken) / effectiveBandwidthBytesPerUs
	// Derive per-GPU KV bytes per token from model config using the prefill pool's TP.
	kvBytesPerTokenF, err := latency.KVBytesPerToken(cs.config.ModelConfig, cs.config.EffectivePrefillTP())
	if err != nil {
		// Unreachable: NewClusterSimulator validates KVBytesPerToken at construction time
		// when PD disaggregation is enabled. If this fires, it indicates a missing
		// validation in the construction path.
		panic(fmt.Sprintf("unreachable: KVTransferStartedEvent: failed to derive KV bytes per token: %v", err))
	}

	numBlocks := e.parentReq.NumKVBlocks
	// Defer truncation: multiply float64 kvBytesPerToken by blockSize before converting,
	// matching the CalculateKVBlocks pattern to avoid precision loss for fractional
	// BytesPerParam (e.g., INT4=0.5 at high TP).
	blockSizeBytesF := float64(cs.config.BlockSizeTokens) * kvBytesPerTokenF
	transferBytes := float64(numBlocks) * blockSizeBytesF

	bandwidthBytesPerUs := cs.config.PDTransferBandwidthGBps * 1000.0 // GB/s → bytes/μs
	baseLatUs := cs.config.PDTransferBaseLatencyMs * 1000.0            // ms → μs

	// Fair-share: divide effective bandwidth by number of concurrent transfers (INV-P2-2)
	if cs.config.PDTransferContention && cs.activeTransfers > 1 {
		bandwidthBytesPerUs = bandwidthBytesPerUs / float64(cs.activeTransfers)
	}

	var duration int64
	if bandwidthBytesPerUs > 0 {
		duration = int64(math.Ceil(baseLatUs + transferBytes/bandwidthBytesPerUs))
	} else {
		duration = int64(math.Ceil(baseLatUs))
	}
	if duration < 1 {
		duration = 1 // Minimum 1 μs transfer
	}

	if cs.config.PDTransferContention {
		logrus.Debugf("[cluster] KV transfer started for %s: %d blocks, duration=%d μs, activeTransfers=%d",
			e.parentReq.ID, numBlocks, duration, cs.activeTransfers)
	} else {
		logrus.Debugf("[cluster] KV transfer started for %s: %d blocks, duration=%d μs",
			e.parentReq.ID, numBlocks, duration)
	}

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &KVTransferCompletedEvent{
			time:      e.time + duration,
			parentReq: e.parentReq,
		},
		seqID: cs.nextSeqID(),
	})
}

// KVTransferCompletedEvent fires after transfer duration elapses.
// Schedules DecodeEnqueueEvent (disagg path, priority 8).
// Priority 7: after transfer start.
// The decode instance was already selected by DecodeRoutingEvent (priority 3).
type KVTransferCompletedEvent struct {
	time      int64
	parentReq *ParentRequest
}

func (e *KVTransferCompletedEvent) Timestamp() int64 { return e.time }
func (e *KVTransferCompletedEvent) Priority() int     { return 7 }

// Execute records transfer completion and schedules DecodeEnqueueEvent (disagg path).
func (e *KVTransferCompletedEvent) Execute(cs *ClusterSimulator) {
	cs.transfersCompleted++
	e.parentReq.TransferCompleteTime = e.time

	// Contention decrement with negative guard (INV-P2-2).
	// activeTransfers going negative is a hard bookkeeping bug (unlike inFlightRequests,
	// which is warn-only because it recovers from delta mis-accounting). Here we set
	// contentionBookkeepingCorrupted so Run() returns an error — contention metrics are
	// meaningless once the counter is corrupted.
	if cs.config.PDTransferContention {
		cs.activeTransfers--
		if cs.activeTransfers < 0 {
			logrus.Errorf("[cluster] activeTransfers went negative (%d) for %s — resetting to 0; contention bookkeeping corrupted",
				cs.activeTransfers, e.parentReq.ID)
			cs.activeTransfers = 0
			cs.contentionBookkeepingCorrupted = true
		}
	}

	logrus.Debugf("[cluster] KV transfer completed for %s, scheduling decode enqueue on instance %s",
		e.parentReq.ID, e.parentReq.DecodeInstanceID)

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &DecodeEnqueueEvent{
			time:      e.time,
			priority:  8,
			parentReq: e.parentReq,
			skipPath:  false,
		},
		seqID: cs.nextSeqID(),
	})
}
