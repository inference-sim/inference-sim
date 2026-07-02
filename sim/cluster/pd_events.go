package cluster

import (
	"container/heap"
	"fmt"
	"math"

	"github.com/sirupsen/logrus"

	"blis/sim"
	"blis/sim/latency"
	"blis/sim/trace"
)

// PrefillRoutingEvent routes a prefill sub-request to a prefill pool instance.
// Priority 4: after RoutingDecisionEvent (2), before KV transfer events.
type PrefillRoutingEvent struct {
	time      int64
	request   *sim.Request // Prefill sub-request
	parentReq *ParentRequest
}

func (e *PrefillRoutingEvent) Timestamp() int64 { return e.time }
func (e *PrefillRoutingEvent) Priority() int    { return 4 }

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
			// KVTransferCompletedEvent (decode), so one parent request consumes 2 capacity slots —
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
	// The routing policy contract requires Route to return a TargetInstance from the
	// provided snapshot set; a panic here indicates a policy implementation bug.
	panic(fmt.Sprintf("PrefillRoutingEvent: invalid TargetInstance %q returned by routing policy", decision.TargetInstance))
}

// KVTransferStartedEvent fires when a prefill sub-request completes.
// Records transfer initiation, computes duration, reserves KV blocks on the
// decode instance (issue #1343 vLLM WAITING_FOR_REMOTE_KVS parity), and
// schedules completion. Priority 5: after prefill routing.
type KVTransferStartedEvent struct {
	time      int64
	parentReq *ParentRequest
}

func (e *KVTransferStartedEvent) Timestamp() int64 { return e.time }
func (e *KVTransferStartedEvent) Priority() int    { return 5 }

// Execute reserves KV blocks on the decode instance in a
// WaitingForRemoteKVs sub-request, then schedules
// KVTransferCompletedEvent via scheduleTransferCompletion. If the decode
// pod has become non-routable or lacks capacity for the reservation, the
// request is dropped immediately and a degenerate completion event is
// scheduled at the same tick so INV-PD-3 (initiated == completed) still
// holds and in-flight accounting stays correct. This mirrors vLLM v1
// scheduler behavior where decode-side block allocation happens as the
// transfer begins (permalink 1, permalink 3 in issue #1343), so reserved
// blocks reduce available decode-pod KV capacity for the transfer window
// rather than only at transfer completion.
func (e *KVTransferStartedEvent) Execute(cs *ClusterSimulator) {
	// OriginalRequest is always set by NewParentRequest in production.
	// Narrow unit tests that exercise the duration formula in isolation
	// bypass this Execute() method and call scheduleTransferCompletion
	// directly to avoid constructing the full decode-side fixtures — so a
	// nil here indicates a programming error in the event pipeline (parent
	// created without an OriginalRequest), not a recoverable runtime
	// condition.
	if e.parentReq.OriginalRequest == nil {
		panic(fmt.Sprintf("KVTransferStartedEvent: nil OriginalRequest for parent %s — NewParentRequest should have attached it",
			e.parentReq.ID))
	}
	orig := e.parentReq.OriginalRequest

	// InputTokens and OutputTokens are slice-header aliases of orig's fields
	// (#1445) — the sub-request views the same underlying token buffer, no
	// flatten. If Request.InputTokens ever becomes lazy/chained, this site must
	// update.
	decodeSubReq := &sim.Request{
		ID:                 e.parentReq.DecodeSubReqID,
		InputTokens:        orig.InputTokens,
		OutputTokens:       orig.OutputTokens,
		MaxOutputLen:       orig.MaxOutputLen,
		Deadline:           orig.Deadline,
		PrefixGroup:        orig.PrefixGroup,
		State:              sim.StateWaitingForRemoteKVs,
		ArrivalTime:        orig.ArrivalTime,
		TenantID:           orig.TenantID,
		SLOClass:           orig.SLOClass,
		Model:              orig.Model,
		IsDecodeSubRequest: true,
	}

	decodeInstID := string(e.parentReq.DecodeInstanceID)
	var decodeInst *InstanceSimulator
	for _, inst := range cs.instances {
		if string(inst.ID()) == decodeInstID {
			decodeInst = inst
			break
		}
	}

	// Split diagnostics for the two failure modes: a nil instance is a
	// programming error (routing policy returned an ID not in cs.instances);
	// a non-routable instance is a legitimate drain/terminate event during
	// the transfer window.
	if decodeInst == nil {
		logrus.Errorf("[cluster] decode instance %q for %s not found in cs.instances — routing invariant violated; dropping request",
			decodeInstID, e.parentReq.ID)
		e.dropAtStart(cs)
		return
	}
	if !decodeInst.IsRoutable() {
		logrus.Warnf("[cluster] decode instance %s for %s is no longer routable at transfer start — request dropped",
			decodeInstID, e.parentReq.ID)
		e.dropAtStart(cs)
		return
	}

	if ok := decodeInst.ReserveTransferredKV(decodeSubReq); !ok {
		logrus.Warnf("[cluster] decode instance %s: insufficient KV capacity for %s (%d input tokens) at transfer start",
			decodeInstID, decodeSubReq.ID, decodeSubReq.InputLen())
		e.dropAtStart(cs)
		return
	}

	// Reservation succeeded: attach the sub-request to the parent. The
	// KVTransferCompletedEvent promotes it to StateQueued and enqueues
	// without re-allocating KV.
	e.parentReq.DecodeSubReq = decodeSubReq
	scheduleTransferCompletion(cs, e.parentReq, e.time)
}

// dropAtStart records a drop-at-transfer-start outcome: decode pod
// unroutable or reservation failed. The parent's TransferStartTime and
// CompletionTime are stamped at this tick (no TransferCompleteTime is
// set, so post-sim detection can distinguish a start-drop from a
// late-drop). Increments cs.droppedAtDecodeKV for INV-1 accounting
// (flowed into DroppedUnservable at finalization).
//
// Also increments cs.transfersInitiated so the counter's original
// "attempts" semantics — every request reaching KVTransferStartedEvent
// — is preserved across the reservation-at-start change introduced in
// issue #1343. INV-PD-3 (initiated == completed) is maintained by
// scheduling a degenerate KVTransferCompletedEvent at the same tick;
// that event detects the drop via DecodeSubReq == nil and returns after
// incrementing transfersCompleted, without attempting to release KV
// (nothing was reserved) or promote state.
func (e *KVTransferStartedEvent) dropAtStart(cs *ClusterSimulator) {
	cs.droppedAtDecodeKV++
	cs.transfersInitiated++
	e.parentReq.TransferStartTime = e.time
	e.parentReq.CompletionTime = e.time
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &KVTransferCompletedEvent{
			time:      e.time,
			parentReq: e.parentReq,
		},
		seqID: cs.nextSeqID(),
	})
}

// scheduleTransferCompletion performs the contention-tracking, duration
// calculation, and KVTransferCompletedEvent scheduling half of the
// transfer-start pipeline. Factored out of Execute so that narrow unit
// tests can exercise the duration formula without constructing full
// decode-side reservation fixtures. Production callers always reach
// this via KVTransferStartedEvent.Execute after a successful decode-side
// KV reservation.
func scheduleTransferCompletion(cs *ClusterSimulator, parentReq *ParentRequest, startTime int64) {
	cs.transfersInitiated++
	parentReq.TransferStartTime = startTime

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
		panic(fmt.Sprintf("unreachable: scheduleTransferCompletion: failed to derive KV bytes per token: %v", err))
	}

	numBlocks := parentReq.NumKVBlocks
	// Defer truncation: multiply float64 kvBytesPerToken by blockSize before converting,
	// matching the CalculateKVBlocks pattern to avoid precision loss for fractional
	// BytesPerParam (e.g., INT4=0.5 at high TP).
	blockSizeBytesF := float64(cs.config.BlockSizeTokens) * kvBytesPerTokenF
	transferBytes := float64(numBlocks) * blockSizeBytesF

	bandwidthBytesPerUs := cs.config.PDTransferBandwidthGBps * 1000.0 // GB/s → bytes/μs
	baseLatUs := cs.config.PDTransferBaseLatencyMs * 1000.0           // ms → μs

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
			parentReq.ID, numBlocks, duration, cs.activeTransfers)
	} else {
		logrus.Debugf("[cluster] KV transfer started for %s: %d blocks, duration=%d μs",
			parentReq.ID, numBlocks, duration)
	}

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &KVTransferCompletedEvent{
			time:      startTime + duration,
			parentReq: parentReq,
		},
		seqID: cs.nextSeqID(),
	})
}

// KVTransferCompletedEvent fires after transfer duration elapses.
// Promotes the pre-reserved decode sub-request (created and KV-reserved
// by KVTransferStartedEvent, per issue #1343 vLLM WAITING_FOR_REMOTE_KVS
// parity) from StateWaitingForRemoteKVs to StateQueued and enqueues it on
// the decode instance. No KV (re-)allocation happens here — blocks are
// already held by the request.
//
// Priority 6: after transfer start.
type KVTransferCompletedEvent struct {
	time      int64
	parentReq *ParentRequest
}

func (e *KVTransferCompletedEvent) Timestamp() int64 { return e.time }
func (e *KVTransferCompletedEvent) Priority() int    { return 6 }

// Execute promotes the pre-reserved decode sub-request to StateQueued and
// injects it into the decode instance. If the decode instance has
// transitioned to a non-routable state during the transfer window, the
// reserved KV blocks are released and the request is dropped (counted as
// droppedAtDecodeKV). KV was reserved at KVTransferStartedEvent.
//
// Degenerate path (issue #1343): when the parent's DecodeSubReq is nil,
// this event was scheduled by KVTransferStartedEvent.dropAtStart as a
// zero-duration "completion" so INV-PD-3 (initiated == completed) holds
// across the reservation-failed drop. In that case bookkeeping was
// already done at transfer-start time (droppedAtDecodeKV++,
// CompletionTime stamped, no contention increment), so this branch only
// bumps transfersCompleted and returns without touching KV or state.
func (e *KVTransferCompletedEvent) Execute(cs *ClusterSimulator) {
	cs.transfersCompleted++

	// Degenerate completion for drop-at-transfer-start (issue #1343).
	// dropAtStart did not call scheduleTransferCompletion, so activeTransfers
	// was never incremented for this transfer and must not be decremented
	// here. TransferCompleteTime intentionally stays 0 — downstream code
	// uses "TransferCompleteTime == 0 && TransferStartTime > 0" to
	// distinguish a start-drop from a completed (or late-dropped) transfer.
	if e.parentReq.DecodeSubReq == nil {
		return
	}
	decodeSubReq := e.parentReq.DecodeSubReq
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

	decodeInstID := string(e.parentReq.DecodeInstanceID)
	logrus.Debugf("[cluster] KV transfer completed for %s, promoting decode sub-req on decode pod %s",
		e.parentReq.ID, decodeInstID)

	var decodeInst *InstanceSimulator
	for _, inst := range cs.instances {
		if string(inst.ID()) == decodeInstID {
			decodeInst = inst
			break
		}
	}

	// Split diagnostics: nil instance = programming error (routing returned
	// an ID not in cs.instances); non-routable = legitimate drain/terminate
	// during the transfer window. Both paths release KV (when the instance
	// exists) and drop.
	if decodeInst == nil {
		logrus.Errorf("[cluster] decode instance %q for %s not found in cs.instances at transfer complete — routing invariant violated; KV cannot be released",
			decodeInstID, e.parentReq.ID)
		cs.droppedAtDecodeKV++
		e.parentReq.CompletionTime = e.time
		e.parentReq.DecodeSubReq = nil
		return
	}
	if !decodeInst.IsRoutable() {
		// The decode pod became non-routable mid-transfer (e.g., drained or
		// terminated). Release reserved blocks and drop.
		logrus.Warnf("[cluster] decode instance %s for %s is no longer routable at transfer complete — releasing reserved KV and dropping",
			decodeInstID, e.parentReq.ID)
		decodeInst.ReleaseReservedKV(decodeSubReq)
		cs.droppedAtDecodeKV++
		e.parentReq.CompletionTime = e.time
		e.parentReq.DecodeSubReq = nil
		return
	}

	// Promote StateWaitingForRemoteKVs → StateQueued. KV blocks remain
	// allocated; EnqueueDecodeSubRequest (via InjectDecodeOnline) does not
	// re-allocate.
	decodeSubReq.State = sim.StateQueued
	decodeSubReq.AssignedInstance = decodeInstID
	e.parentReq.DecodeEnqueueTime = e.time

	// INV-PD-1 structural guarantee: DecodeEnqueueTime >= TransferCompleteTime.
	// Both TransferCompleteTime and DecodeEnqueueTime are set in this event at the same tick.

	// Record KV transfer after successful promotion (BC-PD-17).
	// Placement here ensures no KVTransferRecord is written for a request
	// whose decode pod became non-routable during the transfer window.
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
			DecodeInstanceID:  decodeInstID,
		})
	}

	cs.inFlightRequests[decodeInstID]++
	// Phase 1B-2a: track decode slot for fair-share (see PrefillRoutingEvent
	// comment for PD slot-doubling semantics). OnStart placement here (after
	// routability re-check) ensures balance: a failed late-drop releases KV
	// without calling OnStart, matching the zero OnComplete calls for the
	// dropped decode sub-request.
	if cs.tenantTracker != nil {
		cs.tenantTracker.OnStart(decodeSubReq.TenantID)
	}
	// Register decode sub-request so detectDecodeCompletions can stamp
	// ParentRequest.CompletionTime and read DecodeSubReq.State/ProgressIndex
	// for timeout detection and context accumulation.
	cs.pendingDecodeCompletions[decodeSubReq.ID] = e.parentReq.ID
	decodeInst.InjectDecodeOnline(decodeSubReq, e.time)
}
