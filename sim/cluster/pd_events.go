package cluster

import (
	"container/heap"
	"fmt"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// PrefillRoutingEvent injects a prefill sub-request on the prefill instance the gateway chose.
// Priority 4: before KV transfer events.
type PrefillRoutingEvent struct {
	time      int64
	request   *sim.Request // Prefill sub-request
	parentReq *ParentRequest
}

func (e *PrefillRoutingEvent) Timestamp() int64 { return e.time }
func (e *PrefillRoutingEvent) Priority() int    { return 4 }

// Execute injects the prefill sub-request on the prefill instance the gateway chose.
func (e *PrefillRoutingEvent) Execute(cs *ClusterSimulator) {
	target := string(e.parentReq.PrefillInstanceID)
	logrus.Debugf("[cluster] prefill req %s → instance %s", e.request.ID, target)

	e.request.AssignedInstance = target
	e.parentReq.PrefillEnqueueTime = e.time

	if cs.trace != nil {
		cs.trace.RecordPrefillRouting(trace.PrefillRoutingRecord{
			ParentRequestID: e.parentReq.ID,
			Clock:           cs.clock,
			ChosenInstance:  target,
		})
	}

	// Register as pending prefill completion for detection in event loop
	cs.pendingPrefillCompletions[e.request.ID] = e.parentReq.ID

	inst := cs.instanceByID(InstanceID(target))
	if inst == nil {
		panic(fmt.Sprintf("PrefillRoutingEvent: gateway chose unknown prefill instance %q", target))
	}
	cs.inFlightRequests[target]++
	inst.InjectRequestOnline(e.request, e.time)
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
		Phase:              orig.Phase,
		Adapter:            orig.Adapter,
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
	// programming error (the gateway chose an ID not in cs.instances);
	// a non-routable instance is a legitimate drain/terminate event during
	// the transfer window.
	if decodeInst == nil {
		logrus.Errorf("[cluster] decode instance %q for %s not found in cs.instances — routing invariant violated; dropping request",
			decodeInstID, e.parentReq.ID)
		e.dropAtStart(cs, NIXLDecodeNeverReceived)
		return
	}
	if !decodeInst.IsRoutable() {
		logrus.Warnf("[cluster] decode instance %s for %s is no longer routable at transfer start — request dropped",
			decodeInstID, e.parentReq.ID)
		// The decode engine never received the request (NIXLReleaseOnDrop decides what that
		// means for the prefill side's held blocks).
		e.dropAtStart(cs, NIXLDecodeNeverReceived)
		return
	}
	// Whether the decode engine can ever take the prompt is the connector's rule
	// (NIXLDecodeAccepts); one it cannot is rejected.
	if !NIXLDecodeAccepts(e.parentReq.NumKVBlocks, decodeInst.TotalKVBlocks()) {
		logrus.Warnf("[cluster] decode instance %s: %s needs %d KV blocks, more than its %d — request dropped",
			decodeInstID, decodeSubReq.ID, e.parentReq.NumKVBlocks, decodeInst.TotalKVBlocks())
		e.dropAtStart(cs, NIXLDecodeRejected)
		return
	}
	// The decode engine holds the handoff among its waiting handoffs and starts its pull by its
	// own admission rule (INV-PD-9): startPull prices it when it starts.
	if cs.remotePulls == nil {
		cs.remotePulls = make(map[string]*ParentRequest)
	}
	cs.remotePulls[decodeSubReq.ID] = e.parentReq
	decodeInst.AddRemote(decodeSubReq, e.time)
}

// startPull starts the transfer of a handoff whose pull an engine driven through engine.Engine
// admitted at now, moving pulledTokens of its prompt, as serveDecodeKVWaiters does for the
// compatibility engine's host loop.
func (cs *ClusterSimulator) startPull(req *sim.Request, pulledTokens, now int64) {
	parent := cs.remotePulls[req.ID]
	if parent == nil {
		panic(fmt.Sprintf("startPull: %s is not a handoff waiting for its pull", req.ID))
	}
	delete(cs.remotePulls, req.ID)
	if inst := cs.instanceByID(parent.DecodeInstanceID); inst != nil && !inst.IsRoutable() {
		// A decode instance that went unroutable (drained) while the handoff waited drops its
		// waiters rather than starting their pulls: its engine had the request, so the prefill
		// side frees when the lease, renewed until now, runs out (NIXLDecodeVanished), as the
		// compatibility host loop's serveDecodeKVWaiters dropped them. The pull the engine just
		// admitted is failed, freeing what it reserved.
		inst.ReleaseReservedKV(req, now)
		(&KVTransferStartedEvent{time: now, parentReq: parent}).dropAtStart(cs, NIXLDecodeVanished)
		return
	}
	parent.DecodeSubReq = req
	parent.PulledKVBlocks = (pulledTokens + cs.config.BlockSizeTokens - 1) / cs.config.BlockSizeTokens
	scheduleTransferCompletion(cs, parent, now)
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
//
// How the handoff ended (outcome) decides when the prefill side frees its held KV: the
// connector's rule (NIXLReleaseOnDrop, INV-PD-7); the cluster releases the hold or
// schedules its lease expiry accordingly (releasePrefillOnDrop).
func (e *KVTransferStartedEvent) dropAtStart(cs *ClusterSimulator, outcome NIXLOutcome) {
	cs.releasePrefillOnDrop(e.parentReq, outcome, e.time)
	cs.droppedAtDecodeKV++
	cs.recordUnserved(e.parentReq.OriginalRequest)
	cs.parentDone(e.parentReq, e.time, DoneDropped)
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

// scheduleTransferCompletion performs the duration calculation and
// KVTransferCompletedEvent scheduling half of the
// transfer-start pipeline. Factored out of Execute so that narrow unit
// tests can exercise the duration formula without constructing full
// decode-side reservation fixtures. Production callers always reach
// this via KVTransferStartedEvent.Execute after a successful decode-side
// KV reservation.
func scheduleTransferCompletion(cs *ClusterSimulator, parentReq *ParentRequest, startTime int64) {
	cs.transfersInitiated++
	parentReq.TransferStartTime = startTime

	// The latency backend prices the handoff (NewClusterSimulator refuses a PD deployment
	// without one). The KV moves in whole blocks, so the token count priced is the blocks'
	// capacity. Without contention every handoff takes its price, as if alone on the fabric;
	// with it, the handoff's demand joins the link arbiter, which decides when it completes.
	numBlocks := parentReq.PulledKVBlocks
	tokens := numBlocks * cs.config.BlockSizeTokens
	duration := max(1, cs.config.PDTransferTime(tokens, parentReq.PrefillInstanceID, parentReq.DecodeInstanceID))
	parentReq.TransferAloneTicks = duration
	if cs.pdLinks != nil {
		flow := cs.config.PDTransferDemand(tokens, parentReq.PrefillInstanceID, parentReq.DecodeInstanceID)
		if alone := AloneTicks(flow); alone != duration {
			// blis-schemas kernel.PDTransferDemand: alone, a demand costs exactly PDTransferTime.
			panic(fmt.Sprintf("KV handoff of %s: its demand costs %d ticks alone but its price is %d; "+
				"the latency backend breaks the PDTransferTime == PDTransferDemand.AloneTime law",
				parentReq.ID, alone, duration))
		}
		cs.pdLinkFlows[cs.pdLinks.Start(startTime, flow)] = parentReq
		logrus.Debugf("[cluster] KV transfer started for %s at %d: %d blocks, %d μs alone, sharing links with %d others",
			parentReq.ID, startTime, numBlocks, duration, cs.pdLinks.InFlight()-1)
		cs.scheduleLinkWake()
		return
	}
	logrus.Debugf("[cluster] KV transfer started for %s: %d blocks, duration=%d μs (priced by the latency backend)",
		parentReq.ID, numBlocks, duration)
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &KVTransferCompletedEvent{time: startTime + duration, parentReq: parentReq},
		seqID: cs.nextSeqID(),
	})
}

// scheduleLinkWake schedules the link arbiter's next drain or completion, superseding any wake
// scheduled before: a handoff starting can move every other handoff's completion.
func (cs *ClusterSimulator) scheduleLinkWake() {
	at, ok := cs.pdLinks.NextWake()
	if !ok {
		return
	}
	cs.pdLinkWakeGen++
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &PDLinkWakeEvent{time: at, gen: cs.pdLinkWakeGen},
		seqID: cs.nextSeqID(),
	})
}

// PDLinkWakeEvent advances the link arbiter to its next drain or completion and completes the
// handoffs that finished, in the order they started. Only the latest scheduled wake acts.
// Priority 6, the completion it stands for.
type PDLinkWakeEvent struct {
	time int64
	gen  int64
}

func (e *PDLinkWakeEvent) Timestamp() int64 { return e.time }
func (e *PDLinkWakeEvent) Priority() int    { return 6 }

// Execute completes the finished handoffs through KVTransferCompletedEvent at this tick, so a
// contended handoff completes by exactly the path an uncontended one does.
func (e *PDLinkWakeEvent) Execute(cs *ClusterSimulator) {
	if e.gen != cs.pdLinkWakeGen {
		return
	}
	for _, id := range cs.pdLinks.Advance(e.time) {
		parent := cs.pdLinkFlows[id]
		delete(cs.pdLinkFlows, id)
		heap.Push(&cs.clusterEvents, clusterEventEntry{
			event: &KVTransferCompletedEvent{time: e.time, parentReq: parent},
			seqID: cs.nextSeqID(),
		})
	}
	cs.scheduleLinkWake()
}

// PDLinkPeakConcurrent is, per link, the most handoffs ever in flight on it together; nil when
// contention is off.
func (cs *ClusterSimulator) PDLinkPeakConcurrent() map[string]int {
	if cs.pdLinks == nil {
		return nil
	}
	return cs.pdLinks.PeakConcurrent()
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
// CompletionTime stamped), so this branch only
// bumps transfersCompleted and returns without touching KV or state.
func (e *KVTransferCompletedEvent) Execute(cs *ClusterSimulator) {
	cs.transfersCompleted++

	// Degenerate completion for drop-at-transfer-start (issue #1343).
	// dropAtStart did not call scheduleTransferCompletion, so there is no
	// transfer to complete. TransferCompleteTime intentionally stays 0 — downstream code
	// uses "TransferCompleteTime == 0 && TransferStartTime > 0" to
	// distinguish a start-drop from a completed (or late-dropped) transfer.
	if e.parentReq.DecodeSubReq == nil {
		return
	}
	decodeSubReq := e.parentReq.DecodeSubReq
	e.parentReq.TransferCompleteTime = e.time

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
		cs.recordUnserved(e.parentReq.OriginalRequest)
		cs.parentDone(e.parentReq, e.time, DoneDropped)
		e.parentReq.CompletionTime = e.time
		e.parentReq.DecodeSubReq = nil
		cs.releasePrefillOnDrop(e.parentReq, NIXLDecodeVanished, e.time)
		return
	}
	if !decodeInst.IsRoutable() {
		// The decode pod became non-routable mid-transfer (e.g., drained or
		// terminated). Release reserved blocks and drop.
		logrus.Warnf("[cluster] decode instance %s for %s is no longer routable at transfer complete — releasing reserved KV and dropping",
			decodeInstID, e.parentReq.ID)
		decodeInst.ReleaseReservedKV(decodeSubReq, e.time)
		cs.droppedAtDecodeKV++
		cs.recordUnserved(e.parentReq.OriginalRequest)
		cs.parentDone(e.parentReq, e.time, DoneDropped)
		e.parentReq.CompletionTime = e.time
		e.parentReq.DecodeSubReq = nil
		// A decode side that went away mid-pull (NIXLReleaseOnDrop decides the release).
		cs.releasePrefillOnDrop(e.parentReq, NIXLDecodeVanished, e.time)
		return
	}
	// The pull is complete: what that does on each side is the connector's rule
	// (NIXLOnPullComplete). The cluster frees the hold it keeps; the decode engine caches what
	// it received when the pull lands (InjectDecodeOnline, its TransferDone).
	if NIXLOnPullComplete().FreePrefillHold {
		cs.releasePrefillHold(e.parentReq)
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
			PulledKVBlocks:    e.parentReq.PulledKVBlocks,
			PrefillInstanceID: string(e.parentReq.PrefillInstanceID),
			DecodeInstanceID:  decodeInstID,
		})
	}

	cs.inFlightRequests[decodeInstID]++
	// Register decode sub-request so detectDecodeCompletions can stamp
	// ParentRequest.CompletionTime and read DecodeSubReq.State/ProgressIndex
	// for timeout detection and context accumulation.
	cs.pendingDecodeCompletions[decodeSubReq.ID] = e.parentReq.ID
	decodeInst.InjectDecodeOnline(decodeSubReq, e.time)
}
