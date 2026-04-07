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

// PrefillRoutingEvent routes a prefill sub-request to a prefill pool instance.
// Priority 4: after DisaggregationDecisionEvent (3), before KV transfer events.
type PrefillRoutingEvent struct {
	time      int64
	request   *sim.Request   // Prefill sub-request
	parentReq *ParentRequest
}

func (e *PrefillRoutingEvent) Timestamp() int64 { return e.time }
func (e *PrefillRoutingEvent) Priority() int     { return 4 }

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
			// DecodeRoutingEvent (decode), so one parent request consumes 2 capacity slots —
			// one on the prefill pool, one on the decode pool. IsOverBudget reflects actual
			// resource occupancy across both pools. OnComplete fires symmetrically via
			// OnRequestDone when each sub-request finishes.
			if cs.tenantTracker != nil {
				cs.tenantTracker.OnStart(e.request.TenantID)
			}
			inst.InjectRequestOnline(e.request, e.time)
			// Notify observer so stateful deciders (e.g., PrefixThresholdDecider) can learn
			// from this prefill routing decision (synchronous call -- cache is always current).
			cs.notifyDisaggregationObserver(e.request, decision.TargetInstance)
			return
		}
	}
	// R6: library panic on programming error — policy must return valid target
	panic(fmt.Sprintf("PrefillRoutingEvent: invalid TargetInstance %q returned by routing policy", decision.TargetInstance))
}

// KVTransferStartedEvent fires when a prefill sub-request completes.
// Records transfer initiation, computes duration, schedules completion.
// Priority 5: after prefill routing.
type KVTransferStartedEvent struct {
	time      int64
	parentReq *ParentRequest
}

func (e *KVTransferStartedEvent) Timestamp() int64 { return e.time }
func (e *KVTransferStartedEvent) Priority() int     { return 5 }

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
// Creates decode sub-request, schedules decode routing.
// Priority 6: after transfer start.
type KVTransferCompletedEvent struct {
	time      int64
	parentReq *ParentRequest
}

func (e *KVTransferCompletedEvent) Timestamp() int64 { return e.time }
func (e *KVTransferCompletedEvent) Priority() int     { return 6 }

// Execute creates the decode sub-request and schedules DecodeRoutingEvent.
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

	logrus.Debugf("[cluster] KV transfer completed for %s, scheduling decode routing", e.parentReq.ID)

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &DecodeRoutingEvent{
			time:         e.time,
			parentReq:    e.parentReq,
			decodeSubReq: decodeSubReq,
		},
		seqID: cs.nextSeqID(),
	})
}

// DecodeRoutingEvent routes a decode sub-request to a decode pool instance.
// Priority 7: after transfer completion.
type DecodeRoutingEvent struct {
	time         int64
	parentReq    *ParentRequest
	decodeSubReq *sim.Request
}

func (e *DecodeRoutingEvent) Timestamp() int64 { return e.time }
func (e *DecodeRoutingEvent) Priority() int     { return 7 }

// Execute routes the decode sub-request to a decode pool instance, pre-allocates KV, and injects.
func (e *DecodeRoutingEvent) Execute(cs *ClusterSimulator) {
	filteredSnapshots := cs.buildPoolFilteredSnapshots(PoolRoleDecode)
	if len(filteredSnapshots) == 0 {
		logrus.Warnf("[cluster] decode req %s: no routable instances in decode pool — request dropped", e.parentReq.ID)
		cs.droppedAtDecodeKV++
		e.parentReq.CompletionTime = e.time
		return
	}
	state := &sim.RouterState{Snapshots: filteredSnapshots, Clock: cs.clock}

	policy := cs.decodeRoutingPolicy
	if policy == nil {
		policy = cs.routingPolicy
	}
	decision := policy.Route(e.decodeSubReq, state)

	logrus.Debugf("[cluster] decode req %s → instance %s", e.decodeSubReq.ID, decision.TargetInstance)

	// Find target decode instance
	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			// Pre-allocate KV blocks for transferred input
			if ok := inst.AllocateTransferredKV(e.decodeSubReq); !ok {
				logrus.Warnf("[cluster] decode instance %s: insufficient KV capacity for %s (%d input tokens)",
					decision.TargetInstance, e.decodeSubReq.ID, len(e.decodeSubReq.InputTokens))
				// R1/INV-1: count the drop so aggregated DroppedUnservable remains accurate.
				cs.droppedAtDecodeKV++
				// Mark parent CompletionTime so ParentRequests() doesn't contain records in limbo.
				e.parentReq.CompletionTime = e.time
				return
			}

			// Set state after successful allocation (R5: no partial state on failure path)
			e.decodeSubReq.AssignedInstance = decision.TargetInstance
			e.parentReq.DecodeInstanceID = InstanceID(decision.TargetInstance)
			e.parentReq.DecodeEnqueueTime = e.time

			// INV-PD-1 structural guarantee: DecodeEnqueueTime >= TransferCompleteTime.
			// KVTransferCompletedEvent (priority 6) schedules DecodeRoutingEvent (priority 7)
			// at the same timestamp (e.time), so both fields are equal by construction.

			// Record KV transfer and decode routing after successful KV allocation (BC-PD-17, BC-PD-19).
			// Placement after AllocateTransferredKV ensures no KVTransferRecord or DecodeRoutingRecord
			// is written for a request that will not begin the decode phase.
			// KVTransferRecord is recorded here so DecodeInstanceID is fully populated.
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
					DecodeInstanceID:  string(e.parentReq.DecodeInstanceID),
				})
				decodeRecord := trace.DecodeRoutingRecord{
					ParentRequestID: e.parentReq.ID,
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

			cs.inFlightRequests[decision.TargetInstance]++
			// Phase 1B-2a: track decode slot for fair-share (see PrefillRoutingEvent comment
			// for PD slot-doubling semantics). OnStart placement here (after AllocateTransferredKV)
			// ensures balance: a failed KV allocation returns early above without calling OnStart,
			// matching the zero OnComplete calls for the dropped decode sub-request.
			if cs.tenantTracker != nil {
				cs.tenantTracker.OnStart(e.decodeSubReq.TenantID)
			}
			// Register decode sub-request so detectDecodeCompletions can stamp ParentRequest.CompletionTime
			// and read DecodeSubReq.State/ProgressIndex for timeout detection and context accumulation.
			e.parentReq.DecodeSubReq = e.decodeSubReq
			cs.pendingDecodeCompletions[e.decodeSubReq.ID] = e.parentReq.ID
			inst.InjectDecodeOnline(e.decodeSubReq, e.time)
			return
		}
	}
	panic(fmt.Sprintf("DecodeRoutingEvent: invalid TargetInstance %q", decision.TargetInstance))
}
