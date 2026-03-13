package cluster

import (
	"container/heap"
	"fmt"
	"math"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
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
		panic(fmt.Sprintf("PrefillRoutingEvent: no instances in prefill pool (poolMembership has %d entries)", len(cs.poolMembership)))
	}
	state := &sim.RouterState{Snapshots: filteredSnapshots, Clock: cs.clock}

	policy := cs.prefillRoutingPolicy
	if policy == nil {
		policy = cs.routingPolicy
	}
	decision := policy.Route(e.request, state)

	logrus.Debugf("[cluster] prefill req %s → instance %s", e.request.ID, decision.TargetInstance)

	e.request.AssignedInstance = decision.TargetInstance
	e.parentReq.PrefillInstanceID = decision.TargetInstance
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
			inst.InjectRequestOnline(e.request, e.time)
			// BC-PD-28: Notify observer after prefill routing so decider can learn prefix (R17, INV-7)
			cs.notifyDisaggregationObserver(e.request, decision.TargetInstance)
			return
		}
	}
	panic(fmt.Sprintf("PrefillRoutingEvent: invalid TargetInstance %q", decision.TargetInstance))
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
// When contention is enabled (--pd-transfer-contention), applies INV-P2-2:
// effective_bandwidth = total_bandwidth / active_transfers when active_transfers > 1;
// total_bandwidth (unchanged) when active_transfers == 1. Division by zero is impossible
// because activeTransfers is incremented to at least 1 before the bandwidth calculation.
func (e *KVTransferStartedEvent) Execute(cs *ClusterSimulator) {
	cs.transfersInitiated++
	e.parentReq.TransferStartTime = e.time

	// Contention tracking: increment BEFORE duration calculation so this
	// transfer is counted in its own fair-share divisor (INV-P2-2).
	if cs.config.PDTransferContention {
		cs.activeTransfers++
		if cs.activeTransfers > cs.peakConcurrentTransfers {
			cs.peakConcurrentTransfers = cs.activeTransfers
		}
		cs.transferDepthSum += int64(cs.activeTransfers)
		cs.transferStartCount++
	}

	// Transfer duration: base_latency_us + (numBlocks * blockSizeTokens * bytesPerToken) / bandwidthBytesPerUs
	numBlocks := e.parentReq.NumKVBlocks
	blockSizeBytes := cs.config.BlockSizeTokens * cs.config.PDKVBytesPerToken
	transferBytes := numBlocks * blockSizeBytes

	// INV-P2-2: apply fair-share divisor only when active_transfers > 1.
	// activeTransfers == 1 (this transfer alone) → full bandwidth, no division needed.
	effectiveBandwidthGBps := cs.config.PDTransferBandwidthGBps
	if cs.config.PDTransferContention && cs.activeTransfers > 1 {
		effectiveBandwidthGBps = cs.config.PDTransferBandwidthGBps / float64(cs.activeTransfers)
	}

	bandwidthBytesPerUs := effectiveBandwidthGBps * 1000.0 // GB/s → bytes/μs
	baseLatUs := cs.config.PDTransferBaseLatencyMs * 1000.0 // ms → μs

	var duration int64
	if bandwidthBytesPerUs > 0 {
		duration = int64(math.Ceil(baseLatUs + float64(transferBytes)/bandwidthBytesPerUs))
	} else {
		duration = int64(math.Ceil(baseLatUs))
	}
	if duration < 1 {
		duration = 1 // Minimum 1 μs transfer
	}

	logrus.Debugf("[cluster] KV transfer started for %s: %d blocks, duration=%d μs (active=%d)",
		e.parentReq.ID, numBlocks, duration, cs.activeTransfers)

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

	// Contention tracking: decrement after transfer completes (INV-P2-2).
	if cs.config.PDTransferContention {
		cs.activeTransfers--
		if cs.activeTransfers < 0 {
			logrus.Errorf("[cluster] activeTransfers went negative (%d) — KVTransferCompletedEvent fired without matching KVTransferStartedEvent (bookkeeping bug, R1)",
				cs.activeTransfers)
			cs.activeTransfers = 0
			// Mark corruption so Run() returns an error rather than delivering
			// silently incorrect contention metrics to the caller (Important #1).
			cs.contentionBookkeepingCorrupted = true
		}
	}

	orig := e.parentReq.OriginalRequest
	decodeSubReq := &sim.Request{
		ID:           e.parentReq.DecodeSubReqID,
		InputTokens:  orig.InputTokens,
		OutputTokens: orig.OutputTokens,
		State:        sim.StateQueued,
		ArrivalTime:  orig.ArrivalTime,
		TenantID:     orig.TenantID,
		SLOClass:     orig.SLOClass,
		Model:        orig.Model,
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
		panic(fmt.Sprintf("DecodeRoutingEvent: no instances in decode pool (poolMembership has %d entries)", len(cs.poolMembership)))
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
				// droppedAtDecodeKV is added to aggregatedMetrics.DroppedUnservable after Run().
				cs.droppedAtDecodeKV++
				// Mark parent CompletionTime so ParentRequests() doesn't contain records in
				// limbo (TransferCompleteTime set but CompletionTime = 0). The parent is
				// permanently dropped — set CompletionTime to the current event time so
				// post-run analysis can distinguish dropped-at-decode from in-flight.
				e.parentReq.CompletionTime = e.time
				return
			}

			// Set state after successful allocation (R5: no partial state on failure path)
			e.decodeSubReq.AssignedInstance = decision.TargetInstance
			e.parentReq.DecodeInstanceID = decision.TargetInstance
			e.parentReq.DecodeEnqueueTime = e.time

			// Record KV transfer and decode routing after successful KV allocation (BC-PD-17, BC-PD-19)
			// Placement after AllocateTransferredKV ensures records only exist for requests that
			// complete the decode phase (R1: no orphan records for dropped requests).
			// KVTransferRecord is recorded here so DecodeInstanceID is fully populated.
			// Both TransferStartTime and TransferCompleteTime were set in earlier event handlers.
			if cs.trace != nil {
				// INV-PD-4 guarantees transfer_start ≤ transfer_complete via timestamp sequencing:
				// KVTransferStartedEvent.Execute() schedules completion at T+duration where duration >= 1µs,
				// so the completion event always fires at a strictly later timestamp.
				// Defensive clamp: if the ordering invariant is ever violated, warn and record 0.
				// The warning makes INV-PD-4 violations detectable in operator logs (R1: no silent data loss).
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
					PrefillInstanceID: e.parentReq.PrefillInstanceID,
					DecodeInstanceID:  e.parentReq.DecodeInstanceID,
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
			// INV-PD-4: register decode sub-request for CompletionTime detection.
			cs.pendingDecodeCompletions[e.decodeSubReq.ID] = e.parentReq.ID
			inst.InjectDecodeOnline(e.decodeSubReq, e.time)
			// Observer not called: prefix was already recorded during PrefillRoutingEvent.
			// Decode sub-request has the same InputTokens, so re-notification is a no-op.
			return
		}
	}
	panic(fmt.Sprintf("DecodeRoutingEvent: invalid TargetInstance %q", decision.TargetInstance))
}
