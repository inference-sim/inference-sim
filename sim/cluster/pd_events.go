package cluster

import (
	"container/heap"
	"fmt"
	"math"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
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
	e.parentReq.PrefillInstanceID = InstanceID(decision.TargetInstance)
	e.parentReq.PrefillEnqueueTime = e.time

	// Register as pending prefill completion for detection in event loop
	cs.pendingPrefillCompletions[e.request.ID] = e.parentReq.ID

	// Find target instance and inject
	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			cs.inFlightRequests[decision.TargetInstance]++
			inst.InjectRequestOnline(e.request, e.time)
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

	// Transfer duration: base_latency_us + (numBlocks * blockSizeTokens * bytesPerToken) / bandwidthBytesPerUs
	numBlocks := e.parentReq.NumKVBlocks
	blockSizeBytes := cs.config.BlockSizeTokens * cs.config.PDKVBytesPerToken
	transferBytes := numBlocks * blockSizeBytes

	bandwidthBytesPerUs := cs.config.PDTransferBandwidthGBps * 1000.0 // GB/s → bytes/μs
	baseLatUs := cs.config.PDTransferBaseLatencyMs * 1000.0            // ms → μs

	var duration int64
	if bandwidthBytesPerUs > 0 {
		duration = int64(math.Ceil(baseLatUs + float64(transferBytes)/bandwidthBytesPerUs))
	} else {
		duration = int64(math.Ceil(baseLatUs))
	}
	if duration < 1 {
		duration = 1 // Minimum 1 μs transfer
	}

	logrus.Debugf("[cluster] KV transfer started for %s: %d blocks, duration=%d μs",
		e.parentReq.ID, numBlocks, duration)

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
				cs.droppedAtDecodeKV++
				// Mark parent CompletionTime so ParentRequests() doesn't contain records in limbo.
				e.parentReq.CompletionTime = e.time
				return
			}

			// Set state after successful allocation (R5: no partial state on failure path)
			e.decodeSubReq.AssignedInstance = decision.TargetInstance
			e.parentReq.DecodeInstanceID = InstanceID(decision.TargetInstance)
			e.parentReq.DecodeEnqueueTime = e.time

			// INV-PD-1 defensive check: decode_enqueue_time >= transfer_complete_time.
			if e.parentReq.DecodeEnqueueTime < e.parentReq.TransferCompleteTime {
				logrus.Warnf("[cluster] INV-PD-1 violated: DecodeEnqueueTime (%d) < TransferCompleteTime (%d) for req %s",
					e.parentReq.DecodeEnqueueTime, e.parentReq.TransferCompleteTime, e.parentReq.ID)
			}

			cs.inFlightRequests[decision.TargetInstance]++
			// INV-PD-4: register decode sub-request for CompletionTime detection.
			cs.pendingDecodeCompletions[e.decodeSubReq.ID] = e.parentReq.ID
			inst.InjectDecodeOnline(e.decodeSubReq, e.time)
			return
		}
	}
	panic(fmt.Sprintf("DecodeRoutingEvent: invalid TargetInstance %q", decision.TargetInstance))
}
