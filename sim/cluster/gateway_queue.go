package cluster

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// FlowKey identifies a per-tenant, per-priority flow queue within the gateway queue.
type FlowKey struct {
	TenantID string
	Priority int
}

// flowEntry holds a request in a flow queue with ordering metadata.
type flowEntry struct {
	request *sim.Request
	seqID   int64
}

// flowQueue is a FIFO queue for requests sharing the same FlowKey.
type flowQueue struct {
	key      FlowKey
	requests []flowEntry
}

// priorityBand groups all flow queues at the same priority level.
type priorityBand struct {
	priority int
	flows    map[string]*flowQueue // tenantID -> per-flow queue
	totalLen int
}

// EnqueueOutcome represents the result of enqueuing a request.
type EnqueueOutcome int

const (
	Enqueued   EnqueueOutcome = iota // request accepted into queue
	ShedVictim                       // request accepted, a sheddable victim was evicted
	Rejected                         // queue full, incoming request cannot displace any entry — not enqueued
)

// String returns a human-readable name for the outcome.
func (o EnqueueOutcome) String() string {
	switch o {
	case Enqueued:
		return "Enqueued"
	case ShedVictim:
		return "ShedVictim"
	case Rejected:
		return "Rejected"
	default:
		return fmt.Sprintf("EnqueueOutcome(%d)", int(o))
	}
}

// GatewayQueue is a per-priority-band, per-flow queue for holding admitted requests
// before routing. Replaces the flat heap with a hierarchical structure:
// bands sorted descending by priority, each containing per-tenant flow queues.
// Used by FlowControlAdmission for saturation-gated dispatch (GIE flow control parity).
// Saturation gating is enforced by the caller (ClusterSimulator.tryDispatchFromGatewayQueue).
type GatewayQueue struct {
	bands                []*priorityBand // sorted descending by priority
	totalLen             int
	maxDepth             int // 0 = unlimited (global)
	maxBandCapacity      int // 0 = unlimited (per-band)
	shedCount            int
	rejectedCount        int
	priorityMap          *sim.SLOPriorityMap
	priorityMode         bool    // true for "priority", false for "fifo"
	usageLimitThreshold  float64 // per-band HoL blocking ceiling (default 1.0 = no HoL)
}

// NewGatewayQueue creates a gateway queue with the given dispatch order and max depth.
// dispatchOrder: "fifo" or "priority". maxDepth: 0 = unlimited.
// If priorityMap is nil, DefaultSLOPriorityMap() is used.
// Panics if dispatchOrder is invalid or maxDepth < 0.
func NewGatewayQueue(dispatchOrder string, maxDepth int, priorityMap *sim.SLOPriorityMap) *GatewayQueue {
	if dispatchOrder != "fifo" && dispatchOrder != "priority" {
		panic(fmt.Sprintf("GatewayQueue: unknown dispatch order %q (must be fifo or priority)", dispatchOrder))
	}
	if maxDepth < 0 {
		panic(fmt.Sprintf("GatewayQueue: maxDepth must be >= 0, got %d", maxDepth))
	}
	if priorityMap == nil {
		priorityMap = sim.DefaultSLOPriorityMap()
	}
	return &GatewayQueue{
		maxDepth:            maxDepth,
		priorityMap:         priorityMap,
		priorityMode:        dispatchOrder == "priority",
		usageLimitThreshold: 1.0,
	}
}

// SetPerBandCapacity sets the maximum number of requests per priority band.
// 0 = unlimited. Must be >= 0.
func (q *GatewayQueue) SetPerBandCapacity(n int) {
	if n < 0 {
		panic(fmt.Sprintf("GatewayQueue: maxBandCapacity must be >= 0, got %d", n))
	}
	q.maxBandCapacity = n
}

// SetUsageLimitThreshold sets the per-band saturation ceiling for HoL blocking.
// Must be in (0, 1.0]. 1.0 = no HoL blocking (default). <1.0 gates lower-priority bands earlier.
// Matches GIE's UsageLimitPolicy: highest band gets ceiling 1.0, lowest gets threshold,
// intermediate bands are linearly interpolated.
func (q *GatewayQueue) SetUsageLimitThreshold(t float64) {
	if t <= 0 || t > 1.0 {
		panic(fmt.Sprintf("GatewayQueue: usageLimitThreshold must be in (0, 1.0], got %f", t))
	}
	q.usageLimitThreshold = t
}

// findOrCreateBand returns the band for the given priority, creating one if needed.
// Bands are maintained in descending priority order via binary search insertion.
func (q *GatewayQueue) findOrCreateBand(priority int) *priorityBand {
	// Binary search for descending order: find first band with priority <= target.
	idx := sort.Search(len(q.bands), func(i int) bool {
		return q.bands[i].priority <= priority
	})
	if idx < len(q.bands) && q.bands[idx].priority == priority {
		return q.bands[idx]
	}
	// Insert new band at idx (maintains descending order).
	band := &priorityBand{
		priority: priority,
		flows:    make(map[string]*flowQueue),
	}
	q.bands = append(q.bands, nil)
	copy(q.bands[idx+1:], q.bands[idx:])
	q.bands[idx] = band
	return band
}

// sortedFlowKeys returns the tenant IDs of a band's flows in sorted order (INV-6 determinism).
func sortedFlowKeys(flows map[string]*flowQueue) []string {
	keys := make([]string, 0, len(flows))
	for k := range flows {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Enqueue adds a request to the gateway queue.
// Capacity checks: per-band first (reject when band full), then global (cross-band shedding).
// Only sheddable (priority < 0) entries are eviction candidates.
// Returns the outcome and the evicted victim (non-nil only for ShedVictim).
func (q *GatewayQueue) Enqueue(req *sim.Request, seqID int64) (EnqueueOutcome, *sim.Request) {
	priority := q.priorityMap.Priority(req.SLOClass)
	tenantID := req.TenantID
	if tenantID == "" {
		tenantID = "default"
	}

	band := q.findOrCreateBand(priority)

	// Step 1: Per-band capacity check.
	// Within a band all entries share the same priority and new arrivals have higher
	// seqIDs, so displacement is never possible — reject directly when band is full.
	if q.maxBandCapacity > 0 && band.totalLen >= q.maxBandCapacity {
		q.rejectedCount++
		return Rejected, nil
	}

	// Step 2: Global capacity check (cross-band shedding).
	// Evict the lowest-priority sheddable entry globally to make room.
	var shedVictim *sim.Request
	if q.maxDepth > 0 && q.totalLen >= q.maxDepth {
		victim, victimFlow, victimBand, victimIdx := q.findGlobalShedVictim(priority, seqID)
		if victim == nil {
			q.rejectedCount++
			return Rejected, nil
		}
		shedVictim = victim.request
		q.removeEntryByIndex(victimFlow, victimBand, victimIdx)
		q.shedCount++
	}

	// Step 3: Enqueue into target band/flow.
	flow, ok := band.flows[tenantID]
	if !ok {
		flow = &flowQueue{key: FlowKey{TenantID: tenantID, Priority: priority}}
		band.flows[tenantID] = flow
	}
	flow.requests = append(flow.requests, flowEntry{request: req, seqID: seqID})
	band.totalLen++
	q.totalLen++

	if shedVictim != nil {
		return ShedVictim, shedVictim
	}
	return Enqueued, nil
}

// findGlobalShedVictim finds the lowest-priority sheddable entry across all bands.
// Only evicts if incoming request has strictly higher priority, or same priority with earlier seqID.
// Uses sorted key iteration for determinism (INV-6).
func (q *GatewayQueue) findGlobalShedVictim(incomingPriority int, incomingSeqID int64) (*flowEntry, *flowQueue, *priorityBand, int) {
	var bestEntry *flowEntry
	var bestFlow *flowQueue
	var bestBand *priorityBand
	var bestIdx int

	// Iterate all bands (sorted descending by priority). Must scan exhaustively
	// to find the absolute lowest-priority sheddable entry across all bands.
	for _, band := range q.bands {
		for _, tid := range sortedFlowKeys(band.flows) {
			flow := band.flows[tid]
			for i := range flow.requests {
				entry := &flow.requests[i]
				if !q.priorityMap.IsSheddable(entry.request.SLOClass) {
					continue
				}
				ePri := q.priorityMap.Priority(entry.request.SLOClass)
				if bestEntry == nil {
					bestEntry = entry
					bestFlow = flow
					bestBand = band
					bestIdx = i
					continue
				}
				bestPri := q.priorityMap.Priority(bestEntry.request.SLOClass)
				// Pick lowest priority; tie-break by highest seqID (latest arrival).
				if ePri < bestPri || (ePri == bestPri && entry.seqID > bestEntry.seqID) {
					bestEntry = entry
					bestFlow = flow
					bestBand = band
					bestIdx = i
				}
			}
		}
	}

	if bestEntry == nil {
		return nil, nil, nil, 0
	}

	// Only displace if incoming has higher priority (or same priority + earlier arrival).
	bestPri := q.priorityMap.Priority(bestEntry.request.SLOClass)
	if incomingPriority > bestPri || (incomingPriority == bestPri && incomingSeqID < bestEntry.seqID) {
		return bestEntry, bestFlow, bestBand, bestIdx
	}
	return nil, nil, nil, 0
}

// removeEntryByIndex removes the entry at the given index from the flow and decrements counts.
func (q *GatewayQueue) removeEntryByIndex(flow *flowQueue, band *priorityBand, idx int) {
	// Remove from slice. The shedding victim is always the last element in its flow
	// (highest seqID from append-only FIFO ordering), so this is a simple truncation
	// and the head (index 0) used by Dequeue is preserved.
	last := len(flow.requests) - 1
	if idx != last {
		panic(fmt.Sprintf("removeEntryByIndex: idx=%d != last=%d — shed victim must be flow tail (seqID ordering violated)", idx, last))
	}
	flow.requests = flow.requests[:last]
	band.totalLen--
	q.totalLen--

	// Clean up empty flow.
	if len(flow.requests) == 0 {
		delete(band.flows, flow.key.TenantID)
	}
}

// Dequeue removes and returns the highest-priority (or earliest for FIFO) request.
// Priority mode: iterates bands highest-first; within each band, picks flow with earliest seqID head.
// FIFO mode: scans all bands for globally-earliest seqID.
// Returns nil if the queue is empty.
func (q *GatewayQueue) Dequeue() *sim.Request {
	if q.totalLen == 0 {
		return nil
	}

	var req *sim.Request
	if q.priorityMode {
		req = q.dequeuePriority()
	} else {
		req = q.dequeueFIFO()
	}
	if req == nil {
		panic(fmt.Sprintf("GatewayQueue.Dequeue: totalLen=%d but dequeue returned nil — counter desync", q.totalLen))
	}
	return req
}

// DequeueGated attempts to dispatch one request with per-band HoL blocking.
// Matches GIE's dispatchCycle: iterates bands highest-first, checks per-band ceiling,
// halts entirely if saturation >= ceiling (prevents priority inversion).
// Returns nil if saturated or queue empty.
func (q *GatewayQueue) DequeueGated(saturation float64) *sim.Request {
	if q.totalLen == 0 {
		return nil
	}

	// Count non-empty bands for ceiling computation.
	nonEmptyBands := 0
	for _, band := range q.bands {
		if band.totalLen > 0 {
			nonEmptyBands++
		}
	}
	if nonEmptyBands == 0 {
		return nil
	}

	// Compute per-band ceilings via linear interpolation:
	// Highest band (rank 0) = 1.0, lowest band (rank N-1) = usageLimitThreshold.
	// Single band always gets ceiling 1.0.
	bandRank := 0
	for _, band := range q.bands {
		if band.totalLen == 0 {
			continue // skip empty bands (work conservation)
		}

		ceiling := 1.0
		if nonEmptyBands > 1 {
			// Linear interpolation: rank 0 → 1.0, rank (N-1) → threshold
			ceiling = 1.0 - float64(bandRank)*(1.0-q.usageLimitThreshold)/float64(nonEmptyBands-1)
		}

		if saturation >= ceiling {
			// HoL blocking: halt entire dispatch cycle to prevent priority inversion.
			return nil
		}

		req := q.dequeueFromBand(band)
		if req != nil {
			return req
		}
		bandRank++
	}
	return nil
}

// dequeuePriority dispatches from the highest non-empty band.
// Within the band, picks the flow head with the earliest seqID (global-strict fairness).
func (q *GatewayQueue) dequeuePriority() *sim.Request {
	for _, band := range q.bands {
		if band.totalLen == 0 {
			continue
		}
		return q.dequeueFromBand(band)
	}
	return nil
}

// dequeueFIFO scans all bands for the globally-earliest seqID.
func (q *GatewayQueue) dequeueFIFO() *sim.Request {
	var bestEntry *flowEntry
	var bestFlow *flowQueue
	var bestBand *priorityBand

	for _, band := range q.bands {
		if band.totalLen == 0 {
			continue
		}
		for _, tid := range sortedFlowKeys(band.flows) {
			flow := band.flows[tid]
			if len(flow.requests) == 0 {
				continue
			}
			head := &flow.requests[0]
			if bestEntry == nil || head.seqID < bestEntry.seqID {
				bestEntry = head
				bestFlow = flow
				bestBand = band
			}
		}
	}
	if bestEntry == nil {
		return nil
	}
	req := bestEntry.request
	// Remove head (index 0).
	bestFlow.requests = bestFlow.requests[1:]
	bestBand.totalLen--
	q.totalLen--
	if len(bestFlow.requests) == 0 {
		delete(bestBand.flows, bestFlow.key.TenantID)
	}
	return req
}

// dequeueFromBand picks the flow head with the earliest seqID within the given band.
func (q *GatewayQueue) dequeueFromBand(band *priorityBand) *sim.Request {
	var bestEntry *flowEntry
	var bestFlow *flowQueue

	for _, tid := range sortedFlowKeys(band.flows) {
		flow := band.flows[tid]
		if len(flow.requests) == 0 {
			continue
		}
		head := &flow.requests[0]
		if bestEntry == nil || head.seqID < bestEntry.seqID {
			bestEntry = head
			bestFlow = flow
		}
	}
	if bestEntry == nil {
		return nil
	}
	req := bestEntry.request
	bestFlow.requests = bestFlow.requests[1:]
	band.totalLen--
	q.totalLen--
	if len(bestFlow.requests) == 0 {
		delete(band.flows, bestFlow.key.TenantID)
	}
	return req
}

// Len returns the number of requests in the gateway queue.
func (q *GatewayQueue) Len() int {
	return q.totalLen
}

// LenByBand returns the number of requests in the band for the given priority.
// Returns 0 if no band exists for the priority.
func (q *GatewayQueue) LenByBand(priority int) int {
	for _, band := range q.bands {
		if band.priority == priority {
			return band.totalLen
		}
	}
	return 0
}

// ShedCount returns the number of requests shed (evicted victims) due to capacity.
func (q *GatewayQueue) ShedCount() int {
	return q.shedCount
}

// RejectedCount returns the number of requests rejected (queue full, incoming could not displace any entry).
func (q *GatewayQueue) RejectedCount() int {
	return q.rejectedCount
}
