package cluster

import (
	"container/heap"
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// gatewayQueueEntry holds a request in the gateway queue with ordering metadata.
type gatewayQueueEntry struct {
	request  *sim.Request
	priority int   // priorityMap.Priority(request.SLOClass)
	seqID    int64 // monotonic sequence for FIFO tie-breaking
}

// gatewayQueueHeap implements heap.Interface for the gateway queue.
// Ordering depends on dispatch mode:
// - "fifo": ordered by seqID only
// - "priority": ordered by (-priority, seqID) — highest priority first, then FIFO
type gatewayQueueHeap struct {
	entries      []gatewayQueueEntry
	priorityMode bool // true for "priority", false for "fifo"
}

func (h *gatewayQueueHeap) Len() int { return len(h.entries) }

func (h *gatewayQueueHeap) Less(i, j int) bool {
	if h.priorityMode {
		if h.entries[i].priority != h.entries[j].priority {
			return h.entries[i].priority > h.entries[j].priority // higher priority first
		}
	}
	return h.entries[i].seqID < h.entries[j].seqID // FIFO within same priority
}

func (h *gatewayQueueHeap) Swap(i, j int) {
	h.entries[i], h.entries[j] = h.entries[j], h.entries[i]
}

func (h *gatewayQueueHeap) Push(x any) {
	h.entries = append(h.entries, x.(gatewayQueueEntry))
}

func (h *gatewayQueueHeap) Pop() any {
	old := h.entries
	n := len(old)
	entry := old[n-1]
	h.entries = old[:n-1]
	return entry
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

// GatewayQueue is a priority-ordered queue for holding admitted requests before routing.
// Implements saturation-gated dispatch for GIE flow control parity.
type GatewayQueue struct {
	heap          gatewayQueueHeap
	maxDepth      int // 0 = unlimited
	shedCount     int // number of requests shed (evicted victims only)
	rejectedCount int // number of requests rejected (queue full, incoming could not displace any entry)
	priorityMap   *sim.SLOPriorityMap
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
	q := &GatewayQueue{
		heap: gatewayQueueHeap{
			priorityMode: dispatchOrder == "priority",
		},
		maxDepth:    maxDepth,
		priorityMap: priorityMap,
	}
	heap.Init(&q.heap)
	return q
}

// Enqueue adds a request to the gateway queue.
// When the queue is at capacity, only sheddable (priority < 0) entries are eviction candidates.
// If no sheddable candidate exists, or the incoming request cannot displace the lowest sheddable
// entry (lower or equal priority), the incoming request is rejected.
// Returns the outcome and the evicted victim (non-nil only for ShedVictim).
func (q *GatewayQueue) Enqueue(req *sim.Request, seqID int64) (EnqueueOutcome, *sim.Request) {
	priority := q.priorityMap.Priority(req.SLOClass)
	entry := gatewayQueueEntry{request: req, priority: priority, seqID: seqID}

	if q.maxDepth > 0 && q.heap.Len() >= q.maxDepth {
		// Find the lowest-priority sheddable entry (priority < 0 only).
		minIdx := -1
		for i := 0; i < q.heap.Len(); i++ {
			if q.heap.entries[i].priority >= 0 {
				continue // non-sheddable — skip
			}
			if minIdx == -1 ||
				q.heap.entries[i].priority < q.heap.entries[minIdx].priority ||
				(q.heap.entries[i].priority == q.heap.entries[minIdx].priority &&
					q.heap.entries[i].seqID > q.heap.entries[minIdx].seqID) {
				minIdx = i
			}
		}

		if minIdx == -1 {
			// No sheddable candidate — reject the incoming request.
			q.rejectedCount++
			return Rejected, nil
		}

		// Only displace if incoming has higher priority (or same priority + earlier arrival).
		minEntry := q.heap.entries[minIdx]
		if priority > minEntry.priority || (priority == minEntry.priority && seqID < minEntry.seqID) {
			victim := minEntry.request
			heap.Remove(&q.heap, minIdx)
			q.shedCount++
			heap.Push(&q.heap, entry)
			return ShedVictim, victim
		}

		// Incoming cannot outpriority any sheddable entry — reject it.
		q.rejectedCount++
		return Rejected, nil
	}

	heap.Push(&q.heap, entry)
	return Enqueued, nil
}

// Dequeue removes and returns the highest-priority (or earliest for FIFO) request.
// Returns nil if the queue is empty.
func (q *GatewayQueue) Dequeue() *sim.Request {
	if q.heap.Len() == 0 {
		return nil
	}
	entry := heap.Pop(&q.heap).(gatewayQueueEntry)
	return entry.request
}

// Len returns the number of requests in the gateway queue.
func (q *GatewayQueue) Len() int {
	return q.heap.Len()
}

// ShedCount returns the number of requests shed (evicted victims) due to capacity.
func (q *GatewayQueue) ShedCount() int {
	return q.shedCount
}

// RejectedCount returns the number of requests rejected (queue full, incoming could not displace any entry).
func (q *GatewayQueue) RejectedCount() int {
	return q.rejectedCount
}
