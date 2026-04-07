package cluster

import (
	"container/heap"
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// gatewayQueueEntry holds a request in the gateway queue with ordering metadata.
type gatewayQueueEntry struct {
	request  *sim.Request
	priority int   // SLOTierPriority(request.SLOClass)
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

// GatewayQueue is a priority-ordered queue for holding admitted requests before routing.
// Implements saturation-gated dispatch for GIE flow control parity.
type GatewayQueue struct {
	heap      gatewayQueueHeap
	maxDepth  int // 0 = unlimited
	shedCount int // number of requests shed due to capacity
}

// NewGatewayQueue creates a gateway queue with the given dispatch order and max depth.
// dispatchOrder: "fifo" or "priority". maxDepth: 0 = unlimited.
// Panics if dispatchOrder is invalid or maxDepth < 0.
func NewGatewayQueue(dispatchOrder string, maxDepth int) *GatewayQueue {
	if dispatchOrder != "fifo" && dispatchOrder != "priority" {
		panic(fmt.Sprintf("GatewayQueue: unknown dispatch order %q (must be fifo or priority)", dispatchOrder))
	}
	if maxDepth < 0 {
		panic(fmt.Sprintf("GatewayQueue: maxDepth must be >= 0, got %d", maxDepth))
	}
	q := &GatewayQueue{
		heap: gatewayQueueHeap{
			priorityMode: dispatchOrder == "priority",
		},
		maxDepth: maxDepth,
	}
	heap.Init(&q.heap)
	return q
}

// Enqueue adds a request to the gateway queue. Returns true if the request was shed
// (queue at capacity and request has lower or equal priority to all queued items).
// When the queue is at capacity, the lowest-priority request is shed (R1: counted, not silent).
func (q *GatewayQueue) Enqueue(req *sim.Request, seqID int64) (shed bool) {
	priority := sim.SLOTierPriority(req.SLOClass)
	entry := gatewayQueueEntry{request: req, priority: priority, seqID: seqID}

	if q.maxDepth > 0 && q.heap.Len() >= q.maxDepth {
		// Find the lowest-priority entry for shedding.
		// The heap is max-priority ordered, so the min is not necessarily at the root.
		minIdx := 0
		for i := 1; i < q.heap.Len(); i++ {
			if q.heap.entries[i].priority < q.heap.entries[minIdx].priority ||
				(q.heap.entries[i].priority == q.heap.entries[minIdx].priority &&
					q.heap.entries[i].seqID > q.heap.entries[minIdx].seqID) {
				minIdx = i
			}
		}
		minEntry := q.heap.entries[minIdx]

		// If new request has higher priority (or same priority but earlier), shed the min
		if priority > minEntry.priority || (priority == minEntry.priority && seqID < minEntry.seqID) {
			heap.Remove(&q.heap, minIdx)
			q.shedCount++
			heap.Push(&q.heap, entry)
			return false // new request kept, old one shed
		}
		// New request is shed
		q.shedCount++
		return true
	}

	heap.Push(&q.heap, entry)
	return false
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

// ShedCount returns the number of requests shed due to capacity.
func (q *GatewayQueue) ShedCount() int {
	return q.shedCount
}
