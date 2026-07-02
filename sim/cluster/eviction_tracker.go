package cluster

import (
	"container/heap"

	"blis/sim"
)

// evictionEntry represents a routed sheddable request eligible for eviction.
type evictionEntry struct {
	req          *sim.Request
	priority     int
	dispatchTime int64
	instanceID   string
	index        int
}

// evictionHeap implements heap.Interface for eviction ordering:
// lowest priority first, newest dispatch time first (tie-breaker).
type evictionHeap []*evictionEntry

func (h evictionHeap) Len() int { return len(h) }

func (h evictionHeap) Less(i, j int) bool {
	if h[i].priority != h[j].priority {
		return h[i].priority < h[j].priority
	}
	return h[i].dispatchTime > h[j].dispatchTime
}

func (h evictionHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *evictionHeap) Push(x interface{}) {
	entry := x.(*evictionEntry)
	entry.index = len(*h)
	*h = append(*h, entry)
}

func (h *evictionHeap) Pop() interface{} {
	old := *h
	n := len(old)
	entry := old[n-1]
	old[n-1] = nil
	entry.index = -1
	*h = old[:n-1]
	return entry
}

// EvictionTracker tracks routed sheddable requests for in-flight eviction.
// Only sheddable requests (priority < 0) are tracked.
type EvictionTracker struct {
	h    evictionHeap
	byID map[string]*evictionEntry
}

func NewEvictionTracker() *EvictionTracker {
	return &EvictionTracker{
		byID: make(map[string]*evictionEntry),
	}
}

// Track adds a routed request to the eviction tracker.
// Only sheddable requests (priority < 0) are tracked; non-sheddable are ignored.
func (et *EvictionTracker) Track(req *sim.Request, instanceID string, pm *sim.SLOPriorityMap) {
	if !pm.IsSheddable(req.SLOClass) {
		return
	}
	entry := &evictionEntry{
		req:          req,
		priority:     pm.Priority(req.SLOClass),
		dispatchTime: req.GatewayDispatchTime,
		instanceID:   instanceID,
	}
	heap.Push(&et.h, entry)
	et.byID[req.ID] = entry
}

// Untrack removes a request from the eviction tracker (e.g., on normal completion).
func (et *EvictionTracker) Untrack(requestID string) {
	entry, ok := et.byID[requestID]
	if !ok {
		return
	}
	heap.Remove(&et.h, entry.index)
	delete(et.byID, requestID)
}

// Pop removes and returns the most-evictable request and its instance ID.
// Returns (nil, "") if the tracker is empty.
func (et *EvictionTracker) Pop() (*sim.Request, string) {
	if et.h.Len() == 0 {
		return nil, ""
	}
	entry := heap.Pop(&et.h).(*evictionEntry)
	delete(et.byID, entry.req.ID)
	return entry.req, entry.instanceID
}

func (et *EvictionTracker) Len() int {
	return et.h.Len()
}
