// Implements the WaitQueue, which holds all requests waiting to be processed.
// Requests are enqueued on arrival

package sim

import (
	"fmt"
	"strings"
)

// WaitQueue represents a FIFO queue of requests waiting to be scheduled for execution.
// In the simulator, this models the pool of incoming requests
// that are waiting for their next opportunity to be processed in a batch.
type WaitQueue struct {
	queue []*Request // FIFO queue of requests
}

// Enqueue adds a request to the back of the wait queue.
func (wq *WaitQueue) Enqueue(r *Request) {
	wq.queue = append(wq.queue, r)
}

func (wq *WaitQueue) String() string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, val := range wq.queue {
		fmt.Fprint(&sb, val)
		if i < len(wq.queue)-1 {
			sb.WriteString(" ")
		}
	}
	sb.WriteString("]")
	return sb.String()
}

// Len returns the number of requests in the queue.
func (wq *WaitQueue) Len() int {
	return len(wq.queue)
}

// Peek returns the request at the front of the queue without removing it.
// Returns nil if the queue is empty.
func (wq *WaitQueue) Peek() *Request {
	if len(wq.queue) == 0 {
		return nil
	}
	return wq.queue[0]
}

// PrependFront inserts a request at the front of the queue.
// Used for preemption: a request evicted from the running batch
// is placed back at the head of the wait queue for immediate rescheduling.
func (wq *WaitQueue) PrependFront(req *Request) {
	if req == nil {
		panic("PrependFront: req must not be nil")
	}
	wq.queue = append([]*Request{req}, wq.queue...)
}

// Items returns the queue contents for iteration.
// The returned slice is the queue's internal storage -- callers within the
// sim package may iterate over it but MUST NOT append to or reslice it.
// For reordering, use Reorder() instead.
func (wq *WaitQueue) Items() []*Request {
	return wq.queue
}

// Reorder applies fn to the queue contents, allowing in-place reordering.
// The InstanceScheduler.OrderQueue method is the primary consumer:
//
//	wq.Reorder(func(reqs []*Request) {
//	    scheduler.OrderQueue(reqs, now)
//	})
//
// fn receives the underlying slice and may sort it in-place.
// fn MUST NOT change the slice length (no append/delete).
func (wq *WaitQueue) Reorder(fn func([]*Request)) {
	if fn == nil {
		panic("Reorder: fn must not be nil")
	}
	n := len(wq.queue)
	fn(wq.queue)
	if len(wq.queue) != n {
		panic(fmt.Sprintf("Reorder: fn changed queue length from %d to %d", n, len(wq.queue)))
	}
}

// Remove removes a specific request from the queue by pointer identity.
// O(n) linear scan — acceptable because timeouts are infrequent relative to step processing.
// Returns true if the request was found and removed, false otherwise.
func (wq *WaitQueue) Remove(req *Request) bool {
	for i, r := range wq.queue {
		if r == req {
			wq.queue = append(wq.queue[:i], wq.queue[i+1:]...)
			return true
		}
	}
	return false
}

// DequeueBatch removes a request from the front of the queue.
// This is used by the scheduler to construct a batch for processing.
func (wq *WaitQueue) DequeueBatch() *Request {
	if len(wq.queue) == 0 {
		return nil
	}
	batch := wq.queue[0]
	wq.queue = wq.queue[1:]
	return batch
}
