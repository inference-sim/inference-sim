// Implements the WaitQueue, which holds all requests waiting to be processed.
// Requests are enqueued on arrival

// TODO: Requests need to be re-queued on preemption.

package sim

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
