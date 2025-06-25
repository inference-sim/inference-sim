// Defines the Batch struct which represents a group of requests processed together
// in a single forward pass.

package sim

// Batch represents a group of requests being processed in a single forward pass.
// This abstraction mirrors the batching behavior of the vLLM scheduler.
type Batch struct {
	Requests []*Request // Requests included in the current batch
}

// NewBatch creates a new Batch instance from a given slice of requests.
func NewBatch(reqs []*Request) *Batch {
	return &Batch{Requests: reqs}
}
