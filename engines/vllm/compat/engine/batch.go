// Moved from inference-sim/sim/engine/batch.go at ca532abca209af41425c007974a5f3ef336f5981.

// Defines the Batch struct which represents a group of requests processed together
// in a single forward pass.

package engine

// Batch represents a group of requests being processed in a single forward pass.
// This abstraction mirrors the batching behavior of the vLLM scheduler.
type Batch struct {
	Requests []*Request // Requests included in the current batch
}
