// request.go
//
// Defines the Request struct that models an individual inference request in the simulation.
// Tracks arrival time, input/output tokens, progress, and timestamps for TTFT/TPOT.

package sim

// Request models a single request's lifecycle in the simulation.
// Each request has:
// - input tokens (prompt)
// - output tokens (pre-specified for simulation)
// - state tracking
// - progress index to track prefill/decode steps
// - TTFT and TPOT timestamps

type Request struct {
	ID          string // Unique identifier for the request
	ArrivalTime int64  // Timestamp when the request enters the system

	InputTokens  []string // Prompt tokens
	OutputTokens []string // Pre-specified output tokens (already known for the simulation)

	State         string // "queued", "running", "completed"
	ProgressIndex int    // Number of tokens already processed (input + output)

	TTFTSet        bool  // Tracks whether TTFT has been set
	FirstTokenTime int64 // Timestamp when first token was generated
}
