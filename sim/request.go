// Defines the Request struct that models an individual inference request in the simulation.
// Tracks arrival time, input/output tokens, progress, and timestamps for TTFT/TPOT.

package sim

import (
	"fmt"
)

// Request models a single request's lifecycle in the simulation.
// Each request has:
// - input tokens (prompt)
// - output tokens (pre-specified for simulation)
// - state tracking
// - progress index to track prefill/decode progress
// - TTFT and TPOT timestamps

type Request struct {
	ID          string // Unique identifier for the request
	ArrivalTime int64  // Timestamp when the request enters the system

	InputTokens  []int // Prompt tokens
	OutputTokens []int // Pre-specified output tokens (already known for the simulation)

	State         string // "queued", "running", "completed"
	ProgressIndex int    // Total number of input tokens processed so far + number of output tokens generated so far

	TTFTSet        bool  // Tracks whether TTFT has been set
	FirstTokenTime int64 // Timestamp when first token was generated
}

// This method returns a human-readable string representation of a Request.
func (req Request) String() string {
	return fmt.Sprintf("Request: (ID: %s, State: %s, ProgressIndex: %v, ArrivalTime: %d)", req.ID, req.State, req.ProgressIndex, req.ArrivalTime)
}
