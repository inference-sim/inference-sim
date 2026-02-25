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

// RequestState represents the lifecycle state of a request.
type RequestState string

const (
	StateQueued    RequestState = "queued"
	StateRunning   RequestState = "running"
	StateCompleted RequestState = "completed"
)

type Request struct {
	ID string // Unique identifier for the request

	InputTokens  []int // Prompt tokens
	OutputTokens []int // Pre-specified output tokens (already known for the simulation)

	State         RequestState // queued, running, completed
	ProgressIndex int64  // Total number of input tokens processed so far + number of output tokens generated so far

	TTFTSet          bool    // Tracks whether TTFT has been set
	FirstTokenTime   int64   // Timestamp when first token was generated
	ArrivalTime      int64   // Timestamp in ticks when the request arrives in the simulator
	ScheduledStepIdx int     // Step index when this request got scheduled (waiting -> running)
	FinishedStepIdx  int     // Step index when this request finished (running -> completed)
	NumNewTokens     int     // Number of new tokens to be generated in the current step
	ITL              []int64  // List of inter-token latencies
	Priority         float64  // Scheduling priority score, recomputed each step by PriorityPolicy.
	                          // Higher = more urgent. Set by Simulator.Step() and optionally by
	                          // RoutingDecisionEvent as a one-shot cluster-level hint; read by schedulers.
	                          // Only meaningful for queued requests; zero-value (0.0) is the default.

	// Workload metadata (PR10). All fields are zero-value safe for backward compatibility.
	TenantID        string  // Client/tenant identifier (empty for legacy workloads)
	SLOClass        string  // "realtime", "interactive", "batch" (empty for legacy)
	Streaming       bool    // Whether this request uses streaming response mode
	SessionID       string  // Multi-turn session link (empty for single-turn)
	RoundIndex      int     // Round within session (0-based)
	TextTokenCount  int     // Text input tokens (multimodal breakdown)
	ImageTokenCount int     // Image input tokens
	AudioTokenCount int     // Audio input tokens
	VideoTokenCount int     // Video input tokens
	ReasonRatio     float64 // reason_tokens / total_output_tokens (part of OutputTokens, not additional)

	// Cluster routing metadata. Set by RoutingDecisionEvent; zero-value when
	// Request is used outside the cluster routing pipeline (e.g., direct sim.Simulator tests).
	AssignedInstance string // Instance ID this request was routed to

	// Model tag for multi-model routing (empty = default model).
	// Phase 0: carried through the pipeline but not read by any routing policy.
	Model string
}

// This method returns a human-readable string representation of a Request.
func (req Request) String() string {
	return fmt.Sprintf("Request: (ID: %s, State: %s, ProgressIndex: %v, ArrivalTime: %d)", req.ID, req.State, req.ProgressIndex, req.ArrivalTime)
}
