// Defines the Request struct that models an individual inference request in the simulation.
// Tracks arrival time, input/output tokens, progress, and timestamps for TTFT/TPOT.

package sim

import (
	"fmt"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim/internal/tokenid"
)

// TokenID is the simulator's compact representation of a tokenizer ID.
// Re-exported from sim/internal/tokenid as an alias at the package boundary,
// so callers write sim.TokenID. The underlying type (tokenid.TokenID) is a
// defined type (NOT a Go type alias of int32), so assignments from int still
// require an explicit conversion — the compile-time mixed-arithmetic check
// is the point.
type TokenID = tokenid.TokenID

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
	StateQueued              RequestState = "queued"
	StateRunning             RequestState = "running"
	StateCompleted           RequestState = "completed"
	StateTimedOut            RequestState = "timed_out"
	StateWaitingForRemoteKVs RequestState = "waiting_for_remote_kvs"
)

type Request struct {
	ID string // Unique identifier for the request

	InputTokens  []TokenID // Prompt tokens
	OutputTokens []TokenID // Pre-specified output tokens (already known for the simulation)
	MaxOutputLen int   // Client output budget (vLLM max_tokens); 0 = no budget (input-only check, runtime stop enforces limit)

	State         RequestState // queued, running, completed
	ProgressIndex int64        // Total number of input tokens processed so far + number of output tokens generated so far

	TTFTSet          bool    // Tracks whether TTFT has been set
	FirstTokenTime   int64   // Timestamp when first token was generated
	ArrivalTime      int64   // Timestamp in ticks when the request arrives in the simulator
	ScheduledStepIdx int     // Step index when this request got scheduled (waiting -> running)
	FinishedStepIdx  int     // Step index when this request finished (running -> completed)
	NumNewTokens     int     // Number of new tokens to be generated in the current step
	LengthCapped     bool    // Set when force-completed by runtime MaxModelLen cap (BC-5)
	ITL              []int64 // List of inter-token latencies
	Priority         float64 // Instance-level scheduling priority (vLLM convention: lower = more urgent).
	// Set once at EnqueueRequest/EnqueueDecodeSubRequest via SLOPriorityMap.InvertForVLLM;
	// not recomputed per step.

	// Workload metadata (PR10). All fields are zero-value safe for backward compatibility.
	TenantID        string  // Client/tenant identifier (empty for legacy workloads)
	SLOClass        string  // "critical", "standard", "sheddable", "batch", "background" (empty = default)
	SessionID       string  // Multi-turn session link (empty for single-turn)
	RoundIndex      int     // Round within session (0-based)
	TextTokenCount  int     // Text input tokens (multimodal breakdown)
	ImageTokenCount int     // Image input tokens
	AudioTokenCount int     // Audio input tokens
	VideoTokenCount int     // Video input tokens
	ReasonRatio     float64 // reason_tokens / total_output_tokens (part of OutputTokens, not additional)
	ClientID        string  // Client identifier from workload spec (empty for legacy/test workloads)
	PrefixGroup     string  // Shared prefix group name (empty for no prefix)
	PrefixLength    int     // Shared prefix token count; 0 = no prefix. Set during workload generation.
	Streaming       bool    // Whether client expects streaming response

	// Cluster routing metadata. Set by RoutingDecisionEvent; zero-value when
	// Request is used outside the cluster routing pipeline (e.g., direct sim.Simulator tests).
	AssignedInstance string // Instance ID this request was routed to

	// Model tag for multi-model routing (empty = default model).
	// Phase 0: carried through the pipeline but not read by any routing policy.
	Model string

	// Client timeout: absolute tick by which request must complete (0 = no timeout).
	// Computed during workload generation as ArrivalTime + timeout.
	Deadline int64

	// Per-request SLO TTFT target in microseconds (0 = no target).
	// Used by slo-deadline dispatch ordering: deadline = GatewayEnqueueTime + SLOTargetUs.
	// Distinct from Deadline (hard timeout). Set from workload spec or trace.
	SLOTargetUs int64

	// Redirected marks a request that was re-injected by the REDIRECT drain policy.
	// The source instance never completes it (the request was in WaitQ at drain time,
	// so it never ran on the source). The destination instance is the sole completion
	// site and increments CompletedRequests normally.
	// Do NOT skip completion accounting for redirected requests.
	Redirected bool

	// IsDecodeSubRequest is true when this request was created by PD disaggregation
	// after KV transfer from a prefill instance. It enters the decode instance with
	// ProgressIndex already set to len(InputTokens) and KV blocks pre-allocated.
	// Set by KVTransferCompletedEvent before the request is routed and enqueued.
	IsDecodeSubRequest bool

	// Flow control timestamps (issue #882). Zero when flow control is disabled.
	GatewayEnqueueTime  int64 // microseconds: when request entered the gateway queue
	GatewayDispatchTime int64 // microseconds: when request was dispatched from the gateway queue
}

// This method returns a human-readable string representation of a Request.
func (req Request) String() string {
	return fmt.Sprintf("Request: (ID: %s, State: %s, ProgressIndex: %v, ArrivalTime: %d)", req.ID, req.State, req.ProgressIndex, req.ArrivalTime)
}

// IsMultimodal reports whether the request carries any non-text modality
// (image, audio, or video). Derived from the per-modality token counts that
// already exist on Request (populated by workload generation, TraceV2 replay,
// and synthesis). Using a derived method instead of a new boolean field keeps
// a single source of truth and avoids a TraceV2 schema change.
//
// GAP-4 (issue #1264): consumed by MultimodalEncodeDecider.ShouldEncode.
func (req *Request) IsMultimodal() bool {
	return req.ImageTokenCount > 0 || req.AudioTokenCount > 0 || req.VideoTokenCount > 0
}

// InputLen returns the total number of input tokens without forcing materialization.
// Canonical replacement for `len(req.InputTokens)` and `util.Len64(req.InputTokens)`
// at call sites that need only the length (#1445).
func (req *Request) InputLen() int64 {
	return int64(len(req.InputTokens))
}

// FullInputTokens returns the full input-token sequence as a flat slice. The slice
// is already flat today (a view into a session-scoped shared buffer when produced
// by multi-turn workloads); the accessor exists as a forward-compatible migration
// point and to mark intent at call sites that need the whole sequence (trace
// export, scorers that hash the full prefix) (#1445).
//
// The returned slice's capacity is capped at its length (three-index slice),
// so a stray `append(req.FullInputTokens(), x)` by a caller allocates a fresh
// array instead of silently overwriting the next round's tokens in the shared
// session buffer. This converts the no-mutation contract into a runtime-
// enforced guarantee.
func (req *Request) FullInputTokens() []TokenID {
	n := len(req.InputTokens)
	return req.InputTokens[:n:n]
}

// InputTokenSlice returns the slice spanning the absolute index range [start, end).
// Replaces direct `req.InputTokens[start:end]` reads — same view into the underlying
// array, no copy (#1445).
//
// Panics with a decorated message if [start, end) is out of bounds. R6 prohibits
// logrus.Fatalf in the sim/ package; the panic surfaces the request ID and length
// so callers can debug from the stack trace.
//
// The returned slice's capacity is capped at `end` (three-index slice), so a
// stray `append(slice, ...)` by a caller allocates a fresh backing array
// instead of overwriting subsequent tokens in the shared session buffer.
func (req *Request) InputTokenSlice(start, end int64) []TokenID {
	n := int64(len(req.InputTokens))
	if start < 0 || end < start || end > n {
		panic(fmt.Sprintf("Request.InputTokenSlice: invalid range [%d, %d) for InputTokens len=%d (request %s)",
			start, end, n, req.ID))
	}
	return req.InputTokens[start:end:end]
}

// GenerateRandomTokenIDs creates a slice of random token IDs in [0, MaxTokenID).
// RNG calls: length × Intn(MaxTokenID).
func GenerateRandomTokenIDs(rng *rand.Rand, length int) []TokenID {
	tokens := make([]TokenID, length)
	for i := range tokens {
		tokens[i] = TokenID(rng.Intn(MaxTokenID))
	}
	return tokens
}
