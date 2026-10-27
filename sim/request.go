// Moved from blis-vllm/compat/engine/request.go at 6b1d026, the host's own: the 0.29/0.31
// compatibility engine now runs behind engine.Engine (blis-vllm compat/contract), and these are
// the host's records, not the engine's (phase 2c).

// Defines the Request struct that models an individual inference request in the simulation.
// Tracks arrival time, input/output tokens, progress, and timestamps for TTFT/TPOT.
package sim

import (
	"fmt"
	"math"
	"math/rand"
)

// TokenID is the simulator's compact representation of a tokenizer ID. It is a defined type
// (NOT a Go type alias of int32), so assignments from int still require an explicit
// conversion — the compile-time mixed-arithmetic check is the point. The engine sees token
// IDs as int32 (engine.Request.Prompt); the driver converts at its edge.
type TokenID int32

// MaxTokenID is the largest token ID a generated request carries (exclusive).
const MaxTokenID = 128000

// Compile-time guard: MaxTokenID must fit in TokenID's underlying int32 range.
// If MaxTokenID is ever raised above math.MaxInt32 (~2.1B), the int→TokenID
// cast in GenerateRandomTokenIDs would silently truncate. This expression
// evaluates to a negative integer constant when MaxTokenID > MaxInt32, which
// is not representable as uint and fails to compile.
const _ = uint(math.MaxInt32 - MaxTokenID)

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

// Phase is a request's benchmark phase, after aiperf's benchmark_phase record tag. Only
// PhaseWarmup is excluded from the measured window; profile and grace requests (sent in
// the profiling phase, completing before or after its sending window closes) count, as
// aiperf counts every valid profiling record (AGENTX_MVP_SPEC §5.3).
type Phase string

const (
	PhaseUnset   Phase = ""
	PhaseWarmup  Phase = "warmup"
	PhaseProfile Phase = "profile"
	PhaseGrace   Phase = "grace"
)

// Measured reports whether a request of this phase counts in the measured aggregates.
func (p Phase) Measured() bool { return p != PhaseWarmup }

type Request struct {
	ID string // Unique identifier for the request

	InputTokens  []TokenID // Prompt tokens
	OutputTokens []TokenID // Pre-specified output tokens (already known for the simulation)
	MaxOutputLen int       // Client output budget (vLLM max_tokens); 0 = no budget (input-only check, runtime stop enforces limit)

	State         RequestState // queued, running, completed
	ProgressIndex int64        // Total number of input tokens processed so far + number of output tokens generated so far

	TTFTSet          bool    // Tracks whether TTFT has been set
	FirstTokenTime   int64   // Timestamp when first token was generated
	ArrivalTime      int64   // Timestamp in ticks when the request arrives in the simulator
	ScheduledStepIdx int     // Step index when this request got scheduled (waiting -> running)
	FinishedStepIdx  int     // Step index when this request finished (running -> completed)
	DepartureTime    int64   // Tick its final step ends (its RequestLeftEvent time); 0 until it completes
	NumNewTokens     int     // Number of new tokens to be generated in the current step
	LengthCapped     bool    // Set when force-completed by runtime MaxModelLen cap (BC-5)
	ITL              []int64 // List of inter-token latencies
	// GeneratedTokens is the number of output tokens the engine generated, set when the request
	// completes; it is what a model server reports as usage.completion_tokens.
	GeneratedTokens int
	Priority        float64 // Instance-level scheduling priority (vLLM convention: lower = more urgent).
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
	// Agentic identifies a request of an agentic session-tree replay (#1937); nil otherwise.
	Agentic *AgenticTag

	// Cluster routing metadata. Set by RoutingDecisionEvent; zero-value when
	// Request is used outside the cluster routing pipeline (e.g., direct sim.Simulator tests).
	AssignedInstance string // Instance ID this request was routed to

	// Model tag for multi-model routing (empty = default model).
	// Phase 0: carried through the pipeline but not read by any routing policy.
	Model string

	// Adapter is the LoRA adapter id serving this request (registry key; #1464).
	// Empty ("") = base-model-only request (no adapter) — the no-op default that
	// preserves byte-identity for adapter-blind workloads (INV-6). When non-empty it
	// MUST resolve to a declared adapter in the registry (validated at workload
	// generation). Set from the owning client/cohort's Adapter during generation.
	Adapter string

	// Client timeout: absolute tick by which request must complete (0 = no timeout).
	// Computed during workload generation as ArrivalTime + timeout.
	Deadline int64

	// Per-request SLO TTFT target in microseconds (0 = no target).
	// Sent to the router as x-llm-d-slo-ttft-ms (slo-deadline-ordering-policy reads it).
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

	// RetainKVOnFinish keeps the request's KV blocks allocated when it finishes, for an owner
	// that releases them later. A P/D prefill sub-request sets it: NIXL keeps a finished
	// prefill's blocks until the decode side has pulled them or the lease expires
	// (vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:247-266 at
	// v0.31.0), and the cluster releases them then (#1934). False releases them at completion.
	RetainKVOnFinish bool

	// Flow control timestamps (issue #882). Zero when flow control is disabled.
	GatewayEnqueueTime  int64 // microseconds: when request entered the gateway queue
	GatewayDispatchTime int64 // microseconds: when request was dispatched from the gateway queue

	// Phase is the benchmark phase the request was sent in (#1938). The zero value means
	// the request is in the measured window, so a workload that sets no phase is measured
	// whole. A replay with a warm-up sets PhaseWarmup on the requests the measured
	// aggregates of the --metrics-path file must exclude.
	Phase Phase
}

// String returns a human-readable representation of the request.
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

// SequenceTokenSlice is the request's tokens [start, end) over its prompt followed by its
// output.
func (req *Request) SequenceTokenSlice(start, end int64) []TokenID {
	n := req.InputLen()
	if end <= n {
		return req.InputTokenSlice(start, end)
	}
	if end-n > int64(len(req.OutputTokens)) || start < 0 || end < start {
		panic(fmt.Sprintf("Request.SequenceTokenSlice: invalid range [%d, %d) for %d input and %d output tokens (request %s)",
			start, end, n, len(req.OutputTokens), req.ID))
	}
	out := make([]TokenID, 0, end-start)
	if start < n {
		out = append(out, req.InputTokens[start:n]...)
	}
	return append(out, req.OutputTokens[max(start-n, 0):end-n]...)
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
