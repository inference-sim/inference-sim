package sim

// AgenticTag identifies a request of an agentic session-tree replay (blis replay
// --agentic-scenario, #1937): the lane and tree instance it belongs to, the stream
// (conversation) and turn it replays, and how it ended. Phase (Request.Phase) carries the
// benchmark phase; this carries the replay's structure, so a consumer can window or group
// on it. Nil on every request of every other workload, so their output is unchanged.
type AgenticTag struct {
	// Lane is the concurrency lane (0..C-1) whose session-tree slot the request runs under.
	Lane int `json:"lane"`
	// Tree numbers the tree instance: one per lane at start and one per recycle, in the
	// order they were minted. Prompt prefixes are shared only within one tree instance.
	Tree int `json:"tree"`
	// TraceID is the corpus trace the tree replays.
	TraceID string `json:"trace_id"`
	// Stream is the conversation: the trace id for the root, or aiperf's child session id
	// ({trace}::sa:{agent_id}[:fa|aux|wg:..], {trace}::fa|aux|wg:..).
	Stream string `json:"stream"`
	// Kind is "root", "subagent" or "flat" (a flattened worker chain split off the root).
	Kind string `json:"kind"`
	// StreamInstance numbers the stream's instance within the run. A stream normally has
	// one instance per tree; aiperf can start a second one when a snapshot seeds a child
	// whose spawning turn is then sent again (a branch that starts at exactly t*).
	StreamInstance int `json:"stream_instance"`
	// Turn is the turn index within the stream.
	Turn int `json:"turn"`
	// Outcome is how the request ended: "completed"; "error" (a client timeout, which aiperf
	// records as an error); "context_overflow" (refused by the engine's context check, which
	// aiperf skips); or "cancelled" (in flight when the grace period ended: aiperf keeps no
	// record of it). Empty while in flight.
	Outcome string `json:"outcome,omitempty"`
}

// Agentic request outcomes (AgenticTag.Outcome).
const (
	AgenticCompleted       = "completed"
	AgenticError           = "error"
	AgenticContextOverflow = "context_overflow"
	AgenticCancelled       = "cancelled"
)
