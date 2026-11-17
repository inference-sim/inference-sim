// Package trace provides decision-trace recording for cluster-level policy analysis.
// This package has no dependencies on sim/ or sim/cluster/ — it stores pure data types.
package trace

// AdmissionRecord captures a single admission policy decision.
type AdmissionRecord struct {
	RequestID string
	Clock     int64
	Admitted  bool
	Reason    string
}

// CandidateScore captures a counterfactual candidate instance with its score and state.
type CandidateScore struct {
	InstanceID       string
	Score            float64
	QueueDepth       int
	BatchSize        int
	InFlightRequests int
	KVUtilization    float64
	FreeKVBlocks     int64
}

// RoutingRecord captures one routing decision of the embedded router, with optional counterfactual analysis.
type RoutingRecord struct {
	RequestID      string
	Clock          int64
	ChosenInstance string
	Reason         string
	Scores         map[string]float64 // from Placement.Scores, the router's primary-profile scores (may be nil)
	Candidates     []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret         float64            // max(alternative scores) - score(chosen); 0 if chosen is best
}

// DisaggregationRecord captures a PD disaggregation decision.
// When Disaggregate=true, the request follows the disaggregated path. Paired
// PrefillRoutingRecord and KVTransferRecord are recorded for the same RequestID
// except in two drop scenarios:
//
//  1. No routable prefill pool instances (routingRejections++): no downstream records.
//  2. Decode-side drop (droppedAtDecodeKV++): PrefillRoutingRecord exists but
//     KVTransferRecord is absent. Two sub-cases, both counted the same:
//     - At KVTransferStartedEvent: decode pod became non-routable, or
//     ReserveTransferredKV fails (insufficient decode KV). Drop fires at
//     transfer start; no KVTransferCompletedEvent is scheduled. Issue #1343.
//     - At KVTransferCompletedEvent: decode pod became non-routable between
//     transfer start and transfer complete. Reserved KV is released.
//
// To detect case 1: check for the absence of a PrefillRoutingRecord with a
// matching ParentRequestID in the trace. To detect case 2: check the
// absence of a KVTransferRecord for a given ParentRequestID.
// Note: DecodeRoutingRecord is never emitted; the router chooses the decode pod together
// with the prefill pod (no second routing decision).
type DisaggregationRecord struct {
	RequestID string
	// Clock is the simulation time at which the router's routing (and
	// disaggregation) decision was dispatched.
	Clock        int64
	Disaggregate bool // true = routed to prefill pool; false = standard routing to decode pool
}

// PrefillRoutingRecord captures a prefill pool routing decision with optional counterfactual analysis.
// ParentRequestID equals the RequestID in the corresponding DisaggregationRecord for this request.
type PrefillRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → routing score (higher = more preferred). Never set today:
	// the prefill choice is recorded without scores.
	Scores     map[string]float64 // nil
	Candidates []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64            // max(alternative scores) - score(chosen); 0 if chosen is best; always >= 0
}

// DecodeRoutingRecord captures a decode pool routing decision with optional counterfactual analysis.
// ParentRequestID equals the RequestID in the corresponding DisaggregationRecord for this request.
type DecodeRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → routing score (higher = more preferred). Never emitted
	// today (see DisaggregationRecord).
	Scores     map[string]float64 // nil
	Candidates []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64            // max(alternative scores) - score(chosen); 0 if chosen is best; always >= 0
}

// EncodeRoutingRecord captures an encode pool routing decision.
// Emitted once per encoder the router chose for a request (recordRouting,
// sim/cluster/gateway.go). Under option A (zero-duration encode), the record
// reflects a routing decision only — no encode sub-request is injected.
// ParentRequestID equals the request ID (one encode record per parent).
type EncodeRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → composite routing score (higher = more preferred).
	// Nil when scoring is disabled.
	Scores     map[string]float64
	Candidates []CandidateScore // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64          // max(alternative scores) - score(chosen); >= 0
}

// KVTransferRecord captures a KV cache transfer event between prefill and decode instances.
// TransferDuration is always >= 0; negative values are clamped to 0 with a warning in
// KVTransferCompletedEvent.Execute() (sim/cluster/pd_events.go) if INV-PD-4 is ever violated.
type KVTransferRecord struct {
	ParentRequestID   string
	TransferStartTime int64 // microseconds (sim clock)
	TransferDuration  int64 // microseconds; >= 0 (clamped at recording site)
	NumKVBlocks       int64 // the prompt's KV blocks
	// PulledKVBlocks are the blocks the decode side's pull moved: NumKVBlocks less those its
	// own prefix cache already held (INV-PD-8). Added beside NumKVBlocks, which keeps its
	// meaning.
	PulledKVBlocks    int64
	PrefillInstanceID string
	DecodeInstanceID  string
}

// ModelScaleRecord is one model-autoscaler scale decision as the actuator applied it (#743). It
// is recorded for every decision with a non-zero Delta.
type ModelScaleRecord struct {
	Clock       int64 // simulation time of the actuation, microseconds
	ModelID     string
	GPUType     string // the variant targeted
	TPDegree    int
	Delta       int    // the decision's change; never 0
	OldReplicas int    // the model's replicas (not terminated or draining) just before it was applied
	NewReplicas int    // the same count just after
	Trigger     string // "required_capacity" for a scale-up, "spare_capacity" for a scale-down
	// The analyzer's signals for the model at the tick that made the decision.
	Utilization      float64
	RequiredCapacity float64
	SpareCapacity    float64
}

// ClusterScaleRecord is one cluster-autoscaler node decision as the cluster applied it (#743). It
// is recorded for every decision with a non-zero Delta.
type ClusterScaleRecord struct {
	Clock        int64 // simulation time of the decision, microseconds
	PoolName     string
	OldNodeCount int // non-terminated nodes of the pool before
	NewNodeCount int // and after
	// TriggerReason is why: "pending_threshold" for a node added for a waiting placement.
	TriggerReason  string
	PendingCount   int     // pending placements at the decision
	GPUUtilization float64 // the pool's allocated GPUs over its GPUs at the decision
}
