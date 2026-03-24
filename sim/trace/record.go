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

// RoutingRecord captures a single routing policy decision with optional counterfactual analysis.
type RoutingRecord struct {
	RequestID      string
	Clock          int64
	ChosenInstance string
	Reason         string
	Scores         map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates     []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret         float64            // max(alternative scores) - score(chosen); 0 if chosen is best
}

// DisaggregationRecord captures a PD disaggregation decision.
// When Disaggregate=true, the request follows the disaggregated path. Paired
// PrefillRoutingRecord, KVTransferRecord, and DecodeRoutingRecord are recorded
// for the same RequestID except in three drop scenarios:
//
//  1. No routable prefill pool instances (routingRejections++): no downstream records.
//  2. No routable decode pool instances (droppedAtDecodeKV++): PrefillRoutingRecord
//     exists but KVTransferRecord and DecodeRoutingRecord are absent.
//  3. AllocateTransferredKV fails on the decode instance (droppedAtDecodeKV++):
//     same as case 2.
//
// To detect case 1: check cs.RoutingRejections() > 0 or the absence of a
// PrefillRoutingRecord for a given RequestID in the trace. To detect cases 2
// and 3: check the absence of KVTransferRecord for a given ParentRequestID.
type DisaggregationRecord struct {
	RequestID    string
	Clock        int64
	Disaggregate bool // true = routed to prefill pool; false = standard routing
}

// PrefillRoutingRecord captures a prefill pool routing decision with optional counterfactual analysis.
// ParentRequestID equals the RequestID in the corresponding DisaggregationRecord for this request.
type PrefillRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → composite routing score (higher = more preferred).
	// Values are raw weighted-scorer outputs; not normalized. Nil when scoring is disabled.
	Scores     map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64            // max(alternative scores) - score(chosen); 0 if chosen is best; always >= 0
}

// DecodeRoutingRecord captures a decode pool routing decision with optional counterfactual analysis.
// ParentRequestID equals the RequestID in the corresponding DisaggregationRecord for this request.
type DecodeRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → composite routing score (higher = more preferred).
	// Values are raw weighted-scorer outputs; not normalized. Nil when scoring is disabled.
	Scores     map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64            // max(alternative scores) - score(chosen); 0 if chosen is best; always >= 0
}

// KVTransferRecord captures a KV cache transfer event between prefill and decode instances.
// TransferDuration is always >= 0; negative values are clamped to 0 with a warning in
// DecodeRoutingEvent.Execute() (sim/cluster/pd_events.go) if INV-PD-4 is ever violated.
type KVTransferRecord struct {
	ParentRequestID   string
	TransferStartTime int64 // microseconds (sim clock)
	TransferDuration  int64 // microseconds; >= 0 (clamped at recording site)
	NumKVBlocks       int64
	PrefillInstanceID string
	DecodeInstanceID  string
}
