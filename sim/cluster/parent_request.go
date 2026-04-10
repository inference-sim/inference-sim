package cluster

import "github.com/inference-sim/inference-sim/sim"

// ParentRequest tracks the disaggregated lifecycle of a request that was split
// into prefill and decode sub-requests. Owned by ClusterSimulator.
type ParentRequest struct {
	ID              string       // Original request ID
	OriginalRequest *sim.Request // Pointer to the original request (for metadata)
	PrefillSubReqID string
	DecodeSubReqID  string
	DecodeSubReq    *sim.Request // Pointer to the decode sub-request (set by DecodeRoutingEvent)
	NumKVBlocks     int64        // KV blocks to transfer (ceil(inputLen / blockSize))

	// Phase timestamps (microseconds). Zero means phase not yet reached.
	ArrivalTime              int64
	DecodeRouteDecisionTime  int64 // When the decode instance was selected (decode-first step)
	DisaggDecisionTime       int64 // When the disaggregation decision was made
	PrefillEnqueueTime       int64
	PrefillCompleteTime      int64
	TransferStartTime        int64
	TransferCompleteTime     int64
	DecodeEnqueueTime        int64
	// CompletionTime has three meanings depending on outcome:
	//   - Successful decode: set by detectDecodeCompletions to
	//     clusterClock + decodeInstance.PostDecodeFixedOverhead() when the decode
	//     sub-request finishes its last step. Includes PostDecodeFixedOverhead so
	//     that projectPDMetrics() computes the same client-visible E2E as non-PD
	//     recordRequestCompletion (issue #846). For blackbox/roofline/cross-model
	//     (overhead=0), equals the raw cluster clock tick.
	//   - Dropped at decode KV allocation: set to the DecodeRoutingEvent time (the
	//     point when the drop was detected). Since decode never ran, this reflects
	//     the drop-detection time, not a decode completion time.
	//     Use DecodeInstanceID == "" to distinguish dropped requests.
	//   - Decode timeout: set by detectDecodeCompletions to the cluster clock at
	//     timeout detection time. The session is cancelled via sessionCallback.
	CompletionTime int64

	// SkippedDisaggregation is true when the disaggregation decider chose to serve
	// the request locally on the selected decode instance (cache hit path).
	// When true: no prefill sub-request is created; PrefillSubReqID/DecodeSubReqID
	// are populated but unused; the original request is injected directly.
	SkippedDisaggregation bool

	// Instance assignment
	PrefillInstanceID InstanceID
	DecodeInstanceID  InstanceID
}

// NewParentRequest creates a ParentRequest from the original request.
// blockSizeTokens must be > 0 (validated by KVCacheConfig constructor).
func NewParentRequest(req *sim.Request, blockSizeTokens int64) *ParentRequest {
	if blockSizeTokens <= 0 {
		panic("NewParentRequest: blockSizeTokens must be > 0")
	}
	inputLen := int64(len(req.InputTokens))
	numBlocks := (inputLen + blockSizeTokens - 1) / blockSizeTokens
	return &ParentRequest{
		ID:              req.ID,
		OriginalRequest: req,
		PrefillSubReqID: req.ID + "_prefill",
		DecodeSubReqID:  req.ID + "_decode",
		NumKVBlocks:     numBlocks,
		ArrivalTime:     req.ArrivalTime,
	}
}
