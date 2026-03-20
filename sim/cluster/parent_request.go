package cluster

import "github.com/inference-sim/inference-sim/sim"

// ParentRequest tracks the disaggregated lifecycle of a request that was split
// into prefill and decode sub-requests. Owned by ClusterSimulator.
type ParentRequest struct {
	ID              string       // Original request ID
	OriginalRequest *sim.Request // Pointer to the original request (for metadata)
	PrefillSubReqID string
	DecodeSubReqID  string
	NumKVBlocks     int64 // KV blocks to transfer (ceil(inputLen / blockSize))

	// Phase timestamps (microseconds). Zero means phase not yet reached.
	ArrivalTime          int64
	PrefillEnqueueTime   int64
	PrefillCompleteTime  int64
	TransferStartTime    int64
	TransferCompleteTime int64
	DecodeEnqueueTime    int64
	// CompletionTime has two meanings depending on outcome:
	//   - Successful decode: set by detectDecodeCompletions when the decode sub-request
	//     finishes its last step. CompletionTime == actual decode completion time.
	//   - Dropped at decode KV allocation: set to the DecodeRoutingEvent time (the point
	//     when the drop was detected). CompletionTime < actual decode time (which never
	//     happened). Use DecodeInstanceID == "" to distinguish dropped requests.
	CompletionTime int64

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
