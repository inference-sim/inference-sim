package cluster

import "github.com/inference-sim/inference-sim/sim"

// ParentRequest tracks the disaggregated lifecycle of a request that was split
// into prefill and decode sub-requests. Owned by ClusterSimulator.
type ParentRequest struct {
	gatewayDone     bool         // the gateway has been told the request ended
	ID              string       // Original request ID
	OriginalRequest *sim.Request // Pointer to the original request (for metadata)
	PrefillSubReqID string
	PrefillSubReq   *sim.Request // Pointer to the prefill sub-request, set when the parent is created; its DepartureTime is when the KV the handoff moves exists.
	DecodeSubReqID  string
	DecodeSubReq    *sim.Request // Pointer to the decode sub-request. Set by KVTransferStartedEvent on successful ReserveTransferredKV (issue #1343); KVTransferCompletedEvent promotes it from StateWaitingForRemoteKVs to StateQueued and enqueues it on the decode instance. Nil after a late-drop (decode pod unroutable at transfer complete) when reserved KV is released.
	// DecodeWaiting is the decode sub-request the decode engine holds among its waiting handoffs,
	// before its pull starts (cs.remotePulls). It is the request machine's waiting-for-decode-KV
	// state (SIM-8): DecodeSubReq stays nil until the pull starts, and the gateway can evict the
	// parent from here (evictParent).
	DecodeWaiting *sim.Request
	NumKVBlocks   int64 // the prompt's KV blocks (ceil(inputLen / blockSize))
	// PulledKVBlocks are the blocks the decode side's NIXL pull moves: the prompt's blocks less
	// those the decode instance's prefix cache already held when it reserved them
	// (InstanceSimulator.ReserveTransferredKV). Set at transfer start; the handoff is priced
	// for these, and NewParentRequest sets it to NumKVBlocks until then.
	PulledKVBlocks int64

	// Phase timestamps (microseconds). Zero means phase not yet reached.
	ArrivalTime          int64
	PrefillEnqueueTime   int64
	PrefillCompleteTime  int64
	TransferStartTime    int64
	TransferCompleteTime int64
	// TransferAloneTicks is what the handoff would take alone on the fabric
	// (the Environment's demand for the pull, alone). TransferCompleteTime - TransferStartTime exceeds it
	// only by the time contended links added (INV-P2-2): the handoff's queueing delay.
	TransferAloneTicks int64
	DecodeEnqueueTime  int64
	// CompletionTime has four meanings depending on outcome:
	//   - Successful decode: set by detectDecodeCompletions to
	//     DecodeSubReq.DepartureTime + decodeInstance.PostDecodeFixedOverhead(): the
	//     end of the decode sub-request's last step (#1903), plus PostDecodeFixedOverhead
	//     so that projectPDMetrics() computes the same client-visible E2E as non-PD
	//     recordRequestCompletion (issue #846).
	//   - Dropped at transfer start (issue #1343): set to KVTransferStartedEvent
	//     time when ReserveTransferredKV fails or the decode pod is non-routable.
	//     Signature: TransferStartTime > 0 && CompletionTime == TransferStartTime
	//     && TransferCompleteTime == 0. No KVTransferCompletedEvent is scheduled.
	//   - Dropped at transfer complete: set to KVTransferCompletedEvent time when
	//     the decode pod transitioned non-routable during the transfer window.
	//     Reserved KV is released before the drop. Signature:
	//     TransferCompleteTime > 0 && DecodeSubReq == nil.
	//   - Decode timeout: set by detectDecodeCompletions to the cluster clock at
	//     timeout detection time. The session is cancelled via sessionCallback.
	CompletionTime int64

	// Instance assignment.
	// Both are the router's choice, set by dispatchDisaggregated (sim/cluster/gateway.go)
	// before the prefill sub-request is injected.
	PrefillInstanceID InstanceID
	DecodeInstanceID  InstanceID

	// P2P is whether the handoff goes through vLLM's p2p KV tier (p2pHandoff): the decode engine
	// loads the prompt from the prefill engine's WRITE, and the handoff's pull is not a transfer of
	// the cluster's. The WRITEs that served it are counted in P2PWrites; the first started at
	// P2PWriteStart and the last landed at P2PWriteLand, and P2PWriteAloneTicks is what they would
	// take alone on the fabric, summed. A p2p handoff with no WRITE moved no KV: the decode engine
	// held the prompt already, or its peer connect was rejected and it recomputed (inference-sim#2214).
	P2P                         bool
	P2PWrites                   int
	P2PWriteStart, P2PWriteLand int64
	P2PWriteAloneTicks          int64

	// EncodeInstanceIDs are the encoders the router chose for a multimodal request, in the
	// router's order (x-llm-d-encoder-hosts-ports), empty when it was not encoded.
	EncodeInstanceIDs []InstanceID
}

// NewParentRequest creates a ParentRequest from the original request.
// blockSizeTokens must be > 0 (validated by KVCacheConfig constructor).
func NewParentRequest(req *sim.Request, blockSizeTokens int64) *ParentRequest {
	if blockSizeTokens <= 0 {
		panic("NewParentRequest: blockSizeTokens must be > 0")
	}
	inputLen := req.InputLen()
	numBlocks := (inputLen + blockSizeTokens - 1) / blockSizeTokens
	return &ParentRequest{
		ID:              req.ID,
		OriginalRequest: req,
		PrefillSubReqID: req.ID + "_prefill",
		DecodeSubReqID:  req.ID + "_decode",
		NumKVBlocks:     numBlocks,
		PulledKVBlocks:  numBlocks,
		ArrivalTime:     req.ArrivalTime,
	}
}
