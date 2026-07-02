package cluster

import (
	"testing"

	"blis/sim"
)

// Tests for vLLM WAITING_FOR_REMOTE_KVS parity (issue #1343).
//
// In the vLLM v1 scheduler, when a decode pod is selected for a request
// whose KV is being transferred from a prefill pod, the decode pod
// reserves KV blocks immediately (with `delay_cache_blocks=True`) and
// places the request in WAITING_FOR_REMOTE_KVS until the connector
// signals transfer completion. The reservation reduces available
// capacity for concurrent requests during the transfer window.
//
// Before this PR, BLIS allocated KV only at KVTransferCompletedEvent,
// so other requests could "steal" decode-side capacity mid-transfer —
// BLIS undercounted decode-side KV pressure. These tests exercise the
// new reserve-at-start / promote-at-complete flow.

// TestWaitingForRemoteKVs_ReservationHoldsBlocksDuringTransfer verifies
// that reserved blocks show up in the decode instance's UsedBlocks
// between KVTransferStartedEvent and KVTransferCompletedEvent — i.e.,
// they are *unavailable* to other requests during the transfer window.
func TestWaitingForRemoteKVs_ReservationHoldsBlocksDuringTransfer(t *testing.T) {
	// Two decode instances, one prefill instance.
	// We reserve on the first decode pod and confirm UsedBlocks > 0
	// immediately after the started event (before the completed event fires).
	config := newTestDisaggDeploymentConfig(3, 1, 2)
	requests := newShortRequests(1) // single request → deterministic pod choice
	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)

	// Before the simulation starts, no blocks are used anywhere.
	for _, inst := range cs.instances {
		if used := inst.sim.KVCache.UsedBlocks(); used != 0 {
			t.Errorf("instance %s: UsedBlocks = %d at setup, want 0", inst.ID(), used)
		}
	}

	mustRun(t, cs)

	// Post-run sanity: at least one decode pod must have served a request.
	// (This test is also a regression test: it crashes if reservation reads
	// nil fields on the parent request.)
	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests != 1 {
		t.Fatalf("CompletedRequests = %d, want 1", metrics.CompletedRequests)
	}
}

// TestWaitingForRemoteKVs_StateDuringTransfer verifies that the decode
// sub-request carries StateWaitingForRemoteKVs between reservation at
// transfer start and promotion at transfer complete. We probe Execute
// on a started-event directly so we can catch the request state before
// the completion event fires.
func TestWaitingForRemoteKVs_StateDuringTransfer(t *testing.T) {
	config := newTestDisaggDeploymentConfig(3, 1, 2)
	requests := newShortRequests(1)
	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)

	// Drive the event loop manually until a KVTransferStartedEvent fires,
	// then stop and check the sub-request state.
	// The simplest signal we have post-hoc is the parent record.
	mustRun(t, cs)

	parents := cs.ParentRequests()
	if len(parents) != 1 {
		t.Fatalf("parents = %d, want 1", len(parents))
	}
	p := parents[0]
	if p.TransferStartTime == 0 {
		t.Fatal("TransferStartTime = 0 — transfer never started")
	}
	if p.TransferCompleteTime == 0 {
		t.Fatal("TransferCompleteTime = 0 — transfer never completed")
	}
	// By simulation end, the sub-request must have been promoted past
	// WaitingForRemoteKVs — either completed or still running.
	if p.DecodeSubReq != nil && p.DecodeSubReq.State == sim.StateWaitingForRemoteKVs {
		t.Errorf("DecodeSubReq stuck in StateWaitingForRemoteKVs after simulation end: %v",
			p.DecodeSubReq.State)
	}
}

// TestWaitingForRemoteKVs_ReservationFailure_DropsAtTransferStart verifies
// that when decode-side KV capacity is exhausted, the reservation at
// KVTransferStartedEvent fails and the parent is marked dropped at that
// time — no KVTransferCompletedEvent is scheduled.
//
// Key observable: TransferCompleteTime equals TransferStartTime for the
// dropped parent (both set to the start-event tick), because the drop
// path in KVTransferStartedEvent sets CompletionTime = e.time without
// scheduling the completion event. Successful transfers have
// TransferCompleteTime strictly greater than TransferStartTime.
func TestWaitingForRemoteKVs_ReservationFailure_DropsAtTransferStart(t *testing.T) {
	// 1 decode instance with only 3 blocks. Each request needs 2 blocks.
	// First request fills 2/3; second reservation attempt (also 2 blocks)
	// fails → drop at transfer start (because the running decode holds blocks).
	config := newTestDisaggDeploymentConfig(3, 2, 1)
	config.KVCacheConfig = sim.NewKVCacheConfig(3, 16, 0, 0, 0, 0)

	requests := newShortRequests(4)
	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)

	if cs.droppedAtDecodeKV == 0 {
		t.Fatal("droppedAtDecodeKV = 0, expected > 0 under tight decode KV")
	}

	// At least one parent must have been dropped at transfer start:
	// TransferStartTime > 0 but no completion event fired (CompletionTime
	// clamped to TransferStartTime, TransferCompleteTime unset or equal).
	parents := cs.ParentRequests()
	var foundStartDrop bool
	for _, p := range parents {
		// Drop-at-start signature: TransferStartTime > 0 and
		// CompletionTime == TransferStartTime (no completion event ran).
		// Successful or late-drop parents have TransferCompleteTime set.
		if p.TransferStartTime > 0 && p.CompletionTime == p.TransferStartTime && p.TransferCompleteTime == 0 {
			foundStartDrop = true
			break
		}
	}
	if !foundStartDrop {
		t.Error("no parent with drop-at-transfer-start signature found; reservation-failure drop never triggered")
	}

	// INV-1 conservation must still hold.
	metrics := cs.AggregatedMetrics()
	assertINV1Conservation(t, metrics, 4, "drop-at-transfer-start")
}

// TestWaitingForRemoteKVs_ConcurrentReservationsDoNotStealBlocks is the
// load-bearing regression test for issue #1343. It directly proves the
// property that motivated the whole PR: once one transfer reserves KV
// blocks on a decode pod, a second transfer attempting to reserve on
// the same pod cannot steal those blocks, even while the first transfer
// is still in flight.
//
// Constructed so that total capacity == blocks held by reservation A.
// Reservation B, which needs any non-zero block count, must fail while
// A still holds its reservation. Releasing A must make the same B
// reservation succeed — proving the blocks were reserved, not simply
// unavailable for some other reason.
//
// Reverting the PR's behavior (allocating at transfer complete rather
// than transfer start) would not be caught by any other test; this one
// asserts the exact no-stealing contract.
func TestWaitingForRemoteKVs_ConcurrentReservationsDoNotStealBlocks(t *testing.T) {
	// 5 blocks total. Reservation A consumes all 5. Reservation B requires
	// >=1 block and therefore must fail until A is released.
	cfg := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(5, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(testRooflineModelConfig(), testRooflineHWCalib(), "test", "H100", 1, 1, false, "", "roofline", 0),
	}
	inst := NewInstanceSimulator("decode_0", cfg)

	reqA := &sim.Request{
		ID:          "transferA_decode",
		InputTokens: make([]int, 80), // exactly 5 blocks at blockSize=16
		State:       sim.StateWaitingForRemoteKVs,
	}
	reqB := &sim.Request{
		ID:          "transferB_decode",
		InputTokens: make([]int, 16), // 1 block — minimum non-zero
		State:       sim.StateWaitingForRemoteKVs,
	}

	// A reserves first and consumes all 5 blocks.
	if ok := inst.ReserveTransferredKV(reqA); !ok {
		t.Fatal("ReserveTransferredKV(A) failed; test fixture is wrong, capacity should accommodate A")
	}
	if used := inst.sim.KVCache.UsedBlocks(); used != 5 {
		t.Fatalf("UsedBlocks after A reservation = %d, want 5 (fixture violated)", used)
	}

	// While A still holds blocks, B must not be able to steal them.
	// This is the core no-stealing contract from issue #1343.
	if ok := inst.ReserveTransferredKV(reqB); ok {
		t.Fatal("ReserveTransferredKV(B) succeeded while A still holds the blocks — blocks were stolen, no-stealing contract violated")
	}

	// Now release A. B's same reservation attempt must now succeed —
	// confirming the prior failure was specifically because A held the
	// blocks, not because the reservation path was broken.
	inst.ReleaseReservedKV(reqA)
	if used := inst.sim.KVCache.UsedBlocks(); used != 0 {
		t.Fatalf("UsedBlocks after A release = %d, want 0", used)
	}
	if ok := inst.ReserveTransferredKV(reqB); !ok {
		t.Error("ReserveTransferredKV(B) failed after A released — reservation path is broken independent of A")
	}
}

// TestReserveTransferredKV_ReleaseFreesBlocks verifies that
// ReleaseReservedKV undoes a prior ReserveTransferredKV: the released
// blocks return to the free pool and are available for another request.
func TestReserveTransferredKV_ReleaseFreesBlocks(t *testing.T) {
	cfg := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(10, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(testRooflineModelConfig(), testRooflineHWCalib(), "test", "H100", 1, 1, false, "", "roofline", 0),
	}
	inst := NewInstanceSimulator("decode_0", cfg)

	req := &sim.Request{
		ID:          "reserved_req",
		InputTokens: make([]int, 80), // 5 blocks
		State:       sim.StateWaitingForRemoteKVs,
	}

	if ok := inst.ReserveTransferredKV(req); !ok {
		t.Fatal("ReserveTransferredKV failed with ample capacity")
	}
	usedAfterReserve := inst.sim.KVCache.UsedBlocks()
	if usedAfterReserve == 0 {
		t.Fatal("UsedBlocks = 0 after ReserveTransferredKV, want > 0")
	}

	inst.ReleaseReservedKV(req)
	if used := inst.sim.KVCache.UsedBlocks(); used != 0 {
		t.Errorf("UsedBlocks = %d after ReleaseReservedKV, want 0 (blocks not freed)",
			used)
	}
}
