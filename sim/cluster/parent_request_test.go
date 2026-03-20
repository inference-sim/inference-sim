package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestNewParentRequest_NumKVBlocks verifies ceiling-division math for KV block computation.
func TestNewParentRequest_NumKVBlocks(t *testing.T) {
	tests := []struct {
		name           string
		inputLen       int
		blockSize      int64
		wantNumBlocks  int64
	}{
		{"empty input", 0, 16, 0},
		{"one token", 1, 16, 1},
		{"exact multiple", 32, 16, 2},
		{"one below multiple", 31, 16, 2},
		{"one above multiple", 33, 16, 3},
		{"block size 1", 5, 1, 5},
		{"single block exact", 16, 16, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &sim.Request{
				ID:          "req-1",
				InputTokens: make([]int, tc.inputLen),
				ArrivalTime: 1000,
			}
			pr := NewParentRequest(req, tc.blockSize)
			if pr.NumKVBlocks != tc.wantNumBlocks {
				t.Errorf("NewParentRequest(inputLen=%d, blockSize=%d).NumKVBlocks = %d, want %d",
					tc.inputLen, tc.blockSize, pr.NumKVBlocks, tc.wantNumBlocks)
			}
		})
	}
}

// TestNewParentRequest_SubRequestIDs verifies sub-request ID derivation convention.
func TestNewParentRequest_SubRequestIDs(t *testing.T) {
	req := &sim.Request{ID: "req-42", InputTokens: make([]int, 10), ArrivalTime: 500}
	pr := NewParentRequest(req, 16)

	if pr.PrefillSubReqID != "req-42_prefill" {
		t.Errorf("PrefillSubReqID = %q, want %q", pr.PrefillSubReqID, "req-42_prefill")
	}
	if pr.DecodeSubReqID != "req-42_decode" {
		t.Errorf("DecodeSubReqID = %q, want %q", pr.DecodeSubReqID, "req-42_decode")
	}
}

// TestNewParentRequest_ArrivalTimePropagated verifies ArrivalTime is copied from original request.
func TestNewParentRequest_ArrivalTimePropagated(t *testing.T) {
	req := &sim.Request{ID: "req-1", InputTokens: make([]int, 10), ArrivalTime: 12345}
	pr := NewParentRequest(req, 16)

	if pr.ArrivalTime != 12345 {
		t.Errorf("ArrivalTime = %d, want 12345", pr.ArrivalTime)
	}
}

// TestNewParentRequest_OriginalRequestPreserved verifies the original request pointer is stored.
func TestNewParentRequest_OriginalRequestPreserved(t *testing.T) {
	req := &sim.Request{ID: "req-1", InputTokens: make([]int, 10), ArrivalTime: 100}
	pr := NewParentRequest(req, 16)

	if pr.OriginalRequest != req {
		t.Error("OriginalRequest does not point to the original request")
	}
	if pr.ID != req.ID {
		t.Errorf("ID = %q, want %q", pr.ID, req.ID)
	}
}

// TestNewParentRequest_PanicOnZeroBlockSize verifies panic guard on blockSizeTokens == 0.
func TestNewParentRequest_PanicOnZeroBlockSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for blockSizeTokens == 0")
		}
	}()
	req := &sim.Request{ID: "req-1", InputTokens: make([]int, 10)}
	NewParentRequest(req, 0)
}

// TestNewParentRequest_PanicOnNegativeBlockSize verifies panic guard on blockSizeTokens < 0.
func TestNewParentRequest_PanicOnNegativeBlockSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for blockSizeTokens < 0")
		}
	}()
	req := &sim.Request{ID: "req-1", InputTokens: make([]int, 10)}
	NewParentRequest(req, -1)
}
