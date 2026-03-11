package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestParentRequest_NewParentRequest(t *testing.T) {
	req := &sim.Request{
		ID:          "req_0",
		InputTokens: make([]int, 100),
		ArrivalTime: 1000,
	}
	parent := NewParentRequest(req, 16) // blockSizeTokens=16

	if parent.ID != "req_0" {
		t.Errorf("parent ID = %q, want %q", parent.ID, "req_0")
	}
	if parent.PrefillSubReqID != "req_0_prefill" {
		t.Errorf("prefill sub-req ID = %q, want %q", parent.PrefillSubReqID, "req_0_prefill")
	}
	if parent.DecodeSubReqID != "req_0_decode" {
		t.Errorf("decode sub-req ID = %q, want %q", parent.DecodeSubReqID, "req_0_decode")
	}
	// ceil(100/16) = 7
	if parent.NumKVBlocks != 7 {
		t.Errorf("NumKVBlocks = %d, want %d", parent.NumKVBlocks, 7)
	}
	if parent.ArrivalTime != 1000 {
		t.Errorf("ArrivalTime = %d, want 1000", parent.ArrivalTime)
	}
}

func TestParentRequest_ZeroInputTokens(t *testing.T) {
	req := &sim.Request{
		ID:          "req_empty",
		InputTokens: nil,
	}
	parent := NewParentRequest(req, 16)
	if parent.NumKVBlocks != 0 {
		t.Errorf("NumKVBlocks = %d, want 0 for empty input", parent.NumKVBlocks)
	}
}

func TestFilterSnapshotsByPool(t *testing.T) {
	membership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRolePrefill,
		"instance_2": PoolRoleDecode,
		"instance_3": PoolRoleDecode,
	}
	snapshots := []sim.RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 1},
		{ID: "instance_1", QueueDepth: 2},
		{ID: "instance_2", QueueDepth: 3},
		{ID: "instance_3", QueueDepth: 4},
	}

	prefill := FilterSnapshotsByPool(snapshots, membership, PoolRolePrefill)
	if len(prefill) != 2 {
		t.Fatalf("prefill snapshots = %d, want 2", len(prefill))
	}
	if prefill[0].ID != "instance_0" || prefill[1].ID != "instance_1" {
		t.Errorf("prefill IDs = [%s, %s], want [instance_0, instance_1]", prefill[0].ID, prefill[1].ID)
	}

	decode := FilterSnapshotsByPool(snapshots, membership, PoolRoleDecode)
	if len(decode) != 2 {
		t.Fatalf("decode snapshots = %d, want 2", len(decode))
	}
	if decode[0].ID != "instance_2" || decode[1].ID != "instance_3" {
		t.Errorf("decode IDs = [%s, %s], want [instance_2, instance_3]", decode[0].ID, decode[1].ID)
	}
}
