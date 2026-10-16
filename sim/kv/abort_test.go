package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestAbortKVBlocks: aborting a reservation whose KV never arrived forgets the prefix it alone
// would have served (a removed KV event), but a prefix already cached before the reservation stays
// cached, with no event: its earlier block still holds the content.
func TestAbortKVBlocks(t *testing.T) {
	tokens := []sim.TokenID{11, 12, 13, 14}
	for _, tc := range []struct {
		name         string
		cachedBefore bool
	}{{"new prefix", false}, {"prefix already cached", true}} {
		t.Run(tc.name, func(t *testing.T) {
			kvc := NewKVCacheState(8, 2)
			kvc.EnableKVEvents()
			if tc.cachedBefore {
				cached := &sim.Request{ID: "cached", InputTokens: tokens}
				if !kvc.AllocateKVBlocks(cached, 0, int64(len(tokens)), nil) {
					t.Fatal("allocate")
				}
				kvc.ReleaseKVBlocks(cached)
			}
			before := kvc.CachedBlockHashes()
			_ = kvc.TakeKVEvents()

			transfer := &sim.Request{ID: "transfer", InputTokens: tokens}
			if !kvc.AllocateKVBlocks(transfer, 0, int64(len(tokens)), nil) {
				t.Fatal("reserve")
			}
			kvc.AbortKVBlocks(transfer)

			if kvc.UsedBlocks() != 0 {
				t.Errorf("%d blocks in use after the abort", kvc.UsedBlocks())
			}
			after := kvc.CachedBlockHashes()
			stored, removed := 0, 0
			for _, e := range kvc.TakeKVEvents() {
				if e.Removed {
					removed++
				} else {
					stored++
				}
			}
			if tc.cachedBefore {
				if len(after) != len(before) || len(after) != 2 {
					t.Fatalf("cached %d hashes before, %d after; the earlier prefix must stay cached", len(before), len(after))
				}
				for i := range after {
					if after[i] != before[i] {
						t.Fatalf("cached hashes changed: %v -> %v", before, after)
					}
				}
				for _, h := range after {
					if blk := kvc.Blocks[kvc.HashToBlock[h]]; blk.Hash != h {
						t.Errorf("hash %s maps to block %d, which holds %q", h[:8], blk.ID, blk.Hash)
					}
				}
				if stored != 0 || removed != 0 {
					t.Errorf("events: %d stored, %d removed; want none (the cached content never changed)", stored, removed)
				}
			} else {
				if len(after) != 0 {
					t.Errorf("%d hashes cached for KV that never arrived", len(after))
				}
				if stored != removed || stored != 2 {
					t.Errorf("events: %d stored, %d removed; want each store cancelled", stored, removed)
				}
			}
		})
	}
}
