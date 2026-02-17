package sim

import "testing"

func TestKVStore_SingleTier_BackwardCompatible(t *testing.T) {
	// BC-1: GIVEN KVCPUBlocks == 0 (default)
	cfg := SimConfig{TotalKVBlocks: 10, BlockSizeTokens: 2}
	store := NewKVStore(cfg)

	// WHEN we allocate, use, and release blocks through the interface
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	ok := store.AllocateKVBlocks(req, 0, 4, []int64{})
	if !ok {
		t.Fatal("AllocateKVBlocks should succeed")
	}
	if store.UsedBlocks() != 2 {
		t.Errorf("UsedBlocks() = %d, want 2 after allocating 2 blocks", store.UsedBlocks())
	}
	store.ReleaseKVBlocks(req)

	// THEN behavior matches pre-interface KVCacheState: used returns to 0, capacity unchanged
	if store.UsedBlocks() != 0 {
		t.Errorf("UsedBlocks() = %d, want 0 after release", store.UsedBlocks())
	}
	if store.TotalCapacity() != 10 {
		t.Errorf("TotalCapacity() = %d, want 10", store.TotalCapacity())
	}
	// Single-tier: transfer latency and thrashing are always 0
	if store.PendingTransferLatency() != 0 {
		t.Errorf("PendingTransferLatency() = %d, want 0 (single-tier)", store.PendingTransferLatency())
	}
	if store.KVThrashingRate() != 0 {
		t.Errorf("KVThrashingRate() = %f, want 0 (single-tier)", store.KVThrashingRate())
	}
}

func TestKVStore_CacheHitRate_ReflectsLookups(t *testing.T) {
	// BC-5: GIVEN a fresh KVStore
	store := NewKVStore(SimConfig{TotalKVBlocks: 10, BlockSizeTokens: 2})

	// Initial: no lookups → rate is 0
	if rate := store.CacheHitRate(); rate != 0 {
		t.Errorf("initial CacheHitRate() = %f, want 0", rate)
	}

	// WHEN we allocate blocks (all misses) then release and re-lookup (hits)
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	store.AllocateKVBlocks(req, 0, 4, []int64{})
	store.ReleaseKVBlocks(req)
	cached := store.GetCachedBlocks([]int{1, 2, 3, 4})

	// THEN cache hits occurred and CacheHitRate is between 0 and 1
	if len(cached) == 0 {
		t.Error("expected cache hits for previously allocated prefix")
	}
	rate := store.CacheHitRate()
	if rate <= 0 || rate >= 1 {
		t.Errorf("CacheHitRate() = %f, want 0 < rate < 1 (mix of hits and misses)", rate)
	}
}
