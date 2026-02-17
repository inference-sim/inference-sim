package sim

import (
	"testing"
)

func TestKVCacheState_ImplementsKVStore(t *testing.T) {
	// GIVEN a KVCacheState
	kvc := NewKVCacheState(100, 16)

	// WHEN assigned to KVStore interface
	var store KVStore = kvc

	// THEN all accessor methods return expected values
	if store.BlockSize() != 16 {
		t.Errorf("BlockSize() = %d, want 16", store.BlockSize())
	}
	if store.TotalCapacity() != 100 {
		t.Errorf("TotalCapacity() = %d, want 100", store.TotalCapacity())
	}
	if store.UsedBlocks() != 0 {
		t.Errorf("UsedBlocks() = %d, want 0", store.UsedBlocks())
	}
	if store.CacheHitRate() != 0 {
		t.Errorf("CacheHitRate() = %f, want 0 (no lookups)", store.CacheHitRate())
	}
	if store.PendingTransferLatency() != 0 {
		t.Errorf("PendingTransferLatency() = %d, want 0 (single-tier)", store.PendingTransferLatency())
	}
	if store.KVThrashingRate() != 0 {
		t.Errorf("KVThrashingRate() = %f, want 0 (single-tier)", store.KVThrashingRate())
	}
}

func TestKVCacheState_CacheHitRate_ReflectsLookups(t *testing.T) {
	// GIVEN a KVCacheState with block size 2
	kvc := NewKVCacheState(10, 2)
	var store KVStore = kvc

	// Initial: no lookups
	if rate := store.CacheHitRate(); rate != 0 {
		t.Errorf("initial CacheHitRate() = %f, want 0", rate)
	}

	// WHEN we allocate blocks for a request (all misses)
	req := &Request{
		ID:          "req1",
		InputTokens: []int{1, 2, 3, 4}, // 2 blocks
	}
	ok := store.AllocateKVBlocks(req, 0, 4, []int64{})
	if !ok {
		t.Fatal("AllocateKVBlocks failed")
	}

	// THEN CacheMisses should be > 0 (blocks were freshly allocated)
	if kvc.CacheMisses == 0 {
		t.Error("expected CacheMisses > 0 after fresh allocation")
	}

	// WHEN we release and re-lookup the same prefix
	store.ReleaseKVBlocks(req)
	cached := store.GetCachedBlocks([]int{1, 2, 3, 4})

	// THEN we should get cache hits (blocks are still in hash table)
	if len(cached) == 0 {
		t.Error("expected cache hits for previously allocated prefix")
	}
	if kvc.CacheHits == 0 {
		t.Error("expected CacheHits > 0 after prefix lookup")
	}

	// AND CacheHitRate should be between 0 and 1
	rate := store.CacheHitRate()
	if rate <= 0 || rate >= 1 {
		t.Errorf("CacheHitRate() = %f, want 0 < rate < 1", rate)
	}
}
