package sim

import "testing"

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
