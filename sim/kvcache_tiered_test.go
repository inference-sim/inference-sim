package sim

import (
	"fmt"
	"testing"
)

func TestTieredKVCache_OffloadTriggered_WhenGPUExceedsThreshold(t *testing.T) {
	// BC-2: GIVEN 10 GPU blocks, 10 CPU blocks, threshold 0.5
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.5, 100.0, 0)

	// WHEN we allocate blocks filling >50% GPU, then release
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}} // 6 blocks
	if !tiered.AllocateKVBlocks(req, 0, 12, []int64{}) {
		t.Fatal("allocation should succeed")
	}
	tiered.ReleaseKVBlocks(req)

	// THEN GPU utilization should be at or below threshold
	gpuUtil := float64(tiered.UsedBlocks()) / float64(tiered.TotalCapacity())
	if gpuUtil > 0.5 {
		t.Errorf("GPU utilization after offload = %f, want <= 0.5", gpuUtil)
	}
}

func TestTieredKVCache_CPUFull_OffloadStopsGracefully(t *testing.T) {
	// BC-10: GIVEN 10 GPU blocks, 2 CPU blocks, threshold 0.3
	tiered := NewTieredKVCache(NewKVCacheState(10, 2), 2, 0.3, 100.0, 0)

	// WHEN we allocate many blocks and release (offload should be limited by CPU capacity)
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
	if !tiered.AllocateKVBlocks(req, 0, 16, []int64{}) {
		t.Fatal("allocation should succeed")
	}
	tiered.ReleaseKVBlocks(req)

	// THEN no panic occurred and GPU capacity is preserved
	if tiered.TotalCapacity() != 10 {
		t.Errorf("TotalCapacity() = %d, want 10 (unchanged)", tiered.TotalCapacity())
	}
}

func TestTieredKVCache_Conservation_AllocateReleaseCycle(t *testing.T) {
	// BC-9: GIVEN a tiered cache
	tiered := NewTieredKVCache(NewKVCacheState(10, 2), 5, 0.5, 100.0, 0)

	// WHEN we run multiple allocate-release cycles
	for i := 0; i < 5; i++ {
		req := &Request{ID: fmt.Sprintf("r%d", i), InputTokens: []int{i*2 + 1, i*2 + 2, i*2 + 3, i*2 + 4}}
		if !tiered.AllocateKVBlocks(req, 0, 4, []int64{}) {
			t.Fatalf("allocation %d failed", i)
		}
		tiered.ReleaseKVBlocks(req)
	}

	// THEN UsedBlocks returns to 0 (all blocks released, conservation holds)
	if tiered.UsedBlocks() != 0 {
		t.Errorf("UsedBlocks() = %d after all releases, want 0", tiered.UsedBlocks())
	}
	if tiered.TotalCapacity() != 10 {
		t.Errorf("TotalCapacity() = %d, want 10 (unchanged)", tiered.TotalCapacity())
	}
}

func TestTieredKVCache_TransferLatency_QueryAndClear(t *testing.T) {
	// BC-4: GIVEN a tiered cache with offloaded blocks that get reloaded
	tiered := NewTieredKVCache(NewKVCacheState(10, 2), 10, 0.3, 1.0, 10)

	// Allocate, release (triggers offload), then fill GPU
	reqs := make([]*Request, 4)
	for i := 0; i < 4; i++ {
		reqs[i] = &Request{ID: fmt.Sprintf("r%d", i), InputTokens: []int{i*4 + 1, i*4 + 2, i*4 + 3, i*4 + 4}}
		tiered.AllocateKVBlocks(reqs[i], 0, 4, []int64{})
	}
	tiered.ReleaseKVBlocks(reqs[0])
	tiered.ReleaseKVBlocks(reqs[1])
	// Fill remaining GPU
	for i := 10; i < 20; i++ {
		filler := &Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 100, i*2 + 101}}
		tiered.AllocateKVBlocks(filler, 0, 2, []int64{})
	}
	// Trigger reload by allocating when GPU is full
	newReq := &Request{ID: "new", InputTokens: []int{200, 201}}
	tiered.AllocateKVBlocks(newReq, 0, 2, []int64{})

	// WHEN we query PendingTransferLatency
	lat1 := tiered.PendingTransferLatency()
	lat2 := tiered.PendingTransferLatency()

	// THEN second call returns 0 (query-and-clear semantics)
	if lat2 != 0 {
		t.Errorf("second PendingTransferLatency() = %d, want 0 (query-and-clear)", lat2)
	}
	_ = lat1 // first call captures accumulated latency
}

func TestTieredKVCache_ZeroBandwidth_Panics(t *testing.T) {
	// BC-11: GIVEN bandwidth == 0
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for zero bandwidth")
		}
	}()
	// WHEN creating TieredKVCache
	NewTieredKVCache(NewKVCacheState(10, 2), 10, 0.5, 0, 0)
	// THEN it panics
}

func TestTieredKVCache_ThrashingDetected_WhenReloadWithinWindow(t *testing.T) {
	// BC-6: GIVEN 10 GPU blocks (block_size=2), threshold 0.3
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.3, 100.0, 0)
	tiered.SetClock(100)

	// Step 1: Allocate target prefix [1,2,3,4] (2 blocks) + 6 other blocks to fill GPU
	target := &Request{ID: "target", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(target, 0, 4, []int64{})
	others := make([]*Request, 3)
	for i := 0; i < 3; i++ {
		others[i] = &Request{ID: fmt.Sprintf("o%d", i), InputTokens: []int{i*4 + 10, i*4 + 11, i*4 + 12, i*4 + 13}}
		tiered.AllocateKVBlocks(others[i], 0, 4, []int64{})
	}
	// GPU: 8 used (80%), 2 free (blocks 8,9 — never allocated, no hash)

	// Step 2: Release target → GPU drops to 6 used (60% > 30%), offload triggers
	// Target's 2 blocks (with hashes) go to free list, then offloaded to CPU
	tiered.ReleaseKVBlocks(target)

	// Verify something was offloaded
	offloadsAfterRelease := tiered.offloadCount
	if offloadsAfterRelease == 0 {
		t.Fatal("expected offload to occur after release")
	}

	// Step 3: Advance clock within 1000-tick window
	tiered.SetClock(600)

	// Step 4: Fill GPU so target prefix can't be allocated fresh
	// GPU has 6 used + 4 free (2 from target release + 2 original). Fill 3 more.
	for i := 0; i < 3; i++ {
		filler := &Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 100, i*2 + 101}}
		tiered.AllocateKVBlocks(filler, 0, 2, []int64{})
	}
	// GPU: 9 used, 1 free. Target prefix [1,2,3,4] needs 2 blocks but only 1 free.

	// Step 5: Re-request the SAME prefix — GPU fails, triggers CPU reload
	sameReq := &Request{ID: "retry", InputTokens: []int{1, 2, 3, 4}}
	cached := tiered.GetCachedBlocks([]int{1, 2, 3, 4})
	start := int64(len(cached)) * tiered.BlockSize()
	tiered.AllocateKVBlocks(sameReq, start, 4, cached)

	// THEN thrashing rate should be > 0 (offload at clock=100, reload at clock=600)
	rate := tiered.KVThrashingRate()
	if rate <= 0 {
		t.Errorf("KVThrashingRate() = %f, want > 0 for offload+reload within 1000 ticks", rate)
	}
}

func TestTieredKVCache_NoCacheHitsDoubleCount(t *testing.T) {
	// GIVEN a tiered cache where GPU allocation will initially fail, triggering CPU reload
	gpu := NewKVCacheState(4, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.5, 1.0, 100)

	// Populate prefix cache
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req1, 0, 4, []int64{})
	tiered.ReleaseKVBlocks(req1)
	gpu.CacheHits = 0

	// WHEN calling GetCachedBlocks multiple times (simulating tiered retry path)
	_ = tiered.GetCachedBlocks([]int{1, 2, 3, 4})
	_ = tiered.GetCachedBlocks([]int{1, 2, 3, 4})

	// THEN CacheHits is 0 — GetCachedBlocks is pure (BC-6)
	if gpu.CacheHits != 0 {
		t.Errorf("GPU CacheHits = %d after 2 GetCachedBlocks calls, want 0", gpu.CacheHits)
	}
}

func TestTieredKVCache_NegativeBandwidth_Panics(t *testing.T) {
	// BC-12 (partial): GIVEN negative bandwidth
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative bandwidth")
		}
	}()
	NewTieredKVCache(NewKVCacheState(10, 2), 10, 0.5, -1.0, 0)
}
