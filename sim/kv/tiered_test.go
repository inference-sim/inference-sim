package kv

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestTieredKVCache_OffloadTriggered_WhenGPUExceedsThreshold(t *testing.T) {
	// BC-2: GIVEN 10 GPU blocks, 10 CPU blocks, threshold 0.5
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.5, 100.0, 0)

	// WHEN we allocate blocks filling >50% GPU, then release
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}} // 6 blocks
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
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
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
		req := &sim.Request{ID: fmt.Sprintf("r%d", i), InputTokens: []int{i*2 + 1, i*2 + 2, i*2 + 3, i*2 + 4}}
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

func TestTieredKVCache_TransferLatency_ConsumeClearsAccumulated(t *testing.T) {
	// BC-2: GIVEN a tiered cache where blocks are offloaded to CPU, then reloaded to GPU
	// Setup mirrors TestTieredKVCache_ThrashingDetected (known to trigger CPU-to-GPU reload)
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.3, 100.0, 10)
	tiered.SetClock(100)

	// Fill GPU to 80% (4 requests × 2 blocks = 8 used, 2 free)
	target := &sim.Request{ID: "target", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(target, 0, 4, []int64{})
	for i := 0; i < 3; i++ {
		other := &sim.Request{ID: fmt.Sprintf("o%d", i), InputTokens: []int{i*4 + 10, i*4 + 11, i*4 + 12, i*4 + 13}}
		tiered.AllocateKVBlocks(other, 0, 4, []int64{})
	}

	// Release target → offload its cached blocks to CPU (util 60% > 30%)
	tiered.ReleaseKVBlocks(target)
	if tiered.offloadCount == 0 {
		t.Fatal("setup error: offload should have triggered")
	}

	// Fill GPU so only 1 free block remains (6 used + 3 fillers = 9 used, 1 free)
	tiered.SetClock(2000)
	for i := 0; i < 3; i++ {
		filler := &sim.Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 100, i*2 + 101}}
		tiered.AllocateKVBlocks(filler, 0, 2, []int64{})
	}

	// WHEN requesting the same prefix triggers CPU-to-GPU reload
	sameReq := &sim.Request{ID: "retry", InputTokens: []int{1, 2, 3, 4}}
	cached := tiered.GetCachedBlocks([]int{1, 2, 3, 4})
	start := int64(len(cached)) * tiered.BlockSize()
	tiered.AllocateKVBlocks(sameReq, start, 4, cached)

	// THEN first ConsumePendingTransferLatency returns non-zero (reload accumulated latency)
	lat1 := tiered.ConsumePendingTransferLatency()
	if lat1 == 0 {
		t.Error("ConsumePendingTransferLatency() should return non-zero after CPU-to-GPU reload")
	}

	// AND second consume returns 0 (read-and-clear semantics)
	lat2 := tiered.ConsumePendingTransferLatency()
	if lat2 != 0 {
		t.Errorf("second ConsumePendingTransferLatency() = %d, want 0 (cleared)", lat2)
	}
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
	target := &sim.Request{ID: "target", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(target, 0, 4, []int64{})
	others := make([]*sim.Request, 3)
	for i := 0; i < 3; i++ {
		others[i] = &sim.Request{ID: fmt.Sprintf("o%d", i), InputTokens: []int{i*4 + 10, i*4 + 11, i*4 + 12, i*4 + 13}}
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
		filler := &sim.Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 100, i*2 + 101}}
		tiered.AllocateKVBlocks(filler, 0, 2, []int64{})
	}
	// GPU: 9 used, 1 free. Target prefix [1,2,3,4] needs 2 blocks but only 1 free.

	// Step 5: Re-request the SAME prefix — GPU fails, triggers CPU reload
	sameReq := &sim.Request{ID: "retry", InputTokens: []int{1, 2, 3, 4}}
	cached := tiered.GetCachedBlocks([]int{1, 2, 3, 4})
	start := int64(len(cached)) * tiered.BlockSize()
	tiered.AllocateKVBlocks(sameReq, start, 4, cached)

	// THEN thrashing rate should be > 0 (offload at clock=100, reload at clock=600)
	rate := tiered.KVThrashingRate()
	if rate <= 0 {
		t.Errorf("KVThrashingRate() = %f, want > 0 for offload+reload within 1000 ticks", rate)
	}
}

func TestTieredKVCache_GetCachedBlocks_DoesNotAffectHitRate(t *testing.T) {
	// GIVEN a tiered cache after one allocation cycle
	gpu := NewKVCacheState(4, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.5, 1.0, 100)

	// Populate prefix cache
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req1, 0, 4, []int64{})
	tiered.ReleaseKVBlocks(req1)

	rateBefore := tiered.CacheHitRate()

	// WHEN calling GetCachedBlocks multiple times via tiered interface (BC-6)
	// This simulates the reload retry path where GetCachedBlocks is called twice
	_ = tiered.GetCachedBlocks([]int{1, 2, 3, 4})
	_ = tiered.GetCachedBlocks([]int{1, 2, 3, 4})

	// THEN CacheHitRate is unchanged — GetCachedBlocks is a pure query
	rateAfter := tiered.CacheHitRate()
	if rateAfter != rateBefore {
		t.Errorf("CacheHitRate changed from %f to %f after GetCachedBlocks calls (should be pure query)", rateBefore, rateAfter)
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
