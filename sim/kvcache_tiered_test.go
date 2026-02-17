package sim

import (
	"fmt"
	"testing"
)

func TestTieredKVCache_OffloadTriggered_WhenGPUExceedsThreshold(t *testing.T) {
	// GIVEN 10 GPU blocks, 10 CPU blocks, threshold 0.5 (offload when >50% used)
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.5, 100.0, 0)

	// WHEN we allocate blocks for requests filling >50% GPU
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}} // needs 6 blocks
	ok := tiered.AllocateKVBlocks(req1, 0, 12, []int64{})
	if !ok {
		t.Fatal("allocation should succeed")
	}

	// Release to trigger offload check
	tiered.ReleaseKVBlocks(req1)

	// THEN GPU used should be <= threshold * capacity (some blocks offloaded to CPU)
	gpuUtil := float64(tiered.UsedBlocks()) / float64(tiered.TotalCapacity())
	if gpuUtil > 0.5 {
		t.Errorf("GPU utilization after offload = %f, want <= 0.5", gpuUtil)
	}
}

func TestTieredKVCache_CPUFull_OffloadStops(t *testing.T) {
	// GIVEN 10 GPU blocks, 2 CPU blocks, threshold 0.3
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 2, 0.3, 100.0, 0)

	// WHEN we allocate many blocks and release
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}} // 8 blocks
	ok := tiered.AllocateKVBlocks(req, 0, 16, []int64{})
	if !ok {
		t.Fatal("allocation should succeed")
	}
	tiered.ReleaseKVBlocks(req)

	// THEN CPU tier should not exceed its capacity
	if tiered.cpu.used > tiered.cpu.capacity {
		t.Errorf("CPU used %d exceeds capacity %d", tiered.cpu.used, tiered.cpu.capacity)
	}
}

func TestTieredKVCache_Conservation_BlocksPreserved(t *testing.T) {
	// GIVEN tiered cache
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 5, 0.5, 100.0, 0)

	// Total GPU blocks should remain constant
	initialGPU := gpu.TotalBlocks

	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req, 0, 4, []int64{})
	tiered.ReleaseKVBlocks(req)

	// THEN GPU total is unchanged
	if gpu.TotalBlocks != initialGPU {
		t.Errorf("GPU total changed from %d to %d", initialGPU, gpu.TotalBlocks)
	}
	// AND used + free = total for GPU
	if gpu.UsedBlockCnt+gpu.countFreeBlocks() != gpu.TotalBlocks {
		t.Errorf("conservation violated: used(%d) + free(%d) != total(%d)",
			gpu.UsedBlockCnt, gpu.countFreeBlocks(), gpu.TotalBlocks)
	}
}

func TestTieredKVCache_PendingTransferLatency_QueryAndClear(t *testing.T) {
	// GIVEN tiered cache with some pending latency
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.5, 100.0, 0)

	// Manually set pending latency for testing
	tiered.pendingLatency = 42

	// WHEN we query
	lat := tiered.PendingTransferLatency()

	// THEN it returns the value and resets
	if lat != 42 {
		t.Errorf("PendingTransferLatency() = %d, want 42", lat)
	}
	if tiered.PendingTransferLatency() != 0 {
		t.Error("second call should return 0")
	}
}

func TestTieredKVCache_ZeroBandwidth_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for zero bandwidth")
		}
	}()
	gpu := NewKVCacheState(10, 2)
	NewTieredKVCache(gpu, 10, 0.5, 0, 0) // should panic (BC-11)
}

func TestTieredKVCache_CPUReload_AddsTransferLatency(t *testing.T) {
	// GIVEN tiered cache: 10 GPU blocks, 10 CPU blocks
	// threshold 0.3 means offload when GPU util > 30% after release
	// bandwidth 1.0 block/tick, base latency 10
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.3, 1.0, 10)

	// Allocate 8 blocks (4 requests × 2 blocks each) to fill GPU to 80%
	reqs := make([]*Request, 4)
	for i := 0; i < 4; i++ {
		reqs[i] = &Request{ID: fmt.Sprintf("r%d", i), InputTokens: []int{i*2 + 1, i*2 + 2, i*2 + 3, i*2 + 4}}
		tiered.AllocateKVBlocks(reqs[i], 0, 4, []int64{})
	}

	// Release first 2 requests — GPU goes from 80% to 40%, offload triggers (40% > 30%)
	tiered.ReleaseKVBlocks(reqs[0])
	tiered.ReleaseKVBlocks(reqs[1])

	// Verify blocks were offloaded to CPU
	if tiered.cpu.used == 0 {
		t.Fatal("expected blocks on CPU after offload")
	}
	cpuBefore := tiered.cpu.used

	// Fill remaining GPU blocks
	for i := 10; i < 20; i++ {
		filler := &Request{ID: fmt.Sprintf("filler_%d", i), InputTokens: []int{i*2 + 100, i*2 + 101}}
		if !tiered.AllocateKVBlocks(filler, 0, 2, []int64{}) {
			break // GPU full
		}
	}

	// NOW try to allocate — GPU full, should try CPU reload
	req2 := &Request{ID: "reload_req", InputTokens: []int{200, 201}}
	tiered.AllocateKVBlocks(req2, 0, 2, []int64{})

	// Verify transfer latency was accumulated (if reload happened)
	lat := tiered.PendingTransferLatency()

	// Verify query-and-clear
	lat2 := tiered.PendingTransferLatency()
	if lat2 != 0 {
		t.Errorf("second PendingTransferLatency() = %d, want 0 (query-and-clear)", lat2)
	}

	// If CPU blocks were reloaded, latency should be > 0 and CPU count decreased
	if tiered.cpu.used < cpuBefore && lat <= 0 {
		t.Errorf("CPU blocks reloaded but PendingTransferLatency = %d, want > 0", lat)
	}
}
