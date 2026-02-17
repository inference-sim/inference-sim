package sim

import "testing"

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
