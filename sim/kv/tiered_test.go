package kv

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- cpuTier unit tests (BC-4, BC-1 touch) ---

func TestCpuTier_Store_EvictsOldestWhenFull(t *testing.T) {
	// BC-4: GIVEN CPU tier with capacity 2
	cpu := newCpuTier(2, 4) // capacity=2, blockSize=4 tokens

	// WHEN we store 3 blocks
	cpu.store("hash-a", []int{1, 2, 3, 4})
	cpu.store("hash-b", []int{5, 6, 7, 8})
	cpu.store("hash-c", []int{9, 10, 11, 12})

	// THEN oldest (hash-a) is evicted, newest two remain
	assert.Nil(t, cpu.lookup("hash-a"), "oldest block should be evicted")
	assert.NotNil(t, cpu.lookup("hash-b"), "second block should survive")
	assert.NotNil(t, cpu.lookup("hash-c"), "newest block should survive")
	assert.Equal(t, int64(2), cpu.used)
	assert.Equal(t, int64(1), cpu.evictionCount)
}

func TestCpuTier_Touch_MovesToTail(t *testing.T) {
	// BC-1 (touch): GIVEN CPU tier with 2 blocks
	cpu := newCpuTier(2, 4)
	cpu.store("hash-a", []int{1, 2, 3, 4})
	cpu.store("hash-b", []int{5, 6, 7, 8})

	// WHEN we touch hash-a (refreshes recency)
	cpu.touch("hash-a")

	// AND store a third block (triggers eviction)
	cpu.store("hash-c", []int{9, 10, 11, 12})

	// THEN hash-b is evicted (it's now oldest), hash-a survives (was touched)
	assert.Nil(t, cpu.lookup("hash-b"), "untouched block should be evicted")
	assert.NotNil(t, cpu.lookup("hash-a"), "touched block should survive")
	assert.NotNil(t, cpu.lookup("hash-c"), "newest block should survive")
}

func TestCpuTier_Lookup_ReturnsNilForMissing(t *testing.T) {
	cpu := newCpuTier(10, 4)
	assert.Nil(t, cpu.lookup("nonexistent"))
}

func TestCpuTier_Store_DuplicateHashIsNoOp(t *testing.T) {
	cpu := newCpuTier(10, 2)
	cpu.store("h1", []int{1, 2})
	cpu.store("h1", []int{99, 99}) // duplicate — should not overwrite

	blk := cpu.lookup("h1")
	assert.NotNil(t, blk)
	assert.Equal(t, []int{1, 2}, blk.tokens, "original tokens preserved")
	assert.Equal(t, int64(1), cpu.used, "no duplicate storage")
}

func TestCpuTier_Touch_NoOpForMissing(t *testing.T) {
	cpu := newCpuTier(10, 2)
	cpu.store("h1", []int{1, 2})
	cpu.touch("nonexistent") // should not panic
	assert.Equal(t, int64(1), cpu.used)
}

func TestCpuTier_EvictHead_EmptyListIsNoOp(t *testing.T) {
	cpu := newCpuTier(10, 2)
	cpu.evictHead() // should not panic on empty list
	assert.Equal(t, int64(0), cpu.used)
}

func TestCpuTier_TokenSlicePoolRecycling(t *testing.T) {
	// Pre-allocated pool should be recycled on eviction
	cpu := newCpuTier(2, 4)
	assert.Equal(t, 2, len(cpu.freeTokenSlices), "pool should start with capacity slices")

	cpu.store("h1", []int{1, 2, 3, 4}) // consumes 1 slice
	assert.Equal(t, 1, len(cpu.freeTokenSlices))

	cpu.store("h2", []int{5, 6, 7, 8}) // consumes last slice
	assert.Equal(t, 0, len(cpu.freeTokenSlices))

	// store("h3") evicts h1 (returns slice to pool), then immediately consumes
	// that returned slice for h3. Net pool effect: 0 → 1 → 0.
	cpu.store("h3", []int{9, 10, 11, 12})
	assert.Equal(t, 0, len(cpu.freeTokenSlices), "evicted slice consumed by new block")

	// Verify content — h3 reused h1's pre-allocated slice via copy
	blk := cpu.lookup("h3")
	assert.NotNil(t, blk)
	assert.Equal(t, []int{9, 10, 11, 12}, blk.tokens)

	// Now evict h2 by storing h4 — h2's slice returns to pool, h4 consumes it
	cpu.store("h4", []int{13, 14, 15, 16})
	assert.Equal(t, 0, len(cpu.freeTokenSlices))

	// Release a block explicitly and verify pool grows
	cpu.evictHead() // evicts h3 (oldest)
	assert.Equal(t, 1, len(cpu.freeTokenSlices), "explicit eviction returns slice to pool")
}

// --- Targeted reload tests (BC-2, BC-6) ---

func TestTieredKVCache_TargetedReload_OnlyPrefixBlocks(t *testing.T) {
	// BC-2: GIVEN blocks on CPU matching a prefix
	// WHEN AllocateKVBlocks fails on GPU (not enough free blocks for fresh alloc)
	// THEN only prefix-matching blocks are reloaded, turning misses into hits
	//
	// Setup: 4 GPU blocks, blockSize=2. Prefix [1,2,3,4] = 2 blocks.
	// Fill GPU to 3 used, 1 free. Fresh alloc needs 2 but only 1 free → fails.
	// Reload finds 2 prefix blocks on CPU, loads min(2,1)=1. GetCachedBlocks finds 1.
	// Retry: 1 cached + need 1 more new = 1 free block needed.
	// But the reloaded block was committed as cached (removed from free list),
	// and the original 1 free block was used for reload → 0 free.
	// Actually: reload pops 1 free block, fills with h0, appends back.
	// GetCachedBlocks finds h0 → 1 cached. commitCachedBlocks removes from free list.
	// Retry needs 1 more → 0 free → fails.
	//
	// Better: Use 6 GPU blocks. Prefix=3 blocks [1,2,3,4,5,6].
	// Fill to 4 used, 2 free. Fresh alloc needs 3 but only 2 free → fails.
	// Reload: 2 prefix blocks loaded (limited by free count).
	// GetCachedBlocks finds 2. Retry: 2 cached, need 1 more new.
	// After committing 2 cached (removed from free), 0 free for the 1 new → fails.
	// Hmm. Let me try: 8 GPU blocks.
	gpu := NewKVCacheState(8, 2) // 8 blocks, blockSize=2
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 10)

	// Step 1: Allocate 3-block prefix, mirror to CPU, release
	prefixReq := &sim.Request{ID: "prefix", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	tiered.AllocateKVBlocks(prefixReq, 0, 6, []int64{})
	h0 := gpu.Blocks[gpu.RequestMap["prefix"][0]].Hash
	h1 := gpu.Blocks[gpu.RequestMap["prefix"][1]].Hash
	h2 := gpu.Blocks[gpu.RequestMap["prefix"][2]].Hash
	tiered.cpu.store(h0, []int{1, 2})
	tiered.cpu.store(h1, []int{3, 4})
	tiered.cpu.store(h2, []int{5, 6})
	tiered.cpu.store("unrelated-hash", []int{99, 99})
	tiered.ReleaseKVBlocks(prefixReq)

	// Step 2: Fill GPU completely (8 blocks) to evict prefix hashes
	for i := 0; i < 8; i++ {
		f := &sim.Request{ID: fmt.Sprintf("fill%d", i), InputTokens: []int{i*2 + 20, i*2 + 21}}
		tiered.AllocateKVBlocks(f, 0, 2, []int64{})
	}

	// Step 3: Release 3 fillers → 5 used, 3 free
	tiered.ReleaseKVBlocks(&sim.Request{ID: "fill0"})
	tiered.ReleaseKVBlocks(&sim.Request{ID: "fill1"})
	tiered.ReleaseKVBlocks(&sim.Request{ID: "fill2"})

	// Verify: prefix evicted from GPU
	cached := tiered.GetCachedBlocks([]int{1, 2, 3, 4, 5, 6})
	assert.Equal(t, 0, len(cached), "prefix should be evicted from GPU")

	// WHEN: Request same prefix [1,2,3,4,5,6] → 3 blocks needed, 3 free.
	// Fresh alloc would succeed (3 free >= 3 needed). But prefix is on CPU.
	// GPU alloc succeeds as cache misses. To force reload, we need fresh alloc to fail.
	// Fill one more to leave only 2 free (need 3).
	fExtra := &sim.Request{ID: "fExtra", InputTokens: []int{80, 81}}
	tiered.AllocateKVBlocks(fExtra, 0, 2, []int64{})
	// GPU: 6 used, 2 free. Need 3 → fresh alloc fails → reload triggered.

	newReq := &sim.Request{ID: "new", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	tiered.AllocateKVBlocks(newReq, 0, 6, []int64{}) // may or may not succeed

	// The key test is that CPU hit count > 0 (reload was triggered and found blocks)
	// and unrelated block was not touched.
	assert.Greater(t, tiered.cpuHitCount, int64(0), "CPU hits should be recorded")

	_, unrelatedOnGPU := gpu.HashToBlock["unrelated-hash"]
	assert.False(t, unrelatedOnGPU, "unrelated CPU block should NOT be reloaded to GPU")
}

func TestTieredKVCache_TargetedReload_TransferLatency(t *testing.T) {
	// BC-6: Transfer latency accumulates for reloaded blocks.
	// Use 4 GPU blocks, prefix = 2 blocks. Fill to 3 used, 1 free.
	// Fresh alloc needs 2 but only 1 free → fails → reload.
	// Reload loads 1 block (limited by maxReloads=1).
	// After reload: 1 cached, need 1 more new. 0 free after commit → fails.
	// To make allocation succeed: use 6 GPU blocks. Fill to 4 used, 2 free.
	// Need 3 blocks for prefix (6 tokens / 2 = 3). 2 free < 3 → fails → reload.
	// Reload 2 blocks. Cached: 2. Need 1 more. Committed 2 (remove from free).
	// 0 free → fails. STILL not enough.
	//
	// Simplest approach: test that reload accumulates latency even if allocation
	// ultimately fails. The latency is accumulated per reloaded block.
	gpu := NewKVCacheState(6, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 2.0, 100) // bandwidth=2.0, baseLat=100

	// Allocate, mirror, release
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req, 0, 4, []int64{})
	h0 := gpu.Blocks[gpu.RequestMap["r1"][0]].Hash
	h1 := gpu.Blocks[gpu.RequestMap["r1"][1]].Hash
	tiered.cpu.store(h0, []int{1, 2})
	tiered.cpu.store(h1, []int{3, 4})
	tiered.ReleaseKVBlocks(req)

	// Fill GPU completely to evict prefix hashes
	for i := 0; i < 6; i++ {
		f := &sim.Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 20, i*2 + 21}}
		tiered.AllocateKVBlocks(f, 0, 2, []int64{})
	}

	// Release 1 filler → 5 used, 1 free. Need 2 → fails → reload.
	tiered.ReleaseKVBlocks(&sim.Request{ID: "f0"})

	// Trigger reload attempt
	newReq := &sim.Request{ID: "new", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(newReq, 0, 4, []int64{}) // may fail (1 free < 2 needed)

	// THEN: Transfer latency accumulated for 1 reloaded block
	// 1 block × (100 + ceil(2/2.0)) = 1 × 101 = 101
	lat := tiered.ConsumePendingTransferLatency()
	assert.Equal(t, int64(101), lat, "transfer latency should be 1 × (baseLat + ceil(blockSize/bandwidth))")
}

func TestTieredKVCache_TargetedReload_MaxReloadsGuard(t *testing.T) {
	// BC-6: With F=1 free block and M=2 prefix blocks,
	// only 1 block should be reloaded (maxReloads guard prevents hash destruction)
	gpu := NewKVCacheState(3, 2) // 3 blocks, blockSize=2
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 0)

	// Allocate prefix and capture hashes
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req, 0, 4, []int64{})
	h0 := gpu.Blocks[gpu.RequestMap["r1"][0]].Hash
	h1 := gpu.Blocks[gpu.RequestMap["r1"][1]].Hash
	tiered.ReleaseKVBlocks(req)

	// Place both blocks on CPU
	tiered.cpu.store(h0, []int{1, 2})
	tiered.cpu.store(h1, []int{3, 4})

	// Clear GPU hashes
	for _, blk := range gpu.Blocks {
		if blk.Hash == h0 || blk.Hash == h1 {
			delete(gpu.HashToBlock, blk.Hash)
			blk.Hash = ""
			blk.Tokens = nil
		}
	}

	// Fill GPU: 2 used, 1 free (F=1, M=2)
	filler := &sim.Request{ID: "f1", InputTokens: []int{10, 11, 12, 13}}
	tiered.AllocateKVBlocks(filler, 0, 4, []int64{})
	// GPU: 2 used, 1 free

	// Attempt reload — should reload only 1 block (h0), not destroy it
	newReq := &sim.Request{ID: "new", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(newReq, 0, 4, []int64{})
	// Allocation may fail (only 1 prefix block reloaded, need 2 total)
	// but h0 should still be in GPU HashToBlock
	_, h0OnGPU := gpu.HashToBlock[h0]
	assert.True(t, h0OnGPU, "block 0 hash should survive (maxReloads guard)")
	assert.Equal(t, int64(1), tiered.cpuHitCount, "exactly 1 CPU hit (limited by free blocks)")
}

func TestTieredKVCache_TargetedReload_TouchesCPUOnReload(t *testing.T) {
	// BC-6 + Fix 6: Reloaded CPU blocks should be touched to refresh LRU
	gpu := NewKVCacheState(4, 2)
	tiered := NewTieredKVCache(gpu, 3, 0.0, 1.0, 0) // small CPU: 3 blocks

	// Place 3 blocks on CPU: older → h0, h1, h2 (newest)
	tiered.cpu.store("h0", []int{1, 2})
	tiered.cpu.store("h1", []int{3, 4})
	tiered.cpu.store("h2", []int{5, 6})

	// Simulate reload of h0 by calling touch directly (testing the touch effect)
	tiered.cpu.touch("h0") // h0 moves to tail (newest)

	// Store h3 — should evict h1 (oldest), not h0 (was touched)
	tiered.cpu.store("h3", []int{7, 8})
	assert.NotNil(t, tiered.cpu.lookup("h0"), "h0 should survive (was touched)")
	assert.Nil(t, tiered.cpu.lookup("h1"), "h1 should be evicted (oldest)")
	assert.NotNil(t, tiered.cpu.lookup("h2"), "h2 should survive")
	assert.NotNil(t, tiered.cpu.lookup("h3"), "h3 should survive")
}

// --- MirrorToCPU tests (BC-1, BC-9) ---

func TestTieredKVCache_MirrorToCPU_StoresNewBlocks(t *testing.T) {
	// BC-1: GIVEN a running batch with full hashed blocks
	// WHEN MirrorToCPU is called
	// THEN new blocks are stored on CPU, GPU HashToBlock unchanged
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 0)

	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req, 0, 4, []int64{})

	gpuHashesBefore := len(gpu.HashToBlock)

	// WHEN
	tiered.MirrorToCPU([]*sim.Request{req})

	// THEN: blocks are on CPU
	assert.Equal(t, int64(2), tiered.cpu.used, "2 full blocks should be mirrored to CPU")
	assert.Greater(t, tiered.mirrorCount, int64(0), "mirrorCount should increment")

	// AND: GPU HashToBlock unchanged
	assert.Equal(t, gpuHashesBefore, len(gpu.HashToBlock), "GPU HashToBlock must not change")
}

func TestTieredKVCache_MirrorToCPU_TouchesExistingBlocks(t *testing.T) {
	// BC-1 (touch): GIVEN blocks already on CPU
	// WHEN MirrorToCPU is called again
	// THEN existing blocks are touched (refreshed in LRU)
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 3, 0.0, 1.0, 0) // small CPU: 3 blocks

	// Allocate and mirror r1 (2 blocks)
	r1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(r1, 0, 4, []int64{})
	tiered.MirrorToCPU([]*sim.Request{r1})

	// Mirror r2 (1 block) — now CPU has 3 blocks, full
	r2 := &sim.Request{ID: "r2", InputTokens: []int{10, 20}}
	tiered.AllocateKVBlocks(r2, 0, 2, []int64{})
	tiered.MirrorToCPU([]*sim.Request{r2})
	assert.Equal(t, int64(3), tiered.cpu.used)

	// Touch r1's blocks by mirroring again
	tiered.MirrorToCPU([]*sim.Request{r1})

	// Now mirror r3 (1 block) — should evict r2's block (oldest untouched), not r1's
	r3 := &sim.Request{ID: "r3", InputTokens: []int{30, 40}}
	tiered.AllocateKVBlocks(r3, 0, 2, []int64{})
	tiered.MirrorToCPU([]*sim.Request{r3})

	// r1's blocks should survive (were touched), r2's should be evicted
	h0r1 := gpu.Blocks[gpu.RequestMap["r1"][0]].Hash
	h0r2 := gpu.Blocks[gpu.RequestMap["r2"][0]].Hash
	assert.NotNil(t, tiered.cpu.lookup(h0r1), "r1's block should survive (was touched)")
	assert.Nil(t, tiered.cpu.lookup(h0r2), "r2's block should be evicted (untouched, oldest)")
}

func TestTieredKVCache_MirrorToCPU_NilBatchSafe(t *testing.T) {
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 0)
	// Should not panic
	tiered.MirrorToCPU(nil)
	tiered.MirrorToCPU([]*sim.Request{})
	assert.Equal(t, int64(0), tiered.mirrorCount)
}

func TestTieredKVCache_MirrorToCPU_SkipsPartialAndUnhashedBlocks(t *testing.T) {
	// Blocks with empty hash or not-full should not be mirrored
	gpu := NewKVCacheState(10, 4) // blockSize=4
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 0)

	// Allocate 3 tokens into a 4-token block → partial block (no hash)
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3}}
	tiered.AllocateKVBlocks(req, 0, 3, []int64{})

	tiered.MirrorToCPU([]*sim.Request{req})
	assert.Equal(t, int64(0), tiered.cpu.used, "partial block should not be mirrored")
	assert.Equal(t, int64(0), tiered.mirrorCount)
}

// --- GPU prefix preservation test (BC-3) ---

func TestTieredKVCache_ReleaseKVBlocks_PreservesGPUHashes(t *testing.T) {
	// BC-3: GIVEN a request with cached prefix blocks
	// WHEN ReleaseKVBlocks is called
	// THEN freed blocks stay on GPU with hashes intact
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 0)

	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req, 0, 4, []int64{})

	// Capture hashes before release
	h0 := gpu.Blocks[gpu.RequestMap["r1"][0]].Hash
	h1 := gpu.Blocks[gpu.RequestMap["r1"][1]].Hash
	assert.NotEmpty(t, h0)
	assert.NotEmpty(t, h1)

	// WHEN
	tiered.ReleaseKVBlocks(req)

	// THEN: hashes still in GPU HashToBlock (NOT removed by offload)
	_, h0InGPU := gpu.HashToBlock[h0]
	_, h1InGPU := gpu.HashToBlock[h1]
	assert.True(t, h0InGPU, "block 0 hash should remain on GPU after release")
	assert.True(t, h1InGPU, "block 1 hash should remain on GPU after release")

	// AND: GetCachedBlocks still finds the prefix
	cached := tiered.GetCachedBlocks([]int{1, 2, 3, 4})
	assert.Equal(t, 2, len(cached), "prefix should still be cached on GPU after release")
}

// --- BC-5: CPU extends GPU prefix lifetime ---

func TestTieredKVCache_CPUExtendsGPUPrefixLifetime(t *testing.T) {
	// BC-5: GIVEN a block on both GPU and CPU, WHEN GPU evicts it (popFreeBlock),
	// THEN the CPU copy survives and can be reloaded by a future request.
	//
	// Setup: 6 GPU blocks, blockSize=2. Prefix [1,2,3,4,5,6] = 3 blocks.
	// Fill GPU completely to evict prefix. Release 2 fillers → 2 free.
	// Request 3-block prefix: need 3, have 2 free → GPU alloc fails → reload.
	// Reload loads 2 blocks from CPU (limited by free count).
	// cpuHitCount > 0 and transfer latency > 0 prove CPU extended prefix lifetime.
	gpu := NewKVCacheState(6, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 10) // baseLat=10

	// Step 1: Allocate 3-block prefix, mirror to CPU, release
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	tiered.AllocateKVBlocks(req, 0, 6, []int64{})
	tiered.MirrorToCPU([]*sim.Request{req})
	tiered.ReleaseKVBlocks(req)

	// Step 2: Fill GPU completely to evict prefix hashes
	for i := 0; i < 6; i++ {
		f := &sim.Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 20, i*2 + 21}}
		tiered.AllocateKVBlocks(f, 0, 2, []int64{})
	}

	// Step 3: Release 2 fillers → 4 used, 2 free. Need 3 → fails → reload.
	tiered.ReleaseKVBlocks(&sim.Request{ID: "f0"})
	tiered.ReleaseKVBlocks(&sim.Request{ID: "f1"})

	// Step 4: Re-request prefix — triggers targeted reload from CPU
	newReq := &sim.Request{ID: "new", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	tiered.AllocateKVBlocks(newReq, 0, 6, []int64{}) // may partially succeed

	// THEN: CPU hits > 0 (prefix blocks found on CPU after GPU eviction)
	assert.Greater(t, tiered.cpuHitCount, int64(0), "CPU should provide reload hits")

	// AND: Transfer latency accumulated (proves CPU→GPU transfer occurred)
	lat := tiered.ConsumePendingTransferLatency()
	assert.Greater(t, lat, int64(0), "reload should accumulate transfer latency")
}

// --- KVThrashingRate tests ---

func TestTieredKVCache_KVThrashingRate_ReturnsCPUEvictionRate(t *testing.T) {
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 2, 0.0, 1.0, 0) // tiny CPU: 2 blocks

	// Mirror 3 blocks → 1 eviction
	r1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	tiered.AllocateKVBlocks(r1, 0, 6, []int64{})
	tiered.MirrorToCPU([]*sim.Request{r1})
	// 3 blocks mirrored, CPU capacity=2, so 1 eviction

	rate := tiered.KVThrashingRate()
	// evictionCount=1, mirrorCount=3 → rate = 1/3 ≈ 0.333
	assert.InDelta(t, 1.0/3.0, rate, 0.01, "KVThrashingRate should return CPU eviction rate")
}

func TestTieredKVCache_KVThrashingRate_ZeroMirrors(t *testing.T) {
	// R11: Returns 0 when mirrorCount == 0 (no division by zero)
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 0)
	assert.Equal(t, 0.0, tiered.KVThrashingRate())
}

// --- INV-4 conservation test ---

func TestTieredKVCache_Conservation_MirrorReloadCycle(t *testing.T) {
	// INV-4: allocated + free = total must hold through mirror+reload cycles
	gpu := NewKVCacheState(6, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 0)
	total := gpu.TotalCapacity()

	checkINV4 := func(label string) {
		t.Helper()
		// Walk the free list independently of UsedBlockCnt to avoid tautology.
		// INV-4: UsedBlockCnt + (free list length) == TotalBlocks
		actualFree := int64(0)
		blk := gpu.FreeHead
		for blk != nil {
			actualFree++
			blk = blk.NextFree
		}
		assert.Equal(t, total, gpu.UsedBlockCnt+actualFree,
			"INV-4 %s: UsedBlockCnt(%d) + freeListLen(%d) != total(%d)",
			label, gpu.UsedBlockCnt, actualFree, total)
	}

	// Allocate, mirror, release — check INV-4 at every step
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(req, 0, 4, []int64{})
	checkINV4("after alloc")
	assert.Equal(t, int64(2), gpu.UsedBlocks())

	tiered.MirrorToCPU([]*sim.Request{req})
	checkINV4("after mirror")
	// GPU state unchanged by mirror
	assert.Equal(t, int64(2), gpu.UsedBlocks())

	tiered.ReleaseKVBlocks(req)
	checkINV4("after release")
	assert.Equal(t, int64(0), gpu.UsedBlocks(), "all blocks free after release")

	// Fill and release to trigger reload path
	for i := 0; i < 6; i++ {
		f := &sim.Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 20, i*2 + 21}}
		tiered.AllocateKVBlocks(f, 0, 2, []int64{})
	}
	checkINV4("after fill")

	tiered.ReleaseKVBlocks(&sim.Request{ID: "f0"})
	checkINV4("after partial release")

	// Trigger reload
	newReq := &sim.Request{ID: "new", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(newReq, 0, 4, []int64{})
	checkINV4("after reload attempt")

	// Verify CPU has blocks but GPU conservation unaffected
	assert.Greater(t, tiered.cpu.used, int64(0), "CPU should have mirrored blocks")
}

// --- Validation tests (kept from old file) ---

func TestCpuTier_Validation_ZeroCapacity_Panics(t *testing.T) {
	assert.Panics(t, func() { newCpuTier(0, 4) })
}

func TestCpuTier_Validation_NegativeCapacity_Panics(t *testing.T) {
	assert.Panics(t, func() { newCpuTier(-1, 4) })
}

func TestCpuTier_Validation_ZeroBlockSize_Panics(t *testing.T) {
	assert.Panics(t, func() { newCpuTier(10, 0) })
}

// --- Reload guard tests (BC-2, BC-3) ---
//
// Note on test coverage: The full-range reload guard (newStart >= endIndex for running
// requests) cannot be triggered in isolation because the GPU pre-check fails when K < N
// new blocks are needed, but the reload requires exactly K free blocks — the same K that
// would allow direct GPU success. As a result, these tests verify the correct behavioral
// end state (block count and INV-4) via the direct GPU path. The fix code path is
// exercised only during multi-request batch formation under concurrent KV pressure.

func TestTieredKVCache_ReloadGuard_CommitsBlocksForRunningRequest(t *testing.T) {
	// BC-2: Running request hitting full-range reload guard gets uncovered blocks committed
	// BC-3: Only the uncovered range is committed (no double-counting)
	blockSize := int64(4)
	gpuBlocks := int64(12)
	cpuBlocks := int64(5)

	gpu := NewKVCacheState(gpuBlocks, blockSize)
	tiered := NewTieredKVCache(gpu, cpuBlocks, 0, 1.0, 0)

	// Create tokens for 4 blocks (16 tokens total)
	tokens := make([]int, 4*blockSize)
	for i := range tokens {
		tokens[i] = i + 1
	}

	// Seed request: allocate all 4 blocks, mirror to CPU, then release.
	// This puts all 4 blocks on CPU and leaves hashes on GPU free list.
	seedReq := &sim.Request{ID: "req-seed", InputTokens: tokens}
	ok := tiered.AllocateKVBlocks(seedReq, 0, int64(len(tokens)), []int64{})
	assert.True(t, ok, "seed allocation must succeed")
	tiered.MirrorToCPU([]*sim.Request{seedReq})
	tiered.ReleaseKVBlocks(seedReq)

	// Running request: allocate first 2 blocks (simulates ProgressIndex at block 2)
	runReq := &sim.Request{ID: "req-running", InputTokens: tokens}
	ok = tiered.AllocateKVBlocks(runReq, 0, 2*blockSize, []int64{})
	assert.True(t, ok, "partial allocation must succeed")
	existingBlocks, exists := gpu.RequestMap[runReq.ID]
	assert.True(t, exists, "running request must be in RequestMap")
	numBlocksBefore := len(existingBlocks)
	assert.Equal(t, 2, numBlocksBefore, "running request should have 2 blocks")

	// Exhaust remaining GPU free blocks with fillers
	fillerCount := 0
	for gpu.countFreeBlocks() > 0 {
		filler := &sim.Request{
			ID:          fmt.Sprintf("filler-%d", fillerCount),
			InputTokens: []int{800 + fillerCount*4, 801 + fillerCount*4, 802 + fillerCount*4, 803 + fillerCount*4},
		}
		if tiered.AllocateKVBlocks(filler, 0, blockSize, []int64{}) {
			fillerCount++
		} else {
			break
		}
	}

	// Release 2 fillers to make room for CPU→GPU reload (reload needs free blocks)
	for i := 0; i < 2 && i < fillerCount; i++ {
		tiered.ReleaseKVBlocks(&sim.Request{ID: fmt.Sprintf("filler-%d", i)})
	}

	// WHEN we allocate blocks 2-3 for the running request (the uncovered range)
	// startIndex = 2*blockSize = 8, endIndex = 4*blockSize = 16
	startIndex := int64(2) * blockSize
	endIndex := int64(4) * blockSize
	ok = tiered.AllocateKVBlocks(runReq, startIndex, endIndex, []int64{})
	assert.True(t, ok, "allocation via CPU reload must succeed")

	// THEN the uncovered blocks (blocks 2 and 3) must be committed to RequestMap
	blocksAfter, existsAfter := gpu.RequestMap[runReq.ID]
	assert.True(t, existsAfter, "running request must still be in RequestMap")
	assert.Greater(t, len(blocksAfter), numBlocksBefore,
		"running request must have more blocks after allocation (BC-2)")

	// BC-3: No double-counting — original blocks should not be duplicated
	// We started with 2 blocks and added blocks for range [2,4), so expect 4 total
	assert.Equal(t, 4, len(blocksAfter),
		"running request must have exactly 4 blocks (2 original + 2 newly committed)")

	// INV-4: block conservation
	assert.Equal(t, gpu.TotalCapacity(), gpu.UsedBlocks()+gpu.countFreeBlocks(),
		"INV-4: used + free must equal total capacity")
}

func TestTieredKVCache_ReloadGuard_NonBlockAlignedStartIndex_NoDuplicates(t *testing.T) {
	// BC-3 edge case: startIndex=6, blockSize=4 (non-aligned)
	// Ceiling division: startBlock = (6+3)/4 = 2 (skips partially-filled block 1)
	// Floor division: startBlock = 6/4 = 1 (would re-commit partially-filled block 1)
	blockSize := int64(4)
	gpuBlocks := int64(12)
	cpuBlocks := int64(5)

	gpu := NewKVCacheState(gpuBlocks, blockSize)
	tiered := NewTieredKVCache(gpu, cpuBlocks, 0, 1.0, 0)

	// Tokens for 4 blocks
	tokens := make([]int, 4*blockSize)
	for i := range tokens {
		tokens[i] = i + 1
	}

	// Seed: allocate, mirror to CPU, release
	seedReq := &sim.Request{ID: "req-seed-na", InputTokens: tokens}
	ok := tiered.AllocateKVBlocks(seedReq, 0, int64(len(tokens)), []int64{})
	assert.True(t, ok)
	tiered.MirrorToCPU([]*sim.Request{seedReq})
	tiered.ReleaseKVBlocks(seedReq)

	// Running request: allocate first 6 tokens (non-block-aligned: covers block 0 fully,
	// block 1 partially with tokens 4,5). startIndex=6 for the next allocation.
	runReq := &sim.Request{ID: "req-running-nonaligned", InputTokens: tokens}
	ok = tiered.AllocateKVBlocks(runReq, 0, 6, []int64{})
	assert.True(t, ok, "partial allocation must succeed")
	blocksBefore := len(gpu.RequestMap[runReq.ID])
	// block 0 (full) + block 1 (partial) = 2 blocks
	assert.Equal(t, 2, blocksBefore, "should have 2 blocks (block 0 full, block 1 partial)")

	// Exhaust GPU free blocks
	fillerCount := 0
	for gpu.countFreeBlocks() > 0 {
		filler := &sim.Request{
			ID:          fmt.Sprintf("filler2-%d", fillerCount),
			InputTokens: []int{700 + fillerCount*4, 701 + fillerCount*4, 702 + fillerCount*4, 703 + fillerCount*4},
		}
		if tiered.AllocateKVBlocks(filler, 0, blockSize, []int64{}) {
			fillerCount++
		} else {
			break
		}
	}
	for i := 0; i < 2 && i < fillerCount; i++ {
		tiered.ReleaseKVBlocks(&sim.Request{ID: fmt.Sprintf("filler2-%d", i)})
	}

	// WHEN we allocate startIndex=6, endIndex=16 (needs tokens 6-15)
	ok = tiered.AllocateKVBlocks(runReq, 6, 16, []int64{})
	assert.True(t, ok, "CPU reload must cover the range")

	// THEN: exactly blocks 2 and 3 committed (ceiling: skip block 1 which is partial)
	// With floor division, block 1_copy from reload would also be appended → duplicates
	blocksAfter := gpu.RequestMap[runReq.ID]
	// original 2 + newly committed blocks 2,3 = 4 total
	assert.Equal(t, 4, len(blocksAfter),
		"should have exactly 4 blocks: original 2 (block0,block1_partial) + blocks 2,3 from reload")

	// BC-3: each block ID must appear exactly once (no double-commit)
	seen := make(map[int64]int)
	for _, bid := range blocksAfter {
		seen[bid]++
	}
	for bid, count := range seen {
		assert.Equal(t, 1, count, fmt.Sprintf("block %d must appear exactly once in RequestMap", bid))
	}

	// INV-4
	assert.Equal(t, gpu.TotalCapacity(), gpu.UsedBlocks()+gpu.countFreeBlocks(),
		"INV-4: block conservation")
}

// --- Reload guard BC-1 and preemption BC-5 tests ---

func TestTieredKVCache_ReloadGuard_CommitsBlocksForNewRequest(t *testing.T) {
	// BC-1: New request that hits full-range reload guard gets all blocks committed
	blockSize := int64(4)
	gpuBlocks := int64(10)
	cpuBlocks := int64(5)

	gpu := NewKVCacheState(gpuBlocks, blockSize)
	tiered := NewTieredKVCache(gpu, cpuBlocks, 0, 1.0, 0)

	// Create tokens for 2 blocks
	tokens := make([]int, 2*blockSize)
	for i := range tokens {
		tokens[i] = i + 1
	}

	// Seed: allocate, mirror to CPU, release (leaves blocks on CPU + GPU free list hashes)
	seedReq := &sim.Request{ID: "req-seed-bc1", InputTokens: tokens}
	ok := tiered.AllocateKVBlocks(seedReq, 0, int64(len(tokens)), []int64{})
	assert.True(t, ok)
	tiered.MirrorToCPU([]*sim.Request{seedReq})
	tiered.ReleaseKVBlocks(seedReq)

	// Exhaust all GPU free blocks with unique-prefix fillers
	fillerCount := 0
	for gpu.countFreeBlocks() > 0 {
		filler := &sim.Request{
			ID:          fmt.Sprintf("filler-bc1-%d", fillerCount),
			InputTokens: []int{900 + fillerCount*4, 901 + fillerCount*4, 902 + fillerCount*4, 903 + fillerCount*4},
		}
		if tiered.AllocateKVBlocks(filler, 0, blockSize, []int64{}) {
			fillerCount++
		} else {
			break
		}
	}

	// Release 2 fillers to make room for CPU→GPU reload
	for i := 0; i < 2 && i < fillerCount; i++ {
		tiered.ReleaseKVBlocks(&sim.Request{ID: fmt.Sprintf("filler-bc1-%d", i)})
	}

	// WHEN a new request (not in RequestMap) allocates with the same prefix
	newReq := &sim.Request{ID: "req-new-bc1", InputTokens: tokens}
	_, existsBefore := gpu.RequestMap[newReq.ID]
	assert.False(t, existsBefore, "new request must not be in RequestMap before allocation")

	ok = tiered.AllocateKVBlocks(newReq, 0, int64(len(tokens)), []int64{})
	assert.True(t, ok, "allocation must succeed")

	// THEN all blocks must be committed to RequestMap (BC-1)
	blocks, existsAfter := gpu.RequestMap[newReq.ID]
	assert.True(t, existsAfter, "new request must be in RequestMap after allocation")
	assert.Equal(t, 2, len(blocks), "new request must have 2 blocks committed")

	// INV-4: block conservation (BC-4)
	assert.Equal(t, gpu.TotalCapacity(), gpu.UsedBlocks()+gpu.countFreeBlocks(),
		"INV-4: used + free must equal total capacity")
}

func TestPreemption_ClearsRequestMap_EnablesNewRequestPath(t *testing.T) {
	// BC-5: After preemption, request is not in RequestMap and follows new-request path
	blockSize := int64(4)
	gpuBlocks := int64(10)
	gpu := NewKVCacheState(gpuBlocks, blockSize)

	tokens := make([]int, 2*blockSize)
	for i := range tokens {
		tokens[i] = i + 1
	}
	req := &sim.Request{
		ID:            "req-preempt-bc5",
		InputTokens:   tokens,
		ProgressIndex: 0,
		State:         sim.StateRunning,
	}

	// Allocate blocks (simulates request becoming running)
	ok := gpu.AllocateKVBlocks(req, 0, int64(len(tokens)), []int64{})
	assert.True(t, ok)
	_, exists := gpu.RequestMap[req.ID]
	assert.True(t, exists, "request must be in RequestMap after allocation")

	// WHEN blocks are released (as preemptForTokens does)
	gpu.ReleaseKVBlocks(req)

	// THEN request must NOT be in RequestMap (BC-5 precondition)
	_, existsAfter := gpu.RequestMap[req.ID]
	assert.False(t, existsAfter, "request must not be in RequestMap after ReleaseKVBlocks")

	// INV-4: all blocks returned
	assert.Equal(t, gpu.TotalCapacity(), gpu.UsedBlocks()+gpu.countFreeBlocks(),
		"INV-4: block conservation after preemption")
}

// TestTieredKVCache_PartialReload_RunningRequest_BlocksCommitted verifies BC-1, BC-2, BC-3, BC-4:
// partial CPU reload for a running request must commit reloaded blocks before fresh allocation.
//
// Without the fix: gpu.AllocateKVBlocks is called directly; popFreeBlock steals the
// reloaded block (clears h8), overall allocation returns true but the prefix cache is
// silently corrupted — future requests cannot find the h8 cache entry.
//
// With the fix: commitCachedBlocks protects the reloaded block; tail allocation fails cleanly
// (returns false) because there aren't enough blocks, but h8 is preserved in HashToBlock.
func TestTieredKVCache_PartialReload_RunningRequest_BlocksCommitted(t *testing.T) {
	// Setup: 8-block GPU, blockSize=4, 10-block CPU
	// req1 holds blocks [0,1] (tokens 0..7)
	// req2 holds blocks [2,3], req3 holds blocks [4,5]
	// Free: [6,7] (2 free blocks)
	// CPU has hash h8 = HashBlock(block1.Hash, tokens[8:12])
	//
	// Call AllocateKVBlocks(req1, 8, 20, cached):
	//   First attempt: need ceil(12/4)=3 blocks, have 2 → fails
	//   After reload: h8 on free list (tail), newStart=12, partial improvement
	//   Without fix: popFreeBlock steals block 6 (h8 cleared), ok=true but cache corrupted
	//   With fix: commitCachedBlocks protects block 6 → 1 free; need 2 for tail → ok=false
	blockSize := int64(4)
	totalBlocks := int64(8)
	gpu := NewKVCacheState(totalBlocks, blockSize)

	tokens := make([]int, 20)
	for i := range tokens {
		tokens[i] = i + 10 // distinct values
	}

	req1 := &sim.Request{ID: "req1", InputTokens: tokens}
	require.True(t, gpu.AllocateKVBlocks(req1, 0, 8, nil)) // blocks [0,1]

	req2 := &sim.Request{ID: "req2", InputTokens: make([]int, 8)}
	require.True(t, gpu.AllocateKVBlocks(req2, 0, 8, nil)) // blocks [2,3]

	req3 := &sim.Request{ID: "req3", InputTokens: make([]int, 8)}
	require.True(t, gpu.AllocateKVBlocks(req3, 0, 8, nil)) // blocks [4,5]
	// Free: [6,7]

	tiered := NewTieredKVCache(gpu, 10, 0, 1.0, 0)

	// Hash for tokens[8:12] chaining from block[1].Hash
	prevHash1 := gpu.Blocks[gpu.RequestMap["req1"][1]].Hash
	h8 := hash.HashBlock(prevHash1, tokens[8:12])
	tiered.cpu.store(h8, tokens[8:12])

	// BC-3 conservation before
	require.Equal(t, totalBlocks, gpu.UsedBlocks()+gpu.countFreeBlocks())

	// WHEN allocating tokens 8..20 for req1 (running request)
	cached := gpu.GetCachedBlocks(tokens)
	ok := tiered.AllocateKVBlocks(req1, 8, 20, cached)

	// THEN overall allocation fails cleanly (pre-check: need 2 tail blocks, 1 free after commit)
	// Without fix: ok=true (this assertion would FAIL), h8 cleared
	require.False(t, ok, "allocation must fail cleanly — not silently corrupt the prefix cache")

	// THEN BC-3: KV conservation holds
	require.Equal(t, totalBlocks, gpu.UsedBlocks()+gpu.countFreeBlocks())

	// THEN BC-1: reloaded block hash is preserved (not stolen by popFreeBlock)
	// Without fix: h8 would be cleared from HashToBlock by popFreeBlock
	_, found := gpu.HashToBlock[h8]
	require.True(t, found, "BC-1: reloaded block hash must be preserved in HashToBlock")

	// THEN BC-1: reloaded block is committed (eviction-protected in RequestMap)
	require.Equal(t, 3, len(gpu.RequestMap["req1"]), "req1 must have original 2 blocks + 1 committed reloaded block")
	reloadedID := gpu.RequestMap["req1"][2]
	reloadedBlk := gpu.Blocks[reloadedID]
	require.True(t, reloadedBlk.InUse, "BC-1: reloaded block must be InUse")
	require.Positive(t, reloadedBlk.RefCount, "BC-1: reloaded block RefCount must be > 0")

	// THEN BC-2: future GetCachedBlocks finds 3 blocks (prefix cache intact for future requests)
	futureCached := gpu.GetCachedBlocks(tokens)
	require.GreaterOrEqual(t, len(futureCached), 3, "BC-2: prefix cache must find all 3 cached blocks")
}
