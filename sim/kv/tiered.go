package kv

import (
	"fmt"
	"math"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// cpuBlock represents a KV block mirrored from GPU to CPU tier.
// Identified by content hash (not GPU block ID) for content-addressable reload.
type cpuBlock struct {
	hash   string    // prefix hash (map key, identifies content)
	tokens []sim.TokenID // token content (for GPU reload); pre-allocated slice, copy-into
	prev   *cpuBlock // LRU doubly-linked list: older block
	next   *cpuBlock // LRU doubly-linked list: newer block
}

// cpuTier is an LRU cache of mirrored GPU blocks, keyed by content hash.
// All operations are O(1) via hash map + doubly-linked list.
type cpuTier struct {
	blocks    map[string]*cpuBlock // hash → block, O(1) lookup
	lruHead   *cpuBlock            // oldest (evict first)
	lruTail   *cpuBlock            // newest (most recently stored/touched)
	capacity  int64
	used      int64
	blockSize int64 // tokens per block (for pre-allocation)

	// Pre-allocated token slices for CPU blocks (eliminates per-mirror GC pressure).
	// Pool of free slices returned on eviction, consumed on store.
	freeTokenSlices [][]sim.TokenID

	evictionCount int64 // total CPU LRU evictions
}

// newCpuTier creates a CPU tier with pre-allocated token storage.
func newCpuTier(capacity int64, blockSize int64) *cpuTier {
	if capacity <= 0 {
		panic(fmt.Sprintf("newCpuTier: capacity must be > 0, got %d", capacity))
	}
	if blockSize <= 0 {
		panic(fmt.Sprintf("newCpuTier: blockSize must be > 0, got %d", blockSize))
	}
	slices := make([][]sim.TokenID, capacity)
	for i := int64(0); i < capacity; i++ {
		slices[i] = make([]sim.TokenID, blockSize)
	}
	return &cpuTier{
		blocks:          make(map[string]*cpuBlock),
		capacity:        capacity,
		blockSize:       blockSize,
		freeTokenSlices: slices,
	}
}

// store adds a block to the CPU tier. If at capacity, evicts LRU-oldest first.
// If the hash already exists, this is a no-op (use touch instead).
func (c *cpuTier) store(hash string, tokens []sim.TokenID) {
	if _, exists := c.blocks[hash]; exists {
		return // already present — caller should use touch
	}
	// Evict if at capacity
	if c.used >= c.capacity {
		c.evictHead()
	}
	// Get a pre-allocated token slice from pool, or allocate as fallback
	var tokSlice []sim.TokenID
	if len(c.freeTokenSlices) > 0 {
		tokSlice = c.freeTokenSlices[len(c.freeTokenSlices)-1]
		c.freeTokenSlices = c.freeTokenSlices[:len(c.freeTokenSlices)-1]
		copy(tokSlice, tokens)
	} else {
		tokSlice = append([]sim.TokenID{}, tokens...) // fallback: allocate
	}
	blk := &cpuBlock{hash: hash, tokens: tokSlice}
	c.blocks[hash] = blk
	c.appendToTail(blk)
	c.used++
}

// touch moves an existing block to the LRU tail (most recently used).
// No-op if hash not found.
func (c *cpuTier) touch(hash string) {
	blk, exists := c.blocks[hash]
	if !exists {
		return
	}
	c.unlink(blk)
	c.appendToTail(blk)
}

// lookup returns the cpuBlock for a hash, or nil if not found.
func (c *cpuTier) lookup(hash string) *cpuBlock {
	return c.blocks[hash]
}

// evictHead removes the LRU-oldest block and returns its token slice to the pool.
func (c *cpuTier) evictHead() {
	if c.lruHead == nil {
		return
	}
	victim := c.lruHead
	c.unlink(victim)
	delete(c.blocks, victim.hash)
	c.used--
	c.evictionCount++
	// Return token slice to pool
	c.freeTokenSlices = append(c.freeTokenSlices, victim.tokens)
	victim.tokens = nil
}

// appendToTail inserts a block at the LRU tail (most recent).
func (c *cpuTier) appendToTail(blk *cpuBlock) {
	blk.next = nil
	blk.prev = c.lruTail
	if c.lruTail != nil {
		c.lruTail.next = blk
	} else {
		c.lruHead = blk
	}
	c.lruTail = blk
}

// unlink removes a block from the LRU doubly-linked list.
func (c *cpuTier) unlink(blk *cpuBlock) {
	if blk.prev != nil {
		blk.prev.next = blk.next
	} else {
		c.lruHead = blk.next
	}
	if blk.next != nil {
		blk.next.prev = blk.prev
	} else {
		c.lruTail = blk.prev
	}
	blk.prev = nil
	blk.next = nil
}

// TieredKVCache composes a GPU KVCacheState with a CPU tier that mirrors in-use blocks.
// GPU prefix cache is preserved on release (vLLM v1 model). CPU tier serves as a secondary
// cache that extends prefix lifetime beyond GPU eviction.
type TieredKVCache struct {
	gpu               *KVCacheState
	cpu               *cpuTier
	transferBandwidth float64
	baseLatency       int64

	// Transfer latency accumulator (query-and-clear)
	pendingLatency int64

	// Metrics counters
	cpuHitCount  int64
	cpuMissCount int64
	mirrorCount  int64 // total blocks stored to CPU via MirrorToCPU
}

// NewTieredKVCache creates a TieredKVCache.
// Panics if gpu is nil, cpuBlocks is non-positive, bandwidth is non-positive/NaN/Inf, or threshold is NaN/Inf.
// The threshold parameter is deprecated in the vLLM v1 mirror model and is ignored.
// A deprecation warning is logged if threshold != 0.
func NewTieredKVCache(gpu *KVCacheState, cpuBlocks int64, threshold, bandwidth float64, baseLat int64) *TieredKVCache {
	if gpu == nil {
		panic("NewTieredKVCache: gpu must not be nil")
	}
	if bandwidth <= 0 || math.IsNaN(bandwidth) || math.IsInf(bandwidth, 0) {
		panic(fmt.Sprintf("NewTieredKVCache: KVTransferBandwidth must be finite and > 0, got %v", bandwidth))
	}
	if math.IsNaN(threshold) || math.IsInf(threshold, 0) {
		panic(fmt.Sprintf("NewTieredKVCache: KVOffloadThreshold must be finite, got %v", threshold))
	}
	if cpuBlocks <= 0 {
		panic(fmt.Sprintf("NewTieredKVCache: cpuBlocks must be > 0, got %d", cpuBlocks))
	}
	if baseLat < 0 {
		panic(fmt.Sprintf("NewTieredKVCache: baseLat must be >= 0, got %d", baseLat))
	}
	// BC-7: Log deprecation warning if threshold is set to non-default
	if threshold != 0 {
		logrus.Warn("KVOffloadThreshold is deprecated in vLLM v1 mirror model and will be ignored. " +
			"GPU prefix cache is now preserved on release; CPU tier is populated via MirrorToCPU.")
	}
	return &TieredKVCache{
		gpu:               gpu,
		cpu:               newCpuTier(cpuBlocks, gpu.BlockSizeTokens),
		transferBandwidth: bandwidth,
		baseLatency:       baseLat,
	}
}

func (t *TieredKVCache) AllocateKVBlocks(req *sim.Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	// BC-D3.3 (#1586): consult the CPU offload tier on EVERY request, before GPU
	// allocation — mirroring vLLM's offload connector (get_num_new_matched_tokens on
	// the normal waiting-queue path, with no reference to GPU pressure). Previously the
	// reload ran only after GPU allocation failed, so under light GPU load a prefix
	// resident on CPU-but-evicted-from-GPU was silently recomputed and counted as a
	// miss — making cache-hit accounting load-shaped rather than load-independent.
	//
	// Resume the prefix hash chain just past the GPU-resident prefix the caller already
	// matched (cachedBlocks), using its last block's hash as the chain seed. This keeps
	// the common fully-GPU-cached case at one hash + one CPU lookup instead of
	// re-hashing the whole prefix from block 0 (hot-path mitigation, evidence P).
	reloadStartBlock := int64(len(cachedBlocks))
	reloadPrevHash := ""
	if reloadStartBlock > 0 {
		reloadPrevHash = t.gpu.Blocks[cachedBlocks[reloadStartBlock-1]].Hash
	}
	// A continuing (running) request already owns its prefix blocks (cachedBlocks is empty
	// on that path); resume past what it owns so we never reload a block position it
	// already holds — in particular its partially-filled last block, whose empty hash
	// would also break the chain seed. Reloading such a position would place a duplicate,
	// full copy on the free list that the running-request path never claims but the GPU
	// pre-check still budgets for, spuriously failing the allocation.
	if owned, ok := t.gpu.RequestMap[req.ID]; ok && int64(len(owned)) > reloadStartBlock {
		reloadStartBlock = int64(len(owned))
		reloadPrevHash = t.gpu.Blocks[owned[len(owned)-1]].Hash
	}
	if t.reloadPrefixFromCPU(req.FullInputTokens(), reloadStartBlock, reloadPrevHash) {
		// Re-compute cached blocks now that CPU content is back on the GPU free list.
		newCached := t.gpu.GetCachedBlocks(req.FullInputTokens())
		newStart := int64(len(newCached)) * t.gpu.BlockSize()
		if newStart > startIndex {
			if newStart >= endIndex {
				// Entire requested range is cached after reload.
				// Commit the appropriate block range to RequestMap.
				endBlock := min((endIndex+t.gpu.BlockSize()-1)/t.gpu.BlockSize(), int64(len(newCached)))
				if _, exists := t.gpu.RequestMap[req.ID]; exists {
					// Running request: commit only the uncovered range using ceiling
					// division to skip the partially-filled last block.
					// ceil(startIndex/blockSize) ensures we don't re-commit the block
					// that is already (fully or partially) tracked in RequestMap —
					// e.g., startIndex=6, blockSize=4: ceil=2 (correct), floor=1 (re-commits
					// partial block 1, double-counting its RefCount).
					// newCached[startBlock:endBlock] is guaranteed non-overlapping with
					// existing RequestMap entries because commitCachedBlocks operates on
					// reload-sourced blocks (different physical blocks than the running
					// request's own partially-filled block).
					startBlock := (startIndex + t.gpu.BlockSize() - 1) / t.gpu.BlockSize()
					if startBlock < endBlock {
						t.gpu.commitCachedBlocks(req.ID, newCached[startBlock:endBlock])
					}
				} else {
					// New request: commit all cached blocks from block 0.
					t.gpu.commitCachedBlocks(req.ID, newCached[:endBlock])
				}
				return true
			}
			// Partial improvement: commit reloaded prefix blocks before allocating tail.
			// Without this, reloaded blocks sit on the GPU free list with RefCount=0 and
			// can be evicted by the subsequent popFreeBlock calls in AllocateKVBlocks,
			// destroying their hashes (R1 silent data loss). Also fixes hash chain: fresh
			// blocks' prevHash must chain from the last reloaded block, which only happens
			// if those blocks are in RequestMap[req.ID] before the fresh allocation loop.
			newStartBlock := newStart / t.gpu.BlockSize()
			if _, exists := t.gpu.RequestMap[req.ID]; exists {
				// Running request: skip blocks already in RequestMap (ceiling division
				// avoids double-committing the partially-filled last block;
				// same ceiling division as the full-range reload path above.
				startBlock := (startIndex + t.gpu.BlockSize() - 1) / t.gpu.BlockSize()
				if startBlock < newStartBlock {
					t.gpu.commitCachedBlocks(req.ID, newCached[startBlock:newStartBlock])
				}
			} else {
				// New request: commit all reloaded blocks from block 0.
				t.gpu.commitCachedBlocks(req.ID, newCached[:newStartBlock])
			}
			return t.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
		}
		// Reload produced no prefix hit beyond startIndex — allocate with the
		// caller's original params (any space freed by the reload still helps).
		return t.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	}
	// No CPU-resident continuation: allocate on GPU exactly as the single-tier path.
	// cpuMissCount++ preserves the pre-#1586 semantic — incremented only when the CPU
	// tier could not help AND the GPU allocation fails.
	ok := t.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	if !ok {
		t.cpuMissCount++
	}
	return ok
}

// reloadPrefixFromCPU attempts to reload prefix-matching blocks from CPU to GPU.
// Computes hierarchical block hashes for the given token prefix and checks CPU for each.
// Reloaded blocks are placed back on the GPU free list with valid hashes (not allocated).
// Returns true if any blocks were reloaded.
//
// The scan begins at startBlock, whose predecessor's block hash is prevHash (the empty
// string when startBlock == 0). Callers pass the length of the already-GPU-resident
// prefix and that prefix's last block hash so the scan resumes at the first uncached
// block instead of re-hashing blocks already on GPU (#1586 hot-path mitigation). The
// hierarchical hash chain is identical to a scan from block 0, since GPU-resident blocks
// would only be skipped anyway.
//
// The maxReloads guard ensures we never pop the same GPU free block twice —
// each reload uses a distinct free block. Without this, pop+append creates
// a cycle where block A's hash is destroyed on the second pop.
func (t *TieredKVCache) reloadPrefixFromCPU(tokens []sim.TokenID, startBlock int64, prevHash string) bool {
	n := util.Len64(tokens) / t.gpu.BlockSize()
	maxReloads := t.gpu.countFreeBlocks() // limit to distinct free blocks
	reloaded := false
	reloadCount := int64(0)
	for i := startBlock; i < n; i++ {
		start := i * t.gpu.BlockSize()
		end := start + t.gpu.BlockSize()
		h := hash.HashBlock(prevHash, tokens[start:end])

		// Already on GPU — skip
		if _, inGPU := t.gpu.HashToBlock[h]; inGPU {
			prevHash = h
			continue
		}

		// Check CPU
		cpuBlk := t.cpu.lookup(h)
		if cpuBlk == nil {
			break // First miss — hierarchical hashing means later blocks are useless
		}

		// Guard: don't re-pop a previously-reloaded block
		if reloadCount >= maxReloads {
			break
		}

		// Reload: pop GPU free block, fill with CPU content, re-append to free list
		gpuBlk := t.gpu.popFreeBlock()
		if gpuBlk == nil {
			break
		}

		// Lazy hash deletion (vLLM parity): clear old hash before filling
		// with CPU content. Maintains consistency with main allocation path
		// (cache.go lazy deletion). Without this, HashToBlock retains a stale
		// entry mapping the old hash to this block's ID even though the block
		// is about to be overwritten with different content.
		if gpuBlk.Hash != "" {
			delete(t.gpu.HashToBlock, gpuBlk.Hash)
			gpuBlk.Hash = ""
		}

		gpuBlk.Tokens = append(gpuBlk.Tokens[:0], cpuBlk.tokens...)
		gpuBlk.Hash = h
		gpuBlk.RefCount = 0
		gpuBlk.InUse = false
		t.gpu.HashToBlock[h] = gpuBlk.ID
		t.gpu.appendToFreeList(gpuBlk)

		// Accumulate transfer latency
		blockSize := float64(t.gpu.BlockSize())
		transferTicks := int64(math.Ceil(blockSize / t.transferBandwidth))
		t.pendingLatency += t.baseLatency + transferTicks

		// Touch CPU block to refresh LRU recency (block is actively needed)
		t.cpu.touch(h)

		t.cpuHitCount++
		reloaded = true
		reloadCount++
		prevHash = h
	}
	return reloaded
}

func (t *TieredKVCache) GetCachedBlocks(tokens []sim.TokenID) []int64 {
	return t.gpu.GetCachedBlocks(tokens)
}

// SnapshotCachedBlocksFn returns a snapshot query function for the GPU tier.
// See KVCacheState.SnapshotCachedBlocksFn for details.
func (t *TieredKVCache) SnapshotCachedBlocksFn() func([]sim.TokenID) int {
	return t.gpu.SnapshotCachedBlocksFn()
}

func (t *TieredKVCache) ReleaseKVBlocks(req *sim.Request) {
	t.gpu.ReleaseKVBlocks(req)
	// No offload — freed blocks stay on GPU free list with hashes intact (BC-3).
	// Hashes are cleared only when popFreeBlock() reuses the slot.
}

func (t *TieredKVCache) BlockSize() int64    { return t.gpu.BlockSize() }
func (t *TieredKVCache) UsedBlocks() int64   { return t.gpu.UsedBlocks() }
func (t *TieredKVCache) TotalCapacity() int64 { return t.gpu.TotalCapacity() }

func (t *TieredKVCache) CacheHitRate() float64 {
	// gpu.CacheHits already includes CPU-reloaded blocks (they appear as GPU
	// cache hits on the retry allocation after reload). cpuHitCount is a
	// diagnostic counter for reload events, not additive to the hit rate.
	totalHits := t.gpu.CacheHits
	totalMisses := t.gpu.CacheMisses + t.cpuMissCount
	total := totalHits + totalMisses
	if total == 0 {
		return 0
	}
	return float64(totalHits) / float64(total)
}

// PendingTransferLatency returns the accumulated transfer latency without clearing it.
// This is a pure query — no side effects. Use ConsumePendingTransferLatency to read and clear.
func (t *TieredKVCache) PendingTransferLatency() int64 {
	return t.pendingLatency
}

// ConsumePendingTransferLatency returns the accumulated transfer latency and resets it to zero.
// Called by Simulator.Step() to apply latency to the current step.
func (t *TieredKVCache) ConsumePendingTransferLatency() int64 {
	lat := t.pendingLatency
	t.pendingLatency = 0
	return lat
}

// KVThrashingRate returns the CPU eviction rate: cpuEvictionCount / mirrorCount.
// Semantic change from pre-v1: was thrashingCount/offloadCount (rapid offload→reload).
// Now measures CPU tier eviction pressure. Returns 0 when mirrorCount == 0 (R11).
func (t *TieredKVCache) KVThrashingRate() float64 {
	if t.mirrorCount == 0 {
		return 0
	}
	return float64(t.cpu.evictionCount) / float64(t.mirrorCount)
}

// SetClock is a no-op in vLLM v1 model (thrashing detection removed).
func (t *TieredKVCache) SetClock(_ int64) {}

// MirrorToCPU copies newly-completed full blocks from batch requests to CPU tier.
// For each request in the batch, all full blocks with hashes are processed:
// - New blocks (not yet on CPU): stored at LRU tail
// - Existing blocks (already on CPU): touched (moved to LRU tail)
// GPU HashToBlock is never modified (read-only copy).
// Called by Simulator.Step() after executeBatchStep(), before processCompletions().
func (t *TieredKVCache) MirrorToCPU(batch []*sim.Request) {
	for _, req := range batch {
		blockIDs, exists := t.gpu.RequestMap[req.ID]
		if !exists {
			continue
		}
		for _, blockID := range blockIDs {
			blk := t.gpu.Blocks[blockID]
			// Only mirror full blocks with computed hashes
			if blk.Hash == "" || util.Len64(blk.Tokens) < t.gpu.BlockSize() {
				continue
			}
			if t.cpu.lookup(blk.Hash) != nil {
				// Already on CPU — touch to refresh LRU recency
				t.cpu.touch(blk.Hash)
			} else {
				// New block — store on CPU
				t.cpu.store(blk.Hash, blk.Tokens)
				t.mirrorCount++
			}
		}
	}
}
