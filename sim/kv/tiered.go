package kv

import (
	"fmt"
	"math"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// cpuBlock represents a KV block mirrored from GPU to CPU tier.
// Identified by content hash (not GPU block ID) for content-addressable reload.
type cpuBlock struct {
	hash   string    // prefix hash (map key, identifies content)
	tokens []int     // token content (for GPU reload); pre-allocated slice, copy-into
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
	freeTokenSlices [][]int

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
	slices := make([][]int, capacity)
	for i := int64(0); i < capacity; i++ {
		slices[i] = make([]int, blockSize)
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
func (c *cpuTier) store(hash string, tokens []int) {
	if _, exists := c.blocks[hash]; exists {
		return // already present — caller should use touch
	}
	// Evict if at capacity
	if c.used >= c.capacity {
		c.evictHead()
	}
	// Get a pre-allocated token slice from pool, or allocate as fallback
	var tokSlice []int
	if len(c.freeTokenSlices) > 0 {
		tokSlice = c.freeTokenSlices[len(c.freeTokenSlices)-1]
		c.freeTokenSlices = c.freeTokenSlices[:len(c.freeTokenSlices)-1]
		copy(tokSlice, tokens)
	} else {
		tokSlice = append([]int{}, tokens...) // fallback: allocate
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
	ok := t.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	if ok {
		return true
	}
	// GPU allocation failed — try to reload blocks from CPU to GPU hash table.
	// After reload, re-check cached blocks: reloaded hashes may now match the request's prefix,
	// reducing the number of new blocks needed.
	reloaded := t.tryReloadFromCPU()
	if reloaded {
		// Re-compute cached blocks now that CPU content is back on GPU
		newCached := t.gpu.GetCachedBlocks(req.InputTokens)
		newStart := int64(len(newCached)) * t.gpu.BlockSize()
		if newStart > startIndex {
			if newStart >= endIndex {
				// Entire requested range is cached after reload.
				// For new requests, commit cached blocks (capped at endIndex)
				// to RequestMap so ReleaseKVBlocks can track them.
				// Running requests already have blocks in RequestMap.
				if _, exists := t.gpu.RequestMap[req.ID]; !exists {
					blocksNeeded := (endIndex + t.gpu.BlockSize() - 1) / t.gpu.BlockSize()
					if blocksNeeded > int64(len(newCached)) {
						blocksNeeded = int64(len(newCached))
					}
					t.gpu.commitCachedBlocks(req.ID, newCached[:blocksNeeded])
				}
				return true
			}
			// More cache hits after reload — retry with reduced allocation range
			return t.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
		}
		// No new cache hits — retry with original params (reload freed up space)
		return t.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	}
	t.cpuMissCount++
	return false
}

// tryReloadFromCPU is a stub — replaced by reloadPrefixFromCPU in Task 3.
func (t *TieredKVCache) tryReloadFromCPU() bool {
	return false
}

func (t *TieredKVCache) GetCachedBlocks(tokens []int) []int64 {
	return t.gpu.GetCachedBlocks(tokens)
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
	totalHits := t.gpu.CacheHits + t.cpuHitCount
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
// Stub — full implementation in Task 4.
func (t *TieredKVCache) MirrorToCPU(_ []*sim.Request) {}
