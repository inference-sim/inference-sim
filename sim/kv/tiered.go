package kv

import (
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// offloadedBlock represents a KV block offloaded from GPU to CPU tier.
type offloadedBlock struct {
	OriginalID  int64  // block ID on GPU tier
	Tokens      []int  // token content (for reload)
	Hash        string // prefix hash (for cache hit detection)
	OffloadTime int64  // clock time when offloaded (for thrashing detection)
}

// cpuTier is a simple capacity-tracked block store for CPU-offloaded blocks.
type cpuTier struct {
	blocks   map[int64]*offloadedBlock // keyed by original GPU block ID
	capacity int64
	used     int64
}

// TieredKVCache composes a GPU KVCacheState with a simple CPU tier.
// Delegates all normal operations to GPU; offloads cached free blocks when GPU utilization exceeds threshold.
type TieredKVCache struct {
	gpu              *KVCacheState
	cpu              cpuTier
	offloadThreshold float64
	transferBandwidth float64
	baseLatency      int64
	clock            int64 // updated externally for thrashing detection

	// Transfer latency accumulator (query-and-clear)
	pendingLatency int64

	// Metrics counters
	offloadCount int64
	cpuHitCount  int64
	cpuMissCount   int64
	thrashingCount int64
}

// NewTieredKVCache creates a TieredKVCache.
// Panics if bandwidth is zero (would cause division by zero).
func NewTieredKVCache(gpu *KVCacheState, cpuBlocks int64, threshold, bandwidth float64, baseLat int64) *TieredKVCache {
	if bandwidth <= 0 {
		panic(fmt.Sprintf("KVTransferBandwidth must be > 0, got %f", bandwidth))
	}
	return &TieredKVCache{
		gpu: gpu,
		cpu: cpuTier{
			blocks:   make(map[int64]*offloadedBlock),
			capacity: cpuBlocks,
		},
		offloadThreshold:  threshold,
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
			// More cache hits after reload — retry with reduced allocation range
			return t.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
		}
		// No new cache hits — retry with original params (reload freed up space)
		return t.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	}
	t.cpuMissCount++
	return false
}

// tryReloadFromCPU attempts to reload blocks from CPU to GPU hash table.
// Iterates in deterministic order (sorted by block ID) for reproducibility.
// Returns true if any blocks were reloaded.
//
// Limitation: each reload reuses a GPU free block (pop + re-append). If only 1 GPU free
// block is available and multiple CPU blocks are reloaded, each reload overwrites the
// previous block's content in the same GPU slot. The last-reloaded block survives in the
// hash table; earlier ones are lost. This is acceptable because GetCachedBlocks walks
// prefixes sequentially — only the longest contiguous cached prefix matters.
func (t *TieredKVCache) tryReloadFromCPU() bool {
	// Sort CPU block IDs for deterministic iteration order
	cpuBlockIDs := make([]int64, 0, len(t.cpu.blocks))
	for id := range t.cpu.blocks {
		cpuBlockIDs = append(cpuBlockIDs, id)
	}
	sort.Slice(cpuBlockIDs, func(i, j int) bool { return cpuBlockIDs[i] < cpuBlockIDs[j] })

	reloaded := false
	for _, cpuBlockID := range cpuBlockIDs {
		offloaded := t.cpu.blocks[cpuBlockID]
		if offloaded.Hash == "" {
			continue
		}
		// Check if this hash is already on GPU (no need to reload)
		if _, inGPU := t.gpu.HashToBlock[offloaded.Hash]; inGPU {
			continue
		}
		// Reload: pop a GPU free block, fill with CPU content
		blk := t.gpu.popFreeBlock()
		if blk == nil {
			break // no GPU free blocks available
		}
		blk.Tokens = append([]int{}, offloaded.Tokens...)
		blk.Hash = offloaded.Hash
		blk.RefCount = 0
		blk.InUse = false
		t.gpu.HashToBlock[offloaded.Hash] = blk.ID
		t.gpu.appendToFreeList(blk)

		// Accumulate transfer latency: baseLatency + ceil(blockSize / bandwidth)
		// Uses float64 to avoid division by zero when bandwidth < 1.0
		blockSize := float64(t.gpu.BlockSize())
		transferTicks := int64(math.Ceil(blockSize / t.transferBandwidth))
		t.pendingLatency += t.baseLatency + transferTicks

		// Check thrashing (BC-6): offload followed by reload within 1000 ticks
		if t.clock-offloaded.OffloadTime < 1000 {
			t.thrashingCount++
		}

		// Remove from CPU
		delete(t.cpu.blocks, cpuBlockID)
		t.cpu.used--
		t.cpuHitCount++
		reloaded = true
	}
	return reloaded
}

func (t *TieredKVCache) GetCachedBlocks(tokens []int) []int64 {
	return t.gpu.GetCachedBlocks(tokens)
}

func (t *TieredKVCache) ReleaseKVBlocks(req *sim.Request) {
	t.gpu.ReleaseKVBlocks(req)
	t.maybeOffload()
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

// KVThrashingRate returns the fraction of offloads followed by a reload within 1000 ticks.
func (t *TieredKVCache) KVThrashingRate() float64 {
	if t.offloadCount == 0 {
		return 0
	}
	return float64(t.thrashingCount) / float64(t.offloadCount)
}

// SetClock updates the clock for thrashing detection timestamps.
func (t *TieredKVCache) SetClock(clock int64) {
	t.clock = clock
}

// maybeOffload preserves cached content by moving free blocks with hashes from GPU to CPU.
// Activated when GPU utilization exceeds threshold after a release. Since offloaded blocks
// are already free, this doesn't reduce UsedBlocks — the threshold controls activation,
// and the loop runs until no more cached free blocks remain or CPU is full.
func (t *TieredKVCache) maybeOffload() {
	for t.gpu.TotalCapacity() > 0 {
		util := float64(t.gpu.UsedBlocks()) / float64(t.gpu.TotalCapacity())
		if util <= t.offloadThreshold {
			break
		}
		if t.cpu.used >= t.cpu.capacity {
			break // CPU full (BC-10)
		}
		// Find a free block with cached content to offload
		blk := t.findCachedFreeBlock()
		if blk == nil {
			break // No cached free blocks to offload
		}
		// Copy to CPU
		t.cpu.blocks[blk.ID] = &offloadedBlock{
			OriginalID:  blk.ID,
			Tokens:      append([]int{}, blk.Tokens...),
			Hash:        blk.Hash,
			OffloadTime: t.clock,
		}
		t.cpu.used++
		t.offloadCount++
		// Remove from GPU free list and hash table
		t.gpu.removeFromFreeList(blk)
		delete(t.gpu.HashToBlock, blk.Hash)
		blk.Hash = ""
		blk.Tokens = nil
		// Re-add to free list as empty block (at tail — will be popped last)
		t.gpu.appendToFreeList(blk)
	}
}

// findCachedFreeBlock walks the GPU free list to find a block with cached content (Hash != "").
func (t *TieredKVCache) findCachedFreeBlock() *KVBlock {
	blk := t.gpu.FreeHead
	for blk != nil {
		if blk.Hash != "" {
			return blk
		}
		blk = blk.NextFree
	}
	return nil
}
