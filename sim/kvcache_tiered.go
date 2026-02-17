package sim

import "fmt"

// OffloadedBlock represents a KV block offloaded from GPU to CPU tier.
type OffloadedBlock struct {
	OriginalID  int64  // block ID on GPU tier
	Tokens      []int  // token content (for reload)
	Hash        string // prefix hash (for cache hit detection)
	OffloadTime int64  // clock time when offloaded (for thrashing detection)
}

// cpuTier is a simple capacity-tracked block store for CPU-offloaded blocks.
type cpuTier struct {
	blocks   map[int64]*OffloadedBlock // keyed by original GPU block ID
	capacity int64
	used     int64
}

// TieredKVCache composes a GPU KVCacheState with a simple CPU tier.
// Delegates all normal operations to GPU; offloads LRU blocks when pressure exceeds threshold.
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
			blocks:   make(map[int64]*OffloadedBlock),
			capacity: cpuBlocks,
		},
		offloadThreshold:  threshold,
		transferBandwidth: bandwidth,
		baseLatency:       baseLat,
	}
}

func (t *TieredKVCache) AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	ok := t.gpu.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
	if ok {
		return true
	}
	// GPU allocation failed — reload logic added in Task 5
	return false
}

func (t *TieredKVCache) GetCachedBlocks(tokens []int) []int64 {
	return t.gpu.GetCachedBlocks(tokens)
}

func (t *TieredKVCache) ReleaseKVBlocks(req *Request) {
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

func (t *TieredKVCache) PendingTransferLatency() int64 {
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

// maybeOffload moves LRU free blocks from GPU to CPU until GPU utilization ≤ threshold.
func (t *TieredKVCache) maybeOffload() {
	for t.gpu.TotalCapacity() > 0 {
		util := float64(t.gpu.UsedBlocks()) / float64(t.gpu.TotalCapacity())
		if util <= t.offloadThreshold {
			break
		}
		if t.cpu.used >= t.cpu.capacity {
			break // CPU full (BC-10)
		}
		// Find a free block on GPU to offload (from free list head)
		blk := t.gpu.FreeHead
		if blk == nil {
			break // No free blocks to offload
		}
		// Only offload blocks that have cached content (hash != "")
		if blk.Hash == "" {
			break // Empty block, nothing useful to offload
		}
		// Copy to CPU
		t.cpu.blocks[blk.ID] = &OffloadedBlock{
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
