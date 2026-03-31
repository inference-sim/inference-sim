// Package kv implements block-based KV cache management for the BLIS simulator.
// It provides single-tier GPU (KVCacheState) and two-tier GPU+CPU (TieredKVCache)
// implementations of the sim.KVStore interface. Both support prefix caching with
// SHA256-based block hashing, LRU eviction, and transactional allocation with rollback.
package kv

import (
	"fmt"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// ToDo: Multi-modality is not yet supported
// Please see a description of prefix caching with images here:
// https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html#automatic-prefix-caching

// KVBlock represents a unit of KV cache storage.
// Each block stores a fixed number of tokens and is tracked by a prefix hash.
// A block becomes eligible for caching once it is full.
type KVBlock struct {
	ID           int64    // Unique ID of the block
	RefCount     int      // Number of active requests referencing this block
	InUse        bool     // Whether the block is currently in use by an active (batched) request
	Hash         string   // Prefix hash identifying this block's content and its lineage (if full)
	Tokens       []int    // Actual tokens stored in this block; full if len(Tokens) == BlockSizeTokens
	TierPriority int      // SLO tier priority of owning request (higher = more protected from eviction)
	PrevFree     *KVBlock // LRU doubly linked list: previous free block
	NextFree     *KVBlock // LRU doubly linked list: next free block
}

// KVCacheState maintains global KV cache status across all requests.
// It implements prefix caching, reverse-order LRU eviction, and tracks the number of used blocks.
type KVCacheState struct {
	TotalBlocks     int64              // Total KV blocks available on GPU
	BlockSizeTokens int64              // Tokens per block
	Blocks          []*KVBlock         // All KV blocks
	RequestMap      map[string][]int64 // RequestID -> block sequence
	HashToBlock     map[string]int64   // Hash -> block ID
	FreeHead        *KVBlock           // Head of free list
	FreeTail        *KVBlock           // Tail of free list
	UsedBlockCnt    int64              // Total number of used blocks (tracked incrementally)
	CacheHits       int64              // blocks found via prefix cache (PR12)
	CacheMisses     int64              // blocks not found, allocated fresh (PR12)
}

// NewKVCacheState initializes the KVCacheState and places all blocks in the free list in order.
func NewKVCacheState(totalBlocks int64, blockSizeTokens int64) *KVCacheState {
	if totalBlocks <= 0 {
		panic(fmt.Sprintf("NewKVCacheState: TotalKVBlocks must be > 0, got %d", totalBlocks))
	}
	if blockSizeTokens <= 0 {
		panic(fmt.Sprintf("NewKVCacheState: BlockSizeTokens must be > 0, got %d", blockSizeTokens))
	}
	kvc := &KVCacheState{
		TotalBlocks:     totalBlocks,
		BlockSizeTokens: blockSizeTokens,
		Blocks:          make([]*KVBlock, totalBlocks),
		RequestMap:      make(map[string][]int64),
		HashToBlock:     make(map[string]int64),
	}
	for i := int64(0); i < totalBlocks; i++ {
		blk := &KVBlock{ID: i}
		kvc.Blocks[i] = blk
		kvc.appendToFreeList(blk)
	}
	return kvc
}

// appendToFreeList inserts a block at the tail of the free list.
func (kvc *KVCacheState) appendToFreeList(block *KVBlock) {
	block.NextFree = nil
	// in a doubly linked list, either both head and tail will be nil, or neither or nil
	if kvc.FreeTail != nil {
		// non-empty list; append block at end
		kvc.FreeTail.NextFree = block
		block.PrevFree = kvc.FreeTail
		kvc.FreeTail = block
	} else {
		// empty list; create list with a single block
		kvc.FreeHead = block
		kvc.FreeTail = block
		block.PrevFree = nil
	}
}

// prependToFreeList adds a block to the HEAD of the free list.
// Used by rollbackAllocation to restore blocks to their original position
// (blocks were popped from the head, so rollback prepends them back).
func (kvc *KVCacheState) prependToFreeList(block *KVBlock) {
	block.PrevFree = nil
	block.NextFree = kvc.FreeHead
	if kvc.FreeHead != nil {
		kvc.FreeHead.PrevFree = block
	}
	kvc.FreeHead = block
	if kvc.FreeTail == nil {
		kvc.FreeTail = block
	}
}

// removeFromFreeList detaches a block from the LRU free list.
func (kvc *KVCacheState) removeFromFreeList(block *KVBlock) {
	if block.PrevFree != nil {
		// a - b - block - c => a - b - c
		block.PrevFree.NextFree = block.NextFree
	} else {
		// block - c - d => c - d
		kvc.FreeHead = block.NextFree
	}
	if block.NextFree != nil {
		// a - b - block - c => a - b - c
		block.NextFree.PrevFree = block.PrevFree
	} else {
		// a - b - block => a - b
		kvc.FreeTail = block.PrevFree
	}
	block.NextFree = nil
	block.PrevFree = nil
}

// GetCachedBlocks attempts to reuse previously cached full blocks.
// It returns block IDs for the longest contiguous cached prefix.
// This is a pure query — it does not modify any state.
// CacheHits are counted by AllocateKVBlocks when cached blocks are committed.
//
// Uses hierarchical block hashing: each block's hash chains the previous
// block's hash, so each iteration hashes only blockSize tokens plus a
// fixed-length prev hash (O(K * blockSize) total, down from O(K^2 * blockSize)
// with flat-prefix hashing). Breaks on first miss.
func (kvc *KVCacheState) GetCachedBlocks(tokens []int) (blockIDs []int64) {
	n := util.Len64(tokens) / kvc.BlockSizeTokens
	prevHash := ""
	for i := int64(0); i < n; i++ {
		start := i * kvc.BlockSizeTokens
		end := start + kvc.BlockSizeTokens
		h := hash.HashBlock(prevHash, tokens[start:end])
		blockId, ok := kvc.HashToBlock[h]
		if !ok {
			break
		}
		blockIDs = append(blockIDs, blockId)
		prevHash = h
	}
	return
}

// AllocateKVBlocks handles KV Block allocation for both prefill and decode.
// If the latest block is full, a new one is allocated. Otherwise push to latest allocated block.
// start and endIndex are by original requests' index
// endIndex is non-inclusive
func (kvc *KVCacheState) AllocateKVBlocks(req *sim.Request, startIndex int64, endIndex int64, cachedBlocks []int64) bool {
	reqID := req.ID
	logrus.Debugf("AllocateBlock for ReqID: %s, Num Inputs: %d, startIndex = %d, endIndex = %d", req.ID, len(req.InputTokens), startIndex, endIndex)

	var newTokens []int
	var numNewBlocks int64
	if req.ProgressIndex < util.Len64(req.InputTokens) {
		// request is in prefill (could be chunked)
		newTokens = req.InputTokens[startIndex:endIndex]

		// Compute blocks needed, accounting for tokens that will be absorbed
		// into the request's existing partially-filled last block. Without this,
		// the pre-check over-estimates by up to 1 block, causing false rejections
		// when free blocks are tight (#492).
		effectiveTokens := util.Len64(newTokens)
		if ids, hasBlocks := kvc.RequestMap[reqID]; hasBlocks && len(ids) > 0 {
			lastBlk := kvc.Blocks[ids[len(ids)-1]]
			// spare < BlockSizeTokens excludes empty blocks (0 tokens stored)
			if spare := kvc.BlockSizeTokens - util.Len64(lastBlk.Tokens); spare > 0 && spare < kvc.BlockSizeTokens {
				effectiveTokens -= min(spare, effectiveTokens)
			}
		}
		if effectiveTokens > 0 {
			numNewBlocks = (effectiveTokens + kvc.BlockSizeTokens - 1) / kvc.BlockSizeTokens
		}

		// Cannot allocate enough KV cache blocks
		if numNewBlocks > kvc.countFreeBlocks() {
			logrus.Warnf("KV cache full: cannot allocate %d new blocks for req %s", numNewBlocks, req.ID)
			return false
		}
	} else {
		// request is in decode
		newTokens = append(newTokens, req.OutputTokens[startIndex-util.Len64(req.InputTokens)])
	}
	// Rollback tracking: if allocation fails mid-way, undo all mutations
	var cachedMutations []cachedBlockMutation
	var newlyAllocated []newBlockMutation

	newTokenProgressIndex := int64(0)
	for newTokenProgressIndex < util.Len64(newTokens) { // non-inclusive endIndex
		ids, ok := kvc.RequestMap[reqID]
		latestBlk := &KVBlock{}
		if ok {
			// KV cache has already seen this request before. The latest block needs to be filled first,
			// followed by new blocks. Caching cannot happen here.
			latestBlk = kvc.Blocks[ids[len(ids)-1]]

		} else {
			// KV cache is seeing this request for the first time (beginning of prefill)
			// append the cached blocks to this request's ID map

			for _, blockId := range cachedBlocks {
				blk := kvc.Blocks[blockId]
				wasInUse := blk.InUse
				blk.RefCount++
				if !blk.InUse {
					blk.InUse = true
					kvc.UsedBlockCnt++
					kvc.removeFromFreeList(blk)
				}
				cachedMutations = append(cachedMutations, cachedBlockMutation{block: blk, wasInUse: wasInUse})
				if p := sim.SLOTierPriority(req.SLOClass); p > blk.TierPriority {
					blk.TierPriority = p
				}
				kvc.CacheHits++
				logrus.Debugf("Hit KV Cache for req: %s of length: %d", req.ID, util.Len64(cachedBlocks)*kvc.BlockSizeTokens)
				kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blockId)
			}
		}
		if len(latestBlk.Tokens) > 0 && util.Len64(latestBlk.Tokens) < kvc.BlockSizeTokens {
			// latest block is not full yet, append tokens to the latest block
			remaining := kvc.BlockSizeTokens - util.Len64(latestBlk.Tokens)
			toksToAppend := newTokens[newTokenProgressIndex:min(newTokenProgressIndex+remaining, util.Len64(newTokens))]
			latestBlk.Tokens = append(latestBlk.Tokens, toksToAppend...)
			newTokenProgressIndex += util.Len64(toksToAppend)
			logrus.Debugf("Appending to latest blk: req: %s, newTokenProgressIndex = %d, appended=%d tokens", req.ID, newTokenProgressIndex, util.Len64(toksToAppend))
			if util.Len64(latestBlk.Tokens) == kvc.BlockSizeTokens {
				// latestBlk is full — compute its hierarchical hash.
				// Chain from the previous block's hash (or "" if first block).
				prevHash := ""
				if len(ids) >= 2 {
					prevHash = kvc.Blocks[ids[len(ids)-2]].Hash
				}
				h := hash.HashBlock(prevHash, latestBlk.Tokens)
				latestBlk.Hash = h
				kvc.HashToBlock[h] = latestBlk.ID
			}
		} else {
			// latest block is full or request is coming in for the first time.
			// allocate new block(s) for the request.
			// Recompute blocks needed from remaining tokens (after partial fill consumed some).
			remainingTokens := util.Len64(newTokens) - newTokenProgressIndex
			if remainingTokens <= 0 {
				break
			}
			numNewBlocks = (remainingTokens + kvc.BlockSizeTokens - 1) / kvc.BlockSizeTokens

			// Initialize prevHash for hierarchical block hashing.
			// Chain from the last existing block for this request (if any).
			prevHash := ""
			if existingIDs, has := kvc.RequestMap[reqID]; has && len(existingIDs) > 0 {
				prevHash = kvc.Blocks[existingIDs[len(existingIDs)-1]].Hash
			}

			for i := int64(0); i < numNewBlocks; i++ {
				// Save the original prefix hash before popFreeBlock clears it.
				// This allows rollback to restore cached prefix hashes.
				var originalHash string
				if kvc.FreeHead != nil {
					originalHash = kvc.FreeHead.Hash
				}
				blk := kvc.popFreeBlock()
				if blk == nil {
					kvc.rollbackAllocation(reqID, cachedMutations, newlyAllocated)
					return false
				}
				// start and end are the range of tokens in blk
				start := newTokenProgressIndex
				end := newTokenProgressIndex + kvc.BlockSizeTokens
				logrus.Debugf("Assigning new blocks: req = %s, newTokenProgressIndex = %d, ogStartIdx= %d, ogEndIdx = %d, startBlk=%d, endBlk=%d", req.ID, newTokenProgressIndex, startIndex, endIndex, start, end)
				if end > util.Len64(newTokens) {
					end = util.Len64(newTokens)
				}
				tok := newTokens[start:end]
				blk.Tokens = append([]int{}, tok...) // copy tokens
				blk.RefCount = 1
				blk.InUse = true
				blk.TierPriority = sim.SLOTierPriority(req.SLOClass)
				kvc.UsedBlockCnt++
				kvc.CacheMisses++

				if util.Len64(blk.Tokens) == kvc.BlockSizeTokens && req.ProgressIndex < util.Len64(req.InputTokens) {
					// Only compute prefix hash during prefill (not decode).
					// During decode, blocks hold output tokens that should not
					// participate in prefix caching (input sequences only).
					h := hash.HashBlock(prevHash, blk.Tokens)
					blk.Hash = h
					kvc.HashToBlock[h] = blk.ID
					prevHash = h
				}
				newlyAllocated = append(newlyAllocated, newBlockMutation{block: blk, originalHash: originalHash})
				// allocated is the block IDs allocated for this request
				kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blk.ID)
				newTokenProgressIndex = end
			}
		}
	}

	return true
}

// popFreeBlock evicts a block from the free list and prepares it for reuse.
func (kvc *KVCacheState) popFreeBlock() *KVBlock {
	head := kvc.FreeHead
	if head == nil {
		return nil
	}
	kvc.removeFromFreeList(head)
	if head.Hash != "" {
		delete(kvc.HashToBlock, head.Hash)
		head.Hash = ""
	}
	head.Tokens = nil
	return head
}

// countFreeBlocks returns the number of blocks not currently in use.
func (kvc *KVCacheState) countFreeBlocks() int64 {
	return kvc.TotalBlocks - kvc.UsedBlockCnt
}

// cachedBlockMutation tracks a cached block's state before mutation for rollback.
type cachedBlockMutation struct {
	block    *KVBlock
	wasInUse bool
}

// newBlockMutation tracks a newly allocated block and its pre-pop hash for rollback.
type newBlockMutation struct {
	block        *KVBlock
	originalHash string // hash before popFreeBlock cleared it (empty if block had no hash)
}

// rollbackAllocation undoes all mutations from a failed AllocateKVBlocks call.
// Restores UsedBlockCnt, CacheMisses, CacheHits, RefCount, InUse, free list, HashToBlock, and RequestMap.
// Also restores prefix hashes that were destroyed by popFreeBlock during allocation.
func (kvc *KVCacheState) rollbackAllocation(reqID string, cachedMutations []cachedBlockMutation, newlyAllocated []newBlockMutation) {
	// Undo new block allocations (reverse order so first-popped block
	// ends up at the free list head, restoring original LRU order)
	for i := len(newlyAllocated) - 1; i >= 0; i-- {
		m := newlyAllocated[i]
		blk := m.block
		// Remove any hash set during allocation
		if blk.Hash != "" {
			delete(kvc.HashToBlock, blk.Hash)
		}
		// Restore original hash that popFreeBlock destroyed
		blk.Hash = m.originalHash
		if blk.Hash != "" {
			kvc.HashToBlock[blk.Hash] = blk.ID
		}
		blk.InUse = false
		blk.RefCount = 0
		blk.Tokens = nil
		kvc.UsedBlockCnt--
		kvc.CacheMisses--
		kvc.prependToFreeList(blk)
	}
	// Undo cached block mutations (reverse order so blocks are appended
	// to the tail in the opposite order they were removed, restoring
	// the original free list tail ordering)
	for i := len(cachedMutations) - 1; i >= 0; i-- {
		cm := cachedMutations[i]
		cm.block.RefCount--
		kvc.CacheHits--
		if !cm.wasInUse && cm.block.RefCount == 0 {
			cm.block.InUse = false
			kvc.UsedBlockCnt--
			kvc.appendToFreeList(cm.block)
		}
	}
	// Clean up RequestMap
	delete(kvc.RequestMap, reqID)
}

// commitCachedBlocks registers a slice of cached blocks into a request's RequestMap.
// Increments RefCount, sets InUse, removes from free list, records cache hits,
// and appends block IDs to RequestMap.
//
// For new requests (reqID not yet in RequestMap), pass the full cached block slice.
// For running requests (reqID already in RequestMap), pass only the uncovered range
// (e.g., newCached[startBlock:endBlock]) to avoid double-counting RefCount for
// blocks the request already owns.
//
// NOTE: This method does NOT track mutations for rollback. It is used by
// TieredKVCache in two paths: (1) the full-reload path, which returns true
// immediately after calling this (so no rollback is needed); (2) the
// partial-improvement path, which calls gpu.AllocateKVBlocks afterwards.
// In path (2), if AllocateKVBlocks fails at its pre-check, no rollback
// fires and the committed state is stable. If the pre-check passes (which
// can occur when partial-block spare capacity reduces the needed count),
// the allocation succeeds and the committed blocks are part of the final
// state. In either case the committed state is stable. Mid-loop failure
// is impossible in BLIS's single-threaded DES: once the pre-check passes,
// countFreeBlocks() cannot decrease before the allocation loop runs.
// The inline equivalent in AllocateKVBlocks feeds cachedMutations for
// rollback support — do not replace that inline code with this method.
func (kvc *KVCacheState) commitCachedBlocks(req *sim.Request, cachedBlocks []int64) {
	for _, blockID := range cachedBlocks {
		blk := kvc.Blocks[blockID]
		blk.RefCount++
		if !blk.InUse {
			blk.InUse = true
			kvc.UsedBlockCnt++
			kvc.removeFromFreeList(blk)
		}
		if p := sim.SLOTierPriority(req.SLOClass); p > blk.TierPriority {
			blk.TierPriority = p
		}
		kvc.CacheHits++
		kvc.RequestMap[req.ID] = append(kvc.RequestMap[req.ID], blockID)
	}
}

// ReleaseKVBlocks deallocates blocks used by a completed request.
// Each block's refcount is decremented and may be returned to the free list.
func (kvc *KVCacheState) ReleaseKVBlocks(req *sim.Request) {
	ids := kvc.RequestMap[req.ID]
	delete(kvc.RequestMap, req.ID)
	// From https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html
	// Freed blocks are added to the tail of the free queue in reverse order.
	// Later blocks can only be reused if all preceding blocks also match
	// (hierarchical hashing), so evicting them first preserves the more
	// broadly reusable earlier blocks.
	for i := len(ids) - 1; i >= 0; i-- {
		blockId := ids[i]
		blk := kvc.Blocks[blockId]
		blk.RefCount--
		if blk.RefCount == 0 {
			blk.InUse = false
			blk.TierPriority = 0
			kvc.UsedBlockCnt--
			kvc.appendToFreeList(blk)
		}
	}
}

// BlockSize returns the number of tokens per block.
func (kvc *KVCacheState) BlockSize() int64 { return kvc.BlockSizeTokens }

// UsedBlocks returns the number of blocks currently in use.
func (kvc *KVCacheState) UsedBlocks() int64 { return kvc.UsedBlockCnt }

// TotalCapacity returns the total number of blocks.
func (kvc *KVCacheState) TotalCapacity() int64 { return kvc.TotalBlocks }

// CacheHitRate returns the cumulative cache hit rate.
// Returns 0 if no lookups have been performed.
func (kvc *KVCacheState) CacheHitRate() float64 {
	total := kvc.CacheHits + kvc.CacheMisses
	if total == 0 {
		return 0
	}
	return float64(kvc.CacheHits) / float64(total)
}

// PendingTransferLatency returns 0 for single-tier cache (no transfers).
func (kvc *KVCacheState) PendingTransferLatency() int64 { return 0 }

// KVThrashingRate returns 0 for single-tier cache (no offload/reload).
func (kvc *KVCacheState) KVThrashingRate() float64 { return 0 }

// SetClock is a no-op for single-tier KV cache (no time-dependent behavior).
func (kvc *KVCacheState) SetClock(_ int64) {}

// ConsumePendingTransferLatency returns 0 for single-tier cache (no transfers).
func (kvc *KVCacheState) ConsumePendingTransferLatency() int64 { return 0 }

// MirrorToCPU is a no-op for single-tier KV cache (no CPU tier).
func (kvc *KVCacheState) MirrorToCPU(_ []*sim.Request) {}
