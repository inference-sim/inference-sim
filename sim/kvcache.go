// sim/kvcache.go
package sim

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
)

// ToDo: Multi-modality is not yet supported
// Please see a description of prefix caching with images here:
// https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html#automatic-prefix-caching

// KVBlock represents a unit of KV cache storage.
// Each block stores a fixed number of tokens and is tracked by a prefix hash.
// A block becomes eligible for caching once it is full.
type KVBlock struct {
	ID       int64    // Unique ID of the block
	RefCount int      // Number of active requests referencing this block
	InUse    bool     // Whether the block is currently in use by an active (batched) request
	Hash     string   // Prefix hash identifying this block's content and its lineage (if full)
	Tokens   []int    // Actual tokens stored in this block; full if len(Tokens) == BlockSizeTokens
	PrevFree *KVBlock // LRU doubly linked list: previous free block
	NextFree *KVBlock // LRU doubly linked list: next free block
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

func Len64[T any](v []T) int64 {
	return int64(len(v))
}

// NewKVCacheState initializes the KVCacheState and places all blocks in the free list in order.
func NewKVCacheState(totalBlocks int64, blockSizeTokens int64) *KVCacheState {
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

// hashTokens returns a SHA256 hash of the joined token sequence.
func hashTokens(tokens []int) string {
	h := sha256.New()

	// string version of token ids to be joined
	var tokenStrings strings.Builder

	for i, token := range tokens {
		if i > 0 {
			// Add a | delimiter before all tokens except the first
			tokenStrings.WriteString("|")
		}
		tokenStrings.WriteString(strconv.Itoa(token))
	}

	h.Write([]byte(tokenStrings.String()))
	return hex.EncodeToString(h.Sum(nil))
}

// GetCachedBlocks attempts to reuse previously cached full blocks.
// It returns block IDs for the longest contiguous cached prefix.
// This is a pure query — it does not modify any state.
// CacheHits are counted by AllocateKVBlocks when cached blocks are committed.
func (kvc *KVCacheState) GetCachedBlocks(tokens []int) (blockIDs []int64) {
	n := Len64(tokens) / kvc.BlockSizeTokens
	for i := int64(0); i < n; i++ {
		chunk := tokens[:(i+1)*kvc.BlockSizeTokens]
		h := hashTokens(chunk)
		blockId, ok := kvc.HashToBlock[h]
		if !ok {
			break
		}
		blockIDs = append(blockIDs, blockId)
	}
	return
}

// AllocateKVBlocks handles KV Block allocation for both prefill and decode.
// If the latest block is full, a new one is allocated. Otherwise push to latest allocated block.
// start and endIndex are by original requests' index
// endIndex is non-inclusive
func (kvc *KVCacheState) AllocateKVBlocks(req *Request, startIndex int64, endIndex int64, cachedBlocks []int64) bool {
	reqID := req.ID
	logrus.Debugf("AllocateBlock for ReqID: %s, Num Inputs: %d, startIndex = %d, endIndex = %d", req.ID, len(req.InputTokens), startIndex, endIndex)

	var newTokens []int
	var numNewBlocks = int64(1)
	if req.ProgressIndex < Len64(req.InputTokens) {
		// request is in prefill (could be chunked)
		newTokens = req.InputTokens[startIndex:endIndex]
		// check if enough memory for tokens to be cached
		numNewBlocks = (Len64(newTokens) + kvc.BlockSizeTokens - 1) / kvc.BlockSizeTokens

		// Cannot allocate enough KV cache blocks
		if numNewBlocks > kvc.countFreeBlocks() {
			logrus.Warnf("KV cache full: cannot allocate %d new blocks for req %s", numNewBlocks, req.ID)
			return false
		}
	} else {
		// request is in decode
		newTokens = append(newTokens, req.OutputTokens[startIndex-Len64(req.InputTokens)])
	}
	// Rollback tracking: if allocation fails mid-way, undo all mutations
	var cachedMutations []cachedBlockMutation
	var newlyAllocated []newBlockMutation

	newTokenProgressIndex := int64(0)
	for newTokenProgressIndex < Len64(newTokens) { // non-inclusive endIndex
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
				kvc.CacheHits++
				logrus.Debugf("Hit KV Cache for req: %s of length: %d", req.ID, Len64(cachedBlocks)*kvc.BlockSizeTokens)
				kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blockId)
			}
		}
		if len(latestBlk.Tokens) > 0 && Len64(latestBlk.Tokens) < kvc.BlockSizeTokens {
			// latest block is not full yet, append tokens to the latest block
			remaining := kvc.BlockSizeTokens - Len64(latestBlk.Tokens)
			toksToAppend := newTokens[newTokenProgressIndex:min(newTokenProgressIndex+remaining, Len64(newTokens))]
			latestBlk.Tokens = append(latestBlk.Tokens, toksToAppend...)
			newTokenProgressIndex += Len64(toksToAppend)
			logrus.Debugf("Appending to latest blk: req: %s, newTokenProgressIndex = %d, appended=%d tokens", req.ID, newTokenProgressIndex, Len64(toksToAppend))
			if Len64(latestBlk.Tokens) == kvc.BlockSizeTokens {
				// latesBlk is full
				fullTokens := []int{}
				for _, blockId := range ids {
					fullTokens = append(fullTokens, kvc.Blocks[blockId].Tokens...)
				}
				h := hashTokens(fullTokens)
				latestBlk.Hash = h
				kvc.HashToBlock[h] = latestBlk.ID
			}
		} else {
			// latest block is full or request is coming in for the first time.
			// allocate new block(s) for the request
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
				if end > Len64(newTokens) {
					end = Len64(newTokens)
				}
				tok := newTokens[start:end]
				blk.Tokens = append([]int{}, tok...) // copy tokens
				blk.RefCount = 1
				blk.InUse = true
				kvc.UsedBlockCnt++
				kvc.CacheMisses++

				if Len64(blk.Tokens) == kvc.BlockSizeTokens && req.ProgressIndex < Len64(req.InputTokens) {
					// Only compute prefix hash during prefill (not decode).
					// During decode, startIndex >= len(InputTokens) and absoluteEnd
					// would be out of range for req.InputTokens.
					absoluteEnd := startIndex + end
					fullPrefix := req.InputTokens[:absoluteEnd]
					h := hashTokens(fullPrefix)
					blk.Hash = h
					kvc.HashToBlock[h] = blk.ID
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
	// Undo new block allocations (reverse order for clean free list state)
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
		kvc.appendToFreeList(blk)
	}
	// Undo cached block mutations
	for _, cm := range cachedMutations {
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

// ReleaseKVBlocks deallocates blocks used by a completed request.
// Each block's refcount is decremented and may be returned to the free list.
func (kvc *KVCacheState) ReleaseKVBlocks(req *Request) {
	ids := kvc.RequestMap[req.ID]
	delete(kvc.RequestMap, req.ID)
	// From https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html
	/*
		When a request is finished, we free all its blocks if no other
		requests are using them (reference count = 0).
		In this example, we free request 1 and block 2, 3, 4, 8
		associated with it. We can see that the freed blocks are added
		to the tail of the free queue in the reverse order.
		This is because the last block of a request must hash more tokens
		and is less likely to be reused by other requests.
		As a result, it should be evicted first.
	*/
	for i := len(ids) - 1; i >= 0; i-- {
		blockId := ids[i]
		blk := kvc.Blocks[blockId]
		blk.RefCount--
		if blk.RefCount == 0 {
			blk.InUse = false
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
