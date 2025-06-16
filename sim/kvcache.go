// sim/kvcache.go
package sim

import (
	"crypto/sha256"
	"encoding/hex"
	"log"
	"strconv"
	"strings"
)

// ToDo: Multi-modality is not yet supported
// Please see a description of prefix caching with images here:
// https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html#automatic-prefix-caching

// KVBlock represents a unit of KV cache storage.
// Each block stores a fixed number of tokens and is tracked by a prefix hash.
// A block becomes eligible for caching once it is full.
type KVBlock struct {
	ID       int      // Unique ID of the block
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
	TotalBlocks     int              // Total KV blocks available on GPU
	BlockSizeTokens int              // Tokens per block
	Blocks          []*KVBlock       // All KV blocks
	RequestMap      map[string][]int // RequestID -> block sequence
	HashToBlock     map[string]int   // Hash -> block ID
	FreeHead        *KVBlock         // Head of free list
	FreeTail        *KVBlock         // Tail of free list
	UsedBlockCnt    int              // Total number of used blocks (tracked incrementally)
}

// NewKVCacheState initializes the KVCacheState and places all blocks in the free list in order.
func NewKVCacheState(totalBlocks int, blockSizeTokens int) *KVCacheState {
	kvc := &KVCacheState{
		TotalBlocks:     totalBlocks,
		BlockSizeTokens: blockSizeTokens,
		Blocks:          make([]*KVBlock, totalBlocks),
		RequestMap:      make(map[string][]int),
		HashToBlock:     make(map[string]int),
	}
	for i := 0; i < totalBlocks; i++ {
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
// It returns block IDs and the number of fully matched cached prefixes.
// This is a pure method and does not modify kvcache state.
func (kvc *KVCacheState) GetCachedBlocks(tokens []int) (blockIDs []int) {
	n := len(tokens) / kvc.BlockSizeTokens
	for i := 0; i < n; i++ {
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

// AllocateKVBlocks reserves cache blocks for a request.
// It reuses cached blocks and allocates new ones from the free list as needed.
// Each full block added is hashed and recorded in the prefix table.
func (kvc *KVCacheState) AllocateKVBlocks(req *Request) bool {
	// given a request, find IDs of cached blocks, the remaining tokens, and blocks needed for remaining tokens
	cachedBlocks := kvc.GetCachedBlocks(req.InputTokens)
	remainingTokens := req.InputTokens[len(cachedBlocks)*kvc.BlockSizeTokens:]
	numRemainingBlocks := (len(remainingTokens) + kvc.BlockSizeTokens - 1) / kvc.BlockSizeTokens

	// Cannot allocate enough KV cache blocks
	if numRemainingBlocks > kvc.countFreeBlocks() {
		log.Printf("Not enough KV cache space to allocate %v new blocks", numRemainingBlocks)
		return false
	}

	// allocated is the block IDs allocated for this request
	allocated := make([]int, 0, len(cachedBlocks)+numRemainingBlocks)

	// Reuse cached blocks (increment refcount, remove from free list if needed)
	for _, blockId := range cachedBlocks {
		blk := kvc.Blocks[blockId]
		blk.RefCount++
		if !blk.InUse {
			blk.InUse = true
			kvc.UsedBlockCnt++
			kvc.removeFromFreeList(blk)
		}
		// allocated is the block IDs allocated for this request
		allocated = append(allocated, blockId)
	}

	// Allocate new blocks for the remaining (non-cached) portion of the request
	for i := 0; i < numRemainingBlocks; i++ {
		blk := kvc.popFreeBlock()
		if blk == nil {
			return false
		}
		// start and end are the range of tokens in blk
		start := (len(cachedBlocks) + i) * kvc.BlockSizeTokens
		end := (len(cachedBlocks) + i + 1) * kvc.BlockSizeTokens
		if end > len(req.InputTokens) {
			end = len(req.InputTokens)
		}
		tok := req.InputTokens[start:end]
		blk.Tokens = append([]int{}, tok...) // copy tokens
		blk.RefCount = 1
		blk.InUse = true
		kvc.UsedBlockCnt++

		if len(blk.Tokens) == kvc.BlockSizeTokens {
			fullPrefix := req.InputTokens[:end]
			h := hashTokens(fullPrefix)
			blk.Hash = h
			kvc.HashToBlock[h] = blk.ID
		}
		// allocated is the block IDs allocated for this request
		allocated = append(allocated, blk.ID)
	}
	kvc.RequestMap[req.ID] = allocated
	return true
}

// AppendToken adds a new (decoded) token to the latest request block.
// If the latest block is full, a new one is allocated.
func (kvc *KVCacheState) AppendToken(reqID string, token int) bool {
	ids := kvc.RequestMap[reqID]
	if len(ids) == 0 {
		// ToDo: Log an error here
		// This can only happen if both inputs and outputs are empty which is invalid
		return false
	}
	latestBlk := kvc.Blocks[ids[len(ids)-1]]
	if len(latestBlk.Tokens) < kvc.BlockSizeTokens {
		// latest block is not full yet
		// append the token to the latest block
		latestBlk.Tokens = append(latestBlk.Tokens, token)
		if len(latestBlk.Tokens) == kvc.BlockSizeTokens {
			fullTokens := []int{}
			for _, blockId := range ids {
				fullTokens = append(fullTokens, kvc.Blocks[blockId].Tokens...)
			}
			h := hashTokens(fullTokens)
			latestBlk.Hash = h
			kvc.HashToBlock[h] = latestBlk.ID
		}
		return true
	}

	// latest block is full
	// allocate new block
	newBlk := kvc.popFreeBlock()
	if newBlk == nil {
		return false
	}
	newBlk.Tokens = []int{token}
	newBlk.RefCount = 1
	newBlk.InUse = true
	kvc.UsedBlockCnt++
	kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], newBlk.ID)
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
func (kvc *KVCacheState) countFreeBlocks() int {
	return kvc.TotalBlocks - kvc.UsedBlockCnt
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
