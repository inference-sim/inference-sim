// Package kv implements block-based KV cache management for the BLIS simulator.
// It provides single-tier GPU (KVCacheState) and two-tier GPU+CPU (TieredKVCache)
// implementations of the sim.KVStore interface. Both support prefix caching with
// SHA256-based block hashing, LRU eviction, and check-then-act allocation (vLLM parity).
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
	ID       int64    // Unique ID of the block
	RefCount int      // Number of active requests referencing this block
	InUse    bool     // Whether the block is currently in use by an active (batched) request
	Hash     string   // Prefix hash identifying this block's content and its lineage (if full)
	Tokens   []sim.TokenID // Actual tokens stored in this block; full if len(Tokens) == BlockSizeTokens
	PrevFree *KVBlock // LRU doubly linked list: previous free block
	NextFree *KVBlock // LRU doubly linked list: next free block
}

// KVCacheState maintains global KV cache status across all requests.
// It implements prefix caching, reverse-order LRU eviction, and tracks free blocks
// via a direct counter on the free list (FreeBlockCnt), mirroring vLLM's
// FreeKVCacheBlockQueue.num_free_blocks.
type KVCacheState struct {
	TotalBlocks     int64              // Total KV blocks available on GPU
	BlockSizeTokens int64              // Tokens per block
	Blocks          []*KVBlock         // All KV blocks
	RequestMap      map[string][]int64 // RequestID -> block sequence
	HashToBlock     map[string]int64   // Hash -> block ID
	FreeHead        *KVBlock           // Head of free list
	FreeTail        *KVBlock           // Tail of free list
	FreeBlockCnt    int64              // Direct count of blocks in free list (vLLM parity)
	CacheHits       int64              // blocks found via prefix cache (PR12)
	CacheMisses     int64              // blocks not found, allocated fresh (PR12)
	// remoteKV are the requests whose blocks a remote pull is still filling
	// (AllocateRemoteKV): their full blocks carry hashes but are not registered in
	// HashToBlock until CacheRemoteKV, so no other request hits on KV not yet present.
	remoteKV map[string]bool

	// SequenceState sizes each request's bounded per-sequence state -- window rings,
	// compressor and indexer state, bounded recurrent state -- as the latency kernel prices it
	// (#1941): resized on every allocation to what the scheduled step holds, gated at
	// admission, freed with the request. Zero value: none. StateBlocks holds the blocks per
	// request; they carry no tokens and no hash, so prefix caching never sees them.
	SequenceState sim.KVSequenceState
	StateBlocks   map[string][]int64

	// Speculative sizes each allocation by vLLM's slot count rather than by the tokens
	// stored (#1627): a request holds blocks for endIndex + LookaheadTokens slots, endIndex
	// being its computed tokens plus everything scheduled this step (the whole verify width
	// for a speculating decode). Blocks held past its stored tokens are SpareBlocks, blank,
	// and the first used when its tokens need a new block. Zero value: off.
	Speculative sim.KVSpeculation
	SpareBlocks map[string][]int64
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
		StateBlocks:     make(map[string][]int64),
		SpareBlocks:     make(map[string][]int64),
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
	kvc.FreeBlockCnt++
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
	kvc.FreeBlockCnt--
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
//
// CO-CHANGE: SnapshotCachedBlocksFn (same file) replicates this algorithm on a
// frozen map[string]int64 snapshot. If this loop, the hash chain logic, or the
// break condition changes, update SnapshotCachedBlocksFn to match.
func (kvc *KVCacheState) GetCachedBlocks(tokens []sim.TokenID) (blockIDs []int64) {
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

// SnapshotCachedBlocksFn returns a function that queries a frozen copy of the
// current HashToBlock map. The returned function counts consecutive cached prefix
// blocks for given tokens using the snapshot, NOT the live state.
// Used for stale cache signal simulation (issue #919).
//
// The snapshot captures HashToBlock at call time. Subsequent allocations/releases
// do NOT affect the returned function's results.
//
// CO-CHANGE: GetCachedBlocks (same file) implements the same algorithm on live
// HashToBlock state. If this loop, the hash chain logic, or the break condition
// changes, update GetCachedBlocks to match.
func (kvc *KVCacheState) SnapshotCachedBlocksFn() func([]sim.TokenID) int {
	snapshot := make(map[string]int64, len(kvc.HashToBlock))
	for k, v := range kvc.HashToBlock {
		snapshot[k] = v
	}
	blockSize := kvc.BlockSizeTokens
	return func(tokens []sim.TokenID) int {
		n := int64(len(tokens)) / blockSize
		prevHash := ""
		count := 0
		for i := int64(0); i < n; i++ {
			start := i * blockSize
			end := start + blockSize
			h := hash.HashBlock(prevHash, tokens[start:end])
			if _, ok := snapshot[h]; !ok {
				break
			}
			count++
			prevHash = h
		}
		return count
	}
}

// AllocateKVBlocks handles KV Block allocation for both prefill and decode.
// If the latest block is full, a new one is allocated. Otherwise push to latest allocated block.
// start and endIndex are by original requests' index
// endIndex is non-inclusive
func (kvc *KVCacheState) AllocateKVBlocks(req *sim.Request, startIndex int64, endIndex int64, cachedBlocks []int64) bool {
	reqID := req.ID
	logrus.Debugf("AllocateBlock for ReqID: %s, Num Inputs: %d, startIndex = %d, endIndex = %d", req.ID, req.InputLen(), startIndex, endIndex)

	var newTokens []sim.TokenID
	var numNewBlocks int64
	if kvc.admissionGateRefuses(req, cachedBlocks) {
		return false
	}
	// stateNeed is the change in the request's bounded-state blocks this step (negative when
	// its window rings shrink, as remove_skipped_blocks frees them before allocating).
	stateNeed := kvc.stateDelta(reqID, startIndex, endIndex)
	if req.ProgressIndex < req.InputLen() {
		// request is in prefill (could be chunked)
		newTokens = req.InputTokenSlice(startIndex, endIndex)

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

		// Account for cached blocks that will leave the free list when claimed.
		// Mirrors vLLM's num_evictable_blocks (single_type_kv_cache_manager.py:124-127).
		// Invariant: cachedBlocks (from GetCachedBlocks) only contains blocks not yet
		// owned by the calling request — they are prefix-cache hits with RefCount >= 0.
		// Blocks with !InUse (RefCount == 0) sit on the free list and will be removed
		// when claimed, reducing countFreeBlocks(). We must budget for this.
		var cachedFromFreeList int64
		for _, blockID := range cachedBlocks {
			blk := kvc.Blocks[blockID]
			if !blk.InUse {
				cachedFromFreeList++
			}
		}

		held := int64(len(kvc.RequestMap[reqID]))
		if held == 0 && len(newTokens) > 0 {
			held = util.Len64(cachedBlocks) // claimed below, on a first allocation
		}
		fromFree := cachedFromFreeList + stateNeed + kvc.slotBlocksFromFree(reqID, endIndex, held, numNewBlocks)
		if fromFree > kvc.countFreeBlocks() {
			logrus.Debugf("KV cache full: cannot allocate %d new + %d cached + %d state blocks (%d from free) for req %s",
				numNewBlocks, cachedFromFreeList, stateNeed, fromFree, req.ID)
			return false
		}
	} else {
		// request is in decode
		newTokens = append(newTokens, req.OutputTokens[startIndex-req.InputLen()])

		// Decode pre-check: if the last block is full and no free blocks exist,
		// fail fast without any state mutation. Mirrors vLLM's universal pre-check
		// (kv_cache_manager.py:334-336 / single_type_kv_cache_manager.py:95-101).
		//
		// For decode, get_num_blocks_to_allocate returns max(cdiv(tokens, blockSize) - len(blocks), 0).
		// This simplifies to: last block full → need 1 new block; last block has spare → need 0.
		//
		// When no RequestMap entry exists (e.g., preempted request whose blocks were
		// released but ProgressIndex was not reset), we need 1 new block unconditionally
		// since there is no existing block with spare capacity.
		//
		// So does any growth of its bounded state (#1941).
		if ids, hasBlocks := kvc.RequestMap[reqID]; hasBlocks && len(ids) > 0 {
			lastBlk := kvc.Blocks[ids[len(ids)-1]]
			var newBlk int64
			if util.Len64(lastBlk.Tokens) == kvc.BlockSizeTokens {
				newBlk = 1
			}
			need := stateNeed + kvc.slotBlocksFromFree(reqID, endIndex, int64(len(ids)), newBlk)
			if need > kvc.countFreeBlocks() {
				logrus.Debugf("KV cache full: cannot allocate decode block for req %s (%d needed, %d free)", reqID, need, kvc.countFreeBlocks())
				return false
			}
		} else {
			// No existing blocks — need 1 new block for this decode token.
			if stateNeed+kvc.slotBlocksFromFree(reqID, endIndex, 0, 1) > kvc.countFreeBlocks() {
				logrus.Debugf("KV cache full: cannot allocate decode block for req %s (no existing blocks, %d free)", reqID, kvc.countFreeBlocks())
				return false
			}
		}
	}
	// A shrinking state frees its blocks before any are taken, as vLLM's allocate_slots runs
	// remove_skipped_blocks before it allocates (kv_cache_manager.py:546-556 at v0.31.0).
	if stateNeed < 0 {
		kvc.resizeState(reqID, kvc.stateTarget(startIndex, endIndex))
	}
	newTokenProgressIndex := int64(0)
	for newTokenProgressIndex < util.Len64(newTokens) { // non-inclusive endIndex
		ids, ok := kvc.RequestMap[reqID]
		latestBlk := &KVBlock{}
		if ok {
			// Running request path: KV cache has already seen this request before.
			// The latest block needs to be filled first, followed by new blocks.
			// Note: Running requests do NOT re-claim cached blocks (vLLM parity).
			// Preempted requests reset ProgressIndex to 0 and re-enter via the !ok
			// path, where they claim all cached blocks upfront.
			latestBlk = kvc.Blocks[ids[len(ids)-1]]
		} else {
			// KV cache is seeing this request for the first time (beginning of prefill)
			// append the cached blocks to this request's ID map

			for _, blockId := range cachedBlocks {
				blk := kvc.Blocks[blockId]
				blk.RefCount++
				if !blk.InUse {
					blk.InUse = true
					kvc.removeFromFreeList(blk)
				}
				kvc.CacheHits++
				logrus.Debugf("Hit KV Cache for req: %s of length: %d", req.ID, util.Len64(cachedBlocks)*kvc.BlockSizeTokens)
				kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blockId)
			}

			// Update latestBlk to the last claimed block if we claimed any
			if len(cachedBlocks) > 0 {
				ids = kvc.RequestMap[reqID] // Refresh ids after appending cached blocks
				latestBlk = kvc.Blocks[ids[len(ids)-1]]
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
				if !kvc.remoteKV[reqID] {
					kvc.HashToBlock[h] = latestBlk.ID
				}
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
				blk := kvc.takeBlock(reqID)
				if blk == nil {
					panic(fmt.Sprintf("popFreeBlock returned nil after pre-check passed for req %s: INV-4 violation", reqID))
				}

				// Lazy hash deletion (vLLM parity): clear old hash before filling
				// with new content. Matches vLLM's _maybe_evict_cached_block
				// semantics (block_pool.py:331-356). Hash was preserved in
				// popFreeBlock so preempted requests could find their cached
				// prefix blocks on readmission.
				// Only the entry that names this block: a block whose content another
				// block also holds may never have been registered (AllocateRemoteKV).
				kvc.dropHash(blk)

				// start and end are the range of tokens in blk
				start := newTokenProgressIndex
				end := newTokenProgressIndex + kvc.BlockSizeTokens
				logrus.Debugf("Assigning new blocks: req = %s, newTokenProgressIndex = %d, ogStartIdx= %d, ogEndIdx = %d, startBlk=%d, endBlk=%d", req.ID, newTokenProgressIndex, startIndex, endIndex, start, end)
				if end > util.Len64(newTokens) {
					end = util.Len64(newTokens)
				}
				tok := newTokens[start:end]
				blk.Tokens = append([]sim.TokenID{}, tok...) // copy tokens
				blk.RefCount = 1
				blk.InUse = true
				kvc.CacheMisses++

				if util.Len64(blk.Tokens) == kvc.BlockSizeTokens && req.ProgressIndex < req.InputLen() {
					// Only compute prefix hash during prefill (not decode).
					// During decode, blocks hold output tokens that should not
					// participate in prefix caching (input sequences only).
					h := hash.HashBlock(prevHash, blk.Tokens)
					blk.Hash = h
					if !kvc.remoteKV[reqID] {
						kvc.HashToBlock[h] = blk.ID
					}
					prevHash = h
				}
				// allocated is the block IDs allocated for this request
				kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blk.ID)
				newTokenProgressIndex = end
			}
		}
	}
	// Grown last: a cached prefix block can sit on the free list, and is claimed above.
	if stateNeed > 0 {
		kvc.resizeState(reqID, kvc.stateTarget(startIndex, endIndex))
	}
	kvc.reserveSlots(reqID, endIndex)

	return true
}

// sequenceFits reports whether an allocation of a request over [startIndex, endIndex),
// claiming cachedBlocks on a first allocation, passes the admission gate and finds every
// block it needs on the free list: its token blocks not already held or claimable in use,
// the growth of its bounded state (#1941) from stateStart (loadedStateStart) and its slot
// reservation (#1627), counted as AllocateKVBlocks' own pre-check counts them. The offload
// tiers ask it before they reload anything, so a refusal leaves no side effect (vLLM's
// allocate_slots likewise refuses before it touches a block). Always true when the store
// charges neither term, which keeps those runs byte-identical.
func (kvc *KVCacheState) sequenceFits(req *sim.Request, startIndex, stateStart, endIndex int64, cachedBlocks []int64) bool {
	if kvc.SequenceState.Hold == nil && !kvc.Speculative.Enabled {
		return true
	}
	if kvc.admissionGateRefuses(req, cachedBlocks) {
		return false
	}
	reqID := req.ID
	held := util.Len64(kvc.RequestMap[reqID])
	var claimed, claimedFromFree int64
	if held == 0 {
		claimed = util.Len64(cachedBlocks)
		for _, id := range cachedBlocks {
			if !kvc.Blocks[id].InUse {
				claimedFromFree++
			}
		}
	}
	tokenBlocks := max((endIndex+kvc.BlockSizeTokens-1)/kvc.BlockSizeTokens-held-claimed, 0)
	need := claimedFromFree + kvc.stateDelta(reqID, stateStart, endIndex) +
		kvc.slotBlocksFromFree(reqID, endIndex, held+claimed, tokenBlocks)
	return need <= kvc.countFreeBlocks()
}

// loadedStateStart is the token a new admission's bounded state is sized from when an offload
// tier loads its prefix: past the loaded tokens, which are computed, not scheduled. vLLM sizes
// a window group's blocks from total_computed_tokens, the local and the external (loaded) hit
// together, and skips the blocks the window has passed (get_num_blocks_to_allocate,
// single_type_kv_cache_manager.py:178-245 at v0.31.0), so a loaded prefix holds only its
// window. reloadableEnd is the tier's ReloadablePrefixEnd; the result is capped, as batch
// formation caps the computed baseline, below the prompt's last token and at endIndex.
// Sizing it from startIndex instead charges the whole loaded prefix as in flight, which on a
// pool smaller than that charge refuses the request forever, however empty the pool (#1955).
func (kvc *KVCacheState) loadedStateStart(req *sim.Request, startIndex, endIndex, reloadableEnd int64, ok bool) int64 {
	if !ok {
		return startIndex
	}
	return max(startIndex, min(reloadableEnd, max(req.InputLen()-1, 0), endIndex))
}

// chargeSequence sizes a request's bounded state for [startIndex, endIndex) and tops up its
// slot reservation, for an offload tier that admits or extends a request by committing
// reloaded blocks alone, without AllocateKVBlocks. The tier checked sequenceFits before
// reloading, so the free list holds them; a shortfall is an INV-4 bug.
func (kvc *KVCacheState) chargeSequence(reqID string, startIndex, endIndex int64) {
	if delta := kvc.stateDelta(reqID, startIndex, endIndex); delta != 0 {
		if delta > kvc.countFreeBlocks() {
			panic(fmt.Sprintf("growing req %s's state by %d blocks with %d free after sequenceFits passed: INV-4 violation",
				reqID, delta, kvc.countFreeBlocks()))
		}
		kvc.resizeState(reqID, kvc.stateTarget(startIndex, endIndex))
	}
	if slots := kvc.slotBlocksFromFree(reqID, endIndex, util.Len64(kvc.RequestMap[reqID]), 0); slots > kvc.countFreeBlocks() {
		panic(fmt.Sprintf("reserving req %s %d slot blocks with %d free after sequenceFits passed: INV-4 violation",
			reqID, slots, kvc.countFreeBlocks()))
	}
	kvc.reserveSlots(reqID, endIndex)
}

// stateTarget is the bounded-state blocks a request holds once [startIndex, endIndex) is
// scheduled: SequenceState.Hold of its computed and in-flight tokens; 0 without a state.
func (kvc *KVCacheState) stateTarget(startIndex, endIndex int64) int64 {
	if kvc.SequenceState.Hold == nil {
		return 0
	}
	return kvc.SequenceState.Hold(startIndex, max(endIndex-startIndex, 0))
}

// stateDelta is how many blocks a request's bounded state grows by (negative: shrinks by)
// when [startIndex, endIndex) is scheduled.
func (kvc *KVCacheState) stateDelta(reqID string, startIndex, endIndex int64) int64 {
	if kvc.SequenceState.Hold == nil {
		return 0
	}
	return kvc.stateTarget(startIndex, endIndex) - util.Len64(kvc.StateBlocks[reqID])
}

// admissionGateRefuses is vLLM's full_sequence_must_fit check on a request's first
// allocation (scheduler_reserve_full_isl, on by default): the whole prompt's blocks and its
// bounded state at SequenceState.Gate -- each window group at min(cdiv(num_tokens),
// max_admission_blocks_per_request) -- must be free, or the request is not admitted
// (kv_cache_manager.py:517-534, single_type_kv_cache_manager.py:216-229 at v0.31.0). It
// gates only; what the request then holds grows step by step. It applies to every model, as
// scheduler_reserve_full_isl does (vllm/config/scheduler.py:191, default True): a model
// without bounded state gates on its prompt blocks alone. Without it a long prompt is admitted
// on its first chunk into a pool that cannot hold the rest, and its next chunk preempts
// (INV-PD-9's queue-not-thrash law).
func (kvc *KVCacheState) admissionGateRefuses(req *sim.Request, cachedBlocks []int64) bool {
	if len(kvc.RequestMap[req.ID]) > 0 || len(kvc.StateBlocks[req.ID]) > 0 {
		return false
	}
	var claimedFromFree int64
	for _, id := range cachedBlocks {
		if !kvc.Blocks[id].InUse {
			claimedFromFree++
		}
	}
	promptBlocks := max((req.InputLen()+kvc.BlockSizeTokens-1)/kvc.BlockSizeTokens-util.Len64(cachedBlocks), 0)
	need := claimedFromFree + promptBlocks
	if kvc.SequenceState.Gate != nil {
		need += kvc.SequenceState.Gate(req.InputLen())
	}
	if need > kvc.countFreeBlocks() {
		logrus.Debugf("KV admission gate: req %s needs %d blocks for its full prompt and state, %d free", req.ID, need, kvc.countFreeBlocks())
		return true
	}
	return false
}

// slotTarget is the blocks a speculating request must hold once endIndex tokens are
// scheduled: vLLM's num_tokens_need_slot = computed + new + num_lookahead_tokens, in blocks
// (vllm/v1/core/kv_cache_manager.py allocate_slots at v0.31.0). 0 when the reservation is off.
func (kvc *KVCacheState) slotTarget(endIndex int64) int64 {
	if !kvc.Speculative.Enabled {
		return 0
	}
	slots := endIndex + kvc.Speculative.LookaheadTokens
	if m := kvc.Speculative.MaxModelLen; m > 0 {
		slots = min(slots, max(m, endIndex)) // min(... + lookahead, max_model_len), kv_cache_manager.py:536-538
	}
	return (slots + kvc.BlockSizeTokens - 1) / kvc.BlockSizeTokens
}

// slotBlocksFromFree is how many blocks an allocation takes off the free list for a
// request's tokens and slot reservation: its newBlocks token blocks, less the spare blocks it
// already holds, plus whatever more the reservation needs past the heldAfter+newBlocks token
// blocks it will hold. Without a reservation it is newBlocks.
func (kvc *KVCacheState) slotBlocksFromFree(reqID string, endIndex, heldTokenBlocks, newBlocks int64) int64 {
	spare := util.Len64(kvc.SpareBlocks[reqID])
	fromSpare := min(spare, newBlocks)
	topUp := max(kvc.slotTarget(endIndex)-(heldTokenBlocks+newBlocks)-(spare-fromSpare), 0)
	return newBlocks - fromSpare + topUp
}

// takeBlock returns the block a request's next tokens go in: one of its spare blocks when it
// holds any, else one off the free list.
func (kvc *KVCacheState) takeBlock(reqID string) *KVBlock {
	if spare := kvc.SpareBlocks[reqID]; len(spare) > 0 {
		blk := kvc.Blocks[spare[0]]
		if len(spare) == 1 {
			delete(kvc.SpareBlocks, reqID)
		} else {
			kvc.SpareBlocks[reqID] = spare[1:]
		}
		return blk
	}
	return kvc.popFreeBlock()
}

// reserveSlots tops a request's spare blocks up to its slot target. The caller's pre-check
// guarantees the free list can supply them.
func (kvc *KVCacheState) reserveSlots(reqID string, endIndex int64) {
	short := kvc.slotTarget(endIndex) - util.Len64(kvc.RequestMap[reqID]) - util.Len64(kvc.SpareBlocks[reqID])
	for i := int64(0); i < short; i++ {
		kvc.SpareBlocks[reqID] = append(kvc.SpareBlocks[reqID], kvc.takeBlankBlock(reqID).ID)
	}
}

// takeBlankBlock takes a block off the free list to hold no tokens yet. Its stale hash is
// dropped at once, so no prefix may match it while it is held.
func (kvc *KVCacheState) takeBlankBlock(reqID string) *KVBlock {
	blk := kvc.popFreeBlock()
	if blk == nil {
		panic(fmt.Sprintf("popFreeBlock returned nil reserving blocks for req %s after the pre-check passed: INV-4 violation", reqID))
	}
	kvc.dropHash(blk)
	blk.RefCount = 1
	blk.InUse = true
	return blk
}

// resizeState grows or shrinks a request's bounded-state blocks to target. Growth takes
// blank blocks off the free list (the caller's pre-check guarantees them); shrinking returns
// the last ones to it.
func (kvc *KVCacheState) resizeState(reqID string, target int64) {
	held := kvc.StateBlocks[reqID]
	for util.Len64(held) < target {
		held = append(held, kvc.takeBlankBlock(reqID).ID)
	}
	for util.Len64(held) > target {
		blk := kvc.Blocks[held[len(held)-1]]
		held = held[:len(held)-1]
		blk.RefCount = 0
		blk.InUse = false
		kvc.appendToFreeList(blk)
	}
	if len(held) == 0 {
		delete(kvc.StateBlocks, reqID)
		return
	}
	kvc.StateBlocks[reqID] = held
}

// dropHash clears a block's prefix hash before it is reused, removing the index entry only
// when the entry still names this block: another block may have taken the hash over since --
// a fully cached prompt recomputes its last block (#1943) -- and that live mapping must
// survive the stale copy's reuse. vLLM's BlockHashToBlockMap.pop(key, block_id) likewise
// removes only the popped block's entry (vllm/v1/core/block_pool.py:101-118 at v0.31.0).
func (kvc *KVCacheState) dropHash(blk *KVBlock) {
	if blk.Hash == "" {
		return
	}
	if id, ok := kvc.HashToBlock[blk.Hash]; ok && id == blk.ID {
		delete(kvc.HashToBlock, blk.Hash)
	}
	blk.Hash = ""
}

// popFreeBlock evicts a block from the free list and prepares it for reuse.
// Hash entries are preserved (lazy deletion) - they will be cleared when the
// block is filled with new content in AllocateKVBlocks allocation loop.
// Matches vLLM's block_pool.py:313-318 + _maybe_evict_cached_block semantics.
func (kvc *KVCacheState) popFreeBlock() *KVBlock {
	head := kvc.FreeHead
	if head == nil {
		return nil
	}
	kvc.removeFromFreeList(head)
	// Hash stays intact - will be cleared when block is filled with new content (vLLM parity)
	head.Tokens = nil
	return head
}

// countFreeBlocks returns the number of blocks not currently in use.
// This is a direct read of the free list counter (vLLM parity), not arithmetic derivation.
func (kvc *KVCacheState) countFreeBlocks() int64 {
	return kvc.FreeBlockCnt
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
// NOTE: With the check-then-act allocation pattern, rollback does not exist.
// This method is used by TieredKVCache before calling gpu.AllocateKVBlocks.
// The inner AllocateKVBlocks pre-check sees the reduced FreeBlockCnt (already
// decremented by removeFromFreeList during commitCachedBlocks), so the pre-check
// correctly accounts for the committed blocks. In BLIS's single-threaded DES,
// FreeBlockCnt cannot decrease between the inner pre-check and allocation loop.
func (kvc *KVCacheState) commitCachedBlocks(reqID string, cachedBlocks []int64) {
	for _, blockID := range cachedBlocks {
		blk := kvc.Blocks[blockID]
		blk.RefCount++
		if !blk.InUse {
			blk.InUse = true
			kvc.removeFromFreeList(blk)
		}
		kvc.CacheHits++
		kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blockID)
	}
}

// ReleaseKVBlocks deallocates blocks used by a completed request.
// Each block's refcount is decremented and may be returned to the free list.
func (kvc *KVCacheState) ReleaseKVBlocks(req *sim.Request) {
	// The bounded-state and spare blocks hold nothing reusable, so they are freed first
	// and are the first of this request's blocks to be reused.
	for _, blank := range []map[string][]int64{kvc.StateBlocks, kvc.SpareBlocks} {
		for _, id := range blank[req.ID] {
			blk := kvc.Blocks[id]
			blk.RefCount = 0
			blk.InUse = false
			kvc.appendToFreeList(blk)
		}
		delete(blank, req.ID)
	}
	ids := kvc.RequestMap[req.ID]
	delete(kvc.RequestMap, req.ID)
	delete(kvc.remoteKV, req.ID)
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
			kvc.appendToFreeList(blk)
		}
	}
}

// BlockSize returns the number of tokens per block.
func (kvc *KVCacheState) BlockSize() int64 { return kvc.BlockSizeTokens }

// UsedBlocks returns the number of blocks currently in use.
// Derived from TotalBlocks - FreeBlockCnt (read-only for callers).
func (kvc *KVCacheState) UsedBlocks() int64 { return kvc.TotalBlocks - kvc.FreeBlockCnt }

// RequestBlocks implements sim.RequestBlockReporter: the blocks reqID holds.
func (kvc *KVCacheState) RequestBlocks(reqID string) []int64 { return kvc.RequestMap[reqID] }

var (
	_ sim.RequestBlockReporter = (*KVCacheState)(nil)
	_ sim.RemoteKVReserver     = (*KVCacheState)(nil)
)

// AllocateRemoteKV implements sim.RemoteKVReserver: allocate as AllocateKVBlocks does, but
// register none of the request's block hashes until CacheRemoteKV -- vLLM allocates a
// request loading KV asynchronously with delay_cache_blocks (sched/scheduler.py:1216 at
// v0.31.0) and caches its blocks only once the load finishes (_update_waiting_for_remote_kv,
// :3052-3085), so no request hits on blocks a pull is still filling.
func (kvc *KVCacheState) AllocateRemoteKV(req *sim.Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	if kvc.remoteKV == nil {
		kvc.remoteKV = make(map[string]bool)
	}
	kvc.remoteKV[req.ID] = true
	if !kvc.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks) {
		delete(kvc.remoteKV, req.ID)
		return false
	}
	return true
}

// CacheRemoteKV implements sim.RemoteKVReserver: the pull has landed, so register the
// request's full blocks in the prefix cache (vLLM's cache_blocks). A hash another block
// already registered keeps that block.
func (kvc *KVCacheState) CacheRemoteKV(reqID string) {
	if !kvc.remoteKV[reqID] {
		return
	}
	delete(kvc.remoteKV, reqID)
	for _, id := range kvc.RequestMap[reqID] {
		if h := kvc.Blocks[id].Hash; h != "" {
			if _, taken := kvc.HashToBlock[h]; !taken {
				kvc.HashToBlock[h] = id
			}
		}
	}
}

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

// verifyBlockConservation walks the free list and block InUse flags independently
// to verify INV-4: freeListLen + inUseCount == TotalBlocks.
// Returns nil if conservation holds, or an error describing the violation.
// Intended for debug-mode step-boundary assertions.
func (kvc *KVCacheState) verifyBlockConservation() error {
	freeListLen := int64(0)
	node := kvc.FreeHead
	for node != nil {
		freeListLen++
		node = node.NextFree
	}

	inUseCount := int64(0)
	for _, blk := range kvc.Blocks {
		if blk.InUse {
			inUseCount++
		}
	}

	if freeListLen+inUseCount != kvc.TotalBlocks {
		return fmt.Errorf("block conservation violated: freeList=%d + inUse=%d != total=%d",
			freeListLen, inUseCount, kvc.TotalBlocks)
	}

	if freeListLen != kvc.FreeBlockCnt {
		return fmt.Errorf("FreeBlockCnt drift: counter=%d, actual free list=%d",
			kvc.FreeBlockCnt, freeListLen)
	}

	return nil
}
