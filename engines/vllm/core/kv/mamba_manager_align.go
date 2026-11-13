// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/single_type_kv_cache_manager.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/kv_cache_interface.py (get_mamba_prefill_checkpoint_position,
// is_mamba_prefill_checkpoint_valid) and vllm/v1/core/kv_cache_coordinator.py.

package kv

import (
	"fmt"
)

// This file is the MambaManager in "align" mode (single_type_kv_cache_manager.py:1415-2112
// @5df63a72) for one KV cache group, with the parts of SingleTypeKVCacheManager it relies on
// (:52-724). It is pure bookkeeping over a Pool and the group's block table; the Manager calls
// it where vLLM's coordinator calls the single-type manager:
//
//	vLLM                                        here
//	find_longest_cache_hit (classmethod)        mambaFindLongestHit
//	add_local_computed_blocks                   (*mambaAlign).addLocalComputedBlocks
//	get_num_blocks_to_allocate                  (*mambaAlign).numBlocksToAllocate
//	allocate_new_blocks                         (*mambaAlign).allocateNewBlocks
//	remove_skipped_blocks                       (*mambaAlign).removeSkippedBlocks
//	get_num_skipped_tokens                      (*mambaAlign).skippedTokens
//	cache_blocks                                (*mambaAlign).cacheBlocks
//	new_step_starts                             (*mambaAlign).newStepStarts
//	pop_blocks_for_free / free                  (*mambaAlign).popBlocksForFree / free
//
// Defaults this port assumes, each the value vLLM v0.32.0 ships:
//   - The retention interval is the config's (RetentionInterval): nil is dense retention, 0
//     vLLM's default (only the prompt's replay boundaries keep a state), a positive multiple
//     of the scheduler block size a periodic state (reachable_block_mask, :1479-1553).
//   - enable_mamba_shared_prefix_checkpoint is false (kv_cache_manager.py:186-195), so the
//     junction branch of _cache_partial_tail_block never runs; request.shared_prefix_boundary
//     itself is still read where vLLM reads it.
//   - No external KV connector and no boundary-state offload, so _pending_boundary_state_offloads,
//     _producer_partial_tail_reqs and finalize_partial_tail_offload (:1912-1925, :1935-1953,
//     :2002-2012) are dropped: they feed a connector and change no block table.
//   - DCP = PCP = 1 (:1458-1459).
//
// Not ported, refused by name: fine-grained partial hash hits (the coordinator's
// enable_partial_hash_hits, kv_cache_coordinator.py:684-706, which lowers
// cache_hit_alignment_tokens to the hash block size and makes a Mamba hit end inside a
// block). They arise only when the Mamba block is larger than the hash block, that is with
// groups of different block sizes under TP or DCP. They need the copy-on-write path
// (_partial_hit_reqs, _apply_cow, move_block_hashes) and a fine-grained full-attention
// lookup in the Manager as well; newMambaAlign refuses a group whose block size is not the
// hash block size, and addLocalComputedBlocks refuses a hit that ends inside a block.
//
// EAGLE's drop_eagle_checkpoint_block (kv_cache_coordinator.py:173-180) is ported: it is one
// argument of get_mamba_prefill_checkpoint_position, and the caller sets it in the config.

// mambaAlignConfig is what MambaManager.__init__ reads from the MambaSpec and the coordinator.
type mambaAlignConfig struct {
	// HashBlockSize is the block pool's hash_block_size.
	HashBlockSize int
	// SchedulerBlockSize is cache_hit_alignment_tokens when partial hash hits are off
	// (kv_cache_coordinator.py:699-706): hits land on it.
	SchedulerBlockSize int
	// NumSpeculativeBlocks is MambaSpec.num_speculative_blocks.
	NumSpeculativeBlocks int
	// NumPrefillCheckpointBlocks and PrefillCheckpointAlignment are the spec's fields of the
	// same names; an alignment of 0 is None (is_mamba_prefill_checkpoint_valid, :1119-1121).
	NumPrefillCheckpointBlocks int
	PrefillCheckpointAlignment int
	// DropEagleCheckpointBlock is set by the coordinator when EAGLE is on (:173-180).
	DropEagleCheckpointBlock bool
	// RetentionInterval is prefix_cache_retention_interval: nil is dense, 0 keeps the replay
	// boundaries' states only, a positive multiple of SchedulerBlockSize a state per segment
	// as well (kv_cache_coordinator.py:182-188).
	RetentionInterval *int
	// ReplayEagle is bool(eagle_group_ids), which shifts the replay boundaries
	// (kv_cache_coordinator.py:68-87, 365-370).
	ReplayEagle bool
}

type mambaCheckpoint struct{ position, idx int }

// mambaAlign is a MambaManager with mamba_cache_mode "align".
type mambaAlign struct {
	pool *Pool
	g    *group
	cfg  mambaAlignConfig
	// blockSize is the manager's block_size (:108).
	blockSize int
	// alignment is cache_hit_alignment_tokens.
	alignment int
	// hasCheckpointBlocks is has_prefill_checkpoint_blocks (:1431-1434).
	hasCheckpointBlocks bool

	cachedThisStep map[string]bool // cached_blocks_this_step, by hash key
	// lastStateBlockIdx is the index of the block the previous step ran in (:1441).
	lastStateBlockIdx map[string]int
	numRetired        map[string]int
	allocated         map[string]bool // _allocated_block_reqs
	checkpoints       map[string]mambaCheckpoint
}

// mambaReq is the part of a Request the manager reads when it caches blocks.
type mambaReq struct {
	ID string
	// Hashes are request.block_hashes, at the hash block size.
	Hashes []BlockHash
	// NumPromptTokens is request.num_prompt_tokens.
	NumPromptTokens int
	// Run carries request.all_token_ids and the adapter for BlockStored events.
	Run StoredRun
	// SharedPrefixBoundary is request.shared_prefix_boundary.
	SharedPrefixBoundary int
}

// newMambaAlign is MambaManager.__init__ for align mode over group g's block table.
func newMambaAlign(pool *Pool, g *group, cfg mambaAlignConfig) (*mambaAlign, error) {
	if g.blockSize <= 0 || cfg.HashBlockSize <= 0 || cfg.SchedulerBlockSize <= 0 {
		return nil, fmt.Errorf("kv: mamba align needs positive block sizes, got group %d, hash %d, scheduler %d",
			g.blockSize, cfg.HashBlockSize, cfg.SchedulerBlockSize)
	}
	if g.blockSize > cfg.HashBlockSize {
		return nil, fmt.Errorf("kv: enable_partial_hash_hits is not modelled: the mamba block size %d exceeds the hash block size %d "+
			"(kv_cache_coordinator.py:684-706@5df63a72)", g.blockSize, cfg.HashBlockSize)
	}
	if g.blockSize != cfg.HashBlockSize || cfg.SchedulerBlockSize%g.blockSize != 0 {
		return nil, fmt.Errorf("kv: mamba block size %d must equal the hash block size %d and divide the scheduler block size %d",
			g.blockSize, cfg.HashBlockSize, cfg.SchedulerBlockSize)
	}
	if r := cfg.RetentionInterval; r != nil && (*r < 0 || *r%cfg.SchedulerBlockSize != 0) {
		return nil, fmt.Errorf("kv: prefix_cache_retention_interval (%d) must be non-negative and a multiple of scheduler_block_size (%d) "+
			"(kv_cache_coordinator.py:57-62@5df63a72)", *r, cfg.SchedulerBlockSize)
	}
	if cfg.NumSpeculativeBlocks < 0 || cfg.NumPrefillCheckpointBlocks < 0 || cfg.PrefillCheckpointAlignment < 0 {
		return nil, fmt.Errorf("kv: mamba align counts must not be negative: %+v", cfg)
	}
	return &mambaAlign{
		pool: pool, g: g, cfg: cfg, blockSize: g.blockSize, alignment: cfg.SchedulerBlockSize,
		hasCheckpointBlocks: cfg.NumPrefillCheckpointBlocks > 0,
		cachedThisStep:      map[string]bool{}, lastStateBlockIdx: map[string]int{}, numRetired: map[string]int{},
		allocated: map[string]bool{}, checkpoints: map[string]mambaCheckpoint{},
	}, nil
}

// mambaPrefillCheckpointPosition is get_mamba_prefill_checkpoint_position
// (kv_cache_interface.py:1110-1121@5df63a72): the reusable boundary of a prefill.
func mambaPrefillCheckpointPosition(numTokens, hashBlockSize int, dropEagleBlock bool) int {
	// Without EAGLE, one token is left to recompute on resend; EAGLE's block drop already
	// leaves tokens to recompute.
	proofLimit := numTokens - 1
	if dropEagleBlock {
		proofLimit = numTokens
	}
	pos := proofLimit / hashBlockSize * hashBlockSize
	if dropEagleBlock {
		pos -= hashBlockSize
	}
	return max(pos, 0)
}

// mambaPrefillCheckpointValid is is_mamba_prefill_checkpoint_valid
// (kv_cache_interface.py:1124-1148@5df63a72): whether a backend can export the checkpoint in
// the query [queryStart, queryEnd). An alignment of 0 is None.
func mambaPrefillCheckpointValid(queryStart, queryEnd, checkpointPosition, hashBlockSize, mambaBlockSize,
	checkpointAlignment int) bool {
	if checkpointAlignment == 0 {
		return false
	}
	initialStateCol := floorDiv(queryStart-1, mambaBlockSize)
	checkpointCol := cdiv(queryEnd, mambaBlockSize) - 2
	return queryStart%hashBlockSize == 0 &&
		checkpointCol > initialStateCol &&
		queryStart+hashBlockSize <= checkpointPosition &&
		queryStart < checkpointPosition && checkpointPosition < queryEnd &&
		(checkpointPosition-queryStart)%checkpointAlignment == 0
}

// floorDiv is Python's // for a positive divisor.
func floorDiv(a, b int) int {
	q := a / b
	if a%b != 0 && a < 0 {
		q--
	}
	return q
}

// mambaFindLongestHit is MambaManager.find_longest_cache_hit (:1450-1520) for the groups
// groupIDs, which one spec group looks up together. hashes are the request's block hashes
// (hash block size = the Mamba block size), maxLength the longest hit allowed, alignment the
// coordinator's cache_hit_alignment_tokens. It returns, per group, the blocks of the hit with
// null blocks in front of the one state block that ends the hit, and the hit length in
// tokens (0 and empty lists on a miss). The EAGLE margin is not applied for a Mamba spec
// group (kv_cache_coordinator.py:853-857, 951-952): its finder never drops.
func mambaFindLongestHit(pool *Pool, hashes []BlockHash, maxLength int, groupIDs []uint32, blockSize,
	alignment int) ([][]*Block, int) {
	if alignment < blockSize {
		panic(fmt.Sprintf("kv: enable_partial_hash_hits is not modelled: alignment %d is below the mamba block size %d", alignment, blockSize))
	}
	computed := make([][]*Block, len(groupIDs))
	maxNumBlocks := min(maxLength/blockSize, len(hashes))
	// Search from right to left and stop at the first match (:1500-1519).
	for i := maxNumBlocks - 1; i >= 0; i-- {
		cached := make([]*Block, 0, len(groupIDs))
		for _, gid := range groupIDs {
			b := pool.CachedIn(hashes[i], gid)
			if b == nil {
				cached = nil
				break
			}
			cached = append(cached, b)
		}
		if cached == nil {
			continue
		}
		if blockSize != alignment && (i+1)*blockSize%alignment != 0 {
			continue
		}
		for j := range computed {
			// The hit length logic later assumes hit_length = len(blocks) * block_size, so null
			// blocks fill the front (:1512-1517).
			for k := 0; k < i; k++ {
				computed[j] = append(computed[j], pool.null)
			}
			computed[j] = append(computed[j], cached[j])
		}
		return computed, (i + 1) * blockSize
	}
	return computed, 0
}

// replayBoundaries is replay_boundaries (kv_cache_coordinator.py:68-87): the positions a later
// request replaying the prompt can resume at.
func replayBoundaries(numPrompt, schedulerBlockSize int, useEagle bool) []int {
	if !useEagle {
		return []int{numPrompt - 1}
	}
	b := schedulerBlockSize
	resend := (numPrompt - 1) / b * b
	extension := numPrompt / b * b
	lo, hi := max(resend-b, 0), max(extension-b, 0)
	if lo == hi {
		return []int{lo}
	}
	return []int{min(lo, hi), max(lo, hi)}
}

// reachableBlockMask is MambaManager.reachable_block_mask (:1522-1558) for blocks
// [start, end): nil caches every block (dense retention); otherwise the mask keeps one state
// per retention segment and the states ending on the reachable boundaries.
func reachableBlockMask(start, end, alignment, blockSize int, retention *int, reachable []int) []bool {
	if retention == nil || alignment == 0 {
		return nil
	}
	mask := make([]bool, end-start)
	if *retention != 0 {
		perSegment := *retention / blockSize
		if perSegment <= 1 {
			// Every block is a boundary.
			return nil
		}
		first := (start+perSegment)/perSegment*perSegment - 1
		for i := first - start; i < len(mask); i += perSegment {
			mask[i] = true
		}
	}
	for _, b := range reachable {
		aligned := b / alignment * alignment
		bb := aligned/blockSize - 1
		if start <= bb && bb < end {
			mask[bb-start] = true
		}
	}
	return mask
}

// skippedTokens is get_num_skipped_tokens (:1955-1960): a Mamba request keeps only the state
// of its last computed token.
func (a *mambaAlign) skippedTokens(numComputed int) int { return numComputed - 1 }

// running is request_id in num_cached_block.
func (a *mambaAlign) running(id string) bool {
	_, ok := a.g.numCached[id]
	return ok
}

// addLocalComputedBlocks is SingleTypeKVCacheManager.add_local_computed_blocks (:253-326) for
// a first-time allocation: the hit's null blocks below the last computed token are dropped
// and re-padded, the state block is touched and appended. Only a hit that ends on a block
// boundary is modelled (see the file comment).
func (a *mambaAlign) addLocalComputedBlocks(id string, newComputed []*Block, numLocalComputed, numExternal int) {
	if numLocalComputed%a.blockSize != 0 {
		panic(fmt.Sprintf("kv: enable_partial_hash_hits is not modelled: a local hit of %d tokens ends inside a %d-token mamba block",
			numLocalComputed, a.blockSize))
	}
	reqBlocks := a.g.reqBlocks[id]
	if len(reqBlocks) != 0 {
		panic("kv: add_local_computed_blocks needs a request without blocks")
	}
	skippedBlocks := a.skippedTokens(numLocalComputed+numExternal) / a.blockSize
	if a.skippedTokens(numLocalComputed+numExternal) < 0 {
		skippedBlocks = 0
	}
	if skippedBlocks > 0 {
		newComputed = newComputed[min(skippedBlocks, len(newComputed)):]
	}
	a.pool.Touch(newComputed)
	for i := 0; i < skippedBlocks; i++ {
		reqBlocks = append(reqBlocks, a.pool.null)
	}
	reqBlocks = append(reqBlocks, newComputed...)
	a.g.reqBlocks[id] = reqBlocks
	a.g.numCached[id] = len(reqBlocks)
}

// needsInternalCheckpoint is _needs_internal_checkpoint (:1627-1667).
func (a *mambaAlign) needsInternalCheckpoint(id string, queryStart, queryEnd, checkpointPosition, prefillEnd int) bool {
	checkpointIdx := cdiv(queryEnd, a.blockSize) - 2
	blocks := a.g.reqBlocks[id]
	if !a.hasCheckpointBlocks || queryEnd < prefillEnd {
		return false
	}
	if !mambaPrefillCheckpointValid(queryStart, queryEnd, checkpointPosition, a.cfg.HashBlockSize, a.blockSize,
		a.cfg.PrefillCheckpointAlignment) || checkpointIdx < 0 {
		return false
	}
	return checkpointIdx >= len(blocks) || blocks[checkpointIdx].Null ||
		(a.allocated[id] && checkpointIdx >= len(blocks)-a.cfg.NumSpeculativeBlocks)
}

// numBlocksToAllocate is get_num_blocks_to_allocate, align branch (:1669-1782): the free
// blocks the call needs, counting the prefix-hit blocks that are free (a touch takes them off
// the free queue). newComputed is this group's hit, totalComputed the computed tokens
// including external ones, numLocalComputed the local hit's, numTokensMain the main model's
// token count (draft slots are not allocated in align mode, :1704-1709). admissionCap marks the
// full-sequence gate: it must not record the request's checkpoint (:1759-1766). prefillEnd is
// the scheduler's, ignored under dense retention (the coordinator zeroes it).
//
// A hit on a block another request cached in this same step cannot be used: the call returns
// one more than the pool holds, so the Manager sees no room and schedules the request next
// step (:1685-1692).
func (a *mambaAlign) numBlocksToAllocate(id string, newComputed []*Block, totalComputed, numLocalComputed,
	numTokensMain int, admissionCap bool, prefillEnd int) int {
	if n := len(newComputed); n > 0 && newComputed[n-1].Hash != nil && a.cachedThisStep[string(newComputed[n-1].Hash)] {
		return a.pool.NumBlocks() + 1
	}
	numTokens := numTokensMain
	required := cdiv(numTokens, a.blockSize) + a.cfg.NumSpeculativeBlocks
	numNew := required - len(newComputed) - len(a.g.reqBlocks[id])
	if numLocalComputed%a.blockSize != 0 && len(newComputed) > 0 {
		panic("kv: enable_partial_hash_hits is not modelled: a local hit ends inside a mamba block")
	}
	checkpointPosition := mambaPrefillCheckpointPosition(numTokens, a.alignment, a.cfg.DropEagleCheckpointBlock)
	if !a.needsInternalCheckpoint(id, totalComputed, numTokens, checkpointPosition, prefillEnd) {
		checkpointPosition = 0
	}
	checkpointBlock := 0
	if checkpointPosition > 0 {
		checkpointBlock = 1
	}
	if !admissionCap {
		if checkpointPosition > 0 {
			a.checkpoints[id] = mambaCheckpoint{checkpointPosition, cdiv(numTokens, a.blockSize) - 2}
		} else {
			delete(a.checkpoints, id)
		}
	}
	if numNew > 0 {
		physicalCap := 1 + checkpointBlock
		if !a.allocated[id] {
			physicalCap += a.cfg.NumSpeculativeBlocks
		}
		numNew = min(numNew, physicalCap)
	}
	evictable := 0
	for _, b := range newComputed {
		if b.RefCnt == 0 && !b.Null {
			evictable++
		}
	}
	return numNew + evictable
}

// allocateNewBlocks is allocate_new_blocks, align branch (:1784-1893). It extends the request's
// block table to cdiv(numTokensMain, block_size) + num_speculative_blocks entries: null
// blocks for the states no one needs, scratch blocks moved forward, and fresh blocks from the
// pool. It returns the blocks the worker's table gained, in table order: the nulls and the
// relocated scratch blocks included (returned_blocks = req_blocks[prev_block_len:], :1859).
func (a *mambaAlign) allocateNewBlocks(id string, numTokensMain int) []*Block {
	req := a.g.reqBlocks[id]
	numRequired := cdiv(numTokensMain, a.blockSize) + a.cfg.NumSpeculativeBlocks
	_, hasCheckpoint := a.checkpoints[id]
	checkpointBlock := 0
	if hasCheckpoint {
		checkpointBlock = 1
	}
	// numRequired may be below len(req) if blocks were over-allocated last round.
	if numRequired <= len(req) && !hasCheckpoint {
		a.g.reqBlocks[id] = req
		a.allocated[id] = true
		return nil
	}
	prev := len(req)
	blocksAllocated := a.allocated[id]
	numSpec := a.cfg.NumSpeculativeBlocks
	if blocksAllocated {
		// The running state sits at the block before the speculative ones (:1812-1816).
		a.lastStateBlockIdx[id] = prev - 1 - numSpec
	} else if prev > 0 {
		// A new request that hit the prefix cache keeps the hit state in its last block.
		a.lastStateBlockIdx[id] = prev - 1
	}
	numSkipped := numRequired - numSpec - 1
	if prev < numSkipped {
		// Null blocks, except the internal checkpoint's.
		for i := prev; i < numSkipped-checkpointBlock; i++ {
			req = append(req, a.pool.null)
		}
	}
	if blocksAllocated {
		// Relocate the exclusively owned scratch blocks this step leaves behind; in a
		// checkpoint step the one at the checkpoint index stays (:1836-1848).
		for idx := prev - numSpec; idx < prev; idx++ {
			if idx < numSkipped-checkpointBlock {
				req = a.relocateSpeculativeBlock(req, idx)
			} else {
				break
			}
		}
	}
	numNew := max(numRequired-len(req), 0)
	maxNew := 1 + checkpointBlock
	if !blocksAllocated {
		maxNew += numSpec
	}
	if numNew > maxNew {
		panic(fmt.Sprintf("kv: mamba align needs %d new blocks, at most %d expected", numNew, maxNew))
	}
	newBlocks := a.pool.New(numNew)
	returned := append([]*Block(nil), req[prev:]...)
	req = append(req, newBlocks...)
	a.g.reqBlocks[id] = req
	a.allocated[id] = true
	return append(returned, newBlocks...)
}

// relocateSpeculativeBlock is _relocate_speculative_block (:1895-1905): the block moves to
// the end of the table and a null block takes its place.
func (a *mambaAlign) relocateSpeculativeBlock(req []*Block, idx int) []*Block {
	b := req[idx]
	if !a.pool.IsBlockWritable(b) {
		panic("kv: speculative mamba blocks must be exclusively owned and unhashed before relocation")
	}
	req = append(req, b)
	req[idx] = a.pool.null
	return req
}

// removeBlocksInRange is _remove_blocks_in_range, align branch (:1560-1585): free the live
// blocks in [first, last), tail first, behind null blocks; blocks already retired are not
// revisited.
func (a *mambaAlign) removeBlocksInRange(id string, first, last int) {
	blocks := a.g.reqBlocks[id]
	first = max(first, a.numRetired[id])
	last = min(last, len(blocks))
	if first >= last {
		return
	}
	var freed []*Block
	for i := last - 1; i >= first; i-- {
		if blocks[i].Null {
			continue
		}
		freed = append(freed, blocks[i])
		blocks[i] = a.pool.null
	}
	if len(freed) > 0 {
		a.pool.Free(freed)
	}
	a.numRetired[id] = last
}

// removeSkippedBlocks is remove_skipped_blocks (:1587-1625) with the base class's
// (:600-640): blocks below the last computed token are retired, and the state block of two
// steps ago is freed once the running state has moved past it.
func (a *mambaAlign) removeSkippedBlocks(id string, processedComputed int) {
	if skipped := a.skippedTokens(processedComputed); skipped > 0 {
		numSkippedBlocks := min(skipped/a.blockSize, len(a.g.reqBlocks[id]))
		a.removeBlocksInRange(id, 0, numSkippedBlocks)
	}
	idx, ok := a.lastStateBlockIdx[id]
	if ok && idx < cdiv(processedComputed, a.blockSize)-1 {
		blocks := a.g.reqBlocks[id]
		if blocks[idx] != a.pool.null {
			a.pool.Free([]*Block{blocks[idx]})
			blocks[idx] = a.pool.null
		}
	}
}

// popBlocksForFree is pop_blocks_for_free (:1935-1953) with the base class's (:562-580): the
// request's bookkeeping is dropped and its blocks returned in allocation order, not yet freed.
func (a *mambaAlign) popBlocksForFree(id string) []*Block {
	delete(a.allocated, id)
	delete(a.lastStateBlockIdx, id)
	delete(a.numRetired, id)
	delete(a.checkpoints, id)
	blocks := a.g.reqBlocks[id]
	delete(a.g.reqBlocks, id)
	delete(a.g.numCached, id)
	return blocks
}

// free is SingleTypeKVCacheManager.free (:581-589): the blocks go back tail first.
func (a *mambaAlign) free(id string) {
	blocks := a.popBlocksForFree(id)
	rev := make([]*Block, len(blocks))
	for i, b := range blocks {
		rev[len(blocks)-1-i] = b
	}
	a.pool.Free(rev)
}

// newStepStarts is new_step_starts (:1971-1972).
func (a *mambaAlign) newStepStarts() { a.cachedThisStep = map[string]bool{} }

// cacheBlocks is MambaManager.cache_blocks (:1962-2000) with the base class's (:468-516) under
// dense retention: the full blocks up to numTokens are cached (null blocks skipped), the
// checkpoint block is re-keyed at its boundary, and every hash cached now is marked as cached
// this step, so a request admitted in the same step does not hit it.
func (a *mambaAlign) cacheBlocks(r mambaReq, numTokens int) {
	before := a.g.numCached[r.ID]
	numFull := numTokens / a.blockSize
	if before < numFull && before < len(r.Hashes) {
		blocks := a.g.reqBlocks[r.ID]
		hashes := r.Hashes
		numFull = min(numFull, len(hashes))
		reachable := replayBoundaries(r.NumPromptTokens, a.cfg.SchedulerBlockSize, a.cfg.ReplayEagle)
		if r.SharedPrefixBoundary != 0 {
			reachable = append(reachable, r.SharedPrefixBoundary)
		}
		mask := reachableBlockMask(before, numFull, a.alignment, a.blockSize, a.cfg.RetentionInterval, reachable)
		a.pool.CacheFullMaskedIn(a.g.id, blocks, hashes, before, numFull, r.Run, a.blockSize, mask)
		a.g.numCached[r.ID] = numFull
	}
	after := a.g.numCached[r.ID]
	if key := a.cachePartialTailBlock(r, numTokens); key != nil {
		a.cachedThisStep[string(key)] = true
	}
	if after > before {
		blocks := a.g.reqBlocks[r.ID]
		for idx := before; idx < after; idx++ {
			b := blocks[idx]
			// Null blocks (skipped states) carry no hash and are not cached-this-step.
			if b.Null || b.Hash == nil {
				continue
			}
			a.cachedThisStep[string(b.Hash)] = true
		}
	}
}

// cachePartialTailBlock is _cache_partial_tail_block (:2014-2112) with the shared-prefix
// junction branches dropped (that setting is off) and the partial-hit bookkeeping dropped (a
// hash block equals the Mamba block, so a boundary never falls inside one). It returns the
// hash key it registered, or nil.
func (a *mambaAlign) cachePartialTailBlock(r mambaReq, numTokens int) []byte {
	hashBlockSize := a.cfg.HashBlockSize
	// Re-key the reserved block at its exported checkpoint boundary.
	if ck, ok := a.checkpoints[r.ID]; ok {
		blocks := a.g.reqBlocks[r.ID]
		if ck.idx < 0 || ck.idx >= len(blocks) {
			panic(fmt.Sprintf("kv: checkpoint block index %d outside the %d-block table", ck.idx, len(blocks)))
		}
		block := blocks[ck.idx]
		// The prompt's final chunk caches only up to its last full block, so its checkpoint is
		// the prompt's reusable boundary, not a transient one.
		isPromptCheckpoint := ck.position == mambaPrefillCheckpointPosition(r.NumPromptTokens, a.alignment,
			a.cfg.DropEagleCheckpointBlock)
		isRetainedCheckpoint := false
		if ck.position%a.blockSize == 0 {
			bi := ck.position/a.blockSize - 1
			m := reachableBlockMask(bi, bi+1, a.alignment, a.blockSize, a.cfg.RetentionInterval, nil)
			isRetainedCheckpoint = m == nil || m[0]
		}
		if !isPromptCheckpoint && !isRetainedCheckpoint && ck.position != r.SharedPrefixBoundary {
			// The transient state overwrites this slot, so neither it nor the full-block hash
			// registered earlier this step is reusable.
			a.pool.MaybeEvictCached(block)
			return nil
		}
		if block.Hash != nil && block.HashTokens == ck.position {
			return nil
		}
		return a.pool.CachePartialBlock(a.g.id, block, r.Hashes, ck.position, hashBlockSize, r.Run, true)
	}
	if numTokens <= 0 || numTokens%hashBlockSize != 0 {
		return nil
	}
	latestPromptHashBoundary := mambaPrefillCheckpointPosition(r.NumPromptTokens, hashBlockSize,
		a.cfg.DropEagleCheckpointBlock)
	if numTokens != latestPromptHashBoundary {
		return nil
	}
	blockIdx := cdiv(numTokens, a.blockSize) - 1
	blocks := a.g.reqBlocks[r.ID]
	if blockIdx >= len(blocks) {
		return nil
	}
	source := blocks[blockIdx]
	if source.Null {
		return nil
	}
	return a.pool.CachePartialBlock(a.g.id, source, r.Hashes, numTokens, hashBlockSize, r.Run, false)
}
