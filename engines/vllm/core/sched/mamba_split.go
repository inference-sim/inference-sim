// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/kv_cache_interface.py (get_mamba_prefill_checkpoint_position and
// is_mamba_prefill_checkpoint_valid, :1110-1148).

package sched

import "github.com/inference-sim/blis-vllm/core/kv"

// MambaSplitConfig is what Scheduler.__init__ reads to set up the Mamba block-aligned split
// (vllm/v1/core/sched/scheduler.py:375-410@5df63a72), as plain inputs.
type MambaSplitConfig struct {
	// Groups are the KV cache groups (kv_cache_config.kv_cache_groups); the Mamba ones have
	// Kind kv.Mamba and carry MambaCacheMode, MambaCheckpointBlocks, MambaCheckpointAlignment.
	Groups []kv.Group
	// BlockSize is cache_config.block_size, the engine's block size (:451).
	BlockSize int
	// SchedulerBlockSize is the Scheduler's self.block_size (:209), which differs from BlockSize
	// only under context parallelism; 0 means BlockSize.
	SchedulerBlockSize int
	// HashBlockSize is hash_block_size (:325-327).
	HashBlockSize int
	// MaxNumScheduledTokens is max_num_scheduled_tokens, the per-step token budget (:137, :508).
	MaxNumScheduledTokens int
	// LongPrefillTokenThreshold is scheduler_config.long_prefill_token_threshold as configured
	// (:509), not the per-step threshold schedule() derives.
	LongPrefillTokenThreshold int
	// EagleBlockDrop is use_eagle_block_drop (:316).
	EagleBlockDrop bool
	// PartialHashHits is kv_cache_manager.coordinator.enable_partial_hash_hits
	// (kv_cache_coordinator.py:684-692@5df63a72), which the KV manager decides.
	PartialHashHits bool
	// SharedPrefixCheckpoint is kv_cache_manager.mamba_shared_prefix_checkpoint
	// (kv_cache_manager.py:186-191@5df63a72): enable_mamba_shared_prefix_checkpoint, and an
	// eagle group, partial hash hits, and no re-prefillable tail. It is false by default.
	SharedPrefixCheckpoint bool
}

// MambaSplit is the scheduler state behind _mamba_block_aligned_split
// (scheduler.py:375-410, 443-545@5df63a72).
type MambaSplit struct {
	cfg MambaSplitConfig
	// hasMamba is has_mamba_layers.
	hasMamba bool
	// need is need_mamba_block_aligned_split.
	need bool
	// alignment is mamba_prefill_checkpoint_alignment: the first Mamba group's, 0 for None.
	alignment int
	// hasCheckpointBlocks is mamba_has_prefill_checkpoint_blocks.
	hasCheckpointBlocks bool
	// partialCacheHit is mamba_partial_cache_hit.
	partialCacheHit bool
	// sharedPrefix is mamba_shared_prefix_checkpoint.
	sharedPrefix bool
}

// NewMambaSplit derives the split's state from the KV groups (scheduler.py:375-410).
func NewMambaSplit(c MambaSplitConfig) *MambaSplit {
	if c.SchedulerBlockSize == 0 {
		c.SchedulerBlockSize = c.BlockSize
	}
	m := &MambaSplit{cfg: c, hasCheckpointBlocks: true}
	first := true
	align := false
	for _, g := range c.Groups {
		if g.Kind != kv.Mamba {
			continue
		}
		m.hasMamba = true
		if first {
			// The first Mamba group's alignment and mode stand for all (:379-387); a model
			// whose Mamba groups disagree is vLLM's own TODO.
			m.alignment = g.MambaCheckpointAlignment
			align = g.MambaCacheMode == "align"
			first = false
		}
		if g.MambaCheckpointBlocks <= 0 {
			m.hasCheckpointBlocks = false
		}
	}
	m.hasCheckpointBlocks = m.hasMamba && m.hasCheckpointBlocks
	m.need = m.hasMamba && align
	m.partialCacheHit = m.need && c.HashBlockSize < c.SchedulerBlockSize && c.PartialHashHits
	m.sharedPrefix = m.partialCacheHit && c.SharedPrefixCheckpoint
	return m
}

// Needed is need_mamba_block_aligned_split: whether schedule() calls Split.
func (m *MambaSplit) Needed() bool { return m != nil && m.need }

// PartialCacheHit is mamba_partial_cache_hit.
func (m *MambaSplit) PartialCacheHit() bool { return m.partialCacheHit }

// SharedPrefixCheckpoint is mamba_shared_prefix_checkpoint.
func (m *MambaSplit) SharedPrefixCheckpoint() bool { return m.sharedPrefix }

// MambaSplitRequest is the request state Split reads.
type MambaSplitRequest struct {
	NumComputed int // request.num_computed_tokens
	NumPrompt   int // request.num_prompt_tokens
	NumTokens   int // request.num_tokens, prompt plus output
	// SharedPrefixBoundary is request.shared_prefix_boundary, 0 when none (request.py:206).
	SharedPrefixBoundary int
}

// Split is _mamba_block_aligned_split (scheduler.py:443-545): it clips a prefill chunk of
// numNew tokens so that it ends where Mamba state must be cached, and returns the chunk's
// length. newLocal and external are num_new_local_computed_tokens and
// num_external_computed_tokens of a request being admitted, 0 for a running one.
func (m *MambaSplit) Split(r MambaSplitRequest, numNew, newLocal, external int) int {
	start := r.NumComputed + newLocal + external
	// Split only during prefill: num_tokens - 1 extends this to a resumed request replaying
	// its output tokens (:464).
	prefillEnd := max(r.NumPrompt, r.NumTokens-1)
	if start >= prefillEnd {
		return numNew
	}

	blockSize := m.cfg.BlockSize
	ckptBlockSize := m.cfg.SchedulerBlockSize
	if m.partialCacheHit {
		ckptBlockSize = m.cfg.HashBlockSize
	}
	lastCachePosition := mambaPrefillCheckpointPosition(r.NumTokens, blockSize, m.cfg.EagleBlockDrop)

	end := start + numNew
	checkpointPosition := mambaPrefillCheckpointPosition(prefillEnd, ckptBlockSize, m.cfg.EagleBlockDrop)
	useInternalCheckpoint := m.hasCheckpointBlocks && end >= prefillEnd &&
		mambaPrefillCheckpointValid(start, end, checkpointPosition, m.cfg.HashBlockSize, blockSize, m.alignment)
	if useInternalCheckpoint {
		lastCachePosition = 0
	}
	// Chunk ends must be block aligned, except the prompt's last chunk. A block too wide for
	// one chunk advances sub-block and re-aligns at the next boundary (:503-514).
	if end < prefillEnd {
		maxPrefillTokens := m.cfg.MaxNumScheduledTokens
		if t := m.cfg.LongPrefillTokenThreshold; t > 0 {
			maxPrefillTokens = min(maxPrefillTokens, t)
		}
		alignedEnd := end / blockSize * blockSize
		if alignedEnd > start || blockSize <= maxPrefillTokens {
			end = alignedEnd
		}
	}

	nextBlockBoundary := (start/blockSize + 1) * blockSize
	tailBoundary := 0
	if m.partialCacheHit && !useInternalCheckpoint {
		tailBoundary = mambaPrefillCheckpointPosition(r.NumPrompt, m.cfg.HashBlockSize, m.cfg.EagleBlockDrop)
	}
	junction := r.SharedPrefixBoundary
	// Block-floored: a sub-block junction's state is not separately cacheable. Python's //
	// floors toward minus infinity, as floorDiv does, for a junction below start.
	blockFloored := start + floorDiv(junction-start, blockSize)*blockSize
	junctionStop := blockFloored
	if m.sharedPrefix && junction <= r.NumPrompt {
		junctionStop = junction
	}

	var stops [5]int
	if start%blockSize != 0 && !useInternalCheckpoint {
		stops[0] = nextBlockBoundary
	}
	stops[1] = lastCachePosition
	if lastCachePosition < tailBoundary && tailBoundary < r.NumPrompt {
		stops[2] = tailBoundary
	}
	if start < junction && junction < end {
		stops[3] = junctionStop
	}
	// Stop at the earliest mandatory position strictly inside the chunk (:543-544).
	for _, s := range stops {
		if start < s && s < end {
			end = min(end, s)
		}
	}
	return max(end-start, 0)
}

func floorDiv(a, b int) int {
	q := a / b
	if a%b != 0 && (a < 0) != (b < 0) {
		q--
	}
	return q
}

// mambaPrefillCheckpointPosition is get_mamba_prefill_checkpoint_position
// (vllm/v1/kv_cache_interface.py:1110-1122@5df63a72). Unexported and local to this file: the
// KV side may carry the same function; keep one.
func mambaPrefillCheckpointPosition(numTokens, hashBlockSize int, dropEagleBlock bool) int {
	// Without EAGLE, leave one token to recompute on resend; EAGLE's block drop already does.
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
// (vllm/v1/kv_cache_interface.py:1125-1148@5df63a72): whether a backend can export the
// checkpoint in this query. alignment 0 stands for None. Unexported and local to this file.
func mambaPrefillCheckpointValid(queryStart, queryEnd, checkpointPosition, hashBlockSize, mambaBlockSize, alignment int) bool {
	if alignment <= 0 {
		return false
	}
	initialStateCol := floorDiv(queryStart-1, mambaBlockSize)
	checkpointCol := cdiv(queryEnd, mambaBlockSize) - 2
	return queryStart%hashBlockSize == 0 &&
		checkpointCol > initialStateCol &&
		queryStart+hashBlockSize <= checkpointPosition &&
		queryStart < checkpointPosition && checkpointPosition < queryEnd &&
		(checkpointPosition-queryStart)%alignment == 0
}
