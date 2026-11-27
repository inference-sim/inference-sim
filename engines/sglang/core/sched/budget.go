// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/prefill_budget.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package sched

import "github.com/inference-sim/inference-sim/engines/sglang/core/cache"

// budget is PrefillBudget (prefill_budget.py:90-318@c4af57ff) over a single token pool. Its
// offsets are floats because the running requests' expected decode tokens are scaled by
// new_token_ratio (schedule_policy.py:1240-1251@c4af57ff); cache.FixedPrefillBudget holds them as
// integers and serves only the whole-token case.
type budget struct {
	alloc         cache.Allocator
	tree          cache.PrefixCache
	pageSize      int
	totalOffset   float64
	currentOffset float64
}

func newBudget(a cache.Allocator, t cache.PrefixCache, mixedDecodeTokens int) *budget {
	return &budget{alloc: a, tree: t, pageSize: a.PageSize(),
		totalOffset: float64(mixedDecodeTokens), currentOffset: float64(mixedDecodeTokens)}
}

func (b *budget) ceilPaged(n int) int { return ceilTo(n, b.pageSize) }

// availableAndEvictable is _available_and_evictable (:144-154).
func (b *budget) availableAndEvictable() float64 {
	evictable := b.tree.EvictableSize()
	if b.tree.SupportsMamba() {
		evictable = b.tree.FullEvictableSize()
	}
	return float64(b.alloc.AvailableSize() + evictable)
}

func (b *budget) remainingTotal() float64   { return b.availableAndEvictable() - b.totalOffset }
func (b *budget) remainingCurrent() float64 { return b.availableAndEvictable() - b.currentOffset }

// hasCapacity is has_capacity (:167-171), without context-parallel sharding.
func (b *budget) hasCapacity() bool { return b.remainingTotal() > 0 && b.remainingCurrent() > 0 }

// checkPrefill is check_prefill (:193-216) on one pool: it admits when the request's total fits
// under what remains, and leaves the chunk limit as it was.
func (b *budget) checkPrefill(totalTokens int, chunkLimit *int) (bool, *int) {
	if float64(totalTokens) < b.remainingTotal() {
		return true, chunkLimit
	}
	return false, nil
}

// availableChunkTokens is available_chunk_tokens (:259-268@c4af57ff): the chunk limit, cut to what
// remains in total. With nothing remaining the continuation parks (false) rather than take the
// headroom the running decodes were promised.
func (b *budget) availableChunkTokens(chunkLimit int) (int, bool) {
	available := min(chunkLimit, int(b.remainingTotal()))
	if available > 0 {
		return available, true
	}
	return 0, false
}

// reserve is reserve (:283-318@c4af57ff). A request cut short by a chunk cap takes the rest of its
// prompt and its decode tokens from the total as well when commitUnfinished is set, which only a
// continuing chunk under a chunk cap sets (add_chunked_req): later requests of the pass then see
// the KV that the rest will need.
func (b *budget) reserve(extendInputLen int, maxNewTokens float64, extraTokens int, unfinishedInputLen int, deferredMaxNewTokens float64, commitUnfinished bool) {
	immediate := b.ceilPaged(extendInputLen) + b.pageSize + extraTokens
	b.totalOffset += float64(immediate) + maxNewTokens
	b.currentOffset += float64(immediate)
	if commitUnfinished {
		b.totalOffset += float64(b.ceilPaged(unfinishedInputLen)) + deferredMaxNewTokens
	}
}
