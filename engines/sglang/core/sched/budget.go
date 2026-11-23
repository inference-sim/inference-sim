// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/prefill_budget.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package sched

import "github.com/inference-sim/inference-sim/engines/sglang/core/cache"

// budget is PrefillBudget (prefill_budget.py:62-168@297e54fd) over a single token pool. Its
// offsets are floats because the running requests' expected decode tokens are scaled by
// new_token_ratio (schedule_policy.py:785-795@297e54fd); cache.FixedPrefillBudget holds them as
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

// availableAndEvictable is _available_and_evictable (:91-97).
func (b *budget) availableAndEvictable() float64 {
	evictable := b.tree.EvictableSize()
	if b.tree.SupportsMamba() {
		evictable = b.tree.FullEvictableSize()
	}
	return float64(b.alloc.AvailableSize() + evictable)
}

func (b *budget) remainingTotal() float64   { return b.availableAndEvictable() - b.totalOffset }
func (b *budget) remainingCurrent() float64 { return b.availableAndEvictable() - b.currentOffset }

// hasCapacity is has_capacity (:111-112).
func (b *budget) hasCapacity() bool { return b.remainingTotal() > 0 && b.remainingCurrent() > 0 }

// checkPrefill is check_prefill (:114-128) on one pool: it admits when the request's total fits
// under what remains, and leaves the chunk limit as it was.
func (b *budget) checkPrefill(totalTokens int, chunkLimit *int) (bool, *int) {
	if float64(totalTokens) < b.remainingTotal() {
		return true, chunkLimit
	}
	return false, nil
}

// availableChunkTokens is available_chunk_tokens (:140-144): the chunk limit, cut to what remains
// in total; a continuing chunk with nothing left still makes progress to release its KV.
func (b *budget) availableChunkTokens(chunkLimit int) int {
	available := min(chunkLimit, int(b.remainingTotal()))
	if available > 0 {
		return available
	}
	return chunkLimit
}

// reserve is reserve (:156-168).
func (b *budget) reserve(extendInputLen int, maxNewTokens float64, extraTokens int) {
	immediate := b.ceilPaged(extendInputLen) + b.pageSize + extraTokens
	b.totalOffset += float64(immediate) + maxNewTokens
	b.currentOffset += float64(immediate)
}
