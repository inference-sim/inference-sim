// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/prefill_budget.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package cache

// FixedPrefillBudget is PrefillBudget (prefill_budget.py:62-168@297e54fd): the budget of one
// prefill pass over a single token pool. The offsets are what admitted requests will take, which
// the pool does not yet show; what remains is read live from the allocator and the tree cache,
// so a lock or a free between two admissions is seen by the second.
type FixedPrefillBudget struct {
	alloc         Allocator
	tree          PrefixCache
	pageSize      int
	totalOffset   int
	currentOffset int
}

// NewPrefillBudget is PrefillBudget.__init__ (prefill_budget.py:68-74@297e54fd): both offsets
// start at the decode tokens mixed into the batch.
func NewPrefillBudget(a Allocator, tree PrefixCache, numMixedDecodeTokens int) *FixedPrefillBudget {
	return &FixedPrefillBudget{alloc: a, tree: tree, pageSize: a.PageSize(),
		totalOffset: numMixedDecodeTokens, currentOffset: numMixedDecodeTokens}
}

// availableAndEvictable is prefill_budget.py:91-97@297e54fd.
func (b *FixedPrefillBudget) availableAndEvictable() int {
	evictable := 0
	if b.tree.SupportsMamba() {
		evictable = b.tree.FullEvictableSize()
	} else {
		evictable = b.tree.EvictableSize()
	}
	return b.alloc.AvailableSize() + evictable
}

// RemainingTotal is prefill_budget.py:99-101@297e54fd.
func (b *FixedPrefillBudget) RemainingTotal() int { return b.availableAndEvictable() - b.totalOffset }

// RemainingCurrent is prefill_budget.py:103-105@297e54fd.
func (b *FixedPrefillBudget) RemainingCurrent() int {
	return b.availableAndEvictable() - b.currentOffset
}

// HasCapacity is prefill_budget.py:111-112@297e54fd.
func (b *FixedPrefillBudget) HasCapacity() bool {
	return b.RemainingTotal() > 0 && b.RemainingCurrent() > 0
}

// CanAllocatePrefill is prefill_budget.py:130-138@297e54fd.
func (b *FixedPrefillBudget) CanAllocatePrefill(pagedInput int) bool {
	return pagedInput <= min(b.RemainingCurrent(), b.RemainingTotal())
}

// Reserve is prefill_budget.py:156-168@297e54fd: the extend rounded up to whole pages, one page
// and extraTokens now; the expected decode tokens as well against the total.
func (b *FixedPrefillBudget) Reserve(extendInputLen, maxNewTokens, extraTokens int) {
	extend := ceilDiv(extendInputLen, b.pageSize) * b.pageSize
	immediate := extend + b.pageSize + extraTokens
	b.totalOffset += immediate + maxNewTokens
	b.currentOffset += immediate
}
