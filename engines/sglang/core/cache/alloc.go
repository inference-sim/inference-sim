// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/allocator/token.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/mem_cache/allocator/paged.py,
// python/sglang/srt/mem_cache/allocator/base.py and python/sglang/srt/mem_cache/common.py
// (_evict_until_allocatable).

package cache

import (
	"fmt"
	"slices"
)

// TokenAllocator is TokenToKVPoolAllocator (allocator/token.py:28-96@297e54fd): a free list of
// single-token slots. Slot 0 is never handed out; SGLang writes padded tokens' dummy outputs to
// it (token.py:43).
type TokenAllocator struct {
	size     int
	needSort bool
	kv       KVPool
	free     []int64
	// release holds freed slots while needSort is set, until an alloc runs short
	// (base.py:222-229).
	release []int64
	// grouping is free_group not None: frees are held in group until FreeGroupEnd.
	grouping bool
	group    [][]int64
}

// NewTokenAllocator is TokenToKVPoolAllocator.__init__ (token.py:31-40@297e54fd).
func NewTokenAllocator(size int, kv KVPool, needSort bool) *TokenAllocator {
	a := &TokenAllocator{size: size, needSort: needSort, kv: kv}
	a.Clear()
	return a
}

func (a *TokenAllocator) Size() int          { return a.size }
func (a *TokenAllocator) PageSize() int      { return 1 }
func (a *TokenAllocator) GetKVCache() KVPool { return a.kv }

// Clear is token.py:42-48@297e54fd: slots 1..size free, in order.
func (a *TokenAllocator) Clear() {
	a.free = make([]int64, a.size)
	for i := range a.free {
		a.free[i] = int64(i + 1)
	}
	a.grouping, a.group = false, nil
	a.release = nil
}

// AvailableSize is token.py:50-52@297e54fd.
func (a *TokenAllocator) AvailableSize() int { return len(a.free) + len(a.release) }

// Alloc is token.py:54-63@297e54fd: the first needSize free slots.
func (a *TokenAllocator) Alloc(needSize int) ([]int64, bool) {
	if a.needSort && needSize > len(a.free) {
		a.mergeAndSortFree()
	}
	if needSize > len(a.free) {
		return nil, false
	}
	out := slices.Clone(a.free[:needSize])
	a.free = a.free[needSize:]
	return out, true
}

// AllocExtend is base.py:255-256@297e54fd: only the paged allocator extends by pages.
func (a *TokenAllocator) AllocExtend(prefixLens, seqLens []int, lastLoc []int64, extendNumTokens int) ([]int64, bool) {
	panic("alloc_extend is only for paged allocator")
}

// AllocDecode is base.py:258-259@297e54fd.
func (a *TokenAllocator) AllocDecode(seqLens []int, lastLoc []int64) ([]int64, bool) {
	panic("alloc_decode is only for paged allocator")
}

// mergeAndSortFree is base.py:222-229@297e54fd.
func (a *TokenAllocator) mergeAndSortFree() {
	if len(a.release) > 0 {
		a.free = append(slices.Clone(a.free), a.release...)
		slices.Sort(a.free)
		a.release = nil
	}
}

// Free is token.py:65-75@297e54fd: freed slots go to the back of the free list, to the release
// list while needSort is set, or to the open group.
func (a *TokenAllocator) Free(slots []int64) {
	if len(slots) == 0 {
		return
	}
	switch {
	case a.grouping:
		a.group = append(a.group, slices.Clone(slots))
	case a.needSort:
		a.release = append(a.release, slots...)
	default:
		a.free = append(slices.Clone(a.free), slots...)
	}
}

// FreeSegment is base.py:271-281@297e54fd.
func (a *TokenAllocator) FreeSegment(slots []int64, startPos int) {
	a.Free(slots)
}

// FreeSegments is base.py:283-290@297e54fd.
func (a *TokenAllocator) FreeSegments(segments []Segment) {
	for _, s := range pageDisjoint(segments, 1) {
		a.FreeSegment(s.Slots, s.StartPos)
	}
}

// FreeGroupBegin is base.py:205-207@297e54fd.
func (a *TokenAllocator) FreeGroupBegin() {
	if a.grouping {
		panic("free groups cannot be nested")
	}
	a.grouping, a.group = true, nil
}

// FreeGroupEnd is base.py:209-212@297e54fd: the group's frees, in order, as one free.
func (a *TokenAllocator) FreeGroupEnd() {
	pending := a.group
	a.grouping, a.group = false, nil
	if len(pending) > 0 {
		a.Free(slices.Concat(pending...))
	}
}

func (a *TokenAllocator) EvictToFreeTokens(tree PrefixCache, numTokens int) {
	evictToFreeTokens(a, tree, numTokens)
}

func (a *TokenAllocator) CheckDecodeCapacity(numTokens int, tree PrefixCache) bool {
	return checkDecodeCapacity(a, tree, numTokens)
}

func (a *TokenAllocator) RequestSlotReserve(hasReqPoolSlot bool, prefillTokens int) int { return 0 }

func (a *TokenAllocator) CreatePrefillBudget(tree PrefixCache, numMixedDecodeTokens int) PrefillBudget {
	return NewPrefillBudget(a, tree, numMixedDecodeTokens)
}

func (a *TokenAllocator) MaxNewTokensForMemory(inputTokens, maxNewTokens, tokenCapacity int, slidingWindowSize, chunkSize *int) (int, bool) {
	return maxNewTokensForMemory(1, inputTokens, maxNewTokens, tokenCapacity), true
}

// VerifyByteAccounting is base.py:157-160@297e54fd: a static pool has no byte model.
func (a *TokenAllocator) VerifyByteAccounting() []string { return []string{} }

// PagedAllocator is PagedTokenToKVPoolAllocator (allocator/paged.py:136-421@297e54fd): a free
// list of pages, each page_size consecutive slots. Page 0 is never handed out (paged.py:396).
type PagedAllocator struct {
	size, pageSize, numPages int
	needSort                 bool
	kv                       KVPool
	// free is page ids, not slots.
	free []int64
	// staged holds freed pages while needSort is set (paged.py:402-404).
	staged    [][]int64
	numStaged int
	grouping  bool
	// group is free_group (slots freed by Free); pageGroup is free_page_ids_group (pages freed by
	// FreeSegment).
	group     [][]int64
	pageGroup [][]int64
}

// NewPagedAllocator is PagedTokenToKVPoolAllocator.__init__ (paged.py:143-166@297e54fd).
func NewPagedAllocator(size, pageSize int, kv KVPool, needSort bool) *PagedAllocator {
	if pageSize < 1 {
		panic(fmt.Sprintf("page size %d", pageSize))
	}
	a := &PagedAllocator{size: size, pageSize: pageSize, numPages: size / pageSize, needSort: needSort, kv: kv}
	a.Clear()
	return a
}

func (a *PagedAllocator) Size() int          { return a.size }
func (a *PagedAllocator) PageSize() int      { return a.pageSize }
func (a *PagedAllocator) GetKVCache() KVPool { return a.kv }

// Clear is paged.py:395-404@297e54fd: pages 1..num_pages free, in order.
func (a *PagedAllocator) Clear() {
	a.free = make([]int64, a.numPages)
	for i := range a.free {
		a.free[i] = int64(i + 1)
	}
	a.grouping, a.group, a.pageGroup = false, nil, nil
	a.staged, a.numStaged = nil, 0
}

// AvailableSize is paged.py:168-169@297e54fd, in slots.
func (a *PagedAllocator) AvailableSize() int { return (len(a.free) + a.numStaged) * a.pageSize }

// mergeAndSortFree is paged.py:174-179@297e54fd.
func (a *PagedAllocator) mergeAndSortFree() {
	if len(a.staged) == 0 {
		return
	}
	all := slices.Concat(append([][]int64{a.free}, a.staged...)...)
	slices.Sort(all)
	a.free = all
	a.staged, a.numStaged = nil, 0
}

// Alloc is paged.py:181-202@297e54fd: needSize/page_size whole pages from the front of the free
// list, as the slots they hold.
func (a *PagedAllocator) Alloc(needSize int) ([]int64, bool) {
	numPages := needSize / a.pageSize
	if numPages > len(a.free) {
		a.mergeAndSortFree()
	}
	if numPages > len(a.free) {
		return nil, false
	}
	pages := a.free[:numPages]
	a.free = a.free[numPages:]
	out := make([]int64, 0, numPages*a.pageSize)
	for _, p := range pages {
		for j := 0; j < a.pageSize; j++ {
			out = append(out, p*int64(a.pageSize)+int64(j))
		}
	}
	return out, true
}

// AllocExtend is alloc_extend (paged.py:204-277@297e54fd) with the CPU kernel,
// alloc_extend_naive (paged.py:41-114): each request's new tokens first fill the page its prefix
// ends in (after lastLoc), then take whole new pages, then one more page for the tail. The
// requests take new pages from the front of the free list in batch order.
func (a *PagedAllocator) AllocExtend(prefixLens, seqLens []int, lastLoc []int64, extendNumTokens int) ([]int64, bool) {
	bs := len(prefixLens)
	if len(seqLens) != bs || len(lastLoc) != bs {
		panic(fmt.Sprintf("alloc_extend: %d prefixes, %d sequences, %d last slots", bs, len(seqLens), len(lastLoc)))
	}
	ps := a.pageSize
	numNewPages := 0
	for i := range bs {
		numNewPages += ceilDiv(seqLens[i], ps) - ceilDiv(prefixLens[i], ps)
	}
	// A strict upper bound on the pages needed; only when it exceeds the free list is the exact
	// count checked (paged.py:220-239).
	if maxNewPages := extendNumTokens/ps + bs + 1; maxNewPages > len(a.free) {
		a.mergeAndSortFree()
		if maxNewPages > len(a.free) && numNewPages > len(a.free) {
			return nil, false
		}
	}
	out := make([]int64, 0, extendNumTokens)
	nextPage := 0
	for i := range bs {
		prefix, seq, last := prefixLens[i], seqLens[i], lastLoc[i]
		ceilPrefix := ceilDiv(prefix, ps) * ps
		floorSeq := seq / ps * ps
		num1 := min(seq, ceilPrefix) - prefix
		num2, num3 := 0, 0
		if prefix+num1 != seq {
			num2, num3 = floorSeq-ceilPrefix, seq-floorSeq
		}
		for l := 0; l < num1; l++ {
			out = append(out, last+1+int64(l))
		}
		for l := 0; l < num2; l++ {
			out = append(out, a.free[nextPage+l/ps]*int64(ps)+int64(l%ps))
		}
		nextPage += num2 / ps
		if num3 > 0 {
			for l := 0; l < num3; l++ {
				out = append(out, a.free[nextPage]*int64(ps)+int64(l))
			}
			nextPage++
		}
	}
	if len(out) != extendNumTokens {
		panic(fmt.Sprintf("alloc_extend: %d slots for %d extend tokens", len(out), extendNumTokens))
	}
	a.free = a.free[numNewPages:]
	return out, true
}

// AllocDecode is alloc_decode (paged.py:279-326@297e54fd) with the CPU kernel,
// alloc_decode_naive (paged.py:117-133): a request whose new token starts a page takes the next
// free page, in batch order; every other request's slot follows lastLoc.
func (a *PagedAllocator) AllocDecode(seqLens []int, lastLoc []int64) ([]int64, bool) {
	bs := len(seqLens)
	if len(lastLoc) != bs {
		panic(fmt.Sprintf("alloc_decode: %d sequences, %d last slots", bs, len(lastLoc)))
	}
	if bs > len(a.free) {
		a.mergeAndSortFree()
	}
	ps := a.pageSize
	numNewPages := 0
	for _, s := range seqLens {
		if s%ps == 1 {
			numNewPages++
		}
	}
	if numNewPages > len(a.free) {
		return nil, false
	}
	out := make([]int64, bs)
	rank := 0
	for i, s := range seqLens {
		if numNewPages > 0 && floorMod(s-1, ps) == 0 {
			out[i] = a.free[min(rank, numNewPages-1)] * int64(ps)
			rank++
		} else {
			out[i] = lastLoc[i] + 1
		}
	}
	a.free = a.free[numNewPages:]
	return out, true
}

func floorMod(a, b int) int {
	m := a % b
	if m != 0 && (m < 0) != (b < 0) {
		m += b
	}
	return m
}

// Free is paged.py:328-338@297e54fd: the distinct pages the slots lie on, in ascending order
// (torch.unique sorts), are released.
func (a *PagedAllocator) Free(slots []int64) {
	if len(slots) == 0 {
		return
	}
	if a.grouping {
		a.group = append(a.group, slices.Clone(slots))
		return
	}
	a.releasePages(uniquePages(slots, a.pageSize))
}

func uniquePages(slots []int64, pageSize int) []int64 {
	pages := make([]int64, len(slots))
	for i, s := range slots {
		pages[i] = floorDiv(s, int64(pageSize))
	}
	slices.Sort(pages)
	return slices.Compact(pages)
}

func floorDiv(a, b int64) int64 {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

// FreeSegment is paged.py:340-357@297e54fd: a segment starting on a page boundary frees the page
// of every page_size-th slot, in the segment's order, with no deduplication.
func (a *PagedAllocator) FreeSegment(slots []int64, startPos int) {
	if len(slots) == 0 {
		return
	}
	if startPos%a.pageSize != 0 {
		panic(fmt.Sprintf("segment start %d is not page-aligned", startPos))
	}
	var pages []int64
	for i := 0; i < len(slots); i += a.pageSize {
		pages = append(pages, floorDiv(slots[i], int64(a.pageSize)))
	}
	a.freePageIDs(pages)
}

// freePageIDs is paged.py:359-369@297e54fd.
func (a *PagedAllocator) freePageIDs(pages []int64) {
	if len(pages) == 0 {
		return
	}
	if a.grouping {
		a.pageGroup = append(a.pageGroup, slices.Clone(pages))
		return
	}
	a.releasePages(pages)
}

// releasePages is paged.py:375-380@297e54fd: released pages go to the front of the free list,
// or to the staged list while needSort is set.
func (a *PagedAllocator) releasePages(pages []int64) {
	if a.needSort {
		a.staged = append(a.staged, slices.Clone(pages))
		a.numStaged += len(pages)
		return
	}
	a.free = slices.Concat(pages, a.free)
}

// FreeSegments is base.py:283-290@297e54fd.
func (a *PagedAllocator) FreeSegments(segments []Segment) {
	for _, s := range pageDisjoint(segments, a.pageSize) {
		a.FreeSegment(s.Slots, s.StartPos)
	}
}

// FreeGroupBegin is paged.py:382-384@297e54fd.
func (a *PagedAllocator) FreeGroupBegin() {
	if a.grouping {
		panic("free groups cannot be nested")
	}
	a.grouping, a.group, a.pageGroup = true, nil, nil
}

// FreeGroupEnd is paged.py:386-393@297e54fd: the group's slot frees are released first, then its
// page frees, each ahead of the free list.
func (a *PagedAllocator) FreeGroupEnd() {
	pending := a.group
	a.grouping, a.group = false, nil
	if len(pending) > 0 {
		a.Free(slices.Concat(pending...))
	}
	if len(a.pageGroup) > 0 {
		a.releasePages(slices.Concat(a.pageGroup...))
		a.pageGroup = nil
	}
}

func (a *PagedAllocator) EvictToFreeTokens(tree PrefixCache, numTokens int) {
	evictToFreeTokens(a, tree, numTokens)
}

func (a *PagedAllocator) CheckDecodeCapacity(numTokens int, tree PrefixCache) bool {
	return checkDecodeCapacity(a, tree, numTokens)
}

func (a *PagedAllocator) RequestSlotReserve(hasReqPoolSlot bool, prefillTokens int) int { return 0 }

func (a *PagedAllocator) CreatePrefillBudget(tree PrefixCache, numMixedDecodeTokens int) PrefillBudget {
	return NewPrefillBudget(a, tree, numMixedDecodeTokens)
}

func (a *PagedAllocator) MaxNewTokensForMemory(inputTokens, maxNewTokens, tokenCapacity int, slidingWindowSize, chunkSize *int) (int, bool) {
	return maxNewTokensForMemory(a.pageSize, inputTokens, maxNewTokens, tokenCapacity), true
}

func (a *PagedAllocator) VerifyByteAccounting() []string { return []string{} }

// pageDisjoint is base.py:306-317@297e54fd: the non-empty segments, each starting on a page
// after the last page of the one before.
func pageDisjoint(segments []Segment, pageSize int) []Segment {
	var out []Segment
	prevEnd := -1
	for _, s := range segments {
		n := len(s.Slots)
		if n == 0 {
			continue
		}
		if prevEnd >= 0 && s.StartPos/pageSize <= (prevEnd-1)/pageSize {
			panic(fmt.Sprintf("segment at %d shares a page with the one ending at %d", s.StartPos, prevEnd))
		}
		prevEnd = s.StartPos + n
		out = append(out, s)
	}
	return out
}

// evictToFreeTokens is base.py:132-145@297e54fd: evict the shortfall from a tree that shares
// prefixes. The follow-up loop of _evict_until_allocatable (common.py:292-318) runs only under
// page-interleave sharding, which is refused (refuse.go), so it never runs here.
func evictToFreeTokens(a Allocator, tree PrefixCache, numTokens int) {
	if tree == nil || !tree.SupportsPrefixSharing() {
		return
	}
	if shortfall := numTokens - a.AvailableSize(); shortfall > 0 {
		tree.EvictForAlloc(EvictParams{NumTokens: shortfall})
	}
}

// checkDecodeCapacity is base.py:147-155@297e54fd.
func checkDecodeCapacity(a Allocator, tree PrefixCache, numTokens int) bool {
	a.EvictToFreeTokens(tree, numTokens)
	return a.AvailableSize() >= numTokens
}

// maxNewTokensForMemory is base.py:86-103@297e54fd: generation clipped to what an empty pool
// holds after the prompt's pages, one page and the padding slot.
func maxNewTokensForMemory(pageSize, inputTokens, maxNewTokens, tokenCapacity int) int {
	pagedInput := ceilDiv(inputTokens, pageSize) * pageSize
	return max(0, min(maxNewTokens, tokenCapacity-pagedInput-pageSize-1))
}

// ceilDiv is -(-a // b) for b > 0.
func ceilDiv(a, b int) int {
	q := a / b
	if a%b != 0 && (a > 0) == (b > 0) {
		q++
	}
	return q
}

// MambaFullCacheDonor is base.py:192-194@297e54fd: a static pool funds no Mamba state.
func (a *TokenAllocator) MambaFullCacheDonor() MambaFullCacheDonor { return nil }

// GetExtendAllocationDemand is base.py:144-153@297e54fd: the conservative count.
func (a *TokenAllocator) GetExtendAllocationDemand(prefixLens, seqLens []int, conservativeNumTokens, shardSize int) int {
	return conservativeNumTokens
}

// MambaFullCacheDonor is base.py:192-194@297e54fd.
func (a *PagedAllocator) MambaFullCacheDonor() MambaFullCacheDonor { return nil }

// GetExtendAllocationDemand is base.py:144-153@297e54fd; the paged allocator does not override it.
func (a *PagedAllocator) GetExtendAllocationDemand(prefixLens, seqLens []int, conservativeNumTokens, shardSize int) int {
	return conservativeNumTokens
}
