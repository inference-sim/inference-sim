// Original to blis-sglang.

package cache

import (
	"slices"
	"testing"
)

func mustPanic(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: no refusal", what)
		}
	}()
	f()
}

func slotsOf(t *testing.T, a Allocator, n int) []int64 {
	t.Helper()
	got, ok := a.Alloc(n)
	if !ok {
		t.Fatalf("Alloc(%d) failed with %d available", n, a.AvailableSize())
	}
	return got
}

// A token allocator hands out exactly what is free, and with needSort it merges its released slots
// into the free list only when a request outruns what is free (token.py:44-62@297e54fd), so a
// released slot lower than every free one is taken first only then.
func TestTokenAllocatorMergesReleasedSlotsOnlyWhenShort(t *testing.T) {
	a := NewTokenAllocator(3, nil, false)
	if got := slotsOf(t, a, 3); !slices.Equal(got, []int64{1, 2, 3}) {
		t.Errorf("all three slots: %v", got)
	}
	if _, ok := a.Alloc(1); ok {
		t.Error("a slot from an empty allocator")
	}

	mk := func() *TokenAllocator {
		s := NewTokenAllocator(3, nil, true)
		slotsOf(t, s, 1) // takes slot 1
		s.Free([]int64{1})
		return s
	}
	if got := slotsOf(t, mk(), 1); !slices.Equal(got, []int64{2}) {
		t.Errorf("one slot while two are free took %v; the released slot 1 is not merged yet", got)
	}
	if got := slotsOf(t, mk(), 2); !slices.Equal(got, []int64{2, 3}) {
		t.Errorf("exactly the free slots took %v; the released slot 1 is not merged yet", got)
	}
	if got := slotsOf(t, mk(), 3); !slices.Equal(got, []int64{1, 2, 3}) {
		t.Errorf("one more than free took %v, want the released slot merged in order", got)
	}
}

// A paged allocator takes exactly the pages that are free; with needSort it stages freed pages
// and merges them in only when a request outruns the free list (paged.py:181-202).
func TestPagedAllocatorMergesStagedPagesOnlyWhenShort(t *testing.T) {
	mustPanic(t, "page size 0", func() { NewPagedAllocator(8, 0, nil, false) })
	one := NewPagedAllocator(3, 1, nil, false)
	if got := slotsOf(t, one, 3); !slices.Equal(got, []int64{1, 2, 3}) {
		t.Errorf("page size 1, all pages: %v", got)
	}
	if _, ok := one.Alloc(1); ok {
		t.Error("a page from an empty allocator")
	}
	mk := func() *PagedAllocator {
		p := NewPagedAllocator(12, 4, nil, true) // pages 1, 2, 3
		slotsOf(t, p, 4)                         // page 1
		p.Free([]int64{4, 5, 6, 7})              // staged
		return p
	}
	if got := slotsOf(t, mk(), 8); !slices.Equal(got, []int64{8, 9, 10, 11, 12, 13, 14, 15}) {
		t.Errorf("exactly the free pages took %v; staged page 1 is not merged yet", got)
	}
	if got := slotsOf(t, mk(), 12); !slices.Equal(got[:4], []int64{4, 5, 6, 7}) || len(got) != 12 {
		t.Errorf("one page more than free took %v, want staged page 1 merged first", got)
	}
}

// alloc_extend takes new pages from the free list in batch order, merging staged pages only when
// the bound on the pages needed outruns the free list, and refusing only when the exact count does
// (paged.py:204-277).
func TestPagedAllocExtendBoundsTheNewPages(t *testing.T) {
	p := func() *PagedAllocator {
		a := NewPagedAllocator(16, 4, nil, true) // pages 1..4
		slotsOf(t, a, 4)
		a.Free([]int64{4, 5, 6, 7}) // page 1 staged; free is 2, 3, 4
		return a
	}
	// The bound is 4/4 + 1 + 1 = 3 pages, which the three free pages cover: no merge.
	if got, ok := p().AllocExtend([]int{0}, []int{4}, []int64{-1}, 4); !ok || !slices.Equal(got, []int64{8, 9, 10, 11}) {
		t.Errorf("extend by one page took %v, %v; staged page 1 must stay staged", got, ok)
	}
	// The bound is 12/4 + 2 = 5 pages, over the three free, but the exact count is three pages,
	// which the three free cover: the extend succeeds.
	a := NewPagedAllocator(16, 4, nil, false)
	slotsOf(t, a, 4) // free is pages 2, 3, 4
	got, ok := a.AllocExtend([]int{0}, []int{12}, []int64{-1}, 12)
	if !ok || !slices.Equal(got, []int64{8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}) {
		t.Errorf("extend by exactly the free pages: %v, %v", got, ok)
	}
	if _, ok := a.AllocExtend([]int{0}, []int{4}, []int64{-1}, 4); ok {
		t.Error("an extend with no free page succeeded")
	}
	mustPanic(t, "more sequences than prefixes", func() {
		NewPagedAllocator(16, 4, nil, false).AllocExtend([]int{0}, []int{4, 4}, []int64{-1}, 4)
	})
	mustPanic(t, "more last slots than prefixes", func() {
		NewPagedAllocator(16, 4, nil, false).AllocExtend([]int{0}, []int{4}, []int64{-1, -1}, 4)
	})
}

// alloc_decode gives a new page to each request whose token starts one, in batch order, merging
// staged pages when the batch outruns the free list (paged.py:279-326).
func TestPagedAllocDecodeTakesNewPagesInOrder(t *testing.T) {
	a := NewPagedAllocator(16, 4, nil, true) // pages 1..4
	slotsOf(t, a, 4)
	a.Free([]int64{4, 5, 6, 7}) // page 1 staged; free is 2, 3, 4
	got, ok := a.AllocDecode([]int{1, 1, 1}, []int64{0, 0, 0})
	if !ok || !slices.Equal(got, []int64{8, 12, 16}) {
		t.Errorf("three requests on three free pages: %v, %v; staged page 1 must stay staged", got, ok)
	}
	b := NewPagedAllocator(16, 4, nil, true)
	slotsOf(t, b, 4)
	b.Free([]int64{4, 5, 6, 7})
	if got, ok := b.AllocDecode([]int{1, 1, 1, 1}, []int64{0, 0, 0, 0}); !ok || !slices.Equal(got, []int64{4, 8, 12, 16}) {
		t.Errorf("four requests on three free pages and one staged: %v, %v", got, ok)
	}
	if _, ok := NewPagedAllocator(8, 4, nil, false).AllocDecode([]int{1, 1, 1}, []int64{0, 0, 0}); ok {
		t.Error("three new pages out of two")
	}
	// Mid-page tokens follow their last slot; at page size 1 every token does.
	if got, ok := NewPagedAllocator(8, 4, nil, false).AllocDecode([]int{2, 1}, []int64{9, 0}); !ok || !slices.Equal(got, []int64{10, 4}) {
		t.Errorf("mid-page then page start: %v, %v", got, ok)
	}
	if got, ok := NewPagedAllocator(8, 1, nil, false).AllocDecode([]int{5, 6}, []int64{9, 3}); !ok || !slices.Equal(got, []int64{10, 4}) {
		t.Errorf("page size 1: %v, %v", got, ok)
	}
	mustPanic(t, "last slots of another length", func() {
		NewPagedAllocator(8, 4, nil, false).AllocDecode([]int{1}, []int64{0, 0})
	})
}

// The floored division and modulo and the ceiling division are Python's, whatever the signs.
func TestFloorAndCeilingArithmeticIsPythons(t *testing.T) {
	for _, c := range []struct{ a, b, mod, ceil int }{
		{7, 2, 1, 4}, {-7, 2, 1, -3}, {7, -2, -1, -3}, {-7, -2, -1, 4}, {8, 2, 0, 4}, {-8, 2, 0, -4}, {0, 3, 0, 0}, {1, 4, 1, 1}, {-1, 4, 3, 0},
	} {
		if got := floorMod(c.a, c.b); got != c.mod {
			t.Errorf("floorMod(%d, %d) = %d, want %d", c.a, c.b, got, c.mod)
		}
		wantDiv := int64((c.a - c.mod) / c.b)
		if got := floorDiv(int64(c.a), int64(c.b)); got != wantDiv {
			t.Errorf("floorDiv(%d, %d) = %d, want %d", c.a, c.b, got, wantDiv)
		}
		if got := ceilDiv(c.a, c.b); got != c.ceil {
			t.Errorf("ceilDiv(%d, %d) = %d, want %d", c.a, c.b, got, c.ceil)
		}
	}
}

// Free groups defer a pool's frees to the group's end, in order, and two freed segments may not
// share a page (paged.py:386-393, base.py:283-290).
func TestFreeGroupsAndSegmentsKeepTheirOrderAndPages(t *testing.T) {
	p := NewPagedAllocator(16, 4, nil, false) // pages 1..4
	slotsOf(t, p, 16)
	p.FreeGroupBegin()
	p.Free([]int64{8, 9})
	p.FreeSegment([]int64{12, 13, 14, 15}, 0)
	p.FreeGroupEnd()
	if got := slotsOf(t, p, 8); !slices.Equal(got, []int64{12, 13, 14, 15, 8, 9, 10, 11}) {
		t.Errorf("after a group: %v, want the segment's page then the slots' page, each ahead of the free list", got)
	}
	// An empty group frees nothing.
	p.FreeGroupBegin()
	p.FreeGroupEnd()
	if _, ok := p.Alloc(4); ok {
		t.Error("an empty group freed a page")
	}
	// Segments on pages 0 and 1 are disjoint; on one page they are refused.
	q := NewPagedAllocator(16, 4, nil, false)
	slotsOf(t, q, 16)
	q.FreeSegments([]Segment{{Slots: []int64{4, 5, 6, 7}, StartPos: 0}, {Slots: []int64{8, 9, 10, 11}, StartPos: 4}})
	mustPanic(t, "a segment starting on the page the previous one ends in", func() {
		q.FreeSegments([]Segment{{Slots: []int64{4, 5, 6, 7, 8, 9}, StartPos: 0}, {Slots: []int64{12, 13, 14, 15}, StartPos: 4}})
	})
	t1 := NewTokenAllocator(4, nil, false)
	slotsOf(t, t1, 4)
	t1.FreeGroupBegin()
	t1.Free([]int64{3})
	t1.Free([]int64{1})
	t1.FreeGroupEnd()
	if got := slotsOf(t, t1, 2); !slices.Equal(got, []int64{3, 1}) {
		t.Errorf("token group freed %v, want the frees in order", got)
	}
}

// A prefill budget has capacity only while both what remains in total and what remains now are
// positive (prefill_budget.py:62-168@297e54fd).
func TestPrefillBudgetNeedsBothRemaindersPositive(t *testing.T) {
	a := NewTokenAllocator(20, nil, false)
	b := a.CreatePrefillBudget(budgetTree{}, 0)
	if !b.HasCapacity() {
		t.Error("no capacity with 20 free")
	}
	b.Reserve(0, 19, 0) // one page now, 20 in total: nothing left in total, 19 now
	if b.RemainingTotal() != 0 || b.RemainingCurrent() != 19 || b.HasCapacity() {
		t.Errorf("total %d, current %d, capacity %v", b.RemainingTotal(), b.RemainingCurrent(), b.HasCapacity())
	}
}
