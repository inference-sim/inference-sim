// Original to blis-sglang.

package cache

import "testing"

// budgetTree is a tree cache with fixed evictable figures, for a budget to read.
type budgetTree struct {
	PrefixCache
	evictable, fullEvictable int
	mamba                    bool
}

func (b budgetTree) EvictableSize() int     { return b.evictable }
func (b budgetTree) FullEvictableSize() int { return b.fullEvictable }
func (b budgetTree) SupportsMamba() bool    { return b.mamba }

// A prefill budget (prefill_budget.py:62-168@297e54fd) reads what is free and evictable live, and
// draws down a reservation of the extend in whole pages, one page and the expected decode tokens.
func TestPrefillBudgetDrawsDownReservations(t *testing.T) {
	a := NewPagedAllocator(64, 4, nil, false)
	b := a.CreatePrefillBudget(budgetTree{evictable: 8, fullEvictable: 100}, 2)
	if got := b.RemainingTotal(); got != 64+8-2 {
		t.Errorf("remaining total %d, want 70: 64 free, 8 evictable, 2 mixed decode tokens", got)
	}
	b.Reserve(5, 10, 1) // 5 tokens are 2 pages (8), plus a page (4) and 1 extra: 13 now, 23 in total
	if got, want := b.RemainingCurrent(), 70-13; got != want {
		t.Errorf("remaining current %d, want %d", got, want)
	}
	if got, want := b.RemainingTotal(), 70-23; got != want {
		t.Errorf("remaining total %d, want %d", got, want)
	}
	if !b.CanAllocatePrefill(47) || b.CanAllocatePrefill(48) {
		t.Error("a prefill is allocatable only up to the smaller remainder, 47")
	}
	// Slots taken from the allocator after the budget was made are seen at once.
	if _, ok := a.Alloc(48); !ok {
		t.Fatal("alloc")
	}
	if b.HasCapacity() {
		t.Errorf("capacity remains with %d total, %d current", b.RemainingTotal(), b.RemainingCurrent())
	}
	// A tree with Mamba state offers only its full-attention evictable tokens.
	m := a.CreatePrefillBudget(budgetTree{evictable: 8, fullEvictable: 3, mamba: true}, 0)
	if got := m.RemainingTotal(); got != 16+3 {
		t.Errorf("remaining total with Mamba %d, want 19", got)
	}
}
