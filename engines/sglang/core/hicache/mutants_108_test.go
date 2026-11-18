// Original to blis-sglang.

package hicache

import (
	"slices"
	"testing"
)

// Behavioural tests of the host pool's sizing and free list (blis-sglang#108): each fails on one
// generated mutant of host.go that the golden traces do not reach.

func hostWithCapacity(t *testing.T, tokens, page int) *HostPool {
	t.Helper()
	p, err := NewHostPool(HostSizing{PageSize: page, HostTokenCapacity: &tokens})
	if err != nil {
		t.Fatalf("NewHostPool(capacity %d, page %d): %v", tokens, page, err)
	}
	return p
}

func TestHostPoolRefusesANonPositivePageSize(t *testing.T) {
	for _, page := range []int{0, -1} {
		tokens := 64
		for name, s := range map[string]HostSizing{
			"capacity": {PageSize: page, HostTokenCapacity: &tokens},
			"ratio":    {PageSize: page, DeviceCapacityTokens: 100, HostToDeviceRatio: 2},
		} {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("page size %d sized by %s: panicked instead of refusing: %v", page, name, r)
					}
				}()
				if p, err := NewHostPool(s); err == nil {
					t.Errorf("page size %d sized by %s: accepted (size %d)", page, name, p.Size())
				}
			}()
		}
	}
}

func TestHostPoolTakesAPageAlignedPositiveCapacityAsGiven(t *testing.T) {
	p := hostWithCapacity(t, 256, 64)
	if p.Size() != 256 || p.PageNum() != 4 || p.AvailableSize() != 256 {
		t.Fatalf("capacity 256, page 64: size %d, pages %d, available %d; want 256, 4, 256",
			p.Size(), p.PageNum(), p.AvailableSize())
	}
	p = hostWithCapacity(t, 64, 64)
	if p.Size() != 64 || p.PageNum() != 1 {
		t.Fatalf("capacity 64, page 64: size %d, pages %d; want 64, 1", p.Size(), p.PageNum())
	}
}

func TestHostPoolRefusesACapacityThatIsNotPositiveAndPageAligned(t *testing.T) {
	for _, c := range []int{0, -64, -100, 1, 100, 65} {
		c := c
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("capacity %d, page 64: panicked instead of refusing: %v", c, r)
				}
			}()
			if p, err := NewHostPool(HostSizing{PageSize: 64, HostTokenCapacity: &c}); err == nil {
				t.Errorf("capacity %d, page 64: accepted (size %d)", c, p.Size())
			}
		}()
	}
}

func TestHostPoolHandsOutEveryFreeSlotInOneAllocation(t *testing.T) {
	p := hostWithCapacity(t, 8, 1)
	if got, ok := p.Alloc(9); ok {
		t.Fatalf("Alloc(9) of 8 slots succeeded: %v", got)
	}
	got, ok := p.Alloc(8)
	if !ok || !slices.Equal(got, []int64{0, 1, 2, 3, 4, 5, 6, 7}) {
		t.Fatalf("Alloc(8) of 8 free slots = %v, %v; want all eight in order", got, ok)
	}
	if p.AvailableSize() != 0 {
		t.Fatalf("available after taking every slot = %d, want 0", p.AvailableSize())
	}
	// The same holds when the last slots come back as released ones.
	if _, err := p.Free(got[:3]); err != nil {
		t.Fatal(err)
	}
	got, ok = p.Alloc(3)
	if !ok || !slices.Equal(got, []int64{0, 1, 2}) {
		t.Fatalf("Alloc(3) with exactly 3 released slots = %v, %v; want [0 1 2]", got, ok)
	}
}

// A released run is added behind the slots that are still free, never in place of them
// (base.py:526-536).
func TestHostPoolKeepsFreeSlotsAheadOfOneReleasedRun(t *testing.T) {
	p := hostWithCapacity(t, 8, 1)
	if _, ok := p.Alloc(6); !ok { // slots 0..5; free: 6, 7
		t.Fatal("Alloc(6) failed")
	}
	if _, err := p.Free([]int64{0, 1, 2}); err != nil {
		t.Fatal(err)
	}
	got, ok := p.Alloc(4) // 2 free are too few: merge, then the free ones lead
	if !ok || !slices.Equal(got, []int64{6, 7, 0, 1}) {
		t.Fatalf("Alloc(4) = %v, %v; want [6 7 0 1]", got, ok)
	}
	got, ok = p.Alloc(1)
	if !ok || !slices.Equal(got, []int64{2}) || p.AvailableSize() != 0 {
		t.Fatalf("Alloc(1) = %v, %v, available %d; want [2], true, 0", got, ok, p.AvailableSize())
	}
}

// Several released runs are all kept, in the order they were freed, when the free list is empty.
func TestHostPoolKeepsEveryReleasedRunWhenNothingIsFree(t *testing.T) {
	p := hostWithCapacity(t, 8, 1)
	if _, ok := p.Alloc(8); !ok {
		t.Fatal("Alloc(8) failed")
	}
	for _, run := range [][]int64{{3, 4}, {1}, {6, 5}} {
		if _, err := p.Free(run); err != nil {
			t.Fatal(err)
		}
	}
	got, ok := p.Alloc(5)
	if !ok || !slices.Equal(got, []int64{3, 4, 1, 6, 5}) {
		t.Fatalf("Alloc(5) = %v, %v; want [3 4 1 6 5]", got, ok)
	}
}

func TestHostPoolHandsOutOneReleasedRunWhenNothingIsFree(t *testing.T) {
	p := hostWithCapacity(t, 8, 1)
	if _, ok := p.Alloc(8); !ok {
		t.Fatal("Alloc(8) failed")
	}
	if _, err := p.Free([]int64{5, 6}); err != nil {
		t.Fatal(err)
	}
	got, ok := p.Alloc(2)
	if !ok || !slices.Equal(got, []int64{5, 6}) {
		t.Fatalf("Alloc(2) = %v, %v; want [5 6]", got, ok)
	}
}
