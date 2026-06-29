package workload

import (
	"reflect"
	"testing"
)

func TestSessionTokenBufferAppendAndSlice(t *testing.T) {
	b := newSessionTokenBuffer()
	if b.Len() != 0 {
		t.Fatalf("empty Len = %d, want 0", b.Len())
	}
	s1, e1 := b.Append([]int{1, 2, 3})
	if s1 != 0 || e1 != 3 {
		t.Fatalf("Append #1 range = (%d,%d), want (0,3)", s1, e1)
	}
	s2, e2 := b.Append([]int{4, 5})
	if s2 != 3 || e2 != 5 {
		t.Fatalf("Append #2 range = (%d,%d), want (3,5)", s2, e2)
	}
	got := b.Slice(0, 5)
	want := []int{1, 2, 3, 4, 5}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Slice(0,5) = %v, want %v", got, want)
	}
	if got := b.Slice(2, 4); !reflect.DeepEqual(got, []int{3, 4}) {
		t.Fatalf("Slice(2,4) = %v, want [3 4]", got)
	}
}

func TestSessionTokenBufferLinearGrowth(t *testing.T) {
	// Append N rounds, each of size K. Total appended = N*K. Verify cap stays
	// within a small constant factor of total bytes (Go's append doubles
	// capacity, so cap ≤ 2*Len after the last grow; we allow ≤3*Len slack).
	b := newSessionTokenBuffer()
	const rounds = 20
	const perRound = 100
	for i := 0; i < rounds; i++ {
		chunk := make([]int, perRound)
		for j := range chunk {
			chunk[j] = i*perRound + j
		}
		b.Append(chunk)
	}
	totalLen := int64(rounds * perRound)
	if b.Len() != totalLen {
		t.Fatalf("Len after %d rounds = %d, want %d", rounds, b.Len(), totalLen)
	}
	if b.bufCap() > totalLen*3 {
		t.Fatalf("bufCap = %d, want ≤ 3*Len = %d (linear growth bound)", b.bufCap(), totalLen*3)
	}
}

func TestSessionTokenBufferEmptyAppend(t *testing.T) {
	b := newSessionTokenBuffer()
	s, e := b.Append(nil)
	if s != 0 || e != 0 {
		t.Fatalf("Append(nil) range = (%d,%d), want (0,0)", s, e)
	}
	s, e = b.Append([]int{})
	if s != 0 || e != 0 {
		t.Fatalf("Append([]int{}) range = (%d,%d), want (0,0)", s, e)
	}
}

func TestSessionTokenBufferSliceIsView(t *testing.T) {
	// Two slices spanning the same range MUST point at the same backing array —
	// that's the storage-win guarantee.
	b := newSessionTokenBuffer()
	b.Append([]int{10, 20, 30, 40, 50})
	a := b.Slice(0, 3)
	c := b.Slice(0, 3)
	if len(a) > 0 && &a[0] != &c[0] {
		t.Fatalf("Slice(0,3) returned distinct backing arrays — buffer is copying, not aliasing")
	}
}

// TestSessionTokenBufferSliceAliasesAcrossAppends asserts the load-bearing
// invariant for the O(R) storage win: slices returned by Slice() before a
// subsequent Append() remain aliases of the same backing array, AS LONG AS the
// buffer has capacity for the append (no reallocation). We use
// newSessionTokenBufferWithCapacity to control this.
func TestSessionTokenBufferSliceAliasesAcrossAppends(t *testing.T) {
	b := newSessionTokenBufferWithCapacity(100)
	b.Append([]int{1, 2, 3})
	first := b.Slice(0, 3)
	b.Append([]int{4, 5, 6})
	second := b.Slice(0, 6)
	if len(first) == 0 || len(second) == 0 {
		t.Fatal("empty slice — test invalid")
	}
	if &first[0] != &second[0] {
		t.Fatalf("Slice() returned distinct backing arrays after Append — pre-allocated capacity should prevent realloc")
	}
}

// TestNewSessionTokenBufferWithCapacityRejectsNegative verifies R3 validation:
// a negative capacity is a caller calculation bug and must panic, not be
// silently clamped (NEW-MOD-4, #1445).
func TestNewSessionTokenBufferWithCapacityRejectsNegative(t *testing.T) {
	defer func() {
		if rec := recover(); rec == nil {
			t.Fatalf("expected panic for negative capacity, got none")
		}
	}()
	_ = newSessionTokenBufferWithCapacity(-1)
}

// TestSessionTokenBufferSliceValidBoundaries asserts that valid degenerate
// ranges (empty range at the end of the buffer; empty range at the start
// of a non-empty buffer) return empty slices without panic (MIN-R5-2, #1445).
func TestSessionTokenBufferSliceValidBoundaries(t *testing.T) {
	b := newSessionTokenBuffer()
	b.Append([]int{1, 2, 3})
	// Empty range at the current end (start == end == buf.Len()).
	got := b.Slice(3, 3)
	if len(got) != 0 {
		t.Fatalf("Slice(3, 3) returned len=%d, want 0", len(got))
	}
	// Empty range at the start of a non-empty buffer.
	got = b.Slice(0, 0)
	if len(got) != 0 {
		t.Fatalf("Slice(0, 0) returned len=%d, want 0", len(got))
	}
}

// TestSessionTokenBufferSliceBounds asserts that out-of-bounds Slice() calls
// panic with a decorated message (IMP-1, #1445).
func TestSessionTokenBufferSliceBounds(t *testing.T) {
	b := newSessionTokenBuffer()
	b.Append([]int{1, 2, 3})
	cases := []struct {
		name       string
		start, end int64
	}{
		{"end > len", 0, 5},
		{"start > end", 2, 1},
		{"negative start", -1, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if rec := recover(); rec == nil {
					t.Fatalf("expected panic for Slice(%d, %d)", tc.start, tc.end)
				}
			}()
			_ = b.Slice(tc.start, tc.end)
		})
	}
}
