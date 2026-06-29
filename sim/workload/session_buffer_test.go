package workload

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestSessionTokenBufferAppendAndSlice(t *testing.T) {
	b := newSessionTokenBuffer()
	if b.Len() != 0 {
		t.Fatalf("empty Len = %d, want 0", b.Len())
	}
	s1, e1 := b.Append([]sim.TokenID{1, 2, 3})
	if s1 != 0 || e1 != 3 {
		t.Fatalf("Append #1 range = (%d,%d), want (0,3)", s1, e1)
	}
	s2, e2 := b.Append([]sim.TokenID{4, 5})
	if s2 != 3 || e2 != 5 {
		t.Fatalf("Append #2 range = (%d,%d), want (3,5)", s2, e2)
	}
	got := b.Slice(0, 5)
	want := []sim.TokenID{1, 2, 3, 4, 5}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Slice(0,5) = %v, want %v", got, want)
	}
	if got := b.Slice(2, 4); !reflect.DeepEqual(got, []sim.TokenID{3, 4}) {
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
		chunk := make([]sim.TokenID, perRound)
		for j := range chunk {
			chunk[j] = sim.TokenID(i*perRound + j)
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
	s, e = b.Append([]sim.TokenID{})
	if s != 0 || e != 0 {
		t.Fatalf("Append([]sim.TokenID{}) range = (%d,%d), want (0,0)", s, e)
	}
}

func TestSessionTokenBufferSliceIsView(t *testing.T) {
	// Two slices spanning the same range MUST point at the same backing array —
	// that's the storage-win guarantee.
	b := newSessionTokenBuffer()
	b.Append([]sim.TokenID{10, 20, 30, 40, 50})
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
	b.Append([]sim.TokenID{1, 2, 3})
	first := b.Slice(0, 3)
	b.Append([]sim.TokenID{4, 5, 6})
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

// TestSessionTokenBufferForcedReallocOrphanContentCorrect documents the
// reallocation hazard described on sessionTokenBuffer: when Append exceeds
// capacity, Go reallocates the backing array and previously-returned
// Slice() results become orphaned views of the OLD array. The orphaned
// slices must remain CONTENT-CORRECT — the data they reference is
// unchanged, even though further appends are no longer visible through
// them. This locks in the "intentional, bounded" hazard documented in
// session_buffer.go (susiejojo human review, #1445).
func TestSessionTokenBufferForcedReallocOrphanContentCorrect(t *testing.T) {
	// Start with cap=4. The first Slice() returns a view of [1,2,3,4].
	b := newSessionTokenBufferWithCapacity(4)
	b.Append([]sim.TokenID{1, 2, 3, 4})
	orphanCandidate := b.Slice(0, 4)
	orphanFirstAddr := &orphanCandidate[0]

	// Force reallocation: append more than the remaining capacity. After this,
	// the buffer's backing array has changed.
	b.Append([]sim.TokenID{5, 6, 7, 8, 9, 10})
	newView := b.Slice(0, 10)

	// New view points to a different backing array (reallocation happened).
	if len(newView) > 0 && &newView[0] == orphanFirstAddr {
		t.Fatalf("expected reallocation after exceeding capacity, but new slice still aliases orphan")
	}

	// Orphan slice remains content-correct: still spans [1,2,3,4].
	want := []sim.TokenID{1, 2, 3, 4}
	if !reflect.DeepEqual(orphanCandidate, want) {
		t.Fatalf("orphan slice content drifted: got %v, want %v", orphanCandidate, want)
	}

	// New view also content-correct.
	wantNew := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if !reflect.DeepEqual(newView, wantNew) {
		t.Fatalf("new slice content: got %v, want %v", newView, wantNew)
	}
}

// TestRequestInputTokenSliceCappedCapacity verifies the three-index slice
// guarantee at the user-facing accessor layer: appending to a slice returned
// by Request.InputTokenSlice() allocates a fresh backing array instead of
// overwriting the buffer's contents. The buffer's own internal Slice() keeps
// full capacity for memory measurement; the user-facing accessor caps it
// (susiejojo human review, #1445).
func TestRequestInputTokenSliceCappedCapacity(t *testing.T) {
	b := newSessionTokenBufferWithCapacity(100)
	b.Append([]sim.TokenID{10, 20, 30})
	// Mimic how reasoning.go/session.go wire the buffer view onto a Request.
	req := &sim.Request{ID: "req1", InputTokens: b.Slice(0, 3)}

	view := req.InputTokenSlice(0, 3)
	if cap(view) != 3 {
		t.Fatalf("Request.InputTokenSlice cap = %d, want 3 (three-index slice should cap at len)", cap(view))
	}
	full := req.FullInputTokens()
	if cap(full) != 3 {
		t.Fatalf("Request.FullInputTokens cap = %d, want 3", cap(full))
	}
	// Append to the user-facing view — must reallocate, not overwrite the buffer.
	_ = append(view, sim.TokenID(999))
	// Appending to the buffer via the proper API must yield 50, not 999.
	b.Append([]sim.TokenID{50})
	if got := b.Slice(3, 4)[0]; got != 50 {
		t.Fatalf("after legitimate Append: buf[3] = %d, want 50 (three-index Request accessor should have prevented overwrite)", got)
	}
}

// TestSessionTokenBufferSliceValidBoundaries asserts that valid degenerate
// ranges (empty range at the end of the buffer; empty range at the start
// of a non-empty buffer) return empty slices without panic (MIN-R5-2, #1445).
func TestSessionTokenBufferSliceValidBoundaries(t *testing.T) {
	b := newSessionTokenBuffer()
	b.Append([]sim.TokenID{1, 2, 3})
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
	b.Append([]sim.TokenID{1, 2, 3})
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
