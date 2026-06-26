package workload

import (
	"reflect"
	"testing"
)

func TestSessionTokenBufferAppendAndSlice(t *testing.T) {
	b := NewSessionTokenBuffer()
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
	b := NewSessionTokenBuffer()
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
	if int64(b.Cap()) > totalLen*3 {
		t.Fatalf("Cap = %d, want ≤ 3*Len = %d (linear growth bound)", b.Cap(), totalLen*3)
	}
}

func TestSessionTokenBufferEmptyAppend(t *testing.T) {
	b := NewSessionTokenBuffer()
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
	b := NewSessionTokenBuffer()
	b.Append([]int{10, 20, 30, 40, 50})
	a := b.Slice(0, 3)
	c := b.Slice(0, 3)
	if len(a) > 0 && &a[0] != &c[0] {
		t.Fatalf("Slice(0,3) returned distinct backing arrays — buffer is copying, not aliasing")
	}
}
