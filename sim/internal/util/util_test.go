package util

import "testing"

func TestLen64_IntSlice(t *testing.T) {
	got := Len64([]int{1, 2, 3})
	if got != 3 {
		t.Errorf("Len64([]int{1,2,3}) = %d, want 3", got)
	}
}

func TestLen64_EmptySlice(t *testing.T) {
	got := Len64([]int{})
	if got != 0 {
		t.Errorf("Len64([]int{}) = %d, want 0", got)
	}
}
