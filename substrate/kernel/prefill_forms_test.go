package latencykernel

import (
	"math/rand"
	"testing"
)

// The block sums are exact: against a per-position loop over random ranges, ratios, caps and
// blocks.
func TestTheBlockSumsMatchAPerPositionLoop(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 2000; i++ {
		lo := 1 + r.Intn(5000)
		hi := lo + r.Intn(3000) - 50
		ratio := 1 + r.Intn(130)
		cap := r.Intn(700)
		block := []int{1, 64}[r.Intn(2)]
		var want, wantCeil float64
		for p := lo; p <= hi; p++ {
			m := p / ratio
			if cap > 0 {
				m = min(m, cap)
			}
			want += float64((m + block - 1) / block)
			wantCeil += float64((p + ratio - 1) / ratio)
		}
		if got := sumBlocks(lo, hi, ratio, cap, block); got != want {
			t.Fatalf("sumBlocks(%d, %d, %d, %d, %d) = %v, want %v", lo, hi, ratio, cap,
				block, got, want)
		}
		if got := sumCeil(lo, hi, ratio); hi >= lo && got != wantCeil {
			t.Fatalf("sumCeil(%d, %d, %d) = %v, want %v", lo, hi, ratio, got, wantCeil)
		}
	}
}
