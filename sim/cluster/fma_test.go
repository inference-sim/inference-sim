package cluster

import (
	"fmt"
	"testing"
)

// X6: standard output is byte-identical on linux/amd64, linux/arm64 and darwin/arm64, and the
// Go compiler fuses x*y + z into one rounding on arm64 only. Fairness is summed over products,
// so it is written with each product rounded first; this holds the result to that reference on
// every platform (inference-sim#2078). On amd64 the test cannot fail; on arm64 it fails if a
// product is fused into its sum.
func TestJainFairnessIndexRoundsEachProductBeforeTheSum(t *testing.T) {
	seed := uint64(88172645463325252)
	next := func() float64 {
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		return 0.1 + float64(seed%1_000_003)/997.0
	}
	for trial := 0; trial < 200; trial++ {
		th := map[string]float64{}
		var xs []float64
		for i := 0; i < 7; i++ {
			x := next()
			th[fmt.Sprintf("k%d", i)] = x
			xs = append(xs, x)
		}
		// The function sums in key order, which is k0..k6 here.
		var sumX, sumX2 float64
		for _, x := range xs {
			sumX += x
			sumX2 += float64(x * x)
		}
		want := float64(sumX*sumX) / float64(7*sumX2)
		if got := JainFairnessIndex(th); got != want {
			t.Fatalf("trial %d: JainFairnessIndex = %.17g, want %.17g (each product rounded before the sum)", trial, got, want)
		}
	}
}
