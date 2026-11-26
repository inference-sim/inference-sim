
package latencykernel

import (
	"math"
	"math/rand"
	"testing"
)

// Two balls over two bins end 2-0 or 0-2 with probability 1/2 and 1-1 with probability 1/2, so the
// busiest bin holds 1.5 on average against a mean of 1.
func TestBallsInBinsSmallestCaseByHand(t *testing.T) {
	if got := ballsInBins(2, 2); math.Abs(got-1.5) > 1e-12 {
		t.Errorf("ballsInBins(2, 2) = %v, want 1.5", got)
	}
	// One rank, or no ball, is no imbalance.
	if ballsInBins(10, 1) != 1 || ballsInBins(0, 8) != 1 {
		t.Errorf("one rank or no balls must price as even")
	}
}

// The expectation agrees with simulation of the multinomial on both sides of the exact/normal switch, falls
// toward 1 as the step grows, and never leaves [1, r].
func TestBallsInBinsMatchesSimulationAndShrinksWithTheStep(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, c := range []struct{ n, r int }{{16, 8}, {64, 8}, {128, 8}, {129, 8}, {512, 8}, {4096, 16}} {
		const trials = 4000
		sum := 0.0
		for i := 0; i < trials; i++ {
			bins := make([]int, c.r)
			for j := 0; j < c.n; j++ {
				bins[rng.Intn(c.r)]++
			}
			m := 0
			for _, v := range bins {
				m = max(m, v)
			}
			sum += float64(m)
		}
		sim := sum / trials / (float64(c.n) / float64(c.r))
		if got := ballsInBins(c.n, c.r); math.Abs(got-sim) > 0.03*sim {
			t.Errorf("n=%d r=%d: expectation %.4f, simulation %.4f", c.n, c.r, got, sim)
		}
	}
	prev := math.Inf(1)
	for _, n := range []int{8, 32, 128, 129, 512, 8192, 65536} {
		f := ballsInBins(n, 8)
		if f < 1 || f > 8 || f > prev+1e-9 {
			t.Errorf("n=%d: factor %v not within [1, 8] and non-increasing (previous %v)", n, f, prev)
		}
		prev = f
	}
}
