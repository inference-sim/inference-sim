package latency

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestActivatedExpertFraction_BoundaryCases pins the three properties #1849 names as
// the contract of the coupon-collector fraction, from first principles rather than
// from captured output: B=1 ⇒ k/N, B large ⇒ → 1, and dense/degenerate inputs are
// inert.
func TestActivatedExpertFraction_BoundaryCases(t *testing.T) {
	tests := []struct {
		name       string
		numExperts int
		kEff       int
		tokens     float64
		want       float64
		tol        float64
	}{
		// B = 1: a single token routes to exactly k of the N experts, so the expected
		// activated fraction is k/N with no collision correction at all.
		{"B=1 Mixtral 8x7B (N=8,k=2)", 8, 2, 1, 2.0 / 8.0, 1e-12},
		{"B=1 DeepSeek-scale router (N=256,k=8)", 256, 8, 1, 8.0 / 256.0, 1e-12},
		{"B=1 N=128 k=1", 128, 1, 1, 1.0 / 128.0, 1e-12},

		// B = 2 for N=8,k=2 by hand: 1 − (6/8)² = 1 − 0.5625 = 0.4375. Two tokens
		// activate 3.5 of 8 experts on average (4 routed slots, 0.5 expected collision).
		{"B=2 N=8 k=2 hand-computed", 8, 2, 2, 0.4375, 1e-12},

		// B large: saturation — every resident expert is streamed, recovering the
		// pre-#1849 batch-independent behaviour exactly.
		{"B=1000 N=8 k=2 saturates", 8, 2, 1000, 1.0, 1e-9},
		{"B=100000 N=256 k=8 saturates", 256, 8, 100000, 1.0, 1e-9},

		// Degenerate inputs.
		{"zero tokens activates nothing", 8, 2, 0, 0, 0},
		{"negative tokens activates nothing", 8, 2, -3, 0, 0},
		{"dense model (no routed experts) is inert", 0, 1, 64, 1, 0},
		{"k == N routes to everything", 8, 8, 1, 1, 0},
		{"k > N (malformed config) clamps, never NaN", 8, 16, 1, 1, 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := activatedExpertFraction(tc.numExperts, tc.kEff, tc.tokens)
			require.False(t, math.IsNaN(got), "fraction must never be NaN")
			assert.InDelta(t, tc.want, got, tc.tol)
		})
	}
}

// TestActivatedExpertFraction_MonotoneAndBounded asserts the two laws the weight term
// relies on across the whole batch range, for several router shapes: the fraction is
// non-decreasing in B (so decode step time rises with the running batch) and stays in
// [0, 1] (so it can only ever scale the resident count DOWN — the #1419/#1548
// shard-group ceiling is never exceeded).
func TestActivatedExpertFraction_MonotoneAndBounded(t *testing.T) {
	routers := []struct {
		name string
		n, k int
	}{
		{"Mixtral 8x7B", 8, 2},
		{"Qwen3-235B-style", 128, 8},
		{"DeepSeek-V3-scale", 256, 8},
		{"single-expert routing", 64, 1},
	}

	for _, r := range routers {
		t.Run(r.name, func(t *testing.T) {
			prev := 0.0
			for b := 1; b <= 4096; b *= 2 {
				got := activatedExpertFraction(r.n, r.k, float64(b))
				assert.GreaterOrEqual(t, got, prev, "fraction must not decrease as B grows (B=%d)", b)
				assert.LessOrEqual(t, got, 1.0, "fraction must never exceed 1 (B=%d)", b)
				assert.Greater(t, got, 0.0, "any token activates some expert (B=%d)", b)
				prev = got
			}
		})
	}
}

// TestActivatedExpertFraction_LinearRegime checks the small-batch asymptotic the
// derivation claims: while B·k ≪ N, collisions are negligible and the expected
// distinct-expert count N·fraction is ≈ B·k. This is the regime where the pre-#1849
// batch-independent term over-charged by ~N/k.
func TestActivatedExpertFraction_LinearRegime(t *testing.T) {
	const n, k = 256, 8
	for _, b := range []float64{1, 2, 3} {
		nEff := float64(n) * activatedExpertFraction(n, k, b)
		// Collision-free would be exactly b·k; with B·k ≪ N the shortfall is a few %.
		assert.InDelta(t, b*k, nEff, 0.06*b*k, "B=%v should be near-linear in B·k", b)
		assert.LessOrEqual(t, nEff, b*k, "collisions can only reduce the distinct count")
	}
}
