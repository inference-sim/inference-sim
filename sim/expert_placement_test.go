package sim

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// floatEq compares two MoE-load floats with a tolerance loose enough to absorb
// IEEE-754 rounding in the (moeGroupSize-1)/moeGroupSize factor but tight enough
// to catch any genuine formula error.
const floatEqTol = 1e-9

// TestBalancedPlacement_Resolve verifies BalancedPlacement.Resolve against
// hand-computed ExpertLoad for representative (globalTokens, kEff, numExperts,
// moeGroupSize, dp) tuples — including the degenerate moeGroupSize == 1 — per
// design §4 / issue #1418.
//
// The balanced (imbalanceFactor == 1) formulas under test:
//
//	PerGPUComputeTokens = globalTokens · kEff / moeGroupSize
//	PerGPUExpertCount   = numExperts / moeGroupSize
//	PerGPUCommTokens    = (globalTokens / dp) · kEff · (moeGroupSize-1)/moeGroupSize · 2
func TestBalancedPlacement_Resolve(t *testing.T) {
	tests := []struct {
		name         string
		globalTokens float64
		kEff         float64
		numExperts   int
		moeGroupSize int
		dp           int
		wantCompute  float64
		wantExperts  float64
		wantComm     float64
	}{
		{
			// EP-off-style: TP·DP = 4, dp = 2, top-2 routing.
			name: "group4_dp2", globalTokens: 1000, kEff: 2, numExperts: 8, moeGroupSize: 4, dp: 2,
			wantCompute: 500,  // 1000·2/4
			wantExperts: 2,    // 8/4
			wantComm:    1500, // (1000/2)·2·(3/4)·2
		},
		{
			// Degenerate single-GPU MoE group: per-GPU == global, no all-to-all.
			name: "group1_dp1_degenerate", globalTokens: 512, kEff: 4, numExperts: 16, moeGroupSize: 1, dp: 1,
			wantCompute: 2048, // 512·4/1
			wantExperts: 16,   // 16/1
			wantComm:    0,    // (512/1)·4·(0/1)·2
		},
		{
			// Pure TP MoE group (dp == 1), top-1 routing.
			name: "group8_dp1", globalTokens: 2048, kEff: 1, numExperts: 64, moeGroupSize: 8, dp: 1,
			wantCompute: 256,  // 2048·1/8
			wantExperts: 8,    // 64/8
			wantComm:    3584, // (2048/1)·1·(7/8)·2
		},
		{
			// dp = 4 splits comm volume per source rank; compute/experts unaffected by dp.
			name: "group8_dp4", globalTokens: 4000, kEff: 2, numExperts: 128, moeGroupSize: 8, dp: 4,
			wantCompute: 1000, // 4000·2/8
			wantExperts: 16,   // 128/8
			wantComm:    3500, // (4000/4)·2·(7/8)·2
		},
		{
			// Zero routed tokens (empty step) — all loads vanish.
			name: "zero_tokens", globalTokens: 0, kEff: 2, numExperts: 8, moeGroupSize: 4, dp: 2,
			wantCompute: 0,
			wantExperts: 2, // expert weight residency is token-independent
			wantComm:    0,
		},
	}

	p := BalancedPlacement{}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := p.Resolve(tc.globalTokens, tc.kEff, tc.numExperts, tc.moeGroupSize, tc.dp)
			assert.InDelta(t, tc.wantCompute, got.PerGPUComputeTokens, floatEqTol, "PerGPUComputeTokens")
			assert.InDelta(t, tc.wantExperts, got.PerGPUExpertCount, floatEqTol, "PerGPUExpertCount")
			assert.InDelta(t, tc.wantComm, got.PerGPUCommTokens, floatEqTol, "PerGPUCommTokens")
		})
	}
}

// TestBalancedPlacement_ComputeConservationLaw asserts the routed-compute
// conservation law from issue #1418: summing the busiest-GPU compute over the
// whole MoE group recovers the global routed-token·activation count. Under the
// balanced strategy every GPU carries an equal share, so
// PerGPUComputeTokens · moeGroupSize == globalTokens · kEff exactly.
//
// This is the companion invariant test to the golden table above (CLAUDE.md
// BDD/TDD rule 4): it validates the formula from first principles rather than
// re-encoding the same arithmetic.
func TestBalancedPlacement_ComputeConservationLaw(t *testing.T) {
	cases := []struct {
		globalTokens float64
		kEff         float64
		numExperts   int
		moeGroupSize int
		dp           int
	}{
		{1000, 2, 8, 4, 2},
		{512, 4, 16, 1, 1},
		{2048, 1, 64, 8, 1},
		{4000, 2, 128, 8, 4},
		{333, 3, 7, 5, 5},
	}

	p := BalancedPlacement{}
	for _, c := range cases {
		got := p.Resolve(c.globalTokens, c.kEff, c.numExperts, c.moeGroupSize, c.dp)
		want := c.globalTokens * c.kEff
		assert.InDelta(t, want, got.PerGPUComputeTokens*float64(c.moeGroupSize), 1e-6,
			"routed compute must be conserved across the MoE group")
	}
}

// TestBalancedPlacement_CommConservationLaw is the comm-term companion to the
// compute conservation law (CLAUDE.md BDD/TDD rule 4): it validates the dp and
// moeGroupSize scaling of PerGPUCommTokens from first principles rather than
// re-encoding the golden-table arithmetic.
//
// Multiplying the busiest-GPU per-rank comm volume back up by the group size and
// the dp degree must recover the dp/group-independent total dispatch+combine
// volume:
//
//	PerGPUCommTokens · moeGroupSize · dp == globalTokens · kEff · (moeGroupSize-1) · 2
//
// The right-hand side has no dp or 1/moeGroupSize dependence, so this independently
// pins both the /dp per-rank split and the (group-1)/group all-to-all factor.
func TestBalancedPlacement_CommConservationLaw(t *testing.T) {
	cases := []struct {
		globalTokens float64
		kEff         float64
		numExperts   int
		moeGroupSize int
		dp           int
	}{
		{1000, 2, 8, 4, 2},
		{512, 4, 16, 1, 1}, // degenerate: both sides collapse to 0
		{2048, 1, 64, 8, 1},
		{4000, 2, 128, 8, 4},
		{333, 3, 7, 5, 5},
	}

	p := BalancedPlacement{}
	for _, c := range cases {
		got := p.Resolve(c.globalTokens, c.kEff, c.numExperts, c.moeGroupSize, c.dp)
		want := c.globalTokens * c.kEff * float64(c.moeGroupSize-1) * 2
		assert.InDelta(t, want, got.PerGPUCommTokens*float64(c.moeGroupSize)*float64(c.dp), 1e-6,
			"dispatch+combine comm volume must be conserved across the MoE group and dp ranks")
	}
}

// TestBalancedPlacement_Resolve_FractionalKEff exercises a fractional kEff, which
// the interface explicitly admits ("possibly fractional" — e.g. a shared+routed
// mix or an averaged top-k). The loads must scale linearly in kEff with no
// integer truncation.
func TestBalancedPlacement_Resolve_FractionalKEff(t *testing.T) {
	p := BalancedPlacement{}
	// globalTokens=1000, kEff=2.5, group=4, dp=2.
	got := p.Resolve(1000, 2.5, 8, 4, 2)
	assert.InDelta(t, 625, got.PerGPUComputeTokens, floatEqTol, "1000·2.5/4")
	assert.InDelta(t, 2, got.PerGPUExpertCount, floatEqTol, "8/4 (kEff-independent)")
	assert.InDelta(t, 1875, got.PerGPUCommTokens, floatEqTol, "(1000/2)·2.5·(3/4)·2")
}

// TestBalancedPlacement_Resolve_RejectsDegenerateDivisors verifies the library
// boundary guard (R1): a divisor < 1 would otherwise silently emit +Inf/NaN loads
// into downstream step-time math. Production callers never hit this (the config
// EffectiveMoEGroupSize / EffectiveDP helpers clamp to >= 1), but a direct caller
// passing 0 must fail loudly.
func TestBalancedPlacement_Resolve_RejectsDegenerateDivisors(t *testing.T) {
	p := BalancedPlacement{}
	assert.Panics(t, func() { p.Resolve(1000, 2, 8, 0, 1) }, "moeGroupSize=0 must panic")
	assert.Panics(t, func() { p.Resolve(1000, 2, 8, 4, 0) }, "dp=0 must panic")
	assert.Panics(t, func() { p.Resolve(1000, 2, 8, -1, 1) }, "negative moeGroupSize must panic")
	assert.NotPanics(t, func() { p.Resolve(1000, 2, 8, 1, 1) }, "minimal valid divisors must not panic")
}

// TestBalancedPlacement_NonNegativeAndFinite verifies that for any physically
// meaningful input (group >= 1, dp >= 1, tokens/kEff >= 0) every returned load
// is non-negative and finite — guarding against a sign error or a divide-by-zero
// leaking NaN/Inf into downstream step-time math.
func TestBalancedPlacement_NonNegativeAndFinite(t *testing.T) {
	p := BalancedPlacement{}
	for _, group := range []int{1, 2, 4, 8, 16} {
		for _, dp := range []int{1, 2, 4} {
			got := p.Resolve(1234.5, 2, 32, group, dp)
			for name, v := range map[string]float64{
				"PerGPUComputeTokens": got.PerGPUComputeTokens,
				"PerGPUExpertCount":   got.PerGPUExpertCount,
				"PerGPUCommTokens":    got.PerGPUCommTokens,
			} {
				assert.GreaterOrEqual(t, v, 0.0, "%s must be non-negative (group=%d dp=%d)", name, group, dp)
				assert.False(t, math.IsNaN(v) || math.IsInf(v, 0), "%s must be finite (group=%d dp=%d)", name, group, dp)
			}
		}
	}
}

// TestBalancedPlacement_SatisfiesInterface is a compile-time assertion that
// BalancedPlacement implements the ExpertPlacement contract. It is a behavioral
// guard, not a structural one: if the interface signature drifts, this fails to
// compile, flagging the consumer-contract break (#C wires this field).
func TestBalancedPlacement_SatisfiesInterface(t *testing.T) {
	var _ ExpertPlacement = BalancedPlacement{}
}
