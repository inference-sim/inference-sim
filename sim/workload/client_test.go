package workload

import (
	"math"
	"testing"
)

// floatsClose is a small helper used by the rate-normalization tests.
func floatsClose(a, b, eps float64) bool { return math.Abs(a-b) <= eps }

// TestNormalizeRateFractions_SingleUnboundedGroup preserves the historical
// behaviour: when every client is active for the full run, the aggregate
// rate is split proportionally by RateFraction.
func TestNormalizeRateFractions_SingleUnboundedGroup(t *testing.T) {
	clients := []ClientSpec{
		{ID: "a", RateFraction: 0.7},
		{ID: "b", RateFraction: 0.3},
	}
	// aggregateRate = 40 req/s => total should be 40 * 1e-6 req/µs.
	rates := normalizeRateFractions(clients, 40)
	want := []float64{28.0 / 1e6, 12.0 / 1e6}
	for i := range clients {
		if !floatsClose(rates[i], want[i], 1e-12) {
			t.Errorf("client %s: rate = %g, want %g", clients[i].ID, rates[i], want[i])
		}
	}
}

// TestNormalizeRateFractions_NonOverlappingPhases is the scenario from
// issue #1144. Phase 1 (clients A+B, windows 0-50s, fractions 0.7+0.3)
// and Phase 2 (client C alone, 50-100s, fraction 1.0) do not overlap —
// the peak-overlap sum is 1.0 and the per-phase rates must total
// aggregateRate, not aggregateRate/2.
func TestNormalizeRateFractions_NonOverlappingPhases(t *testing.T) {
	clients := []ClientSpec{
		{
			ID:           "phase1-A",
			RateFraction: 0.7,
			Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
				{StartUs: 0, EndUs: 50_000_000},
			}},
		},
		{
			ID:           "phase1-B",
			RateFraction: 0.3,
			Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
				{StartUs: 0, EndUs: 50_000_000},
			}},
		},
		{
			ID:           "phase2",
			RateFraction: 1.0,
			Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
				{StartUs: 50_000_000, EndUs: 100_000_000},
			}},
		},
	}
	rates := normalizeRateFractions(clients, 40)
	want := []float64{28.0 / 1e6, 12.0 / 1e6, 40.0 / 1e6}
	for i := range clients {
		if !floatsClose(rates[i], want[i], 1e-12) {
			t.Errorf("client %s: rate = %g, want %g", clients[i].ID, rates[i], want[i])
		}
	}
}

// TestNormalizeRateFractions_OverlappingWindows keeps the old normalization
// when lifecycle windows do overlap: the concurrent sum of active
// fractions must not exceed aggregateRate, so the divisor is the peak.
func TestNormalizeRateFractions_OverlappingWindows(t *testing.T) {
	clients := []ClientSpec{
		{
			ID:           "a",
			RateFraction: 0.5,
			Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
				{StartUs: 0, EndUs: 80_000_000},
			}},
		},
		{
			ID:           "b",
			RateFraction: 0.5,
			Lifecycle: &LifecycleSpec{Windows: []ActiveWindow{
				{StartUs: 20_000_000, EndUs: 100_000_000},
			}},
		},
	}
	// Peak overlap (both active 20-80s) sums to 1.0, so each client
	// should run at aggregateRate/2 when overlapping.
	rates := normalizeRateFractions(clients, 40)
	want := []float64{20.0 / 1e6, 20.0 / 1e6}
	for i := range clients {
		if !floatsClose(rates[i], want[i], 1e-12) {
			t.Errorf("client %s: rate = %g, want %g", clients[i].ID, rates[i], want[i])
		}
	}
}

// TestNormalizeRateFractions_ZeroAggregate yields zero rates without panicking.
func TestNormalizeRateFractions_ZeroAggregate(t *testing.T) {
	clients := []ClientSpec{
		{ID: "a", RateFraction: 0},
		{ID: "b", RateFraction: 0},
	}
	rates := normalizeRateFractions(clients, 40)
	if len(rates) != 2 || rates[0] != 0 || rates[1] != 0 {
		t.Errorf("expected [0 0], got %v", rates)
	}
}
