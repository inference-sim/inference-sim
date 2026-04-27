package workload

import (
	"math"
	"testing"
)

func TestNormalizeRateFractions(t *testing.T) {
	const aggregateRate = 100.0 // 100 req/s
	const toReqPerUs = 1e6     // multiply rate (req/µs) to get req/s

	tests := []struct {
		name    string
		clients []ClientSpec
		// wantRates are expected rates in req/s (we multiply the returned req/µs by 1e6)
		wantRates []float64
	}{
		{
			name: "BC-4: no lifecycle windows — global normalization",
			clients: []ClientSpec{
				{ID: "a", RateFraction: 0.7},
				{ID: "b", RateFraction: 0.3},
			},
			wantRates: []float64{70.0, 30.0}, // 100 * 0.7, 100 * 0.3
		},
		{
			name: "BC-4: global normalization with non-unit sum",
			clients: []ClientSpec{
				{ID: "a", RateFraction: 7.0},
				{ID: "b", RateFraction: 3.0},
			},
			wantRates: []float64{70.0, 30.0}, // normalized: 7/10=0.7, 3/10=0.3
		},
		{
			name: "BC-1: non-overlapping phases — each phase at aggregate_rate",
			clients: []ClientSpec{
				{ID: "p1a", RateFraction: 0.7, Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{{StartUs: 0, EndUs: 50_000_000}},
				}},
				{ID: "p1b", RateFraction: 0.3, Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{{StartUs: 0, EndUs: 50_000_000}},
				}},
				{ID: "p2", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{{StartUs: 50_000_000, EndUs: 100_000_000}},
				}},
			},
			// Phase 1 coActiveSum = 0.7+0.3 = 1.0 → rates unchanged
			// Phase 2 coActiveSum = 1.0 → rate unchanged
			wantRates: []float64{70.0, 30.0, 100.0},
		},
		{
			name: "BC-2: sub-1.0 fraction solo phase — normalized to full rate",
			clients: []ClientSpec{
				{ID: "solo", RateFraction: 0.7, Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{{StartUs: 0, EndUs: 50_000_000}},
				}},
			},
			// coActiveSum = 0.7, rate = 100 * 0.7/0.7 = 100
			wantRates: []float64{100.0},
		},
		{
			name: "BC-3: always-on + phased mix",
			clients: []ClientSpec{
				{ID: "always", RateFraction: 0.5}, // no lifecycle = always-on
				{ID: "phased", RateFraction: 0.5, Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{{StartUs: 0, EndUs: 50_000_000}},
				}},
			},
			// Both overlap (always-on overlaps everything)
			// coActiveSum for both = 0.5 + 0.5 = 1.0
			wantRates: []float64{50.0, 50.0},
		},
		{
			name: "known-limitation: always-on + two non-overlapping phases — per-phase total < aggregate_rate",
			clients: []ClientSpec{
				{ID: "always", RateFraction: 0.5}, // always-on, overlaps both phases
				{ID: "p1", RateFraction: 0.5, Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{{StartUs: 0, EndUs: 50_000_000}},
				}},
				{ID: "p2", RateFraction: 0.5, Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{{StartUs: 50_000_000, EndUs: 100_000_000}},
				}},
			},
			// always-on coActiveSum = 0.5+0.5+0.5 = 1.5 → rate = 100*0.5/1.5 ≈ 33.33
			// p1 coActiveSum = 0.5+0.5 = 1.0 → rate = 100*0.5/1.0 = 50
			// p2 coActiveSum = 0.5+0.5 = 1.0 → rate = 100*0.5/1.0 = 50
			// Per-phase total ≈ 83.33, not 100. This is a known limitation of
			// per-client normalization; a fully phase-aware algorithm would fix it.
			wantRates: []float64{100.0 / 3, 50.0, 50.0},
		},
		{
			name: "edge: all zero fractions",
			clients: []ClientSpec{
				{ID: "a", RateFraction: 0},
				{ID: "b", RateFraction: 0},
			},
			wantRates: []float64{0, 0},
		},
		{
			name: "edge: single client no lifecycle",
			clients: []ClientSpec{
				{ID: "only", RateFraction: 1.0},
			},
			wantRates: []float64{100.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rates := normalizeRateFractions(tt.clients, aggregateRate)
			if len(rates) != len(tt.wantRates) {
				t.Fatalf("got %d rates, want %d", len(rates), len(tt.wantRates))
			}
			for i, got := range rates {
				gotReqPerSec := got * toReqPerUs
				want := tt.wantRates[i]
				if math.Abs(gotReqPerSec-want) > 0.01 {
					t.Errorf("client %d (%s): rate = %.4f req/s, want %.4f req/s",
						i, tt.clients[i].ID, gotReqPerSec, want)
				}
			}
		})
	}
}

func TestWindowsOverlap(t *testing.T) {
	tests := []struct {
		name string
		a, b *LifecycleSpec
		want bool
	}{
		{"both nil", nil, nil, true},
		{"a nil b has windows", nil, &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 0, EndUs: 100}}}, true},
		{"a has windows b nil", &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 0, EndUs: 100}}}, nil, true},
		{"a empty windows", &LifecycleSpec{}, &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 0, EndUs: 100}}}, true},
		{"overlapping", &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 0, EndUs: 100}}}, &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 50, EndUs: 150}}}, true},
		{"adjacent no overlap", &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 0, EndUs: 100}}}, &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 100, EndUs: 200}}}, false},
		{"disjoint", &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 0, EndUs: 50}}}, &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 100, EndUs: 200}}}, false},
		{"multi-window one overlaps", &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 0, EndUs: 50}, {StartUs: 200, EndUs: 300}}}, &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 250, EndUs: 350}}}, true},
		{"multi-window none overlap", &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 0, EndUs: 50}, {StartUs: 200, EndUs: 250}}}, &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 100, EndUs: 150}}}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := windowsOverlap(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("windowsOverlap = %v, want %v", got, tt.want)
			}
		})
	}
}
