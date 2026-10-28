package main

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/tools/harness"
)

// The observation harness.Run scores at each point -- on a prefix-caching point, the main phase
// after the warm-up's state (Primed) -- is what BLIS's TPOT shape error is computed from, anchored
// at the sweep's lowest concurrency and excluding that anchor; AISimulate's comes from its own
// published relatives. Regression: recomputed by hand from the primed observations.
func TestThePrimedObservationEntersTheShapeError(t *testing.T) {
	sw := harness.Sweep{Points: []harness.Point{
		{Concurrency: 4, MeasuredRelative: 1, AISimulateRelative: 1.1},
		{Concurrency: 8, MeasuredRelative: 1.25, AISimulateRelative: 1.32},
		{Concurrency: 16, MeasuredRelative: 1.6, AISimulateRelative: 1.98},
	}}
	obs := []harness.Observation{{MeanTPOTUs: 1000, Primed: true}, {MeanTPOTUs: 1300, Primed: true},
		{MeanTPOTUs: 1500, Primed: true}}
	mine, theirs, concOf, predicted, why := tpotShapeErrors(sw, obs)
	if why != "" {
		t.Fatal(why)
	}
	near := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
	wantMine := []float64{(1.3/1.25 - 1) * 100, (1.5/1.6 - 1) * 100}
	wantTheirs := []float64{(1.32/1.1/1.25 - 1) * 100, (1.98/1.1/1.6 - 1) * 100}
	for i := range wantMine {
		if !near(mine[i], wantMine[i]) || !near(theirs[i], wantTheirs[i]) {
			t.Errorf("point %d: BLIS %v AISimulate %v, want %v and %v", i+1, mine[i], theirs[i], wantMine[i], wantTheirs[i])
		}
	}
	if len(mine) != 2 || concOf[0] != 8 || concOf[1] != 16 || !near(predicted[0], 1) || !near(predicted[2], 1.5) {
		t.Errorf("the anchor must be excluded and the concurrencies kept: %v %v %v", mine, concOf, predicted)
	}
	if _, _, _, _, why := tpotShapeErrors(harness.Sweep{Points: []harness.Point{{AISimulateRelative: 0}}},
		obs[:1]); why == "" {
		t.Error("a non-positive AISimulate anchor is accepted")
	}
}
