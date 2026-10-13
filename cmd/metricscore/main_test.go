package main

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel/harness"
)

// The observation harness.Run scores at a point -- on a prefix-caching point, the main phase
// after the warm-up's state (Primed) -- is the one that enters BOTH the absolute error (mape,
// against the measured absolute latency) and the shape error (against the measured relatives,
// anchored at the lowest concurrency). Regression: the errors are recomputed by hand from the
// primed observations, so an implementation that read any other value -- a cold run, a
// different point's -- fails.
func TestThePrimedObservationEntersBothErrors(t *testing.T) {
	sw := harness.Sweep{Points: []harness.Point{
		{Concurrency: 4, MeasuredRelative: 1, MeasuredTTFTRelative: 1},
		{Concurrency: 8, MeasuredRelative: 1.25, MeasuredTTFTRelative: 2},
	}}
	obs := []harness.Observation{
		{MeanTPOTUs: 1000, MeanTTFTUs: 50000, Primed: true},
		{MeanTPOTUs: 1100, MeanTTFTUs: 90000, Primed: true},
	}
	var measured harness.Absolutes
	if err := json.Unmarshal([]byte(`{"absolute_tpot_us":{"4":800,"8":1000},"absolute_ttft_us":{"4":40000,"8":100000}}`), &measured); err != nil {
		t.Fatal(err)
	}
	if v, ok := measured.At(harness.MetricTPOT, 8); !ok || v != 1000 {
		t.Fatalf("test absolutes did not decode: %v %v", v, ok)
	}
	shape, mape, bad := simulatedErrors(sw, obs, &measured, map[int]bool{})
	if bad != "" {
		t.Fatal(bad)
	}
	near := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
	wantMape := map[harness.Metric][]float64{
		harness.MetricTPOT: {(1000.0/800 - 1) * 100, (1100.0/1000 - 1) * 100},
		harness.MetricTTFT: {(50000.0/40000 - 1) * 100, (90000.0/100000 - 1) * 100},
	}
	wantShape := map[harness.Metric][]float64{
		harness.MetricTPOT: {(1100.0/1000/1.25 - 1) * 100},
		harness.MetricTTFT: {(90000.0/50000/2 - 1) * 100},
	}
	for _, m := range harness.AllMetrics {
		for i := range wantMape[m] {
			if len(mape[m]) != len(wantMape[m]) || !near(mape[m][i], wantMape[m][i]) {
				t.Errorf("%s mape %v, want %v from the primed observations", m, mape[m], wantMape[m])
			}
		}
		if len(shape[m]) != 1 || !near(shape[m][0], wantShape[m][0]) {
			t.Errorf("%s shape %v, want %v from the primed observations", m, shape[m], wantShape[m])
		}
	}
	// A spike point leaves TTFT, and only TTFT.
	shape, mape, _ = simulatedErrors(sw, obs, &measured, map[int]bool{1: true})
	if len(shape[harness.MetricTTFT]) != 0 || len(mape[harness.MetricTTFT]) != 1 || len(shape[harness.MetricTPOT]) != 1 {
		t.Errorf("a TTFT spike changed more than TTFT: shape %v mape %v", shape, mape)
	}
}
