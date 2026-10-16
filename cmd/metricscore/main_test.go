package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
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

// Each point's errors are where the scorer puts them, hand-computed: mape against the
// measured absolute, shape against the measured relative with each side anchored at the lowest
// concurrency. And each absence is where the scorer says it is: no shape at the anchor, no TTFT
// error at a spike, no mape where -absolutes has no measured value. simulatedErrors pools these
// same values, so TestThePrimedObservationEntersBothErrors and this test pin one computation.
func TestPointErrorsPutEachErrorOnItsPoint(t *testing.T) {
	sw := harness.Sweep{Points: []harness.Point{
		{Concurrency: 4, MeasuredRelative: 1, MeasuredTTFTRelative: 1},
		{Concurrency: 8, MeasuredRelative: 1.25, MeasuredTTFTRelative: 2},
		{Concurrency: 16, MeasuredRelative: 1.5, MeasuredTTFTRelative: 3},
	}}
	obs := []harness.Observation{
		{MeanTPOTUs: 1000, MeanTTFTUs: 50000},
		{MeanTPOTUs: 1100, MeanTTFTUs: 90000},
		{MeanTPOTUs: 1300, MeanTTFTUs: 160000},
	}
	var measured harness.Absolutes // no absolute at c=16
	if err := json.Unmarshal([]byte(`{"absolute_tpot_us":{"4":800,"8":1000},"absolute_ttft_us":{"4":40000,"8":100000}}`), &measured); err != nil {
		t.Fatal(err)
	}
	near := func(got *float64, want float64) bool { return got != nil && math.Abs(*got-want) < 1e-9 }

	per, bad := pointErrors(sw, obs, &measured, map[int]bool{})
	if bad != "" || len(per) != 3 {
		t.Fatalf("errors %v, bad %q", per, bad)
	}
	if !near(per[0].TPOTErrorPct, (1000.0/800-1)*100) || !near(per[1].TTFTErrorPct, (90000.0/100000-1)*100) {
		t.Errorf("mape is not pred/measured-1 at its own point: %+v %+v", per[0], per[1])
	}
	if !near(per[1].TPOTShapeErrorPct, (1100.0/1000/1.25-1)*100) || !near(per[2].TTFTShapeErrorPct, (160000.0/50000/3-1)*100) {
		t.Errorf("shape is not anchored at the lowest concurrency: %+v %+v", per[1], per[2])
	}
	if per[0].TPOTShapeErrorPct != nil || per[0].TTFTShapeErrorPct != nil {
		t.Errorf("the anchor carries a shape error: %+v", per[0])
	}
	if per[2].TPOTErrorPct != nil || per[2].TTFTErrorPct != nil || per[2].TPOTShapeErrorPct == nil {
		t.Errorf("a point with no measured absolute must lose mape and only mape: %+v", per[2])
	}

	per, _ = pointErrors(sw, obs, &measured, map[int]bool{1: true})
	if per[1].TTFTErrorPct != nil || per[1].TTFTShapeErrorPct != nil || per[1].TPOTErrorPct == nil || per[1].TPOTShapeErrorPct == nil {
		t.Errorf("a TTFT spike changed more, or less, than TTFT: %+v", per[1])
	}
}

// A record reports the sweep it came from, the tier it was scored in, the observation that
// was scored as the prediction, and the measured value it was scored against -- null, not 0,
// where there is none, since a zero measured latency would be scored by a consumer.
func TestPointRecordsReportWhatWasScored(t *testing.T) {
	sw := harness.Sweep{Scenario: "s.yaml", Model: "m", GPU: "h200", Label: "1k1k", Framework: "vllm",
		Parallelism: map[string]int{"tp": 8}, Points: []harness.Point{
			{Concurrency: 4, MeasuredRelative: 1, MeasuredTTFTRelative: 1},
			{Concurrency: 8, MeasuredRelative: 1.25, MeasuredTTFTRelative: 2},
		}}
	obs := []harness.Observation{{MeanTPOTUs: 1000, MeanTTFTUs: 50000}, {MeanTPOTUs: 1100, MeanTTFTUs: 90000}}
	var measured harness.Absolutes // TTFT absent at c=8
	if err := json.Unmarshal([]byte(`{"absolute_tpot_us":{"4":800,"8":1000},"absolute_ttft_us":{"4":40000}}`), &measured); err != nil {
		t.Fatal(err)
	}
	recs, bad := pointRecords(sw, harness.TierMeasuredOrResolved, obs, &measured, map[int]bool{})
	if bad != "" || len(recs) != 2 {
		t.Fatalf("records %v, bad %q", recs, bad)
	}
	r := recs[1]
	if r.Scenario != "s.yaml" || r.Concurrency != 8 || r.Tier != harness.TierMeasuredOrResolved.String() ||
		r.Parallelism["tp"] != 8 || r.PredictedTPOTUs != 1100 || r.PredictedTTFTUs != 90000 ||
		r.MeasuredTPOTUs == nil || *r.MeasuredTPOTUs != 1000 {
		t.Errorf("record does not report the scored sweep, tier, prediction and measurement: %+v", r)
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"measured_ttft_us":null`) || !strings.Contains(string(b), `"ttft_error_pct":null`) {
		t.Errorf("an absent measurement is not written as null: %s", b)
	}
}

// The same report writes the same bytes (INV-6): -json output is diffable run to run.
func TestJSONReportIsByteStable(t *testing.T) {
	sw := harness.Sweep{Scenario: "s", Parallelism: map[string]int{"tp": 8, "ep": 8, "dp": 1},
		Points: []harness.Point{{Concurrency: 4, MeasuredRelative: 1, MeasuredTTFTRelative: 1}}}
	recs, _ := pointRecords(sw, harness.TierApproximate, []harness.Observation{{MeanTPOTUs: 1, MeanTTFTUs: 1}}, nil, nil)
	r := jsonReport{SchemaVersion: 1, Framework: "vllm", Points: recs}
	dir := t.TempDir()
	var outs []string
	for i := 0; i < 2; i++ {
		path := filepath.Join(dir, fmt.Sprintf("r%d.json", i))
		if err := writeJSONReport(path, r); err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		outs = append(outs, string(b))
	}
	if outs[0] != outs[1] {
		t.Errorf("the same report wrote different bytes:\n%s\n---\n%s", outs[0], outs[1])
	}
}
