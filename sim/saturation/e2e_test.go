// sim/saturation/e2e_test.go
package saturation_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/saturation"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// TestE2E_ExtractorParity_ByteIdenticalTrace closes the INV-13 loop hermetically:
// the sim-side extractor (Metrics.CompletedRequestMetrics) and the observe-side
// extractor (workload.TraceRecordsToRequestMetrics ∘ RequestsToTraceRecords) must
// feed ReplayOneDetector to produce BYTE-IDENTICAL trace files. This proves the
// "same (ArrivedAt,E2E,ID) triples ⇒ same trace bytes" implication directly,
// rather than only asserting the triples match (tracev2_test) and the pipeline is
// deterministic (replay_test) separately.
func TestE2E_ExtractorParity_ByteIdenticalTrace(t *testing.T) {
	type spec struct {
		id        int
		arrivalUs int64
		e2eUs     int64
	}
	specs := []spec{
		{0, 1_000_000, 150_000},
		{1, 2_500_000, 300_000},
		{2, 500_000, 50_000}, // out of arrival order
	}

	// Sim side: a Metrics as run/replay would populate.
	m := &sim.Metrics{
		Requests:     make(map[string]sim.RequestMetrics),
		RequestE2Es:  make(map[string]float64),
		RequestTTFTs: make(map[string]float64),
	}
	simReqs := make([]*sim.Request, 0, len(specs))
	for _, s := range specs {
		id := fmt.Sprintf("request_%d", s.id)
		m.CompletedRequests++
		m.Requests[id] = sim.RequestMetrics{ID: id, ArrivedAt: float64(s.arrivalUs) / 1e6}
		m.RequestE2Es[id] = float64(s.e2eUs) // ticks; /1e3 → ms in CompletedRequestMetrics
		m.RequestTTFTs[id] = 0
		simReqs = append(simReqs, &sim.Request{
			ID:             id,
			ArrivalTime:    s.arrivalUs,
			TTFTSet:        true,
			FirstTokenTime: s.e2eUs,
			ITL:            []int64{},
			State:          sim.StateCompleted,
		})
	}

	writeTrace := func(reqs []sim.RequestMetrics) []byte {
		det, err := saturation.BuildDetector("composite", saturation.SaturationConfig{})
		if err != nil {
			t.Fatalf("BuildDetector: %v", err)
		}
		c := saturation.NewInMemoryCollector()
		saturation.ReplayOneDetector(det, reqs, c)
		path := filepath.Join(t.TempDir(), "t.json")
		if err := saturation.WriteCombinedReport(path, c); err != nil {
			t.Fatalf("WriteCombinedReport: %v", err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		return data
	}

	runBytes := writeTrace(m.CompletedRequestMetrics())
	observeBytes := writeTrace(workload.TraceRecordsToRequestMetrics(workload.RequestsToTraceRecords(simReqs)))

	if string(runBytes) != string(observeBytes) {
		t.Errorf("run-side and observe-side traces differ:\n--- run ---\n%s\n--- observe ---\n%s", runBytes, observeBytes)
	}
}

// TestE2E_ReplayComposite_WritesTrace verifies the full replay → collect → write
// pipeline (#1516): a composite detector streamed over completed requests writes
// a non-empty {"trace":[...]} file whose last verdict reflects a rising latency
// trend. Smoothly increasing latency across 24 completions satisfies the
// quartile-monotone filter → OVERLOADED.
func TestE2E_ReplayComposite_WritesTrace(t *testing.T) {
	requests := make([]sim.RequestMetrics, 24)
	for i := 0; i < 24; i++ {
		requests[i] = sim.RequestMetrics{
			ID:        fmt.Sprintf("request_%d", i),
			ArrivedAt: float64(i),
			E2E:       float64(100 + i*5), // 100ms → 215ms
		}
	}

	det, err := saturation.BuildDetector("composite", saturation.SaturationConfig{})
	if err != nil {
		t.Fatalf("BuildDetector: %v", err)
	}
	collector := saturation.NewInMemoryCollector()
	saturation.ReplayOneDetector(det, requests, collector)

	tmpFile := t.TempDir() + "/trace.json"
	if err := saturation.WriteCombinedReport(tmpFile, collector); err != nil {
		t.Fatalf("WriteCombinedReport: %v", err)
	}

	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("read trace: %v", err)
	}
	var report saturation.CombinedReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("unmarshal trace: %v", err)
	}

	// 24 requests → 48 events → 48 verdict records.
	if len(report.Trace) != 48 {
		t.Fatalf("expected 48 trace records (2 events × 24 requests), got %d", len(report.Trace))
	}
	last := report.Trace[len(report.Trace)-1]
	if last.Detector != "composite" {
		t.Errorf("expected detector=composite, got %q", last.Detector)
	}
	if last.Result.Level != saturation.Overloaded {
		t.Errorf("expected final verdict OVERLOADED for rising latency, got %v (lt=%.3f)",
			last.Result.Level, last.Result.Signals["latency_trend"])
	}
}

// TestE2E_ReplayThreshold_BelowAndAbove verifies the threshold detector's final
// verdict through the replay pipeline for stable vs overloaded latency.
func TestE2E_ReplayThreshold_BelowAndAbove(t *testing.T) {
	below := []sim.RequestMetrics{
		{ID: "request_0", ArrivedAt: 0, E2E: 3000},
		{ID: "request_1", ArrivedAt: 1, E2E: 4000},
		{ID: "request_2", ArrivedAt: 2, E2E: 3500},
	}
	above := []sim.RequestMetrics{
		{ID: "request_0", ArrivedAt: 0, E2E: 6000},
		{ID: "request_1", ArrivedAt: 1, E2E: 7000},
		{ID: "request_2", ArrivedAt: 2, E2E: 8000},
	}

	for _, tc := range []struct {
		name  string
		reqs  []sim.RequestMetrics
		level saturation.Level
	}{
		{"below", below, saturation.Stable},
		{"above", above, saturation.Overloaded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			det, err := saturation.BuildDetector("threshold", saturation.SaturationConfig{})
			if err != nil {
				t.Fatalf("BuildDetector: %v", err)
			}
			collector := saturation.NewInMemoryCollector()
			saturation.ReplayOneDetector(det, tc.reqs, collector)
			recs := collector.Records()
			if len(recs) == 0 {
				t.Fatal("expected non-empty trace")
			}
			if got := recs[len(recs)-1].Result.Level; got != tc.level {
				t.Errorf("expected final level %v, got %v", tc.level, got)
			}
		})
	}
}

// TestE2E_ReplayEmptyInput_WritesEmptyTrace verifies the empty-input contract:
// zero completed requests writes a valid {"trace":[]} file, not an error.
func TestE2E_ReplayEmptyInput_WritesEmptyTrace(t *testing.T) {
	det, err := saturation.BuildDetector("backlog-drift", saturation.SaturationConfig{})
	if err != nil {
		t.Fatalf("BuildDetector: %v", err)
	}
	collector := saturation.NewInMemoryCollector()
	saturation.ReplayOneDetector(det, nil, collector)

	tmpFile := t.TempDir() + "/empty.json"
	if err := saturation.WriteCombinedReport(tmpFile, collector); err != nil {
		t.Fatalf("WriteCombinedReport: %v", err)
	}
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("read trace: %v", err)
	}
	var report saturation.CombinedReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("unmarshal trace: %v", err)
	}
	if len(report.Trace) != 0 {
		t.Errorf("expected empty trace, got %d records", len(report.Trace))
	}
}
