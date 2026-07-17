package workload

import (
	"os"
	"path/filepath"
	"testing"
)

// T041 (US5, #1470): DT rank-table/reference import + per-config TTFT and
// throughput comparison report with an error metric (MAPE). Fail-first contract
// test — defines the API before implementation.

func TestLoadDTReference_ParsesAggregates(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ref.json")
	// Minimal DT reference: adapter_aware + adapter_blind aggregate blocks.
	json := `{
	  "model": "qwen-2.5-7b-instruct",
	  "adapter_aware": {"ttft": 10696.3, "output_throughput": 1034.5, "total_throughput": 3142.9},
	  "adapter_blind": {"ttft": 3251.7, "output_throughput": 1144.6, "total_throughput": 3463.8}
	}`
	if err := os.WriteFile(path, []byte(json), 0o644); err != nil {
		t.Fatal(err)
	}
	ref, err := LoadDTReference(path)
	if err != nil {
		t.Fatalf("LoadDTReference: %v", err)
	}
	if ref.Model != "qwen-2.5-7b-instruct" {
		t.Errorf("model = %q, want qwen-2.5-7b-instruct", ref.Model)
	}
	if ref.AdapterAware.TTFTMs != 10696.3 {
		t.Errorf("aware ttft = %v, want 10696.3", ref.AdapterAware.TTFTMs)
	}
	if ref.AdapterBlind.OutputThroughput != 1144.6 {
		t.Errorf("blind output_throughput = %v, want 1144.6", ref.AdapterBlind.OutputThroughput)
	}
}

func TestCompareAdapterReference_ProducesPerMetricMAPE(t *testing.T) {
	ref := &DTReference{
		Model:        "m",
		AdapterAware: DTAggregate{TTFTMs: 1000, OutputThroughput: 100},
		AdapterBlind: DTAggregate{TTFTMs: 500, OutputThroughput: 110},
	}
	// BLIS aware exactly matches DT aware → 0% MAPE on both metrics.
	aware := BLISAggregate{TTFTMs: 1000, OutputThroughput: 100}
	rep := CompareAdapterReference(ref, aware, nil, 0.20)
	if rep == nil {
		t.Fatal("nil report")
	}
	if len(rep.Metrics) != 2 {
		t.Fatalf("expected ttft+throughput metrics, got %d", len(rep.Metrics))
	}
	for _, m := range rep.Metrics {
		if m.MAPE != 0 {
			t.Errorf("metric %s MAPE = %v, want 0 (exact match)", m.Metric, m.MAPE)
		}
		if !m.Within {
			t.Errorf("metric %s should be within threshold at 0 MAPE", m.Metric)
		}
	}
	if !rep.AllWithin {
		t.Error("AllWithin should be true when every metric is within threshold")
	}
}

func TestCompareAdapterReference_FlagsExceedances(t *testing.T) {
	ref := &DTReference{
		Model:        "m",
		AdapterAware: DTAggregate{TTFTMs: 10000, OutputThroughput: 1000},
		AdapterBlind: DTAggregate{TTFTMs: 3000, OutputThroughput: 1100},
	}
	// BLIS TTFT wildly off (base mismatch), throughput close — mirrors the real
	// PR7 finding. Threshold 20%.
	aware := BLISAggregate{TTFTMs: 160, OutputThroughput: 1150}
	rep := CompareAdapterReference(ref, aware, nil, 0.20)
	byName := map[string]AdapterMetricComparison{}
	for _, m := range rep.Metrics {
		byName[m.Metric] = m
	}
	if byName["ttft"].Within {
		t.Errorf("ttft MAPE %.3f should exceed 0.20 (base mismatch)", byName["ttft"].MAPE)
	}
	if !byName["throughput"].Within {
		t.Errorf("throughput MAPE %.3f should be within 0.20", byName["throughput"].MAPE)
	}
	if rep.AllWithin {
		t.Error("AllWithin must be false when ttft exceeds threshold (no silent pass)")
	}
}

func TestCompareAdapterReference_DeltaNormalizedDiagnostic(t *testing.T) {
	ref := &DTReference{
		Model:        "m",
		AdapterAware: DTAggregate{TTFTMs: 1000, OutputThroughput: 90},
		AdapterBlind: DTAggregate{TTFTMs: 500, OutputThroughput: 100}, // DT tput ratio 0.90
	}
	aware := BLISAggregate{TTFTMs: 200, OutputThroughput: 90}
	blind := &BLISAggregate{TTFTMs: 100, OutputThroughput: 100} // BLIS tput ratio 0.90 → delta 0%
	rep := CompareAdapterReference(ref, aware, blind, 0.20)
	var tput AdapterMetricComparison
	for _, m := range rep.Metrics {
		if m.Metric == "throughput" {
			tput = m
		}
	}
	if tput.DeltaMAPE == nil {
		t.Fatal("delta-normalized MAPE should be populated when blind aggregate is provided")
	}
	if *tput.DeltaMAPE > 1e-9 {
		t.Errorf("throughput delta MAPE = %v, want ~0 (both ratios 0.90)", *tput.DeltaMAPE)
	}
}
