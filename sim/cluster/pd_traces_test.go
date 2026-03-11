package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestPDTrace_NonDisaggMode_NoDisaggRecords verifies BC-PD-18:
// when disaggregation is not configured, no PD-specific trace records are emitted.
func TestPDTrace_NonDisaggMode_NoDisaggRecords(t *testing.T) {
	// GIVEN non-disaggregated simulation with trace enabled
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances: 4,
		TraceLevel:   "decisions",
	}
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)

	// WHEN run
	mustRun(t, cs)

	// THEN no disaggregation-specific trace records
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace with trace-level decisions")
	}
	if len(tr.Disaggregations) != 0 {
		t.Errorf("expected 0 disaggregation records in non-disagg mode, got %d", len(tr.Disaggregations))
	}
	if len(tr.PrefillRoutings) != 0 {
		t.Errorf("expected 0 prefill routing records in non-disagg mode, got %d", len(tr.PrefillRoutings))
	}
	if len(tr.DecodeRoutings) != 0 {
		t.Errorf("expected 0 decode routing records in non-disagg mode, got %d", len(tr.DecodeRoutings))
	}
	if len(tr.KVTransfers) != 0 {
		t.Errorf("expected 0 KV transfer records in non-disagg mode, got %d", len(tr.KVTransfers))
	}
	// Existing admission/routing records still present (BC-TRACE-COMPAT)
	if len(tr.Admissions) != 5 {
		t.Errorf("expected 5 admission records, got %d", len(tr.Admissions))
	}
	if len(tr.Routings) != 5 {
		t.Errorf("expected 5 routing records, got %d", len(tr.Routings))
	}
}

// TestPDTrace_DisaggMode_DisaggDecisionRecorded verifies disaggregation decisions are recorded.
func TestPDTrace_DisaggMode_DisaggDecisionRecorded(t *testing.T) {
	// GIVEN disaggregated simulation with trace enabled
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests)

	// WHEN run
	mustRun(t, cs)

	// THEN exactly 3 disaggregation records (one per request), all Disaggregate=true
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	if len(tr.Disaggregations) != 3 {
		t.Errorf("expected 3 disaggregation records (one per request), got %d", len(tr.Disaggregations))
	}
	for i, r := range tr.Disaggregations {
		if !r.Disaggregate {
			t.Errorf("disaggregation[%d]: Disaggregate=false, want true (AlwaysDisaggregate)", i)
		}
		if r.RequestID == "" {
			t.Errorf("disaggregation[%d]: RequestID empty", i)
		}
	}
}
