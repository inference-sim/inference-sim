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

// TestPDTrace_DisaggMode_AllRecordTypesPresent verifies BC-PD-17:
// all 4 PD trace record types are emitted for each disaggregated request.
func TestPDTrace_DisaggMode_AllRecordTypesPresent(t *testing.T) {
	const numRequests = 5
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	requests := newTestRequests(numRequests)
	cs := NewClusterSimulator(config, requests)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// BC-PD-17: one record of each type per disaggregated request
	if len(tr.Disaggregations) != numRequests {
		t.Errorf("Disaggregations: expected %d, got %d", numRequests, len(tr.Disaggregations))
	}
	if len(tr.PrefillRoutings) != numRequests {
		t.Errorf("PrefillRoutings: expected %d, got %d", numRequests, len(tr.PrefillRoutings))
	}
	if len(tr.KVTransfers) != numRequests {
		t.Errorf("KVTransfers: expected %d, got %d", numRequests, len(tr.KVTransfers))
	}
	if len(tr.DecodeRoutings) != numRequests {
		t.Errorf("DecodeRoutings: expected %d, got %d", numRequests, len(tr.DecodeRoutings))
	}

	// Verify KVTransfer records have both instance IDs and non-zero duration
	for i, kv := range tr.KVTransfers {
		if kv.PrefillInstanceID == "" {
			t.Errorf("KVTransfers[%d]: PrefillInstanceID empty", i)
		}
		if kv.DecodeInstanceID == "" {
			t.Errorf("KVTransfers[%d]: DecodeInstanceID empty", i)
		}
		if kv.TransferDuration <= 0 {
			t.Errorf("KVTransfers[%d]: TransferDuration=%d, want > 0", i, kv.TransferDuration)
		}
		if kv.NumKVBlocks <= 0 {
			t.Errorf("KVTransfers[%d]: NumKVBlocks=%d, want > 0", i, kv.NumKVBlocks)
		}
	}

	// Verify per-pool routing records have non-empty chosen instances
	for i, r := range tr.PrefillRoutings {
		if r.ChosenInstance == "" {
			t.Errorf("PrefillRoutings[%d]: ChosenInstance empty", i)
		}
		if r.ParentRequestID == "" {
			t.Errorf("PrefillRoutings[%d]: ParentRequestID empty", i)
		}
	}
	for i, r := range tr.DecodeRoutings {
		if r.ChosenInstance == "" {
			t.Errorf("DecodeRoutings[%d]: ChosenInstance empty", i)
		}
		if r.ParentRequestID == "" {
			t.Errorf("DecodeRoutings[%d]: ParentRequestID empty", i)
		}
	}
}

// TestPDTrace_DisaggMode_Counterfactual verifies BC-PD-19:
// per-pool routing records have counterfactual candidates when k > 0.
func TestPDTrace_DisaggMode_Counterfactual(t *testing.T) {
	// GIVEN disaggregated simulation with k=2, 2 prefill + 2 decode instances
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	config.CounterfactualK = 2
	config.RoutingPolicy = "weighted"
	config.RoutingScorerConfigs = sim.DefaultScorerConfigs()
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// THEN prefill routing records have candidates (BC-PD-19)
	for i, r := range tr.PrefillRoutings {
		if len(r.Candidates) == 0 {
			t.Errorf("PrefillRoutings[%d]: expected candidates with k=2, got none", i)
		}
		if len(r.Candidates) > 2 {
			t.Errorf("PrefillRoutings[%d]: expected ≤2 candidates, got %d", i, len(r.Candidates))
		}
		if r.Regret < 0 {
			t.Errorf("PrefillRoutings[%d]: Regret=%f, want ≥0", i, r.Regret)
		}
	}

	// THEN decode routing records have candidates (BC-PD-19)
	for i, r := range tr.DecodeRoutings {
		if len(r.Candidates) == 0 {
			t.Errorf("DecodeRoutings[%d]: expected candidates with k=2, got none", i)
		}
		if len(r.Candidates) > 2 {
			t.Errorf("DecodeRoutings[%d]: expected ≤2 candidates, got %d", i, len(r.Candidates))
		}
		if r.Regret < 0 {
			t.Errorf("DecodeRoutings[%d]: Regret=%f, want ≥0", i, r.Regret)
		}
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
