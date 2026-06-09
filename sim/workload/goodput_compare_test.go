package workload

import "testing"

// TestComputeGoodputComparison_Empty verifies BC-9 no-op path: no targets => nil report.
func TestComputeGoodputComparison_Empty(t *testing.T) {
	got := ComputeGoodputComparison(nil, nil, nil, nil, nil, 1.0)
	if got != nil {
		t.Errorf("expected nil report when no targets configured, got %+v", got)
	}
}

// TestComputeGoodputComparison_E2EOnly verifies the basic compare path:
// real-side meets E2E threshold, sim-side does not, both contribute to denominator.
func TestComputeGoodputComparison_E2EOnly(t *testing.T) {
	records := []TraceRecord{
		{RequestID: 1, SLOClass: "default", Status: "ok", SendTimeUs: 0, FirstChunkTimeUs: 100_000, LastChunkTimeUs: 1_000_000}, // 1s real E2E (passes 5s)
	}
	simResults := map[int]SimResult{
		1: {RequestID: 1, TTFT: 100_000, E2E: 10_000_000}, // 10s sim E2E (fails 5s)
	}
	matched := map[int]bool{1: true}
	targets := map[string]SLODimTargets{"default": {E2EMs: 5000}}

	got := ComputeGoodputComparison(records, simResults, matched, nil, targets, 10.0)
	if got == nil {
		t.Fatal("expected non-nil report")
	}
	def := got.PerClass["default"]
	if def.Count != 1 {
		t.Errorf("count: got %d, want 1", def.Count)
	}
	if def.RealSLOAttainment != 1.0 {
		t.Errorf("real attainment: got %v, want 1.0", def.RealSLOAttainment)
	}
	if def.SimSLOAttainment != 0.0 {
		t.Errorf("sim attainment: got %v, want 0.0", def.SimSLOAttainment)
	}
	if def.RealGoodputRPS != 0.1 {
		t.Errorf("real goodput_rps: got %v, want 0.1", def.RealGoodputRPS)
	}
	if def.SimGoodputRPS != 0.0 {
		t.Errorf("sim goodput_rps: got %v, want 0.0", def.SimGoodputRPS)
	}
	if got.SkippedITL {
		t.Errorf("SkippedITL: should be false when ITL not configured")
	}
}

// TestComputeGoodputComparison_ITLMissing verifies BC-9 ITL fallback:
// when ITL is configured but missing on either side, SkippedITL is true and
// only TTFT/E2E rows are computed.
func TestComputeGoodputComparison_ITLMissing(t *testing.T) {
	records := []TraceRecord{
		{RequestID: 1, SLOClass: "critical", Status: "ok", SendTimeUs: 0, FirstChunkTimeUs: 50_000, LastChunkTimeUs: 1_000_000},
	}
	simResults := map[int]SimResult{
		1: {RequestID: 1, TTFT: 50_000, E2E: 1_000_000, ITLMeanUs: 0}, // ITL missing on sim side
	}
	matched := map[int]bool{1: true}
	targets := map[string]SLODimTargets{"critical": {TTFTMs: 100, E2EMs: 5000, ITLMs: 50}}

	got := ComputeGoodputComparison(records, simResults, matched, nil /* no ITL records on real side */, targets, 1.0)
	if got == nil {
		t.Fatal("expected non-nil report")
	}
	if !got.SkippedITL {
		t.Errorf("SkippedITL: got false, want true")
	}
	// real-side ITL missing → real not "good"; sim-side ITL missing → sim not "good"
	crit := got.PerClass["critical"]
	if crit.RealSLOAttainment != 0.0 {
		t.Errorf("real attainment: got %v, want 0.0 (ITL missing fails the AND)", crit.RealSLOAttainment)
	}
	if crit.SimSLOAttainment != 0.0 {
		t.Errorf("sim attainment: got %v, want 0.0 (ITL missing fails the AND)", crit.SimSLOAttainment)
	}
}
