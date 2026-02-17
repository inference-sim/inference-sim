package workload

import (
	"math"
	"testing"
)

func TestComputeCalibration_PerfectMatch_ZeroMAPE(t *testing.T) {
	real := []float64{100, 200, 300, 400, 500}
	sim := []float64{100, 200, 300, 400, 500}

	report, err := ComputeCalibration(real, sim, "ttft")
	if err != nil {
		t.Fatal(err)
	}
	if report.MAPE != 0.0 {
		t.Errorf("MAPE = %f, want 0.0", report.MAPE)
	}
	if report.PearsonR != 1.0 {
		t.Errorf("PearsonR = %f, want 1.0", report.PearsonR)
	}
	if report.Quality != "excellent" {
		t.Errorf("quality = %q, want excellent", report.Quality)
	}
}

func TestComputeCalibration_KnownError_CorrectMAPE(t *testing.T) {
	real := []float64{100, 200, 300}
	sim := []float64{110, 220, 330}

	report, err := ComputeCalibration(real, sim, "e2e")
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(report.MAPE-0.10) > 0.001 {
		t.Errorf("MAPE = %f, want 0.10", report.MAPE)
	}
	if report.BiasDirection != "over-predict" {
		t.Errorf("bias = %q, want over-predict", report.BiasDirection)
	}
}

func TestComputeCalibration_EmptySlice_ReturnsError(t *testing.T) {
	_, err := ComputeCalibration(nil, nil, "ttft")
	if err == nil {
		t.Fatal("expected error for empty slices")
	}
}

func TestComputeCalibration_MismatchedLengths_ReturnsError(t *testing.T) {
	_, err := ComputeCalibration([]float64{1, 2}, []float64{1}, "ttft")
	if err == nil {
		t.Fatal("expected error for mismatched lengths")
	}
}

func TestComputeCalibration_RealZero_SkippedInMAPE(t *testing.T) {
	// Real has a 0; MAPE should skip it
	real := []float64{0, 200, 300}
	sim := []float64{10, 220, 330}

	report, err := ComputeCalibration(real, sim, "ttft")
	if err != nil {
		t.Fatal(err)
	}
	// MAPE computed only on 200→220 and 300→330 (10% each)
	if math.Abs(report.MAPE-0.10) > 0.001 {
		t.Errorf("MAPE = %f, want 0.10 (skipping real=0)", report.MAPE)
	}
}

func TestPrepareCalibrationPairs_MatchesByRequestID(t *testing.T) {
	realRecords := []TraceRecord{
		{RequestID: 0, ArrivalTimeUs: 0, FirstChunkTimeUs: 500, LastChunkTimeUs: 1000, SendTimeUs: 10},
		{RequestID: 1, ArrivalTimeUs: 100000, FirstChunkTimeUs: 100800, LastChunkTimeUs: 101500, SendTimeUs: 100010},
	}
	simResults := []SimResult{
		{RequestID: 1, TTFT: 750, E2E: 1400}, // out of order
		{RequestID: 0, TTFT: 450, E2E: 950},
	}

	pairs, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})
	if err != nil {
		t.Fatal(err)
	}

	if pairs.MatchedCount != 2 {
		t.Fatalf("matched = %d, want 2", pairs.MatchedCount)
	}
	// Request 0: real TTFT = 500 - 10 = 490, sim TTFT = 450
	if pairs.TTFT.Real[0] != 490 || pairs.TTFT.Sim[0] != 450 {
		t.Errorf("request 0 TTFT: real=%.0f sim=%.0f, want 490/450", pairs.TTFT.Real[0], pairs.TTFT.Sim[0])
	}
}

func TestPrepareCalibrationPairs_AppliesNetworkAdjustment(t *testing.T) {
	realRecords := []TraceRecord{
		{RequestID: 0, FirstChunkTimeUs: 6000, SendTimeUs: 100, LastChunkTimeUs: 7000},
	}
	simResults := []SimResult{
		{RequestID: 0, TTFT: 500, E2E: 900},
	}

	pairs, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{
		NetworkRTTUs: 5000,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Sim TTFT = 500 + 5000 = 5500 (client-perspective)
	if pairs.TTFT.Sim[0] != 5500 {
		t.Errorf("sim TTFT with network = %.0f, want 5500", pairs.TTFT.Sim[0])
	}
	// Real TTFT = 6000 - 100 = 5900
	if pairs.TTFT.Real[0] != 5900 {
		t.Errorf("real TTFT = %.0f, want 5900", pairs.TTFT.Real[0])
	}
}

func TestPrepareCalibrationPairs_ExcludesWarmUp(t *testing.T) {
	realRecords := make([]TraceRecord, 5)
	simResults := make([]SimResult, 5)
	for i := 0; i < 5; i++ {
		realRecords[i] = TraceRecord{RequestID: i, FirstChunkTimeUs: int64(i*1000 + 500), SendTimeUs: int64(i * 1000), LastChunkTimeUs: int64(i*1000 + 1000)}
		simResults[i] = SimResult{RequestID: i, TTFT: 450, E2E: 900}
	}

	pairs, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{
		WarmUpRequests: 2,
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(pairs.TTFT.Real) != 3 {
		t.Errorf("expected 3 pairs after warm-up exclusion, got %d", len(pairs.TTFT.Real))
	}
	if pairs.ExcludedWarmUp != 2 {
		t.Errorf("excluded warm-up = %d, want 2", pairs.ExcludedWarmUp)
	}
}

func TestPrepareCalibrationPairs_UnmatchedRequests(t *testing.T) {
	realRecords := []TraceRecord{
		{RequestID: 0, FirstChunkTimeUs: 500, SendTimeUs: 0, LastChunkTimeUs: 1000},
		{RequestID: 1, FirstChunkTimeUs: 1500, SendTimeUs: 1000, LastChunkTimeUs: 2000},
		{RequestID: 2, FirstChunkTimeUs: 2500, SendTimeUs: 2000, LastChunkTimeUs: 3000},
	}
	simResults := []SimResult{
		{RequestID: 0, TTFT: 450, E2E: 900},
		{RequestID: 1, TTFT: 480, E2E: 950},
	}

	pairs, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})
	if err != nil {
		t.Fatal(err)
	}

	if pairs.MatchedCount != 2 {
		t.Errorf("matched = %d, want 2", pairs.MatchedCount)
	}
	if pairs.UnmatchedReal != 1 {
		t.Errorf("unmatched real = %d, want 1", pairs.UnmatchedReal)
	}
}

func TestPrepareCalibrationPairs_DetectsTokenMismatch(t *testing.T) {
	realRecords := []TraceRecord{
		{RequestID: 0, InputTokens: 512, OutputTokens: 128, FirstChunkTimeUs: 500, SendTimeUs: 0, LastChunkTimeUs: 1000},
	}
	simResults := []SimResult{
		{RequestID: 0, TTFT: 450, E2E: 900, InputTokens: 500, OutputTokens: 128},
	}

	pairs, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})
	if err != nil {
		t.Fatal(err)
	}

	if pairs.TokenMismatchCount != 1 {
		t.Errorf("token mismatch count = %d, want 1", pairs.TokenMismatchCount)
	}
}

func TestBuildCalibrationReport_IncludesAllAnnotations(t *testing.T) {
	pairs := &CalibrationPairs{
		TTFT:               LatencyPair{Real: []float64{100, 200, 300}, Sim: []float64{110, 210, 310}},
		E2E:                LatencyPair{Real: []float64{500, 600, 700}, Sim: []float64{520, 630, 710}},
		TokenMismatchCount: 1,
		MatchedCount:       3,
		ExcludedWarmUp:     2,
	}
	report, err := BuildCalibrationReport(pairs, &ConfigMatchInfo{
		Matched:   []string{"max_num_seqs=256"},
		Defaulted: []string{"block_size (not in trace header)"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.KnownLimitations) == 0 {
		t.Error("KnownLimitations must not be empty")
	}
	if len(report.ConfigMatch.Matched) != 1 || len(report.ConfigMatch.Defaulted) != 1 {
		t.Error("ConfigMatch not populated correctly")
	}
	if report.TraceInfo.TokenMismatches != 1 {
		t.Errorf("TokenMismatches = %d, want 1", report.TraceInfo.TokenMismatches)
	}
	if report.Metrics["ttft"] == nil || report.Metrics["e2e"] == nil {
		t.Error("expected TTFT and E2E metric comparisons in report")
	}
	if report.TraceInfo.MatchedPairs != 3 {
		t.Errorf("matched pairs = %d, want 3", report.TraceInfo.MatchedPairs)
	}
}
