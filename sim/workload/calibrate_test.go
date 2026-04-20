package workload

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestComputeCalibration_PerfectMatch_ZeroMAPE(t *testing.T) {
	real := []float64{100, 200, 300, 400, 500}
	sim := []float64{100, 200, 300, 400, 500}

	report, err := ComputeCalibration(real, sim, "ttft")
	if err != nil {
		t.Fatal(err)
	}
	if report.RequestLevel.MAPE != 0.0 {
		t.Errorf("RequestLevel.MAPE = %f, want 0.0", report.RequestLevel.MAPE)
	}
	if report.RequestLevel.PearsonR != 1.0 {
		t.Errorf("RequestLevel.PearsonR = %f, want 1.0", report.RequestLevel.PearsonR)
	}
	if report.RequestLevel.Quality != "excellent" {
		t.Errorf("RequestLevel.Quality = %q, want excellent", report.RequestLevel.Quality)
	}
}

func TestComputeCalibration_KnownError_CorrectMAPE(t *testing.T) {
	real := []float64{100, 200, 300}
	sim := []float64{110, 220, 330}

	report, err := ComputeCalibration(real, sim, "e2e")
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(report.RequestLevel.MAPE-0.10) > 0.001 {
		t.Errorf("RequestLevel.MAPE = %f, want 0.10", report.RequestLevel.MAPE)
	}
	if report.RequestLevel.BiasDirection != "over-predict" {
		t.Errorf("RequestLevel.BiasDirection = %q, want over-predict", report.RequestLevel.BiasDirection)
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
	if math.Abs(report.RequestLevel.MAPE-0.10) > 0.001 {
		t.Errorf("RequestLevel.MAPE = %f, want 0.10 (skipping real=0)", report.RequestLevel.MAPE)
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

	pairs, _, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})
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

	pairs, _, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{
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

	pairs, _, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{
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

	pairs, _, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})
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

	pairs, _, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})
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

func TestCalibration_WithITL(t *testing.T) {
	// GIVEN trace records with ITL data and matching sim results
	traceRecords := []TraceRecord{
		{RequestID: 0, InputTokens: 100, OutputTokens: 50, FirstChunkTimeUs: 1000, LastChunkTimeUs: 1100, SendTimeUs: 0},
		{RequestID: 1, InputTokens: 100, OutputTokens: 50, FirstChunkTimeUs: 2000, LastChunkTimeUs: 2100, SendTimeUs: 0},
	}
	simResults := []SimResult{
		{RequestID: 0, InputTokens: 100, OutputTokens: 50, TTFT: 1000, E2E: 1100},
		{RequestID: 1, InputTokens: 100, OutputTokens: 50, TTFT: 2000, E2E: 2100},
	}
	itlRecords := []ITLRecord{
		{RequestID: 0, ChunkIndex: 0, TimestampUs: 1000},
		{RequestID: 0, ChunkIndex: 1, TimestampUs: 1020},
		{RequestID: 0, ChunkIndex: 2, TimestampUs: 1040},
		{RequestID: 1, ChunkIndex: 0, TimestampUs: 2000},
		{RequestID: 1, ChunkIndex: 1, TimestampUs: 2020},
		{RequestID: 1, ChunkIndex: 2, TimestampUs: 2040},
	}

	// WHEN preparing calibration pairs with ITL
	config := &CalibrationConfig{}
	pairs, err := PrepareCalibrationPairsWithITL(traceRecords, simResults, itlRecords, config)
	if err != nil {
		t.Fatalf("PrepareCalibrationPairsWithITL failed: %v", err)
	}

	// THEN ITL pairs are populated with correct values
	if len(pairs.ITL.Real) != 2 {
		t.Errorf("ITL.Real length = %d, want 2", len(pairs.ITL.Real))
	}
	if len(pairs.ITL.Sim) != 2 {
		t.Errorf("ITL.Sim length = %d, want 2", len(pairs.ITL.Sim))
	}

	// Verify real ITL: mean of deltas {1020-1000, 1040-1020} = {20, 20} = 20.0
	if len(pairs.ITL.Real) > 0 {
		expectedRealITL := 20.0
		if math.Abs(pairs.ITL.Real[0]-expectedRealITL) > 0.01 {
			t.Errorf("ITL.Real[0] = %.2f, want %.2f (mean of chunk deltas)", pairs.ITL.Real[0], expectedRealITL)
		}
	}

	// Verify sim ITL: (E2E - TTFT) / (OutputTokens - 1) = (1100 - 1000) / (50 - 1) ≈ 2.04
	if len(pairs.ITL.Sim) > 0 {
		expectedSimITL := 100.0 / 49.0
		if math.Abs(pairs.ITL.Sim[0]-expectedSimITL) > 0.01 {
			t.Errorf("ITL.Sim[0] = %.2f, want %.2f ((E2E-TTFT)/(OutputTokens-1))", pairs.ITL.Sim[0], expectedSimITL)
		}
	}

	// WHEN building calibration report
	report, err := BuildCalibrationReport(pairs, &ConfigMatchInfo{})
	if err != nil {
		t.Fatalf("BuildCalibrationReport failed: %v", err)
	}

	// THEN report includes ITL metric
	itlMetric, ok := report.Metrics["itl"]
	if !ok {
		t.Fatal("report.Metrics[\"itl\"] not found")
	}
	if itlMetric.NumRequests == 0 {
		t.Error("ITL metric has NumRequests=0")
	}
}

func TestComputeCalibration_PopulatesMeanAndMedian(t *testing.T) {
	// GIVEN real and sim vectors where mean ≠ median (skewed distribution)
	real := []float64{100, 200, 300, 400, 1000} // mean=400, median=300
	sim := []float64{110, 210, 310, 410, 1100}  // mean=428, median=310

	// WHEN computing calibration
	report, err := ComputeCalibration(real, sim, "ttft")

	// THEN mean and median are correctly computed (BC-1, BC-2)
	if err != nil {
		t.Fatalf("ComputeCalibration failed: %v", err)
	}
	if report.WorkloadLevel.RealMean != 400.0 {
		t.Errorf("WorkloadLevel.RealMean = %f, want 400.0", report.WorkloadLevel.RealMean)
	}
	if report.WorkloadLevel.SimMean != 428.0 {
		t.Errorf("WorkloadLevel.SimMean = %f, want 428.0", report.WorkloadLevel.SimMean)
	}
	// Median is P50 (3rd element in sorted 5-element array)
	if report.WorkloadLevel.RealMedian != 300.0 {
		t.Errorf("WorkloadLevel.RealMedian = %f, want 300.0", report.WorkloadLevel.RealMedian)
	}
	if report.WorkloadLevel.SimMedian != 310.0 {
		t.Errorf("WorkloadLevel.SimMedian = %f, want 310.0", report.WorkloadLevel.SimMedian)
	}
}

func TestComputeCalibration_ErrorFields_CorrectSignAndMagnitude(t *testing.T) {
	tests := []struct {
		name               string
		real               []float64
		sim                []float64
		wantMeanError      float64
		wantMeanPctError   float64
		wantMedianError    float64
		wantMedianPctError float64
		tolerance          float64
	}{
		{
			name:               "over-predict",
			real:               []float64{100, 200, 300},
			sim:                []float64{110, 220, 330},
			wantMeanError:      20.0, // 220 - 200
			wantMeanPctError:   0.10, // 20 / 200
			wantMedianError:    20.0, // 220 - 200
			wantMedianPctError: 0.10, // 20 / 200
			tolerance:          0.01,
		},
		{
			name:               "under-predict",
			real:               []float64{100, 200, 300},
			sim:                []float64{90, 180, 270},
			wantMeanError:      -20.0, // 180 - 200
			wantMeanPctError:   0.10,  // | -20 | / 200
			wantMedianError:    -20.0, // 180 - 200
			wantMedianPctError: 0.10,  // | -20 | / 200
			tolerance:          0.01,
		},
		{
			name:               "perfect-match",
			real:               []float64{100, 200, 300},
			sim:                []float64{100, 200, 300},
			wantMeanError:      0.0,
			wantMeanPctError:   0.0,
			wantMedianError:    0.0,
			wantMedianPctError: 0.0,
			tolerance:          0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// WHEN computing calibration
			report, err := ComputeCalibration(tt.real, tt.sim, "test")
			if err != nil {
				t.Fatalf("ComputeCalibration failed: %v", err)
			}

			// THEN error fields match expected values (BC-3 through BC-6)
			if math.Abs(report.WorkloadLevel.MeanError-tt.wantMeanError) > tt.tolerance {
				t.Errorf("WorkloadLevel.MeanError = %f, want %f", report.WorkloadLevel.MeanError, tt.wantMeanError)
			}
			if math.Abs(report.WorkloadLevel.MeanPercentError-tt.wantMeanPctError) > tt.tolerance {
				t.Errorf("WorkloadLevel.MeanPercentError = %f, want %f", report.WorkloadLevel.MeanPercentError, tt.wantMeanPctError)
			}
			if math.Abs(report.WorkloadLevel.MedianError-tt.wantMedianError) > tt.tolerance {
				t.Errorf("WorkloadLevel.MedianError = %f, want %f", report.WorkloadLevel.MedianError, tt.wantMedianError)
			}
			if math.Abs(report.WorkloadLevel.MedianPercentError-tt.wantMedianPctError) > tt.tolerance {
				t.Errorf("WorkloadLevel.MedianPercentError = %f, want %f", report.WorkloadLevel.MedianPercentError, tt.wantMedianPctError)
			}
		})
	}
}

func TestCalibration_WithITL_NegativeDelta_ClockSkew(t *testing.T) {
	// GIVEN trace records with ITL data containing negative deltas (clock skew)
	traceRecords := []TraceRecord{
		{RequestID: 0, InputTokens: 100, OutputTokens: 50, FirstChunkTimeUs: 1000, LastChunkTimeUs: 1100, SendTimeUs: 0},
		{RequestID: 1, InputTokens: 100, OutputTokens: 50, FirstChunkTimeUs: 2000, LastChunkTimeUs: 2100, SendTimeUs: 0},
	}
	simResults := []SimResult{
		{RequestID: 0, InputTokens: 100, OutputTokens: 50, TTFT: 1000, E2E: 1100},
		{RequestID: 1, InputTokens: 100, OutputTokens: 50, TTFT: 2000, E2E: 2100},
	}
	itlRecords := []ITLRecord{
		// Request 0: all negative deltas (timestamps go backward)
		{RequestID: 0, ChunkIndex: 0, TimestampUs: 1000},
		{RequestID: 0, ChunkIndex: 1, TimestampUs: 990},
		{RequestID: 0, ChunkIndex: 2, TimestampUs: 980},
		// Request 1: partial negative deltas (one valid, one negative)
		{RequestID: 1, ChunkIndex: 0, TimestampUs: 2000},
		{RequestID: 1, ChunkIndex: 1, TimestampUs: 1990}, // negative delta
		{RequestID: 1, ChunkIndex: 2, TimestampUs: 2020}, // positive delta
	}

	// WHEN preparing calibration pairs with ITL
	config := &CalibrationConfig{}
	pairs, err := PrepareCalibrationPairsWithITL(traceRecords, simResults, itlRecords, config)
	if err != nil {
		t.Fatalf("PrepareCalibrationPairsWithITL failed: %v", err)
	}

	// THEN request 0 is dropped (all deltas negative)
	if pairs.ITLDropped != 1 {
		t.Errorf("ITLDropped = %d, want 1 (request 0 with all negative deltas)", pairs.ITLDropped)
	}

	// THEN request 1 is included with only the positive delta
	if len(pairs.ITL.Real) != 1 {
		t.Errorf("ITL.Real length = %d, want 1 (request 1 with one valid delta)", len(pairs.ITL.Real))
	}
	if len(pairs.ITL.Real) > 0 {
		// Request 1 has one valid delta: chunk[2] - chunk[1] = 2020 - 1990 = 30
		// (negative delta chunk[1] - chunk[0] = 1990 - 2000 = -10 is skipped)
		expectedRealITL := 30.0
		if math.Abs(pairs.ITL.Real[0]-expectedRealITL) > 0.01 {
			t.Errorf("ITL.Real[0] = %.2f, want %.2f (one valid delta)", pairs.ITL.Real[0], expectedRealITL)
		}
	}
}

func TestComputeCalibration_ZeroRealMean_GuardsDivision(t *testing.T) {
	// GIVEN real values are all zero (degenerate input)
	real := []float64{0, 0, 0}
	sim := []float64{10, 20, 30}

	// WHEN computing calibration
	report, err := ComputeCalibration(real, sim, "test")

	// THEN MeanPercentError is set to 0 (BC-11, R11)
	if err != nil {
		t.Fatalf("ComputeCalibration failed: %v", err)
	}
	if report.WorkloadLevel.RealMean != 0.0 {
		t.Errorf("WorkloadLevel.RealMean = %f, want 0.0", report.WorkloadLevel.RealMean)
	}
	if report.WorkloadLevel.MeanPercentError != 0.0 {
		t.Errorf("WorkloadLevel.MeanPercentError = %f, want 0.0 (guarded division)", report.WorkloadLevel.MeanPercentError)
	}
	// MeanError should still be computed (not guarded)
	if report.WorkloadLevel.MeanError != 20.0 {
		t.Errorf("WorkloadLevel.MeanError = %f, want 20.0", report.WorkloadLevel.MeanError)
	}
}

func TestComputeCalibration_ZeroRealMedian_GuardsDivision(t *testing.T) {
	// GIVEN real median is zero (degenerate distribution: 0 at P50)
	real := []float64{0, 0, 100} // median = 0 (middle value)
	sim := []float64{10, 10, 110}

	// WHEN computing calibration
	report, err := ComputeCalibration(real, sim, "test")

	// THEN MedianPercentError is set to 0 (BC-12, R11)
	if err != nil {
		t.Fatalf("ComputeCalibration failed: %v", err)
	}
	if report.WorkloadLevel.RealMedian != 0.0 {
		t.Errorf("WorkloadLevel.RealMedian = %f, want 0.0", report.WorkloadLevel.RealMedian)
	}
	if report.WorkloadLevel.MedianPercentError != 0.0 {
		t.Errorf("WorkloadLevel.MedianPercentError = %f, want 0.0 (guarded division)", report.WorkloadLevel.MedianPercentError)
	}
	// MedianError should still be computed (not guarded)
	if report.WorkloadLevel.MedianError != 10.0 {
		t.Errorf("WorkloadLevel.MedianError = %f, want 10.0", report.WorkloadLevel.MedianError)
	}
}

func TestMetricComparison_JSONRoundTrip_IncludesNewFields(t *testing.T) {
	// GIVEN a MetricComparison with nested workload-level and request-level structs
	original := &MetricComparison{
		WorkloadLevel: WorkloadAggregates{
			RealP50:            5000,
			SimP50:             5100,
			RealP90:            8000,
			SimP90:             8200,
			RealP95:            9000,
			SimP95:             9100,
			RealP99:            11000,
			SimP99:             10800,
			RealMean:           5200,
			SimMean:            5400,
			RealMedian:         5000,
			SimMedian:          5100,
			MeanError:          200,
			MeanPercentError:   0.038,
			MedianError:        100,
			MedianPercentError: 0.020,
		},
		RequestLevel: PredictionQuality{
			MAPE:          0.12,
			PearsonR:      0.92,
			BiasDirection: "over-predict",
			Quality:       "good",
		},
		NumRequests: 100,
	}

	// WHEN marshaling to JSON and back
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var decoded MetricComparison
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// THEN all workload-level fields round-trip correctly (BC-7)
	if decoded.WorkloadLevel.RealMean != original.WorkloadLevel.RealMean {
		t.Errorf("WorkloadLevel.RealMean = %f, want %f", decoded.WorkloadLevel.RealMean, original.WorkloadLevel.RealMean)
	}
	if decoded.WorkloadLevel.SimMean != original.WorkloadLevel.SimMean {
		t.Errorf("WorkloadLevel.SimMean = %f, want %f", decoded.WorkloadLevel.SimMean, original.WorkloadLevel.SimMean)
	}
	if decoded.WorkloadLevel.MeanError != original.WorkloadLevel.MeanError {
		t.Errorf("WorkloadLevel.MeanError = %f, want %f", decoded.WorkloadLevel.MeanError, original.WorkloadLevel.MeanError)
	}
	if decoded.WorkloadLevel.MeanPercentError != original.WorkloadLevel.MeanPercentError {
		t.Errorf("WorkloadLevel.MeanPercentError = %f, want %f", decoded.WorkloadLevel.MeanPercentError, original.WorkloadLevel.MeanPercentError)
	}
	if decoded.WorkloadLevel.MedianError != original.WorkloadLevel.MedianError {
		t.Errorf("WorkloadLevel.MedianError = %f, want %f", decoded.WorkloadLevel.MedianError, original.WorkloadLevel.MedianError)
	}
	if decoded.WorkloadLevel.MedianPercentError != original.WorkloadLevel.MedianPercentError {
		t.Errorf("WorkloadLevel.MedianPercentError = %f, want %f", decoded.WorkloadLevel.MedianPercentError, original.WorkloadLevel.MedianPercentError)
	}
	// Verify request-level fields (BC-9)
	if decoded.RequestLevel.MAPE != original.RequestLevel.MAPE {
		t.Errorf("RequestLevel.MAPE = %f, want %f", decoded.RequestLevel.MAPE, original.RequestLevel.MAPE)
	}
	if decoded.RequestLevel.PearsonR != original.RequestLevel.PearsonR {
		t.Errorf("RequestLevel.PearsonR = %f, want %f", decoded.RequestLevel.PearsonR, original.RequestLevel.PearsonR)
	}

	// THEN JSON includes expected nested structure
	jsonStr := string(data)
	if !strings.Contains(jsonStr, "\"workload_level\"") {
		t.Errorf("JSON missing workload_level object: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, "\"request_level\"") {
		t.Errorf("JSON missing request_level object: %s", jsonStr)
	}
	expectedKeys := []string{"real_mean", "sim_mean", "mean_error", "mean_percent_error", "median_error", "median_percent_error", "mape", "pearson_r"}
	for _, key := range expectedKeys {
		if !strings.Contains(jsonStr, key) {
			t.Errorf("JSON missing key %q: %s", key, jsonStr)
		}
	}

	// THEN only canonical nested keys exist at top level (no flat deprecated keys)
	// Parse into map to check top-level keys
	var topLevel map[string]interface{}
	if err := json.Unmarshal(data, &topLevel); err != nil {
		t.Fatalf("Failed to unmarshal for top-level key check: %v", err)
	}
	// Verify only canonical keys exist at top level
	expectedTopLevelKeys := map[string]bool{"workload_level": true, "request_level": true, "num_requests": true}
	for key := range topLevel {
		if !expectedTopLevelKeys[key] {
			t.Errorf("JSON contains unexpected top-level key %q (only workload_level, request_level, count should exist): %s", key, jsonStr)
		}
	}
	if len(topLevel) != 3 {
		t.Errorf("JSON should have exactly 3 top-level keys (workload_level, request_level, count), got %d: %v", len(topLevel), topLevel)
	}
}
