package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestDistribution_FromValues_ComputesCorrectStats verifies BC-2.
func TestDistribution_FromValues_ComputesCorrectStats(t *testing.T) {
	tests := []struct {
		name      string
		values    []float64
		wantCount int
		wantMin   float64
		wantMax   float64
		wantMean  float64
	}{
		{
			name:      "single value",
			values:    []float64{100.0},
			wantCount: 1,
			wantMin:   100.0,
			wantMax:   100.0,
			wantMean:  100.0,
		},
		{
			name:      "multiple values",
			values:    []float64{10.0, 20.0, 30.0, 40.0, 50.0},
			wantCount: 5,
			wantMin:   10.0,
			wantMax:   50.0,
			wantMean:  30.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDistribution(tt.values)
			if d.Count != tt.wantCount {
				t.Errorf("Count: got %d, want %d", d.Count, tt.wantCount)
			}
			if d.Min != tt.wantMin {
				t.Errorf("Min: got %f, want %f", d.Min, tt.wantMin)
			}
			if d.Max != tt.wantMax {
				t.Errorf("Max: got %f, want %f", d.Max, tt.wantMax)
			}
			if d.Mean != tt.wantMean {
				t.Errorf("Mean: got %f, want %f", d.Mean, tt.wantMean)
			}
			// P99 of [10,20,30,40,50] should be close to 50
			if tt.name == "multiple values" && d.P99 < 40.0 {
				t.Errorf("P99: got %f, expected >= 40.0", d.P99)
			}
		})
	}
}

// TestDistribution_EmptyValues_ReturnsZero verifies edge case.
func TestDistribution_EmptyValues_ReturnsZero(t *testing.T) {
	d := NewDistribution([]float64{})
	if d.Count != 0 {
		t.Errorf("Count: got %d, want 0", d.Count)
	}
	if d.Mean != 0 {
		t.Errorf("Mean: got %f, want 0", d.Mean)
	}
}

// TestCollectRawMetrics_BasicAggregation verifies BC-1.
func TestCollectRawMetrics_BasicAggregation(t *testing.T) {
	// GIVEN aggregated metrics with known TTFT and E2E values
	m := sim.NewMetrics()
	m.RequestTTFTs = map[string]float64{
		"r0": 1000.0,
		"r1": 2000.0,
		"r2": 3000.0,
	}
	m.RequestE2Es = map[string]float64{
		"r0": 5000.0,
		"r1": 10000.0,
		"r2": 15000.0,
	}
	m.CompletedRequests = 3
	m.TotalOutputTokens = 300
	m.SimEndedTime = 1_000_000 // 1 second

	// WHEN collecting RawMetrics
	raw := CollectRawMetrics(m, nil, 0)

	// THEN TTFT distribution should be populated
	if raw.TTFT.Count != 3 {
		t.Errorf("TTFT.Count: got %d, want 3", raw.TTFT.Count)
	}
	if raw.TTFT.Min != 1000.0 {
		t.Errorf("TTFT.Min: got %f, want 1000.0", raw.TTFT.Min)
	}

	// THEN throughput should be computed
	wantRPS := 3.0 / 1.0
	if math.Abs(raw.RequestsPerSec-wantRPS) > 0.01 {
		t.Errorf("RequestsPerSec: got %f, want %f", raw.RequestsPerSec, wantRPS)
	}
	wantTPS := 300.0 / 1.0
	if math.Abs(raw.TokensPerSec-wantTPS) > 0.01 {
		t.Errorf("TokensPerSec: got %f, want %f", raw.TokensPerSec, wantTPS)
	}
}

// TestCollectRawMetrics_ZeroCompleted_ReturnsEmptyDistributions verifies edge case.
func TestCollectRawMetrics_ZeroCompleted_ReturnsEmptyDistributions(t *testing.T) {
	m := sim.NewMetrics()
	raw := CollectRawMetrics(m, nil, 0)
	if raw.TTFT.Count != 0 {
		t.Errorf("TTFT.Count: got %d, want 0", raw.TTFT.Count)
	}
	if raw.RequestsPerSec != 0 {
		t.Errorf("RequestsPerSec: got %f, want 0", raw.RequestsPerSec)
	}
}

// TestCollectRawMetrics_RejectedRequests verifies rejected count is captured.
func TestCollectRawMetrics_RejectedRequests(t *testing.T) {
	m := sim.NewMetrics()
	raw := CollectRawMetrics(m, nil, 42)
	if raw.RejectedRequests != 42 {
		t.Errorf("RejectedRequests: got %d, want 42", raw.RejectedRequests)
	}
}

// TestComputeFitness_WeightedScore verifies BC-3.
func TestComputeFitness_WeightedScore(t *testing.T) {
	raw := &RawMetrics{
		RequestsPerSec: 100.0,
		TTFT:           Distribution{P99: 5000.0},
	}

	weights := map[string]float64{
		"throughput": 1.0,
	}

	result := ComputeFitness(raw, weights)

	// THEN Score should be in (0, 1] range (normalized)
	// throughput=100, reference=100 → 100/(100+100) = 0.5
	if result.Score <= 0 || result.Score > 1.0 {
		t.Errorf("Score: got %f, expected in (0, 1]", result.Score)
	}
	if math.Abs(result.Components["throughput"]-0.5) > 0.01 {
		t.Errorf("throughput component: got %f, expected ~0.5", result.Components["throughput"])
	}
}

// TestComputeFitness_LatencyInversion verifies latency metrics are inverted (lower is better).
func TestComputeFitness_LatencyInversion(t *testing.T) {
	lowLatency := &RawMetrics{TTFT: Distribution{P99: 1000.0}}   // 1ms
	highLatency := &RawMetrics{TTFT: Distribution{P99: 10000.0}} // 10ms

	weights := map[string]float64{"p99_ttft": 1.0}

	lowResult := ComputeFitness(lowLatency, weights)
	highResult := ComputeFitness(highLatency, weights)

	// THEN lower latency should produce higher fitness score
	if lowResult.Score <= highResult.Score {
		t.Errorf("Expected low latency score (%f) > high latency score (%f)", lowResult.Score, highResult.Score)
	}
	// 1ms: 1/(1+1000/1000) = 0.5
	if math.Abs(lowResult.Score-0.5) > 0.01 {
		t.Errorf("1ms latency score: got %f, expected ~0.5", lowResult.Score)
	}
}

// TestComputeFitness_MultiObjective verifies throughput and latency have comparable scale.
func TestComputeFitness_MultiObjective(t *testing.T) {
	raw := &RawMetrics{
		RequestsPerSec: 100.0,
		TTFT:           Distribution{P99: 1000.0},
	}
	weights := map[string]float64{"throughput": 0.5, "p99_ttft": 0.5}
	result := ComputeFitness(raw, weights)

	// Both at reference → both contribute 0.5 * 0.5 = 0.25, total ≈ 0.5
	if math.Abs(result.Score-0.5) > 0.01 {
		t.Errorf("Multi-objective score: got %f, expected ~0.5", result.Score)
	}
}

// TestComputeFitness_UnknownKey_Ignored verifies EC-1.
func TestComputeFitness_UnknownKey_Ignored(t *testing.T) {
	raw := &RawMetrics{RequestsPerSec: 100.0}
	weights := map[string]float64{"nonexistent": 1.0}

	result := ComputeFitness(raw, weights)
	if result.Score != 0 {
		t.Errorf("Score: got %f, expected 0 for unknown key", result.Score)
	}
}

// TestParseFitnessWeights_ValidInput verifies parsing.
func TestParseFitnessWeights_ValidInput(t *testing.T) {
	weights, err := ParseFitnessWeights("throughput:0.5,p99_ttft:0.3")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(weights) != 2 {
		t.Fatalf("expected 2 weights, got %d", len(weights))
	}
	if weights["throughput"] != 0.5 {
		t.Errorf("throughput: got %f, want 0.5", weights["throughput"])
	}
	if weights["p99_ttft"] != 0.3 {
		t.Errorf("p99_ttft: got %f, want 0.3", weights["p99_ttft"])
	}
}

// TestParseFitnessWeights_Empty verifies EC-2.
func TestParseFitnessWeights_Empty(t *testing.T) {
	weights, err := ParseFitnessWeights("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(weights) != 0 {
		t.Errorf("expected empty map, got %d entries", len(weights))
	}
}

// TestParseFitnessWeights_InvalidFormat verifies error on bad input.
func TestParseFitnessWeights_InvalidFormat(t *testing.T) {
	_, err := ParseFitnessWeights("throughput:abc")
	if err == nil {
		t.Error("expected error for non-numeric weight")
	}
}
