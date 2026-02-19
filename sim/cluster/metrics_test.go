package cluster

import (
	"math"
	"strings"
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

	result, err := ComputeFitness(raw, weights)
	if err != nil {
		t.Fatal(err)
	}

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

	lowResult, err := ComputeFitness(lowLatency, weights)
	if err != nil {
		t.Fatal(err)
	}
	highResult, err := ComputeFitness(highLatency, weights)
	if err != nil {
		t.Fatal(err)
	}

	// THEN lower latency should produce higher fitness score
	if lowResult.Score <= highResult.Score {
		t.Errorf("Expected low latency score (%f) > high latency score (%f)", lowResult.Score, highResult.Score)
	}
	// 1ms: 1/(1+1000/1000) = 0.5
	if math.Abs(lowResult.Score-0.5) > 0.01 {
		t.Errorf("1ms latency score: got %f, expected ~0.5", lowResult.Score)
	}
}

// TestComputeFitness_MultiObjective verifies throughput and latency contribute comparable weight.
func TestComputeFitness_MultiObjective(t *testing.T) {
	// GIVEN metrics at reference values for both throughput and latency
	raw := &RawMetrics{
		RequestsPerSec: 100.0,
		TTFT:           Distribution{P99: 1000.0},
	}
	weights := map[string]float64{"throughput": 0.5, "p99_ttft": 0.5}
	result, err := ComputeFitness(raw, weights)
	if err != nil {
		t.Fatal(err)
	}

	// THEN both components should contribute roughly equally
	throughputComponent := result.Components["throughput"]
	latencyComponent := result.Components["p99_ttft"]
	ratio := throughputComponent / latencyComponent
	// At reference values, both should produce similar normalized scores (within 2×)
	if ratio < 0.5 || ratio > 2.0 {
		t.Errorf("Components not comparable: throughput=%f, latency=%f, ratio=%f (expected 0.5-2.0)",
			throughputComponent, latencyComponent, ratio)
	}

	// THEN total score should be meaningful (not dominated by one component)
	if result.Score <= 0 || result.Score > 1.0 {
		t.Errorf("Multi-objective score out of range: got %f, expected (0, 1]", result.Score)
	}
}

// TestComputeFitness_UnknownKey_ReturnsError verifies BC-7.
func TestComputeFitness_UnknownKey_ReturnsError(t *testing.T) {
	raw := &RawMetrics{RequestsPerSec: 100.0}
	weights := map[string]float64{"nonexistent": 1.0}

	_, err := ComputeFitness(raw, weights)
	if err == nil {
		t.Error("expected error for unknown key, got nil")
	}
	// Error message should list valid keys to help the user fix the typo
	if err != nil && !strings.Contains(err.Error(), "throughput") {
		t.Errorf("error message should list valid keys, got: %v", err)
	}
}

func TestComputeFitness_MixedKnownUnknown_ReturnsError(t *testing.T) {
	raw := &RawMetrics{RequestsPerSec: 100.0}
	weights := map[string]float64{"throughput": 0.5, "invalid_key": 0.5}

	_, err := ComputeFitness(raw, weights)
	if err == nil {
		t.Error("expected error when any key is unknown")
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

// TestDetectPriorityInversions_InvertedRequests verifies BC-8.
func TestDetectPriorityInversions_InvertedRequests(t *testing.T) {
	m := sim.NewMetrics()
	m.Requests["high"] = sim.RequestMetrics{ID: "high", ArrivedAt: 100}
	m.RequestE2Es["high"] = 50000.0
	m.Requests["low"] = sim.RequestMetrics{ID: "low", ArrivedAt: 200}
	m.RequestE2Es["low"] = 5000.0

	inversions := detectPriorityInversions([]*sim.Metrics{m})

	if inversions < 0 {
		t.Errorf("inversions should be >= 0, got %d", inversions)
	}
	// Earlier request (high) has 10× worse E2E than later request (low) → inversion detected
	if inversions == 0 {
		t.Error("expected at least 1 inversion for 10× E2E difference")
	}
}

// TestDetectHOLBlocking_ImbalancedInstances verifies BC-9.
func TestDetectHOLBlocking_ImbalancedInstances(t *testing.T) {
	perInstance := []*sim.Metrics{
		makeMetricsWithQueueDepth([]int{50, 50, 50, 50}),
		makeMetricsWithQueueDepth([]int{1, 1, 1, 1}),
		makeMetricsWithQueueDepth([]int{2, 2, 2, 2}),
	}

	blocking := detectHOLBlocking(perInstance)

	if blocking <= 0 {
		t.Errorf("expected HOL blocking events > 0, got %d", blocking)
	}
}

// TestDetectHOLBlocking_BalancedInstances_NoBlocking verifies no false positives.
func TestDetectHOLBlocking_BalancedInstances_NoBlocking(t *testing.T) {
	perInstance := []*sim.Metrics{
		makeMetricsWithQueueDepth([]int{10, 10, 10}),
		makeMetricsWithQueueDepth([]int{11, 11, 11}),
		makeMetricsWithQueueDepth([]int{9, 9, 9}),
	}

	blocking := detectHOLBlocking(perInstance)

	if blocking != 0 {
		t.Errorf("expected 0 HOL blocking events for balanced instances, got %d", blocking)
	}
}

func makeMetricsWithQueueDepth(depths []int) *sim.Metrics {
	m := sim.NewMetrics()
	m.NumWaitQRequests = depths
	return m
}

// TestPathological_RejectAll_AllRejected verifies BC-4 + rejection counting.
func TestPathological_RejectAll_AllRejected(t *testing.T) {
	config := newTestDeploymentConfig(2)
	config.AdmissionPolicy = "reject-all"

	cs := NewClusterSimulator(config, newTestWorkload(20), "")
	cs.Run()

	raw := CollectRawMetrics(cs.AggregatedMetrics(), cs.PerInstanceMetrics(), cs.RejectedRequests())

	// ALL requests should be rejected
	if raw.RejectedRequests == 0 {
		t.Error("expected rejected requests > 0 with reject-all policy")
	}
	// No requests should complete
	if cs.AggregatedMetrics().CompletedRequests != 0 {
		t.Errorf("expected 0 completed requests, got %d", cs.AggregatedMetrics().CompletedRequests)
	}
}

// TestPathological_AlwaysBusiest_CausesImbalance verifies BC-6 + BC-9.
func TestPathological_AlwaysBusiest_CausesImbalance(t *testing.T) {
	config := newTestDeploymentConfig(3)
	config.RoutingPolicy = "always-busiest"

	cs := NewClusterSimulator(config, newTestWorkload(20), "")
	cs.Run()

	raw := CollectRawMetrics(cs.AggregatedMetrics(), cs.PerInstanceMetrics(), cs.RejectedRequests())

	// With always-busiest routing, all requests should pile onto one instance.
	perInstance := cs.PerInstanceMetrics()
	maxCompleted := 0
	minCompleted := int(^uint(0) >> 1)
	for _, m := range perInstance {
		if m.CompletedRequests > maxCompleted {
			maxCompleted = m.CompletedRequests
		}
		if m.CompletedRequests < minCompleted {
			minCompleted = m.CompletedRequests
		}
	}

	if maxCompleted <= minCompleted && raw.HOLBlockingEvents == 0 {
		t.Logf("maxCompleted=%d, minCompleted=%d, HOL=%d", maxCompleted, minCompleted, raw.HOLBlockingEvents)
		t.Error("expected significant load imbalance or HOL blocking with always-busiest")
	}
}
