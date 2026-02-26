package cluster

import (
	"bytes"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
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
	raw := CollectRawMetrics(m, nil, 0, "")

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
	raw := CollectRawMetrics(m, nil, 0, "")
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
	raw := CollectRawMetrics(m, nil, 42, "")
	if raw.RejectedRequests != 42 {
		t.Errorf("RejectedRequests: got %d, want 42", raw.RejectedRequests)
	}
}

// TestCollectRawMetrics_DroppedUnservable verifies dropped count is captured.
func TestCollectRawMetrics_DroppedUnservable(t *testing.T) {
	// GIVEN aggregated metrics with dropped requests
	m := sim.NewMetrics()
	m.DroppedUnservable = 3

	// WHEN collecting raw metrics
	raw := CollectRawMetrics(m, nil, 0, "")

	// THEN DroppedUnservable is captured
	if raw.DroppedUnservable != 3 {
		t.Errorf("DroppedUnservable: got %d, want 3", raw.DroppedUnservable)
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
	// Throughput component should be in (0, 1) range (normalized)
	if result.Components["throughput"] <= 0 || result.Components["throughput"] >= 1.0 {
		t.Errorf("throughput component: got %f, expected in (0, 1)", result.Components["throughput"])
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
	// Low latency score should be in a reasonable normalized range (0, 1]
	if lowResult.Score <= 0 || lowResult.Score > 1.0 {
		t.Errorf("1ms latency score: got %f, expected in (0, 1]", lowResult.Score)
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

// TestParseFitnessWeights_InvalidValues_ReturnsError verifies BC-1, BC-2, BC-3.
func TestParseFitnessWeights_InvalidValues_ReturnsError(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"NaN value", "throughput:NaN"},
		{"positive Inf", "throughput:Inf"},
		{"negative Inf", "throughput:-Inf"},
		{"explicit +Inf", "throughput:+Inf"},
		{"negative weight", "throughput:-0.5"},
		{"negative one", "p99_ttft:-1"},
		{"NaN after valid", "throughput:0.5,p99_ttft:NaN"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseFitnessWeights(tt.input)
			if err == nil {
				t.Errorf("expected error for input %q, got nil", tt.input)
			}
		})
	}
}

// TestParseFitnessWeights_ZeroWeight_Accepted verifies BC-4.
func TestParseFitnessWeights_ZeroWeight_Accepted(t *testing.T) {
	weights, err := ParseFitnessWeights("throughput:0,p99_ttft:0.3")
	if err != nil {
		t.Fatalf("zero weight should be accepted, got error: %v", err)
	}
	if weights["throughput"] != 0.0 {
		t.Errorf("throughput: got %f, want 0.0", weights["throughput"])
	}
	if weights["p99_ttft"] != 0.3 {
		t.Errorf("p99_ttft: got %f, want 0.3", weights["p99_ttft"])
	}
}

// TestCollectRawMetrics_ConstantPriority_SuppressesInversions verifies that
// priority inversion counter returns 0 when using constant priority policy,
// since there are no meaningful priorities to invert.
func TestCollectRawMetrics_ConstantPriority_SuppressesInversions(t *testing.T) {
	// GIVEN per-instance metrics with requests that would normally trigger inversions
	m := sim.NewMetrics()
	m.Requests["early"] = sim.RequestMetrics{ID: "early", ArrivedAt: 100}
	m.RequestE2Es["early"] = 50000.0 // 10× slower than "late"
	m.Requests["late"] = sim.RequestMetrics{ID: "late", ArrivedAt: 200}
	m.RequestE2Es["late"] = 5000.0

	aggregated := sim.NewMetrics()
	aggregated.CompletedRequests = 2
	aggregated.SimEndedTime = 1_000_000

	// WHEN collecting with constant priority policy
	raw := CollectRawMetrics(aggregated, []*sim.Metrics{m}, 0, "constant")

	// THEN priority inversions should be suppressed
	if raw.PriorityInversions != 0 {
		t.Errorf("expected 0 priority inversions with constant policy, got %d", raw.PriorityInversions)
	}
}

// TestCollectRawMetrics_SLOBasedPriority_DetectsInversions verifies that
// priority inversion counter still works for non-constant priority policies.
func TestCollectRawMetrics_SLOBasedPriority_DetectsInversions(t *testing.T) {
	// GIVEN per-instance metrics with requests that would trigger inversions
	m := sim.NewMetrics()
	m.Requests["early"] = sim.RequestMetrics{ID: "early", ArrivedAt: 100}
	m.RequestE2Es["early"] = 50000.0
	m.Requests["late"] = sim.RequestMetrics{ID: "late", ArrivedAt: 200}
	m.RequestE2Es["late"] = 5000.0

	aggregated := sim.NewMetrics()
	aggregated.CompletedRequests = 2
	aggregated.SimEndedTime = 1_000_000

	// WHEN collecting with slo-based priority policy
	raw := CollectRawMetrics(aggregated, []*sim.Metrics{m}, 0, "slo-based")

	// THEN priority inversions should be detected
	if raw.PriorityInversions == 0 {
		t.Error("expected priority inversions > 0 with slo-based policy")
	}
}

// TestDetectPriorityInversions_InvertedRequests verifies BC-8.
func TestDetectPriorityInversions_InvertedRequests(t *testing.T) {
	m := sim.NewMetrics()
	m.Requests["high"] = sim.RequestMetrics{ID: "high", ArrivedAt: 100}
	m.RequestE2Es["high"] = 50000.0
	m.Requests["low"] = sim.RequestMetrics{ID: "low", ArrivedAt: 200}
	m.RequestE2Es["low"] = 5000.0

	inversions := detectPriorityInversions([]*sim.Metrics{m}, "slo-based")

	if inversions < 0 {
		t.Errorf("inversions should be >= 0, got %d", inversions)
	}
	// Earlier request (high) has 10× worse E2E than later request (low) → inversion detected
	if inversions == 0 {
		t.Error("expected at least 1 inversion for 10× E2E difference")
	}
}

// TestDetectPriorityInversions_MissingE2E_WarnsAndCountsMatched verifies BC-1.
// Requests with no E2E entry are skipped with a warning, but matched requests
// are still evaluated for inversions.
func TestDetectPriorityInversions_MissingE2E_WarnsAndCountsMatched(t *testing.T) {
	// GIVEN an instance with 3 requests but only 2 have E2E data
	m := sim.NewMetrics()
	m.Requests["r1"] = sim.RequestMetrics{ID: "r1", ArrivedAt: 100}
	m.Requests["r2"] = sim.RequestMetrics{ID: "r2", ArrivedAt: 200}
	m.Requests["r3"] = sim.RequestMetrics{ID: "r3", ArrivedAt: 300}
	// r1 has 10× worse E2E than r2 → inversion
	m.RequestE2Es["r1"] = 50000.0
	m.RequestE2Es["r2"] = 5000.0
	// r3 has NO E2E entry → should be skipped with warning

	// Capture log output
	var buf bytes.Buffer
	logrus.SetOutput(&buf)
	defer logrus.SetOutput(os.Stderr)

	// WHEN detecting priority inversions
	inversions := detectPriorityInversions([]*sim.Metrics{m}, "slo-based")

	// THEN inversions are counted from matched requests (r1 vs r2)
	assert.GreaterOrEqual(t, inversions, 1, "should detect inversion between r1 and r2")

	// AND a warning was logged about the skipped request
	assert.Contains(t, buf.String(), "missing E2E", "should warn about requests with missing E2E data")
}

// TestDetectPriorityInversions_MixedSLO_NoFalsePositives verifies BC-4 (#292):
// Mixed-SLO workloads must not produce false positives from cross-class comparisons.
func TestDetectPriorityInversions_MixedSLO_NoFalsePositives(t *testing.T) {
	// GIVEN requests from two SLO classes with naturally different E2E
	m := sim.NewMetrics()
	// Critical requests: fast (low E2E)
	m.Requests["rt1"] = sim.RequestMetrics{ID: "rt1", ArrivedAt: 100, SLOClass: "critical"}
	m.RequestE2Es["rt1"] = 5000.0
	m.Requests["rt2"] = sim.RequestMetrics{ID: "rt2", ArrivedAt: 300, SLOClass: "critical"}
	m.RequestE2Es["rt2"] = 4500.0
	// Batch requests: slow (high E2E) — this is expected, not an inversion
	m.Requests["b1"] = sim.RequestMetrics{ID: "b1", ArrivedAt: 200, SLOClass: "batch"}
	m.RequestE2Es["b1"] = 50000.0
	m.Requests["b2"] = sim.RequestMetrics{ID: "b2", ArrivedAt: 400, SLOClass: "batch"}
	m.RequestE2Es["b2"] = 48000.0

	// WHEN detecting with slo-based priority
	inversions := detectPriorityInversions([]*sim.Metrics{m}, "slo-based")

	// THEN no inversions (within each class, requests are ordered correctly)
	if inversions != 0 {
		t.Errorf("expected 0 inversions for correctly-ordered mixed-SLO workload, got %d", inversions)
	}
}

// TestDetectPriorityInversions_WithinSLOClass_StillDetected verifies BC-7:
// Inversions within a single SLO class must still be detected.
func TestDetectPriorityInversions_WithinSLOClass_StillDetected(t *testing.T) {
	m := sim.NewMetrics()
	// Two critical requests where earlier one has much worse E2E
	m.Requests["rt1"] = sim.RequestMetrics{ID: "rt1", ArrivedAt: 100, SLOClass: "critical"}
	m.RequestE2Es["rt1"] = 50000.0 // 10× worse than rt2
	m.Requests["rt2"] = sim.RequestMetrics{ID: "rt2", ArrivedAt: 200, SLOClass: "critical"}
	m.RequestE2Es["rt2"] = 5000.0

	inversions := detectPriorityInversions([]*sim.Metrics{m}, "slo-based")

	if inversions == 0 {
		t.Error("expected at least 1 inversion within the same SLO class")
	}
}

// TestDetectPriorityInversions_EmptySLOClass_UsesDefault verifies BC-7:
// Legacy workloads with empty SLOClass are grouped as "default" and
// existing detection behavior is preserved.
func TestDetectPriorityInversions_EmptySLOClass_UsesDefault(t *testing.T) {
	m := sim.NewMetrics()
	// Legacy requests with no SLO class (empty string)
	m.Requests["r1"] = sim.RequestMetrics{ID: "r1", ArrivedAt: 100}
	m.RequestE2Es["r1"] = 50000.0
	m.Requests["r2"] = sim.RequestMetrics{ID: "r2", ArrivedAt: 200}
	m.RequestE2Es["r2"] = 5000.0

	inversions := detectPriorityInversions([]*sim.Metrics{m}, "slo-based")

	if inversions == 0 {
		t.Error("expected inversion detected for legacy (empty SLO class) requests")
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

// TestDetectHOLBlocking_AllTrafficOneInstance_Detected verifies BC-3 (#291):
// When all traffic goes to a single instance, HOL blocking MUST be detected.
func TestDetectHOLBlocking_AllTrafficOneInstance_Detected(t *testing.T) {
	// GIVEN 4 instances where only instance 0 has traffic
	perInstance := []*sim.Metrics{
		makeMetricsWithQueueDepth([]int{50, 50, 50, 50}), // instance 0: all traffic
		makeMetricsWithQueueDepth([]int{}),                // instance 1: no traffic
		makeMetricsWithQueueDepth([]int{}),                // instance 2: no traffic
		makeMetricsWithQueueDepth([]int{}),                // instance 3: no traffic
	}

	// WHEN detecting HOL blocking
	blocking := detectHOLBlocking(perInstance)

	// THEN HOL blocking MUST be detected (this is the most extreme case)
	if blocking == 0 {
		t.Error("expected HOL blocking > 0 when all traffic goes to one instance, got 0")
	}
}

// TestDetectHOLBlocking_PartialConcentration_Detected verifies the fix
// handles partial concentration (2 active + 2 idle) correctly.
func TestDetectHOLBlocking_PartialConcentration_Detected(t *testing.T) {
	// GIVEN 4 instances with partial concentration
	perInstance := []*sim.Metrics{
		makeMetricsWithQueueDepth([]int{40, 40, 40}), // instance 0: heavy traffic
		makeMetricsWithQueueDepth([]int{5, 5, 5}),    // instance 1: light traffic
		makeMetricsWithQueueDepth([]int{}),            // instance 2: no traffic
		makeMetricsWithQueueDepth([]int{}),            // instance 3: no traffic
	}

	// WHEN detecting HOL blocking
	blocking := detectHOLBlocking(perInstance)

	// THEN HOL blocking should be detected
	// Mean is (40+5+0+0)/4 = 11.25; instance 0 avg=40 > 2*11.25=22.5
	if blocking == 0 {
		t.Error("expected HOL blocking > 0 for partial concentration, got 0")
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

	cs := NewClusterSimulator(config, newTestRequests(20))
	if err := cs.Run(); err != nil {
		t.Fatalf("cs.Run: %v", err)
	}

	raw := CollectRawMetrics(cs.AggregatedMetrics(), cs.PerInstanceMetrics(), cs.RejectedRequests(), "")

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

	cs := NewClusterSimulator(config, newTestRequests(20))
	if err := cs.Run(); err != nil {
		t.Fatalf("cs.Run: %v", err)
	}

	raw := CollectRawMetrics(cs.AggregatedMetrics(), cs.PerInstanceMetrics(), cs.RejectedRequests(), "")

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

// TestJainFairnessIndex_AllZeroThroughputs_ReturnsPerfectFairness verifies BC-13:
// all-zero throughputs means all tenants treated identically → perfectly fair.
func TestJainFairnessIndex_AllZeroThroughputs_ReturnsPerfectFairness(t *testing.T) {
	throughputs := map[string]float64{"t1": 0, "t2": 0, "t3": 0}
	jfi := JainFairnessIndex(throughputs)
	if jfi != 1.0 {
		t.Errorf("JainFairnessIndex(all-zero) = %f, want 1.0 (perfectly fair)", jfi)
	}
}

// TestJainFairnessIndex_Empty_ReturnsZero verifies edge case: no tenants.
func TestJainFairnessIndex_Empty_ReturnsZero(t *testing.T) {
	jfi := JainFairnessIndex(map[string]float64{})
	if jfi != 0 {
		t.Errorf("JainFairnessIndex(empty) = %f, want 0", jfi)
	}
}
