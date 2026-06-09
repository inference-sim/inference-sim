package cluster

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

func TestComputePerSLODistributions_SegregatesCorrectly(t *testing.T) {
	m := sim.NewMetrics()
	for i := 0; i < 50; i++ {
		id := fmt.Sprintf("rt_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "critical"}
		m.RequestTTFTs[id] = float64(100 + i)
		m.RequestE2Es[id] = float64(500 + i)
	}
	for i := 0; i < 150; i++ {
		id := fmt.Sprintf("batch_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "batch"}
		m.RequestTTFTs[id] = float64(200 + i)
		m.RequestE2Es[id] = float64(1000 + i)
	}

	sloDistributions := ComputePerSLODistributions(m)

	if sloDistributions["critical"] == nil {
		t.Fatal("expected critical class")
	}
	if sloDistributions["batch"] == nil {
		t.Fatal("expected batch class")
	}
	if sloDistributions["critical"].TTFT.Count != 50 {
		t.Errorf("critical TTFT count = %d, want 50", sloDistributions["critical"].TTFT.Count)
	}
	if sloDistributions["batch"].TTFT.Count != 150 {
		t.Errorf("batch TTFT count = %d, want 150", sloDistributions["batch"].TTFT.Count)
	}
}

func TestComputePerSLODistributions_PopulatesITL(t *testing.T) {
	// BC-7: ITL distribution populated alongside TTFT/E2E.
	m := sim.NewMetrics()
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "critical"}
		m.RequestTTFTs[id] = 100
		m.RequestE2Es[id] = 1000
		m.RequestITLs[id] = float64(50 + i)
	}
	result := ComputePerSLODistributions(m)
	if result["critical"] == nil {
		t.Fatal("expected critical class")
	}
	if result["critical"].ITL.Count != 5 {
		t.Errorf("ITL.Count = %d, want 5", result["critical"].ITL.Count)
	}
	if result["critical"].ITL.Min != 50 || result["critical"].ITL.Max != 54 {
		t.Errorf("ITL min/max = %f/%f, want 50/54", result["critical"].ITL.Min, result["critical"].ITL.Max)
	}
}

func TestJainFairnessIndex_EqualThroughput_ReturnsOne(t *testing.T) {
	throughputs := map[string]float64{"t1": 100, "t2": 100, "t3": 100}
	jfi := JainFairnessIndex(throughputs)
	if math.Abs(jfi-1.0) > 0.001 {
		t.Errorf("JFI = %f, want 1.0 for equal throughputs", jfi)
	}
}

func TestJainFairnessIndex_UnequalThroughput_LessThanOne(t *testing.T) {
	throughputs := map[string]float64{"t1": 1000, "t2": 1, "t3": 1}
	jfi := JainFairnessIndex(throughputs)
	if jfi > 0.5 {
		t.Errorf("JFI = %f, expected < 0.5 for very unequal throughputs", jfi)
	}
}

func TestJainFairnessIndex_EmptyMap_ReturnsZero(t *testing.T) {
	jfi := JainFairnessIndex(map[string]float64{})
	if jfi != 0 {
		t.Errorf("JFI = %f, want 0 for empty map", jfi)
	}
}

func TestComputePerSLODistributions_MissingRequests_StillComputesPresent(t *testing.T) {
	m := sim.NewMetrics()
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.RequestTTFTs[id] = float64(100 + i)
		m.RequestE2Es[id] = float64(500 + i)
	}
	for i := 0; i < 7; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "batch"}
	}

	result := ComputePerSLODistributions(m)

	if result["batch"] == nil {
		t.Fatal("expected batch class in result")
	}
	if result["batch"].TTFT.Count != 7 {
		t.Errorf("TTFT count = %d, want 7 (only present requests)", result["batch"].TTFT.Count)
	}
	if result["batch"].E2E.Count != 7 {
		t.Errorf("E2E count = %d, want 7 (only present requests)", result["batch"].E2E.Count)
	}
}

// --- SLOAttainmentMultiDim tests (BC-1..4, BC-N2) -------------------------

// makeMetrics builds a sim.Metrics with the given per-class requests and latencies.
// Each entry: class string, count int, ttftMs, itlMs, e2eMs float64 (per request).
func makeMetricsForSLO(t *testing.T, entries []struct {
	class                  string
	count                  int
	ttftMs, itlMs, e2eMs   float64
}) *sim.Metrics {
	t.Helper()
	m := sim.NewMetrics()
	idx := 0
	for _, e := range entries {
		for i := 0; i < e.count; i++ {
			id := fmt.Sprintf("%s_%d", e.class, idx)
			idx++
			m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: e.class}
			m.RequestTTFTs[id] = e.ttftMs
			m.RequestITLs[id] = e.itlMs
			m.RequestE2Es[id] = e.e2eMs
		}
	}
	return m
}

func TestSLOAttainmentMultiDim_AllDimsMet_ReturnsPerfectScore(t *testing.T) {
	// BC-1: all configured dimensions met => attainment = 1.0
	m := makeMetricsForSLO(t, []struct {
		class                string
		count                int
		ttftMs, itlMs, e2eMs float64
	}{
		{"critical", 10, 50, 30, 2000},
	})
	results := BuildLatencyResults(m)
	injected := map[string]int64{"critical": 10}
	targets := map[string]workload.SLODimTargets{
		"critical": {TTFTMs: 100, ITLMs: 50, E2EMs: 5000},
	}
	overall, perClass := SLOAttainmentMultiDim(results, injected, targets)
	if overall != 1.0 {
		t.Errorf("overall = %f, want 1.0", overall)
	}
	if perClass["critical"].Good != 10 {
		t.Errorf("Good = %d, want 10", perClass["critical"].Good)
	}
	if perClass["critical"].Injected != 10 {
		t.Errorf("Injected = %d, want 10", perClass["critical"].Injected)
	}
	for _, dim := range []string{"ttft", "itl", "e2e"} {
		if got := perClass["critical"].ByDim[dim]; got != 1.0 {
			t.Errorf("ByDim[%s] = %f, want 1.0", dim, got)
		}
	}
}

func TestSLOAttainmentMultiDim_PartialDims_OnlyConfiguredGate(t *testing.T) {
	// BC-3: ITLMs == 0 means ITL is not gated. Request with high ITL is still good.
	m := makeMetricsForSLO(t, []struct {
		class                string
		count                int
		ttftMs, itlMs, e2eMs float64
	}{
		{"batch", 5, 80, 9999, 4000},
	})
	results := BuildLatencyResults(m)
	injected := map[string]int64{"batch": 5}
	targets := map[string]workload.SLODimTargets{
		"batch": {TTFTMs: 100, E2EMs: 5000}, // ITLMs=0 => not gated
	}
	overall, perClass := SLOAttainmentMultiDim(results, injected, targets)
	if overall != 1.0 {
		t.Errorf("overall = %f, want 1.0 (ITL not gated)", overall)
	}
	if _, present := perClass["batch"].ByDim["itl"]; present {
		t.Errorf("ByDim should not contain 'itl' when ITLMs == 0")
	}
}

func TestSLOAttainmentMultiDim_AndCombination_AllDimsMustPass(t *testing.T) {
	// BC-4: a request good only if every non-zero dim passes.
	// 10 requests in class A: 5 with all dims passing, 5 fail TTFT only.
	m := sim.NewMetrics()
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("good_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "A"}
		m.RequestTTFTs[id] = 50
		m.RequestITLs[id] = 30
		m.RequestE2Es[id] = 2000
	}
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("bad_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "A"}
		m.RequestTTFTs[id] = 999 // exceeds threshold
		m.RequestITLs[id] = 30
		m.RequestE2Es[id] = 2000
	}
	results := BuildLatencyResults(m)
	injected := map[string]int64{"A": 10}
	targets := map[string]workload.SLODimTargets{
		"A": {TTFTMs: 100, ITLMs: 50, E2EMs: 5000},
	}
	overall, perClass := SLOAttainmentMultiDim(results, injected, targets)
	if math.Abs(overall-0.5) > 1e-9 {
		t.Errorf("overall = %f, want 0.5", overall)
	}
	if perClass["A"].Good != 5 {
		t.Errorf("Good = %d, want 5", perClass["A"].Good)
	}
	if got := perClass["A"].ByDim["ttft"]; math.Abs(got-0.5) > 1e-9 {
		t.Errorf("ByDim[ttft] = %f, want 0.5", got)
	}
	if got := perClass["A"].ByDim["itl"]; got != 1.0 {
		t.Errorf("ByDim[itl] = %f, want 1.0", got)
	}
	if got := perClass["A"].ByDim["e2e"]; got != 1.0 {
		t.Errorf("ByDim[e2e] = %f, want 1.0", got)
	}
}

func TestSLOAttainmentMultiDim_UnconfiguredClass_ExcludedFromDenominator(t *testing.T) {
	// BC-N2: unconfigured class contributes neither numerator nor denominator.
	m := makeMetricsForSLO(t, []struct {
		class                string
		count                int
		ttftMs, itlMs, e2eMs float64
	}{
		{"critical", 10, 50, 30, 2000},
		{"free", 100, 9999, 9999, 99999}, // would all fail if gated
	})
	results := BuildLatencyResults(m)
	injected := map[string]int64{"critical": 10, "free": 100}
	targets := map[string]workload.SLODimTargets{
		"critical": {TTFTMs: 100, E2EMs: 5000},
	}
	overall, perClass := SLOAttainmentMultiDim(results, injected, targets)
	if overall != 1.0 {
		t.Errorf("overall = %f, want 1.0 (free class excluded)", overall)
	}
	if _, ok := perClass["free"]; ok {
		t.Errorf("perClass must not contain unconfigured class 'free'")
	}
	if perClass["critical"].Injected != 10 {
		t.Errorf("Injected for critical = %d, want 10 (denominator must not include free)", perClass["critical"].Injected)
	}
}

func TestSLOAttainmentMultiDim_EmptyClassMapsToDefault(t *testing.T) {
	// BC-2: SLOClass == "" maps to "default" target.
	m := makeMetricsForSLO(t, []struct {
		class                string
		count                int
		ttftMs, itlMs, e2eMs float64
	}{
		{"", 8, 50, 30, 2000},
	})
	results := BuildLatencyResults(m)
	injected := map[string]int64{"": 8}
	targets := map[string]workload.SLODimTargets{
		"default": {TTFTMs: 100, E2EMs: 5000},
	}
	overall, perClass := SLOAttainmentMultiDim(results, injected, targets)
	if overall != 1.0 {
		t.Errorf("overall = %f, want 1.0", overall)
	}
	if perClass["default"].Good != 8 {
		t.Errorf("Good for 'default' = %d, want 8", perClass["default"].Good)
	}
}

func TestSLOAttainmentMultiDim_DenominatorIsInjected_NotCompleted(t *testing.T) {
	// BC-1 (denominator): saturation drops count as violations.
	// 10 injected, 7 completed and meeting all dims => attainment = 7/10 = 0.7.
	m := makeMetricsForSLO(t, []struct {
		class                string
		count                int
		ttftMs, itlMs, e2eMs float64
	}{
		{"critical", 7, 50, 30, 2000},
	})
	results := BuildLatencyResults(m)
	injected := map[string]int64{"critical": 10} // 3 dropped before completion
	targets := map[string]workload.SLODimTargets{
		"critical": {TTFTMs: 100, E2EMs: 5000},
	}
	overall, perClass := SLOAttainmentMultiDim(results, injected, targets)
	if math.Abs(overall-0.7) > 1e-9 {
		t.Errorf("overall = %f, want 0.7", overall)
	}
	if perClass["critical"].Good != 7 {
		t.Errorf("Good = %d, want 7", perClass["critical"].Good)
	}
	if perClass["critical"].Injected != 10 {
		t.Errorf("Injected = %d, want 10", perClass["critical"].Injected)
	}
}

func TestSLOAttainmentMultiDim_DeterministicAcrossRuns(t *testing.T) {
	// INV-6: bit-identical output across iterations (R2 sanity).
	m := makeMetricsForSLO(t, []struct {
		class                string
		count                int
		ttftMs, itlMs, e2eMs float64
	}{
		{"A", 50, 50, 30, 2000},
		{"B", 75, 80, 40, 3000},
		{"C", 25, 90, 100, 4500},
	})
	results := BuildLatencyResults(m)
	injected := map[string]int64{"A": 50, "B": 100, "C": 30}
	targets := map[string]workload.SLODimTargets{
		"A": {TTFTMs: 100, ITLMs: 50, E2EMs: 5000},
		"B": {TTFTMs: 100, E2EMs: 5000},
		"C": {E2EMs: 5000},
	}
	first, _ := SLOAttainmentMultiDim(results, injected, targets)
	for i := 0; i < 100; i++ {
		got, _ := SLOAttainmentMultiDim(results, injected, targets)
		if got != first {
			t.Fatalf("iter %d: overall = %v, want %v (non-deterministic)", i, got, first)
		}
	}
}

func TestSLOAttainmentMultiDim_NoConfiguredClasses_ReturnsZero(t *testing.T) {
	// Edge: empty targets => 0 overall, empty perClass.
	m := makeMetricsForSLO(t, []struct {
		class                string
		count                int
		ttftMs, itlMs, e2eMs float64
	}{
		{"A", 10, 50, 30, 2000},
	})
	results := BuildLatencyResults(m)
	injected := map[string]int64{"A": 10}
	overall, perClass := SLOAttainmentMultiDim(results, injected, map[string]workload.SLODimTargets{})
	if overall != 0 {
		t.Errorf("overall = %f, want 0 (no targets)", overall)
	}
	if len(perClass) != 0 {
		t.Errorf("perClass = %v, want empty", perClass)
	}
}
