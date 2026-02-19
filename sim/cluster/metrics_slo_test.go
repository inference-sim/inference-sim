package cluster

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestComputePerSLODistributions_SegregatesCorrectly(t *testing.T) {
	m := sim.NewMetrics()
	// Add requests with different SLO classes
	for i := 0; i < 50; i++ {
		id := fmt.Sprintf("rt_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "realtime"}
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

	if sloDistributions["realtime"] == nil {
		t.Fatal("expected realtime class")
	}
	if sloDistributions["batch"] == nil {
		t.Fatal("expected batch class")
	}
	if sloDistributions["realtime"].TTFT.Count != 50 {
		t.Errorf("realtime TTFT count = %d, want 50", sloDistributions["realtime"].TTFT.Count)
	}
	if sloDistributions["batch"].TTFT.Count != 150 {
		t.Errorf("batch TTFT count = %d, want 150", sloDistributions["batch"].TTFT.Count)
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
	// Extreme unfairness: one tenant gets everything
	throughputs := map[string]float64{"t1": 1000, "t2": 1, "t3": 1}
	jfi := JainFairnessIndex(throughputs)
	// Should be close to 1/3 (very unfair)
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

func TestSLOAttainment_AllMeet_ReturnsOne(t *testing.T) {
	m := sim.NewMetrics()
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "batch"}
		m.RequestE2Es[id] = 500 // well under 1000 target
	}
	targets := map[string]float64{"batch": 1000}
	attainment := SLOAttainment(m, targets)
	if attainment != 1.0 {
		t.Errorf("attainment = %f, want 1.0 when all meet SLO", attainment)
	}
}

func TestSLOAttainment_SomeMiss_FractionalResult(t *testing.T) {
	m := sim.NewMetrics()
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "realtime"}
		if i < 7 {
			m.RequestE2Es[id] = 100 // meets 200 target
		} else {
			m.RequestE2Es[id] = 300 // exceeds 200 target
		}
	}
	targets := map[string]float64{"realtime": 200}
	attainment := SLOAttainment(m, targets)
	if math.Abs(attainment-0.7) > 0.01 {
		t.Errorf("attainment = %f, want 0.7 (7/10 meet SLO)", attainment)
	}
}

func TestComputePerSLODistributions_MissingRequests_StillComputesPresent(t *testing.T) {
	// GIVEN metrics with 10 TTFT entries and 10 E2E entries,
	// but only 7 have corresponding entries in the Requests map
	m := sim.NewMetrics()
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.RequestTTFTs[id] = float64(100 + i)
		m.RequestE2Es[id] = float64(500 + i)
	}
	// Only 7 out of 10 have Requests entries
	for i := 0; i < 7; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "batch"}
	}

	// WHEN computing per-SLO distributions
	result := ComputePerSLODistributions(m)

	// THEN distributions are computed for the 7 present requests
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

func TestSLOAttainment_MissingRequests_CountedAsViolation(t *testing.T) {
	// GIVEN 10 requests in RequestE2Es but only 7 in Requests map
	m := sim.NewMetrics()
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.RequestE2Es[id] = 100 // all would meet SLO
	}
	// Only register 7 in Requests map
	for i := 0; i < 7; i++ {
		id := fmt.Sprintf("req_%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, SLOClass: "batch"}
	}
	targets := map[string]float64{"batch": 200}

	// WHEN computing SLO attainment
	attainment := SLOAttainment(m, targets)

	// THEN attainment should be 7/10 = 0.7 (dropped requests are violations)
	if math.Abs(attainment-0.7) > 0.01 {
		t.Errorf("attainment = %f, want 0.7 (3 missing requests should count as violations)", attainment)
	}
}
