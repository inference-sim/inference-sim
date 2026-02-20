package workload

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestGenerateRequests_SingleClient_ProducesRequests(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "1",
		Seed:          42,
		Category:      "language",
		AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			TenantID:     "t1",
			SLOClass:     "batch",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	horizon := int64(1e6) // 1 second

	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	for i, req := range requests {
		if req.TenantID != "t1" {
			t.Errorf("request %d: TenantID = %q, want %q", i, req.TenantID, "t1")
			break
		}
		if req.SLOClass != "batch" {
			t.Errorf("request %d: SLOClass = %q, want %q", i, req.SLOClass, "batch")
			break
		}
		if len(req.InputTokens) == 0 || len(req.OutputTokens) == 0 {
			t.Errorf("request %d: empty token slices", i)
			break
		}
	}
	// Verify sorted by arrival time
	for i := 1; i < len(requests); i++ {
		if requests[i].ArrivalTime < requests[i-1].ArrivalTime {
			t.Errorf("requests not sorted: [%d].ArrivalTime=%d < [%d].ArrivalTime=%d",
				i, requests[i].ArrivalTime, i-1, requests[i-1].ArrivalTime)
			break
		}
	}
	// Verify sequential IDs
	for i, req := range requests {
		expected := fmt.Sprintf("request_%d", i)
		if req.ID != expected {
			t.Errorf("request %d: ID = %q, want %q", i, req.ID, expected)
			break
		}
	}
}

func TestGenerateRequests_Deterministic_SameSeedSameOutput(t *testing.T) {
	// BC-1: same seed + spec = identical requests
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	horizon := int64(1e6)

	r1, _ := GenerateRequests(spec, horizon, 0)
	r2, _ := GenerateRequests(spec, horizon, 0)

	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
		if len(r1[i].InputTokens) != len(r2[i].InputTokens) {
			t.Errorf("request %d: input len %d vs %d", i, len(r1[i].InputTokens), len(r2[i].InputTokens))
			break
		}
	}
}

func TestGenerateRequests_Deterministic_WithMaxRequests(t *testing.T) {
	// Determinism invariant: same seed + maxRequests > 0 = identical output.
	// This exercises the generate-all-then-truncate path.
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{
			{ID: "a", TenantID: "a", RateFraction: 0.7,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "b", TenantID: "b", RateFraction: 0.3,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
		},
	}
	maxReqs := int64(150)
	r1, err := GenerateRequests(spec, 100e6, maxReqs)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := GenerateRequests(spec, 100e6, maxReqs)
	if err != nil {
		t.Fatal(err)
	}

	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
		if r1[i].TenantID != r2[i].TenantID {
			t.Errorf("request %d: tenant %q vs %q", i, r1[i].TenantID, r2[i].TenantID)
			break
		}
	}
}

func TestGenerateRequests_TwoClients_RateProportional(t *testing.T) {
	// BC-2: client rate fractions produce proportional request counts
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{
			{ID: "a", TenantID: "a", RateFraction: 0.7,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "b", TenantID: "b", RateFraction: 0.3,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
		},
	}
	requests, err := GenerateRequests(spec, 10e6, 0) // 10 seconds
	if err != nil {
		t.Fatal(err)
	}
	countA := 0
	for _, r := range requests {
		if r.TenantID == "a" {
			countA++
		}
	}
	fracA := float64(countA) / float64(len(requests))
	if math.Abs(fracA-0.7) > 0.05 {
		t.Errorf("client A fraction = %.3f, want ≈ 0.7 (within 5%%)", fracA)
	}
}

func TestGenerateRequests_RateFractionNormalization(t *testing.T) {
	// Rate fractions that don't sum to 1.0 should be auto-normalized
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{
			{ID: "a", TenantID: "a", RateFraction: 7.0,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "b", TenantID: "b", RateFraction: 3.0,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
		},
	}
	requests, err := GenerateRequests(spec, 10e6, 0)
	if err != nil {
		t.Fatal(err)
	}
	countA := 0
	for _, r := range requests {
		if r.TenantID == "a" {
			countA++
		}
	}
	fracA := float64(countA) / float64(len(requests))
	if math.Abs(fracA-0.7) > 0.05 {
		t.Errorf("normalized fraction A = %.3f, want ≈ 0.7", fracA)
	}
}

func TestGenerateRequests_ZeroHorizon_ReturnsEmpty(t *testing.T) {
	// EC-5: horizon=0 returns empty
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	requests, err := GenerateRequests(spec, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 0 {
		t.Errorf("expected 0 requests for horizon=0, got %d", len(requests))
	}
}

func TestGenerateRequests_PrefixGroup_SharedPrefix(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 20.0,
		Clients: []ClientSpec{
			{ID: "a", TenantID: "a", RateFraction: 0.5, PrefixGroup: "shared",
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 50, "max": 200}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "b", TenantID: "b", RateFraction: 0.5, PrefixGroup: "shared",
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 50, "max": 200}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
		},
	}
	requests, err := GenerateRequests(spec, 1e6, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) < 2 {
		t.Fatal("need at least 2 requests")
	}

	// All requests in same prefix group should share the same prefix tokens
	// (first N tokens should be identical)
	prefixLen := 50 // default prefix length for shared groups
	first := requests[0].InputTokens
	for i := 1; i < len(requests); i++ {
		other := requests[i].InputTokens
		if len(first) < prefixLen || len(other) < prefixLen {
			continue
		}
		for j := 0; j < prefixLen; j++ {
			if first[j] != other[j] {
				t.Errorf("request %d: prefix token %d differs: %d vs %d", i, j, first[j], other[j])
				break
			}
		}
		break // just check first pair
	}
}

func TestGenerateRequests_MaxRequests_CapsOutput(t *testing.T) {
	// BC-1, BC-6: maxRequests caps total output even with long horizon
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	horizon := int64(100e6) // 100 seconds — would produce ~10000 requests
	maxReqs := int64(50)

	requests, err := GenerateRequests(spec, horizon, maxReqs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if int64(len(requests)) > maxReqs {
		t.Errorf("len(requests) = %d, want <= %d", len(requests), maxReqs)
	}
	if len(requests) == 0 {
		t.Error("expected at least some requests")
	}
}

func TestGenerateRequests_ZeroMaxRequests_UsesHorizonOnly(t *testing.T) {
	// BC-2: maxRequests=0 means unlimited — horizon controls
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	horizon := int64(1e6) // 1 second

	requests, err := GenerateRequests(spec, horizon, 0) // 0 = unlimited
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Error("expected requests with unlimited maxRequests and finite horizon")
	}
}

func TestGenerateRequests_MaxRequests_PreservesClientProportions(t *testing.T) {
	// BC-3: With maxRequests cap, both clients must appear in proportional amounts.
	// Bug #278: Sequential generation starves later clients.
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{
			{ID: "batch", TenantID: "tenant-A", SLOClass: "batch", RateFraction: 0.7,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "realtime", TenantID: "tenant-B", SLOClass: "realtime", RateFraction: 0.3,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
		},
	}
	maxReqs := int64(200)
	requests, err := GenerateRequests(spec, 100e6, maxReqs) // long horizon, capped by maxReqs
	if err != nil {
		t.Fatal(err)
	}

	// BC-4: total output must be capped
	if int64(len(requests)) != maxReqs {
		t.Errorf("len(requests) = %d, want %d", len(requests), maxReqs)
	}

	// BC-3: both SLO classes must appear
	countBatch := 0
	countRealtime := 0
	for _, r := range requests {
		switch r.SLOClass {
		case "batch":
			countBatch++
		case "realtime":
			countRealtime++
		}
	}

	if countRealtime == 0 {
		t.Fatal("realtime client produced 0 requests — starvation bug (#278)")
	}

	// Check proportions are approximately 70/30 (within ±10%)
	fracBatch := float64(countBatch) / float64(len(requests))
	if fracBatch < 0.6 || fracBatch > 0.8 {
		t.Errorf("batch fraction = %.3f, want ≈ 0.7 (±10%%)", fracBatch)
	}
}

func TestGenerateRequests_MaxRequests_ReasoningClientNotStarved(t *testing.T) {
	// BC-7: Reasoning (multi-turn) clients must not be starved by maxRequests cap.
	reasoningSpec := &ReasoningSpec{
		ReasonRatioDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 0.3, "std_dev": 0.1, "min": 0.1, "max": 0.9}},
		MultiTurn:       &MultiTurnSpec{MaxRounds: 2, ContextGrowth: "accumulate"},
	}
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{
			{ID: "standard", TenantID: "std", SLOClass: "batch", RateFraction: 0.7,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "reasoning", TenantID: "rsn", SLOClass: "realtime", RateFraction: 0.3,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
				Reasoning:  reasoningSpec},
		},
	}
	maxReqs := int64(100)
	requests, err := GenerateRequests(spec, 100e6, maxReqs)
	if err != nil {
		t.Fatal(err)
	}

	// Total capped
	if int64(len(requests)) > maxReqs {
		t.Errorf("len(requests) = %d, want <= %d", len(requests), maxReqs)
	}

	// Reasoning client must appear
	countReasoning := 0
	for _, r := range requests {
		if r.TenantID == "rsn" {
			countReasoning++
		}
	}
	if countReasoning == 0 {
		t.Fatal("reasoning client produced 0 requests — starvation bug")
	}
}

func TestRequestNewFields_ZeroValueDefault(t *testing.T) {
	req := &sim.Request{ID: "test", State: sim.StateQueued}
	if req.TenantID != "" || req.SLOClass != "" || req.SessionID != "" {
		t.Error("new fields should have zero-value defaults")
	}
	if req.Streaming || req.RoundIndex != 0 || req.ReasonRatio != 0 {
		t.Error("new bool/int/float fields should have zero-value defaults")
	}
}
