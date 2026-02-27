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
			{ID: "realtime", TenantID: "tenant-B", SLOClass: "critical", RateFraction: 0.3,
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
	countCritical := 0
	for _, r := range requests {
		switch r.SLOClass {
		case "batch":
			countBatch++
		case "critical":
			countCritical++
		}
	}

	if countCritical == 0 {
		t.Fatal("critical client produced 0 requests — starvation bug (#278)")
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
			{ID: "reasoning", TenantID: "rsn", SLOClass: "critical", RateFraction: 0.3,
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

func TestGenerateRequests_SameSeed_ProducesIdenticalRequests(t *testing.T) {
	// Blocking prerequisite: Verify GenerateRequests is deterministic
	// (same seed = identical output). Validates INV-6 through sim/workload/ path.
	spec1 := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}

	reqs1, err := GenerateRequests(spec1, 1_000_000, 10)
	if err != nil {
		t.Fatalf("first generation: %v", err)
	}

	spec2 := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}

	reqs2, err := GenerateRequests(spec2, 1_000_000, 10)
	if err != nil {
		t.Fatalf("second generation: %v", err)
	}

	// INV-6: Same seed must produce identical requests
	if len(reqs1) != len(reqs2) {
		t.Fatalf("request count mismatch: %d vs %d", len(reqs1), len(reqs2))
	}
	for i := range reqs1 {
		if reqs1[i].ArrivalTime != reqs2[i].ArrivalTime {
			t.Errorf("request[%d] arrival time mismatch: %d vs %d", i, reqs1[i].ArrivalTime, reqs2[i].ArrivalTime)
		}
		if len(reqs1[i].InputTokens) != len(reqs2[i].InputTokens) {
			t.Errorf("request[%d] input token count mismatch: %d vs %d", i, len(reqs1[i].InputTokens), len(reqs2[i].InputTokens))
		}
		if len(reqs1[i].OutputTokens) != len(reqs2[i].OutputTokens) {
			t.Errorf("request[%d] output token count mismatch: %d vs %d", i, len(reqs1[i].OutputTokens), len(reqs2[i].OutputTokens))
		}
	}
}

func TestGenerateRequests_V1SpecAutoUpgrade_EndToEnd(t *testing.T) {
	// BC-1 end-to-end: v1 spec with deprecated tier names auto-upgrades and generates
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{
			{
				ID: "realtime-client", RateFraction: 0.5, SLOClass: "realtime",
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
			},
			{
				ID: "interactive-client", RateFraction: 0.5, SLOClass: "interactive",
				Model: "llama-3.1-8b",
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 200}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			},
		},
	}

	reqs, err := GenerateRequests(spec, 1_000_000, 20)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqs) == 0 {
		t.Fatal("expected requests to be generated")
	}

	// Verify SLO classes were upgraded
	hasCritical, hasStandard := false, false
	for _, req := range reqs {
		switch req.SLOClass {
		case "critical":
			hasCritical = true
		case "standard":
			hasStandard = true
		case "realtime", "interactive":
			t.Errorf("found deprecated SLO class %q — should have been upgraded", req.SLOClass)
		}
	}
	if !hasCritical {
		t.Error("expected at least one request with SLOClass 'critical'")
	}
	if !hasStandard {
		t.Error("expected at least one request with SLOClass 'standard'")
	}

	// Verify model propagation
	hasModel := false
	for _, req := range reqs {
		if req.Model == "llama-3.1-8b" {
			hasModel = true
			break
		}
	}
	if !hasModel {
		t.Error("expected at least one request with Model 'llama-3.1-8b'")
	}

	// Verify spec was upgraded
	if spec.Version != "2" {
		t.Errorf("spec.Version = %q, want %q", spec.Version, "2")
	}
}

func TestGenerateRequests_ConstantArrival_EvenSpacing(t *testing.T) {
	// BC-3 integration: Constant sampler produces evenly-spaced requests
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0, // 10 req/s
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
		}},
	}

	reqs, err := GenerateRequests(spec, 1_000_000, 5) // 1 second, 5 requests max
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqs) < 2 {
		t.Fatalf("expected at least 2 requests, got %d", len(reqs))
	}

	// Verify even spacing: all IATs should be identical
	expectedIAT := reqs[1].ArrivalTime - reqs[0].ArrivalTime
	for i := 2; i < len(reqs); i++ {
		iat := reqs[i].ArrivalTime - reqs[i-1].ArrivalTime
		if iat != expectedIAT {
			t.Errorf("request[%d] IAT = %d, want %d (constant spacing)", i, iat, expectedIAT)
		}
	}
}

func TestGenerateRequests_V2NewSLOTiers_Generate(t *testing.T) {
	// BC-2: New v2 SLO tiers generate successfully
	tiers := []string{"critical", "standard", "sheddable", "batch", "background"}
	for _, tier := range tiers {
		spec := &WorkloadSpec{
			Version: "2", Seed: 42, AggregateRate: 10.0,
			Clients: []ClientSpec{{
				ID: "c1", RateFraction: 1.0, SLOClass: tier,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
			}},
		}
		reqs, err := GenerateRequests(spec, 1_000_000, 5)
		if err != nil {
			t.Errorf("SLO tier %q: unexpected error: %v", tier, err)
			continue
		}
		for _, req := range reqs {
			if req.SLOClass != tier {
				t.Errorf("SLO tier %q: request has SLOClass %q", tier, req.SLOClass)
			}
		}
	}
}

func TestGenerateRequests_ModelFieldPropagated(t *testing.T) {
	// BC-4: Model field flows from ClientSpec to Request
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", Model: "llama-3.1-8b", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	reqs, err := GenerateRequests(spec, 1_000_000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqs) == 0 {
		t.Fatal("expected at least 1 request")
	}
	for i, req := range reqs {
		if req.Model != "llama-3.1-8b" {
			t.Errorf("request[%d].Model = %q, want %q", i, req.Model, "llama-3.1-8b")
		}
	}
}

func TestGenerateRequests_EmptyModel_DefaultsToEmpty(t *testing.T) {
	// BC-5: Empty model field preserved as empty string
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	reqs, err := GenerateRequests(spec, 1_000_000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, req := range reqs {
		if req.Model != "" {
			t.Errorf("request[%d].Model = %q, want empty string", i, req.Model)
		}
	}
}

func TestGenerateRequests_ZeroAggregateRate_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 0.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		}},
	}
	_, err := GenerateRequests(spec, math.MaxInt64, 0)
	if err == nil {
		t.Fatal("expected error for zero aggregate_rate")
	}
}

func TestGenerateRequests_NaNAggregateRate_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: math.NaN(),
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		}},
	}
	_, err := GenerateRequests(spec, math.MaxInt64, 0)
	if err == nil {
		t.Fatal("expected error for NaN aggregate_rate")
	}
}
