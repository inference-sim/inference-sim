package workload

import (
	"fmt"
	"math"
	"strings"
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
	if req.RoundIndex != 0 || req.ReasonRatio != 0 {
		t.Error("new int/float fields should have zero-value defaults")
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

func TestGenerateRequests_SingleSession_OneSessionPerClient(t *testing.T) {
	// BC-2: SingleSession mode generates exactly one session per client.
	// 3 clients, each with MaxRounds=10 and ThinkTimeUs=100_000.
	var clients []ClientSpec
	for i := 0; i < 3; i++ {
		clients = append(clients, ClientSpec{
			ID: fmt.Sprintf("client-%d", i), TenantID: fmt.Sprintf("t%d", i),
			SLOClass: "batch", RateFraction: 1.0 / 3.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     10,
					ThinkTimeUs:   100_000,
					ContextGrowth: "accumulate",
					SingleSession: true,
				},
			},
		})
	}
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 30.0,
		Clients: clients,
	}
	horizon := int64(2_000_000) // 2 seconds
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}

	// Each client (by TenantID) should have at most MaxRounds requests
	byTenant := make(map[string]int)
	for _, req := range requests {
		byTenant[req.TenantID]++
	}
	for tenant, count := range byTenant {
		if count > 10 {
			t.Errorf("tenant %q: %d requests, want <= 10 (MaxRounds)", tenant, count)
		}
	}
}

func TestGenerateRequests_SingleSession_HorizonTruncation(t *testing.T) {
	// BC-7: Rounds beyond horizon are excluded.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     100,
					ThinkTimeUs:   100_000, // 100 rounds × 100ms = 10s total session
					ContextGrowth: "accumulate",
					SingleSession: true,
				},
			},
		}},
	}
	horizon := int64(1_000_000) // 1 second — should truncate the 10s session
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, req := range requests {
		if req.ArrivalTime >= horizon {
			t.Errorf("request %d: ArrivalTime=%d >= horizon=%d", i, req.ArrivalTime, horizon)
		}
	}
	// Should have significantly fewer than 100 requests (horizon truncation)
	if len(requests) >= 100 {
		t.Errorf("expected fewer than 100 requests due to horizon truncation, got %d", len(requests))
	}
}

func TestGenerateRequests_SingleSession_Deterministic(t *testing.T) {
	// BC-5 / INV-6: SingleSession mode must be deterministic — same seed, same output.
	makeSpec := func() *WorkloadSpec {
		return &WorkloadSpec{
			Version: "2", Seed: 99, AggregateRate: 20.0,
			Clients: []ClientSpec{
				{
					ID: "ss-a", TenantID: "a", SLOClass: "batch", RateFraction: 0.5,
					Arrival:    ArrivalSpec{Process: "poisson"},
					InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
					Reasoning: &ReasoningSpec{
						ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
						MultiTurn: &MultiTurnSpec{
							MaxRounds: 20, ThinkTimeUs: 50_000,
							ContextGrowth: "accumulate", SingleSession: true,
						},
					},
				},
				{
					ID: "ss-b", TenantID: "b", SLOClass: "batch", RateFraction: 0.5,
					Arrival:    ArrivalSpec{Process: "poisson"},
					InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
					Reasoning: &ReasoningSpec{
						ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
						MultiTurn: &MultiTurnSpec{
							MaxRounds: 20, ThinkTimeUs: 50_000,
							ContextGrowth: "accumulate", SingleSession: true,
						},
					},
				},
			},
		}
	}
	horizon := int64(2_000_000)
	r1, err1 := GenerateRequests(makeSpec(), horizon, 0)
	r2, err2 := GenerateRequests(makeSpec(), horizon, 0)
	if err1 != nil || err2 != nil {
		t.Fatalf("errors: %v, %v", err1, err2)
	}
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

func TestGenerateRequests_SingleSession_LifecycleWindowRoundSuppression(t *testing.T) {
	// BC-6: SingleSession rounds that cross the lifecycle window boundary must be suppressed.
	// Session has MaxRounds=50 at ThinkTimeUs=100_000 (5s span), but window is only 1s wide.
	// Only rounds within the window should appear in output.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "ss-lc", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     50,
					ThinkTimeUs:   100_000, // 100ms between rounds → 50 rounds spans 5s
					ContextGrowth: "",
					SingleSession: true,
				},
			},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 1_000_000}}, // 1s window
			},
		}},
	}
	horizon := int64(10_000_000) // 10s
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	// All requests must be within the lifecycle window [0, 1_000_000)
	for i, req := range requests {
		if req.ArrivalTime >= 1_000_000 {
			t.Errorf("request %d: ArrivalTime=%d >= window end 1000000 (BC-6 violation)", i, req.ArrivalTime)
		}
	}
	// Should have significantly fewer than 50 requests (window truncation)
	if len(requests) >= 50 {
		t.Errorf("expected fewer than 50 requests due to window truncation, got %d", len(requests))
	}
	// Should have at least a few (window starts at 0, constant arrival means start near 0)
	if len(requests) < 3 {
		t.Errorf("expected at least 3 requests in 1s window with 100ms spacing, got %d", len(requests))
	}
}

func TestGenerateRequests_ReasoningClient_RespectsLifecycleWindows(t *testing.T) {
	// BC-1/BC-6: Reasoning path must respect lifecycle windows, just like the standard path.
	// Bug: the reasoning arrival loop in generator.go did not call isInActiveWindow(),
	// so multi-turn requests ignored stage boundaries.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "mt-client", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn:       &MultiTurnSpec{MaxRounds: 2, ThinkTimeUs: 10000, ContextGrowth: "accumulate"},
			},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 500_000, EndUs: 1_500_000}},
			},
		}},
	}
	horizon := int64(2_000_000) // 2 seconds
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request within the lifecycle window")
	}
	for i, req := range requests {
		if req.ArrivalTime < 500_000 || req.ArrivalTime >= 1_500_000 {
			t.Errorf("request %d: ArrivalTime=%d outside lifecycle window [500000, 1500000)",
				i, req.ArrivalTime)
		}
	}
}

func TestGenerateRequests_ReasoningClient_PrependsPrefixTokens(t *testing.T) {
	// BC-1/BC-2: Reasoning paths must prepend shared prefix tokens, just like
	// the standard request path. Both SingleSession and multi-session must work.
	for _, singleSession := range []bool{true, false} {
		name := "multi-session"
		if singleSession {
			name = "single-session"
		}
		t.Run(name, func(t *testing.T) {
			spec := &WorkloadSpec{
				Version: "2", Seed: 42, AggregateRate: 10.0,
				Clients: []ClientSpec{{
					ID: "reasoning-pfx", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
					PrefixGroup:  "system-prompt",
					PrefixLength: 20,
					Arrival:      ArrivalSpec{Process: "constant"},
					InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
					Reasoning: &ReasoningSpec{
						ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
						MultiTurn: &MultiTurnSpec{
							MaxRounds:     2,
							ThinkTimeUs:   10_000,
							ContextGrowth: "",
							SingleSession: singleSession,
						},
					},
				}},
			}
			horizon := int64(2_000_000)
			requests, err := GenerateRequests(spec, horizon, 0)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(requests) < 2 {
				t.Fatalf("expected at least 2 requests, got %d", len(requests))
			}

			prefixLen := 20
			for i, req := range requests {
				if len(req.InputTokens) < prefixLen {
					t.Errorf("request %d: input too short (%d < %d)", i, len(req.InputTokens), prefixLen)
					continue
				}
				if req.RoundIndex == 0 && len(req.InputTokens) != prefixLen+50 {
					t.Errorf("request %d (round 0): input len %d, want %d (prefix %d + input 50)",
						i, len(req.InputTokens), prefixLen+50, prefixLen)
				}
			}
			firstPrefix := requests[0].InputTokens[:prefixLen]
			for i := 1; i < len(requests); i++ {
				for j := 0; j < prefixLen; j++ {
					if requests[i].InputTokens[j] != firstPrefix[j] {
						t.Errorf("request %d: prefix token %d = %d, want %d (shared prefix mismatch)",
							i, j, requests[i].InputTokens[j], firstPrefix[j])
						break
					}
				}
			}
		})
	}
}

func TestGenerateRequests_ReasoningClient_PrefixWithAccumulation(t *testing.T) {
	// BC-8: With prefix + context accumulation, the token layout per round is:
	//   Round 0: [prefix | newInput_r0]
	//   Round 1: [prefix | newInput_r0 + output_r0 | newInput_r1]
	// The prefix is NOT part of the accumulated context — it's re-prepended fresh.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 1.0,
		Clients: []ClientSpec{{
			ID: "accum-pfx", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			PrefixGroup:  "sys",
			PrefixLength: 10,
			Arrival:      ArrivalSpec{Process: "constant"},
			InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 20}},
			OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 15}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     2,
					ThinkTimeUs:   10_000,
					ContextGrowth: "accumulate",
					SingleSession: true,
				},
			},
		}},
	}
	horizon := int64(5_000_000)
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(requests))
	}

	prefixLen := 10
	inputLen := 20
	outputLen := 15

	r0 := requests[0]
	if len(r0.InputTokens) != prefixLen+inputLen {
		t.Errorf("round 0: input len %d, want %d", len(r0.InputTokens), prefixLen+inputLen)
	}

	r1 := requests[1]
	expectedR1Len := prefixLen + (inputLen + outputLen) + inputLen
	if len(r1.InputTokens) != expectedR1Len {
		t.Errorf("round 1: input len %d, want %d", len(r1.InputTokens), expectedR1Len)
	}

	for j := 0; j < prefixLen; j++ {
		if r0.InputTokens[j] != r1.InputTokens[j] {
			t.Errorf("prefix token %d: round 0 has %d, round 1 has %d",
				j, r0.InputTokens[j], r1.InputTokens[j])
			break
		}
	}

	r0NewInput := r0.InputTokens[prefixLen:]
	for j := 0; j < inputLen; j++ {
		if r1.InputTokens[prefixLen+j] != r0NewInput[j] {
			t.Errorf("round 1 context token %d: got %d, want %d (round 0's newInput)",
				j, r1.InputTokens[prefixLen+j], r0NewInput[j])
			break
		}
	}
}

func TestGenerateRequests_MultiSession_PerRoundLifecycleFiltering(t *testing.T) {
	// BC-3: Multi-session reasoning must filter individual rounds against
	// lifecycle windows, not just session start times. A session starting
	// inside a window can have later rounds that cross the window boundary.
	//
	// Timeline with rate=1.0 constant arrival:
	//   Session 1 starts at t=1,000,000 (1s). Rounds:
	//     Round 0: t=1,000,000 (in window [0, 2s)) → KEPT
	//     Round 1: t=1,500,025 (in window) → KEPT
	//     Round 2: t=2,000,050 (outside window) → FILTERED
	//     Rounds 3-9: beyond window → FILTERED
	//   Session 2 would start at t=2,000,000 — filtered at session level.
	//   Expected: 2 accepted requests.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 1.0,
		Clients: []ClientSpec{{
			ID: "ms-lc", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     10,
					ThinkTimeUs:   500_000,
					ContextGrowth: "",
					SingleSession: false,
				},
			},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 2_000_000}},
			},
		}},
	}
	horizon := int64(10_000_000)
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	for i, req := range requests {
		if req.ArrivalTime >= 2_000_000 {
			t.Errorf("request %d: ArrivalTime=%d >= window end 2000000 (BC-3 violation)",
				i, req.ArrivalTime)
		}
	}
	if len(requests) >= 10 {
		t.Errorf("expected fewer than 10 requests due to lifecycle window truncation, got %d", len(requests))
	}
}

func TestGenerateRequests_MultiSession_PerRoundHorizonFiltering(t *testing.T) {
	// BC-4: Multi-session reasoning must filter individual rounds against
	// horizon, not just the session start time. No lifecycle windows — exercises
	// the `break` on horizon independently of lifecycle filtering.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 1.0,
		Clients: []ClientSpec{{
			ID: "ms-hz", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     20,
					ThinkTimeUs:   500_000,
					ContextGrowth: "",
					SingleSession: false,
				},
			},
		}},
	}
	horizon := int64(3_000_000) // 3s — session at 1s, rounds at 1.0s, 1.5s, 2.0s, 2.5s, 3.0s...
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	for i, req := range requests {
		if req.ArrivalTime >= horizon {
			t.Errorf("request %d: ArrivalTime=%d >= horizon %d (BC-4 violation)",
				i, req.ArrivalTime, horizon)
		}
	}
	if len(requests) >= 20 {
		t.Errorf("expected fewer than 20 requests due to horizon truncation, got %d", len(requests))
	}
}

func TestGenerateRequests_ReasoningClient_NoPrefixGroup_Unchanged(t *testing.T) {
	// BC-5: Reasoning clients without prefix_group must produce identical output.
	// No prefix should be spuriously prepended.
	spec := &WorkloadSpec{
		Version: "2", Seed: 99, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "no-pfx", TenantID: "t1", SLOClass: "batch", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     3,
					ThinkTimeUs:   10_000,
					ContextGrowth: "",
					SingleSession: false,
				},
			},
			// No PrefixGroup, no Lifecycle
		}},
	}
	horizon := int64(2_000_000)
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	// With no prefix, input tokens should be exactly 50 (constant sampler) for round 0.
	for _, req := range requests {
		if req.RoundIndex == 0 && len(req.InputTokens) != 50 {
			t.Errorf("round 0 request: input len %d, want 50 (no prefix should be added)", len(req.InputTokens))
		}
	}
}

// BC-5: Generator sets MaxOutputLen = len(OutputTokens) for all requests.
func TestGenerateRequests_SetsMaxOutputLen(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		Seed:          42,
		Category:      "language",
		AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "constant"},
			InputDist:    DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist:   DistSpec{Type: "constant", Params: map[string]float64{"value": 30}},
		}},
	}

	requests, err := GenerateRequests(spec, 1_000_000, 42)
	if err != nil {
		t.Fatalf("GenerateRequests: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}

	for i, req := range requests {
		if req.MaxOutputLen != len(req.OutputTokens) {
			t.Errorf("request %d: MaxOutputLen=%d, want len(OutputTokens)=%d",
				i, req.MaxOutputLen, len(req.OutputTokens))
		}
	}
}

// TestGenerateWorkload_ClosedLoop_OnlyRound0 verifies that closed-loop
// reasoning clients produce only round-0 requests with session blueprints.
func TestGenerateWorkload_ClosedLoop_OnlyRound0(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{
			{
				ID: "reasoning", TenantID: "t1", SLOClass: "standard", RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 20}},
				Reasoning: &ReasoningSpec{
					ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
					MultiTurn:       &MultiTurnSpec{MaxRounds: 3, ThinkTimeUs: 1000, ContextGrowth: ""},
				},
				// ClosedLoop defaults to true for reasoning clients
			},
		},
	}

	wl, err := GenerateWorkload(spec, 10_000_000, 50)
	if err != nil {
		t.Fatal(err)
	}

	// All requests should be round 0
	for _, req := range wl.Requests {
		if req.RoundIndex != 0 {
			t.Errorf("closed-loop request %s has RoundIndex=%d, want 0", req.ID, req.RoundIndex)
		}
	}

	// Should have session blueprints
	if len(wl.Sessions) == 0 {
		t.Fatal("expected session blueprints for closed-loop reasoning client")
	}

	// Each blueprint should have MaxRounds=3
	for _, bp := range wl.Sessions {
		if bp.MaxRounds != 3 {
			t.Errorf("blueprint %s MaxRounds=%d, want 3", bp.SessionID, bp.MaxRounds)
		}
	}
}

// TestGenerateWorkload_OpenLoop_AllRounds verifies that clients with
// closed_loop: false preserve the current all-rounds generation behavior.
func TestGenerateWorkload_OpenLoop_AllRounds(t *testing.T) {
	closedLoopFalse := false
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{
			{
				ID: "reasoning-openloop", TenantID: "t1", SLOClass: "standard", RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 20}},
				Reasoning: &ReasoningSpec{
					ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
					MultiTurn:       &MultiTurnSpec{MaxRounds: 3, ThinkTimeUs: 1000, ContextGrowth: ""},
				},
				ClosedLoop: &closedLoopFalse, // explicit opt-out
			},
		},
	}

	wl, err := GenerateWorkload(spec, 10_000_000, 50)
	if err != nil {
		t.Fatal(err)
	}

	// Should have NO session blueprints (open-loop)
	if len(wl.Sessions) != 0 {
		t.Errorf("open-loop: expected 0 session blueprints, got %d", len(wl.Sessions))
	}

	// Should have rounds > 0 (all rounds pre-generated)
	hasLaterRound := false
	for _, req := range wl.Requests {
		if req.RoundIndex > 0 {
			hasLaterRound = true
			break
		}
	}
	if !hasLaterRound {
		t.Error("open-loop: expected requests with RoundIndex > 0 (all rounds pre-generated)")
	}
}

// TestGenerateWorkload_NonSessionWorkload_NoBlueprints verifies that
// non-reasoning workloads produce no session blueprints.
func TestGenerateWorkload_NonSessionWorkload_NoBlueprints(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{
			{
				ID: "standard", TenantID: "t1", SLOClass: "standard", RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 20}},
			},
		},
	}

	wl, err := GenerateWorkload(spec, 10_000_000, 50)
	if err != nil {
		t.Fatal(err)
	}

	if len(wl.Sessions) != 0 {
		t.Errorf("non-session: expected 0 blueprints, got %d", len(wl.Sessions))
	}
	if len(wl.Requests) == 0 {
		t.Error("expected requests for non-session workload")
	}
}

// TestGenerateWorkload_Deadline_NonSessionNoTimeout verifies that non-session
// clients get Deadline=0 (no timeout) by default — backward compatible.
func TestGenerateWorkload_Deadline_NonSessionNoTimeout(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{
			{
				ID: "std", TenantID: "t1", SLOClass: "standard", RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 20}},
			},
		},
	}

	wl, err := GenerateWorkload(spec, 10_000_000, 10)
	if err != nil {
		t.Fatal(err)
	}

	for _, req := range wl.Requests {
		if req.Deadline != 0 {
			t.Errorf("non-session request %s: Deadline=%d, want 0 (no timeout for non-session)", req.ID, req.Deadline)
		}
	}
}

// TestGenerateWorkload_Deadline_SessionDefaultTimeout verifies that session
// (reasoning/multi-turn) clients get default 300s timeout when Timeout is nil.
func TestGenerateWorkload_Deadline_SessionDefaultTimeout(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, Category: "language", AggregateRate: 5.0,
		Clients: []ClientSpec{
			{
				ID: "reasoning", TenantID: "t1", SLOClass: "standard", RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 20}},
				OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				Reasoning: &ReasoningSpec{
					ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
					MultiTurn:       &MultiTurnSpec{MaxRounds: 2, ThinkTimeUs: 1000, ContextGrowth: ""},
				},
			},
		},
	}

	wl, err := GenerateWorkload(spec, 10_000_000, 10)
	if err != nil {
		t.Fatal(err)
	}

	for _, req := range wl.Requests {
		expected := req.ArrivalTime + DefaultTimeoutUs
		if req.Deadline != expected {
			t.Errorf("session request %s: Deadline=%d, want %d (arrival + 300s default)", req.ID, req.Deadline, expected)
		}
	}
}

// TestGenerateWorkload_SessionManager_Integration verifies that GenerateWorkload
// blueprints work correctly with SessionManager end-to-end: round-0 requests
// are generated, blueprints produce valid follow-up rounds via OnComplete.
func TestGenerateWorkload_SessionManager_Integration(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, Category: "language", AggregateRate: 5.0,
		Clients: []ClientSpec{
			{
				ID: "session-client", TenantID: "t1", SLOClass: "standard", RateFraction: 1.0,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 20}},
				OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				Reasoning: &ReasoningSpec{
					ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
					MultiTurn:       &MultiTurnSpec{MaxRounds: 3, ThinkTimeUs: 5000, ContextGrowth: ""},
				},
			},
		},
	}

	wl, err := GenerateWorkload(spec, 10_000_000, 20)
	if err != nil {
		t.Fatal(err)
	}
	if len(wl.Sessions) == 0 {
		t.Fatal("expected session blueprints")
	}

	// Create SessionManager from generated blueprints
	sm := NewSessionManager(wl.Sessions)

	// Simulate round-0 completion → should produce round-1 follow-up
	for _, req := range wl.Requests {
		if req.SessionID == "" {
			continue
		}
		// Simulate completion
		req.State = sim.StateCompleted
		req.ProgressIndex = int64(len(req.InputTokens) + len(req.OutputTokens) - 1)

		follow := sm.OnComplete(req, req.ArrivalTime+5000) // completion at arrival+5ms
		if follow == nil {
			t.Errorf("request %s (session %s, round %d): expected follow-up, got nil",
				req.ID, req.SessionID, req.RoundIndex)
			continue
		}
		if len(follow) != 1 {
			t.Errorf("expected 1 follow-up, got %d", len(follow))
			continue
		}
		r1 := follow[0]
		if r1.SessionID != req.SessionID {
			t.Errorf("follow-up SessionID = %q, want %q", r1.SessionID, req.SessionID)
		}
		if r1.RoundIndex != 1 {
			t.Errorf("follow-up RoundIndex = %d, want 1", r1.RoundIndex)
		}
		if r1.ArrivalTime != req.ArrivalTime+5000+5000 {
			t.Errorf("follow-up ArrivalTime = %d, want %d (completion + ThinkTime)",
				r1.ArrivalTime, req.ArrivalTime+5000+5000)
		}
		if r1.Deadline <= 0 {
			t.Errorf("follow-up Deadline = %d, want > 0", r1.Deadline)
		}
		break // test just one session
	}
}

func TestGenerateRequests_MutualExclusion_ClientsAndServeGen_ReturnsError(t *testing.T) {
	// BC-6: Clients + ServeGenData → error
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		ServeGenData: &ServeGenDataSpec{Path: "data/"},
	}
	_, err := GenerateRequests(spec, 1_000_000, 100)
	if err == nil {
		t.Fatal("expected error for Clients + ServeGenData, got nil")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("error should mention mutual exclusion: %v", err)
	}
}

func TestGenerateRequests_MutualExclusion_ClientsAndInferencePerf_ReturnsError(t *testing.T) {
	// BC-6: Clients + InferencePerf → error
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		InferencePerf: &InferencePerfSpec{
			Stages: []StageSpec{{Rate: 10, Duration: 60}},
			SharedPrefix: &SharedPrefixSpec{
				NumUniqueSystemPrompts: 1, NumUsersPerSystemPrompt: 1,
				SystemPromptLen: 10, QuestionLen: 100, OutputLen: 50,
			},
		},
	}
	_, err := GenerateRequests(spec, 1_000_000, 100)
	if err == nil {
		t.Fatal("expected error for Clients + InferencePerf, got nil")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("error should mention mutual exclusion: %v", err)
	}
}

func TestGenerateRequests_MutualExclusion_ServeGenAndInferencePerf_ReturnsError(t *testing.T) {
	// BC-6: ServeGenData + InferencePerf → error
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		ServeGenData: &ServeGenDataSpec{Path: "data/"},
		InferencePerf: &InferencePerfSpec{
			Stages: []StageSpec{{Rate: 10, Duration: 60}},
			SharedPrefix: &SharedPrefixSpec{
				NumUniqueSystemPrompts: 1, NumUsersPerSystemPrompt: 1,
				SystemPromptLen: 10, QuestionLen: 100, OutputLen: 50,
			},
		},
	}
	_, err := GenerateRequests(spec, 1_000_000, 100)
	if err == nil {
		t.Fatal("expected error for ServeGenData + InferencePerf, got nil")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("error should mention mutual exclusion: %v", err)
	}
}

func TestGenerateRequests_MutualExclusion_AllThreeSources_ReturnsError(t *testing.T) {
	// BC-6: All three sources set → error listing all three
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		ServeGenData: &ServeGenDataSpec{Path: "data/"},
		InferencePerf: &InferencePerfSpec{
			Stages: []StageSpec{{Rate: 10, Duration: 60}},
			SharedPrefix: &SharedPrefixSpec{
				NumUniqueSystemPrompts: 1, NumUsersPerSystemPrompt: 1,
				SystemPromptLen: 10, QuestionLen: 100, OutputLen: 50,
			},
		},
	}
	_, err := GenerateRequests(spec, 1_000_000, 100)
	if err == nil {
		t.Fatal("expected error for all three sources, got nil")
	}
	if !strings.Contains(err.Error(), "clients") || !strings.Contains(err.Error(), "servegen_data") || !strings.Contains(err.Error(), "inference_perf") {
		t.Errorf("error should list all three conflicting sources: %v", err)
	}
}

func TestGenerateRequests_MutualExclusion_CohortsWithClients_Allowed(t *testing.T) {
	// BC-7: Cohorts + Clients is intentional composition, not a conflict
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 0.5,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		Cohorts: []CohortSpec{{
			ID: "cohort1", Population: 2, RateFraction: 0.5,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	reqs, err := GenerateRequests(spec, 1_000_000, 10)
	if err != nil {
		t.Fatalf("unexpected error for Clients + Cohorts: %v", err)
	}
	if len(reqs) == 0 {
		t.Error("expected requests from Clients + Cohorts composition")
	}
}

func TestGenerateRequests_MutualExclusion_CohortsWithInferencePerf_Allowed(t *testing.T) {
	// BC-7: Cohorts + InferencePerf (no explicit Clients) is allowed composition.
	// InferencePerf expands into Clients, then Cohorts compose with them.
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Cohorts: []CohortSpec{{
			ID: "cohort1", Population: 1, RateFraction: 0.5,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		InferencePerf: &InferencePerfSpec{
			Stages: []StageSpec{{Rate: 10, Duration: 60}},
			SharedPrefix: &SharedPrefixSpec{
				NumUniqueSystemPrompts: 1, NumUsersPerSystemPrompt: 1,
				SystemPromptLen: 10, QuestionLen: 100, OutputLen: 50,
			},
		},
	}
	reqs, err := GenerateRequests(spec, 1_000_000, 10)
	if err != nil {
		t.Fatalf("unexpected error for Cohorts + InferencePerf: %v", err)
	}
	if len(reqs) == 0 {
		t.Error("expected requests from Cohorts + InferencePerf composition")
	}
}
