package workload

import (
	"math/rand"
	"testing"
)

func TestGenerateReasoningRequests_MultiTurn_SequentialRounds(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	spec := &ReasoningSpec{
		MultiTurn: &MultiTurnSpec{
			MaxRounds:     3,
			ThinkTimeUs:   1000,
			ContextGrowth: "fixed",
		},
	}
	inputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 50, "max": 200}})
	outputSampler, _ := NewLengthSampler(DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}})

	requests, err := GenerateReasoningRequests(rng, spec, inputSampler, outputSampler, 0, "c1", "t1", "batch")
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 3 {
		t.Fatalf("expected 3 rounds, got %d", len(requests))
	}

	// Verify sequential round indices and same session ID
	sessionID := requests[0].SessionID
	if sessionID == "" {
		t.Error("session ID should be set")
	}
	for i, req := range requests {
		if req.RoundIndex != i {
			t.Errorf("round %d: RoundIndex = %d", i, req.RoundIndex)
		}
		if req.SessionID != sessionID {
			t.Errorf("round %d: different session ID", i)
		}
	}

	// Arrival times should be increasing
	for i := 1; i < len(requests); i++ {
		if requests[i].ArrivalTime <= requests[i-1].ArrivalTime {
			t.Errorf("round %d arrival (%d) should be > round %d (%d)",
				i, requests[i].ArrivalTime, i-1, requests[i-1].ArrivalTime)
		}
	}
}

func TestGenerateReasoningRequests_ContextAccumulate_GrowingInput(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	spec := &ReasoningSpec{
		MultiTurn: &MultiTurnSpec{
			MaxRounds:     3,
			ThinkTimeUs:   1000,
			ContextGrowth: "accumulate",
		},
	}
	inputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 5, "min": 90, "max": 110}})
	outputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 40, "max": 60}})

	requests, err := GenerateReasoningRequests(rng, spec, inputSampler, outputSampler, 0, "c1", "t1", "batch")
	if err != nil {
		t.Fatal(err)
	}

	// With "accumulate", input tokens should grow per round
	for i := 1; i < len(requests); i++ {
		if len(requests[i].InputTokens) <= len(requests[i-1].InputTokens) {
			t.Errorf("round %d input (%d) should be > round %d (%d) with accumulate",
				i, len(requests[i].InputTokens), i-1, len(requests[i-1].InputTokens))
		}
	}
}

func TestGenerateReasoningRequests_ReasonRatio_InRange(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	spec := &ReasoningSpec{
		ReasonRatioDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 20, "min": 0, "max": 100}},
		MultiTurn: &MultiTurnSpec{
			MaxRounds:     5,
			ThinkTimeUs:   1000,
			ContextGrowth: "fixed",
		},
	}
	inputSampler, _ := NewLengthSampler(DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}})
	outputSampler, _ := NewLengthSampler(DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}})

	requests, err := GenerateReasoningRequests(rng, spec, inputSampler, outputSampler, 0, "c1", "t1", "batch")
	if err != nil {
		t.Fatal(err)
	}

	for i, req := range requests {
		if req.ReasonRatio < 0 || req.ReasonRatio > 1.0 {
			t.Errorf("round %d: ReasonRatio = %f, want [0, 1]", i, req.ReasonRatio)
		}
	}
}

// TestGenerateReasoningRequests_Accumulate_SharesPrefixTokens verifies that
// multi-turn requests with context_growth="accumulate" share actual prefix
// tokens across rounds (not just matching lengths with independent random values).
func TestGenerateReasoningRequests_Accumulate_SharesPrefixTokens(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	spec := &ReasoningSpec{
		MultiTurn: &MultiTurnSpec{
			MaxRounds:     3,
			ThinkTimeUs:   1000,
			ContextGrowth: "accumulate",
		},
	}
	inputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 5, "min": 90, "max": 110}})
	outputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 40, "max": 60}})

	requests, err := GenerateReasoningRequests(rng, spec, inputSampler, outputSampler, 0, "c1", "t1", "batch")
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 3 {
		t.Fatalf("expected 3 rounds, got %d", len(requests))
	}

	// Round 0 tokens are the prefix for round 1
	round0Tokens := requests[0].InputTokens
	round1Tokens := requests[1].InputTokens

	// Round 1 must start with round 0's input + round 0's output
	round0OutputLen := len(requests[0].OutputTokens)
	expectedPrefixLen := len(round0Tokens) + round0OutputLen
	if len(round1Tokens) < expectedPrefixLen {
		t.Fatalf("round 1 too short: %d < expected prefix %d", len(round1Tokens), expectedPrefixLen)
	}

	// Verify the shared prefix tokens match exactly
	for i := 0; i < len(round0Tokens); i++ {
		if round1Tokens[i] != round0Tokens[i] {
			t.Errorf("token %d: round 1 has %d, round 0 has %d — prefix tokens must match",
				i, round1Tokens[i], round0Tokens[i])
			break
		}
	}

	// Verify round 0's output tokens appear after round 0's input in round 1
	for i := 0; i < round0OutputLen; i++ {
		idx := len(round0Tokens) + i
		if round1Tokens[idx] != requests[0].OutputTokens[i] {
			t.Errorf("output token %d: round 1 has %d, round 0 output has %d — context must include prior output",
				i, round1Tokens[idx], requests[0].OutputTokens[i])
			break
		}
	}
}

func TestGenerateReasoningRequests_NilSpec_ReturnsNil(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	requests, err := GenerateReasoningRequests(rng, nil, nil, nil, 0, "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if requests != nil {
		t.Error("expected nil for nil spec")
	}
}
