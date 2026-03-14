package workload

import (
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func makeTestBlueprint(sessionID string, maxRounds int, thinkTime int64, contextGrowth string, horizon int64) SessionBlueprint {
	return SessionBlueprint{
		SessionID:     sessionID,
		ClientID:      "test-client",
		MaxRounds:     maxRounds,
		ContextGrowth: contextGrowth,
		ThinkTimeUs:   thinkTime,
		Horizon:       horizon,
		InputSampler:  &constantSampler{value: 10},
		OutputSampler: &constantSampler{value: 5},
		RNG:           rand.New(rand.NewSource(42)),
		TenantID:      "test-tenant",
		SLOClass:      "standard",
		Model:         "test-model",
	}
}

// constantSampler implements LengthSampler for deterministic testing.
type constantSampler struct {
	value int
}

func (s *constantSampler) Sample(_ *rand.Rand) int { return s.value }

// TestSession_RoundGeneration_CorrectArrivalTime verifies BC-6:
// round N+1 arrival time = round N completion tick + ThinkTimeUs.
func TestSession_RoundGeneration_CorrectArrivalTime(t *testing.T) {
	bp := makeTestBlueprint("sess1", 3, 1000, "", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Simulate round 0 completing at tick 5000
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess1", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15, // 10 input + 5 output
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}

	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up request, got %d", len(follow))
	}
	if follow[0].ArrivalTime != 6000 {
		t.Errorf("BC-6: arrival time = %d, want 6000 (5000 + 1000)", follow[0].ArrivalTime)
	}
	if follow[0].RoundIndex != 1 {
		t.Errorf("round index = %d, want 1", follow[0].RoundIndex)
	}
	if follow[0].SessionID != "sess1" {
		t.Errorf("session ID = %q, want sess1", follow[0].SessionID)
	}
}

// TestSession_TimeoutCancels_NoMoreRounds verifies BC-7:
// when a round times out, the session is cancelled.
func TestSession_TimeoutCancels_NoMoreRounds(t *testing.T) {
	bp := makeTestBlueprint("sess2", 5, 1000, "", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 2 times out
	req := &sim.Request{
		ID: "r2", SessionID: "sess2", RoundIndex: 2,
		State: sim.StateTimedOut,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}

	follow := sm.OnComplete(req, 10000)
	if follow != nil {
		t.Errorf("BC-7: expected nil follow-up after timeout, got %d requests", len(follow))
	}

	// Verify session is cancelled — even if we call again, no follow-ups
	req2 := &sim.Request{
		ID: "r2b", SessionID: "sess2", RoundIndex: 2,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow2 := sm.OnComplete(req2, 11000)
	if follow2 != nil {
		t.Errorf("BC-7: expected nil after session cancelled, got %d requests", len(follow2))
	}
}

// TestSession_ContextAccumulation verifies BC-8:
// round N+1 input starts with accumulated context from prior rounds.
func TestSession_ContextAccumulation(t *testing.T) {
	bp := makeTestBlueprint("sess3", 3, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0: 10 input tokens + 5 output tokens
	inputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(99)), 10)
	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(100)), 5)
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess3", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: inputR0, OutputTokens: outputR0,
	}

	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}

	// Round 1's input should start with accumulated context (10 + 5 = 15 tokens from round 0)
	// followed by 10 new tokens from the sampler
	r1Input := follow[0].InputTokens
	if len(r1Input) != 25 { // 15 accumulated + 10 new
		t.Errorf("BC-8: round 1 input length = %d, want 25 (15 accumulated + 10 new)", len(r1Input))
	}

	// Verify the first 10 tokens match round 0's input
	for i := 0; i < 10 && i < len(r1Input); i++ {
		if r1Input[i] != inputR0[i] {
			t.Errorf("BC-8: accumulated token %d = %d, want %d (from round 0 input)", i, r1Input[i], inputR0[i])
			break
		}
	}
}

// TestSession_BeyondHorizon_NotGenerated verifies BC-19:
// follow-up rounds past horizon are not generated.
func TestSession_BeyondHorizon_NotGenerated(t *testing.T) {
	bp := makeTestBlueprint("sess4", 3, 1000, "", 6000) // horizon = 6000
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0 completes at tick 5500. Next round would arrive at 5500+1000=6500 > horizon
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess4", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}

	follow := sm.OnComplete(req0, 5500)
	if follow != nil {
		t.Errorf("BC-19: expected nil (beyond horizon), got %d requests", len(follow))
	}
}

// TestSession_DroppedFollowUp_CancelsSession verifies BC-17:
// a dropped-unservable follow-up cancels the session.
func TestSession_DroppedFollowUp_CancelsSession(t *testing.T) {
	bp := makeTestBlueprint("sess5", 5, 1000, "", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Simulate a dropped request (state is still StateQueued from construction)
	req := &sim.Request{
		ID: "r1", SessionID: "sess5", RoundIndex: 1,
		State: sim.StateQueued,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}

	follow := sm.OnComplete(req, 10000)
	if follow != nil {
		t.Errorf("BC-17: expected nil after drop, got %d requests", len(follow))
	}
}

// TestSession_LengthCapped_ContinuesSession verifies BC-16:
// a length-capped request continues the session (not cancelled).
func TestSession_LengthCapped_ContinuesSession(t *testing.T) {
	bp := makeTestBlueprint("sess6", 3, 1000, "", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Length-capped request: State=Completed, LengthCapped=true, fewer output tokens
	req := &sim.Request{
		ID: "r0", SessionID: "sess6", RoundIndex: 0,
		State: sim.StateCompleted, LengthCapped: true,
		ProgressIndex: 13, // 10 input + 3 actual output (out of 5 oracle)
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}

	follow := sm.OnComplete(req, 5000)
	if len(follow) != 1 {
		t.Fatalf("BC-16: expected 1 follow-up (length-capped continues), got %d", len(follow))
	}
	if follow[0].ArrivalTime != 6000 {
		t.Errorf("BC-16: arrival time = %d, want 6000", follow[0].ArrivalTime)
	}
}

// TestSession_FinalRound_Completes verifies that the final round
// transitions the session to completed (no more follow-ups).
func TestSession_FinalRound_Completes(t *testing.T) {
	bp := makeTestBlueprint("sess7", 2, 1000, "", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0 completes → generates round 1
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess7", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow0 := sm.OnComplete(req0, 5000)
	if len(follow0) != 1 {
		t.Fatalf("expected round 1, got %d", len(follow0))
	}

	// Round 1 completes → no more rounds (MaxRounds=2, current=1 which is final)
	req1 := &sim.Request{
		ID: "r1", SessionID: "sess7", RoundIndex: 1,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow1 := sm.OnComplete(req1, 7000)
	if follow1 != nil {
		t.Errorf("expected nil after final round, got %d", len(follow1))
	}
}

// TestSession_NonSessionRequest_ReturnsNil verifies that non-session
// requests (empty SessionID) are ignored by the manager.
func TestSession_NonSessionRequest_ReturnsNil(t *testing.T) {
	bp := makeTestBlueprint("sess8", 3, 1000, "", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	req := &sim.Request{
		ID: "non-session", SessionID: "",
		State: sim.StateCompleted,
	}
	follow := sm.OnComplete(req, 5000)
	if follow != nil {
		t.Errorf("expected nil for non-session request, got %d", len(follow))
	}
}

// TestNewSessionManager_PanicsOnZeroMaxRounds verifies MaxRounds validation.
func TestNewSessionManager_PanicsOnZeroMaxRounds(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic for MaxRounds=0, got none")
		}
	}()
	bp := makeTestBlueprint("bad", 0, 1000, "", 1_000_000)
	NewSessionManager([]SessionBlueprint{bp})
}
