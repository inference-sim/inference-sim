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

// TestSession_ContextAccumulation_MultiStep verifies BC-1:
// context accumulation across a 3-round chain (round 0 → 1 → 2).
// Extends TestSession_ContextAccumulation (which only tests round 0 → 1).
// NOTE: contextTokens is append-only across ALL rounds — it grows to 15 after
// round 0 (input10 + output5), then to 45 after round 1 (prior15 + input25 + output5).
func TestSession_ContextAccumulation_MultiStep(t *testing.T) {
	bp := makeTestBlueprint("sess-accum3", 3, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	inputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(1)), 10)
	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(2)), 5)
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-accum3", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15, // 10 input + 5 output
		InputTokens: inputR0, OutputTokens: outputR0,
	}
	follow1 := sm.OnComplete(req0, 5000)
	if len(follow1) != 1 {
		t.Fatalf("expected 1 follow-up after round 0, got %d", len(follow1))
	}
	// Round 1: contextTokens(15: r0 input+output) + 10 new = 25
	if len(follow1[0].InputTokens) != 25 {
		t.Errorf("BC-1: round 1 input length = %d, want 25 (10+5+10)", len(follow1[0].InputTokens))
	}

	req1 := &sim.Request{
		ID: "r1", SessionID: "sess-accum3", RoundIndex: 1,
		State: sim.StateCompleted,
		ProgressIndex: int64(len(follow1[0].InputTokens) + len(follow1[0].OutputTokens)), // 25 + 5 = 30
		InputTokens: follow1[0].InputTokens, OutputTokens: follow1[0].OutputTokens,
	}
	follow2 := sm.OnComplete(req1, 12000)
	if len(follow2) != 1 {
		t.Fatalf("expected 1 follow-up after round 1, got %d", len(follow2))
	}
	// Round 2: contextTokens grows to 45 (prior 15 + r1 input 25 + r1 output 5),
	// then + 10 new = 55. contextTokens is append-only across ALL rounds.
	if len(follow2[0].InputTokens) != 55 {
		t.Errorf("BC-1: round 2 input length = %d, want 55 (contextTokens(45)+10)", len(follow2[0].InputTokens))
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

// TestSession_HorizonInterrupted_IsTerminal verifies BC-2:
// after horizon interruption, any further OnComplete call for the same
// session returns nil (the session is terminal, not silently active).
// Extends TestSession_BeyondHorizon_NotGenerated (which only checks the
// first call, not the terminal-state idempotency required by INV-11).
func TestSession_HorizonInterrupted_IsTerminal(t *testing.T) {
	bp := makeTestBlueprint("sess-hz-term", 3, 1000, "", 6000) // horizon = 6000
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-hz-term", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	// Next arrival = 5500 + 1000 = 6500 > horizon → horizon-interrupted
	follow := sm.OnComplete(req0, 5500)
	if follow != nil {
		t.Errorf("BC-2: expected nil (beyond horizon), got %d follow-ups", len(follow))
	}

	// BC-2: any subsequent call must also return nil (terminal, not active).
	// This simulates an implementation bug where session state might reset —
	// a real DES would not call OnComplete twice for the same session/round,
	// but this guard ensures terminal state is idempotent regardless.
	req0b := &sim.Request{
		ID: "r0b", SessionID: "sess-hz-term", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow2 := sm.OnComplete(req0b, 5500)
	if follow2 != nil {
		t.Errorf("BC-2: expected nil on repeat call (session must be terminal), got %d", len(follow2))
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

// TestSession_ThinkTimeSampler_UsedWhenPresent verifies BC-3:
// when ThinkTimeSampler is set, OnComplete uses it instead of constant ThinkTimeUs.
func TestSession_ThinkTimeSampler_UsedWhenPresent(t *testing.T) {
	bp := makeTestBlueprint("tts1", 3, 1000, "", 1_000_000)
	bp.ThinkTimeSampler = &SequenceSampler{values: []int{2000, 3000}}
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "tts1", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}

	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}
	// Should use ThinkTimeSampler value (2000) not constant ThinkTimeUs (1000)
	if follow[0].ArrivalTime != 7000 {
		t.Errorf("BC-3: arrival = %d, want 7000 (5000 + 2000)", follow[0].ArrivalTime)
	}
}

// TestSession_ThinkTimeSampler_NilFallsBack verifies BC-4:
// when ThinkTimeSampler is nil, OnComplete falls back to constant ThinkTimeUs.
func TestSession_ThinkTimeSampler_NilFallsBack(t *testing.T) {
	bp := makeTestBlueprint("tts2", 3, 1000, "", 1_000_000)
	// ThinkTimeSampler is nil by default
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "tts2", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}

	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}
	if follow[0].ArrivalTime != 6000 {
		t.Errorf("BC-4: arrival = %d, want 6000 (5000 + 1000)", follow[0].ArrivalTime)
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

func TestSession_UnlimitedRounds_ContinuesPastMaxRounds(t *testing.T) {
	bp := makeTestBlueprint("sess-unlim", 1, 1000, "", 1_000_000)
	bp.UnlimitedRounds = true
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-unlim", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up with UnlimitedRounds, got %d", len(follow))
	}
}

func TestSession_FollowUpBudget_StopsWhenExhausted(t *testing.T) {
	bp1 := makeTestBlueprint("sess-b1", 1, 1000, "", 1_000_000)
	bp1.UnlimitedRounds = true
	bp2 := makeTestBlueprint("sess-b2", 1, 1000, "", 1_000_000)
	bp2.UnlimitedRounds = true
	sm := NewSessionManager([]SessionBlueprint{bp1, bp2})
	sm.SetFollowUpBudget(2)

	req1 := &sim.Request{
		ID: "r1-0", SessionID: "sess-b1", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow1 := sm.OnComplete(req1, 5000)
	if len(follow1) != 1 {
		t.Fatalf("expected 1 follow-up (budget=2), got %d", len(follow1))
	}

	req2 := &sim.Request{
		ID: "r2-0", SessionID: "sess-b2", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow2 := sm.OnComplete(req2, 6000)
	if len(follow2) != 1 {
		t.Fatalf("expected 1 follow-up (budget=1), got %d", len(follow2))
	}

	req1b := &sim.Request{
		ID: "r1-1", SessionID: "sess-b1", RoundIndex: 1,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow1b := sm.OnComplete(req1b, 8000)
	if follow1b != nil {
		t.Errorf("expected nil after budget exhausted, got %d", len(follow1b))
	}
}

func TestSession_UnlimitedRounds_AllowsZeroMaxRounds(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("unexpected panic for UnlimitedRounds with MaxRounds=0: %v", r)
		}
	}()
	bp := SessionBlueprint{
		SessionID:       "sess-zero",
		ClientID:        "test-client",
		MaxRounds:       0,
		UnlimitedRounds: true,
		ThinkTimeUs:     1000,
		Horizon:         1_000_000,
		InputSampler:    &constantSampler{value: 10},
		OutputSampler:   &constantSampler{value: 5},
		RNG:             rand.New(rand.NewSource(42)),
		TenantID:        "test-tenant",
		SLOClass:        "standard",
		Model:           "test-model",
	}
	NewSessionManager([]SessionBlueprint{bp})
}

func TestSession_NoContextAccumulation_FreshTokens(t *testing.T) {
	bp := makeTestBlueprint("sess-fresh", 1, 1000, "", 1_000_000)
	bp.UnlimitedRounds = true
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-fresh", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}
	if len(follow[0].InputTokens) != 10 {
		t.Errorf("BC-6: follow-up input length = %d, want 10 (fresh, no accumulation)", len(follow[0].InputTokens))
	}
}
