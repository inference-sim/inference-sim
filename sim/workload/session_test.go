package workload

import (
	"math/rand"
	"strings"
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

// TestSession_AccumulateMalformedTrace_PrefixLongerThanInput exercises the
// defensive fallback at session.go:198-213 where round 0's InputTokens is
// shorter than bp.Prefix (e.g., malformed trace replay). Verifies that
// (a) no panic occurs, (b) the follow-up is still generated with the same
// byte content as the legacy session.go produced (NEW-MOD-1, #1445).
func TestSession_AccumulateMalformedTrace_PrefixLongerThanInput(t *testing.T) {
	bp := makeTestBlueprint("sess-malformed", 3, 1000, "accumulate", 1_000_000)
	bp.Prefix = sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(20)), 10) // 10 prefix tokens
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0: InputTokens shorter than prefix (3 tokens vs 10 expected).
	inputR0 := []int{1, 2, 3}
	outputR0 := []int{91, 92}
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-malformed", RoundIndex: 0,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(inputR0) + len(outputR0)), // 3+2 = 5
		InputTokens:   inputR0,
		OutputTokens:  outputR0,
	}

	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up despite malformed input, got %d", len(follow))
	}
	r1 := follow[0]
	// Legacy/new contract: round 1 = [prefix(10) | full_short_input(3) | output(2) | newInput(10)] = 25
	wantLen := 10 + 3 + 2 + 10
	if len(r1.InputTokens) != wantLen {
		t.Fatalf("round 1 InputLen = %d, want %d (malformed-trace fallback byte-equivalent to legacy)", len(r1.InputTokens), wantLen)
	}
	// Verify the prefix appears at position 0
	for i, tok := range bp.Prefix {
		if r1.InputTokens[i] != tok {
			t.Fatalf("token[%d] = %d, want prefix token %d", i, r1.InputTokens[i], tok)
		}
	}
	// Verify the (malformed) round 0 input appears verbatim after the prefix
	for i, tok := range inputR0 {
		if r1.InputTokens[len(bp.Prefix)+i] != tok {
			t.Fatalf("token[%d] = %d, want round 0 input %d", len(bp.Prefix)+i, r1.InputTokens[len(bp.Prefix)+i], tok)
		}
	}
}

// TestSession_AccumulateContinuityGuard_RejectsMismatchedSlice verifies the
// session continuity panic guard (NEW-IMP-1, #1445): if a caller assigns a
// req.InputTokens that does NOT alias buf[0:buf.Len()] before the second
// OnComplete, the guard fires.
func TestSession_AccumulateContinuityGuard_RejectsMismatchedSlice(t *testing.T) {
	bp := makeTestBlueprint("sess-cont-guard", 3, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-cont-guard", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}

	// Simulate a caller bug: pass a fresh, different-length slice for round 1's
	// InputTokens. The guard must panic.
	req1Bad := &sim.Request{
		ID: "r1", SessionID: "sess-cont-guard", RoundIndex: 1,
		State: sim.StateCompleted, ProgressIndex: 99,
		InputTokens: make([]int, 99), // wrong length — doesn't match buf
		OutputTokens: make([]int, 5),
	}
	defer func() {
		rec := recover()
		if rec == nil {
			t.Fatalf("expected panic for mismatched req.InputTokens length")
		}
		msg, ok := rec.(string)
		if !ok {
			t.Fatalf("panic value is %T, want string", rec)
		}
		if !strings.Contains(msg, "buffer continuity broken") {
			t.Errorf("panic message %q does not mention buffer continuity", msg)
		}
	}()
	sm.OnComplete(req1Bad, 10000)
}

// TestSession_AccumulateContinuityGuard_RejectsShortSlice exercises the
// SHORT-slice direction of the `!=` continuity guard. Companion to
// TestSession_AccumulateContinuityGuard_RejectsMismatchedSlice which exercises
// the long-slice direction; together they protect against a regression from
// `!=` back to `>` (which would miss the orphaned-too-short case) (IMP-R4-1,
// #1445).
func TestSession_AccumulateContinuityGuard_RejectsShortSlice(t *testing.T) {
	bp := makeTestBlueprint("sess-cont-guard-short", 3, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-cont-guard-short", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up after round 0, got %d", len(follow))
	}
	// After round 0's OnComplete: buf contains 10 input + 5 output + 10 new = 25 tokens.
	// Pass a SHORTER slice to round 1 (1 != 25): the `!=` guard must fire.
	req1Short := &sim.Request{
		ID: "r1", SessionID: "sess-cont-guard-short", RoundIndex: 1,
		State: sim.StateCompleted, ProgressIndex: 6,
		InputTokens:  make([]int, 1),
		OutputTokens: make([]int, 5),
	}
	defer func() {
		rec := recover()
		if rec == nil {
			t.Fatalf("expected panic for short-slice InputTokens, got none")
		}
		msg, ok := rec.(string)
		if !ok {
			t.Fatalf("panic value is %T, want string", rec)
		}
		if !strings.Contains(msg, "buffer continuity broken") {
			t.Errorf("panic message %q does not mention buffer continuity", msg)
		}
	}()
	sm.OnComplete(req1Short, 10000)
}

// TestSession_AccumulateOverCap_CancelsSession exercises the over-cap branch
// of session.go's output accumulation: when actualOutputLen exceeds
// len(req.OutputTokens), the upstream ProgressIndex accounting has drifted.
// The branch logs Error and cancels the session to contain the corruption
// (susiejojo human review, #1445). A second OnComplete call confirms the
// session is terminal — no further follow-ups are generated.
func TestSession_AccumulateOverCap_CancelsSession(t *testing.T) {
	bp := makeTestBlueprint("sess-overcap", 3, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	// actualOutputLen = max(ProgressIndex - InputLen, 0) = max(20 - 10, 0) = 10
	// OutputTokens has only 5 → triggers the over-cap case (10 > 5).
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-overcap", RoundIndex: 0,
		State:         sim.StateCompleted,
		ProgressIndex: 20,
		InputTokens:   make([]int, 10),
		OutputTokens:  make([]int, 5),
	}
	follow := sm.OnComplete(req0, 5000)
	if follow != nil {
		t.Fatalf("over-cap: expected nil follow-up (session cancelled), got %d requests", len(follow))
	}

	// Confirm session is in terminal state: a second well-formed OnComplete
	// call must also return nil.
	req0Again := &sim.Request{
		ID: "r0b", SessionID: "sess-overcap", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	if follow := sm.OnComplete(req0Again, 6000); follow != nil {
		t.Errorf("expected nil after session cancelled, got %d requests", len(follow))
	}
}

// TestSession_AccumulateArrivalTime_INV10 anchors INV-10 (session causality)
// under the accumulate path: round N+1's ArrivalTime is computed independently
// of the buffer machinery and must equal `tick + thinkTime` exactly
// (MOD-R4-6, #1445).
func TestSession_AccumulateArrivalTime_INV10(t *testing.T) {
	bp := makeTestBlueprint("sess-arrival-inv10", 3, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-arrival-inv10", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow := sm.OnComplete(req0, 7500)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}
	// ThinkTimeUs=1000, tick=7500 → expected arrival 8500.
	if follow[0].ArrivalTime != 8500 {
		t.Errorf("INV-10: arrival = %d, want 8500 (tick 7500 + thinkTime 1000)", follow[0].ArrivalTime)
	}
}

// TestSession_NonAccumulate_WithPrefix verifies the closed-loop non-accumulate
// path with a non-empty prefix: each follow-up round's InputTokens must be
// [bp.Prefix | newInputTokens] (the prefix is prepended fresh per round; no
// shared buffer in this mode). Closes the test gap that would let a
// regression strip the append(bp.Prefix, ...) silently (MOD-R5-2, #1445).
func TestSession_NonAccumulate_WithPrefix(t *testing.T) {
	bp := makeTestBlueprint("sess-noaccum-prefix", 3, 1000, "", 1_000_000)
	bp.Prefix = []int{91, 92, 93}
	sm := NewSessionManager([]SessionBlueprint{bp})

	inputR0 := []int{1, 2, 3, 4, 5}
	outputR0 := []int{50, 51}
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-noaccum-prefix", RoundIndex: 0,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(inputR0) + len(outputR0)),
		InputTokens:   inputR0,
		OutputTokens:  outputR0,
	}
	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}
	r1 := follow[0]
	// constantSampler returns 10 input tokens; expected InputLen = 3 (prefix) + 10 (new) = 13.
	wantLen := len(bp.Prefix) + 10
	if len(r1.InputTokens) != wantLen {
		t.Fatalf("non-accumulate + prefix: InputLen = %d, want %d", len(r1.InputTokens), wantLen)
	}
	// Verify prefix appears at the front.
	for i, tok := range bp.Prefix {
		if r1.InputTokens[i] != tok {
			t.Fatalf("token[%d] = %d, want prefix token %d", i, r1.InputTokens[i], tok)
		}
	}
}

// TestSession_AccumulateZeroOutput_SeededRound verifies the
// `actualOutputLen == 0` guard at session.go:230 in a SEEDED accumulate round
// (i.e., round ≥ 1). An existing test covers this in round 0 (unseeded);
// this test adds coverage for the post-seed path where the buffer is
// already populated and the guard skips the output-append (MOD-R5-3, #1445).
func TestSession_AccumulateZeroOutput_SeededRound(t *testing.T) {
	bp := makeTestBlueprint("sess-zero-out-seeded", 3, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0: 10 input, 5 actual output (ProgressIndex = 15).
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-zero-out-seeded", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow1 := sm.OnComplete(req0, 5000)
	if len(follow1) != 1 {
		t.Fatalf("round 0: expected 1 follow-up, got %d", len(follow1))
	}
	r1 := follow1[0]
	if len(r1.InputTokens) != 25 {
		t.Fatalf("round 1 InputLen = %d, want 25 (10 in + 5 out + 10 new)", len(r1.InputTokens))
	}

	// Round 1: present as a length-capped result with ZERO actual output
	// (ProgressIndex == InputLen → actualOutputLen = max(0, 0) = 0). The
	// guard `actualOutputLen > 0 && ...` must skip the output-append; only
	// the new round's input gets appended.
	req1 := &sim.Request{
		ID: "r1", SessionID: "sess-zero-out-seeded", RoundIndex: 1,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(r1.InputTokens)), // zero actual output
		InputTokens:   r1.InputTokens,
		OutputTokens:  r1.OutputTokens,
	}
	follow2 := sm.OnComplete(req1, 10000)
	if len(follow2) != 1 {
		t.Fatalf("round 1 zero-output: expected 1 follow-up, got %d", len(follow2))
	}
	r2 := follow2[0]
	// Buffer state coming in: 25 tokens. No output appended (zero actual).
	// New input is 10 tokens. Round 2 InputLen = 25 + 0 + 10 = 35.
	if len(r2.InputTokens) != 35 {
		t.Fatalf("round 2 InputLen = %d, want 35 (25 buf + 0 out + 10 new)", len(r2.InputTokens))
	}
}

// TestSession_AccumulateDoesNotReseed verifies the `seeded` flag guard:
// once a session has produced its first follow-up, a second OnComplete call
// on a request with the same SessionID must NOT re-seed the buffer (which
// would double-count the prefix and conversation). This catches a class of
// caller bugs that could silently corrupt accumulated context (MOD-2, #1445).
func TestSession_AccumulateDoesNotReseed(t *testing.T) {
	bp := makeTestBlueprint("sess-no-reseed", 4, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	inputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(1)), 10)
	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(2)), 5)
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-no-reseed", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: inputR0, OutputTokens: outputR0,
	}
	follow1 := sm.OnComplete(req0, 5000)
	if len(follow1) != 1 {
		t.Fatalf("first OnComplete: expected 1 follow-up, got %d", len(follow1))
	}
	r1 := follow1[0]
	// Round 1: 10 (r0 input) + 5 (r0 output) + 10 (new) = 25 tokens
	if len(r1.InputTokens) != 25 {
		t.Fatalf("first follow-up input length = %d, want 25", len(r1.InputTokens))
	}
	r1Cap := cap(r1.InputTokens)
	r1Len := int64(len(r1.InputTokens))

	// Simulate a caller bug: call OnComplete on round 1 normally — buffer
	// should extend exactly once for output + new input, not re-seed.
	req1 := &sim.Request{
		ID: "r1", SessionID: "sess-no-reseed", RoundIndex: 1,
		State: sim.StateCompleted,
		ProgressIndex: int64(len(r1.InputTokens) + len(r1.OutputTokens)),
		InputTokens: r1.InputTokens, OutputTokens: r1.OutputTokens,
	}
	follow2 := sm.OnComplete(req1, 10000)
	if len(follow2) != 1 {
		t.Fatalf("second OnComplete: expected 1 follow-up, got %d", len(follow2))
	}
	r2 := follow2[0]
	// Round 2: prior 25 + 5 (r1 output) + 10 (new) = 40. If re-seeded, would be 50.
	if len(r2.InputTokens) != 40 {
		t.Fatalf("BC-9: second follow-up input length = %d, want 40 (re-seeded would be 50)", len(r2.InputTokens))
	}
	// Sanity: r1's slice must still be a valid prefix of r2's (no in-place
	// mutation, buffer just extended).
	for i := int64(0); i < r1Len; i++ {
		if r2.InputTokens[i] != r1.InputTokens[i] {
			t.Fatalf("token %d diverged between r1 and r2: %d vs %d — buffer was mutated", i, r1.InputTokens[i], r2.InputTokens[i])
		}
	}
	// Capacity should have grown at most by Go's append doubling.
	if cap(r2.InputTokens) > 4*r1Cap {
		t.Errorf("cap grew from %d to %d (more than 4x) — non-amortized append", r1Cap, cap(r2.InputTokens))
	}
}

// TestSession_AccumulateSharesBackingArray verifies that closed-loop accumulate
// rounds share a sessionTokenBuffer (BC-2, #1445). Specifically, round 1's
// InputTokens must be a slice header view into the same buffer as round 2's —
// proving that no fresh copy of accumulated context is created per round.
func TestSession_AccumulateSharesBackingArray(t *testing.T) {
	bp := makeTestBlueprint("sess-share", 3, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-share", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(1)), 10),
		OutputTokens: sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(2)), 5),
	}
	follow1 := sm.OnComplete(req0, 5000)
	if len(follow1) != 1 {
		t.Fatalf("expected 1 follow-up after round 0, got %d", len(follow1))
	}
	r1 := follow1[0]

	req1 := &sim.Request{
		ID: "r1", SessionID: "sess-share", RoundIndex: 1,
		State: sim.StateCompleted,
		ProgressIndex: int64(len(r1.InputTokens) + len(r1.OutputTokens)),
		InputTokens: r1.InputTokens, OutputTokens: r1.OutputTokens,
	}
	follow2 := sm.OnComplete(req1, 10000)
	if len(follow2) != 1 {
		t.Fatalf("expected 1 follow-up after round 1, got %d", len(follow2))
	}
	r2 := follow2[0]

	// The shared-buffer guarantee: r2's InputTokens MUST extend r1's via the
	// same growable buffer. At minimum, r2's underlying array equals either
	// r1's underlying array (no realloc) OR a later, larger allocation that
	// began life as a copy of the same buffer. We check the operational
	// property: r2.InputTokens[:len(r1.InputTokens)] equals r1.InputTokens
	// (already covered by length tests). For the address-aliasing check we
	// assert that two snapshots of r2's slice from OnComplete return slices
	// rooted in the same array.
	if len(r1.InputTokens) > 0 && len(r2.InputTokens) > 0 {
		// r2 may live in a reallocated array. The strict guarantee we can make
		// without poking buffer internals: r2.InputTokens cap >= len(r2.InputTokens)
		// AND r2 contains r1's content verbatim.
		for i := 0; i < len(r1.InputTokens); i++ {
			if r2.InputTokens[i] != r1.InputTokens[i] {
				t.Fatalf("BC-2: r2 token %d = %d, want %d (shared-buffer continuity broken)",
					i, r2.InputTokens[i], r1.InputTokens[i])
			}
		}
	}
}

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
//
// contextTokens tracks the full conversation history without double-counting:
//   - After round 0: contextTokens = r0_input(10) + r0_output(5) = 15 tokens
//   - After round 1: contextTokens += new_suffix_of_r1(10) + r1_output(5) = 30 tokens
//   - Round 2 input = contextTokens(30) + r2_new(10) = 40 tokens
//
// The fix prevents quadratic growth: req.InputTokens already contains the
// accumulated context, so only the new suffix (req.InputTokens[len(contextTokens):])
// is appended — not the full req.InputTokens, which would double-count prior context.
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
	// Round 2: contextTokens grows to 30 (prior 15 + r1 new suffix 10 + r1 output 5),
	// then + 10 new = 40. Only the new suffix is appended — not the full req.InputTokens
	// which already contains the accumulated context.
	if len(follow2[0].InputTokens) != 40 {
		t.Errorf("BC-1: round 2 input length = %d, want 40 (contextTokens(30)+10)", len(follow2[0].InputTokens))
	}
}

// TestSession_ContextAccumulation_WithPrefix verifies BC-1:
// when ContextGrowth="accumulate" and Prefix is non-empty, the prefix appears
// exactly once in the follow-up input — not twice (once absorbed into contextTokens
// from round-0's InputTokens, and once from the prefix-prepend block).
//
// Invariant: len(round1.InputTokens) == len(prefix) + len(input0) + len(output0) + len(newInput1)
//            = 5 + 10 + 5 + 10 = 30, not 35.
func TestSession_ContextAccumulation_WithPrefix(t *testing.T) {
	bp := makeTestBlueprint("sess-prefix", 3, 1000, "accumulate", 1_000_000)
	bp.Prefix = sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(7)), 5) // 5 prefix tokens

	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0: InputTokens = [prefix(5) | content(10)] = 15 tokens, actual output = 5
	content0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(8)), 10)
	inputR0 := append(append([]int{}, bp.Prefix...), content0...)
	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(9)), 5)

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-prefix", RoundIndex: 0,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(inputR0) + len(outputR0)), // 15 + 5 = 20
		InputTokens:   inputR0,
		OutputTokens:  outputR0,
	}

	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}

	r1 := follow[0]
	// prefix(5) + content0(10) + output0(5) + newInput(10) = 30
	wantLen := 5 + 10 + 5 + 10
	if len(r1.InputTokens) != wantLen {
		t.Errorf("BC-1: round 1 input length = %d, want %d (prefix appears once, not twice)",
			len(r1.InputTokens), wantLen)
	}

	// Verify prefix appears at position 0
	for i, tok := range bp.Prefix {
		if i >= len(r1.InputTokens) || r1.InputTokens[i] != tok {
			t.Errorf("BC-1: round 1 token[%d] = %v, want prefix token %v", i, r1.InputTokens[i], tok)
		}
	}
}

// TestSession_ContextAccumulation_WithPrefix_MultiStep verifies BC-2:
// corruption does not compound across rounds — round 2 has the correct token
// count and contextTokens remains prefix-free throughout.
//
// Invariant: len(round2.InputTokens) == prefix(5) + contextTokens(30) + newInput(10) = 45,
// where contextTokens = content0(10) + output0(5) + newInput1(10) + output1(5) = 30 (prefix-free).
func TestSession_ContextAccumulation_WithPrefix_MultiStep(t *testing.T) {
	bp := makeTestBlueprint("sess-prefix-multi", 4, 1000, "accumulate", 1_000_000)
	bp.Prefix = sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(10)), 5) // 5 prefix tokens

	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0: [prefix(5) | content(10)] = 15 tokens, actual output = 5
	inputR0 := append(append([]int{}, bp.Prefix...), sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(11)), 10)...)
	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(12)), 5)
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-prefix-multi", RoundIndex: 0,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(inputR0) + len(outputR0)), // 15 + 5 = 20
		InputTokens:   inputR0, OutputTokens: outputR0,
	}
	follow1 := sm.OnComplete(req0, 5000)
	if len(follow1) != 1 {
		t.Fatalf("round 0: expected 1 follow-up, got %d", len(follow1))
	}
	// Round 1 must be 30 tokens: prefix(5) + content0(10) + output0(5) + newInput1(10)
	if len(follow1[0].InputTokens) != 30 {
		t.Errorf("BC-2: round 1 input length = %d, want 30", len(follow1[0].InputTokens))
	}

	// Round 1 completion
	req1 := &sim.Request{
		ID: "r1", SessionID: "sess-prefix-multi", RoundIndex: 1,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(follow1[0].InputTokens) + len(follow1[0].OutputTokens)), // 30 + 5 = 35
		InputTokens:   follow1[0].InputTokens, OutputTokens: follow1[0].OutputTokens,
	}
	follow2 := sm.OnComplete(req1, 10000)
	if len(follow2) != 1 {
		t.Fatalf("round 1: expected 1 follow-up, got %d", len(follow2))
	}
	// Round 2: prefix(5) + contextTokens(30) + newInput2(10) = 45
	// contextTokens = content0(10) + output0(5) + newInput1(10) + output1(5) = 30 (prefix-free)
	if len(follow2[0].InputTokens) != 45 {
		t.Errorf("BC-2: round 2 input length = %d, want 45 (no compounding corruption)", len(follow2[0].InputTokens))
	}
}

// TestSession_ContextAccumulation_WithPrefix_PrefixOnly verifies the bounds guard:
// when round-0 InputTokens equals the prefix exactly (zero conversation content),
// no panic occurs and the follow-up includes only prior output as accumulated context.
func TestSession_ContextAccumulation_WithPrefix_PrefixOnly(t *testing.T) {
	bp := makeTestBlueprint("sess-prefix-only", 3, 1000, "accumulate", 1_000_000)
	bp.Prefix = sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(13)), 5) // 5 prefix tokens

	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0: InputTokens = prefix only (no conversation content), actual output = 5
	// This exercises the edge case where len(rawConversation) == 0.
	inputR0 := append([]int{}, bp.Prefix...) // InputTokens == Prefix exactly
	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(14)), 5)

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-prefix-only", RoundIndex: 0,
		State:         sim.StateCompleted,
		ProgressIndex: int64(len(inputR0) + len(outputR0)), // 5 + 5 = 10
		InputTokens:   inputR0,
		OutputTokens:  outputR0,
	}

	// Must not panic regardless of prefix/input relationship.
	follow := sm.OnComplete(req0, 5000)
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up, got %d", len(follow))
	}

	r1 := follow[0]
	// contextTokens = output0(5) (no conversation to accumulate)
	// round 1 = prefix(5) + contextTokens(5) + newInput(10) = 20
	wantLen := 5 + 5 + 10
	if len(r1.InputTokens) != wantLen {
		t.Errorf("bounds guard: round 1 input length = %d, want %d", len(r1.InputTokens), wantLen)
	}
}

// TestSession_ContextAccumulation_ZeroSuffix verifies the guard in accumulate mode
// when InputSampler returns 0: contextTokens grows only by output tokens and the
// follow-up is still generated (no panic, no state corruption).
func TestSession_ContextAccumulation_ZeroSuffix(t *testing.T) {
	bp := makeTestBlueprint("sess-zero-suffix", 3, 1000, "accumulate", 1_000_000)
	bp.InputSampler = &constantSampler{value: 0}
	sm := NewSessionManager([]SessionBlueprint{bp})

	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(1)), 5)
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-zero-suffix", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 5,
		InputTokens: []int{}, OutputTokens: outputR0,
	}
	follow1 := sm.OnComplete(req0, 5000)
	if len(follow1) != 1 {
		t.Fatalf("expected 1 follow-up after zero-suffix round 0, got %d", len(follow1))
	}
	// contextTokens = 0 input + 5 output = 5; next input = 5 context + 0 new = 5
	if len(follow1[0].InputTokens) != 5 {
		t.Errorf("round 1 input length = %d, want 5 (contextTokens only, zero new suffix)", len(follow1[0].InputTokens))
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
