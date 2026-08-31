package workload

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// mkReq builds a minimal COMPLETED replayed request for re-export reconstruction tests.
// ProgressIndex = inputLen + outputLen models a fully-generated round (actualOutput ==
// oracle MaxOutputLen), the common non-length-capped case. mkReqAO models a round whose
// actual accumulated output differs from the oracle budget.
func mkReq(id, sessionID string, round, inputLen, outputLen int, arrival int64, state sim.RequestState) *sim.Request {
	return mkReqAO(id, sessionID, round, inputLen, outputLen, outputLen, arrival, state)
}

// mkReqAO is mkReq with an explicit actualOutput (sets ProgressIndex = inputLen +
// actualOutput) distinct from the oracle outputLen (= len(OutputTokens)).
func mkReqAO(id, sessionID string, round, inputLen, outputLen, actualOutput int, arrival int64, state sim.RequestState) *sim.Request {
	return &sim.Request{
		ID:            id,
		SessionID:     sessionID,
		RoundIndex:    round,
		InputTokens:   make([]sim.TokenID, inputLen),
		OutputTokens:  make([]sim.TokenID, outputLen),
		ProgressIndex: int64(inputLen + actualOutput),
		ArrivalTime:   arrival,
		State:         state,
	}
}

// T1 (BC-1): accumulate re-export re-derives per-round DELTAS and re-emits an
// input_tokens_reset marker on a compaction round, from the replayed ABSOLUTE inputs.
// Absolutes 100, 205, 50 with round-0 output 50: delta1 = 205-100-50 = 55; round 2's
// absolute (50) is below 205+30, so its delta clamps to 0 and it carries reset=50.
func TestReExportClosedLoopRecords_Accumulate_DeltasAndResets(t *testing.T) {
	r0 := mkReq("request_0", "s1", 0, 100, 50, 0, sim.StateCompleted)
	r1 := mkReq("session_s1_round_1_1", "s1", 1, 205, 30, 1_000_000, sim.StateCompleted)
	r2 := mkReq("session_s1_round_2_2", "s1", 2, 50, 20, 2_000_000, sim.StateCompleted)
	// Round-0 comes from the `requests` slice; follow-ups from followUpRequests (any order).
	reqs := []*sim.Request{r0, r2, r1}
	think := map[string]int64{
		"session_s1_round_1_1": 1500,
		"session_s1_round_2_2": 2500,
	}

	recs, err := ReExportClosedLoopRecords(reqs, think, "accumulate")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(recs) != 3 {
		t.Fatalf("expected 3 records (all rounds captured), got %d", len(recs))
	}

	// Rounds present in order 0,1,2 with sequential RequestIDs.
	for i, rec := range recs {
		if rec.RoundIndex != i {
			t.Errorf("record %d: RoundIndex = %d, want %d", i, rec.RoundIndex, i)
		}
		if rec.RequestID != i {
			t.Errorf("record %d: RequestID = %d, want %d (sequential)", i, rec.RequestID, i)
		}
		if rec.SessionID != "s1" {
			t.Errorf("record %d: SessionID = %q, want s1", i, rec.SessionID)
		}
		if rec.Model != "" {
			t.Errorf("record %d: Model = %q, want empty (routing safety)", i, rec.Model)
		}
	}

	// Deltas: round 0 = full absolute (100); round 1 = 55; round 2 = 0 (compaction).
	wantDeltas := []int{100, 55, 0}
	for i, want := range wantDeltas {
		if recs[i].InputTokens != want {
			t.Errorf("round %d: InputTokens (delta) = %d, want %d", i, recs[i].InputTokens, want)
		}
	}

	// Output counts preserved (pre-determined len).
	wantOut := []int{50, 30, 20}
	for i, want := range wantOut {
		if recs[i].OutputTokens != want {
			t.Errorf("round %d: OutputTokens = %d, want %d", i, recs[i].OutputTokens, want)
		}
	}

	// Reset marker only on the compaction round (round 2), = the recorded absolute (50).
	if recs[0].InputTokensReset != nil {
		t.Errorf("round 0: InputTokensReset = %v, want nil", *recs[0].InputTokensReset)
	}
	if recs[1].InputTokensReset != nil {
		t.Errorf("round 1: InputTokensReset = %v, want nil", *recs[1].InputTokensReset)
	}
	if recs[2].InputTokensReset == nil || *recs[2].InputTokensReset != 50 {
		t.Errorf("round 2: InputTokensReset = %v, want &50", recs[2].InputTokensReset)
	}

	// Think: nil on round 0, recorded (non-nil) on rounds 1..N so re-replay uses the
	// recorded-think path.
	if recs[0].ThinkTimeUs != nil {
		t.Errorf("round 0: ThinkTimeUs = %v, want nil", *recs[0].ThinkTimeUs)
	}
	if recs[1].ThinkTimeUs == nil || *recs[1].ThinkTimeUs != 1500 {
		t.Errorf("round 1: ThinkTimeUs = %v, want &1500", recs[1].ThinkTimeUs)
	}
	if recs[2].ThinkTimeUs == nil || *recs[2].ThinkTimeUs != 2500 {
		t.Errorf("round 2: ThinkTimeUs = %v, want &2500", recs[2].ThinkTimeUs)
	}
}

// T1b (BC-1, #1630 root-cause): the delta law must use the round's ACTUAL accumulated
// output (ProgressIndex − InputLen), not the oracle len(OutputTokens). A round-0 whose
// actual output (19) is one below its oracle budget (20) must still yield the exact delta
// (abs₁ − abs₀ − actualOutput₀), while the emitted OutputTokens column keeps the oracle
// budget so re-replay reproduces the same completion.
func TestReExportClosedLoopRecords_Accumulate_ActualOutputVsOracle(t *testing.T) {
	// abs 100 → 149; round-0 oracle budget 20 but actual accumulated output 19.
	r0 := mkReqAO("request_0", "s1", 0, 100, 20, 19, 0, sim.StateCompleted)
	r1 := mkReq("session_s1_round_1_1", "s1", 1, 149, 15, 1_000_000, sim.StateCompleted)
	recs, err := ReExportClosedLoopRecords([]*sim.Request{r0, r1}, map[string]int64{"session_s1_round_1_1": 500}, "accumulate")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// delta₁ = 149 − 100 − actualOutput₀(19) = 30 (NOT 149−100−20 = 29).
	if recs[1].InputTokens != 30 {
		t.Errorf("round-1 delta = %d, want 30 (must use actualOutput 19, not oracle 20)", recs[1].InputTokens)
	}
	// Emitted OutputTokens column carries the ORACLE budget (20) so re-replay drives the
	// same completion.
	if recs[0].OutputTokens != 20 {
		t.Errorf("round-0 OutputTokens column = %d, want 20 (oracle MaxOutputLen)", recs[0].OutputTokens)
	}
}

// T2 (BC-3, BC-7): non-accumulate re-export emits ABSOLUTE per-round input, records the
// captured think on follow-ups, passes non-session single-shots through, and propagates a
// session's round-0 prefix metadata onto its follow-up records (no double-count).
func TestReExportClosedLoopRecords_NonAccumulate_AbsoluteThinkAndPrefix(t *testing.T) {
	// Prefix-free multi-round session.
	a0 := mkReq("request_0", "sA", 0, 100, 10, 0, sim.StateCompleted)
	a1 := mkReq("session_sA_round_1_1", "sA", 1, 40, 20, 5_000_000, sim.StateCompleted)
	// Session whose round 0 carries a prefix group; the follow-up input already includes
	// the prefix (len 10) prepended by OnComplete, but with PrefixLength==0.
	b0 := mkReq("request_1", "sB", 0, 30, 5, 0, sim.StateCompleted)
	b0.PrefixGroup = "g"
	b0.PrefixLength = 10
	b1 := mkReq("session_sB_round_1_2", "sB", 1, 25, 8, 6_000_000, sim.StateCompleted) // 10 prefix + 15 new
	// Non-session single-shot.
	ns := mkReq("request_2", "", 0, 12, 3, 0, sim.StateCompleted)

	reqs := []*sim.Request{a0, b0, ns, a1, b1}
	think := map[string]int64{
		"session_sA_round_1_1": 700,
		"session_sB_round_1_2": 800,
	}

	recs, err := ReExportClosedLoopRecords(reqs, think, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(recs) != 5 {
		t.Fatalf("expected 5 records, got %d", len(recs))
	}

	byKey := func(sid string, round int) *TraceRecord {
		for i := range recs {
			if recs[i].SessionID == sid && recs[i].RoundIndex == round {
				return &recs[i]
			}
		}
		t.Fatalf("record %s/round %d not found", sid, round)
		return nil
	}

	// sA: absolute per-round input, no reset, think on round 1.
	if r := byKey("sA", 0); r.InputTokens != 100 || r.InputTokensReset != nil {
		t.Errorf("sA r0: InputTokens=%d reset=%v, want 100/nil", r.InputTokens, r.InputTokensReset)
	}
	if r := byKey("sA", 1); r.InputTokens != 40 || r.ThinkTimeUs == nil || *r.ThinkTimeUs != 700 {
		t.Errorf("sA r1: InputTokens=%d think=%v, want 40/&700", r.InputTokens, r.ThinkTimeUs)
	}

	// sB prefix propagation: round-0 records suffix (30-10=20) with the group; the
	// follow-up inherits prefix_group/prefix_length and records the suffix (25-10=15).
	if r := byKey("sB", 0); r.InputTokens != 20 || r.PrefixGroup != "g" || r.PrefixLength != 10 {
		t.Errorf("sB r0: InputTokens=%d group=%q len=%d, want 20/g/10", r.InputTokens, r.PrefixGroup, r.PrefixLength)
	}
	if r := byKey("sB", 1); r.InputTokens != 15 || r.PrefixGroup != "g" || r.PrefixLength != 10 {
		t.Errorf("sB r1: InputTokens=%d group=%q len=%d, want 15/g/10 (no prefix double-count)", r.InputTokens, r.PrefixGroup, r.PrefixLength)
	}

	// Non-session passthrough (absolute, round 0, no session).
	if r := byKey("", 0); r.InputTokens != 12 {
		t.Errorf("non-session: InputTokens=%d, want 12", r.InputTokens)
	}

	// Sequential RequestIDs.
	for i := range recs {
		if recs[i].RequestID != i {
			t.Errorf("record %d: RequestID=%d, want %d", i, recs[i].RequestID, i)
		}
	}
}

// T3 (R1): a session with a gap in round indices is a captured-run corruption — error
// rather than emit a corpus the loader would reject/misinterpret.
func TestReExportClosedLoopRecords_NonConsecutiveRounds_Error(t *testing.T) {
	r0 := mkReq("request_0", "s1", 0, 10, 5, 0, sim.StateCompleted)
	r2 := mkReq("session_s1_round_2_1", "s1", 2, 20, 5, 1_000_000, sim.StateCompleted) // gap: no round 1
	_, err := ReExportClosedLoopRecords([]*sim.Request{r0, r2}, nil, "accumulate")
	if err == nil {
		t.Fatal("expected error for non-consecutive round indices, got nil")
	}
}

// T4 (INV-6): identical inputs produce byte-identical records across calls.
func TestReExportClosedLoopRecords_Deterministic(t *testing.T) {
	build := func() []*sim.Request {
		return []*sim.Request{
			mkReq("request_0", "s1", 0, 100, 50, 0, sim.StateCompleted),
			mkReq("session_s1_round_2_2", "s1", 2, 50, 20, 2_000_000, sim.StateCompleted),
			mkReq("session_s1_round_1_1", "s1", 1, 205, 30, 1_000_000, sim.StateCompleted),
			mkReq("request_3", "s2", 0, 60, 10, 0, sim.StateCompleted),
			mkReq("session_s2_round_1_9", "s2", 1, 90, 12, 3_000_000, sim.StateCompleted),
		}
	}
	think := map[string]int64{
		"session_s1_round_1_1": 1500,
		"session_s1_round_2_2": 2500,
		"session_s2_round_1_9": 1200,
	}
	a, err := ReExportClosedLoopRecords(build(), think, "accumulate")
	if err != nil {
		t.Fatalf("call A: %v", err)
	}
	b, err := ReExportClosedLoopRecords(build(), think, "accumulate")
	if err != nil {
		t.Fatalf("call B: %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Errorf("records differ across identical calls (INV-6):\nA=%+v\nB=%+v", a, b)
	}
}

// TestReExportClosedLoopRecords_StatusMapping covers reexportStatus's non-"ok" branches
// (susiejojo non-blocking observation on #1645): a completed round → "ok", a terminal
// timed-out round → "timeout", a still-running (incomplete) round → "incomplete". Status
// is informational (the replay loader does not read it), but the mapping should be pinned.
func TestReExportClosedLoopRecords_StatusMapping(t *testing.T) {
	// Accumulate session s1: round 0 completed, round 1 timed out (a terminal round —
	// OnComplete cancels the session, so a timeout is the session's last captured round).
	r0 := mkReq("request_0", "s1", 0, 100, 20, 0, sim.StateCompleted)
	r1 := mkReq("session_s1_round_1_1", "s1", 1, 140, 10, 1_000_000, sim.StateTimedOut)
	// Single-round session s2 still running at horizon → "incomplete" (default branch).
	inc := mkReq("request_2", "s2", 0, 50, 5, 0, sim.StateRunning)

	recs, err := ReExportClosedLoopRecords([]*sim.Request{r0, r1, inc}, map[string]int64{"session_s1_round_1_1": 500}, "accumulate")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Deterministic emission order: session s1 (rounds 0,1) then s2 (round 0).
	if len(recs) != 3 {
		t.Fatalf("expected 3 records, got %d", len(recs))
	}
	want := []struct {
		session string
		round   int
		status  string
	}{
		{"s1", 0, "ok"},
		{"s1", 1, "timeout"},
		{"s2", 0, "incomplete"},
	}
	for i, w := range want {
		if recs[i].SessionID != w.session || recs[i].RoundIndex != w.round {
			t.Fatalf("record %d = %s/round %d, want %s/round %d", i, recs[i].SessionID, recs[i].RoundIndex, w.session, w.round)
		}
		if recs[i].Status != w.status {
			t.Errorf("record %d (%s round %d): Status = %q, want %q", i, w.session, w.round, recs[i].Status, w.status)
		}
	}
}

// TestAccumulatedOutputLen_ClampsNegative pins accumulatedOutputLen's clamp-at-0
// (susiejojo non-blocking observation on #1645): a ProgressIndex below InputLen —
// unreachable under current sim guarantees, but the clamp must hold — yields 0, not a
// negative that would corrupt the delta law; a normal ProgressIndex yields the difference.
func TestAccumulatedOutputLen_ClampsNegative(t *testing.T) {
	neg := &sim.Request{InputTokens: make([]sim.TokenID, 100), ProgressIndex: 40}
	if got := accumulatedOutputLen(neg); got != 0 {
		t.Errorf("accumulatedOutputLen with ProgressIndex(40) < InputLen(100) = %d, want 0 (clamped)", got)
	}
	normal := &sim.Request{InputTokens: make([]sim.TokenID, 100), ProgressIndex: 118}
	if got := accumulatedOutputLen(normal); got != 18 {
		t.Errorf("accumulatedOutputLen with ProgressIndex(118), InputLen(100) = %d, want 18", got)
	}
	// Exactly at the boundary (no output generated yet) → 0.
	boundary := &sim.Request{InputTokens: make([]sim.TokenID, 100), ProgressIndex: 100}
	if got := accumulatedOutputLen(boundary); got != 0 {
		t.Errorf("accumulatedOutputLen at ProgressIndex==InputLen = %d, want 0", got)
	}
}
