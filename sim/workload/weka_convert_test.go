package workload

import (
	"testing"
)

// A 3-main-turn session with a type:"subagent" group interleaved between turns
// 0 and 1. Main turns: in 100/150/215, out 10/20/5; the sub-agent group at
// t=2.0 (dropped) sits between the 0s and 6s main turns. api_time on each main
// turn drives think recompute; think_time is the recorded value (equals the
// recomputed gap on this subagent-free-between-consecutive-mains chain).
const wekaThreeTurnWithSubagent = `{
  "id": "sess-1",
  "models": ["claude-sonnet-4"],
  "block_size": 64,
  "hash_id_scope": "local",
  "requests": [
    {"type":"n","t":0.0,"model":"claude-sonnet-4","in":100,"out":10,"hash_ids":[1,2],"api_time":1.0},
    {"type":"subagent","t":2.0,"agent_id":"sa-1","subagent_type":"search","duration_ms":3000,"total_tokens":500,"requests":[{"type":"n","t":2.5,"in":9999,"out":9999,"api_time":0.5}]},
    {"type":"s","t":6.0,"model":"claude-sonnet-4","in":150,"out":20,"hash_ids":[1,2,3],"api_time":2.0,"think_time":5.0,"ttft":0.3},
    {"type":"n","t":10.0,"model":"claude-sonnet-4","in":215,"out":5,"hash_ids":[1,2,3,4],"api_time":1.0,"think_time":2.0}
  ]
}`

// T1 — reader core: delta reconstruction, RoundIndex over main turns only,
// sub-agent group excluded, Model routing-safety.
func TestConvertWekaSession_DeltaReconstructionAndSubagentSkip(t *testing.T) {
	recs, err := ConvertWekaSession([]byte(wekaThreeTurnWithSubagent), WekaConvertOptions{ContextGrowth: "accumulate", MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertWekaSession: %v", err)
	}
	// 3 main turns → 3 records; the type:"subagent" group produces none.
	if len(recs) != 3 {
		t.Fatalf("got %d records, want 3 (sub-agent group excluded)", len(recs))
	}
	// RoundIndex 0..N over main turns in stream order.
	for i, r := range recs {
		if r.RoundIndex != i {
			t.Errorf("record %d RoundIndex = %d, want %d (main turns only, stream order)", i, r.RoundIndex, i)
		}
	}
	// Round 0: full first prompt.
	if recs[0].InputTokens != 100 || recs[0].OutputTokens != 10 {
		t.Errorf("round0 = in %d out %d, want 100/10", recs[0].InputTokens, recs[0].OutputTokens)
	}
	// Round 1 delta: 150 - 100 - 10 = 40.
	if recs[1].InputTokens != 40 {
		t.Errorf("round1 delta = %d, want 40", recs[1].InputTokens)
	}
	// Round 2 delta: 215 - 150 - 20 = 45.
	if recs[2].InputTokens != 45 {
		t.Errorf("round2 delta = %d, want 45", recs[2].InputTokens)
	}
	// Delta reconstruction law: prefix + running(delta+output) == recorded totals.
	wantTotals := []int{100, 150, 215}
	running := 0
	for i, r := range recs {
		if i == 0 {
			running = r.InputTokens
		} else {
			running += r.InputTokens
		}
		if running != wantTotals[i] {
			t.Errorf("round %d reconstructed input = %d, want %d", i, running, wantTotals[i])
		}
		running += r.OutputTokens
	}
	// SessionID from `id`.
	if recs[0].SessionID != "sess-1" {
		t.Errorf("session id = %q, want sess-1", recs[0].SessionID)
	}
	// Model MUST be empty even though every turn records "claude-sonnet-4":
	// TraceRecord.Model is routing-significant, so a recorded name differing from
	// --model would drop every request at routing.
	for i, r := range recs {
		if r.Model != "" {
			t.Errorf("round %d Model = %q, want empty (recorded model must not reach the routing-significant field)", i, r.Model)
		}
	}
}

// T2 — think recompute equals both the formula and the recorded think_time on a
// subagent-free chain; and the sub-agent group's wall-clock is absorbed into the
// FOLLOWING main turn's think gap (BC-2, BC-3). The fixture above places a
// sub-agent group (t=2..~5) between main turn 0 (t=0, api 1.0) and main turn 1
// (t=6): the recomputed think for main turn 1 is 6.0 − 0.0 − 1.0 = 5.0s, which
// spans and thus ABSORBS the sub-agent's wall-clock. It equals the recorded
// think_time (5.0). Main turn 2: 10.0 − 6.0 − 2.0 = 2.0s == recorded 2.0.
func TestConvertWekaSession_ThinkTimeRecompute(t *testing.T) {
	recs, err := ConvertWekaSession([]byte(wekaThreeTurnWithSubagent), WekaConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertWekaSession: %v", err)
	}
	if recs[0].ThinkTimeUs != 0 {
		t.Errorf("round0 think = %d, want 0 (round 0 has no predecessor)", recs[0].ThinkTimeUs)
	}
	// Round 1 think absorbs the skipped sub-agent group's wall-clock: 5.0s.
	if recs[1].ThinkTimeUs != 5_000_000 {
		t.Errorf("round1 think = %d, want 5e6 (t1−t0−api0 = 6−0−1, absorbs sub-agent wall-clock; == recorded think_time 5.0)", recs[1].ThinkTimeUs)
	}
	// Round 2 think: consecutive main turns, no sub-agent between: 2.0s.
	if recs[2].ThinkTimeUs != 2_000_000 {
		t.Errorf("round2 think = %d, want 2e6 (t2−t1−api1 = 10−6−2; == recorded think_time 2.0)", recs[2].ThinkTimeUs)
	}
}

// T3 — overlapping main turns (round i arrives before round i-1's response
// elapsed → negative raw gap, real in these traces) yield ThinkTimeUs == 0,
// never negative (INV-3, BC-4). Mirrors the input-delta clamp.
func TestConvertWekaSession_NegativeGapThinkClampsToZero(t *testing.T) {
	// Round 1 arrives at t=1.0 but round 0's response ends at 0.0+2.0=2.0 →
	// raw gap 1.0−0.0−2.0 = −1.0s → clamp to 0.
	j := `{"id":"s","requests":[
	  {"type":"n","t":0.0,"in":100,"out":10,"api_time":2.0},
	  {"type":"n","t":1.0,"in":150,"out":20,"api_time":1.0}
	]}`
	recs, err := ConvertWekaSession([]byte(j), WekaConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertWekaSession: %v", err)
	}
	if recs[1].ThinkTimeUs != 0 {
		t.Errorf("round1 think = %d, want 0 (negative raw gap clamped; never negative)", recs[1].ThinkTimeUs)
	}
}

// T4 — --max-think-time caps the recomputed gap (>0); 0 leaves it uncapped.
func TestConvertWekaSession_MaxThinkTimeCapAndUncapped(t *testing.T) {
	// 100s think gap (t1=101, api0=1 → 100s), no sub-agent.
	j := `{"id":"s","requests":[
	  {"type":"n","t":0.0,"in":50,"out":5,"api_time":1.0},
	  {"type":"n","t":101.0,"in":70,"out":8,"api_time":1.0}
	]}`
	capped, err := ConvertWekaSession([]byte(j), WekaConvertOptions{MinRounds: 1, MaxThinkTimeUs: 15_000_000})
	if err != nil {
		t.Fatalf("capped: %v", err)
	}
	if capped[1].ThinkTimeUs != 15_000_000 {
		t.Errorf("capped think = %d, want 15e6", capped[1].ThinkTimeUs)
	}
	uncapped, err := ConvertWekaSession([]byte(j), WekaConvertOptions{MinRounds: 1, MaxThinkTimeUs: 0})
	if err != nil {
		t.Fatalf("uncapped: %v", err)
	}
	if uncapped[1].ThinkTimeUs != 100_000_000 {
		t.Errorf("uncapped think = %d, want 100e6 (no cap when MaxThinkTimeUs == 0)", uncapped[1].ThinkTimeUs)
	}
}

// T5 — non-monotone input (context compaction/trimming): in_1 < in_0 + out_0 →
// raw input delta negative → clamped to 0 by the shared encoder, never negative
// (BC-1 edge). Reconstructed absolute over-counts by the clamped deficit
// (accepted, documented deviation — same as OTel).
func TestConvertWekaSession_NonMonotoneInputClampsToZero(t *testing.T) {
	// Round 1 recorded input (120) < round 0 input+output (200+50=250).
	j := `{"id":"s","requests":[
	  {"type":"n","t":0.0,"in":200,"out":50,"api_time":1.0},
	  {"type":"n","t":5.0,"in":120,"out":8,"api_time":1.0}
	]}`
	recs, err := ConvertWekaSession([]byte(j), WekaConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertWekaSession: %v", err)
	}
	if recs[0].InputTokens != 200 {
		t.Errorf("round0 input = %d, want 200", recs[0].InputTokens)
	}
	if recs[1].InputTokens != 0 {
		t.Errorf("round1 delta = %d, want 0 (clamped, never negative)", recs[1].InputTokens)
	}
	// Reconstructed round-1 absolute = 200 + 50 + 0 = 250 (over-counts recorded 120 by 130).
	reconstructed := recs[0].InputTokens + recs[0].OutputTokens + recs[1].InputTokens
	if reconstructed != 250 {
		t.Errorf("reconstructed round1 absolute = %d, want 250 (over-counts recorded 120 by the clamped 130)", reconstructed)
	}
}

// T6 — `in` is read directly, NEVER derived from len(hash_ids)×64. This is the
// v5 (051926) encoding: in ≤ len(hash_ids)×64. Here in=100 but len(hash_ids)=1
// (which would be 64 if recomputed) → InputTokens must reflect 100 (BC-6).
func TestConvertWekaSession_ReadsInDirectlyNotHashIds(t *testing.T) {
	j := `{"id":"s","requests":[
	  {"type":"n","t":0.0,"in":100,"out":10,"hash_ids":[7],"api_time":1.0}
	]}`
	recs, err := ConvertWekaSession([]byte(j), WekaConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertWekaSession: %v", err)
	}
	if len(recs) != 1 {
		t.Fatalf("got %d records, want 1", len(recs))
	}
	if recs[0].InputTokens != 100 {
		t.Errorf("round0 input = %d, want 100 (read `in` directly; NOT len(hash_ids)×64 = 64)", recs[0].InputTokens)
	}
}

// T7 — a session with fewer than MinRounds usable main turns is skipped
// (nil,nil): both the all-subagent (0 usable main turns) case and a plain
// below-min-rounds case. No panic, no records (BC-8).
func TestConvertWekaSession_MinRoundsAndAllSubagentSkip(t *testing.T) {
	// All-subagent session: 0 usable main turns → skipped even at MinRounds:1.
	allSub := `{"id":"s","requests":[
	  {"type":"subagent","t":0.0,"agent_id":"a","requests":[{"type":"n","t":0.1,"in":10,"out":2,"api_time":0.1}]}
	]}`
	recs, err := ConvertWekaSession([]byte(allSub), WekaConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("all-subagent: %v", err)
	}
	if recs != nil {
		t.Fatalf("all-subagent session got %d records, want nil (0 usable main turns)", len(recs))
	}

	// Single main turn below MinRounds:2 → skipped.
	oneTurn := `{"id":"s","requests":[{"type":"n","t":0.0,"in":50,"out":5,"api_time":1.0}]}`
	recs, err = ConvertWekaSession([]byte(oneTurn), WekaConvertOptions{MinRounds: 2})
	if err != nil {
		t.Fatalf("below-min-rounds: %v", err)
	}
	if recs != nil {
		t.Fatalf("below-min-rounds got %d records, want nil", len(recs))
	}
}

// T7b — a main turn missing `in` (or `out`) is dropped (defense-in-depth; real
// data always carries them). Surviving main turns renumber contiguously and the
// delta spans the survivors — the dropped turn's wall-clock is absorbed into the
// following kept turn's think gap. Mirrors OTel's DropsSpanMissingTokenCount.
func TestConvertWekaSession_DropsMainTurnMissingTokens(t *testing.T) {
	// Middle main turn omits `in` → dropped; survivors are the 0s and 8s turns.
	j := `{"id":"s","requests":[
	  {"type":"n","t":0.0,"in":100,"out":10,"api_time":1.0},
	  {"type":"n","t":4.0,"out":9,"api_time":1.0},
	  {"type":"n","t":8.0,"in":150,"out":20,"api_time":1.0}
	]}`
	recs, err := ConvertWekaSession([]byte(j), WekaConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertWekaSession: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("got %d records, want 2 (main turn with missing `in` dropped)", len(recs))
	}
	// RoundIndex renumbers 0,1 with no gap; delta spans survivors: 150-100-10 = 40.
	if recs[1].RoundIndex != 1 || recs[1].InputTokens != 40 {
		t.Errorf("survivor round1 = ri %d in %d, want 1/40", recs[1].RoundIndex, recs[1].InputTokens)
	}
	// Think for the kept round spans the dropped turn: t2-t0-api0 = 8-0-1 = 7s.
	if recs[1].ThinkTimeUs != 7_000_000 {
		t.Errorf("round1 think = %d, want 7e6 (dropped turn's wall-clock absorbed into the gap)", recs[1].ThinkTimeUs)
	}
}

// T8 — a session with no `id` cannot be identified: ConvertWekaSession errors
// rather than emitting records under an empty session id. The corpus-level
// warn+skip of this error is covered in cmd/convert_weka_test.go.
func TestConvertWekaSession_NoSessionIDErrors(t *testing.T) {
	j := `{"requests":[{"type":"n","t":0.0,"in":50,"out":5,"api_time":1.0}]}`
	if _, err := ConvertWekaSession([]byte(j), WekaConvertOptions{MinRounds: 1}); err == nil {
		t.Fatal("expected error for a weka session with no id, got nil")
	}
}
