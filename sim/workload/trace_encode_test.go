package workload

import "testing"

// TestEncodeSessionToTraceRecords_DeltaLaw verifies the shared encoder's core
// contract (#1479): absolute per-round inputs → per-round deltas, round-0 full,
// non-monotone clamp to 0, Model left empty (routing safety), RequestID left 0,
// and ThinkUs surfaced onto the think_time_us column (#1478).
func TestEncodeSessionToTraceRecords_DeltaLaw(t *testing.T) {
	rounds := []NormalizedRound{
		{InputTokensAbs: 100, OutputTokens: 50, ArrivalUs: 0, Status: "ok"},
		{InputTokensAbs: 200, OutputTokens: 80, ArrivalUs: 5000, ThinkUs: 300, Status: "ok"},
		{InputTokensAbs: 120, OutputTokens: 10, ArrivalUs: 9000, Status: "error"}, // non-monotone → clamp
	}
	recs := EncodeSessionToTraceRecords("sess", rounds)
	if len(recs) != 3 {
		t.Fatalf("got %d records, want 3", len(recs))
	}

	// Delta law: round 0 = full first prompt; round 1 = 200-100-50 = 50;
	// round 2 = 120-200-80 = -160 → clamped to 0.
	if recs[0].InputTokens != 100 {
		t.Errorf("round0 delta = %d, want 100", recs[0].InputTokens)
	}
	if recs[1].InputTokens != 50 {
		t.Errorf("round1 delta = %d, want 50", recs[1].InputTokens)
	}
	if recs[2].InputTokens != 0 {
		t.Errorf("round2 delta = %d, want 0 (non-monotone clamp)", recs[2].InputTokens)
	}

	for i, r := range recs {
		if r.RoundIndex != i {
			t.Errorf("record %d RoundIndex = %d, want %d", i, r.RoundIndex, i)
		}
		if r.RequestID != 0 {
			t.Errorf("record %d RequestID = %d, want 0 (caller assigns global ids)", i, r.RequestID)
		}
		if r.Model != "" {
			t.Errorf("record %d Model = %q, want empty (routing safety)", i, r.Model)
		}
		if r.SessionID != "sess" {
			t.Errorf("record %d SessionID = %q, want sess", i, r.SessionID)
		}
	}

	// ThinkUs → think_time_us column; arrival, output, status propagate.
	if recs[0].ThinkTimeUs != 0 || recs[1].ThinkTimeUs != 300 {
		t.Errorf("ThinkTimeUs = [%d, %d], want [0, 300]", recs[0].ThinkTimeUs, recs[1].ThinkTimeUs)
	}
	if recs[1].ArrivalTimeUs != 5000 {
		t.Errorf("round1 ArrivalTimeUs = %d, want 5000", recs[1].ArrivalTimeUs)
	}
	if recs[2].OutputTokens != 10 || recs[2].Status != "error" {
		t.Errorf("round2 out/status = %d/%q, want 10/error", recs[2].OutputTokens, recs[2].Status)
	}
}
