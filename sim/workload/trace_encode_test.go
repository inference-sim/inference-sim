package workload

import "testing"

// i64p returns a pointer to v, for constructing a RECORDED (non-nil) think time in
// tests (#1608). A recorded &0 (i64p(0)) is distinct from a nil (not-recorded) think:
// TraceRecord.ThinkTimeUs and NormalizedRound.ThinkUs are *int64 presence signals.
func i64p(v int64) *int64 { return &v }

// TestEncodeSessionToTraceRecords_DeltaLaw verifies the shared encoder's core
// contract (#1479): absolute per-round inputs → per-round deltas, round-0 full,
// non-monotone clamp to 0, Model left empty (routing safety), RequestID left 0,
// and ThinkUs surfaced onto the think_time_us column (#1478).
func TestEncodeSessionToTraceRecords_DeltaLaw(t *testing.T) {
	rounds := []NormalizedRound{
		{InputTokensAbs: 100, OutputTokens: 50, ArrivalUs: 0, Status: "ok"}, // round 0: ThinkUs nil (not recorded)
		{InputTokensAbs: 200, OutputTokens: 80, ArrivalUs: 5000, ThinkUs: i64p(300), Status: "ok"},
		{InputTokensAbs: 120, OutputTokens: 10, ArrivalUs: 9000, Status: "error"}, // non-monotone → clamp; ThinkUs nil
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

	// Compaction marker (#1609): the non-monotone round (delta would clamp to a
	// negative) carries the recorded absolute input as a reset target; every
	// other round leaves it nil. Round 0 never carries a reset.
	if recs[0].InputTokensReset != nil {
		t.Errorf("round0 InputTokensReset = %v, want nil", *recs[0].InputTokensReset)
	}
	if recs[1].InputTokensReset != nil {
		t.Errorf("round1 (monotone) InputTokensReset = %v, want nil", *recs[1].InputTokensReset)
	}
	if recs[2].InputTokensReset == nil {
		t.Fatalf("round2 (compaction) InputTokensReset = nil, want &120")
	}
	if *recs[2].InputTokensReset != 120 {
		t.Errorf("round2 InputTokensReset = %d, want 120 (recorded absolute)", *recs[2].InputTokensReset)
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

	// ThinkUs → think_time_us column: nil propagates as nil (not recorded), a set
	// value propagates through (#1608). Round 0 nil; round 1 recorded &300.
	if recs[0].ThinkTimeUs != nil {
		t.Errorf("round0 ThinkTimeUs = %v, want nil (not recorded)", recs[0].ThinkTimeUs)
	}
	if recs[1].ThinkTimeUs == nil || *recs[1].ThinkTimeUs != 300 {
		t.Errorf("round1 ThinkTimeUs = %v, want &300", recs[1].ThinkTimeUs)
	}
	if recs[1].ArrivalTimeUs != 5000 {
		t.Errorf("round1 ArrivalTimeUs = %d, want 5000", recs[1].ArrivalTimeUs)
	}
	if recs[2].OutputTokens != 10 || recs[2].Status != "error" {
		t.Errorf("round2 out/status = %d/%q, want 10/error", recs[2].OutputTokens, recs[2].Status)
	}
}

// TestEncodeSessionToTraceRecords_MonotoneNoReset verifies a strictly-growing
// session emits NO reset markers (#1609): every reconstruction is exact via the
// delta law alone, so the compaction column stays absent (INV-6 byte-identity).
func TestEncodeSessionToTraceRecords_MonotoneNoReset(t *testing.T) {
	rounds := []NormalizedRound{
		{InputTokensAbs: 100, OutputTokens: 50, Status: "ok"},
		{InputTokensAbs: 200, OutputTokens: 80, Status: "ok"}, // 200 >= 100+50
		{InputTokensAbs: 300, OutputTokens: 10, Status: "ok"}, // 300 >= 200+80
	}
	recs := EncodeSessionToTraceRecords("sess", rounds)
	for i, r := range recs {
		if r.InputTokensReset != nil {
			t.Errorf("round%d InputTokensReset = %d, want nil (monotone)", i, *r.InputTokensReset)
		}
	}
	// Deltas: 100, 50 (200-100-50), 20 (300-200-80).
	if recs[1].InputTokens != 50 || recs[2].InputTokens != 20 {
		t.Errorf("deltas = [%d, %d], want [50, 20]", recs[1].InputTokens, recs[2].InputTokens)
	}
}
