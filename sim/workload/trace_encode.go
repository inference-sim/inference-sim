package workload

// NormalizedRound is one LLM call in a captured agentic session, in the
// source-agnostic form the shared TraceV2 encoder consumes. Per-format readers
// (OTel `convert otel` #1479; Weka `convert weka` #1604) parse their own format
// into an ordered slice of these; EncodeSessionToTraceRecords turns them into
// TraceRecords. Readers supply the ABSOLUTE per-round input count — the encoder
// computes the per-round delta — so no reader duplicates the delta law.
type NormalizedRound struct {
	InputTokensAbs int    // absolute recorded prompt tokens for this round
	OutputTokens   int    // recorded output tokens
	ArrivalUs      int64  // arrival time (µs), relative to session start
	ThinkUs        *int64 // recorded pure client think time (µs); nil = not recorded, &0 = recorded zero (#1608)
	Status         string // "ok" or "error"
}

// EncodeSessionToTraceRecords converts one session's ordered NormalizedRounds
// into TraceRecords with per-round input-token DELTAS, for closed-loop replay
// under `session_context_growth: accumulate`. RoundIndex is 0..N in slice order;
// RequestID is left 0 (the caller assigns global ids across sessions).
//
// The delta law: round 0 carries the full first prompt; round N+1 carries
// max(0, in_{N+1} − in_N − out_N). In accumulate replay the buffer appends
// out_N then this delta, reconstructing the recorded absolute input exactly for
// monotonically-growing, non-length-capped sessions.
//
// Context compaction (#1609): when in_{N+1} < in_N + out_N (the model summarized
// or trimmed context, so the delta would go negative) the delta still clamps to
// 0, but the record ALSO carries the recorded absolute input in InputTokensReset
// (a *int64 presence marker; nil otherwise). Accumulate replay re-seeds its
// growing buffer to that absolute at the boundary — restoring exact reconstruction
// on non-monotone rounds and intentionally breaking strict prefix identity across
// the compaction boundary (the summary is not a literal prefix of the pre-compaction
// buffer). The marker is absent for monotone sessions and for spec-built `blis run`
// blueprints, so a trace without it replays byte-identically to before (INV-6).
//
// TraceRecord.Model is left empty deliberately: it is routing-significant at
// replay (buildRouterState filters instances by it), so writing a recorded
// cross-model name would drop every request at routing. Empty makes requests
// inherit --model (#1477). ThinkUs, when a reader sets it (non-nil), is written to
// the think_time_us column (#1478) and preferred over arrival-gap think at replay;
// the OTel reader leaves it nil (no reliable response-complete time → arrival-gap
// fallback), Weka sets it — including a recorded &0 for an overlapping turn, which
// (#1608) is no longer conflated with "not recorded".
func EncodeSessionToTraceRecords(sessionID string, rounds []NormalizedRound) []TraceRecord {
	recs := make([]TraceRecord, 0, len(rounds))
	prevIn, prevOut := 0, 0
	for round, r := range rounds {
		inputDelta := r.InputTokensAbs
		var resetAbs *int64 // #1609: recorded absolute on a compaction round; nil otherwise
		if round > 0 {
			inputDelta = r.InputTokensAbs - prevIn - prevOut
			if inputDelta < 0 {
				// Non-monotone context (compaction/trimming): the delta would go
				// negative. Clamp it to 0 as before (so an old reader still replays,
				// over-counting) AND emit the recorded absolute so a compaction-aware
				// accumulate replay re-seeds the buffer to it (#1609).
				inputDelta = 0
				abs := int64(r.InputTokensAbs)
				resetAbs = &abs
			}
		}
		recs = append(recs, TraceRecord{
			SessionID:        sessionID,
			RoundIndex:       round,
			InputTokens:      inputDelta,
			InputTokensReset: resetAbs,
			OutputTokens:     r.OutputTokens,
			ArrivalTimeUs:    r.ArrivalUs,
			ThinkTimeUs:      r.ThinkUs,
			Status:           r.Status,
			// Model: intentionally empty — see doc comment.
		})
		prevIn, prevOut = r.InputTokensAbs, r.OutputTokens
	}
	return recs
}
