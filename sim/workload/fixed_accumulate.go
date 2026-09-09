package workload

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// LoadTraceV2FixedAccumulateRequests builds a flat slice of pre-baked requests from
// an accumulate corpus (header session_context_growth == "accumulate"), one request
// per session round, injected at its RECORDED arrival time (open-loop, like fixed
// mode) while reconstructing the GROWING accumulate-delta input for each round (like
// closed-loop). This is the loader behind `blis replay --session-mode fixed-accumulate`
// (#1692).
//
// # Why this decoupling is possible
//
// Closed-loop replay reconstructs the growing context inside SessionManager.OnComplete,
// which chains each round's arrival to the sim's own completion time — a self-throttling
// feedback loop that never lets the queue reach its real depth at high concurrency.
// Fixed mode replays the recorded arrivals (breaking the feedback loop) but reads
// input_tokens as ABSOLUTE, so it misreads the deltas of an accumulate corpus.
//
// The insight: the growing buffer needs no live sim output. The trace's delta chain
// (abs_N = abs_{N-1} + out_{N-1} + delta_N) plus input_tokens_reset compaction markers
// fully determines every round's absolute input from trace data alone. So we invert the
// ENCODER's delta law here (EncodeSessionToTraceRecords) — appending each round's
// generated output then its delta (or Reset-ing to the recorded absolute on a compaction
// round) into a session-scoped buffer — and pre-bake every round as a request at its
// recorded arrival. The result carries the real cross-session overlap, so N large
// prefills pile into the scheduler per the real clock and produce genuine queueing delay.
//
// This mirrors SessionManager.OnComplete's buffer mechanics but feeds it the RECORDED
// output (rec.OutputTokens), whereas OnComplete appends the ACTUAL sim output
// (ProgressIndex − InputLen, which a length cap can truncate). The two are the same
// absent capping; feeding the recorded output is the correct choice here because the
// encoder's law is written over the recorded output, so this is its exact inverse (no
// INV-13 claim spans the two modes, and token CONTENT also differs from closed-loop on
// compaction rounds due to RNG draw order — harmless).
//
// # Contract with the encoder (INV-13 with the accumulate closed-loop path)
//
// The reconstruction is the EXACT inverse of EncodeSessionToTraceRecords' delta law:
// round 0 carries the full first prompt; round N+1's recorded delta is
// max(0, in_{N+1} - in_N - out_N), so appending out_N then delta reproduces in_{N+1}.
// On a compaction round (in_{N+1} < in_N + out_N) the encoder clamps the delta to 0 and
// stamps InputTokensReset = in_{N+1}; here we Reset the buffer to a fresh segment of that
// absolute length, matching SessionManager.OnComplete (#1609).
//
// Determinism (INV-6): sessions are processed in insertion order, each with its own RNG
// seeded from a master RNG derived from seed — identical to LoadTraceV2SessionBlueprints —
// so token streams are reproducible.
//
// Non-session (single-shot) records pass through with recorded absolute input at their
// recorded arrival, identical to LoadTraceV2Requests.
//
// Errors (never Fatalf — this is library code, R6): an empty trace, an unknown
// session_context_growth value, or a session with non-consecutive round indices.
func LoadTraceV2FixedAccumulateRequests(trace *TraceV2, seed int64) ([]*sim.Request, error) {
	if trace == nil || len(trace.Records) == 0 {
		return nil, fmt.Errorf("empty trace")
	}
	// This loader is only meaningful for an accumulate corpus. An empty growth value
	// is tolerated (the deltas then equal absolutes for single-round sessions), but an
	// unknown value is a typo footgun — fail loudly, mirroring LoadTraceV2SessionBlueprints.
	contextGrowth := trace.Header.SessionContextGrowth
	if contextGrowth != "" && contextGrowth != "accumulate" {
		return nil, fmt.Errorf("session_context_growth: unknown value %q (valid: \"accumulate\" or empty)", contextGrowth)
	}

	rng := rand.New(rand.NewSource(seed))

	// Injection-origin shift (#1606): re-base injection onto the arrival/deadline origin.
	// 0 for generated traces (send == arrival) ⇒ arrivals equal the recorded ArrivalTimeUs.
	originShift := injectionOriginShift(trace.Records)

	// Shared prefix tokens per prefix group (same as LoadTraceV2Requests).
	prefixTokens := make(map[string][]sim.TokenID)
	for _, rec := range trace.Records {
		if rec.PrefixGroup != "" && rec.PrefixLength > 0 {
			if _, exists := prefixTokens[rec.PrefixGroup]; !exists {
				prefixTokens[rec.PrefixGroup] = sim.GenerateRandomTokenIDs(rng, rec.PrefixLength)
			}
		}
	}

	// Group records by session, preserving insertion order (INV-6).
	sessionMap := make(map[string][]TraceRecord)
	var nonSessionRecords []TraceRecord
	var sessionOrder []string
	for _, rec := range trace.Records {
		if rec.SessionID == "" {
			nonSessionRecords = append(nonSessionRecords, rec)
			continue
		}
		if _, exists := sessionMap[rec.SessionID]; !exists {
			sessionOrder = append(sessionOrder, rec.SessionID)
		}
		sessionMap[rec.SessionID] = append(sessionMap[rec.SessionID], rec)
	}

	var requests []*sim.Request

	for _, sessionID := range sessionOrder {
		rounds := sessionMap[sessionID]
		sort.SliceStable(rounds, func(i, j int) bool {
			return rounds[i].RoundIndex < rounds[j].RoundIndex
		})
		for i, rec := range rounds {
			if rec.RoundIndex != i {
				return nil, fmt.Errorf("session %q has non-consecutive round indices (expected %d, got %d)", sessionID, i, rec.RoundIndex)
			}
		}

		// Per-session RNG for deterministic token IDs (INV-6), seeded from the master
		// RNG exactly as LoadTraceV2SessionBlueprints does.
		sessionRNG := rand.New(rand.NewSource(rng.Int63()))

		r0 := rounds[0]
		var prefix []sim.TokenID
		if r0.PrefixGroup != "" {
			prefix = prefixTokens[r0.PrefixGroup]
		}

		// Seed the growing buffer with round 0's input: prefix (if any) + a generated
		// suffix of effectiveInputTokenCount tokens. Layout mirrors the closed-loop
		// buffer: [prefix | r0_conversation | r0_output | r1_delta | r1_output | ...].
		//
		// Pre-allocate to the session's EXACT peak buffer length (computed from the delta
		// chain) so the growth phase never reallocates — every round's Slice() view then
		// aliases ONE backing array, and the session retains O(peak) tokens, not the
		// O(Σ absolute inputs) an eager per-round copy would retain. On the motivating
		// Weka corpus (ISL p50 ≈ 110K over ~37K rounds) that is the difference between
		// ~0.4 GB and ~16–24 GB. Matches the reasoning.go accumulate pattern (#1445).
		buf := newSessionTokenBufferWithCapacity(accumulatePeakBufferLen(rounds, len(prefix)))
		if len(prefix) > 0 {
			buf.Append(prefix)
		}
		r0Suffix := sim.GenerateRandomTokenIDs(sessionRNG, effectiveInputTokenCount(r0.InputTokens, r0.ServerInputTokens, r0.PrefixGroup))
		buf.Append(r0Suffix)

		// prevOutputTokens holds the previous round's generated output, appended to the
		// buffer BEFORE the current round's delta (the accumulate growth step).
		var prevOutputTokens []sim.TokenID

		for i, rec := range rounds {
			// Reset target for compaction rounds (#1609): a round>0 with an
			// input_tokens_reset marker re-seeds the buffer to that absolute length.
			resetTarget := -1
			if i > 0 && rec.InputTokensReset != nil {
				resetTarget = int(*rec.InputTokensReset)
			}

			var inputTokens []sim.TokenID
			if i == 0 {
				// Round 0 already seeded above.
				inputTokens = buf.Slice(0, buf.Len())
			} else if resetTarget >= 0 {
				// Context compaction: replace the whole buffer with a fresh segment of
				// exactly resetTarget tokens (mirrors SessionManager.OnComplete). The
				// just-completed round's output is intentionally NOT carried forward — it
				// was folded into the compaction the trace recorded as this round's abs.
				resetToks := sim.GenerateRandomTokenIDs(sessionRNG, resetTarget)
				_, inputEnd := buf.Reset(resetToks)
				inputTokens = buf.Slice(0, inputEnd)
			} else {
				// Normal growth: append prev round's output, then this round's delta.
				// A round>0 in an accumulate corpus stores its per-round DELTA directly in
				// rec.InputTokens — read it as-is. (Do NOT route through
				// effectiveInputTokenCount here: that helper returns the ABSOLUTE
				// ServerInputTokens when set, which would be a large silent over-count when
				// consumed as a delta. ServerInputTokens/PrefixGroup are round-0 concerns;
				// the prefix is already folded into the buffer.)
				if len(prevOutputTokens) > 0 {
					buf.Append(prevOutputTokens)
				}
				deltaToks := sim.GenerateRandomTokenIDs(sessionRNG, rec.InputTokens)
				_, inputEnd := buf.Append(deltaToks)
				inputTokens = buf.Slice(0, inputEnd)
			}

			outputTokens := sim.GenerateRandomTokenIDs(sessionRNG, rec.OutputTokens)
			prevOutputTokens = outputTokens

			// inputTokens is a VIEW into the shared session buffer (no copy). This is the
			// #1445 shared-buffer pattern (see reasoning.go): the simulator treats
			// Request.InputTokens as read-only (Request.FullInputTokens/InputTokenSlice
			// hand out capped three-index slices, so a stray append can't overwrite the
			// buffer), and the buffer's reallocation hazard is documented-safe — Append only
			// grows (indices [0,end_i) of an earlier view never move) and Reset allocates a
			// fresh array (earlier views keep pointing at the old, content-correct array).
			// Because the buffer was pre-sized to the peak above, the no-compaction path
			// never reallocates, so all views alias one array; a compaction Reset shrinks it
			// and the pre-reset views remain valid on the old array.
			requests = append(requests, buildFixedAccumulateRequest(rec, sessionID, inputTokens, outputTokens, originShift))
		}
	}

	// Non-session records: identical construction to LoadTraceV2Requests.
	for _, rec := range nonSessionRecords {
		inputTokens := sim.GenerateRandomTokenIDs(rng, effectiveInputTokenCount(rec.InputTokens, rec.ServerInputTokens, rec.PrefixGroup))
		if rec.PrefixGroup != "" {
			if prefix, ok := prefixTokens[rec.PrefixGroup]; ok {
				inputTokens = append(append([]sim.TokenID{}, prefix...), inputTokens...)
			}
		}
		outputTokens := sim.GenerateRandomTokenIDs(rng, rec.OutputTokens)
		requests = append(requests, buildFixedAccumulateRequest(rec, rec.SessionID, inputTokens, outputTokens, originShift))
	}

	// Sort by arrival time to satisfy the RequestSource contract (cluster.Run
	// requires non-decreasing ArrivalTime; SliceRequestSource does not re-sort).
	// This loader emits requests in SESSION-major order (all of session A's rounds,
	// then all of session B's), but a high-concurrency corpus interleaves sessions
	// in time — session B's round 0 may arrive between session A's rounds 0 and 1 —
	// so session order is not arrival order. A stable sort here (matching generator.go)
	// preserves the per-session round order for equal-arrival ties, keeping INV-6
	// determinism. Safe because every request is already fully materialized with a
	// frozen InputTokens slice, so reordering cannot disturb the accumulate
	// reconstruction (which happened above, per session).
	sort.SliceStable(requests, func(i, j int) bool {
		return requests[i].ArrivalTime < requests[j].ArrivalTime
	})

	return requests, nil
}

// accumulatePeakBufferLen returns the maximum shared-buffer length the growth loop in
// LoadTraceV2FixedAccumulateRequests will reach for a session, so the buffer can be
// pre-allocated once and never reallocate (keeping every round's Slice() view aliased to
// one backing array).
//
// The buffer length after round i's input is that round's ABSOLUTE input in_i, and it
// never exceeds in_i mid-round (the transient append of out_{i-1} brings it to
// in_{i-1}+out_{i-1} = in_i − delta_i ≤ in_i). So the peak is max_i(in_i), reconstructed
// from the same delta law the loop applies:
//   - round 0: in_0 = prefixLen + delta_0 (the seeded prefix + round-0 suffix)
//   - reset round: in_i = *InputTokensReset (the recorded absolute)
//   - normal round i>0: in_i = in_{i-1} + out_{i-1} + delta_i
// rounds is assumed sorted ascending and consecutive (validated by the caller).
func accumulatePeakBufferLen(rounds []TraceRecord, prefixLen int) int64 {
	var peak, prevAbs, prevOut int64
	for i, rec := range rounds {
		var abs int64
		switch {
		case i == 0:
			abs = int64(prefixLen) + int64(effectiveInputTokenCount(rec.InputTokens, rec.ServerInputTokens, rec.PrefixGroup))
		case rec.InputTokensReset != nil:
			abs = *rec.InputTokensReset
		default:
			// round>0 delta is rec.InputTokens directly (see the growth branch in
			// LoadTraceV2FixedAccumulateRequests — not effectiveInputTokenCount).
			abs = prevAbs + prevOut + int64(rec.InputTokens)
		}
		if abs > peak {
			peak = abs
		}
		prevAbs, prevOut = abs, int64(rec.OutputTokens)
	}
	return peak
}

// buildFixedAccumulateRequest assembles a sim.Request from a trace record with the
// already-reconstructed input/output token slices. Field set matches
// LoadTraceV2Requests (R4). PrefixGroup/PrefixLength are deliberately NOT carried on
// session rounds: accumulate folds the prefix into round 0's reconstructed input, so
// re-prepending it would double-count. Non-session records keep them (their input is
// suffix-only, exactly as fixed mode).
func buildFixedAccumulateRequest(rec TraceRecord, sessionID string, inputTokens, outputTokens []sim.TokenID, originShift int64) *sim.Request {
	req := &sim.Request{
		ID:               fmt.Sprintf("request_%d", rec.RequestID),
		ArrivalTime:      injectionTime(rec) - originShift, // #1606: on the arrival/deadline origin
		InputTokens:      inputTokens,
		OutputTokens:     outputTokens,
		MaxOutputLen:     len(outputTokens),
		State:            sim.StateQueued,
		ScheduledStepIdx: 0,
		FinishedStepIdx:  0,
		TenantID:         rec.TenantID,
		SLOClass:         rec.SLOClass,
		SessionID:        sessionID,
		RoundIndex:       rec.RoundIndex,
		TextTokenCount:   rec.TextTokens,
		ImageTokenCount:  rec.ImageTokens,
		AudioTokenCount:  rec.AudioTokens,
		VideoTokenCount:  rec.VideoTokens,
		ReasonRatio:      rec.ReasonRatio,
		Model:            rec.Model,
		Deadline:         rec.DeadlineUs,
		SLOTargetUs:      rec.SLOTargetUs,
		ClientID:         rec.ClientID,
		Streaming:        rec.Streaming,
		Adapter:          rec.Adapter,
	}
	// Prefix metadata parity with the other loaders (router prefix-affinity scoring
	// reads PrefixGroup/PrefixLength; the tokens themselves are already baked into
	// InputTokens):
	//   - Non-session records: carry it (suffix-only input), like LoadTraceV2Requests.
	//   - Session ROUND 0: carry it, matching LoadTraceV2SessionBlueprints' round-0
	//     request — so two sessions sharing a system-prompt group still get cross-session
	//     prefix-cache affinity. No-op for converter corpora (PrefixGroup == "").
	//   - Session follow-up rounds (RoundIndex > 0): DO NOT carry it — accumulate folds
	//     the prefix into the growing buffer, so a PrefixLength here would double-count
	//     (mirrors SessionManager.OnComplete, which prepends the prefix but leaves the
	//     follow-up's PrefixGroup/PrefixLength unset).
	if sessionID == "" || rec.RoundIndex == 0 {
		req.PrefixGroup = rec.PrefixGroup
		req.PrefixLength = rec.PrefixLength
	}
	return req
}
