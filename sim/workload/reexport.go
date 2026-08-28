package workload

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// ReExportClosedLoopRecords reconstructs a faithful TraceV2 record set from the
// requests a closed-loop (or pool) replay actually ran — the round-0 requests
// PLUS every SessionManager / pool-driver-generated follow-up — so a
// `blis replay --trace-output` re-export captures ALL rounds (issue #1630),
// not just the initial round-0 wave the previous export sourced.
//
// contextGrowth is the source trace header's session_context_growth:
//   - "accumulate": per-round input is a GROWING buffer, so records carry per-round
//     input-token DELTAS + input_tokens_reset compaction markers, re-derived from the
//     replayed absolute inputs via the shared EncodeSessionToTraceRecords law (#1613).
//     The re-export header MUST also set session_context_growth: accumulate (caller's
//     responsibility) so the corpus re-replays correctly. Model is left empty (routing
//     safety) — every reachable accumulate corpus is converter-produced (empty
//     Model/SLO/deadline/adapter), so this reproduces the converter shape exactly.
//   - "" (non-accumulate): each round's input is independent, so records carry the
//     ABSOLUTE per-round suffix via RequestsToTraceRecords, preserving SLO / model /
//     deadline / adapter / prefix. Follow-up records additionally inherit their
//     session's round-0 prefix_group / prefix_length so the prefix is not double-counted
//     on re-replay (a non-accumulate follow-up bakes the prefix into InputTokens but
//     carries PrefixLength==0; see SessionManager.OnComplete).
//
// thinkUsByReqID maps a follow-up request ID to the think time it used at generation
// (captured by the caller as followUp.ArrivalTime - completionClock). It is consulted
// ONLY for round>0 records, so a pool admission (a round-0 request returned as a
// "follow-up" with think 0) never gets a spurious think value. Writing the captured
// think to the think_time_us column lets closed-loop re-replay reproduce the exact
// per-round arrivals via the recorded-think path (LoadTraceV2SessionBlueprints), rather
// than the arrival-gap derivation which bundles service time into the gap.
//
// RequestIDs are assigned sequentially over a deterministic ordering
// (session-insertion order, rounds sorted ascending; non-session requests last), so
// repeated calls with the same input produce byte-identical records (INV-6). Returns an
// error if any session's rounds are non-consecutive (a "should never happen" guard on
// the captured run; R1 — surface rather than silently emit a corrupt corpus).
func ReExportClosedLoopRecords(reqs []*sim.Request, thinkUsByReqID map[string]int64, contextGrowth string) ([]TraceRecord, error) {
	// Partition into session groups (insertion order preserved for determinism) and
	// non-session single-shot requests (order preserved).
	type sessionGroup struct {
		id   string
		reqs []*sim.Request
	}
	var groups []*sessionGroup
	byID := make(map[string]*sessionGroup)
	var nonSession []*sim.Request
	for _, r := range reqs {
		if r.SessionID == "" {
			nonSession = append(nonSession, r)
			continue
		}
		g, ok := byID[r.SessionID]
		if !ok {
			g = &sessionGroup{id: r.SessionID}
			byID[r.SessionID] = g
			groups = append(groups, g)
		}
		g.reqs = append(g.reqs, r)
	}

	// Sort each session's rounds ascending and validate consecutive 0..N. A gap means
	// a round was lost between capture and here — refuse to emit a corpus whose loader
	// would reject or misinterpret it (R1).
	for _, g := range groups {
		sort.SliceStable(g.reqs, func(i, j int) bool {
			return g.reqs[i].RoundIndex < g.reqs[j].RoundIndex
		})
		for i, r := range g.reqs {
			if r.RoundIndex != i {
				return nil, fmt.Errorf("ReExportClosedLoopRecords: session %q has non-consecutive round indices (expected round %d, got %d) — captured rounds are incomplete", g.id, i, r.RoundIndex)
			}
		}
	}

	var records []TraceRecord

	if contextGrowth == "accumulate" {
		for _, g := range groups {
			rounds := make([]NormalizedRound, len(g.reqs))
			for i, r := range g.reqs {
				var think *int64
				if i > 0 {
					if t, ok := thinkUsByReqID[r.ID]; ok {
						tv := t
						think = &tv
					}
				}
				// The encoder's delta law is delta_k = abs_k − abs_{k-1} − prevOut, and it
				// must be the EXACT inverse of the round's accumulate growth
				// (abs_k = abs_{k-1} + actualOutput_{k-1} + delta_k, session.go). The buffer
				// grows by actualOutputLen = ProgressIndex − InputLen (which can be < the oracle
				// MaxOutputLen — e.g. the final decode token is not always counted, or a
				// length-capped round), NOT len(OutputTokens). Feeding the oracle here would
				// mis-derive every delta by (MaxOutputLen − actualOutput) and desync the
				// reconstruction (issue #1630 round-trip). So feed actualOutputLen for the delta;
				// the emitted OutputTokens column is overwritten with the oracle MaxOutputLen
				// below (re-replay's OutputSampler must drive the same completion → same
				// actualOutput → exact abs).
				rounds[i] = NormalizedRound{
					InputTokensAbs: int(r.InputLen()),
					OutputTokens:   accumulatedOutputLen(r),
					ArrivalUs:      r.ArrivalTime,
					ThinkUs:        think,
					Status:         reexportStatus(r.State),
				}
			}
			// EncodeSessionToTraceRecords re-derives per-round deltas + reset markers (the
			// #1613 law) and sets SessionID/RoundIndex/InputTokens/InputTokensReset/
			// OutputTokens/ArrivalTimeUs/ThinkTimeUs/Status; Model stays empty.
			recs := EncodeSessionToTraceRecords(g.id, rounds)
			for i := range recs {
				// Emit the ORACLE MaxOutputLen in the OutputTokens column (the delta was
				// computed from actualOutputLen above): re-replay's OutputSampler yields this
				// as MaxOutputLen, reproducing the same completion → same actualOutput → the
				// abs sequence reconstructs exactly. Then decorate with SendTimeUs +
				// sim-computed chunk timing (RequestsToTraceRecords parity, for calibrate).
				recs[i].OutputTokens = len(g.reqs[i].OutputTokens)
				decorateReExportTiming(&recs[i], g.reqs[i])
			}
			records = append(records, recs...)
		}
		if len(nonSession) > 0 {
			records = append(records, RequestsToTraceRecords(nonSession)...)
		}
	} else {
		// Non-accumulate: absolute per-round suffix via RequestsToTraceRecords, over a
		// deterministic grouped ordering (sessions in insertion order, rounds ascending;
		// non-session last). records[i] corresponds to ordered[i].
		ordered := make([]*sim.Request, 0, len(reqs))
		for _, g := range groups {
			ordered = append(ordered, g.reqs...)
		}
		ordered = append(ordered, nonSession...)
		records = RequestsToTraceRecords(ordered)

		// Per-session round-0 prefix info (for follow-up prefix propagation, BC-3).
		type prefixInfo struct {
			group  string
			length int
		}
		r0Prefix := make(map[string]prefixInfo, len(groups))
		for _, g := range groups {
			r0 := g.reqs[0]
			r0Prefix[g.id] = prefixInfo{group: r0.PrefixGroup, length: r0.PrefixLength}
		}

		for i, r := range ordered {
			if r.SessionID == "" || r.RoundIndex == 0 {
				continue // round 0 has no think and correct prefix metadata already
			}
			// Recorded per-round think (round>0): reproduce arrivals via the recorded-think
			// path on re-replay.
			if t, ok := thinkUsByReqID[r.ID]; ok {
				tv := t
				records[i].ThinkTimeUs = &tv
			}
			// Prefix propagation: a non-accumulate follow-up carries the prefix INSIDE
			// InputTokens but with PrefixLength==0 (SessionManager.OnComplete prepends
			// bp.Prefix but does not set the request's PrefixGroup/PrefixLength). Left as-is,
			// RequestsToTraceRecords records the full (prefix-included) length with
			// prefix_length=0, and the loader would prepend the prefix a SECOND time on
			// re-replay. Inherit the session's round-0 prefix metadata and emit the
			// suffix-only count instead.
			if pi, ok := r0Prefix[r.SessionID]; ok && pi.group != "" && pi.length > 0 {
				if records[i].InputTokens >= pi.length {
					records[i].InputTokens -= pi.length
					records[i].PrefixGroup = pi.group
					records[i].PrefixLength = pi.length
				}
			}
		}
	}

	// Sequential RequestIDs over the final deterministic ordering (uniqueness + INV-6).
	for i := range records {
		records[i].RequestID = i
	}
	return records, nil
}

// accumulatedOutputLen returns the number of output tokens that actually grew the
// session's accumulate buffer for this round — ProgressIndex − InputLen, clamped at 0 —
// mirroring SessionManager.OnComplete's actualOutputLen. This (not len(OutputTokens),
// the oracle budget) is the correct prevOut for the delta/reset law, so the re-derived
// deltas are the exact inverse of the round's accumulate growth (#1630 round-trip).
func accumulatedOutputLen(req *sim.Request) int {
	ao := int(req.ProgressIndex) - int(req.InputLen())
	if ao < 0 {
		return 0
	}
	return ao
}

// reexportStatus maps a replayed request's terminal state to the trace Status string,
// mirroring RequestsToTraceRecords (the sibling re-export). Status is informational —
// the replay loader does not read it — but keeping it faithful aids downstream analysis.
func reexportStatus(state sim.RequestState) string {
	switch state {
	case sim.StateCompleted:
		return "ok"
	case sim.StateTimedOut:
		return "timeout"
	default:
		return "incomplete"
	}
}

// decorateReExportTiming sets the send + client-observable chunk timestamps on an
// accumulate-branch record from the replayed request, using the same formula as
// RequestsToTraceRecords (LastChunkTimeUs = ArrivalTime + FirstTokenTime + sum(ITL),
// guarded by TTFTSet). SendTimeUs == ArrivalTimeUs (no network send in simulation), so
// the re-export's injection origin shift is 0 on re-replay (#1606), byte-identical to a
// generated trace's injection semantics.
func decorateReExportTiming(rec *TraceRecord, req *sim.Request) {
	rec.SendTimeUs = req.ArrivalTime
	if req.TTFTSet {
		rec.FirstChunkTimeUs = req.ArrivalTime + req.FirstTokenTime
		e2e := req.FirstTokenTime
		for _, itl := range req.ITL {
			e2e += itl
		}
		rec.LastChunkTimeUs = req.ArrivalTime + e2e
	}
}
