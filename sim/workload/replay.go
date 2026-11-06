package workload

import (
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// MaxNormalizedInjectionTimeUs returns the largest arrival time among the records
// closed-loop replay injects initially: session round-0 records and all non-session records.
// A request is injected at its arrival_time_us, on the origin its deadline_us is written on,
// so the value is on the clock of the requests LoadTraceV2SessionBlueprints builds (and of
// computeReplayHorizon). cmd uses it to size the preliminary blueprint horizon in O(n)
// without building requests first. Returns 0 for a nil/empty trace.
func MaxNormalizedInjectionTimeUs(trace *TraceV2) int64 {
	if trace == nil || len(trace.Records) == 0 {
		return 0
	}
	var max int64
	for _, rec := range trace.Records {
		if rec.SessionID != "" && rec.RoundIndex != 0 {
			continue // skip follow-up session rounds (not injected initially)
		}
		if inj := rec.ArrivalTimeUs; inj > max {
			max = inj
		}
	}
	return max
}

// LoadTraceV2Requests converts trace v2 records into sim.Request objects
// with synthetic token IDs for simulation replay. Requests in the same
// prefix_group share identical prefix token sequences.
func LoadTraceV2Requests(trace *TraceV2, seed int64) ([]*sim.Request, error) {
	if trace == nil || len(trace.Records) == 0 {
		return nil, fmt.Errorf("empty trace")
	}

	// Generate shared prefix tokens per prefix group using trace-specified length
	prefixTokens := make(map[string][]sim.TokenID)
	for _, rec := range trace.Records {
		if rec.PrefixGroup != "" && rec.PrefixLength > 0 {
			if _, exists := prefixTokens[rec.PrefixGroup]; !exists {
				prefixTokens[rec.PrefixGroup] = sim.GenerateRandomTokenIDs(prefixStream(seed, rec.PrefixGroup), rec.PrefixLength)
			}
		}
	}

	requests := make([]*sim.Request, 0, len(trace.Records))
	for _, rec := range trace.Records {
		// Synthetic token IDs from the record's own streams (replayRecordStreams).
		rs := replayRecordStreams(seed, rec)
		inputTokens := tokens(rs, purposeInputTokens, rec.InputTokens)

		// Prepend prefix if in a group
		if rec.PrefixGroup != "" {
			if prefix, ok := prefixTokens[rec.PrefixGroup]; ok {
				inputTokens = append(append([]sim.TokenID{}, prefix...), inputTokens...)
			}
		}

		outputTokens := tokens(rs, purposeOutputTokens, rec.OutputTokens)

		req := &sim.Request{
			ID:               fmt.Sprintf("request_%d", rec.RequestID),
			ArrivalTime:      rec.ArrivalTimeUs,
			InputTokens:      inputTokens,
			OutputTokens:     outputTokens,
			MaxOutputLen:     len(outputTokens),
			State:            sim.StateQueued,
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
			TenantID:         rec.TenantID,
			SLOClass:         rec.SLOClass,
			SessionID:        rec.SessionID,
			RoundIndex:       rec.RoundIndex,
			TextTokenCount:   rec.TextTokens,
			ImageTokenCount:  rec.ImageTokens,
			AudioTokenCount:  rec.AudioTokens,
			VideoTokenCount:  rec.VideoTokens,
			ReasonRatio:      rec.ReasonRatio,
			Model:            rec.Model,      // BC-3, BC-6: model identity from trace; empty = default model
			Deadline:         rec.DeadlineUs, // BC-4, BC-5: client timeout; 0 = no timeout
			SLOTargetUs:      rec.SLOTargetUs,
			ClientID:         rec.ClientID,
			PrefixGroup:      rec.PrefixGroup,
			PrefixLength:     rec.PrefixLength,
			Streaming:        rec.Streaming,
			Adapter:          rec.Adapter, // #1464: adapter identity from trace; "" = base-model-only
		}
		requests = append(requests, req)
	}
	return requests, nil
}

// sessionHasRecordedThinkTime reports whether any non-round-0 record in a session
// carries a recorded think_time_us (#1478). Round 0 has no predecessor, so its think
// time is meaningless and ignored. When true, closed-loop replay prefers the recorded
// per-round think time over arrival-gap derivation.
//
// PRESENCE, not non-zero (#1608): ThinkTimeUs is now a *int64, so a recorded &0 is
// distinguishable from an absent (nil) value. A session whose every round genuinely
// recorded think == 0 (real for Weka's overlap-clamped rounds) now reads as recorded
// here and uses those zeros (back-to-back rounds) — it no longer falls back to the
// arrival-gap path, which bundles service time into the gap (see the gap-derivation
// note below), the very effect think_time_us exists to avoid. Only a truly absent
// (nil) value falls back. Pinned by
// TestLoadTraceV2SessionBlueprints_AllRecordedZeroThink_UsesRecordedZeros.
func sessionHasRecordedThinkTime(rounds []TraceRecord) bool {
	for i := 1; i < len(rounds); i++ {
		if rounds[i].ThinkTimeUs != nil {
			return true
		}
	}
	return false
}

// LoadTraceV2SessionBlueprints groups trace records by session and builds
// SessionBlueprints with SequenceSamplers for deterministic token replay.
// Returns round-0 requests (plus all non-session requests) for initial injection,
// and blueprints for the SessionManager.
//
// thinkTimeSampler != nil: use this sampler for all sessions' think-time draws.
// thinkTimeSampler == nil: derive per-round think time from trace arrival gaps.
//
//	NOTE: gap-derived think time = ArrivalTimeUs[i] - ArrivalTimeUs[i-1], which
//	equals (service_time[i-1] + client_think_time) when the trace records a real
//	client's arrivals. It is NOT pure client think time. Pass a sampler built via
//	ParseThinkTimeDist to supply the actual client-side think time when replaying
//	such a trace with accurate inter-round spacing.
//
// horizon <= 0: defaults to math.MaxInt64.
func LoadTraceV2SessionBlueprints(trace *TraceV2, seed int64, thinkTimeSampler LengthSampler, horizon int64) ([]*sim.Request, []SessionBlueprint, error) {
	if trace == nil || len(trace.Records) == 0 {
		return nil, nil, fmt.Errorf("empty trace")
	}
	if horizon <= 0 {
		horizon = math.MaxInt64
	}

	// Generate shared prefix tokens per prefix group (same as LoadTraceV2Requests)
	prefixTokens := make(map[string][]sim.TokenID)
	for _, rec := range trace.Records {
		if rec.PrefixGroup != "" && rec.PrefixLength > 0 {
			if _, exists := prefixTokens[rec.PrefixGroup]; !exists {
				prefixTokens[rec.PrefixGroup] = sim.GenerateRandomTokenIDs(prefixStream(seed, rec.PrefixGroup), rec.PrefixLength)
			}
		}
	}

	// Group records by session, preserving insertion order for deterministic output (INV-6)
	type sessionRounds struct {
		records []TraceRecord
	}
	sessionMap := make(map[string]*sessionRounds)
	var nonSessionRecords []TraceRecord
	var sessionOrder []string

	for _, rec := range trace.Records {
		if rec.SessionID == "" {
			nonSessionRecords = append(nonSessionRecords, rec)
			continue
		}
		sr, exists := sessionMap[rec.SessionID]
		if !exists {
			sr = &sessionRounds{}
			sessionMap[rec.SessionID] = sr
			sessionOrder = append(sessionOrder, rec.SessionID)
		}
		sr.records = append(sr.records, rec)
	}

	// Sort each session's records by RoundIndex and validate consecutive indices
	for sid, sr := range sessionMap {
		sort.Slice(sr.records, func(i, j int) bool {
			return sr.records[i].RoundIndex < sr.records[j].RoundIndex
		})
		for i, rec := range sr.records {
			if rec.RoundIndex != i {
				return nil, nil, fmt.Errorf("session %q has non-consecutive round indices (expected %d, got %d)", sid, i, rec.RoundIndex)
			}
		}
	}

	var requests []*sim.Request
	var blueprints []SessionBlueprint

	// Growth mode from header (design §5): "accumulate" → growing shared prefix,
	// shrunk back to the recorded absolute at a context-compaction boundary when the
	// trace carries input_tokens_reset markers (#1609).
	// Validate up front: an unrecognized value (e.g. a typo like "Accumulate") would
	// otherwise fall through to the non-accumulate branch silently, disabling the
	// feature with no feedback — an operator footgun. Fail loudly instead (R1).
	contextGrowth := trace.Header.SessionContextGrowth
	if contextGrowth != "" && contextGrowth != "accumulate" {
		return nil, nil, fmt.Errorf("session_context_growth: unknown value %q (valid: \"accumulate\" or empty)", contextGrowth)
	}

	for _, sessionID := range sessionOrder {
		sr := sessionMap[sessionID]
		rounds := sr.records
		if len(rounds) == 0 {
			continue
		}

		// Build per-round token sequences.
		inputSeq := make([]int, len(rounds))
		outputSeq := make([]int, len(rounds))
		for i, rec := range rounds {
			inputSeq[i] = rec.InputTokens
			outputSeq[i] = rec.OutputTokens
		}

		// Per-round client-deadline offsets (#2064), aligned with inputSeq. The trace's
		// deadline_us is absolute on the same origin as arrival_time_us, so a round's
		// offset is deadline_us - arrival_time_us: a value re-anchored to whatever
		// simulated arrival the follow-up lands on (see session.go followUpDeadline).
		// A round with deadline_us == 0 (no deadline), or a defensively negative result,
		// gets the -1 "no deadline" sentinel. Unlike InputResetSampler this is ALWAYS set
		// on the replay path (even when every round is -1): leaving it nil would fall the
		// follow-up back to computeDeadline's 300 s session-client default — exactly the
		// #2064 bug. The run path never reaches this loader, so its blueprints keep a nil
		// sampler and the 300 s default (INV-6).
		deadlineOffsetSeq := make([]int, len(rounds))
		for i, rec := range rounds {
			if rec.DeadlineUs > 0 && rec.DeadlineUs >= rec.ArrivalTimeUs {
				deadlineOffsetSeq[i] = int(rec.DeadlineUs - rec.ArrivalTimeUs)
			} else {
				deadlineOffsetSeq[i] = -1 // no deadline this round
			}
		}
		deadlineOffsetSampler := &SequenceSampler{values: deadlineOffsetSeq[1:]} // rounds 1..N, lockstep with InputSampler

		// Context-compaction reset targets (#1609), aligned with inputSeq. A round's
		// input_tokens_reset marker (non-nil) becomes its absolute re-seed length; a
		// round without one gets the -1 "no reset" sentinel. Only meaningful in
		// accumulate mode, and only built when at least one round 1..N actually carries
		// a marker — so a trace with no compaction column (or a monotone session within
		// a compaction-bearing corpus) leaves InputResetSampler nil and replays
		// byte-identically to pre-#1609 (INV-6).
		var inputResetSampler LengthSampler
		if contextGrowth == "accumulate" {
			resetSeq := make([]int, len(rounds))
			anyReset := false
			for i, rec := range rounds {
				if i > 0 && rec.InputTokensReset != nil {
					resetSeq[i] = int(*rec.InputTokensReset)
					anyReset = true
				} else {
					resetSeq[i] = -1 // no reset this round
				}
			}
			if anyReset {
				inputResetSampler = &SequenceSampler{values: resetSeq[1:]} // rounds 1..N, lockstep with InputSampler
			}
		}

		// Build think time. Precedence (#1478):
		//   1. caller-provided sampler (CLI --think-time-dist / --think-time-ms)
		//   2. recorded per-round think_time_us column (set by agentic-trace converters):
		//      pure client think time, decoupled from inference — preferred over the
		//      arrival-gap derivation, which bundles service time into the gap
		//   3. inter-round arrival gaps (existing default; NOT pure client think time)
		var sessionThinkTimeSampler LengthSampler
		switch {
		case thinkTimeSampler != nil:
			sessionThinkTimeSampler = thinkTimeSampler // stateless: safe to share across sessions
		case sessionHasRecordedThinkTime(rounds):
			// think_time_us on round i is the recorded gap BEFORE round i (from round
			// i-1's end). Round 0 carries no think; rounds 1..N supply the sequence.
			thinkTimes := make([]int, len(rounds)-1)
			for i := 1; i < len(rounds); i++ {
				// derefInt64(nil) == 0: a recorded &0 and (defensively) a nil interior
				// round both yield 0 think. sessionHasRecordedThinkTime guarantees at
				// least one round here is non-nil. No shipped converter produces a mixed
				// interior (OTel leaves every round nil; Weka sets every round 1..N), so
				// the nil-interior branch is unreachable today — the deref pins a defined
				// behavior (0 think) if a future converter ever recorded think sparsely.
				t := derefInt64(rounds[i].ThinkTimeUs)
				if t < 0 {
					t = 0 // defensive; LoadTraceV2 already rejects negatives (INV-3)
				}
				thinkTimes[i-1] = int(t)
			}
			sessionThinkTimeSampler = &SequenceSampler{values: thinkTimes}
		case len(rounds) > 1:
			thinkTimes := make([]int, len(rounds)-1)
			for i := 1; i < len(rounds); i++ {
				gap := rounds[i].ArrivalTimeUs - rounds[i-1].ArrivalTimeUs
				if gap < 0 {
					// Non-monotone timestamps (e.g., clock skew in observed trace).
					// Clamp to 0 rather than propagating a negative think time,
					// which would violate INV-3 (clock monotonicity) in OnComplete.
					gap = 0
				}
				thinkTimes[i-1] = int(gap)
			}
			sessionThinkTimeSampler = &SequenceSampler{values: thinkTimes}
		}

		// The session's streams (DEV-6): round 0's tokens here, each follow-up's in the
		// session manager, every round from its own child (roundStreams).
		session := SessionStreams(seed, sessionID)
		rs0 := roundStreams(session, 0)

		// Build round-0 request.
		r0 := rounds[0]
		inputTokens := tokens(rs0, purposeInputTokens, r0.InputTokens)
		if r0.PrefixGroup != "" {
			if prefix, ok := prefixTokens[r0.PrefixGroup]; ok {
				inputTokens = append(append([]sim.TokenID{}, prefix...), inputTokens...)
			}
		}
		outputTokens := tokens(rs0, purposeOutputTokens, r0.OutputTokens)

		var prefix []sim.TokenID
		if r0.PrefixGroup != "" {
			prefix = prefixTokens[r0.PrefixGroup]
		}

		req := &sim.Request{
			ID:           fmt.Sprintf("request_%d", r0.RequestID),
			ArrivalTime:  r0.ArrivalTimeUs,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			MaxOutputLen: len(outputTokens),
			State:        sim.StateQueued,
			// ScheduledStepIdx, FinishedStepIdx default to 0 (R4: consistent with LoadTraceV2Requests)
			TenantID:        r0.TenantID,
			SLOClass:        r0.SLOClass,
			SessionID:       sessionID,
			RoundIndex:      0,
			TextTokenCount:  r0.TextTokens,
			ImageTokenCount: r0.ImageTokens,
			AudioTokenCount: r0.AudioTokens,
			VideoTokenCount: r0.VideoTokens,
			ReasonRatio:     r0.ReasonRatio,
			Model:           r0.Model,
			Deadline:        r0.DeadlineUs,
			SLOTargetUs:     r0.SLOTargetUs,
			ClientID:        r0.ClientID,
			PrefixGroup:     r0.PrefixGroup,
			PrefixLength:    r0.PrefixLength,
			Streaming:       r0.Streaming,
			Adapter:         r0.Adapter, // #1464: adapter identity from trace; "" = base-model-only
		}
		requests = append(requests, req)

		bp := SessionBlueprint{
			SessionID:             sessionID,
			ClientID:              r0.ClientID,
			MaxRounds:             len(rounds),
			ContextGrowth:         string(contextGrowth),
			ThinkTimeSampler:      sessionThinkTimeSampler,
			Horizon:               horizon,
			InputSampler:          &SequenceSampler{values: inputSeq[1:]},  // rounds 1..N
			OutputSampler:         &SequenceSampler{values: outputSeq[1:]}, // rounds 1..N
			InputResetSampler:     inputResetSampler,                       // #1609; nil unless a round compacts
			DeadlineOffsetSampler: deadlineOffsetSampler,                   // #2064; always set on replay so follow-ups honor the trace deadline
			Streams:               session,
			Prefix:                prefix,
			TenantID:              r0.TenantID,
			SLOClass:              r0.SLOClass,
			Model:                 r0.Model,
			SLOTargetUs:           r0.SLOTargetUs,
			Adapter:               r0.Adapter, // #1464: adapter threads through session follow-up rounds
		}
		blueprints = append(blueprints, bp)
	}

	// Append non-session requests (same construction as LoadTraceV2Requests).
	for _, rec := range nonSessionRecords {
		rs := replayRecordStreams(seed, rec)
		inputTokens := tokens(rs, purposeInputTokens, rec.InputTokens)
		if rec.PrefixGroup != "" {
			if prefix, ok := prefixTokens[rec.PrefixGroup]; ok {
				inputTokens = append(append([]sim.TokenID{}, prefix...), inputTokens...)
			}
		}
		outputTokens := tokens(rs, purposeOutputTokens, rec.OutputTokens)
		req := &sim.Request{
			ID:              fmt.Sprintf("request_%d", rec.RequestID),
			ArrivalTime:     rec.ArrivalTimeUs,
			InputTokens:     inputTokens,
			OutputTokens:    outputTokens,
			MaxOutputLen:    len(outputTokens),
			State:           sim.StateQueued,
			TenantID:        rec.TenantID,
			SLOClass:        rec.SLOClass,
			SessionID:       rec.SessionID,
			RoundIndex:      rec.RoundIndex,
			TextTokenCount:  rec.TextTokens,
			ImageTokenCount: rec.ImageTokens,
			AudioTokenCount: rec.AudioTokens,
			VideoTokenCount: rec.VideoTokens,
			ReasonRatio:     rec.ReasonRatio,
			Model:           rec.Model,
			Deadline:        rec.DeadlineUs,
			SLOTargetUs:     rec.SLOTargetUs,
			ClientID:        rec.ClientID,
			PrefixGroup:     rec.PrefixGroup,
			PrefixLength:    rec.PrefixLength,
			Streaming:       rec.Streaming,
			Adapter:         rec.Adapter, // #1464: adapter identity from trace; "" = base-model-only
		}
		requests = append(requests, req)
	}

	return requests, blueprints, nil
}
