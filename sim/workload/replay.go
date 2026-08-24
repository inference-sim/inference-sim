package workload

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// effectiveInputTokenCount returns the token count to use for generating synthetic
// input token IDs. Priority rules:
//  1. serverInputTokens > 0 && prefixGroup == "": return serverInputTokens (server
//     is authoritative; covers chat-template overhead in blis-observe traces).
//  2. prefixGroup != "": return inputTokens regardless of serverInputTokens (server
//     count includes the prefix; using it as suffix would double-count the prefix
//     that replay.go prepends separately).
//  3. serverInputTokens == 0: return inputTokens (field absent in
//     generated/synthetic traces; not a real measurement).
func effectiveInputTokenCount(inputTokens, serverInputTokens int, prefixGroup string) int {
	if serverInputTokens > 0 && prefixGroup == "" {
		return serverInputTokens
	}
	return inputTokens
}

// injectionTime returns the raw DES injection basis for a trace record.
// For traces where SendTimeUs > 0, uses SendTimeUs as the injection basis.
// For blis observe traces with --concurrency, SendTimeUs is when the HTTP
// request was actually dispatched (after any concurrency-slot wait), which
// matches calibrate's TTFT baseline of first_chunk_time_us - send_time_us.
// For generated traces (blis run), SendTimeUs == ArrivalTimeUs, so both
// branches produce the same result.
// Falls back to ArrivalTimeUs whenever SendTimeUs <= 0:
//   - SendTimeUs == 0: legacy traces or generated traces (blis run) where no
//     real network send occurred.
//   - SendTimeUs < 0: defensive guard against corrupted trace timestamps;
//     a negative DES injection time would violate INV-3 (clock monotonicity).
//
// IMPORTANT (#1606): the value returned here is the RAW basis and may be on a
// different clock than the trace's arrival_time_us / deadline_us columns. In a
// real blis observe trace SendTimeUs is Unix-epoch µs (time.Now().UnixMicro())
// while arrival_time_us / deadline_us are run-relative (t≈0 at observe start).
// Callers building sim.Request.ArrivalTime MUST subtract injectionOriginShift so
// injection lands on the same (arrival) origin as the deadline; using this raw
// value as an absolute DES tick injects at epoch scale and instantly trips the
// past-due deadline guard (issue #1606).
//
// Note: in closed-loop session replay, think-time gaps between rounds are
// derived from ArrivalTimeUs deltas (not SendTimeUs) to preserve client-side
// pacing semantics; only the initial injection point uses the injection basis.
func injectionTime(rec TraceRecord) int64 {
	if rec.SendTimeUs > 0 {
		return rec.SendTimeUs
	}
	return rec.ArrivalTimeUs
}

// injectionOriginShift returns the constant to subtract from each record's
// injectionTime so injection lands on the SAME origin as arrival_time_us — the
// origin the deadline_us column is written on (#1606). It is
// min(injectionTime(rec)) - min(arrival_time_us(rec)) over all records:
//
//   - blis observe traces: send_time_us is Unix-epoch µs while arrival_time_us
//     is run-relative, so the shift ≈ the epoch offset and re-bases injection
//     onto the relative clock. Without it, requests inject at ~1.79e15 ticks and
//     the relative deadline (~3.4e8) is already past → instant timeout, 0
//     completions (issue #1606).
//   - blis run traces: send_time_us == arrival_time_us for every record
//     (tracev2.go), so min(injectionTime) == min(arrival) and the shift is
//     exactly 0 ⇒ injection == arrival, byte-identical to pre-#1606 replay
//     (INV-13 run/replay parity, INV-6 determinism).
//   - all-fallback traces (every send_time_us <= 0): injectionTime == arrival
//     for every record, so the shift is 0 (INV-3: no negative injection tick).
//
// Preserves #1304's send-based intra-trace spacing: the concurrency-slot wait is
// carried in send deltas, which are origin-invariant, so subtracting a single
// constant leaves the spacing intact. The result is >= 0 for every injected
// request whenever arrival_time_us >= 0 (validated by LoadTraceV2), because
// injectionTime(rec) >= min(injectionTime) and we add back min(arrival) >= 0.
// That same arrival_time_us >= 0 validation also bounds this subtraction: both
// mins lie in [0, max epoch µs ~1.79e15], far below int64 max, so minInjection -
// minArrival cannot overflow for any trace that came through LoadTraceV2.
// Returns 0 for an empty record set.
func injectionOriginShift(records []TraceRecord) int64 {
	if len(records) == 0 {
		return 0
	}
	minInjection := injectionTime(records[0])
	minArrival := records[0].ArrivalTimeUs
	for _, rec := range records[1:] {
		if inj := injectionTime(rec); inj < minInjection {
			minInjection = inj
		}
		if rec.ArrivalTimeUs < minArrival {
			minArrival = rec.ArrivalTimeUs
		}
	}
	return minInjection - minArrival
}

// MaxNormalizedInjectionTimeUs returns the largest normalized injection time
// among the records closed-loop replay injects initially — session round-0
// records and all non-session records. "Normalized" == injectionTime(rec) minus
// injectionOriginShift, so the value is on the same relative clock as the
// requests LoadTraceV2SessionBlueprints builds (and as computeReplayHorizon
// reads). cmd uses it to size the preliminary blueprint horizon in O(n) without
// building requests first; deriving it from the normalized injection (not the
// raw arrival_time_us column) keeps the horizon and the injection on one clock
// (#1606). Returns 0 for a nil/empty trace.
func MaxNormalizedInjectionTimeUs(trace *TraceV2) int64 {
	if trace == nil || len(trace.Records) == 0 {
		return 0
	}
	shift := injectionOriginShift(trace.Records)
	var max int64
	for _, rec := range trace.Records {
		if rec.SessionID != "" && rec.RoundIndex != 0 {
			continue // skip follow-up session rounds (not injected initially)
		}
		if inj := injectionTime(rec) - shift; inj > max {
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

	rng := rand.New(rand.NewSource(seed))

	// Injection-origin shift (#1606): re-base injection onto the arrival/deadline
	// origin. 0 for generated traces (send == arrival) ⇒ byte-identical.
	originShift := injectionOriginShift(trace.Records)

	// Generate shared prefix tokens per prefix group using trace-specified length
	prefixTokens := make(map[string][]sim.TokenID)
	for _, rec := range trace.Records {
		if rec.PrefixGroup != "" && rec.PrefixLength > 0 {
			if _, exists := prefixTokens[rec.PrefixGroup]; !exists {
				prefixTokens[rec.PrefixGroup] = sim.GenerateRandomTokenIDs(rng, rec.PrefixLength)
			}
		}
	}

	requests := make([]*sim.Request, 0, len(trace.Records))
	for _, rec := range trace.Records {
		// Generate synthetic token IDs, preferring server-reported count when available.
		inputTokens := sim.GenerateRandomTokenIDs(rng, effectiveInputTokenCount(rec.InputTokens, rec.ServerInputTokens, rec.PrefixGroup))

		// Prepend prefix if in a group
		if rec.PrefixGroup != "" {
			if prefix, ok := prefixTokens[rec.PrefixGroup]; ok {
				inputTokens = append(append([]sim.TokenID{}, prefix...), inputTokens...)
			}
		}

		outputTokens := sim.GenerateRandomTokenIDs(rng, rec.OutputTokens)

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
//	equals (service_time[i-1] + client_think_time) when the trace was produced by
//	blis observe. It is NOT pure client think time. Pass a sampler built via
//	ParseThinkTimeDist to supply the actual client-side think time when replaying
//	an observe-generated trace with accurate inter-round spacing.
//
// horizon <= 0: defaults to math.MaxInt64.
func LoadTraceV2SessionBlueprints(trace *TraceV2, seed int64, thinkTimeSampler LengthSampler, horizon int64) ([]*sim.Request, []SessionBlueprint, error) {
	if trace == nil || len(trace.Records) == 0 {
		return nil, nil, fmt.Errorf("empty trace")
	}
	if horizon <= 0 {
		horizon = math.MaxInt64
	}

	rng := rand.New(rand.NewSource(seed))

	// Injection-origin shift (#1606): a single constant over ALL records so the
	// round-0 and non-session injection sites below share one origin. 0 for
	// generated traces (send == arrival) ⇒ byte-identical to pre-#1606 replay.
	originShift := injectionOriginShift(trace.Records)

	// Generate shared prefix tokens per prefix group (same as LoadTraceV2Requests)
	prefixTokens := make(map[string][]sim.TokenID)
	for _, rec := range trace.Records {
		if rec.PrefixGroup != "" && rec.PrefixLength > 0 {
			if _, exists := prefixTokens[rec.PrefixGroup]; !exists {
				prefixTokens[rec.PrefixGroup] = sim.GenerateRandomTokenIDs(rng, rec.PrefixLength)
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

	// Growth mode from header (design §5): "accumulate" → strict growing prefix.
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

		// Build per-round token sequences, preferring server-reported count when available.
		inputSeq := make([]int, len(rounds))
		outputSeq := make([]int, len(rounds))
		for i, rec := range rounds {
			inputSeq[i] = effectiveInputTokenCount(rec.InputTokens, rec.ServerInputTokens, rec.PrefixGroup)
			outputSeq[i] = rec.OutputTokens
		}

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

		// Per-session RNG for deterministic token ID generation (INV-6)
		sessionRNG := rand.New(rand.NewSource(rng.Int63()))

		// Build round-0 request, preferring server-reported count when available.
		r0 := rounds[0]
		inputTokens := sim.GenerateRandomTokenIDs(sessionRNG, effectiveInputTokenCount(r0.InputTokens, r0.ServerInputTokens, r0.PrefixGroup))
		if r0.PrefixGroup != "" {
			if prefix, ok := prefixTokens[r0.PrefixGroup]; ok {
				inputTokens = append(append([]sim.TokenID{}, prefix...), inputTokens...)
			}
		}
		outputTokens := sim.GenerateRandomTokenIDs(sessionRNG, r0.OutputTokens)

		var prefix []sim.TokenID
		if r0.PrefixGroup != "" {
			prefix = prefixTokens[r0.PrefixGroup]
		}

		req := &sim.Request{
			ID:           fmt.Sprintf("request_%d", r0.RequestID),
			ArrivalTime:  injectionTime(r0) - originShift, // #1606: on the arrival/deadline origin
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
			SessionID:         sessionID,
			ClientID:          r0.ClientID,
			MaxRounds:         len(rounds),
			ContextGrowth:     contextGrowth,
			ThinkTimeSampler:  sessionThinkTimeSampler,
			Horizon:           horizon,
			InputSampler:      &SequenceSampler{values: inputSeq[1:]},  // rounds 1..N
			OutputSampler:     &SequenceSampler{values: outputSeq[1:]}, // rounds 1..N
			InputResetSampler: inputResetSampler,                       // #1609; nil unless a round compacts
			RNG:               sessionRNG,
			Prefix:            prefix,
			TenantID:          r0.TenantID,
			SLOClass:          r0.SLOClass,
			Model:             r0.Model,
			SLOTargetUs:       r0.SLOTargetUs,
			Adapter:           r0.Adapter, // #1464: adapter threads through session follow-up rounds
		}
		blueprints = append(blueprints, bp)
	}

	// Append non-session requests (same construction as LoadTraceV2Requests).
	for _, rec := range nonSessionRecords {
		inputTokens := sim.GenerateRandomTokenIDs(rng, effectiveInputTokenCount(rec.InputTokens, rec.ServerInputTokens, rec.PrefixGroup))
		if rec.PrefixGroup != "" {
			if prefix, ok := prefixTokens[rec.PrefixGroup]; ok {
				inputTokens = append(append([]sim.TokenID{}, prefix...), inputTokens...)
			}
		}
		outputTokens := sim.GenerateRandomTokenIDs(rng, rec.OutputTokens)
		req := &sim.Request{
			ID:              fmt.Sprintf("request_%d", rec.RequestID),
			ArrivalTime:     injectionTime(rec) - originShift, // #1606: on the arrival/deadline origin
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
