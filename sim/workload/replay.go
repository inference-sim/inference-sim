package workload

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// LoadTraceV2Requests converts trace v2 records into sim.Request objects
// with synthetic token IDs for simulation replay. Requests in the same
// prefix_group share identical prefix token sequences.
func LoadTraceV2Requests(trace *TraceV2, seed int64) ([]*sim.Request, error) {
	if trace == nil || len(trace.Records) == 0 {
		return nil, fmt.Errorf("empty trace")
	}

	rng := rand.New(rand.NewSource(seed))

	// Generate shared prefix tokens per prefix group using trace-specified length
	prefixTokens := make(map[string][]int)
	for _, rec := range trace.Records {
		if rec.PrefixGroup != "" && rec.PrefixLength > 0 {
			if _, exists := prefixTokens[rec.PrefixGroup]; !exists {
				prefixTokens[rec.PrefixGroup] = sim.GenerateRandomTokenIDs(rng, rec.PrefixLength)
			}
		}
	}

	requests := make([]*sim.Request, 0, len(trace.Records))
	for _, rec := range trace.Records {
		// Generate synthetic token IDs
		inputTokens := sim.GenerateRandomTokenIDs(rng, rec.InputTokens)

		// Prepend prefix if in a group
		if rec.PrefixGroup != "" {
			if prefix, ok := prefixTokens[rec.PrefixGroup]; ok {
				inputTokens = append(append([]int{}, prefix...), inputTokens...)
			}
		}

		outputTokens := sim.GenerateRandomTokenIDs(rng, rec.OutputTokens)

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
			ClientID:         rec.ClientID,
			PrefixGroup:      rec.PrefixGroup,
			PrefixLength:     rec.PrefixLength,
			Streaming:        rec.Streaming,
			// ServerInputTokens: not propagated to sim.Request (calibration-only field, BC-7)
		}
		requests = append(requests, req)
	}
	return requests, nil
}

// LoadTraceV2SessionBlueprints groups trace records by session and builds
// SessionBlueprints with SequenceSamplers for deterministic token replay.
// Returns round-0 requests (plus all non-session requests) for initial injection,
// and blueprints for the SessionManager.
//
// thinkTimeOverrideUs > 0: use constant think time for all sessions.
// thinkTimeOverrideUs == 0: derive per-round think time from trace arrival gaps.
//   NOTE: gap-derived think time = ArrivalTimeUs[i] - ArrivalTimeUs[i-1], which
//   equals (service_time[i-1] + client_think_time) when the trace was produced by
//   blis observe. It is NOT pure client think time. Use thinkTimeOverrideUs (i.e.
//   --think-time-ms) to supply the actual client-side think time when replaying an
//   observe-generated trace with accurate inter-round spacing.
// horizon <= 0: defaults to math.MaxInt64.
func LoadTraceV2SessionBlueprints(trace *TraceV2, seed int64, thinkTimeOverrideUs int64, horizon int64) ([]*sim.Request, []SessionBlueprint, error) {
	if trace == nil || len(trace.Records) == 0 {
		return nil, nil, fmt.Errorf("empty trace")
	}
	if horizon <= 0 {
		horizon = math.MaxInt64
	}

	rng := rand.New(rand.NewSource(seed))

	// Generate shared prefix tokens per prefix group (same as LoadTraceV2Requests)
	prefixTokens := make(map[string][]int)
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

	for _, sessionID := range sessionOrder {
		sr := sessionMap[sessionID]
		rounds := sr.records
		if len(rounds) == 0 {
			continue
		}

		// Build per-round token sequences
		inputSeq := make([]int, len(rounds))
		outputSeq := make([]int, len(rounds))
		for i, rec := range rounds {
			inputSeq[i] = rec.InputTokens
			outputSeq[i] = rec.OutputTokens
		}

		// Build think time: override or derive from inter-round arrival gaps
		var thinkTimeSampler LengthSampler
		var thinkTimeUs int64
		if thinkTimeOverrideUs > 0 {
			thinkTimeUs = thinkTimeOverrideUs
		} else if len(rounds) > 1 {
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
			thinkTimeSampler = &SequenceSampler{values: thinkTimes}
		}

		// Per-session RNG for deterministic token ID generation (INV-6)
		sessionRNG := rand.New(rand.NewSource(rng.Int63()))

		// Build round-0 request
		r0 := rounds[0]
		inputTokens := sim.GenerateRandomTokenIDs(sessionRNG, r0.InputTokens)
		if r0.PrefixGroup != "" {
			if prefix, ok := prefixTokens[r0.PrefixGroup]; ok {
				inputTokens = append(append([]int{}, prefix...), inputTokens...)
			}
		}
		outputTokens := sim.GenerateRandomTokenIDs(sessionRNG, r0.OutputTokens)

		var prefix []int
		if r0.PrefixGroup != "" {
			prefix = prefixTokens[r0.PrefixGroup]
		}

		req := &sim.Request{
			ID:              fmt.Sprintf("request_%d", r0.RequestID),
			ArrivalTime:     r0.ArrivalTimeUs,
			InputTokens:     inputTokens,
			OutputTokens:    outputTokens,
			MaxOutputLen:    len(outputTokens),
			State:           sim.StateQueued,
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
			ClientID:        r0.ClientID,
			PrefixGroup:     r0.PrefixGroup,
			PrefixLength:    r0.PrefixLength,
			Streaming:       r0.Streaming,
		}
		requests = append(requests, req)

		bp := SessionBlueprint{
			SessionID:        sessionID,
			ClientID:         r0.ClientID,
			MaxRounds:        len(rounds),
			ThinkTimeUs:      thinkTimeUs,
			ThinkTimeSampler: thinkTimeSampler,
			Horizon:          horizon,
			InputSampler:     &SequenceSampler{values: inputSeq[1:]},  // rounds 1..N
			OutputSampler:    &SequenceSampler{values: outputSeq[1:]}, // rounds 1..N
			RNG:              sessionRNG,
			Prefix:           prefix,
			TenantID:         r0.TenantID,
			SLOClass:         r0.SLOClass,
			Model:            r0.Model,
		}
		blueprints = append(blueprints, bp)
	}

	// Append non-session requests (same construction as LoadTraceV2Requests)
	for _, rec := range nonSessionRecords {
		inputTokens := sim.GenerateRandomTokenIDs(rng, rec.InputTokens)
		if rec.PrefixGroup != "" {
			if prefix, ok := prefixTokens[rec.PrefixGroup]; ok {
				inputTokens = append(append([]int{}, prefix...), inputTokens...)
			}
		}
		outputTokens := sim.GenerateRandomTokenIDs(rng, rec.OutputTokens)
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
			ClientID:        rec.ClientID,
			PrefixGroup:     rec.PrefixGroup,
			PrefixLength:    rec.PrefixLength,
			Streaming:       rec.Streaming,
		}
		requests = append(requests, req)
	}

	return requests, blueprints, nil
}
