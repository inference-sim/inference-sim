package workload

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
)

// GenerateReasoningRequests generates multi-turn conversation requests.
// For each session: round 0 is a normal request; subsequent rounds have
// increasing input lengths if context_growth == "accumulate".
//
// When prefix is non-empty it is prepended to every round's InputTokens
// (req.PrefixLength is set to len(prefix)). Under accumulate, prefix becomes
// the first chunk in the shared session buffer so all rounds share the same
// prefix bytes — eliminating the legacy O(R) prefix-copy tax per round
// (#1445).
func GenerateReasoningRequests(
	rng *rand.Rand,
	spec *ReasoningSpec,
	inputSampler, outputSampler LengthSampler,
	startTime int64,
	clientID, tenantID, sloClass, model string,
	prefix []int,
) ([]*sim.Request, error) {
	if spec == nil || spec.MultiTurn == nil {
		return nil, nil
	}
	mt := spec.MultiTurn
	if mt.MaxRounds <= 0 {
		return nil, nil
	}

	// Sample reason ratio distribution
	var reasonRatioSampler LengthSampler
	if spec.ReasonRatioDist.Type != "" {
		var err error
		reasonRatioSampler, err = NewLengthSampler(spec.ReasonRatioDist)
		if err != nil {
			return nil, fmt.Errorf("reason ratio distribution: %w", err)
		}
	}

	sessionID := fmt.Sprintf("sess_%d", rng.Int63())
	var requests []*sim.Request
	currentTime := startTime
	// Shared session token buffer (#1445). Replaces the prior O(R²) eager copy
	// pattern. Each accumulate round's InputTokens is a flat sub-slice into
	// buf's underlying array; buf grows by amortized-O(1) append, so total
	// per-session memory is O(R) including the prefix (seeded once below).
	//
	// We pre-allocate to a heuristic capacity based on the first round's
	// sampled lengths × MaxRounds. This eliminates within-loop reallocation
	// for typical workloads (input/output sampler variance is bounded), making
	// every round's Slice() result point into the SAME backing array. Even if
	// the estimate is exceeded, Go's amortized-O(1) append keeps total memory
	// O(R); the only cost is a one-time copy.
	var buf *SessionTokenBuffer
	prefixLen := int64(len(prefix))

	// Pre-sample round 0 to size the buffer; these values feed round 0 below.
	round0Input := inputSampler.Sample(rng)
	round0Output := outputSampler.Sample(rng)

	if mt.ContextGrowth == "accumulate" {
		estCap := prefixLen + int64(mt.MaxRounds)*int64(round0Input+round0Output)
		buf = NewSessionTokenBufferWithCapacity(estCap)
		if prefixLen > 0 {
			buf.Append(prefix)
		}
	} else {
		buf = NewSessionTokenBuffer()
	}

	for round := 0; round < mt.MaxRounds; round++ {
		var inputLen, outputLen int
		if round == 0 {
			inputLen, outputLen = round0Input, round0Output
		} else {
			inputLen = inputSampler.Sample(rng)
			outputLen = outputSampler.Sample(rng)
		}

		// Generate this round's new tokens
		newInputTokens := sim.GenerateRandomTokenIDs(rng, inputLen)
		outputTokens := sim.GenerateRandomTokenIDs(rng, outputLen)

		// Build input.
		// - Accumulate: route through the shared buffer (prefix already seeded
		//   at index 0). Round N's InputTokens spans [0, end_of_rN_input).
		// - Non-accumulate: each round is fresh; if prefix is set, prepend it
		//   (one O(prefix+input) copy per round; O(R) total — not the
		//   quadratic pattern we're eliminating).
		var inputTokens []int
		if mt.ContextGrowth == "accumulate" {
			_, inputEnd := buf.Append(newInputTokens)
			inputTokens = buf.Slice(0, inputEnd)
		} else if prefixLen > 0 {
			inputTokens = make([]int, 0, len(prefix)+inputLen)
			inputTokens = append(inputTokens, prefix...)
			inputTokens = append(inputTokens, newInputTokens...)
		} else {
			inputTokens = newInputTokens
		}

		// Reason ratio
		reasonRatio := 0.0
		if reasonRatioSampler != nil {
			// Sample as integer percentage then convert
			pct := reasonRatioSampler.Sample(rng)
			reasonRatio = math.Min(1.0, math.Max(0.0, float64(pct)/100.0))
		}

		req := &sim.Request{
			ID:           "", // assigned later
			ArrivalTime:  currentTime,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			MaxOutputLen: len(outputTokens),
			State:        sim.StateQueued,
			TenantID:     tenantID,
			SLOClass:     sloClass,
			Model:        model,
			ClientID:     clientID,
			SessionID:    sessionID,
			RoundIndex:   round,
			ReasonRatio:  reasonRatio,
			PrefixLength: int(prefixLen),
		}
		requests = append(requests, req)

		// Update accumulated context for next round: this round's input is
		// already in the buffer; append its output so the next round sees
		// the full history.
		if mt.ContextGrowth == "accumulate" {
			buf.Append(outputTokens)
		}

		// Next round arrives after think time
		currentTime += mt.ThinkTimeUs
		// Add estimated completion time (simple heuristic: 1µs per output token)
		currentTime += int64(outputLen)
	}
	return requests, nil
}
