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
func GenerateReasoningRequests(
	rng *rand.Rand,
	spec *ReasoningSpec,
	inputSampler, outputSampler LengthSampler,
	startTime int64,
	clientID, tenantID, sloClass string,
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
	var contextPrefix []int // accumulated context from prior rounds (actual tokens)

	for round := 0; round < mt.MaxRounds; round++ {
		inputLen := inputSampler.Sample(rng)
		outputLen := outputSampler.Sample(rng)

		// Generate this round's new tokens
		newInputTokens := sim.GenerateRandomTokenIDs(rng, inputLen)
		outputTokens := sim.GenerateRandomTokenIDs(rng, outputLen)

		// Build input: prepend accumulated context if accumulating
		var inputTokens []int
		if mt.ContextGrowth == "accumulate" && round > 0 {
			inputTokens = append(append([]int{}, contextPrefix...), newInputTokens...)
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

		req := sim.NewRequest("", currentTime, inputTokens, outputTokens)
		req.TenantID = tenantID
		req.SLOClass = sloClass
		req.SessionID = sessionID
		req.RoundIndex = round
		req.ReasonRatio = reasonRatio
		requests = append(requests, req)

		// Update accumulated context for next round: append this round's
		// new input + output tokens so the next round sees the full history.
		if mt.ContextGrowth == "accumulate" {
			contextPrefix = append(contextPrefix, newInputTokens...)
			contextPrefix = append(contextPrefix, outputTokens...)
		}

		// Next round arrives after think time
		currentTime += mt.ThinkTimeUs
		// Add estimated completion time (simple heuristic: 1µs per output token)
		currentTime += int64(outputLen)
	}
	return requests, nil
}
