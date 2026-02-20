package workload

import (
	"fmt"
	"math/rand"

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

	// Generate shared prefix tokens per prefix group
	prefixTokens := make(map[string][]int)
	for _, rec := range trace.Records {
		if rec.PrefixGroup != "" {
			if _, exists := prefixTokens[rec.PrefixGroup]; !exists {
				prefixTokens[rec.PrefixGroup] = sim.GenerateRandomTokenIDs(rng, 50)
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
			State:            sim.StateQueued,
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
			TenantID:         rec.TenantID,
			SLOClass:         rec.SLOClass,
			Streaming:        rec.Streaming,
			SessionID:        rec.SessionID,
			RoundIndex:       rec.RoundIndex,
			TextTokenCount:   rec.TextTokens,
			ImageTokenCount:  rec.ImageTokens,
			AudioTokenCount:  rec.AudioTokens,
			VideoTokenCount:  rec.VideoTokens,
			ReasonRatio:      rec.ReasonRatio,
		}
		requests = append(requests, req)
	}
	return requests, nil
}
