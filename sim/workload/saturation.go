package workload

import (
	"math"

	sim "github.com/inference-sim/inference-sim/sim"
)

// ComputeSimEndUs computes the simulation end time from request completion times.
// Returns max(all completion times, horizon) where horizon is used as a floor if > 0.
// This is the canonical simEndUs calculation used by run, replay, and calibrate commands.
func ComputeSimEndUs(requests []*sim.Request, horizon int64) int64 {
	simEndUs := int64(0)
	for _, req := range requests {
		completionUs := req.ArrivalTime
		if req.TTFTSet {
			completionUs += req.FirstTokenTime
			// Only sum ITL if request has valid TTFT (prevents malformed data from inflating simEndUs)
			for _, itl := range req.ITL {
				completionUs += itl
			}
		}
		if completionUs > simEndUs {
			simEndUs = completionUs
		}
	}
	// Use horizon as floor if explicitly set and larger
	if horizon > 0 && horizon < math.MaxInt64 && horizon > simEndUs {
		simEndUs = horizon
	}
	return simEndUs
}
