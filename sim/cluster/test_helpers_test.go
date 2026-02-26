package cluster

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
)

// testGenerateRequests replicates the exact algorithm from the old
// generateRequestsFromDistribution using SubsystemWorkload RNG,
// preserving byte-identical test request sequences during legacy retirement.
func testGenerateRequests(seed, horizon int64, rate float64,
	numReqs, prefix, pMean, pStd, pMin, pMax, oMean, oStd, oMin, oMax int,
) []*sim.Request {
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(seed))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkload)

	var requests []*sim.Request
	currentTime := int64(0)
	reqIdx := 0

	prefixTokens := sim.GenerateRandomTokenIDs(workloadRNG, prefix)

	for currentTime < horizon && reqIdx < numReqs {
		promptLen := generateLengthGauss(workloadRNG, pMean, pStd, pMin, pMax)
		prompt := sim.GenerateRandomTokenIDs(workloadRNG, promptLen)
		input := append(append([]int{}, prefixTokens...), prompt...)

		outputLen := generateLengthGauss(workloadRNG, oMean, oStd, oMin, oMax)
		output := sim.GenerateRandomTokenIDs(workloadRNG, outputLen)

		requests = append(requests, &sim.Request{
			ID:               fmt.Sprintf("request_%v", reqIdx),
			ArrivalTime:      currentTime,
			InputTokens:      input,
			OutputTokens:     output,
			State:            sim.StateQueued,
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
		})

		currentTime += int64(1 / rate)
		reqIdx++
		if currentTime > horizon {
			break
		}
	}
	return requests
}

// generateLengthGauss samples a length from a clamped Gaussian distribution.
// Replicated from the deleted sim/workload_config.go for test backward compat.
func generateLengthGauss(rng *rand.Rand, mean, std, min, max int) int {
	if min == max {
		return min
	}
	val := rng.NormFloat64()*float64(std) + float64(mean)
	clampedVal := math.Min(float64(max), val)
	clampedVal = math.Max(float64(min), clampedVal)
	return int(math.Round(clampedVal))
}

// newTestRequests creates test requests matching the old newTestWorkload(n) behavior:
// rate=10/1e6, seed=42, horizon=MaxInt64, no prefix, prompt mean=100 std=20 [10,200],
// output mean=50 std=10 [10,100].
func newTestRequests(n int) []*sim.Request {
	return testGenerateRequests(42, math.MaxInt64, 10.0/1e6, n,
		0, 100, 20, 10, 200, 50, 10, 10, 100)
}
