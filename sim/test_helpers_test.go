package sim

import (
	"fmt"
	"math"
	"math/rand"
)

// testGenerateRequests replicates the exact algorithm from the old
// generateRequestsFromDistribution using SubsystemWorkload RNG,
// preserving byte-identical test request sequences during legacy retirement.
//
// INTENTIONAL DUPLICATION: an identical copy exists in sim/cluster/test_helpers_test.go.
// Both use SubsystemWorkload (legacy) rather than SubsystemWorkloadGen (production).
// This is deliberate: existing tests validate behavior against known sequences.
// The RNG stream change is documented as a deviation in the PR description.
// TODO: consolidate into sim/internal/testutil/ once golden dataset is regenerated.
func testGenerateRequests(seed, horizon int64, rate float64,
	numReqs, prefix, pMean, pStd, pMin, pMax, oMean, oStd, oMin, oMax int,
) []*Request {
	rng := NewPartitionedRNG(NewSimulationKey(seed))
	workloadRNG := rng.ForSubsystem(SubsystemWorkload)

	var requests []*Request
	currentTime := int64(0)
	reqIdx := 0

	prefixTokens := GenerateRandomTokenIDs(workloadRNG, prefix)

	for currentTime < horizon && reqIdx < numReqs {
		promptLen := generateLengthGauss(workloadRNG, pMean, pStd, pMin, pMax)
		prompt := GenerateRandomTokenIDs(workloadRNG, promptLen)
		input := append(append([]int{}, prefixTokens...), prompt...)

		outputLen := generateLengthGauss(workloadRNG, oMean, oStd, oMin, oMax)
		output := GenerateRandomTokenIDs(workloadRNG, outputLen)

		requests = append(requests, &Request{
			ID:               fmt.Sprintf("request_%v", reqIdx),
			ArrivalTime:      currentTime,
			InputTokens:      input,
			OutputTokens:     output,
			State:            StateQueued,
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

// injectRequests is a test helper that injects pre-generated requests into a simulator.
func injectRequests(sim *Simulator, requests []*Request) {
	for _, req := range requests {
		sim.InjectArrival(req)
	}
}
