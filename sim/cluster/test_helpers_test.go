package cluster

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
	"github.com/inference-sim/inference-sim/sim/internal/testutil/fakelatency"
)

// testGenerateRequests replicates the exact algorithm from the old
// generateRequestsFromDistribution using SubsystemWorkload RNG,
// preserving byte-identical test request sequences during legacy retirement.
//
// INTENTIONAL DUPLICATION: an identical copy exists in sim/test_helpers_test.go.
// Both use SubsystemWorkload (legacy) rather than SubsystemWorkloadGen (production).
// This is deliberate: existing tests validate behavior against known sequences.
// The RNG stream change is documented as a deviation in the PR description.
// TODO: consolidate into sim/internal/testutil/ once golden dataset is regenerated.
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
		input := append(append([]sim.TokenID{}, prefixTokens...), prompt...)

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

// testModelConfig returns a dense sim.ModelConfig for tests that need one. Step time comes
// from the fake latency model (testFakeLatency).
func testModelConfig() sim.ModelConfig {
	return sim.ModelConfig{}
}

// testFakeLatency is the latency model every cluster behavior test prices steps with:
// the deterministic, stateless fake from sim/internal/testutil (default coefficients).
// Set it as SimConfig.LatencyModel so instances never build a pricing backend.
func testFakeLatency() sim.LatencyModel { return fakelatency.New() }

// testFakeZeroQueueing is testFakeLatency with no arrival-to-queue delay, for tests that
// need a request to enter the wait queue at its arrival tick.
func testFakeZeroQueueing() sim.LatencyModel {
	c := testutil.DefaultFakeLatency()
	c.QueueingTicks = 0
	return fakelatency.WithCoeffs(c)
}

// testPDTransferTime is the KV-handoff price every PD behavior test injects as
// DeploymentConfig.PDTransferTime: a fixed setup cost plus a per-token term, so a larger
// handoff takes longer and every handoff takes at least one tick. The simulator only decides
// when a transfer happens; how long it takes is the pricer's, so tests supply one.
func testPDTransferTime(tokens int64, _, _ InstanceID) int64 { return 50 + tokens/50 }

// newTestRequests creates test requests matching the old newTestWorkload(n) behavior:
// rate=10/1e6, seed=42, horizon=MaxInt64, no prefix, prompt mean=100 std=20 [10,200],
// output mean=50 std=10 [10,100].
func newTestRequests(n int) []*sim.Request {
	return testGenerateRequests(42, math.MaxInt64, 10.0/1e6, n,
		0, 100, 20, 10, 200, 50, 10, 10, 100)
}
