package cluster

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// testGenerateRequests replicates the exact algorithm from the old
// generateRequestsFromDistribution, drawing everything from one math/rand stream seeded with the
// test seed, so the golden dataset's request sequences stay byte-identical. It is test input,
// not the workload generator: production draws come from named streams (sim.NewStream).
func testGenerateRequests(seed, horizon int64, rate float64,
	numReqs, prefix, pMean, pStd, pMin, pMax, oMean, oStd, oMin, oMax int,
) []*sim.Request {
	workloadRNG := rand.New(rand.NewSource(seed))

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

// testModelConfig returns a dense cengine.ModelConfig for tests that need one. Step time comes
// from the fake latency model (testFakeLatency).
func testModelConfig() cengine.ModelConfig {
	return cengine.ModelConfig{}
}

// testFakeLatency is the latency model every cluster behavior test prices steps with:
// the deterministic, stateless fake from sim/internal/testutil (default coefficients).
// Set it as SimConfig.LatencyModel so instances never build a pricing backend.
func testFakeLatency() cengine.LatencyModel { return newFakeLatencyModel() }

// testFakeZeroQueueing is testFakeLatency with no arrival-to-queue delay, for tests that
// need a request to enter the wait queue at its arrival tick.
func testFakeZeroQueueing() cengine.LatencyModel {
	c := testutil.DefaultFakeLatency()
	c.QueueingTicks = 0
	return fakeLatencyWithCoeffs(c)
}

// testPDTransferTime is the KV-handoff price every PD behavior test injects as
// DeploymentConfig.PDTransferTime: a fixed setup cost plus a per-token term, so a larger
// handoff takes longer and every handoff takes at least one tick. The simulator only decides
// when a transfer happens; how long it takes is the pricer's, so tests supply one.
func testPDTransferTime(tokens int64, _, _ plan.Endpoint) int64 { return 50 + tokens/50 }

// testInstanceEndpoint places instance_N on node N of its pool, so every pair of instances is on
// two nodes and a pull's far end names the prefill instance it came from.
func testInstanceEndpoint(id InstanceID) plan.Endpoint {
	n, err := strconv.Atoi(strings.TrimPrefix(string(id), "instance_"))
	if err != nil {
		panic(fmt.Sprintf("testInstanceEndpoint: %q", id))
	}
	return plan.Endpoint{Node: n}
}

// setPDPrice makes every handoff of d take ticks(tokens, from, to) alone, for the blocks' token
// capacity, from the endpoint of the prefill instance to the decode engine's own, and gives d the
// endpoints a PD deployment needs. The decode engine states the move and the driver's Environment
// prices it; a test supplies the price.
func setPDPrice(d *DeploymentConfig, ticks func(tokens int64, from, to plan.Endpoint) int64) {
	setPullEnv(d, func(block int64) testutil.PullEnv { return testutil.FixedPull(block, ticks) })
}

// setPullEnv is setPDPrice for a test that states the whole demand (payload, floor, links).
func setPullEnv(d *DeploymentConfig, mk func(blockTokens int64) testutil.PullEnv) {
	d.InstanceEndpoint = testInstanceEndpoint
	d.NewEngine = compatEngineFactoryWith(d.DPLockstepRanks, func(b int64) env.Environment { return mk(b) })
}

// newTestRequests creates test requests matching the old newTestWorkload(n) behavior:
// rate=10/1e6, seed=42, horizon=MaxInt64, no prefix, prompt mean=100 std=20 [10,200],
// output mean=50 std=10 [10,100].
func newTestRequests(n int) []*sim.Request {
	return testGenerateRequests(42, math.MaxInt64, 10.0/1e6, n,
		0, 100, 20, 10, 200, 50, 10, 10, 100)
}
