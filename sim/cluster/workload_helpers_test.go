package cluster

import (
	"fmt"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
)

// Shared workload helpers of the cluster tests. They lived beside BLIS's native prefix-routing
// tests (prefix_routing_test.go), which went with the native router; the measured-window and
// cache-hit-rate tests (#1938, #1904) still use them.

// makeSharedPrefixRequests creates numRequests requests where sharedFraction
// share a common prefix of prefixLen tokens. The rest have unique tokens.
// Each request has prefixLen+suffixLen input tokens and outputLen output tokens.
func makeSharedPrefixRequests(numRequests int, sharedFraction float64,
	prefixLen, suffixLen, outputLen int, interarrivalUs int64) []*sim.Request {
	sharedPrefix := make([]sim.TokenID, prefixLen)
	for i := range sharedPrefix {
		sharedPrefix[i] = sim.TokenID(1000 + i)
	}
	sharedCount := int(float64(numRequests) * sharedFraction)
	var requests []*sim.Request
	for i := 0; i < numRequests; i++ {
		var inputTokens []sim.TokenID
		if i < sharedCount {
			inputTokens = append([]sim.TokenID{}, sharedPrefix...)
			for j := 0; j < suffixLen; j++ {
				inputTokens = append(inputTokens, sim.TokenID(50000+i*1000+j))
			}
		} else {
			inputTokens = make([]sim.TokenID, prefixLen+suffixLen)
			for j := range inputTokens {
				inputTokens[j] = sim.TokenID(90000 + i*1000 + j)
			}
		}
		outputTokens := make([]sim.TokenID, outputLen)
		for j := range outputTokens {
			outputTokens[j] = sim.TokenID(j)
		}
		requests = append(requests, &sim.Request{
			ID:           fmt.Sprintf("request_%d", i),
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			State:        sim.StateQueued,
			ArrivalTime:  int64(i) * interarrivalUs,
		})
	}
	return requests
}

// baseDeploymentConfig returns a cluster config with realistic processing times.
// Steps are priced by the shared fake latency model (testFakeLatency).
func baseDeploymentConfig(numInstances int) DeploymentConfig {
	return func() DeploymentConfig {
		d := compatDeployment(cengine.SimConfig{
			Horizon:             50000000, // 50 seconds
			Seed:                42,
			KVCacheConfig:       cengine.NewKVCacheConfig(2000, 16, 0, 0, 0, 0),
			BatchConfig:         cengine.NewBatchConfig(64, 65536, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "test-model", "", 1, 1, false, 0),
		}, 0)
		d.NumInstances = numInstances
		d.TraceLevel = "decisions"
		return d
	}()
}

func copyRequests(reqs []*sim.Request) []*sim.Request {
	out := make([]*sim.Request, len(reqs))
	for i, r := range reqs {
		cp := *r
		cp.InputTokens = append([]sim.TokenID{}, r.InputTokens...)
		cp.OutputTokens = append([]sim.TokenID{}, r.OutputTokens...)
		out[i] = &cp
	}
	return out
}

func getRoutingDistribution(cs *ClusterSimulator) map[string]int {
	dist := make(map[string]int)
	for _, inst := range cs.Instances() {
		dist[string(inst.ID())] = inst.Metrics().CompletedRequests
	}
	return dist
}

func maxValue(m map[string]int) int {
	best := 0
	for _, v := range m {
		if v > best {
			best = v
		}
	}
	return best
}

// pinnedGateway is the test stand-in for perfect prefix affinity: every request goes to
// instance_0, so every other instance stays idle.
func pinnedGateway() *fakeGateway { return &fakeGateway{pin: "instance_0"} }
