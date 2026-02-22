package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// makeSharedPrefixRequests creates numRequests requests where sharedFraction
// share a common prefix of prefixLen tokens. The rest have unique tokens.
// Each request has totalLen total input tokens and outputLen output tokens.
func makeSharedPrefixRequests(numRequests int, sharedFraction float64,
	prefixLen, suffixLen, outputLen int, interarrivalUs int64) []*sim.Request {

	// Create the shared prefix
	sharedPrefix := make([]int, prefixLen)
	for i := range sharedPrefix {
		sharedPrefix[i] = 1000 + i
	}

	sharedCount := int(float64(numRequests) * sharedFraction)
	var requests []*sim.Request
	for i := 0; i < numRequests; i++ {
		var inputTokens []int
		if i < sharedCount {
			// Shared prefix + unique suffix
			inputTokens = append([]int{}, sharedPrefix...)
			for j := 0; j < suffixLen; j++ {
				inputTokens = append(inputTokens, 50000+i*1000+j)
			}
		} else {
			// Completely unique tokens
			inputTokens = make([]int, prefixLen+suffixLen)
			for j := range inputTokens {
				inputTokens[j] = 90000 + i*1000 + j
			}
		}
		outputTokens := make([]int, outputLen)
		for j := range outputTokens {
			outputTokens[j] = j
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
// BetaCoeffs/AlphaCoeffs match the existing cluster test conventions.
func baseDeploymentConfig(numInstances int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:            50000000, // 50 seconds
			Seed:               42,
			TotalKVBlocks:      2000,
			BlockSizeTokens:    16,
			MaxRunningReqs:     64,
			MaxScheduledTokens: 65536,
			BetaCoeffs:         []float64{1000, 10, 5},
			AlphaCoeffs:        []float64{100, 50, 25},
			Model:              "test-model",
		},
		NumInstances: numInstances,
		TraceLevel:   "decisions",
	}
}

// TestPrefixAffinityRouting_LongPrefix_ConcentratesVsLoadOnly tests the core
// hypothesis: with a long shared prefix (256 tokens = 16 blocks), prefix-affinity
// dominant routing MUST produce a more concentrated distribution than load-only.
//
// This uses a realistic cluster simulation with proper processing times so queues
// build up and the load scorer actually differentiates instances.
func TestPrefixAffinityRouting_LongPrefix_ConcentratesVsLoadOnly(t *testing.T) {
	numInstances := 4
	numRequests := 40

	// 256-token shared prefix (16 blocks at block_size=16) + 64-token suffix (4 blocks)
	// Prefix match ratio = 16/20 = 0.80
	// At 500µs between requests (2000 req/s), queues build up with realistic betas.
	requests := makeSharedPrefixRequests(
		numRequests,
		1.0,   // 100% shared prefix for maximum signal
		256,   // prefixLen
		64,    // suffixLen
		16,    // outputLen
		500,   // 500µs interarrival = 2000 req/s
	)

	config := baseDeploymentConfig(numInstances)

	// Experiment A: prefix-affinity dominant
	affinityConfig := config
	affinityConfig.RoutingPolicy = "weighted"
	affinityConfig.RoutingScorerConfigs = []sim.ScorerConfig{
		{Name: "prefix-affinity", Weight: 5.0},
		{Name: "queue-depth", Weight: 1.0},
	}
	guideLLM := &sim.GuideLLMConfig{Rate: 0.002, NumRequests: numRequests}
	affinityCS := NewClusterSimulator(affinityConfig, guideLLM, "")
	affinityCS.SetPreGeneratedRequests(copyRequests(requests))
	require.NoError(t, affinityCS.Run())
	affinityDist := getRoutingDistribution(affinityCS)

	// Experiment B: load-only
	loadConfig := config
	loadConfig.RoutingPolicy = "weighted"
	loadConfig.RoutingScorerConfigs = []sim.ScorerConfig{
		{Name: "queue-depth", Weight: 1.0},
	}
	loadCS := NewClusterSimulator(loadConfig, guideLLM, "")
	loadCS.SetPreGeneratedRequests(copyRequests(requests))
	require.NoError(t, loadCS.Run())
	loadDist := getRoutingDistribution(loadCS)

	// Experiment C: round-robin baseline
	rrConfig := config
	rrConfig.RoutingPolicy = "round-robin"
	rrCS := NewClusterSimulator(rrConfig, guideLLM, "")
	rrCS.SetPreGeneratedRequests(copyRequests(requests))
	require.NoError(t, rrCS.Run())
	rrDist := getRoutingDistribution(rrCS)

	t.Logf("Prefix-affinity (5:1): %v (max=%d)", affinityDist, maxValue(affinityDist))
	t.Logf("Load-only (queue-depth):     %v (max=%d)", loadDist, maxValue(loadDist))
	t.Logf("Round-robin:                 %v (max=%d)", rrDist, maxValue(rrDist))

	// Core assertion: prefix-affinity dominant should be more concentrated
	// than load-only for prefix-heavy workloads.
	assert.Greater(t, maxValue(affinityDist), maxValue(loadDist),
		"prefix-affinity dominant MUST concentrate more than load-only for prefix-heavy workloads")

	// Secondary: load-only should spread more evenly than prefix-affinity
	// (load scorer actively balances while prefix-affinity attracts to cached instance)
	affinityUsed := countNonZero(affinityDist)
	loadUsed := countNonZero(loadDist)
	t.Logf("Instances used: prefix-affinity=%d, load-only=%d", affinityUsed, loadUsed)
}

// TestPrefixAffinityRouting_ShortPrefix_NoAdvantage verifies that with a very
// short shared prefix (32 tokens = 2 blocks out of 20), prefix-affinity does
// NOT dominate — load balancing should still be the primary signal because
// 2/20 = 0.10 match ratio is a weak signal.
func TestPrefixAffinityRouting_ShortPrefix_NoAdvantage(t *testing.T) {
	numInstances := 4
	numRequests := 40

	// 32-token shared prefix (2 blocks) + 288-token suffix (18 blocks)
	// Match ratio = 2/20 = 0.10 — very weak signal
	requests := makeSharedPrefixRequests(
		numRequests, 1.0, 32, 288, 16, 500,
	)

	config := baseDeploymentConfig(numInstances)

	// Prefix-affinity dominant
	affinityConfig := config
	affinityConfig.RoutingPolicy = "weighted"
	affinityConfig.RoutingScorerConfigs = []sim.ScorerConfig{
		{Name: "prefix-affinity", Weight: 5.0},
		{Name: "queue-depth", Weight: 1.0},
	}
	guideLLM := &sim.GuideLLMConfig{Rate: 0.002, NumRequests: numRequests}
	affinityCS := NewClusterSimulator(affinityConfig, guideLLM, "")
	affinityCS.SetPreGeneratedRequests(copyRequests(requests))
	require.NoError(t, affinityCS.Run())
	affinityDist := getRoutingDistribution(affinityCS)

	// Load-only
	loadConfig := config
	loadConfig.RoutingPolicy = "weighted"
	loadConfig.RoutingScorerConfigs = []sim.ScorerConfig{
		{Name: "queue-depth", Weight: 1.0},
	}
	loadCS := NewClusterSimulator(loadConfig, guideLLM, "")
	loadCS.SetPreGeneratedRequests(copyRequests(requests))
	require.NoError(t, loadCS.Run())
	loadDist := getRoutingDistribution(loadCS)

	t.Logf("Short prefix — affinity (5:1): %v (max=%d)", affinityDist, maxValue(affinityDist))
	t.Logf("Short prefix — load-only:      %v (max=%d)", loadDist, maxValue(loadDist))

	// With a weak prefix signal (0.10), the load scorer should dominate
	// and the distributions should be similar. Soft assertion: prefix-affinity
	// should NOT concentrate significantly more than load-only.
	assert.LessOrEqual(t, maxValue(affinityDist), maxValue(loadDist)+5,
		"short prefix should not concentrate significantly more than load-only")
}

// TestPrefixAffinityRouting_MultiTurn_SessionAffinity verifies that
// prefix-affinity keeps same-session rounds on the same instance,
// while load-only scatters them. This is the multi-turn cache locality test.
func TestPrefixAffinityRouting_MultiTurn_SessionAffinity(t *testing.T) {
	numInstances := 4
	numSessions := 20
	roundsPerSession := 5

	// Build multi-turn requests: each session has 5 rounds with accumulating context
	var requests []*sim.Request
	reqID := 0
	baseTime := int64(0)
	for s := 0; s < numSessions; s++ {
		sessionID := fmt.Sprintf("session_%d", s)

		// Generate the session's base prefix (unique per session)
		prefix := make([]int, 128) // 128 tokens = 8 blocks
		for i := range prefix {
			prefix[i] = s*10000 + i
		}

		var contextPrefix []int
		for r := 0; r < roundsPerSession; r++ {
			// Build input: context prefix + new tokens for this round
			newTokens := make([]int, 64) // 64 new tokens per round = 4 blocks
			for i := range newTokens {
				newTokens[i] = s*10000 + r*1000 + 5000 + i
			}
			var inputTokens []int
			if r == 0 {
				inputTokens = append([]int{}, prefix...)
				inputTokens = append(inputTokens, newTokens...)
			} else {
				inputTokens = append(append([]int{}, contextPrefix...), newTokens...)
			}

			outputTokens := make([]int, 32)
			for i := range outputTokens {
				outputTokens[i] = i
			}

			requests = append(requests, &sim.Request{
				ID:           fmt.Sprintf("request_%d", reqID),
				InputTokens:  inputTokens,
				OutputTokens: outputTokens,
				State:        sim.StateQueued,
				ArrivalTime:  baseTime,
				SessionID:    sessionID,
				RoundIndex:   r,
			})
			reqID++

			// Accumulate context for next round
			if r == 0 {
				contextPrefix = append(append([]int{}, prefix...), newTokens...)
			} else {
				contextPrefix = append(contextPrefix, newTokens...)
			}
			contextPrefix = append(contextPrefix, outputTokens...)

			baseTime += 200 // 200µs between rounds (fast multi-turn)
		}
		baseTime += 1000 // 1ms between sessions
	}

	config := baseDeploymentConfig(numInstances)
	config.Horizon = baseTime + 50000000

	// Count per-session instance scattering
	countSessionAffinity := func(cs *ClusterSimulator) float64 {
		// Map request → instance from per-instance metrics
		reqToInst := make(map[string]string)
		for _, inst := range cs.Instances() {
			m := inst.Metrics()
			for reqID := range m.Requests {
				reqToInst[reqID] = string(inst.ID())
			}
		}

		// For each session, count how many unique instances were used
		sessInsts := make(map[string]map[string]bool)
		for _, req := range requests {
			if inst, ok := reqToInst[req.ID]; ok {
				if sessInsts[req.SessionID] == nil {
					sessInsts[req.SessionID] = make(map[string]bool)
				}
				sessInsts[req.SessionID][inst] = true
			}
		}

		// Average instances per session (1.0 = perfect affinity, higher = scattered)
		total := 0.0
		count := 0
		for _, insts := range sessInsts {
			total += float64(len(insts))
			count++
		}
		if count == 0 {
			return 0
		}
		return total / float64(count)
	}

	guideLLM := &sim.GuideLLMConfig{Rate: 0.005, NumRequests: len(requests)}

	// Experiment A: prefix-affinity dominant
	affinityConfig := config
	affinityConfig.RoutingPolicy = "weighted"
	affinityConfig.RoutingScorerConfigs = []sim.ScorerConfig{
		{Name: "prefix-affinity", Weight: 5.0},
		{Name: "queue-depth", Weight: 1.0},
	}
	affinityCS := NewClusterSimulator(affinityConfig, guideLLM, "")
	affinityCS.SetPreGeneratedRequests(copyRequests(requests))
	require.NoError(t, affinityCS.Run())
	affinityAffinity := countSessionAffinity(affinityCS)

	// Experiment B: load-only
	loadConfig := config
	loadConfig.RoutingPolicy = "weighted"
	loadConfig.RoutingScorerConfigs = []sim.ScorerConfig{
		{Name: "queue-depth", Weight: 1.0},
	}
	loadCS := NewClusterSimulator(loadConfig, guideLLM, "")
	loadCS.SetPreGeneratedRequests(copyRequests(requests))
	require.NoError(t, loadCS.Run())
	loadAffinity := countSessionAffinity(loadCS)

	t.Logf("Avg instances/session — prefix-affinity: %.2f, load-only: %.2f", affinityAffinity, loadAffinity)
	t.Logf("(1.0 = perfect session affinity, %.1f = fully scattered)", float64(numInstances))

	// Prefix-affinity should have LOWER scatter (closer to 1.0 = sessions stick to one instance)
	assert.Less(t, affinityAffinity, loadAffinity,
		"prefix-affinity should keep sessions together (lower scatter) vs load-only")
}

// --- helpers ---

func copyRequests(reqs []*sim.Request) []*sim.Request {
	out := make([]*sim.Request, len(reqs))
	for i, r := range reqs {
		cp := *r
		cp.InputTokens = append([]int{}, r.InputTokens...)
		cp.OutputTokens = append([]int{}, r.OutputTokens...)
		out[i] = &cp
	}
	return out
}

func getRoutingDistribution(cs *ClusterSimulator) map[string]int {
	dist := make(map[string]int)
	for _, inst := range cs.Instances() {
		m := inst.Metrics()
		dist[string(inst.ID())] = m.CompletedRequests
	}
	return dist
}

func maxValue(m map[string]int) int {
	max := 0
	for _, v := range m {
		if v > max {
			max = v
		}
	}
	return max
}

func countNonZero(m map[string]int) int {
	count := 0
	for _, v := range m {
		if v > 0 {
			count++
		}
	}
	return count
}
