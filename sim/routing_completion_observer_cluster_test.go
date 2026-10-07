package sim_test

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// End-to-end through cluster.NewClusterSimulator's STARTUP wiring site: a scorer
// that subscribes to request completions is told once per completed request, with
// the instance it was routed to. Lives in sim_test (not sim/cluster) because only
// package sim's test build can register a scorer (export_test.go).

func completionTestConfig(n int, scorer string) cluster.DeploymentConfig {
	return cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       math.MaxInt64,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(
				sim.ModelConfig{NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 8, BytesPerParam: 2},
				sim.HardwareCalib{TFlopsPeak: 989.0, BwPeakTBs: 3.35, MfuPrefill: 0.55, MfuDecode: 0.30},
				"test-model", "H100", 1, 1, false, "", "roofline", 0),
		},
		NumInstances:         n,
		RoutingPolicy:        "weighted",
		RoutingScorerConfigs: []sim.ScorerConfig{{Name: scorer, Weight: 1}, {Name: "queue-depth", Weight: 1}},
	}
}

func completionTestRequests(n int) []*sim.Request {
	reqs := make([]*sim.Request, n)
	for i := range reqs {
		in := make([]sim.TokenID, 50+i)
		out := make([]sim.TokenID, 20+i%7)
		for j := range in {
			in[j] = sim.TokenID(i*1000 + j)
		}
		reqs[i] = &sim.Request{ID: fmt.Sprintf("r%02d", i), ArrivalTime: int64(i) * 2000,
			InputTokens: in, OutputTokens: out, State: sim.StateQueued}
	}
	return reqs
}

type routedEvent struct {
	instance string
	tick     int64
}

func runWithCompletionScorer(t *testing.T, subscribe bool) (*cluster.ClusterSimulator, map[string]string, map[string][]routedEvent) {
	t.Helper()
	routed := map[string]string{}
	completed := map[string][]routedEvent{}
	var onComplete func(*sim.Request, string, int64)
	if subscribe {
		onComplete = func(req *sim.Request, inst string, tick int64) {
			completed[req.ID] = append(completed[req.ID], routedEvent{inst, tick})
		}
	}
	neutral := func(_ *sim.Request, snaps []sim.RoutingSnapshot) map[string]float64 {
		s := make(map[string]float64, len(snaps))
		for _, sn := range snaps {
			s[sn.ID] = 0.5
		}
		return s
	}
	restore := sim.RegisterCompletionScorerForTest("test-completion-e2e", neutral,
		func(req *sim.Request, target string) { routed[req.ID] = target }, onComplete)
	defer restore()
	cs := cluster.NewClusterSimulator(completionTestConfig(3, "test-completion-e2e"),
		cluster.NewSliceRequestSource(completionTestRequests(30)), nil)
	require.NoError(t, cs.Run())
	return cs, routed, completed
}

func TestRequestCompletionObserver_ClusterStartupSite_FiresOncePerCompletion(t *testing.T) {
	// no t.Parallel: mutates package-global scorerRegistry
	cs, routed, completed := runWithCompletionScorer(t, true)
	m := cs.AggregatedMetrics()
	require.Equal(t, 30, m.CompletedRequests, "precondition: every request completes")
	require.Len(t, routed, 30, "precondition: every request was routed")
	assert.Len(t, completed, m.CompletedRequests)
	instances := map[string]bool{}
	for id, evs := range completed {
		require.Len(t, evs, 1, "request %s: exactly one completion event", id)
		assert.Equal(t, routed[id], evs[0].instance, "request %s: completion on the routed instance", id)
		assert.Positive(t, evs[0].tick)
		instances[evs[0].instance] = true
	}
	assert.Greater(t, len(instances), 1, "precondition: routing spread over >1 instance")
}

// Subscribing a (no-op) observer must not change simulation results.
func TestRequestCompletionObserver_ClusterResultsUnchanged(t *testing.T) {
	// no t.Parallel: mutates package-global scorerRegistry
	csOn, routedOn, _ := runWithCompletionScorer(t, true)
	csOff, routedOff, _ := runWithCompletionScorer(t, false)
	assert.Equal(t, routedOff, routedOn)
	assert.True(t, reflect.DeepEqual(csOff.AggregatedMetrics(), csOn.AggregatedMetrics()),
		"aggregated metrics must be identical with and without a completion subscriber")
}
