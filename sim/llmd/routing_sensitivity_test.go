package llmd

import (
	"math"
	"testing"
	"time"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// scaledDeployment is testDeployment whose fake engine prices every step, and every per-request
// cost, at scale times the default: the test's stand-in for a small refit of a cost coefficient.
func scaledDeployment(n int, scale float64) cluster.DeploymentConfig {
	f := testutil.DefaultFakeLatency()
	s := func(v int64) int64 { return int64(math.Round(float64(v) * scale)) }
	f.BaseTicks, f.PerScheduledTokenTicks, f.PerDecodeRequestTicks = s(f.BaseTicks), s(f.PerScheduledTokenTicks), s(f.PerDecodeRequestTicks)
	f.PerContextTokenMilliTicks = s(f.PerContextTokenMilliTicks)
	return deploymentOf(cengine.SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       cengine.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         cengine.NewBatchConfig(256, 2048, 0),
		LatencyModel:        fakeLatency{c: f},
		ModelHardwareConfig: cengine.NewModelHardwareConfig(cengine.ModelConfig{}, "test-model", "H100", 1, 1, false, 0),
	}, n)
}

// splitAt runs 80 requests that share a prefix and arrive every gapUs through the default router
// and returns the share served by the busiest instance, and how many decisions did not pick the
// endpoint with the highest recorded score.
func splitAt(t *testing.T, scale float64, gapUs int64, jitter time.Duration, prefix int) (share float64, offArgmax int) {
	t.Helper()
	reqs := testRequests(80, 13, gapUs, prefix)
	dep := scaledDeployment(2, scale)
	dep.TraceLevel = "decisions"
	cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
	cs.SetGateway(New(Options{Model: "test-model", ScrapeJitter: jitter}))
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	by := map[string]int{}
	for _, r := range reqs {
		by[r.AssignedInstance]++
	}
	most := 0
	for _, n := range by {
		if n > most {
			most = n
		}
	}
	for _, rec := range cs.Trace().Routings {
		best := math.Inf(-1)
		for _, sc := range rec.Scores {
			best = math.Max(best, sc)
		}
		if rec.Scores[rec.ChosenInstance] < best {
			offArgmax++
		}
	}
	return float64(most) / float64(len(reqs)), offArgmax
}

// Pre-registered before the runs (router lane, byte-gate moe-prefix finding): the default router
// (queue-scorer 2, kv-cache-utilization-scorer 2, prefix-cache-scorer 3; max-score picker;
// reference-configs/default.yaml in the fork) is an argmax of weighted scores, so its split is not
// a smooth function of a cost coefficient and need not be. Two things must hold, as in the real
// router:
//
//  1. every decision picks an endpoint whose recorded weighted score is the highest, at every
//     cost scale, scrape phase and prefix length (the host adds no tie-break or threshold of its
//     own);
//  2. when the prefix score alone outweighs the largest possible queue score difference
//     (3 x matched fraction > 2), the whole stream follows the first prefix hit, so a small change
//     of cost cannot move the split at all.
//
// A split that jumps between 42/38 and 0/80 under a 1% cost change therefore needs a cause in the
// router's inputs (scrape phase against the arrival period), not in its decision rule.
func TestRoutingSensitivity_DecisionIsTheArgmaxOfItsScores(t *testing.T) {
	for _, prefix := range []int{0, 48, 128} {
		for _, jitter := range []time.Duration{0, 50 * time.Millisecond} {
			for _, scale := range []float64{80, 100, 100.5, 101, 130} {
				if _, off := splitAt(t, scale, 50000, jitter, prefix); off != 0 {
					t.Errorf("prefix=%d jitter=%v scale=%v: %d decisions did not pick the highest-scored endpoint", prefix, jitter, scale, off)
				}
			}
		}
	}
}

func TestRoutingSensitivity_DominantPrefixPinsTheStreamWhateverTheCost(t *testing.T) {
	for _, jitter := range []time.Duration{0, 50 * time.Millisecond} {
		for _, scale := range []float64{1, 1.01, 20, 100, 101, 102, 240} {
			if share, _ := splitAt(t, scale, 50000, jitter, 128); share != 1 {
				t.Errorf("jitter=%v scale=%v: busiest instance served %.3f of the stream, want all of it (prefix score 3x(>0.67) beats the queue score range 2)", jitter, scale, share)
			}
		}
	}
}
