package llmd

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/internal/testutil/fakelatency"
)

func testDeployment(n int) cluster.DeploymentConfig {
	return cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyModel:        fakelatency.New(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, "test-model", "H100", 1, 1, false, 0),
		},
		NumInstances: n,
	}
}

func testRequests(n int, seed int64, gapUs int64, prefixLen int) []*sim.Request {
	rng := rand.New(rand.NewSource(seed))
	prefix := sim.GenerateRandomTokenIDs(rng, prefixLen)
	reqs := make([]*sim.Request, n)
	for i := range reqs {
		in := append(append([]sim.TokenID{}, prefix...), sim.GenerateRandomTokenIDs(rng, 32+rng.Intn(64))...)
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("request_%d", i),
			ArrivalTime:  int64(i) * gapUs,
			InputTokens:  in,
			OutputTokens: sim.GenerateRandomTokenIDs(rng, 64+rng.Intn(64)),
			State:        sim.StateQueued,
		}
	}
	return reqs
}

type runResult struct {
	cs      *cluster.ClusterSimulator
	gw      *Gateway
	reqs    []*sim.Request
	byInst  map[string]int
	metrics *sim.Metrics
}

func runCluster(t *testing.T, n, numReqs int, opts Options) runResult {
	t.Helper()
	reqs := testRequests(numReqs, 7, 500, 0)
	cs := cluster.NewClusterSimulator(testDeployment(n), cluster.NewSliceRequestSource(reqs), nil)
	if opts.Model == "" {
		opts.Model = "test-model"
	}
	gw := New(opts)
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	by := map[string]int{}
	for _, r := range reqs {
		by[r.AssignedInstance]++
	}
	return runResult{cs: cs, gw: gw, reqs: reqs, byInst: by, metrics: cs.AggregatedMetrics()}
}

func TestGateway_DefaultRouterServesEveryRequest(t *testing.T) {
	res := runCluster(t, 3, 60, Options{})
	if got := res.metrics.CompletedRequests; got != 60 {
		t.Fatalf("completed %d of 60 requests (rejected %d, routing-rejected %d)", got, res.cs.RejectedRequests(), res.cs.RoutingRejections())
	}
	if len(res.byInst) < 3 {
		t.Errorf("the router left an instance idle under queueing load: %v", res.byInst)
	}
	t.Logf("requests per instance: %v", res.byInst)
	if live := res.gw.LiveCoroutines(); live != 0 {
		t.Errorf("%d router coroutines still alive after Stop", live)
	}
}

func TestGateway_Deterministic(t *testing.T) {
	a := runCluster(t, 4, 80, Options{})
	b := runCluster(t, 4, 80, Options{})
	for i := range a.reqs {
		if a.reqs[i].AssignedInstance != b.reqs[i].AssignedInstance {
			t.Fatalf("request %d routed to %s then %s", i, a.reqs[i].AssignedInstance, b.reqs[i].AssignedInstance)
		}
	}
}

func TestGateway_NoGoroutinesLeak(t *testing.T) {
	before := runtime.NumGoroutine()
	runCluster(t, 2, 20, Options{})
	runtime.GC()
	if after := runtime.NumGoroutine(); after > before+2 {
		t.Errorf("goroutines: %d before, %d after a simulation", before, after)
	}
}
