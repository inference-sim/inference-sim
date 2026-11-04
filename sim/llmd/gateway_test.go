package llmd

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

func testDeployment(n int) cluster.DeploymentConfig {
	return deploymentOf(cengine.SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       cengine.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         cengine.NewBatchConfig(256, 2048, 0),
		LatencyModel:        newFakeLatencyModel(),
		ModelHardwareConfig: cengine.NewModelHardwareConfig(cengine.ModelConfig{}, "test-model", "H100", 1, 1, false, 0),
	}, n)
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
	cs := newCluster(testDeployment(n), cluster.NewSliceRequestSource(reqs), nil)
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

func TestGateway_EmptyPromptDoesNotAbortOtherRequests(t *testing.T) {
	reqs := testRequests(3, 7, 500, 0)
	reqs[1].InputTokens = nil
	var done []string
	cs := newCluster(testDeployment(1), cluster.NewSliceRequestSource(reqs[:2]), func(req *sim.Request, now int64) []*sim.Request {
		done = append(done, req.ID)
		if req.ID == reqs[1].ID {
			reqs[2].ArrivalTime = now
			return []*sim.Request{reqs[2]}
		}
		return nil
	})
	cs.SetGateway(New(Options{Model: "test-model"}))
	var unserved []string
	cs.SetUnservedHook(func(req *sim.Request, _ int64) []*sim.Request {
		unserved = append(unserved, req.ID)
		return nil
	})
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 2 || m.DroppedUnservable != 1 {
		t.Fatalf("completed=%d dropped=%d, want 2 and 1", m.CompletedRequests, m.DroppedUnservable)
	}
	if len(done) != 3 {
		t.Fatalf("terminal callbacks=%v, want all three requests", done)
	}
	if len(unserved) != 1 || unserved[0] != reqs[1].ID {
		t.Fatalf("unserved callbacks=%v, want only %s", unserved, reqs[1].ID)
	}
}

func TestGateway_EmptyPromptAtDeadlineTimesOut(t *testing.T) {
	reqs := testRequests(1, 7, 500, 0)
	reqs[0].InputTokens = nil
	reqs[0].Deadline = 100
	cs := newCluster(testDeployment(1), cluster.NewSliceRequestSource(reqs), nil)
	cs.SetGateway(New(Options{Model: "test-model", GatewayLatencyUs: 100}))
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	m := cs.AggregatedMetrics()
	if m.TimedOutRequests != 1 || m.DroppedUnservable != 0 {
		t.Fatalf("timed out=%d dropped=%d, want 1 and 0", m.TimedOutRequests, m.DroppedUnservable)
	}
}

func TestGateway_EmptyPromptWithLatencyBypassesFlowControl(t *testing.T) {
	reqs := testRequests(3, 7, 0, 0)
	reqs[1].InputTokens = nil
	cs := newCluster(testDeployment(1), cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{
		Model:            "test-model",
		GatewayLatencyUs: 100,
		EPPFlags:         append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+globalStrictConfig),
	})
	routerRequests := make(map[string]bool)
	gw.observe = func(kind, detail string) {
		if kind == "request" {
			routerRequests[detail] = true
		}
	}
	cs.SetGateway(gw)

	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if m := cs.AggregatedMetrics(); m.CompletedRequests != 2 || m.DroppedUnservable != 1 {
		t.Fatalf("completed=%d dropped=%d, want 2 and 1", m.CompletedRequests, m.DroppedUnservable)
	}
	if routerRequests[reqs[1].ID] || reqs[1].GatewayEnqueueTime != 0 {
		t.Errorf("empty prompt reached flow control: observed=%v enqueue_time=%d", routerRequests[reqs[1].ID], reqs[1].GatewayEnqueueTime)
	}
	for _, req := range []*sim.Request{reqs[0], reqs[2]} {
		if !routerRequests[req.ID] {
			t.Errorf("valid request %s did not reach the router", req.ID)
		}
		if req.GatewayEnqueueTime != 100 {
			t.Errorf("valid request %s entered flow control at %d, want 100", req.ID, req.GatewayEnqueueTime)
		}
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
