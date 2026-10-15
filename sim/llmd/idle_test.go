package llmd

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// TestFastForwardIdle_IsExact runs workloads with long idle gaps between bursts twice, once with
// the idle fast-forward and once ticking every router loop through every gap, and requires the
// same routing, dispatch times and latencies.
func TestFastForwardIdle_IsExact(t *testing.T) {
	burstsWithGaps := func() []*sim.Request {
		var reqs []*sim.Request
		for b, start := range []int64{0, 30_000_000, 95_000_000} { // bursts 30 s and 65 s apart
			for _, r := range testRequests(40, int64(3+b), 400, 32) {
				r.ID = fmt.Sprintf("%s_b%d", r.ID, b)
				r.ArrivalTime += start
				reqs = append(reqs, r)
			}
		}
		return reqs
	}
	for name, cfg := range map[string]string{"default": "", "flow-control": globalStrictConfig} {
		t.Run(name, func(t *testing.T) {
			run := func(neverSkip bool) (map[string]string, map[string]float64, map[string]float64) {
				reqs := burstsWithGaps()
				dep := testDeployment(3)
				dep.BatchConfig = sim.NewBatchConfig(4, 2048, 0)
				cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
				flags := DefaultEPPFlags()
				if cfg != "" {
					flags = append(flags, "--allow-experimental-plugins", "--config-text="+cfg)
				}
				gw := New(Options{Model: "test-model", EPPFlags: flags})
				gw.neverSkipIdle = neverSkip
				cs.SetGateway(gw)
				if err := cs.Run(); err != nil {
					t.Fatalf("Run: %v", err)
				}
				placed := map[string]string{}
				for _, r := range reqs {
					placed[r.ID] = fmt.Sprintf("%s@%d", r.AssignedInstance, r.GatewayDispatchTime)
				}
				m := cs.AggregatedMetrics()
				return placed, m.RequestTTFTs, m.RequestE2Es
			}
			p1, ttft1, e2e1 := run(false)
			p2, ttft2, e2e2 := run(true)
			if !reflect.DeepEqual(p1, p2) || !reflect.DeepEqual(ttft1, ttft2) || !reflect.DeepEqual(e2e1, e2e2) {
				t.Fatal("the idle fast-forward changed routing or latency")
			}
			if len(ttft1) != 120 {
				t.Fatalf("%d of 120 requests completed", len(ttft1))
			}
		})
	}
}

// TestFastForwardIdle_IsExactWithStatefulUsageLimit: soft-reflective-ceiling-policy keeps a tick
// counter that advances on dispatch cycles while some band is in its alternating regime. With two
// priority bands and saturating bursts separated by long idle gaps, the fast-forward must still
// give the routing, dispatch times and latencies of ticking through every gap: an idle pool is
// below the alternating regime, so the skipped cycles leave the counter unchanged.
func TestFastForwardIdle_IsExactWithStatefulUsageLimit(t *testing.T) {
	for name, extra := range map[string][]string{
		"fresh metrics": nil,
		// Metrics older than the utilization detector's 200 ms staleness count as saturated, so
		// with staggered scrapes an idle pool sits in the policy's alternating regime (review
		// 6200849987).
		"stale while idle": {"--refresh-metrics-interval=500ms"},
	} {
		t.Run(name, func(t *testing.T) { checkStatefulUsageLimitExact(t, extra) })
	}
}

func checkStatefulUsageLimitExact(t *testing.T, extraFlags []string) {
	run := func(neverSkip bool) (map[string]string, map[string]float64) {
		var reqs []*sim.Request
		for b, start := range []int64{0, 40_000_000, 130_000_000} {
			for i, r := range testRequests(300, int64(7+b), 1000, 32) { // 300 ms bursts, across several scrapes
				r.ID = fmt.Sprintf("%s_b%d", r.ID, b)
				r.ArrivalTime += start
				r.SLOClass = map[bool]string{true: "critical", false: "sheddable"}[i%2 == 0]
				reqs = append(reqs, r)
			}
		}
		dep := testDeployment(2)
		dep.BatchConfig = sim.NewBatchConfig(4, 2048, 0)
		cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
		gw := New(Options{
			Model:        "test-model",
			EPPFlags:     append([]string{"--pool-name=vllm", "--allow-experimental-plugins", "--config-text=" + softCeilingConfig}, extraFlags...),
			Manifests:    []byte(poolManifests),
			PodLabels:    func(cluster.InstanceID, cluster.PoolRole) map[string]string { return map[string]string{"app": "vllm"} },
			ScrapeJitter: 400 * time.Millisecond,
		})
		gw.neverSkipIdle = neverSkip
		cs.SetGateway(gw)
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if gw.skipIdle {
			t.Fatal("the idle fast-forward stayed on with soft-reflective-ceiling-policy")
		}
		placed := map[string]string{}
		held := 0
		for _, r := range reqs {
			placed[r.ID] = fmt.Sprintf("%s@%d", r.AssignedInstance, r.GatewayDispatchTime)
			if r.GatewayDispatchTime > r.ArrivalTime {
				held++
			}
		}
		if held == 0 {
			t.Fatal("no request waited in flow control; the bursts do not exercise the usage limit")
		}
		return placed, cs.AggregatedMetrics().RequestTTFTs
	}
	p1, ttft1 := run(false)
	p2, ttft2 := run(true)
	if !reflect.DeepEqual(p1, p2) || !reflect.DeepEqual(ttft1, ttft2) {
		t.Fatal("the idle fast-forward changed routing or latency with a stateful usage-limit policy and two bands")
	}
}
