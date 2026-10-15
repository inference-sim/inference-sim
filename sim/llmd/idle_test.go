package llmd

import (
	"fmt"
	"reflect"
	"testing"

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
	for name, cfg := range map[string]string{"default": "", "flow-control": softCeilingConfig} {
		t.Run(name, func(t *testing.T) {
			run := func(neverSkip bool) (map[string]string, map[string]float64, map[string]float64) {
				reqs := burstsWithGaps()
				dep := testDeployment(3)
				dep.BatchConfig = sim.NewBatchConfig(4, 2048, 0)
				cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
				flags := append([]string{}, DefaultEPPFlags...)
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
