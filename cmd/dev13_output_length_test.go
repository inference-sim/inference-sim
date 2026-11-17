package cmd

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/llmd"
)

// DEV-13 (#2107 I2): the output length the workload sampled for a request is read only by the
// engine's stop check, by the trace writer and by replay. The router, which decides where a
// request goes, sees the client's maximum token count and the prompt, as a real router does.
//
// So a burst whose requests differ only in how long their sampled outputs are (and what those
// outputs say) must be placed on the same instances, with the same prefill figures. A routing
// decision that peeked at the sampled length would send some request elsewhere.
func TestDEV13_PlacementDoesNotDependOnTheSampledOutput(t *testing.T) {
	d := newKernelDeployment(t, 7, nil)
	build := func(outLen func(i int) int, tail sim.TokenID) []*sim.Request {
		var reqs []*sim.Request
		for i := 0; i < 24; i++ {
			in := make([]sim.TokenID, 64+(i%4)*32)
			for j := range in {
				in[j] = sim.TokenID(100 + (i%3)*1000 + j) // three prompt families, so cache-aware routing has a reason to choose
			}
			out := make([]sim.TokenID, outLen(i))
			for j := range out {
				out[j] = tail + sim.TokenID(j)
			}
			reqs = append(reqs, &sim.Request{
				ID: fmt.Sprintf("r%02d", i), ArrivalTime: int64(i) * 50, InputTokens: in,
				OutputTokens: out, MaxOutputLen: 200, State: sim.StateQueued,
			})
		}
		return reqs
	}
	placed := func(reqs []*sim.Request) map[string][3]any {
		cfg := deploymentOf(d.Config)
		cfg.NumInstances = 3
		cs := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(reqs), nil)
		cs.SetGateway(newGateway(routerInputs{EPPFlags: append([]string(nil), llmd.DefaultEPPFlags()...)}, cfg.Model))
		if err := cs.Run(); err != nil {
			t.Fatal(err)
		}
		got := map[string][3]any{}
		for id, rm := range cs.AggregatedMetrics().Requests {
			got[id] = [3]any{rm.HandledBy, rm.NumPrefillTokens, rm.ArrivedAt}
		}
		return got
	}
	a := placed(build(func(int) int { return 8 }, 5000))
	b := placed(build(func(i int) int { return 8 + 13*i }, 9000))
	if len(a) != 24 {
		t.Fatalf("%d of 24 requests completed", len(a))
	}
	used := map[any]bool{}
	for id, pa := range a {
		used[pa[0]] = true
		if pb := b[id]; !reflect.DeepEqual(pa, pb) {
			t.Errorf("%s: placement changed with the sampled output: short outputs %v, long outputs %v", id, pa, pb)
		}
	}
	if len(used) < 2 {
		t.Errorf("every request went to one instance (%v): the check would pass for a router that ignores everything", used)
	}
}
