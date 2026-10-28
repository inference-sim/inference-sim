// Moved from blis-latency-kernel/routing_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/routing"
)

func routingValue(tok, exp float64) *routing.Resolution {
	return &routing.Resolution{Layer: routing.LayerScenario, Source: "test",
		Value: &routing.Value{TokenImbalance: tok, ExpertImbalance: &exp, Source: "test"}}
}

func withRouting(t *testing.T, scenario string, r *routing.Resolution) *Kernel {
	t.Helper()
	in := fixtureInputs(t, scenario)
	in.Routing = r
	k, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// The busiest expert-parallel rank gates every MoE layer's combine, so a resolved token
// imbalance multiplies the routed compute and nothing else, and the expert imbalance
// multiplies the routed weight read and nothing else.
func TestRoutingImbalanceScalesTheBusiestRanksRoutedWork(t *testing.T) {
	const sc = "minimax-m25-h200-ep8.yaml"
	even := fixture(t, sc)
	tokOnly := withRouting(t, sc, routingValue(1.5, 1))
	expOnly := withRouting(t, sc, routingValue(1, 0.5))
	b := decodeBatch(256, 1, 8192)

	sm := func(k *Kernel) float64 { return k.StepTime(b).PerResource[kernel.ResourceSM].Seconds() }
	hbm := func(k *Kernel) float64 { return k.StepTime(b).PerResource[kernel.ResourceHBM].Seconds() }

	if sm(tokOnly) <= sm(even)*1.05 {
		t.Errorf("a 1.5x busiest-rank token load left SM at %.4g against %.4g ms", sm(tokOnly)*1e3, sm(even)*1e3)
	}
	if hbm(tokOnly) != hbm(even) {
		t.Errorf("token imbalance moved the HBM term: %v against %v", hbm(tokOnly), hbm(even))
	}
	if hbm(expOnly) >= hbm(even) {
		t.Errorf("halving the busiest rank's distinct experts did not cut the weight read: %v against %v",
			hbm(expOnly), hbm(even))
	}
	if sm(expOnly) != sm(even) {
		t.Errorf("expert imbalance moved the SM term")
	}
}
