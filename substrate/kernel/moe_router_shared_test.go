package latencykernel

import (
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// The MoE router and the shared expert's MLP are launches vLLM v0.33.0 runs on NVIDIA parts
// (inference-sim/blis-substrate#228): the gate GEMM, and the shared expert's gate/up, activation
// and down. The catalog states them as GEMM and Elementwise nodes, and the kernel counts and
// prices those generically: a decode step of a shared-expert MoE layer pays one launch for each
// and the router's weight read, where the graph that folded the shared expert into the routed
// node and stated no router paid neither.

// foldedMoE is the graph as the deriver stated it before #228: no router, and the shared
// experts folded into the routed GroupedGEMM.
func foldedMoE(t *testing.T, g *model.Graph) *model.Graph {
	t.Helper()
	c := *g
	c.LayerKinds = append([]model.LayerKind(nil), g.LayerKinds...)
	for i := range c.LayerKinds {
		var kept []model.Node
		shared := 0
		for _, n := range g.LayerKinds[i].Nodes {
			switch {
			case n.Role == "router":
			case n.Role == "shared_gate_up":
				shared = n.N / 2
			case strings.HasPrefix(n.Role, "shared_"):
			default:
				kept = append(kept, n)
			}
		}
		for j := range kept {
			if kept[j].Op == model.OpGroupedGEMM && shared > 0 {
				kept[j].SharedExperts = 1
				if shared != kept[j].N {
					kept[j].SharedIntermediateSize = shared
				}
			}
		}
		c.LayerKinds[i].Nodes, c.LayerKinds[i].Edges = kept, nil
	}
	return &c
}

// moeLayers is how many layers of the target's stack run routed experts.
func moeLayers(g *model.Graph) int {
	routed := map[string]bool{}
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Op == model.OpGroupedGEMM {
				routed[lk.ID] = true
			}
		}
	}
	count := 0
	for _, id := range g.Stack.Expand() {
		if routed[id] {
			count++
		}
	}
	return count
}

// withoutDrafts runs the scenario with no speculative decoding, so a step's launches are the
// target's alone.
func withoutDrafts(in Inputs) Inputs {
	return drafting(in, 0)
}

func oneDecode() kernel.Batch {
	return kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
		{Scheduled: 1, Computed: 4000, PromptLen: 4000}}}
}

func TestMoERouterAndSharedExpertLaunchesArePricedInADecodeStep(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		model, chip string
		perLayer    int // router, shared gate/up, its activation, shared down
	}{
		{"glm-5.3", "h200", 4},
		{"deepseek-v3", "h200", 4},
		// No shared expert: the router alone.
		{"mixtral-8x7b-v0.1", "h200", 1},
	} {
		t.Run(tc.model, func(t *testing.T) {
			base := quantCase(t, d, tc.model, tc.chip)
			g := base.Model
			old := base
			old.Model = foldedMoE(t, g)
			kNew, kOld := mustKernel(t, withoutDrafts(base)), mustKernel(t, withoutDrafts(old))
			b := oneDecode()
			layers := moeLayers(g)
			if layers == 0 {
				t.Fatal("no routed layer in the graph")
			}
			if got, want := kNew.stepKernels(b)-kOld.stepKernels(b), tc.perLayer*layers; got != want {
				t.Errorf("a decode step launches %d more kernels, want %d (%d a layer over %d MoE layers)",
					got, want, tc.perLayer, layers)
			}
			sNew, sOld := kNew.StepTime(b), kOld.StepTime(b)
			host := sNew.PerResource[kernel.ResourceHost] - sOld.PerResource[kernel.ResourceHost]
			launches := time.Duration(tc.perLayer*layers) * kNew.launchPerKernel
			if d := host - launches; d < -time.Microsecond || d > time.Microsecond {
				t.Errorf("host time rose by %v, want the %d added launches at %v each, %v",
					host, tc.perLayer*layers, kNew.launchPerKernel, launches)
			}
			if sNew.PerResource[kernel.ResourceHBM] <= sOld.PerResource[kernel.ResourceHBM] {
				t.Errorf("HBM %v -> %v: the router's weight read, whole on every rank, is not charged",
					sOld.PerResource[kernel.ResourceHBM], sNew.PerResource[kernel.ResourceHBM])
			}
			if sNew.NoOverlap-sOld.NoOverlap < launches {
				t.Errorf("the step rose by %v, less than its added launches (%v)",
					sNew.NoOverlap-sOld.NoOverlap, launches)
			}
		})
	}
}
