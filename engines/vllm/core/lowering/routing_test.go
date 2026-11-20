package lowering

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	"github.com/inference-sim/blis-vllm/core/vllmsettings"
)

// Whether expert selection is a launch of its own depends on the experts class vLLM v0.33.0
// resolves (fused_moe/runner/moe_runner.py:620-643@4ba787ec): the lowering states it, so the cost
// model counts the launch only where it happens (inference-sim/blis-vllm#194).
func TestTheLoweringStatesWhetherExpertSelectionIsItsOwnLaunch(t *testing.T) {
	h200 := where{"deepseek-v3", "h200", "tp: 8, pp: 1, dp: 1"}
	b200 := where{"deepseek-v3", "b200", "tp: 8, pp: 1, dp: 1"}
	for _, c := range []struct {
		name string
		at   where
		s    vllmsettings.Settings
		want plan.MoERouting
		note bool
	}{
		// Hopper has no TRT-LLM experts: Triton's or FlashInfer's CUTLASS, both modular.
		{"fp8 block on hopper", h200, vllmsettings.Settings{}, plan.RoutingSeparate, false},
		{"bf16 on hopper", where{"qwen3-30b-a3b", "h200", "tp: 8, pp: 1, dp: 1"}, vllmsettings.Settings{}, plan.RoutingSeparate, false},
		// Blackwell: the TRT-LLM monolithic experts take the router logits.
		{"fp8 block on blackwell", b200, vllmsettings.Settings{}, plan.RoutingFused, true},
		{"fp8 block on blackwell, flashinfer_trtllm", b200, vllmsettings.Settings{MoEBackend: "flashinfer_trtllm"}, plan.RoutingFused, true},
		// A stated modular backend runs the selection first on either part.
		{"fp8 block on blackwell, triton", b200, vllmsettings.Settings{MoEBackend: "triton"}, plan.RoutingSeparate, false},
		{"fp8 block on hopper, deep_gemm", h200, vllmsettings.Settings{MoEBackend: "deep_gemm"}, plan.RoutingSeparate, false},
		// No routed experts: nothing to select.
		{"dense", dense, vllmsettings.Settings{}, plan.RoutingNotStated, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			l := lowered(t, c.at, c.s)
			if got := l.Plan.MoE.Routing; got != c.want {
				t.Errorf("MoE.Routing = %q, want %q", got, c.want)
			}
			if _, ok := noted(l, "moe_routing"); ok != c.note {
				t.Errorf("moe_routing note recorded = %v, want %v", ok, c.note)
			}
		})
	}
}

// The same model resolves modular on one part and monolithic on another, so the plan states
// different routing for one graph.
func TestOneGraphResolvesModularOnHopperAndMonolithicOnBlackwell(t *testing.T) {
	hop := lowered(t, where{"deepseek-v3", "h200", "tp: 8, pp: 1, dp: 1"}, vllmsettings.Settings{})
	bw := lowered(t, where{"deepseek-v3", "b200", "tp: 8, pp: 1, dp: 1"}, vllmsettings.Settings{})
	if hop.Plan.MoE.Routing == bw.Plan.MoE.Routing {
		t.Fatalf("hopper and blackwell both resolve routing %q for deepseek-v3", hop.Plan.MoE.Routing)
	}
}
