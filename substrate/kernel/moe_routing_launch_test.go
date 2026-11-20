package latencykernel

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// Whether expert selection is a launch of its own depends on the experts backend the engine
// resolves: a modular backend runs router.select_experts before the experts, a monolithic one
// takes the router logits (vllm/model_executor/layers/fused_moe/runner/moe_runner.py:620-643
// @4ba787ec). The plan states it (MoEPlan.Routing); the same graph then launches one more kernel
// in each routed MoE layer under a modular resolution than under a monolithic one.
func TestRoutingLaunchSeparatesAModularFromAMonolithicResolution(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := withoutDrafts(quantCase(t, d, "glm-5.3", "h200"))
	layers := moeLayers(base.Model)
	if layers == 0 {
		t.Fatal("no routed layer in the graph")
	}
	routing := func(r plan.MoERouting) *Kernel {
		return mustKernel(t, withPlan(base, func(l *plan.Lowered) { l.Plan.MoE.Routing = r }))
	}
	b := oneDecode()
	notStated, separate, fused := routing(plan.RoutingNotStated), routing(plan.RoutingSeparate), routing(plan.RoutingFused)
	if got, want := separate.stepKernels(b)-fused.stepKernels(b), layers; got != want {
		t.Errorf("a modular resolution launches %d more kernels than a monolithic one, want %d (one a routed layer)", got, want)
	}
	if notStated.stepKernels(b) != fused.stepKernels(b) {
		t.Errorf("an unstated routing counts %d kernels, a fused one %d: neither counts a selection launch",
			notStated.stepKernels(b), fused.stepKernels(b))
	}
	host := separate.StepTime(b).PerResource[kernel.ResourceHost] - fused.StepTime(b).PerResource[kernel.ResourceHost]
	want := time.Duration(layers) * separate.launchPerKernel
	if d := host - want; d < -time.Microsecond || d > time.Microsecond {
		t.Errorf("host time rose by %v, want %d launches at %v each, %v", host, layers, separate.launchPerKernel, want)
	}
}

// A routing word the plan does not know is refused, not read as unstated.
func TestAnUnknownRoutingWordIsRefused(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	in := withPlan(withoutDrafts(quantCase(t, d, "glm-5.3", "h200")),
		func(l *plan.Lowered) { l.Plan.MoE.Routing = "inline" })
	if _, err := New(in); err == nil {
		t.Fatal("a plan whose MoE.Routing is \"inline\" was accepted")
	}
}
