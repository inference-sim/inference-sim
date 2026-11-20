package latencykernel

import (
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// SUB-6, SUB-11: the launches the engine's fused kernels remove (plan.PoolPlan.Launches) are not
// charged. Muse Spark on GB200 at TP8 with expert parallelism is the layout blis-vllm#192 resolves
// for vLLM v0.33.0: three fewer launches a layer at 1-128 tokens and two fewer at 129-256. A
// FULL-graph decode step in a range then costs the stated launches times the per-launch term
// less, a step past every range costs the same, and a pool that states no change prices as before.
func TestTheLaunchesTheEnginesFusedKernelsRemoveAreNotCharged(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	fusions := []plan.LaunchChange{{FromTokens: 1, ToTokens: 128, PerLayer: -3}, {FromTokens: 129, ToTokens: 256, PerLayer: -2}}
	open := func(launches []plan.LaunchChange, decode plan.Launch) *Kernel {
		return mustKernel(t, planCase(t, d, "muse-spark-open-816b-a42b-nvfp4", "gb200-nvl72", 2,
			"tp: 8, pp: 1, dp: 1, enable_expert_parallel: true", func(l *plan.Lowered) {
				l.Plan.Launches = launches
				l.Plan.Batch.Requests = 512
				l.Plan.Graphs.Ceiling = 512
				l.Plan.Graphs.Buckets = []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
				l.Plan.Graphs.Decode = decode
				if decode == plan.LaunchEager {
					l.Plan.Graphs.Other = plan.LaunchEager
				}
			}))
	}
	decodes := func(n int) kernel.Batch {
		b := kernel.Batch{DecodeThreshold: 1}
		for range n {
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 2048, PromptLen: 1024})
		}
		return b
	}
	graph, fused := open(nil, plan.LaunchGraph), open(fusions, plan.LaunchGraph)
	layers := fused.plan.TotalLayers
	for _, c := range []struct {
		requests, perLayer int
	}{{1, -3}, {64, -3}, {128, -3}, {129, -2}, {256, -2}, {257, 0}, {512, 0}} {
		before, after := graph.StepTime(decodes(c.requests)), fused.StepTime(decodes(c.requests))
		want := time.Duration(c.perLayer*layers) * fused.launchPerKernel
		got := after.NoOverlap - before.NoOverlap
		t.Logf("%d decodes: %v -> %v, %v (%d launches)", c.requests, before.NoOverlap, after.NoOverlap, got, c.perLayer*layers)
		if diff := got - want; diff < -time.Nanosecond || diff > time.Nanosecond {
			t.Errorf("%d decodes: the step changes by %v, want %v (%d launches at %v)", c.requests, got,
				want, c.perLayer*layers, fused.launchPerKernel)
		}
	}
	// An eager step also launches each removed kernel from the host, and saves that too.
	eager, eagerFused := open(nil, plan.LaunchEager), open(fusions, plan.LaunchEager)
	b := decodes(64)
	if saved, graphSaved := eager.StepTime(b).NoOverlap-eagerFused.StepTime(b).NoOverlap,
		graph.StepTime(b).NoOverlap-fused.StepTime(b).NoOverlap; saved <= graphSaved {
		t.Errorf("an eager step saves %v from the fused kernels, a graph step %v; eager must save more", saved, graphSaved)
	}
}

// A pipeline stage is charged its own layers' share of the launches the fusions remove.
func TestAPipelineStageSavesItsLayersShareOfTheFusedLaunches(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	open := func(launches []plan.LaunchChange) *Kernel {
		return mustKernel(t, planCase(t, d, "muse-spark-open-816b-a42b-nvfp4", "gb200-nvl72", 2,
			"tp: 4, pp: 2, dp: 1, enable_expert_parallel: true", func(l *plan.Lowered) {
				l.Plan.Launches = launches
				l.Plan.MemoryFraction = 0.9
			}))
	}
	b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 2048, PromptLen: 1024}}}
	base, fused := open(nil).StepTime(b), open([]plan.LaunchChange{{FromTokens: 1, ToTokens: 128, PerLayer: -2}}).StepTime(b)
	if fused.NoOverlap >= base.NoOverlap {
		t.Errorf("a two-stage step costs %v with fused kernels, %v without; it must cost less", fused.NoOverlap, base.NoOverlap)
	}
}

// A plan that removes more launches than a layer makes is refused by name.
func TestAPlanThatFusesMoreLaunchesThanALayerHasIsRefused(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	in := planCase(t, d, "llama-3.1-70b-instruct", "h100", 1, "tp: 4, pp: 1, dp: 1", func(l *plan.Lowered) {
		l.Plan.Launches = []plan.LaunchChange{{FromTokens: 1, ToTokens: 8, PerLayer: -1000}}
	})
	if _, err := New(in); err == nil || !strings.Contains(err.Error(), "Plan.Launches") {
		t.Errorf("err %v, want a refusal naming Plan.Launches", err)
	}
}

// The other layouts blis-vllm#192 resolves: H200 at TP8 with expert parallelism launches two
// fewer kernels a layer, and GB200 at DP8 with expert parallelism one fewer. A FULL-graph decode
// step saves that many launches at the per-launch term.
func TestEachFusedLayoutSavesItsStatedLaunches(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		chip     string
		nodes    int
		parallel string
		perLayer int
	}{
		{"h200", 1, "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true", -2},
		{"h100", 2, "tp: 16, pp: 1, dp: 1, enable_expert_parallel: true", -2},
		{"gb200-nvl72", 2, "tp: 1, pp: 1, dp: 8, enable_expert_parallel: true", -1},
	} {
		open := func(launches []plan.LaunchChange) *Kernel {
			return mustKernel(t, planCase(t, d, "muse-spark-open-816b-a42b-nvfp4", c.chip, c.nodes, c.parallel,
				func(l *plan.Lowered) { l.Plan.Launches = launches }))
		}
		b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 2048, PromptLen: 1024}}}
		base, fused := open(nil), open([]plan.LaunchChange{{FromTokens: 1, ToTokens: 256, PerLayer: c.perLayer}})
		got := fused.StepTime(b).NoOverlap - base.StepTime(b).NoOverlap
		want := time.Duration(c.perLayer*fused.plan.TotalLayers) * fused.launchPerKernel
		t.Logf("%s %s: %v (%d launches)", c.chip, c.parallel, got, c.perLayer*fused.plan.TotalLayers)
		if diff := got - want; diff < -time.Nanosecond || diff > time.Nanosecond {
			t.Errorf("%s %s: the step changes by %v, want %v", c.chip, c.parallel, got, want)
		}
	}
}
