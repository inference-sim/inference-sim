// Original to blis-sglang.

package lower

import (
	"errors"
	"slices"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/contracts/units"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

// resolvedArgs are the ServerArgs SGLang resolves for the DeepSeek-R1 pool of the h200 recipes, a copy a
// test may change before it lowers them.
func resolvedArgs(t *testing.T) config.ServerArgs {
	t.Helper()
	res, err := Resolve(PoolInput{Deployment: pool(t, 8, "mem_fraction_static: 0.82\n"), Model: graph(t, "deepseek-r1-0528"), Chip: h200(t)})
	if err != nil {
		t.Fatal(err)
	}
	return res.Args
}

func lowerArgs(t *testing.T, mutate func(*config.ServerArgs)) (*plan.Lowered, error) {
	t.Helper()
	a := resolvedArgs(t)
	if mutate != nil {
		mutate(&a)
	}
	return Lower(&a, graph(t, "deepseek-r1-0528"), h200(t))
}

func mustLower(t *testing.T, mutate func(*config.ServerArgs)) *plan.Lowered {
	t.Helper()
	l, err := lowerArgs(t, mutate)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func refused(err error) bool {
	var nl *NotLowered
	return errors.As(err, &nl)
}

// The pool index names a pool of the deployment: a negative index and the index one past the last pool
// are both refused, and the last pool is not.
func TestAPoolIndexOutsideTheDeploymentIsRefused(t *testing.T) {
	d := pool(t, 8, "mem_fraction_static: 0.82\n")
	for _, i := range []int{-1, 1} {
		if _, err := Args(PoolInput{Deployment: d, PoolIndex: i}); err == nil {
			t.Errorf("pool index %d of a one-pool deployment was accepted", i)
		}
	}
	if _, err := Args(PoolInput{Deployment: d, PoolIndex: 0}); err != nil {
		t.Errorf("pool index 0 refused: %v", err)
	}
}

// Parallelism beyond tensor and data is refused one dimension at a time; a width of one in a dimension
// is no parallelism and passes.
func TestEachParallelismBeyondTensorAndDataIsRefusedOnItsOwn(t *testing.T) {
	for name, mutate := range map[string]func(*deployment.Parallelism){
		"pp":  func(p *deployment.Parallelism) { p.PP = 2 },
		"dcp": func(p *deployment.Parallelism) { p.DCP = 2 },
		"pcp": func(p *deployment.Parallelism) { p.PCP = 2 },
	} {
		d := pool(t, 8, "mem_fraction_static: 0.82\n")
		mutate(&d.Pools[0].Parallel)
		_, err := Args(PoolInput{Deployment: d})
		if !refused(err) {
			t.Errorf("%s: want a NotLowered refusal, got %v", name, err)
		}
	}
	d := pool(t, 8, "mem_fraction_static: 0.82\n")
	d.Pools[0].Parallel.DCP, d.Pools[0].Parallel.PCP = 1, 1
	if _, err := Args(PoolInput{Deployment: d}); err != nil {
		t.Errorf("a context-parallel width of one is no parallelism: %v", err)
	}
}

// The engine block may restate the pool's width; a width that is not the pool's, or not a whole number,
// is refused.
func TestAStatedWidthMustBeAWholeNumberEqualToThePools(t *testing.T) {
	ok := pool(t, 8, "tp_size: 8.0\n")
	if _, err := Args(PoolInput{Deployment: ok}); err != nil {
		t.Errorf("tp_size 8.0 states the pool's width: %v", err)
	}
	for _, block := range []string{"tp_size: 8.5\n", "tp_size: 4\n"} {
		if _, err := Args(PoolInput{Deployment: pool(t, 8, block)}); err == nil {
			t.Errorf("%q was accepted for a pool of width 8", block)
		}
	}
}

// The lowering needs both the model graph and the chip.
func TestResolveNeedsTheModelAndTheChip(t *testing.T) {
	d := pool(t, 8, "mem_fraction_static: 0.82\n")
	if _, err := Resolve(PoolInput{Deployment: d, Chip: h200(t)}); err == nil {
		t.Error("a pool with no model graph was resolved")
	}
	if _, err := Resolve(PoolInput{Deployment: d, Model: graph(t, "deepseek-r1-0528")}); err == nil {
		t.Error("a pool with no chip was resolved")
	}
}

// A model with no grouped GEMM has no routed experts: the plan states no experts kernel. A graph with a
// recurrent update is a hybrid pool and is refused naming its issue.
func TestTheGraphDecidesWhetherTheExpertsAndTheRecurrentLayersAreLowered(t *testing.T) {
	a := resolvedArgs(t)
	l, err := Lower(&a, graph(t, "deepseek-r1-0528"), h200(t))
	if err != nil || l.Plan.MoE.Experts == (plan.KernelImpl{}) {
		t.Fatalf("a MoE graph must state its experts kernel: %v %+v", err, l)
	}
	dense := graph(t, "deepseek-r1-0528")
	for i := range dense.LayerKinds {
		// Dropping nodes would break the edge indices, so replace them by an op the lowering ignores.
		for j, n := range dense.LayerKinds[i].Nodes {
			if n.Op == model.OpGroupedGEMM {
				dense.LayerKinds[i].Nodes[j].Op = model.OpGEMM
			}
		}
	}
	l, err = Lower(&a, dense, h200(t))
	if err != nil {
		t.Fatal(err)
	}
	if l.Plan.MoE.Experts != (plan.KernelImpl{}) {
		t.Errorf("a graph with no grouped GEMM states experts %+v", l.Plan.MoE.Experts)
	}
	hybrid := graph(t, "deepseek-r1-0528")
	hybrid.Head = append(hybrid.Head, model.Node{Op: model.OpRecurrentUpdate})
	if _, err := Lower(&a, hybrid, h200(t)); !refused(err) {
		t.Errorf("a recurrent layer must be refused naming #2366, got %v", err)
	}
}

func TestTwoBatchAndSingleBatchOverlapAreRefusedOnTheirOwn(t *testing.T) {
	for name, mutate := range map[string]func(*config.ServerArgs){
		"tbo": func(a *config.ServerArgs) { a.EnableTwoBatchOverlap = true },
		"sbo": func(a *config.ServerArgs) { a.EnableSingleBatchOverlap = true },
	} {
		if _, err := lowerArgs(t, mutate); !refused(err) {
			t.Errorf("%s: want a NotLowered refusal, got %v", name, err)
		}
	}
}

// --quantization fp8 is the checkpoint's own format on an fp8 checkpoint; any other method, or fp8 on a
// checkpoint that is not fp8, changes the weights and is refused.
func TestQuantizationIsLoweredOnlyAsTheCheckpointsOwnFormat(t *testing.T) {
	fp8, other := "fp8", "w8a8_int8"
	if l := mustLower(t, func(a *config.ServerArgs) { a.Quantization = &fp8 }); l.Plan.WeightFormat != "fp8" {
		t.Errorf("fp8 on an fp8 checkpoint served as %q", l.Plan.WeightFormat)
	}
	if _, err := lowerArgs(t, func(a *config.ServerArgs) { a.Quantization = &other }); !refused(err) {
		t.Errorf("another method on an fp8 checkpoint must be refused, got %v", err)
	}
	a := resolvedArgs(t)
	a.Quantization = &fp8
	bf16 := graph(t, "deepseek-r1-0528")
	bf16.Global.WeightDType = model.DTypeBF16
	if _, err := Lower(&a, bf16, h200(t)); !refused(err) {
		t.Errorf("fp8 on a bf16 checkpoint must be refused, got %v", err)
	}
}

func TestThePageAndTheMemoryFractionMustResolve(t *testing.T) {
	if _, err := lowerArgs(t, func(a *config.ServerArgs) { a.PageSize = nil }); err == nil {
		t.Error("an unresolved page_size was lowered")
	}
	if _, err := lowerArgs(t, func(a *config.ServerArgs) { a.MemFractionStatic = nil }); err == nil {
		t.Error("an unresolved mem_fraction_static was lowered")
	}
}

// A step schedules the prefill chunk when chunked prefill is on (a positive chunk size, no more than the
// prefill budget), else the prefill budget; a decode-only limit of requests may exceed it.
func TestTheTokensAStepSchedulesAreTheChunkOrTheBudget(t *testing.T) {
	zero, chunk, eight := 0, 300, 8
	for _, tc := range []struct {
		name   string
		chunk  *int
		budget int
		want   units.Tokens
	}{
		{"no chunking: the budget", nil, 16384, 16384},
		{"a zero chunk size is chunking off", &zero, 16384, 16384},
		{"a chunk below the budget", &chunk, 16384, 300},
		{"a chunk and a zero budget: the chunk", &chunk, 0, 300},
	} {
		l := mustLower(t, func(a *config.ServerArgs) {
			a.ChunkedPrefillSize, a.MaxPrefillTokens, a.MaxRunningRequests = tc.chunk, tc.budget, &eight
		})
		if l.Plan.Batch.Tokens != tc.want {
			t.Errorf("%s: a step schedules %d tokens, want %d", tc.name, l.Plan.Batch.Tokens, tc.want)
		}
	}
}

func TestAZeroRequestLimitIsRefused(t *testing.T) {
	zero := 0
	if _, err := lowerArgs(t, func(a *config.ServerArgs) { a.MaxRunningRequests = &zero }); err == nil {
		t.Error("max_running_requests 0 was lowered")
	}
}

func cudaGraphs(decode, prefill string, dbs, pbs []int) func(*config.ServerArgs) {
	return func(a *config.ServerArgs) {
		a.CudaGraphConfig.Decode.Backend, a.CudaGraphConfig.Decode.Bs = decode, dbs
		a.CudaGraphConfig.Prefill.Backend, a.CudaGraphConfig.Prefill.Bs = prefill, pbs
		n := 64
		a.MaxRunningRequests = &n
	}
}

// The pairs of decode and prefill graph backends the cost model prices are lowered to the plan's words;
// every other pair, and a piecewise decode graph, is refused by name.
func TestCudaGraphBackendsAreLoweredToThePlansWords(t *testing.T) {
	type want struct {
		decode, other plan.Launch
		buckets       []int
		ceiling       units.Tokens
	}
	for _, tc := range []struct {
		name     string
		mutate   func(*config.ServerArgs)
		want     want
		refusal  bool
		anyError bool
	}{
		{"full decode, no prefill graph", cudaGraphs("full", "none", []int{1, 8, 64, 128}, nil), want{plan.LaunchGraph, plan.LaunchEager, []int{1, 8, 64}, 64}, false, false},
		{"full decode, breakable prefill", cudaGraphs("full", "breakable", []int{1, 8}, []int{256, 512}), want{plan.LaunchGraph, plan.LaunchPiecewise, []int{1, 8}, 512}, false, false},
		{"full decode, full prefill", cudaGraphs("full", "full", []int{1, 8}, []int{16}), want{plan.LaunchGraph, plan.LaunchGraph, []int{1, 8}, 16}, false, false},
		{"no graphs", cudaGraphs("none", "none", nil, nil), want{plan.LaunchEager, plan.LaunchEager, nil, 0}, false, false},
		{"no decode graph, a prefill graph", cudaGraphs("none", "full", nil, []int{16}), want{}, true, false},
		{"no decode graph, a breakable prefill graph", cudaGraphs("none", "breakable", nil, []int{16}), want{}, true, false},
		{"a breakable decode graph", cudaGraphs("breakable", "none", []int{1}, nil), want{}, true, false},
		{"a decode graph that captures no size", cudaGraphs("full", "none", []int{128}, nil), want{}, false, true},
		{"a prefill graph that captures no size", cudaGraphs("full", "full", []int{1}, nil), want{plan.LaunchGraph, plan.LaunchGraph, []int{1}, 1}, false, false},
	} {
		l, err := lowerArgs(t, tc.mutate)
		switch {
		case tc.refusal:
			if !refused(err) {
				t.Errorf("%s: want a NotLowered refusal, got %v", tc.name, err)
			}
		case tc.anyError:
			if err == nil {
				t.Errorf("%s: want an error", tc.name)
			}
		default:
			if err != nil {
				t.Errorf("%s: %v", tc.name, err)
				continue
			}
			g := l.Plan.Graphs
			if g.Decode != tc.want.decode || g.Other != tc.want.other || g.Ceiling != tc.want.ceiling || !slices.Equal(g.Buckets, tc.want.buckets) {
				t.Errorf("%s: graphs %+v, want decode %v other %v buckets %v ceiling %d", tc.name, g, tc.want.decode, tc.want.other, tc.want.buckets, tc.want.ceiling)
			}
		}
	}

}

// The custom all-reduce handles a width of 2, 4, 6 or 8 unless it is disabled; the allreduce fusion
// removes two launches per layer only for tensor-parallel pools without attention data parallelism.
func TestTheAllReducePathsFollowTheWidthAndTheAttentionParallelism(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*config.ServerArgs)
		sm     bool
		fusion bool
	}{
		{"tp 8", nil, true, true},
		{"tp 1", func(a *config.ServerArgs) { a.TpSize = 1 }, false, false},
		{"tp 3 is no width of the custom all-reduce", func(a *config.ServerArgs) { a.TpSize = 3 }, false, true},
		{"the custom all-reduce disabled", func(a *config.ServerArgs) { a.DisableCustomAllReduce = true }, false, true},
		{"attention data parallelism", func(a *config.ServerArgs) { a.AttnDpSize = 2 }, true, false},
	} {
		l := mustLower(t, tc.mutate)
		if l.Reported.SMAllReduce != tc.sm {
			t.Errorf("%s: SMAllReduce %v, want %v", tc.name, l.Reported.SMAllReduce, tc.sm)
		}
		if got := len(l.Plan.Launches) > 0; got != tc.fusion {
			t.Errorf("%s: fused launches %v, want %v", tc.name, got, tc.fusion)
		}
	}
}
