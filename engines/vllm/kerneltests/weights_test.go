// Moved from blis-latency-kernel/weights_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Resident weights and the rates their GEMMs run at, checked as relations between
// deployments and graphs rather than as figures.

// editNodes applies edit to every node of g's layer kinds that match.
func editNodes(g *model.Graph, match func(model.Node) bool, edit func(*model.Node)) int {
	n := 0
	for i := range g.LayerKinds {
		for j := range g.LayerKinds[i].Nodes {
			if nd := &g.LayerKinds[i].Nodes[j]; match(*nd) {
				edit(nd)
				n++
			}
		}
	}
	return n
}

func role(r string) func(model.Node) bool { return func(n model.Node) bool { return n.Role == r } }

const gib = 1 << 30

// Nemotron-3-Ultra's NVFP4 checkpoint holds about 38 GiB a rank at TP8 x EP8 by its own
// configuration -- and an internal serving campaign measured 44 GiB at load -- where an
// earlier form reported 901 GiB: its routed experts' latent width ignored, and its FP8
// shared expert charged once per local routed expert. Whatever the exact figure, the NVFP4
// checkpoint must fit the part, and must hold less than the same model's BF16 checkpoint
// served at FP8.
func TestMoEWeightsFitTheirCheckpoint(t *testing.T) {
	in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
	in.Deployment.Pools[0].Parallel.EnableExpertParallel = true
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Quantization = "nvfp4" })
	nvfp4 := build(t, in).FixedBytes().Weights

	in = fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
	in.Deployment.Pools[0].Parallel.EnableExpertParallel = true
	withModel(t, &in, "nemotron-3-ultra-550b-a55b-bf16")
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Quantization = "fp8" })
	fp8 := build(t, in).FixedBytes().Weights

	if float64(nvfp4) > 80*gib {
		t.Errorf("the NVFP4 checkpoint holds %.1f GiB a rank, more than an 80 GiB H100",
			float64(nvfp4)/gib)
	}
	if nvfp4 >= fp8 {
		t.Errorf("the NVFP4 checkpoint holds %.1f GiB a rank, not less than the FP8-served "+
			"BF16 checkpoint's %.1f", float64(nvfp4)/gib, float64(fp8)/gib)
	}
}

// latent_size narrows the routed experts: stating it can only lower their weights and work.
func TestLatentSizeNarrowsTheRoutedExperts(t *testing.T) {
	with := build(t, fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml"))
	in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
	editNodes(in.Model, role("experts"), func(n *model.Node) { n.LatentSize = 0 })
	without := build(t, in)
	if with.FixedBytes().Weights >= without.FixedBytes().Weights {
		t.Error("stating latent_size did not lower the routed experts' resident bytes")
	}
	b := decodeBatch(64, 1, 4096)
	if with.StepTime(b).NoOverlap >= without.StepTime(b).NoOverlap {
		t.Error("stating latent_size did not lower the step time")
	}
}

// A shared expert stated as its own node is dense. Its weights do not depend on how many
// routed experts the model has, and it does not set the routing's top_k: doubling the
// routed top_k makes the step dearer whatever node follows it.
func TestASharedExpertNodeIsDense(t *testing.T) {
	residentDelta := func(experts int) int64 {
		in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
		editNodes(in.Model, role("experts"), func(n *model.Node) { n.Experts = experts })
		with := build(t, in).FixedBytes().Weights
		in = fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
		editNodes(in.Model, role("experts"), func(n *model.Node) { n.Experts = experts })
		for i := range in.Model.LayerKinds {
			var kept []model.Node
			for _, n := range in.Model.LayerKinds[i].Nodes {
				if n.Role != "shared_experts" {
					kept = append(kept, n)
				}
			}
			in.Model.LayerKinds[i].Nodes = kept
			in.Model.LayerKinds[i].Edges = chain(len(kept))
		}
		return with - build(t, in).FixedBytes().Weights
	}
	if a, b := residentDelta(512), residentDelta(256); a != b || a <= 0 {
		t.Errorf("the shared expert holds %d bytes beside 512 routed experts and %d beside "+
			"256; it is one dense expert either way", a, b)
	}

	in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
	base := build(t, in)
	in = fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
	editNodes(in.Model, role("experts"), func(n *model.Node) { n.TopK *= 2 })
	doubled := build(t, in)
	b := decodeBatch(64, 1, 4096)
	if doubled.StepTime(b).NoOverlap <= base.StepTime(b).NoOverlap {
		t.Error("doubling the routed top_k did not raise the step time; a shared-expert " +
			"node is setting top_k")
	}
}

// chain is the edges of n nodes run in sequence.
func chain(n int) [][2]int {
	var e [][2]int
	for i := 1; i < n; i++ {
		e = append(e, [2]int{i - 1, i})
	}
	return e
}

// A weight-only kernel computes at its ACTIVATION format's rate: MXFP8 weights (one E8M0
// scale per 32), served by Marlin W8A16 on SM90, no faster than block-scaled FP8.
func TestWeightOnlyGEMMsComputeAtTheirActivationRate(t *testing.T) {
	prefill := kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 8192, PromptLen: 8192}}}
	sm := func(k *Kernel) float64 {
		return k.StepTime(prefill).PerResource[kernel.ResourceSM].Seconds()
	}
	// Four-bit experts against FP8 ones is TestFourBitExpertsOnHopperComputeAtBF16, with
	// expert parallelism on so neither the Marlin padding nor the tensor-parallel slice
	// enters the comparison.
	mx := sm(build(t, fixtureInputs(t, v41)))
	in := fixtureInputs(t, v41)
	in.Model.Global.WeightScale = &model.WeightScale{Elements: 128 * 128, Bytes: 4}
	block := sm(build(t, in))
	if mx <= block {
		t.Errorf("MXFP8 weights on H200 compute in %.2f ms, no slower than block-scaled "+
			"FP8's %.2f; Marlin runs them on BF16", mx*1e3, block*1e3)
	}
}

// FP32 weights run at the TF32 rate, below BF16's.
func TestFP32GEMMsAreSlowerThanBF16(t *testing.T) {
	// V4.1's FP32 GEMMs are its hc_*fn mixing GEMMs, which a measured mHC form replaces
	// (step.go). blis-registry v0.2.0 measures that form on h200, so strip it: this test
	// is about the GEMMs' own rate.
	noMHC := func() Inputs {
		return withoutEntries(fixtureInputs(t, v41), map[string]bool{
			"mhc_floor_pre": true, "mhc_rate_pre": true,
			"mhc_floor_post": true, "mhc_rate_post": true})
	}
	b := kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 8192, PromptLen: 8192}}}
	fp32 := build(t, noMHC()).StepTime(b).PerResource[kernel.ResourceSM]
	in := noMHC()
	if editNodes(in.Model, func(n model.Node) bool { return n.WeightDType == model.DTypeFP32 },
		func(n *model.Node) { n.WeightDType = model.DTypeBF16 }) == 0 {
		t.Fatal("the graph states no FP32 GEMM")
	}
	bf16 := build(t, in).StepTime(b).PerResource[kernel.ResourceSM]
	if fp32 <= bf16 {
		t.Errorf("FP32 mHC GEMMs take %v of SM, no more than at BF16 (%v)", fp32, bf16)
	}
}

// Marlin pads each tensor-parallel slice of an MXFP4 expert's intermediate width to a
// multiple of 128: at TP8 without expert parallelism a rank holds more expert bytes than
// its eighth of an expert-parallel layout's. On a part with a four-bit peak nothing is
// padded, and the two layouts hold the same expert bytes.
func TestMarlinPadsTensorParallelExperts(t *testing.T) {
	held := func(chip string, ep bool) int64 {
		in := fixtureInputs(t, v41)
		if chip != "h200" {
			in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, chip), chip
		}
		in.Deployment.Pools[0].Parallel.EnableExpertParallel = ep
		return build(t, in).FixedBytes().Weights
	}
	if tp, ep := held("h200", false), held("h200", true); tp <= ep {
		t.Errorf("on H200 a TP8 rank holds %d weight bytes, no more than an EP8 rank's %d",
			tp, ep)
	}
	if tp, ep := held("b200", false), held("b200", true); tp > ep {
		t.Errorf("on B200 a TP8 rank holds %d weight bytes, more than an EP8 rank's %d",
			tp, ep)
	}
}

// A replicated node is held whole on every rank: declaring a column-parallel GEMM
// replicated raises the resident bytes and never lowers the step time.
func TestReplicatedNodesAreNotDividedByTP(t *testing.T) {
	base := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	editNodes(in.Model, role("wq_b"), func(n *model.Node) { n.TP = model.TPReplicated })
	repl := build(t, in)
	if repl.FixedBytes().Weights <= base.FixedBytes().Weights {
		t.Error("replicating wq_b did not raise the resident bytes")
	}
	for name, b := range batchShapes() {
		if repl.StepTime(b).NoOverlap < base.StepTime(b).NoOverlap {
			t.Errorf("%s: replicating wq_b made the step cheaper", name)
		}
	}
}

// under_sequence_parallel describes only the model's own sequence parallelism. With it off
// the field changes nothing; with it on, a token-sharded node runs fewer rows, so the step
// is no dearer than if the node saw every token.
func TestTokenShardingAppliesOnlyUnderSequenceParallelism(t *testing.T) {
	clear := func(in *Inputs) {
		editNodes(in.Model, func(n model.Node) bool { return n.UnderSequenceParallel != "" },
			func(n *model.Node) { n.UnderSequenceParallel = "" })
	}
	same := func(a, b *Kernel, what string) {
		for name, bt := range batchShapes() {
			if a.StepTime(bt).NoOverlap != b.StepTime(bt).NoOverlap {
				t.Errorf("%s, %s: the step moved", what, name)
			}
		}
	}
	off := fixtureInputs(t, v41)
	offCleared := fixtureInputs(t, v41)
	clear(&offCleared)
	same(build(t, off), build(t, offCleared), "sequence parallelism off")

	sp := func(edit func(*Inputs)) *Kernel {
		in := fixtureInputs(t, v41)
		pl := &in.Deployment.Pools[0].Parallel
		pl.TP, pl.DP, pl.EnableExpertParallel = 4, 2, true
		if edit != nil {
			edit(&in)
		}
		return build(t, in)
	}
	on, onCleared := sp(nil), sp(clear)
	b := kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 4096, PromptLen: 4096}}}
	if on.StepTime(b).NoOverlap > onCleared.StepTime(b).NoOverlap {
		t.Error("under sequence parallelism, sharding nodes by token made the step dearer")
	}
}

// A collective moves what the graph says it moves: a wider stated payload costs more link
// time.
func TestCollectivesMoveTheirStatedPayload(t *testing.T) {
	b := decodeBatch(64, 1, 4096)
	base := build(t, fixtureInputs(t, v41)).StepTime(b).PerResource[kernel.ResourceNVLink]
	in := fixtureInputs(t, v41)
	if editNodes(in.Model, func(n model.Node) bool {
		return n.Op == model.OpAllGather && n.BytesPerToken > 0
	}, func(n *model.Node) { n.BytesPerToken *= 4 }) == 0 {
		t.Fatal("the graph states no collective payload")
	}
	wide := build(t, in).StepTime(b).PerResource[kernel.ResourceNVLink]
	if wide <= base {
		t.Errorf("quadrupling the engram all-gathers' payload left NVLink at %v (was %v)",
			wide, base)
	}
}

// The embedding is unquantized: a stated BF16 table and an inferred one hold the same, and
// an Engram table lives where engram_config says, host memory by default.
func TestEmbeddingAndEngramResidency(t *testing.T) {
	stated := build(t, fixtureInputs(t, v41)).FixedBytes().Weights
	in := fixtureInputs(t, v41)
	// The draft's lookup aliases the target's table, so it goes with it; an alias holds
	// nothing either way.
	in.Model.Embedding, in.Model.Speculator.Embedding = nil, nil
	if inferred := build(t, in).FixedBytes().Weights; inferred != stated {
		t.Errorf("the stated embedding holds %d bytes and the inferred one %d; vLLM serves "+
			"an FP8 checkpoint's embedding unquantized", stated, inferred)
	}

	off := false
	in = fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.EngramConfig = &engineconfig.EngramConfig{CPUOffload: &off} })
	if hbm := build(t, in).FixedBytes().Weights; hbm <= stated {
		t.Errorf("tables in HBM hold %d bytes, no more than tables in host memory (%d)",
			hbm, stated)
	}
}

// Scales are resident: a block-scaled format holds more than its payload, and stating an
// MXFP4 node's intrinsic scale changes nothing.
func TestScaleBytesAreResident(t *testing.T) {
	base := build(t, fixtureInputs(t, v41)).FixedBytes().Weights
	in := fixtureInputs(t, v41)
	in.Model.Global.WeightScale = nil
	if build(t, in).FixedBytes().Weights >= base {
		t.Error("dropping the FP8 weights' resident scales did not lower the resident bytes")
	}
	in = fixtureInputs(t, v41)
	editNodes(in.Model, role("experts"), func(n *model.Node) {
		n.WeightScale = &model.WeightScale{Elements: 32, Bytes: 1}
	})
	if got := build(t, in).FixedBytes().Weights; got != base {
		t.Errorf("stating MXFP4's own scale moved the resident bytes from %d to %d", base, got)
	}
}
