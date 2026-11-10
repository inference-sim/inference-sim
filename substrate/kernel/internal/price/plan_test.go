package price

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// emitAll answers every condition affirmatively, so a plan built with it contains every
// node. Used where a test is about planning rather than about resolution.
type emitAll struct{}

func (emitAll) Emits(model.EmitCondition) bool { return true }
func (emitAll) Recognizes(c model.EmitCondition) bool {
	return c.Valid()
}

// emitNone drops every conditional node, which is the single-rank layout.
type emitNone struct{}

func (emitNone) Emits(c model.EmitCondition) bool { return c == model.EmitAlways }
func (emitNone) Recognizes(c model.EmitCondition) bool {
	return c.Valid()
}

// emitUnknown recognizes nothing, standing in for a resolver that meets a condition it
// cannot evaluate.
type emitUnknown struct{}

func (emitUnknown) Emits(model.EmitCondition) bool      { return false }
func (emitUnknown) Recognizes(model.EmitCondition) bool { return false }

// hybridGraph mirrors the widest model in the catalog: 108 layers over three kinds, which
// is the case that shows why a plan aggregates by kind.
func hybridGraph() *model.Graph {
	const hidden = 8192
	attn := model.LayerKind{ID: "attention", Nodes: []model.Node{
		{Op: model.OpElementwise},
		{Op: model.OpGEMM, N: 10240, K: hidden},
		{Op: model.OpAttention, HoldsEngineKV: model.Bool(true), AttentionKind: model.AttentionGQA,
			NumQHeads: 64, NumKVHeads: 2, HeadDim: 128},
		{Op: model.OpGEMM, N: hidden, K: hidden},
		{Op: model.OpAllReduce, Emit: model.EmitTensorParallel},
	}}
	moe := model.LayerKind{ID: "moe", Nodes: []model.Node{
		{Op: model.OpElementwise},
		{Op: model.OpGroupedGEMM, Shared: model.Bool(false), N: 5120, K: hidden, Experts: 512, TopK: 22,
			SharedExperts: 1, SharedIntermediateSize: 10240},
		{Op: model.OpAll2All, Emit: model.EmitExpertParallel},
	}}
	mamba := model.LayerKind{ID: "mamba", Nodes: []model.Node{
		{Op: model.OpElementwise},
		{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentMamba2,
			NumHeads: 256, HeadDim: 64, StateSize: 128, NumGroups: 8, ConvKernel: 4,
			IntermediateSize: 16384},
		{Op: model.OpAllReduce, Emit: model.EmitTensorParallel},
	}}
	seq := make([]string, 0, 108)
	for len(seq) < 108 {
		seq = append(seq, "mamba", "moe")
		if len(seq) < 108 && len(seq)%14 == 0 {
			seq = append(seq, "attention")
		}
	}
	return &model.Graph{
		Kind: "ModelGraph", Name: "hybrid",
		Global:     model.GlobalShape{HiddenSize: hidden, VocabSize: 131072, WeightDType: model.DTypeBF16},
		LayerKinds: []model.LayerKind{mamba, moe, attn},
		Stack:      model.Stack{Prologue: seq[:108]},
		Head: []model.Node{
			{Op: model.OpElementwise},
			{Op: model.OpGEMM, N: 131072, K: hidden},
		},
	}
}

// TestPlanAggregatesByKindNotByLayer is the property the hot path depends on. A plan's
// size must follow the number of distinct layer kinds, not the layer count, or step time
// scales with model depth for no reason.
func TestPlanAggregatesByKindNotByLayer(t *testing.T) {
	g := hybridGraph()
	p, err := BuildPlan(g, emitAll{}, 2, 4)
	if err != nil {
		t.Fatalf("BuildPlan: %v", err)
	}
	if p.TotalLayers != 108 {
		t.Errorf("TotalLayers = %d, want 108", p.TotalLayers)
	}
	if len(p.Layers) != 3 {
		t.Fatalf("the plan holds %d entries for a 3-kind stack; step time would scale "+
			"with depth", len(p.Layers))
	}
	// The counts must sum to the expanded stack, or work has been lost or duplicated.
	sum := 0
	for _, l := range p.Layers {
		sum += l.Count
	}
	if sum != 108 {
		t.Errorf("layer counts sum to %d, want 108", sum)
	}
}

// TestPlanIsIndependentOfHowTheStackIsSpelled: a literal sequence and an equivalent
// pattern-and-repeat must produce the same plan. The compression in the catalog is a
// storage choice, and a plan that differed between the two forms would make it a
// behavioural one.
func TestPlanIsIndependentOfHowTheStackIsSpelled(t *testing.T) {
	literal := hybridGraph()
	compressed := hybridGraph()
	// Respell the same sequence as a pattern where it happens to be periodic.
	uniform := make([]string, 0, 60)
	for i := 0; i < 60; i++ {
		uniform = append(uniform, "moe")
	}
	literal.Stack = model.Stack{Prologue: uniform}
	compressed.Stack = model.Stack{Pattern: []string{"moe"}, Repeat: 60}

	a, err := BuildPlan(literal, emitAll{}, 2, 4)
	if err != nil {
		t.Fatalf("literal: %v", err)
	}
	b, err := BuildPlan(compressed, emitAll{}, 2, 4)
	if err != nil {
		t.Fatalf("compressed: %v", err)
	}
	if len(a.Layers) != len(b.Layers) || a.TotalLayers != b.TotalLayers {
		t.Fatalf("the two spellings planned differently: %d/%d against %d/%d",
			len(a.Layers), a.TotalLayers, len(b.Layers), b.TotalLayers)
	}
	for i := range a.Layers {
		if a.Layers[i].Count != b.Layers[i].Count ||
			a.Layers[i].DenseFLOPsPerToken != b.Layers[i].DenseFLOPsPerToken {
			t.Errorf("layer %d differs between spellings", i)
		}
	}
}

// TestConditionalNodesAreResolvedAtPlanTime: a plan holds only nodes that will run, so no
// condition is evaluated per step.
func TestConditionalNodesAreResolvedAtPlanTime(t *testing.T) {
	g := hybridGraph()
	with, err := BuildPlan(g, emitAll{}, 2, 4)
	if err != nil {
		t.Fatalf("with collectives: %v", err)
	}
	without, err := BuildPlan(g, emitNone{}, 2, 4)
	if err != nil {
		t.Fatalf("without collectives: %v", err)
	}
	countCollectives := func(p *Plan) int {
		n := 0
		for _, l := range p.Layers {
			n += len(l.Collectives)
		}
		return n
	}
	if countCollectives(with) == 0 {
		t.Fatal("a tensor- and expert-parallel layout planned no collectives")
	}
	if got := countCollectives(without); got != 0 {
		t.Errorf("a single-rank layout planned %d collectives; they should be resolved "+
			"out rather than evaluated per step", got)
	}
	// The non-collective work must be identical: a layout changes which collectives run,
	// not what a matmul costs.
	for i := range with.Layers {
		if with.Layers[i].DenseFLOPsPerToken != without.Layers[i].DenseFLOPsPerToken {
			t.Errorf("layer %d: dense work changed with the layout", i)
		}
	}
}

// TestUnknownConditionIsAnErrorNotADroppedNode: the failure mode worth preventing. A plan
// that quietly skipped a node it could not evaluate would return a step time that looks
// plausible and omits a cost.
func TestUnknownConditionIsAnErrorNotADroppedNode(t *testing.T) {
	g := hybridGraph()
	if _, err := BuildPlan(g, emitUnknown{}, 2, 4); err == nil {
		t.Fatal("a condition the resolver cannot evaluate was silently dropped")
	}
}

func TestPlanRejectsAStackNamingAnUndeclaredKind(t *testing.T) {
	g := hybridGraph()
	g.Stack = model.Stack{Pattern: []string{"nonexistent"}, Repeat: 4}
	if _, err := BuildPlan(g, emitAll{}, 2, 4); err == nil {
		t.Fatal("a stack naming an undeclared layer kind was accepted")
	}
}

// BenchmarkBuildPlan measures construction, which happens once per deployment. It is
// allowed to be slow; the step path is not.
func BenchmarkBuildPlan(b *testing.B) {
	g := hybridGraph()
	for i := 0; i < b.N; i++ {
		if _, err := BuildPlan(g, emitAll{}, 2, 4); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWalkPlan measures the per-step traversal: what a step time pays to visit every
// planned layer. It must not scale with the 108-layer depth, and it must not allocate.
func BenchmarkWalkPlan(b *testing.B) {
	g := hybridGraph()
	p, err := BuildPlan(g, emitAll{}, 2, 4)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	var flops float64
	for i := 0; i < b.N; i++ {
		flops = 0
		for _, l := range p.Layers {
			flops += float64(l.Count) * (l.DenseFLOPsPerToken +
				float64(l.TopK)*l.ExpertFLOPsPerTokenPerExpert +
				l.SharedExpertFLOPsPerToken)
		}
	}
	if flops == 0 {
		b.Fatal("the traversal computed nothing")
	}
}

// blockSparseGraph mirrors MiniMax-M3's sparse layer: an indexer that scores which blocks
// to read, then the bounded attention that reads them. The indexer comes FIRST, which is
// the order it runs in and the order that used to decide the primary by accident.
func blockSparseGraph() *model.Graph {
	const hidden = 6144
	sparse := model.LayerKind{ID: "sparse", Nodes: []model.Node{
		{Op: model.OpElementwise},
		{Op: model.OpGEMM, N: 9216, K: hidden},
		{Op: model.OpGEMM, N: 640, K: hidden, Role: "index_qk_proj"},
		{Op: model.OpAttention, HoldsEngineKV: model.Bool(false), Role: "block_index_scores",
			AttentionKind: model.AttentionGQA,
			NumQHeads:     4, NumKVHeads: 1, HeadDim: 128},
		{Op: model.OpAttention, HoldsEngineKV: model.Bool(true), AttentionKind: model.AttentionSWA,
			NumQHeads: 64, NumKVHeads: 4, HeadDim: 128, Window: 2176},
		{Op: model.OpGEMM, N: hidden, K: 8192},
	}}
	return &model.Graph{
		Global:     model.GlobalShape{HiddenSize: hidden, VocabSize: 200064},
		LayerKinds: []model.LayerKind{sparse},
		Stack:      model.Stack{Pattern: []string{"sparse"}, Repeat: 57},
	}
}

// pooledIndexerGraph mirrors GLM-5.3-Flash's sparse layer: a k-pool indexer whose scan
// is bounded by the pooling ratio rather than by a window, feeding a latent attention
// bounded by the top-k it selects. ratio is index_kpool.
func pooledIndexerGraph(ratio int) *model.Graph {
	const hidden = 4096
	sparse := model.LayerKind{ID: "sparse", Nodes: []model.Node{
		{Op: model.OpElementwise},
		{Op: model.OpGEMM, N: 2048, K: hidden},
		{Op: model.OpGEMM, N: 4096, K: 1536, Role: "index_wq_b"},
		{Op: model.OpAttention, HoldsEngineKV: model.Bool(false), Role: "block_index_scores",
			AttentionKind: model.AttentionSparseMLA,
			NumQHeads:     32, NumKVHeads: 1, HeadDim: 128,
			CompressRatio: ratio},
		{Op: model.OpAttention, HoldsEngineKV: model.Bool(true), AttentionKind: model.AttentionSparseMLA,
			NumQHeads: 64, NumKVHeads: 1, HeadDim: 512, IndexTopK: 2048},
		{Op: model.OpGEMM, N: hidden, K: 16384},
	}}
	return &model.Graph{
		Global:     model.GlobalShape{HiddenSize: hidden, VocabSize: 154880},
		LayerKinds: []model.LayerKind{sparse},
		Stack:      model.Stack{Pattern: []string{"sparse"}, Repeat: 45},
	}
}

func TestPlanCarriesAPooledScorersCompressionRatio(t *testing.T) {
	// A pooled indexer caches one state per index_kpool tokens, so the ratio is the
	// bound on how many candidates its scan covers. Dropped in planning, the scorer
	// priced over full context -- four times the candidates on the one term in the
	// layer that grows with context.
	for _, ratio := range []int{1, 2, 4, 16} {
		p, err := BuildPlan(pooledIndexerGraph(ratio), emitNone{}, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		var scorer *PlannedAttention
		for i := range p.Layers[0].Attentions {
			if a := &p.Layers[0].Attentions[i]; a.Role == "block_index_scores" {
				scorer = a
			}
		}
		if scorer == nil {
			t.Fatalf("ratio %d: no scorer in the plan", ratio)
		}
		if scorer.CompressRatio != ratio {
			t.Errorf("ratio %d: plan carries CompressRatio %d",
				ratio, scorer.CompressRatio)
		}
	}
}

func TestPlanKeepsEveryAttentionKernelALayerLaunches(t *testing.T) {
	// A layer that launches two attention kernels must report two. An earlier plan
	// assigned the attention fields per node, so the second overwrote the first and the
	// step paid for one of them.
	p, err := BuildPlan(blockSparseGraph(), emitNone{}, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	l := p.Layers[0]
	if got := len(l.Attentions); got != 2 {
		t.Fatalf("layer launches 2 attention kernels, plan has %d", got)
	}
}

func TestPrimaryAttentionIsTheOneHoldingKVWhicheverOrderTheyAppearIn(t *testing.T) {
	// Asserted in BOTH orders on purpose. With the indexer first, a plan that takes the
	// last node seen gets the right answer by luck, so that arrangement alone cannot
	// tell a role-aware choice from the overwrite this replaced. With the indexer last,
	// only a role-aware choice still reports the attention that holds KV.
	for _, tc := range []struct {
		name       string
		indexFirst bool
	}{{"indexer first", true}, {"indexer last", false}} {
		t.Run(tc.name, func(t *testing.T) {
			g := blockSparseGraph()
			nodes := g.LayerKinds[0].Nodes
			var idx, main int
			for i, n := range nodes {
				if n.Op != model.OpAttention {
					continue
				}
				if n.ReadsEngineKV() {
					main = i
				} else {
					idx = i
				}
			}
			if (idx < main) != tc.indexFirst {
				nodes[idx], nodes[main] = nodes[main], nodes[idx]
			}
			p, err := BuildPlan(g, emitNone{}, 2, 4)
			if err != nil {
				t.Fatal(err)
			}
			l := p.Layers[0]
			if l.AttnQHeads != 64 || l.AttnKVHeads != 4 {
				t.Fatalf("primary is %dq/%dkv, want 64q/4kv (the indexer is 4q/1kv)",
					l.AttnQHeads, l.AttnKVHeads)
			}
			if l.AttnKind != model.AttentionSWA || l.AttnWindow != 2176 {
				t.Fatalf("primary should carry the bounded read, got kind=%s window=%d",
					l.AttnKind, l.AttnWindow)
			}
		})
	}
}

func TestIndexerIsNotCountedAsHoldingKV(t *testing.T) {
	// The scorer keeps its own narrow cache, sized by the model's index projections
	// rather than by num_key_value_heads. Counting it as KV-holding doubled MiniMax-M3's
	// KV-holding layer count and let its geometry set the engine's page size.
	p, err := BuildPlan(blockSparseGraph(), emitNone{}, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	var holders int
	for _, a := range p.Layers[0].Attentions {
		if a.HoldsKV {
			holders++
		}
	}
	if holders != 1 {
		t.Fatalf("%d attention kernels claim to hold the engine's KV, want 1", holders)
	}
}

func TestSingleAttentionLayerStillReportsItAsPrimary(t *testing.T) {
	// The common case must be unchanged: one attention, reported in both places.
	p, err := BuildPlan(hybridGraph(), emitNone{}, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range p.Layers {
		if len(l.Attentions) == 0 {
			continue
		}
		if len(l.Attentions) != 1 {
			t.Fatalf("layer has %d attentions, want 1", len(l.Attentions))
		}
		a := l.Attentions[0]
		if a.QHeads != l.AttnQHeads || a.KVHeads != l.AttnKVHeads ||
			a.HeadDim != l.AttnHeadDim || a.Kind != l.AttnKind {
			t.Fatalf("the sole attention and the primary fields disagree")
		}
	}
}

// mixedPrecisionMoE mirrors DeepSeek-V4-Pro: routed experts stored at fp4 beside an fp8
// checkpoint. The node-level override is the only way to express that, and getting it
// wrong doubles the expert weights, which is what made the kernel refuse a deployment
// InferenceX actually ran on 8 GPUs.
func mixedPrecisionMoE(expertDType model.DType) *model.Graph {
	const hidden = 7168
	nodes := []model.Node{
		{Op: model.OpGEMM, Role: "qkv_proj", N: 66560, K: hidden},
		{Op: model.OpGroupedGEMM, Shared: model.Bool(false), Role: "experts", N: 3072, K: hidden,
			Experts: 384, TopK: 6, WeightDType: expertDType},
	}
	return &model.Graph{
		Global:     model.GlobalShape{HiddenSize: hidden, VocabSize: 129280},
		LayerKinds: []model.LayerKind{{ID: "moe", Nodes: nodes}},
		Stack:      model.Stack{Pattern: []string{"moe"}, Repeat: 61},
	}
}

func TestANodeDTypeOverridesTheCheckpointWidthForWeightsOnly(t *testing.T) {
	// fp8 globally (2 bytes passed as the checkpoint width here), experts at fp4.
	const global = 1.0
	base, err := BuildPlan(mixedPrecisionMoE(""), emitNone{}, global, 4)
	if err != nil {
		t.Fatal(err)
	}
	over, err := BuildPlan(mixedPrecisionMoE(model.DTypeNVFP4), emitNone{}, global, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, o := base.Layers[0], over.Layers[0]

	// The expert weights halve, because nvfp4 is half of fp8.
	got, want := o.ExpertWeightBytesPerExpert, b.ExpertWeightBytesPerExpert/2
	if got != want {
		t.Errorf("expert weight bytes = %.0f, want %.0f (half the fp8 figure)", got, want)
	}
	// The DENSE weights do not, because the override is on the expert node alone.
	// Asserted against the ABSOLUTE figure rather than only against the other graph: both
	// graphs share the dense node, so a bug that applied a wrong width to it in BOTH would
	// cancel in a comparison and pass. 66560 * 7168 parameters at the 1-byte global width.
	const wantDense = 66560.0 * 7168 * 1
	for name, l := range map[string]PlannedLayer{"without an override": b, "with one": o} {
		if l.DenseWeightBytes != wantDense {
			t.Errorf("%s: dense weight bytes = %.0f, want %.0f at the global width",
				name, l.DenseWeightBytes, wantDense)
		}
	}
	// FLOPs count arithmetic, not bytes, so a storage width cannot change them.
	if o.ExpertFLOPsPerTokenPerExpert != b.ExpertFLOPsPerTokenPerExpert {
		t.Errorf("expert FLOPs moved with a storage width: %.0f vs %.0f",
			o.ExpertFLOPsPerTokenPerExpert, b.ExpertFLOPsPerTokenPerExpert)
	}
}

func TestAnAbsentNodeDTypeLeavesEveryFigureUnchanged(t *testing.T) {
	// The common case must be untouched: every committed graph states no node dtype, so a
	// plan built without one has to match what it produced before the field existed.
	p, err := BuildPlan(hybridGraph(), emitNone{}, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range p.Layers {
		for _, n := range []float64{l.DenseWeightBytes, l.ExpertWeightBytesPerExpert} {
			if n < 0 {
				t.Fatalf("negative weight bytes: %f", n)
			}
		}
	}
	// hybridGraph's expert node is N=5120 over K=8192, three matrices per gated expert,
	// at the 2 bytes per parameter this plan was built with.
	var found bool
	for _, l := range p.Layers {
		if l.ExpertWeightBytesPerExpert > 0 {
			found = true
			if want := 3.0 * 5120 * 8192 * 2; l.ExpertWeightBytesPerExpert != want {
				t.Errorf("expert bytes = %.0f, want %.0f", l.ExpertWeightBytesPerExpert, want)
			}
		}
	}
	if !found {
		t.Fatal("no expert layer in the plan")
	}
}

// TestDenseGEMMShardAxisComesFromTheGraph pins the classification that decides which
// dimension tensor parallelism divides. Getting it backwards is not a rounding error:
// minimax-m3's mlp_gate_up is (n=24576, k=6144) at hidden 6144, so sharding K instead
// of N would price its 6144-deep reduction as 768-deep at TP=8 -- on the widest GEMM in
// the layer, where a shape-aware efficiency is most sensitive.
//
// The rule is a property of the projection, readable from the shape: a column-parallel
// GEMM takes the whole hidden state and splits its output, so K == hidden; a
// row-parallel GEMM consumes a shard and reduces to the whole hidden state, so
// N == hidden.
func TestDenseGEMMShardAxisComesFromTheGraph(t *testing.T) {
	const hidden = 6144
	nodes := []model.Node{
		{Op: model.OpGEMM, Role: "qkv_proj", N: 9216, K: hidden},
		{Op: model.OpGEMM, Role: "o_proj", N: hidden, K: 8192},
		{Op: model.OpGEMM, Role: "mlp_gate_up", N: 24576, K: hidden},
		{Op: model.OpGEMM, Role: "mlp_down", N: hidden, K: 12288},
		// minimax-m3's indexer projection: column-parallel (K == hidden) yet NARROWER
		// than its reduction. It is here because it separates the rule from a
		// coincidence -- "shard whichever dimension is larger" classifies the other four
		// shapes correctly and this one wrongly.
		{Op: model.OpGEMM, Role: "index_qk_proj", N: 640, K: hidden},
	}
	pl, err := planNodes(nodes, emitAll{}, testOptions(1, 1), hidden)
	if err != nil {
		t.Fatal(err)
	}
	if len(pl.DenseGEMMs) != len(nodes) {
		t.Fatalf("planned %d GEMMs, want %d", len(pl.DenseGEMMs), len(nodes))
	}
	want := map[string]bool{
		"qkv_proj": true, "mlp_gate_up": true, // column-parallel: shard the output
		"o_proj": false, "mlp_down": false, // row-parallel: shard the reduction
		"index_qk_proj": true, // column-parallel despite N < K
	}
	for i, g := range pl.DenseGEMMs {
		role := nodes[i].Role
		if g.ShardN != want[role] {
			t.Errorf("%s (n=%d k=%d): ShardN=%v, want %v",
				role, g.N, g.K, g.ShardN, want[role])
		}
	}
}

// TestDenseGEMMShardAxisFallsBackWithoutHidden pins the degenerate case. A graph with no
// hidden size recorded must not claim a classification it cannot make: the consumer then
// shards the reduction, which is what it did before this field existed.
func TestDenseGEMMShardAxisFallsBackWithoutHidden(t *testing.T) {
	pl, err := planNodes([]model.Node{
		{Op: model.OpGEMM, Role: "unknown", N: 4096, K: 4096},
	}, emitAll{}, testOptions(1, 1), 0)
	if err != nil {
		t.Fatal(err)
	}
	if pl.DenseGEMMs[0].ShardN {
		t.Fatal("with no hidden size the plan must not assert a column-parallel split")
	}
}

// TestConsecutiveColumnParallelGEMMsFuse pins the grouping that matches what the engine
// launches. vLLM concatenates column-parallel projections of the same hidden state along
// the OUTPUT dimension -- QKVParallelLinear and MergedColumnParallelLinear, both in
// vllm/model_executor/layers/linear.py -- so they are one matmul with the widths summed.
//
// It matters because efficiency is strongly non-linear in output width: on AISimulate's
// vLLM fp8 sweep for H200 at 256 rows, measured efficiency is 0.0008 at n=16 against
// 0.0138 at n=256. minimax-m3's layer-1 indexer projection is n=640, which sharded over
// eight ranks is n=80 -- pricing it as its own kernel charges for a fragmentation vLLM
// has already removed.
//
// Three cases, because the rule has three parts and each can be dropped independently.
func TestConsecutiveColumnParallelGEMMsFuse(t *testing.T) {
	const hidden = 6144
	plan := func(nodes ...model.Node) PlannedLayer {
		t.Helper()
		pl, err := planNodes(nodes, emitAll{}, testOptions(1, 1), hidden)
		if err != nil {
			t.Fatal(err)
		}
		return pl
	}

	// Adjacent, both column-parallel, same K: fused, widths summed, FLOPs preserved.
	got := plan(
		model.Node{Op: model.OpGEMM, Role: "qkv_proj", N: 9216, K: hidden},
		model.Node{Op: model.OpGEMM, Role: "index_qk_proj", N: 640, K: hidden},
	)
	if len(got.DenseGEMMs) != 1 {
		t.Fatalf("adjacent column-parallel GEMMs on the same K did not fuse: %d GEMMs",
			len(got.DenseGEMMs))
	}
	if got.DenseGEMMs[0].N != 9216+640 {
		t.Fatalf("fused width %d, want %d", got.DenseGEMMs[0].N, 9216+640)
	}
	wantFLOPs := 2.0 * (9216*hidden + 640*hidden)
	if f := got.DenseGEMMs[0].FLOPsPerToken; f != wantFLOPs {
		t.Fatalf("fusing changed the work: %.0f against %.0f", f, wantFLOPs)
	}

	// Separated by an op the engine cannot fuse across: NOT fused, even though both are
	// column-parallel on the same K. This is minimax-m3's layer-0, where qkv_proj and
	// mlp_gate_up sit either side of the attention.
	got = plan(
		model.Node{Op: model.OpGEMM, Role: "qkv_proj", N: 9216, K: hidden},
		model.Node{Op: model.OpAttention, HoldsEngineKV: model.Bool(true), NumQHeads: 64, NumKVHeads: 4, HeadDim: 128},
		model.Node{Op: model.OpGEMM, Role: "mlp_gate_up", N: 24576, K: hidden},
	)
	if len(got.DenseGEMMs) != 2 {
		t.Fatalf("GEMMs separated by attention fused: %d GEMMs, want 2",
			len(got.DenseGEMMs))
	}

	// A row-parallel GEMM never fuses into a column-parallel one: they shard opposite
	// dimensions, so concatenating their outputs is not something the engine can do.
	//
	// The K values here are deliberately EQUAL. A row-parallel projection usually has a
	// different K, so a rule that tested only "same K" would pass this case by accident
	// -- and one did: dropping the ShardN conditions survived an earlier version of this
	// test, where o_proj carried k=8192 against the column GEMM's k=6144.
	got = plan(
		model.Node{Op: model.OpGEMM, Role: "col", N: 9216, K: hidden},
		model.Node{Op: model.OpGEMM, Role: "row", N: hidden, K: hidden},
	)
	if len(got.DenseGEMMs) != 2 {
		t.Fatalf("a row-parallel GEMM on the same K fused into a column-parallel one: "+
			"%d GEMMs, want 2", len(got.DenseGEMMs))
	}
}

// testOptions plans every weight at dtypeBytes and every format as bf16, the shape the
// planNodes tests below were written against.
func testOptions(dtypeBytes, stateBytes float64) PlanOptions {
	return PlanOptions{
		WeightBytesPerParam: func(n model.Node) float64 { return weightBytes(n, dtypeBytes) },
		Format:              func(model.Node) Format { return Format{DType: model.DTypeBF16} },
		EntryBytes:          func(c model.Cache) int { return c.BytesPerEntry },
		StateBytes:          stateBytes,
	}
}

// A layer that runs attention with no primary -- every attention node states holds_engine_kv
// false, none reads the engine's cache -- is refused rather than planned with its attention
// unpriced.
func TestALayerWithAttentionButNoPrimaryIsRefused(t *testing.T) {
	_, err := planNodes([]model.Node{
		{Op: model.OpAttention, HoldsEngineKV: model.Bool(false), Role: "attention", AttentionKind: model.AttentionGQA,
			NumQHeads: 8, NumKVHeads: 8, HeadDim: 128},
	}, emitAll{}, testOptions(2, 4), 1024)
	if err == nil {
		t.Fatal("a layer whose only attention does not hold the engine cache was planned")
	}
}

// Two routed-expert nodes of one layer computing from different formats cannot be one
// expert term, and are refused rather than the second silently replacing the first.
func TestConflictingExpertFormatsAreRefused(t *testing.T) {
	o := testOptions(1, 4)
	o.Format = func(n model.Node) Format { return Format{DType: n.WeightDType} }
	_, err := planNodes([]model.Node{
		{Op: model.OpGroupedGEMM, Shared: model.Bool(false), N: 512, K: 1024, Experts: 64, TopK: 4,
			WeightDType: model.DTypeFP8},
		{Op: model.OpGroupedGEMM, Shared: model.Bool(false), N: 512, K: 1024, Experts: 64, TopK: 4,
			WeightDType: model.DTypeMXFP4},
	}, emitAll{}, o, 1024)
	if err == nil {
		t.Fatal("two routed-expert formats in one layer were planned as one")
	}
}

// A latent MoE's dispatch and combine carry the routed experts' input width, latent_size,
// not the hidden size, whichever order the graph states the two nodes in; a layer with no
// latent projection keeps the hidden width, and a shared expert's latent_size does not count.
func TestTheMoEDispatchCarriesTheLatentWidth(t *testing.T) {
	const hidden, latent = 8192, 4096
	experts := model.Node{Op: model.OpGroupedGEMM, Shared: model.Bool(false), Role: "experts", N: 4096, K: hidden,
		Experts: 256, TopK: 8, LatentSize: latent}
	a2a := model.Node{Op: model.OpAll2All, Emit: model.EmitExpertParallel}
	for _, c := range []struct {
		what  string
		nodes []model.Node
		want  float64
	}{
		{"experts then dispatch", []model.Node{experts, a2a}, latent * ActivationBytes},
		{"dispatch then experts", []model.Node{a2a, experts}, latent * ActivationBytes},
		{"no latent", []model.Node{{Op: model.OpGroupedGEMM, Shared: model.Bool(false), N: 4096, K: hidden, Experts: 256, TopK: 8}, a2a},
			hidden * ActivationBytes},
	} {
		pl, err := planNodes(c.nodes, emitAll{}, PlanOptions{
			WeightBytesPerParam: func(model.Node) float64 { return 2 },
			Format:              func(model.Node) Format { return Format{DType: model.DTypeBF16} },
		}, hidden)
		if err != nil {
			t.Fatal(err)
		}
		var got float64
		for _, col := range pl.Collectives {
			if col.Op == model.OpAll2All {
				got = col.BytesPerToken
			}
		}
		if got != c.want {
			t.Errorf("%s: the all-to-all carries %v B a token, want %v", c.what, got, c.want)
		}
	}
}

// A share of the stack's piecewise segments are its own layers' split points plus one, so a
// pipeline stage counts the indexers of the sparse-MLA layers it holds and no others.
func TestPiecewiseSegmentsOfAShare(t *testing.T) {
	dense := PlannedLayer{Count: 10, AttnKind: model.AttentionGQA,
		Attentions: []PlannedAttention{{HoldsKV: true}}}
	sparse := PlannedLayer{Count: 6, AttnKind: model.AttentionSparseMLA,
		Attentions: []PlannedAttention{{HoldsKV: true}, {Role: "block_index_scores"}}}
	if got := PiecewiseSegmentsOf(16, []PlannedLayer{dense, sparse}); got != 16+6+1 {
		t.Errorf("whole stack: %d segments, want 23", got)
	}
	sparse.Count = 2
	if got := PiecewiseSegmentsOf(5, []PlannedLayer{{Count: 3, AttnKind: model.AttentionGQA}, sparse}); got != 5+2+1 {
		t.Errorf("a stage of 3 dense and 2 sparse layers: %d segments, want 8", got)
	}
}
