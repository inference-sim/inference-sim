// The kernel's pipeline-parallel and Muse Spark fixes (blis-latency-kernel 223e6a5, 50320f5),
// checked through its unexported state against committed fixtures.

package latencykernel

import (
	"math"
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/kernel/enginesettings"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
)

// The pipeline-parallel fixes, read through the kernel's own state: each test builds a pool
// from a committed fixture, edits its layout and checks one stage-level quantity against what
// vLLM's worker holds or sends.

// ppInputs is a fixture at pp x tp on its own cluster, with a named drafter's graph loaded.
func ppInputs(t *testing.T, fix string, pp, tp int) Inputs {
	t.Helper()
	in := fixtureInputs(t, fix)
	p := &in.Deployment.Pools[0]
	p.Parallel.PP, p.Parallel.TP = pp, tp
	if in.Engine.Speculative != nil && in.Engine.Speculative.DraftModel != "" {
		d, err := schemas.LoadModelGraph(filepath.Join(catalogRoot, "models",
			in.Engine.Speculative.DraftModel, "graph.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		in.Draft = d
	}
	return in
}

// stageWeights is one stage's resident weights as vLLM's worker loads them, recomputed from
// the plan: its layers, the embedding on the first, the head and drafter on the last, and the
// target's embedding there too for an EAGLE-family drafter.
func stageWeights(k *Kernel, st *pipelineStage) float64 {
	var w float64
	for _, l := range st.layers {
		w += float64(l.Count) * k.residentBytes(l)
	}
	if st.first {
		w += k.embeddingResidentBytes(k.graph)
	}
	if st.last {
		w += k.residentBytes(k.plan.Head) + k.draftResidentBytes()
		if k.speculating && needsTargetEmbed(k.plan.Draft.Method) {
			w += k.embeddingResidentBytes(k.graph)
		}
	}
	return w
}

// B1: a recurrent layer's state is held by the stage that holds the layer.
func TestPPRecurrentStateIsTheStagesOwn(t *testing.T) {
	mk := func(pp int) *Kernel {
		in := ppInputs(t, "nemotron3-ultra-h100-agg.yaml", pp, 4)
		in.Deployment.Pools[0].Nodes, in.Scenario.Cluster.Nodes = 1, 1
		k, err := newKernel(in)
		if err != nil {
			t.Fatalf("pp %d: %v", pp, err)
		}
		return k
	}
	one, two := mk(1), mk(2)
	// The state a sequence holds a layer, as the whole model's figure is spread over its
	// layers; a stage holds its own layers' count of it.
	var all, mine float64
	for _, l := range two.plan.Layers {
		all += float64(l.Count) * l.RecurrentStateBytes
	}
	for _, l := range two.stages[two.ppKVStage].layers {
		mine += float64(l.Count) * l.RecurrentStateBytes
	}
	want := float64(one.SequenceBoundedBytes(0)) * mine / all
	got := two.SequenceBoundedBytes(0)
	if math.Abs(float64(got)-want) > 1 || got >= one.SequenceBoundedBytes(0) {
		t.Errorf("pp 2 rank holds %d B of recurrent state a sequence, its stage's layers %.0f, pp 1 %d",
			got, want, one.SequenceBoundedBytes(0))
	}
}

// B2: the binding stage is the one whose own memory, less its own weights, over its own KV a
// token holds the fewest tokens, and FixedBytes and SequenceVariableBytes are that stage's.
func TestPPCapacityIsTheSmallestStagesPool(t *testing.T) {
	for _, c := range []struct {
		fix    string
		pp, tp int
	}{
		{"muse-h100-pp2tp8.yaml", 4, 4},
		{"muse-h100-pp2tp8.yaml", 8, 2},
		{"muse-h100-pp2tp8-dspark.yaml", 2, 8},
		{"minimax-m25-h200-tp8.yaml", 3, 2},
		{"minimax-m25-h200-tp8.yaml", 4, 2},
	} {
		k, err := newKernel(ppInputs(t, c.fix, c.pp, c.tp))
		if err != nil {
			t.Fatalf("%s pp %d tp %d: %v", c.fix, c.pp, c.tp, err)
		}
		budget := k.chip.MemoryGiB * (1 << 30) * assumedMemoryUtilization
		best, at := math.Inf(1), -1
		for i := range k.stages {
			st := &k.stages[i]
			var kv float64
			for j, l := range st.layers {
				if st.kv[j].window == 0 {
					kv += st.kv[j].perToken * float64(l.Count)
				}
			}
			if st.last && k.speculating {
				kv += k.draftKVBytesPerToken
			}
			other := float64(k.captureBytes+k.deepEPBufferBytes(k.graph)) + k.activationBytes(k.graph, st.last)
			if tok := (budget - stageWeights(k, st) - other) / kv; tok < best {
				best, at = tok, i
			}
		}
		st := &k.stages[at]
		fixed := float64(k.FixedBytes().Total())
		if at != k.ppKVStage || math.Abs(fixed-(stageWeights(k, st)+float64(k.captureBytes)+
			float64(k.deepEPBufferBytes(k.graph))+k.activationBytes(k.graph, st.last))) > 2 {
			t.Errorf("%s pp %d tp %d: binding stage %d with %.3f GB fixed; vLLM's smallest pool is stage %d's",
				c.fix, c.pp, c.tp, k.ppKVStage, fixed/1e9, at)
		}
		// vLLM's pool is the binding stage's own: its budget less its own fixed bytes over its own
		// KV a token. The old rule took the most loaded stage's weights and the widest stage's KV
		// separately, which on Muse pp 4 (the first and last stages hold the embedding and the
		// head, the middle two one layer more) priced a rank no worker is.
		kv := k.stages[at].memory.kvPerToken
		tokens := (budget - fixed) / kv
		if math.Abs(tokens-best) > 1 {
			t.Errorf("%s pp %d tp %d: the reported rank holds %.0f tokens, vLLM's smallest pool %.0f",
				c.fix, c.pp, c.tp, tokens, best)
		}
		var maxW, maxKV float64
		for i := range k.stages {
			maxW = math.Max(maxW, stageWeights(k, &k.stages[i]))
			maxKV = math.Max(maxKV, k.stages[i].memory.kvPerToken)
		}
		if float64(k.FixedBytes().Weights) >= maxW-1 && k.ppKVBytesPerToken >= maxKV-1 &&
			stageWeights(k, st) < maxW-1 {
			t.Errorf("%s pp %d tp %d: the rank pairs the most loaded weights with the widest KV", c.fix, c.pp, c.tp)
		}
		sm := k.PipelineStageMemory()
		if len(sm) != c.pp || !sm[at].Binding || sm[at].Fixed != k.FixedBytes() ||
			sm[at].PerBlockBytes != k.SequenceVariableBytes(k.blockSize) {
			t.Errorf("%s pp %d tp %d: PipelineStageMemory %+v does not report the binding stage as FixedBytes does",
				c.fix, c.pp, c.tp, sm)
		}
	}
}

// B3: a graph that states no embedding still holds its inferred table on the first stage.
func TestPPInferredEmbeddingIsOnTheFirstStage(t *testing.T) {
	k, err := newKernel(ppInputs(t, "minimax-m25-h200-tp8.yaml", 2, 4))
	if err != nil {
		t.Fatal(err)
	}
	if k.plan.Embedding.Count > 0 {
		t.Fatal("the fixture's graph states its embedding; the test needs one that does not")
	}
	table := float64(k.graph.Global.VocabSize) * float64(k.graph.Global.HiddenSize) * 2 / k.tp
	var layers float64
	for _, l := range k.stages[0].layers {
		layers += float64(l.Count) * k.residentBytes(l)
	}
	if got := k.stages[0].memory.weights; math.Abs(got-(layers+table)) > 1 {
		t.Errorf("stage 0 holds %.3f GB, its layers %.3f GB and the inferred table %.3f GB", got/1e9,
			layers/1e9, table/1e9)
	}
}

// B4: an unstated boundary is hidden_states and residual, two activations; Muse states its one
// fp32 stream.
func TestPPDefaultBoundaryIsTwoTensors(t *testing.T) {
	k, err := newKernel(ppInputs(t, "minimax-m25-h200-tp8.yaml", 2, 4))
	if err != nil {
		t.Fatal(err)
	}
	if want := 2 * float64(k.graph.Global.HiddenSize) * price.ActivationBytes / 4; k.stages[0].sendBytesPerToken != want {
		t.Errorf("a rank sends %v B a token, want hidden_states + residual over tp 4, %v", k.stages[0].sendBytesPerToken, want)
	}
	m := fixture(t, "muse-h100-pp2tp8.yaml")
	if m.stages[0].sendBytesPerToken != 32768.0/8 {
		t.Errorf("Muse's rank sends %v B a token, want its fp32 stream over tp 8", m.stages[0].sendBytesPerToken)
	}
}

// B5: the receiving stage all-gathers the previous stage's slices over its TP group.
func TestPPReceivingStageAllGathersTheSlices(t *testing.T) {
	k := fixture(t, "muse-h100-pp2tp8.yaml")
	if k.stages[1].recvBytesPerToken != 32768 || k.stages[0].recvBytesPerToken != 0 {
		t.Fatalf("stage receives %v and %v B a token, want 0 and 32768", k.stages[0].recvBytesPerToken,
			k.stages[1].recvBytesPerToken)
	}
	tokens := 8192.0
	on, cross := k.pipelineRecvGatherSeconds(&k.stages[1], tokens)
	want := k.collectiveSeconds(collKey{Op: model.OpAllGather, Group: price.GroupTP}, tokens*32768)
	if on != want || cross != 0 {
		t.Errorf("the receive gather is %v on-node and %v cross-node, want %v on-node", on, cross, want)
	}
	b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 8192, PromptLen: 8192}}}
	e := k.StepTime(b)
	k.stages[1].recvBytesPerToken = 0
	without := k.StepTime(b)
	if e.Stages[1].NoOverlap-without.Stages[1].NoOverlap <= 0 || e.Stages[0] != without.Stages[0] {
		t.Errorf("the gather moved stage 1 by %v and stage 0 by %v", e.Stages[1].NoOverlap-without.Stages[1].NoOverlap,
			e.Stages[0].NoOverlap-without.Stages[0].NoOverlap)
	}
}

// B6: a boundary carries the drafter's auxiliary states captured upstream of it. Muse's DSpark
// reads target layers 14, 30, 46, 58 and 62; at pp 2 (layers [0, 31) and [31, 62)) stage 0
// sends the stream and two fp32 states; at pp 4 ([0, 15), [15, 31), [31, 47), [47, 62)) the
// boundaries carry 1, 2 and 3.
func TestPPBoundaryCarriesTheDrafterAuxStates(t *testing.T) {
	for _, c := range []struct {
		pp, tp int
		aux    []int
	}{{2, 8, []int{2}}, {4, 4, []int{1, 2, 3}}} {
		in := ppInputs(t, "muse-h100-pp2tp8-dspark.yaml", c.pp, c.tp)
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		if got := k.graph.Speculator.AuxHiddenStateLayers; len(got) == 0 {
			// TEMPORARY: until testdata/upstream.lock pins a catalog whose DSpark graph states
			// aux_hidden_state_layers (blis-catalog 74b8f13).
			t.Skip("the pinned catalog's drafter graph states no aux_hidden_state_layers")
		} else if len(got) != 5 {
			t.Fatalf("the drafter graph states aux layers %v, want Muse's five", got)
		}
		for i, n := range c.aux {
			want := (32768 + float64(n)*32768) / float64(c.tp)
			if got := k.stages[i].sendBytesPerToken; got != want {
				t.Errorf("pp %d stage %d sends %v B a token a rank, want the stream and %d aux states, %v",
					c.pp, i, got, n, want)
			}
		}
	}
}

// B8: an in-rack NVLink send is booked to NVLink, as its rate is.
func TestPPInRackSendIsNVLinkTime(t *testing.T) {
	in := ppInputs(t, "muse-gb200-tp8ep8.yaml", 2, 4)
	k, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	st := &k.stages[0]
	if !st.crossesNode || st.overNIC {
		t.Fatalf("GB200 pp 2 x tp 4: crosses a tray %v, leaves the rack %v; want true, false", st.crossesNode, st.overNIC)
	}
	b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 8192, PromptLen: 8192}}}
	if e := k.StepTime(b); e.PerResource[kernel.ResourceNIC] != 0 {
		t.Errorf("an in-rack boundary booked %v of NIC time", e.PerResource[kernel.ResourceNIC])
	}
}

// B9: a grafted drafter's nodes keep the drafter checkpoint's own weight dtype.
func TestPPDrafterKeepsItsOwnDType(t *testing.T) {
	base, err := newKernel(ppInputs(t, "muse-h100-pp2tp8-dspark.yaml", 2, 8))
	if err != nil {
		t.Fatal(err)
	}
	in := ppInputs(t, "muse-h100-pp2tp8-dspark.yaml", 2, 8)
	g := *in.Model
	g.Global.WeightDType = model.DTypeFP8
	in.Model = &g
	fp8, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	if base.draftResidentBytes() != fp8.draftResidentBytes() {
		t.Errorf("the drafter's weights moved with the target's global dtype: %.3f GB against %.3f GB",
			fp8.draftResidentBytes()/1e9, base.draftResidentBytes()/1e9)
	}
}

// B10: under DSpark with pp > 1 the last stage also holds the target's embedding.
func TestPPLastStageHoldsTheTargetEmbedForDSpark(t *testing.T) {
	plain := fixture(t, "muse-h100-pp2tp8.yaml")
	spec, err := newKernel(ppInputs(t, "muse-h100-pp2tp8-dspark.yaml", 2, 8))
	if err != nil {
		t.Fatal(err)
	}
	embed := spec.embeddingResidentBytes(spec.graph)
	last := spec.stages[1].memory.weights - plain.stages[1].memory.weights
	if want := embed + spec.draftResidentBytes(); math.Abs(last-want) > 1 {
		t.Errorf("DSpark adds %.3f GB to the last stage, want the drafter %.3f GB and the target embedding %.3f GB",
			last/1e9, spec.draftResidentBytes()/1e9, embed/1e9)
	}
}

// B12: EPLB's redundant replicas are a stage's own MoE layers'.
func TestPPEPLBRedundancyIsPerStage(t *testing.T) {
	mk := func(pp int) *Kernel {
		in := ppInputs(t, "minimax-m25-h200-ep8.yaml", pp, 8/pp)
		in.Deployment.Pools[0].Parallel.DP = 1
		in.Deployment.Pools[0].Parallel.EnableExpertParallel = true
		eng, err := vllmsettings.OfPool(in.Deployment, 0)
		if err != nil {
			t.Fatal(err)
		}
		eng.EPLB = &enginesettings.EPLB{Enabled: true, NumRedundantExperts: 32}
		in.Engine = &eng
		k, err := New(in)
		if err != nil {
			t.Fatalf("pp %d: %v", pp, err)
		}
		return k
	}
	two := mk(2)
	if want := two.eplbRedundantBytes(two.stages[two.ppKVStage].layers); math.Abs(float64(two.FixedBytes().EPLBRedundant)-want) > 1 {
		t.Errorf("pp 2 rank reports %.3f GB of EPLB replicas, its stage's %.3f GB",
			float64(two.FixedBytes().EPLBRedundant)/1e9, want/1e9)
	}
	if two.FixedBytes().EPLBRedundant <= 0 || two.FixedBytes().EPLBRedundant >= int64(two.eplbRedundantBytes(two.plan.Layers)) {
		t.Errorf("pp 2 rank's EPLB replicas %d are not a share of the whole model's %.0f",
			two.FixedBytes().EPLBRedundant, two.eplbRedundantBytes(two.plan.Layers))
	}
}

// B13: a stage boundary crosses a node when any rank's peer is on another node, for any tp.
func TestPPBoundaryPlacementForAnyTP(t *testing.T) {
	k := fixture(t, "muse-h100-pp2tp8.yaml")
	for _, c := range []struct {
		pp, tp, gpn int
	}{{2, 6, 8}, {3, 6, 8}, {4, 4, 8}, {2, 4, 8}, {4, 3, 8}} {
		k.layout.PP, k.layout.TP, k.layout.GPUsPerNode = c.pp, c.tp, c.gpn
		if err := k.liftPipeline(k.graph); err != nil {
			t.Fatal(err)
		}
		for s, st := range k.stages {
			truth := false
			if !st.last {
				for r := range c.tp {
					truth = truth || (s*c.tp+r)/c.gpn != ((s+1)*c.tp+r)/c.gpn
				}
			}
			if st.crossesNode != truth {
				t.Errorf("pp %d tp %d on %d-GPU nodes, stage %d: crosses %v, the rank layout says %v",
					c.pp, c.tp, c.gpn, s, st.crossesNode, truth)
			}
		}
	}
}

// M6: a sliding-window layer whose cache is inferred is a bounded ring, not KV that grows with
// context: the growing KV a token is the full-attention layers', and each window layer holds
// its window's pages per sequence.
func TestASlidingWindowCacheIsARing(t *testing.T) {
	in := fixtureInputs(t, "muse-gb200-tp8ep8.yaml")
	k, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	var full, swa float64
	var windows int
	for i, l := range k.plan.Layers {
		if l.AttnKind == model.AttentionSWA {
			swa += k.layerKV[i].perToken * float64(l.Count)
			windows = l.AttnWindow
		} else if l.AttnQHeads > 0 {
			full += k.layerKV[i].perToken * float64(l.Count)
		}
	}
	if swa == 0 || k.kvBytesPerToken != full {
		t.Fatalf("KV a token %v, the full-attention layers' %v (window layers %v)", k.kvBytesPerToken, full, swa)
	}
	// A decoding sequence holds each ring at its window plus one token in flight, in pages.
	block := k.blockSize
	pages := (windows-1+1+block-1)/block + 1
	want := float64(pages*block) * swa
	if got := k.SequenceFixedBytes(); math.Abs(float64(got)-want) > 1 {
		t.Errorf("a sequence holds %d B of bounded state, its window rings %.0f", got, want)
	}
	// A P/D handoff moves each ring clipped to its window.
	if got, wantPD := k.pdTransferBytes(100000), 100000*full+float64(min((100000+block-1)/block, (windows+block-1)/block+1)*block)*swa; math.Abs(got-wantPD) > 1 {
		t.Errorf("a 100k handoff moves %.0f B, want the full layers' and the clipped rings' %.0f", got, wantPD)
	}
}

// M7: on Blackwell with TP and no DP, a step of at most 256 tokens computes Muse's MoE pooler
// and unpooler a TP slice at a time and all-gathers the columns; the unpooler, its TP group
// spanning two trays, only from 32 tokens. A larger step, or Hopper, runs them whole.
func TestMusePoolerShardsAtDecodeOnBlackwell(t *testing.T) {
	k := fixture(t, "muse-gb200-tp8ep8.yaml")
	var pool, unpool *price.PlannedGEMM
	for i := range k.plan.Layers[0].DenseGEMMs {
		g := &k.plan.Layers[0].DenseGEMMs[i]
		if g.ShardedAtDecode == nil {
			continue
		}
		if g.ShardedAtDecode.MinTokensAcrossNodes == 0 {
			pool = g
		} else {
			unpool = g
		}
	}
	if pool == nil || unpool == nil {
		// TEMPORARY: until testdata/upstream.lock pins a catalog whose Muse graphs state
		// sharded_at_decode (blis-catalog af32dd0).
		t.Skip("the pinned catalog's Muse graph marks no pooler or unpooler sharded_at_decode")
	}
	for _, c := range []struct {
		tokens       int
		pool, unpool bool
	}{{1, true, false}, {31, true, false}, {32, true, true}, {256, true, true}, {257, false, false}} {
		if got := k.shardsAtDecode(pool, c.tokens); got != c.pool {
			t.Errorf("%d tokens: the pooler shards %v, want %v", c.tokens, got, c.pool)
		}
		if got := k.shardsAtDecode(unpool, c.tokens); got != c.unpool {
			t.Errorf("%d tokens: the unpooler shards %v, want %v", c.tokens, got, c.unpool)
		}
	}
	decode := func(n int) kernel.Batch {
		b := kernel.Batch{DecodeThreshold: 1}
		for range n {
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096})
		}
		return b
	}
	sharded := k.StepTime(decode(64))
	pool.ShardedAtDecode, unpool.ShardedAtDecode = nil, nil
	for i := range k.plan.Layers {
		for j := range k.plan.Layers[i].DenseGEMMs {
			k.plan.Layers[i].DenseGEMMs[j].ShardedAtDecode = nil
		}
	}
	whole := k.StepTime(decode(64))
	if sharded.PerResource[kernel.ResourceHBM] >= whole.PerResource[kernel.ResourceHBM] ||
		sharded.PerResource[kernel.ResourceNVLink] <= whole.PerResource[kernel.ResourceNVLink] {
		t.Errorf("64-token decode: HBM %v sharded against %v whole, NVLink %v against %v; the shard "+
			"reads a tp-th of the weights and adds gathers", sharded.PerResource[kernel.ResourceHBM],
			whole.PerResource[kernel.ResourceHBM], sharded.PerResource[kernel.ResourceNVLink],
			whole.PerResource[kernel.ResourceNVLink])
	}
	h := fixture(t, "muse-h100-pp2tp8.yaml")
	for _, g := range h.plan.Layers[0].DenseGEMMs {
		if g.ShardedAtDecode != nil && h.shardsAtDecode(&g, 64) {
			t.Error("Hopper sharded the pooler")
		}
	}
}
