// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/models/qwen4_exp/nvidia/qsa.py at vLLM v0.34.0+0fda86a35419 (0fda86a35419c9c70dbfada224c838c10fbb62ad).

package kernelmodel

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	latencykernel "github.com/inference-sim/inference-sim/substrate/kernel"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
)

const qwen4 = "Qwen4ExpForConditionalGeneration"

// A Qwen4Exp deployment is refused for each setting the QSA backend refuses, the refusal names
// the setting, and the same setting on another architecture is not refused.
func TestQwen4ExpRefusalsAreTheBackends(t *testing.T) {
	ok := qsaSettings{Arch: qwen4, TP: 8, PP: 1, BlockSize: 64, CacheDType: "fp8"}
	for _, c := range []struct {
		name string
		edit func(*qsaSettings)
		want string // substring of the refusal; empty means accepted
	}{
		{"the defaults", func(*qsaSettings) {}, ""},
		{"block size unstated", func(s *qsaSettings) { s.BlockSize = 0 }, ""},
		{"block size 16", func(s *qsaSettings) { s.BlockSize = 16 }, ""},
		{"block size 24", func(s *qsaSettings) { s.BlockSize = 24 }, "block_size"},
		{"decode context parallelism", func(s *qsaSettings) { s.DCP = 2 }, "decode context"},
		{"prefill context parallelism", func(s *qsaSettings) { s.PCP = 2 }, "prefill context"},
		{"a KV connector", func(s *qsaSettings) { s.KVConnector = true }, "KV connector"},
		{"fp8_e5m2 cache", func(s *qsaSettings) { s.CacheDType = "fp8_e5m2" }, "cache_dtype"},
		{"bfloat16 cache", func(s *qsaSettings) { s.CacheDType = "bfloat16" }, ""},
		{"hc-sp at pp 2", func(s *qsaSettings) { s.EnableHCSP, s.PP = true, 2 }, "enable-hc-sp"},
		{"hc-sp at pp 1", func(s *qsaSettings) { s.EnableHCSP = true }, ""},
	} {
		s := ok
		c.edit(&s)
		err := checkQwen4Exp(s)
		switch {
		case c.want == "" && err != nil:
			t.Errorf("%s: refused: %v", c.name, err)
		case c.want != "" && err == nil:
			t.Errorf("%s: accepted, want a refusal naming %q", c.name, c.want)
		case c.want != "" && !strings.Contains(err.Error(), c.want):
			t.Errorf("%s: refusal does not name %q: %v", c.name, c.want, err)
		}
		if c.want != "" {
			s.Arch = "Qwen3_5MoeForConditionalGeneration"
			if err := checkQwen4Exp(s); err != nil {
				t.Errorf("%s: another architecture was refused: %v", c.name, err)
			}
		}
	}
}

// Hyper-connection sequence parallelism needs tp above 1, and sequence-parallel MoE turns it on
// by itself.
func TestHCSequenceParallelFollowsTheEngine(t *testing.T) {
	for _, c := range []struct {
		tp        int
		hcSP, moe bool
		want      bool
	}{
		{1, true, true, false}, {8, false, false, false}, {8, true, false, true},
		{8, false, true, true}, {2, true, true, true},
	} {
		if got := hcSequenceParallel(c.tp, c.hcSP, c.moe); got != c.want {
			t.Errorf("tp %d enable_hc_sp %v sp_moe %v: %v, want %v", c.tp, c.hcSP, c.moe, got, c.want)
		}
	}
}

func qsaNodes() (model.Node, model.Node) {
	main := model.Node{Op: model.OpAttention, HoldsEngineKV: model.Bool(true), AttentionKind: sparseGQA,
		NumQHeads: 24, NumKVHeads: 2, HeadDim: 256, IndexTopK: 2048}
	scorer := model.Node{Op: model.OpAttention, HoldsEngineKV: model.Bool(false), AttentionKind: sparseGQA,
		NumQHeads: 4, NumKVHeads: 1, HeadDim: 128, CompressRatio: 4}
	return main, scorer
}

// A QSA layer registers its K/V cache, the indexer's key-state ring and its compressed key
// cache. The ring is whole groups of the ratio that hold the open group and the speculative
// rows, so it grows with the number of speculative tokens in steps of the ratio.
func TestQwen4ExpLayerRegistersThreeCaches(t *testing.T) {
	main, scorer := qsaNodes()
	runs, err := qwen4ExpAttentionRuns(main, scorer, 8, 64, 1, "auto", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 3 || runs[0].Spec != "full" || runs[1].Spec != "circular_buffer" || runs[2].Spec != "mla" {
		t.Fatalf("specs %+v", runs)
	}
	if r := runs[0]; r.NumKVHeads != 1 || r.HeadSize != 256 {
		t.Errorf("main cache at tp 8 over 2 KV heads: %+v; the head is replicated, one per rank", r)
	}
	if r := runs[1]; r.BlockSize != 8 || r.HeadSize != 140 || r.ElemBytes != 2 {
		t.Errorf("ring at ratio 4 and 1 speculative token: %+v; want 8 rows of 128 + 12 position slots in bf16", r)
	}
	if r := runs[2]; r.TokensPerState != 4 || r.BlockSize != 64 || r.ElemBytes != 2 || r.HeadSize != 128 {
		t.Errorf("compressed cache: %+v", r)
	}
	for _, c := range []struct{ spec, rows int }{{0, 4}, {3, 8}, {4, 8}, {5, 12}} {
		r, _ := qwen4ExpAttentionRuns(main, scorer, 1, 64, c.spec, "auto", false)
		if r[1].BlockSize != c.rows || r[1].HeadSize != 128 {
			t.Errorf("%d speculative tokens: ring %+v, want %d rows of 128", c.spec, r[1], c.rows)
		}
	}
	if r, _ := qwen4ExpAttentionRuns(main, scorer, 1, 64, 0, "fp8", false); r[2].ElemBytes != 1 {
		t.Errorf("fp8 indexer cache: %+v", r[2])
	}
}

func TestQwen4ExpLayerRefusesWhatTheEngineRefuses(t *testing.T) {
	main, scorer := qsaNodes()
	if _, err := qwen4ExpAttentionRuns(main, scorer, 1, 64, 0, "fp4", false); err == nil {
		t.Error("an fp4 indexer cache was accepted")
	}
	if _, err := qwen4ExpAttentionRuns(main, scorer, 1, 66, 0, "auto", false); err == nil {
		t.Error("a block size the ratio does not divide was accepted")
	}
	scorer.CompressRatio = 0
	if _, err := qwen4ExpAttentionRuns(main, scorer, 1, 64, 0, "auto", false); err == nil {
		t.Error("a scorer without a ratio was accepted")
	}
}

// A sparse-GQA layer beside Gated DeltaNet layers is a hybrid the block-size alignment can size:
// its page is the main K/V cache's (2 KV heads of 256, tp 8 replicates one head per rank) and
// its backend's smallest kernel block is 16 (qsa.py:188-190). Without this it was refused as unmapped.
func TestSparseGQABesideRecurrentLayersIsAHybridTheAlignmentSizes(t *testing.T) {
	main, scorer := qsaNodes()
	main.Caches = []model.Cache{{Name: "kv", Kind: model.CacheKV, Owned: true, BytesPerEntry: 2048}}
	scorer.Caches = []model.Cache{{Name: "index", Kind: model.CacheIndex, Owned: true, BytesPerEntry: 256, TokensPerEntry: 4}}
	gdn := model.Node{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentGDN, NumHeads: 48, StateSize: 128,
		ConvKernel: 4, IntermediateSize: 10240, StateDType: model.DTypeFP32}
	g := &model.Graph{Global: model.GlobalShape{WeightDType: model.DTypeBF16, ContextWindow: 262144},
		LayerKinds: []model.LayerKind{{ID: "rec", Nodes: []model.Node{gdn}}, {ID: "attn", Nodes: []model.Node{scorer, main}}},
		Stack:      model.Stack{Pattern: []string{"rec", "rec", "rec", "attn"}, Repeat: 2}}
	facts, err := attentionOf(g, engineconfig.Engine{BlockSize: 64, CacheDType: "auto"}, 8)
	if err != nil {
		t.Fatal(err)
	}
	if !facts.present || facts.kernelErr != nil || facts.kernelMin != 16 {
		t.Errorf("facts %+v: want attention present, kernel block 16 and no kernel error", facts)
	}
	if want := int64(1 * 256 * 2 * 2); facts.page1Token != want {
		t.Errorf("a token's page is %d B, want %d (one KV head of 256, K and V, bf16)", facts.page1Token, want)
	}
	// Plain GQA of the same shape has the same kernel block.
	main.AttentionKind = model.AttentionGQA
	main.Caches = nil
	g.LayerKinds[1].Nodes = []model.Node{main}
	if f, err := attentionOf(g, engineconfig.Engine{BlockSize: 64, CacheDType: "auto"}, 8); err != nil || f.kernelErr != nil || f.kernelMin != 16 {
		t.Errorf("head size 256 on plain GQA: facts %+v, error %v; want the 16-token kernel block", f, err)
	}
}

// Opening a pool of a Qwen4Exp model refuses at start-up what the QSA backend refuses, with the
// refusal reaching the caller; the same pool of another architecture opens.
func TestOpeningAQwen4ExpPoolRefusesWhatTheBackendRefuses(t *testing.T) {
	const scenario = "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"
	open := func(arch string, edit func(*deployment.Pool)) error {
		in, err := latencykernel.OpenInputs(scenario, repos(), 0)
		if err != nil {
			t.Fatal(err)
		}
		g := *in.Model
		g.Global.Architecture = arch
		in.Model = &g
		edit(&in.Deployment.Pools[0])
		_, err = fromInputs(in)
		return err
	}
	dcp := func(p *deployment.Pool) { p.Parallel.DCP = 2 }
	err := open(qwen4, dcp)
	if err == nil || !strings.Contains(err.Error(), "decode context parallelism") {
		t.Errorf("DCP 2 on Qwen4Exp: %v, want a refusal naming decode context parallelism", err)
	}
	if err := open(qwen4, func(*deployment.Pool) {}); err != nil && strings.Contains(err.Error(), "QSA") {
		t.Errorf("a plain Qwen4Exp pool was refused by the QSA check: %v", err)
	}
	if err := open("GptOssForCausalLM", dcp); err != nil && strings.Contains(err.Error(), "QSA") {
		t.Errorf("another architecture was refused by the QSA check: %v", err)
	}
}

// kvLayersOf maps a hybrid of Gated DeltaNet and QSA layers to the specs vLLM registers, in
// layer order: three recurrent layers, then the QSA layer's K/V cache, ring and compressed keys,
// twice over. Before the QSA branch it refused the scorer's caches as unmapped.
func TestKVLayersOfAQwen4ExpHybridAreTheEnginesSpecs(t *testing.T) {
	main, scorer := qsaNodes()
	main.Caches = []model.Cache{{Name: "kv", Kind: model.CacheKV, Owned: true, BytesPerEntry: 2048}}
	scorer.Caches = []model.Cache{{Name: "index", Kind: model.CacheIndex, Owned: true, BytesPerEntry: 256, TokensPerEntry: 4}}
	gdn := model.Node{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentGDN, NumHeads: 48, StateSize: 128,
		ConvKernel: 4, IntermediateSize: 10240, StateDType: model.DTypeFP32}
	g := &model.Graph{Global: model.GlobalShape{WeightDType: model.DTypeBF16, ContextWindow: 262144},
		LayerKinds: []model.LayerKind{{ID: "rec", Nodes: []model.Node{gdn}}, {ID: "attn", Nodes: []model.Node{scorer, main}}},
		Stack:      model.Stack{Pattern: []string{"rec", "rec", "rec", "attn"}, Repeat: 2}}
	e := engineconfig.Engine{BlockSize: 64}
	plan := mustPlan(t, g, e, 8)
	pool := deployment.Pool{Parallel: deployment.Parallelism{TP: 8, PP: 1, DP: 1}}
	runs, err := kvLayersOf(g, pool, e, deepseekServing{mamba: plan})
	if err != nil {
		t.Fatal(err)
	}
	var specs []string
	count := map[string]int{}
	for _, r := range runs {
		specs = append(specs, r.Spec)
		count[r.Spec] += r.Count
	}
	if want := "mamba full circular_buffer mla mamba full circular_buffer mla"; strings.Join(specs, " ") != want {
		// Consecutive identical runs merge, so the recurrent layers of a block are one run.
		t.Errorf("specs %v, want %s", specs, want)
	}
	if count["mamba"] != 6 || count["full"] != 2 || count["circular_buffer"] != 2 || count["mla"] != 2 {
		t.Errorf("layers per spec %v: want 6 recurrent and 2 each of the QSA layer's three", count)
	}
}

// Qwen4Exp builds two Mamba state layouts, Gated DeltaNet's and the per-layer embedding's short
// convolution, and sizes the model's Mamba page as the larger (get_mamba_specs_from_config,
// qwen4_exp/nvidia/model.py:880-900; platforms/interface.py:1049-1056@4ba787ec). Each layer keeps
// its own state bytes, and both are padded to one page.
func qwen4MambaGraph(arch string, short model.Node) *model.Graph {
	main, scorer := qsaNodes()
	main.Caches = []model.Cache{{Name: "kv", Kind: model.CacheKV, Owned: true, BytesPerEntry: 2048}}
	scorer.Caches = []model.Cache{{Name: "index", Kind: model.CacheIndex, Owned: true, BytesPerEntry: 256, TokensPerEntry: 4}}
	gdn := model.Node{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentGDN, NumHeads: 48, StateSize: 128,
		ConvKernel: 4, IntermediateSize: 10240, StateDType: model.DTypeFP32}
	return &model.Graph{Global: model.GlobalShape{Architecture: arch, WeightDType: model.DTypeBF16, ContextWindow: 262144},
		LayerKinds: []model.LayerKind{{ID: "rec", Nodes: []model.Node{gdn}}, {ID: "ple", Nodes: []model.Node{short, gdn}},
			{ID: "attn", Nodes: []model.Node{scorer, main}}},
		Stack: model.Stack{Pattern: []string{"ple", "rec", "rec", "attn"}, Repeat: 2}}
}

func TestQwen4ExpSizesTheMambaPageAsTheLargerOfItsTwoStateLayouts(t *testing.T) {
	short := model.Node{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentShortConv, ConvKernel: 4,
		ConvDilation: 3, IntermediateSize: 10240}
	e := engineconfig.Engine{BlockSize: 64}
	plan, err := planMamba(qwen4MambaGraph(qwen4, short), e, mambaEnv{tp: 8, dcp: 1, pcp: 1, capability: 9, maxModelLen: 262144})
	if err != nil {
		t.Fatal(err)
	}
	gdnRun, shortRun := plan.runs[recurrentGeometry(qwen4MambaGraph(qwen4, short).LayerKinds[0].Nodes[0])], plan.runs[recurrentGeometry(short)]
	// Gated DeltaNet at tp 8: conv 10240/8 x 3 bf16, temporal 48/8 x 128 x 128 fp32. The short
	// convolution is replicated: 10240 x (3 x 3) bf16.
	if gdnRun.StateBytes != 400896 || shortRun.StateBytes != 184320 {
		t.Errorf("state bytes: gdn %d, short conv %d; want 400896 and 184320", gdnRun.StateBytes, shortRun.StateBytes)
	}
	if gdnRun.PageSizePadded != shortRun.PageSizePadded || gdnRun.BlockSize != shortRun.BlockSize || gdnRun.PageSizePadded == 0 {
		t.Errorf("the two layouts must share one padded page and block: %+v %+v", gdnRun, shortRun)
	}
	if r, err := plan.mambaRun("ple", short); err != nil || r.StateBytes != 184320 {
		t.Errorf("the short-convolution layer's run: %+v, %v", r, err)
	}
	// A speculative row widens the short convolution's window by one row of the channels.
	e.Speculative = &engineconfig.Speculative{Method: "mtp", NumSpecTokens: 1}
	plan, err = planMamba(qwen4MambaGraph(qwen4, short), e, mambaEnv{tp: 8, dcp: 1, pcp: 1, capability: 9, maxModelLen: 262144})
	if err != nil {
		t.Fatal(err)
	}
	if got := plan.runs[recurrentGeometry(short)].StateBytes; got != 10240*(9+1)*2 {
		t.Errorf("short conv state with one speculative token: %d, want %d", got, 10240*10*2)
	}
	// Another architecture's second recurrent geometry is still not mapped.
	if _, err := planMamba(qwen4MambaGraph("Qwen3_5MoeForConditionalGeneration", short), engineconfig.Engine{BlockSize: 64},
		mambaEnv{tp: 8, dcp: 1, pcp: 1, capability: 9, maxModelLen: 262144}); err == nil {
		t.Error("a short convolution beside Gated DeltaNet was mapped for another architecture")
	}
}

// qwen4ExpScenario is a one-pool vLLM scenario for a catalog model on h200 at tp, written to a
// directory of its own; cacheDType "" leaves it unstated.
func qwen4ExpScenario(t *testing.T, modelName string, tp int, cacheDType string) (string, Repos) {
	t.Helper()
	dtype := ""
	if cacheDType != "" {
		dtype = "\n      kv_cache_dtype: " + cacheDType
	}
	nodes := strconv.Itoa((tp + 7) / 8)
	src := `kind: Scenario
name: s
engine: {kind: vllm}
model: ` + modelName + `
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]
cluster: {hardware: h200, fabric: ib-400g, nodes: ` + nodes + `, gpus_per_node: 8}
---
kind: Deployment
name: s
pools:
  - role: colocated
    nodes: ` + nodes + `
    parallel: {tp: ` + strconv.Itoa(tp) + `, pp: 1, dp: 1, enable_expert_parallel: false}
    engine:
      block_size: 16
      max_num_batched_tokens: 8192
      max_num_seqs: 256
      max_model_len: 4096
      gpu_memory_utilization: 0.9
      compilation_config: {cudagraph_mode: PIECEWISE}` + dtype + "\n"
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "s.yaml"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	r := repos()
	r.Scenarios = dir
	return "s.yaml", r
}

// Every cache a Qwen4Exp layer registers states its own element size -- the main K/V cache's is
// the dtype the QSA layer reads (cache_dtype, qsa.py:361-368), the indexer's two are bf16 or
// indexer_kv_dtype -- so the pool's page bytes are the specs' and none is derived from the memory
// model's bytes, which count the graph's stated caches at a width the TP split does not give
// (blis-vllm#310). Laws at every tensor-parallel size the model shards to: it opens, no cache is
// left without an element size, the main cache's follows cache_dtype, and the pool's bytes a
// block do not depend on the memory model's page.
func TestQwen4ExpPoolPageIsTheSpecsAndNotTheMemoryModels(t *testing.T) {
	for _, c := range []struct {
		dtype string
		elem  int
	}{{"", 2}, {"bfloat16", 2}, {"fp8", 1}, {"fp8_e4m3", 1}} {
		for _, tp := range []int{2, 4, 8, 16} {
			scen, r := qwen4ExpScenario(t, "qwen3.8-flash-next", tp, c.dtype)
			m, err := Open(scen, r)
			if err != nil {
				t.Fatalf("tp %d cache_dtype %q: %v", tp, c.dtype, err)
			}
			runs, err := m.KVLayers()
			if err != nil {
				t.Fatalf("tp %d cache_dtype %q: %v", tp, c.dtype, err)
			}
			full := 0
			vr := make([]vllm.KVLayerRun, len(runs))
			for i, run := range runs {
				vr[i] = vllm.KVLayerRun(run)
				if run.Spec == "mamba" {
					continue
				}
				if run.ElemBytes == 0 {
					t.Errorf("tp %d cache_dtype %q: a %s run states no element size: %+v", tp, c.dtype, run.Spec, run)
				}
				if run.Spec == "full" {
					full++
					if run.ElemBytes != c.elem {
						t.Errorf("tp %d cache_dtype %q: the main cache's element is %d B, want %d", tp, c.dtype, run.ElemBytes, c.elem)
					}
				}
			}
			if full == 0 {
				t.Fatalf("tp %d: no main cache among %+v", tp, runs)
			}
			var per [2]int64
			for i, page := range []int64{1 << 20, 3<<20 + 7} {
				if per[i], err = vllm.KVPoolBytesPerBlock(vr, false, m.engine.BlockSize, page); err != nil {
					t.Fatalf("tp %d cache_dtype %q: memory model page %d: %v", tp, c.dtype, page, err)
				}
			}
			if per[0] != per[1] || per[0] <= 0 {
				t.Errorf("tp %d cache_dtype %q: the pool's bytes a block follow the memory model's page: %d and %d", tp, c.dtype, per[0], per[1])
			}
		}
	}
}
