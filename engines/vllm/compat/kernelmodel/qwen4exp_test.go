// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/models/qwen4_exp/nvidia/qsa.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package kernelmodel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
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
// its backend's smallest kernel block is 16, not the FlashAttention-4 block that head size 256
// would otherwise leave undecided (qsa.py:188-190). Without this it was refused as unmapped.
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
	// Plain GQA of the same shape still leaves its kernel block undecided.
	main.AttentionKind = model.AttentionGQA
	main.Caches = nil
	g.LayerKinds[1].Nodes = []model.Node{main}
	if f, _ := attentionOf(g, engineconfig.Engine{BlockSize: 64, CacheDType: "auto"}, 8); f.kernelErr == nil {
		t.Error("head size 256 on plain GQA lost its kernel-block caveat")
	}
}
