package model

import (
	"gopkg.in/yaml.v3"
	"strings"
	"testing"
)

// Qwen3.8-Flash-Next's full-attention layer is sparse grouped-query attention: a main node
// that reads a budget of selected tokens of per-KV-head K and V, and an indexer scorer over
// a key cache compressed by a ratio. Its per-layer embedding branch is a dilated depthwise
// convolution with per-request state and no recurrent state. These tests state what the
// schema accepts and refuses for each, so a graph cannot price one node's read as the
// other's.

func sparseGQAMain() Node {
	return Node{Op: OpAttention, HoldsEngineKV: Bool(true), AttentionKind: AttentionSparseGQA,
		NumQHeads: 24, NumKVHeads: 2, HeadDim: 256, IndexTopK: 2048,
		Caches: []Cache{{Name: "kv", Kind: CacheKV, Owned: true, BytesPerEntry: 2048}}}
}

func sparseGQAScorer() Node {
	return Node{Op: OpAttention, HoldsEngineKV: Bool(false), AttentionKind: AttentionSparseGQA,
		Role: "indexer_scores", NumQHeads: 4, NumKVHeads: 1, HeadDim: 128, CompressRatio: 4,
		Caches: []Cache{
			{Name: "index", Kind: CacheIndex, Owned: true, BytesPerEntry: 128, TokensPerEntry: 4},
			{Name: "ring", Kind: CacheState, Owned: true, BytesPerEntry: 256, Entries: 8},
		}}
}

func shortConv() Node {
	return Node{Op: OpRecurrentUpdate, RecurrentKind: RecurrentShortConv, Role: "ple_conv",
		ConvKernel: 4, ConvDilation: 3, IntermediateSize: 8192}
}

// validateNode places n as the third node of the base graph's first layer kind, so each
// case isolates the rule it names rather than another node's requirement.
func validateNode(t *testing.T, n Node) string {
	t.Helper()
	g := validGraph()
	g.LayerKinds[0].Nodes[2] = n
	p := g.Validate()
	if p.OK() {
		return ""
	}
	return p.Error()
}

func TestSparseGQAAttention(t *testing.T) {
	with := func(n Node, f func(*Node)) Node { f(&n); return n }
	for _, tc := range []struct {
		name string
		node Node
		bad  string // substring of the refusal; empty means valid
	}{
		{"the main node with a budget is valid", sparseGQAMain(), ""},
		{"the scorer with a ratio is valid", sparseGQAScorer(), ""},
		{"the main node without a budget is refused",
			with(sparseGQAMain(), func(n *Node) { n.IndexTopK = 0 }), "index_topk"},
		{"a negative budget is refused",
			with(sparseGQAMain(), func(n *Node) { n.IndexTopK = -1 }), "index_topk"},
		{"the scorer without a ratio is refused",
			with(sparseGQAScorer(), func(n *Node) { n.CompressRatio = 0 }), "compress_ratio"},
		{"a negative ratio is refused",
			with(sparseGQAScorer(), func(n *Node) { n.CompressRatio = -4 }), "compress_ratio"},
		{"the main node with a ratio is refused (the compressed cache is the scorer's)",
			with(sparseGQAMain(), func(n *Node) { n.CompressRatio = 4 }), "compress_ratio"},
		{"the scorer with a budget is refused (the budget is the main node's)",
			with(sparseGQAScorer(), func(n *Node) { n.IndexTopK = 2048 }), "index_topk"},
		{"gqa with a budget is refused",
			Node{Op: OpAttention, HoldsEngineKV: Bool(true), AttentionKind: AttentionGQA,
				NumQHeads: 24, NumKVHeads: 2, HeadDim: 256, IndexTopK: 2048}, "index_topk"},
		{"gqa with a ratio is refused",
			Node{Op: OpAttention, HoldsEngineKV: Bool(true), AttentionKind: AttentionGQA,
				NumQHeads: 24, NumKVHeads: 2, HeadDim: 256, CompressRatio: 4}, "compress_ratio"},
		{"more KV heads than query heads is refused",
			with(sparseGQAMain(), func(n *Node) { n.NumKVHeads = 48 }), "n_kv"},
		{"unlike sparse_mla, several KV heads are valid (K and V are per head)",
			with(sparseGQAMain(), func(n *Node) { n.NumKVHeads = 8 }), ""},
		{"the index cache must hold one entry per ratio tokens",
			with(sparseGQAScorer(), func(n *Node) { n.Caches[0].TokensPerEntry = 8 }), "tokens_per_entry"},
		{"the main node's KV cache holds one entry per token",
			with(sparseGQAMain(), func(n *Node) { n.Caches[0].TokensPerEntry = 4 }), "tokens_per_entry"},
		{"a KV head that cannot be replicated across ranks is stated as for gqa",
			with(sparseGQAMain(), func(n *Node) { n.KVHeadsNotReplicated = true }), ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := validateNode(t, tc.node)
			switch {
			case tc.bad == "" && got != "":
				t.Fatalf("refused:\n%s", got)
			case tc.bad != "" && got == "":
				t.Fatalf("accepted, want a refusal naming %q", tc.bad)
			case tc.bad != "" && !strings.Contains(got, tc.bad):
				t.Fatalf("refusal does not name %q:\n%s", tc.bad, got)
			}
		})
	}
}

func TestShortConvRecurrent(t *testing.T) {
	with := func(n Node, f func(*Node)) Node { f(&n); return n }
	for _, tc := range []struct {
		name string
		node Node
		bad  string
	}{
		{"a dilated short conv is valid", shortConv(), ""},
		{"an undilated one is valid (dilation absent means one)",
			with(shortConv(), func(n *Node) { n.ConvDilation = 0 }), ""},
		{"it needs a kernel", with(shortConv(), func(n *Node) { n.ConvKernel = 0 }), "conv_kernel"},
		{"it needs channels", with(shortConv(), func(n *Node) { n.IntermediateSize = 0 }), "intermediate_size"},
		{"a negative dilation is refused",
			with(shortConv(), func(n *Node) { n.ConvDilation = -1 }), "conv_dilation"},
		{"it keeps no state size", with(shortConv(), func(n *Node) { n.StateSize = 128 }), "state_size"},
		{"it has no heads", with(shortConv(), func(n *Node) { n.NumHeads = 16 }), "n_heads"},
		{"it has no groups", with(shortConv(), func(n *Node) { n.NumGroups = 1 }), "n_groups"},
		{"a gdn node cannot carry a dilation",
			Node{Op: OpRecurrentUpdate, RecurrentKind: RecurrentGDN, NumHeads: 48, StateSize: 128,
				ConvKernel: 4, ConvDilation: 3}, "conv_dilation"},
		{"a GEMM cannot carry a dilation",
			Node{Op: OpGEMM, N: 4096, K: 3072, ConvDilation: 3}, "conv_dilation"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := validateNode(t, tc.node)
			switch {
			case tc.bad == "" && got != "":
				t.Fatalf("refused:\n%s", got)
			case tc.bad != "" && got == "":
				t.Fatalf("accepted, want a refusal naming %q", tc.bad)
			case tc.bad != "" && !strings.Contains(got, tc.bad):
				t.Fatalf("refusal does not name %q:\n%s", tc.bad, got)
			}
		})
	}
}

// The documents the deriver writes spell these nodes in YAML. The same nodes, read from
// that spelling, validate; and the spelling of a wrong one is refused with the field named.
func TestQwen4ExpNodesFromYAML(t *testing.T) {
	const doc = `
- {op: Attention, role: attn, holds_engine_kv: true, kind: sparse_gqa, n_q: 24, n_kv: 2, d_h: 256, index_topk: 2048}
- {op: Attention, role: indexer_scores, holds_engine_kv: false, kind: sparse_gqa, n_q: 4, n_kv: 1, d_h: 128, compress_ratio: 4}
- {op: RecurrentUpdate, role: ple_conv, recurrent_kind: short_conv, conv_kernel: 4, conv_dilation: 3, intermediate_size: 8192}
`
	var nodes []Node
	if err := yaml.Unmarshal([]byte(doc), &nodes); err != nil {
		t.Fatal(err)
	}
	g := validGraph()
	g.LayerKinds[0].Nodes = append(g.LayerKinds[0].Nodes, nodes...)
	if p := g.Validate(); !p.OK() {
		t.Fatalf("the Qwen4Exp nodes were refused:\n%s", p.Error())
	}
	nodes[0].IndexTopK = 0
	g = validGraph()
	g.LayerKinds[0].Nodes = append(g.LayerKinds[0].Nodes, nodes...)
	if p := g.Validate(); p.OK() || !strings.Contains(p.Error(), "index_topk") {
		t.Fatalf("a sparse_gqa main node without a budget was accepted or unnamed: %v", p.Error())
	}
}
