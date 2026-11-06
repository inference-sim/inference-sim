package model

import (
	"strings"
	"testing"
)

// validGraph is a Granite-shaped uniform MoE stack: 224 experts, GQA attention,
// one layer kind repeated. Each negative test mutates one field of a copy, so a
// failure names exactly the rule that fired rather than a document that was wrong
// in several ways at once.
func validGraph() *Graph {
	return &Graph{
		Kind: "ModelGraph",
		Name: "granite-5-230b",
		DerivedFrom: Derivation{
			Format: "hf_config_json", Path: "config.json",
			SHA256:         "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
			DeriverVersion: 1,
		},
		Global: GlobalShape{HiddenSize: 3072, VocabSize: 100352, WeightDType: DTypeFP8},
		LayerKinds: []LayerKind{{
			ID: "attn_moe",
			Nodes: []Node{
				{Op: OpElementwise, Role: "input_norm"},
				{Op: OpGEMM, Role: "qkv_proj", N: 4096, K: 3072},
				{Op: OpAttention, AttentionKind: AttentionGQA, NumQHeads: 48, NumKVHeads: 8, HeadDim: 64},
				{Op: OpGEMM, Role: "o_proj", N: 3072, K: 3072},
				{Op: OpAllReduce, Role: "attn_out", Emit: EmitTensorParallel},
				{Op: OpGroupedGEMM, Role: "experts", N: 1536, K: 3072, Experts: 224, TopK: 8},
				{Op: OpAll2All, Role: "moe", Emit: EmitExpertParallel},
			},
			Edges: [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}},
		}},
		Stack: Stack{Pattern: []string{"attn_moe"}, Repeat: 72},
	}
}

func TestValidGraphPasses(t *testing.T) {
	if p := validGraph().Validate(); !p.OK() {
		t.Fatalf("a valid graph was rejected:\n%s", p.Error())
	}
}

func TestStackLayers(t *testing.T) {
	if got := validGraph().Stack.Layers(); got != 72 {
		t.Errorf("Layers() = %d, want 72", got)
	}
	hybrid := Stack{Pattern: []string{"a", "a", "a", "b"}, Repeat: 10}
	if got := hybrid.Layers(); got != 40 {
		t.Errorf("hybrid Layers() = %d, want 40", got)
	}
}

// TestRejects is the negative suite. A validator that cannot fail proves nothing,
// so every check above has a case here that provokes it.
func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Graph)
	}{
		{"wrong kind", func(g *Graph) { g.Kind = "Model" }},
		{"no name", func(g *Graph) { g.Name = "" }},
		{"short digest", func(g *Graph) { g.DerivedFrom.SHA256 = "abc" }},
		{"no derivation format", func(g *Graph) { g.DerivedFrom.Format = "" }},
		{"zero deriver version", func(g *Graph) { g.DerivedFrom.DeriverVersion = 0 }},
		{"zero hidden size", func(g *Graph) { g.Global.HiddenSize = 0 }},
		{"negative pipeline boundary", func(g *Graph) { g.Global.PipelineBoundaryBytesPerToken = -1 }},
		{"unknown weight dtype", func(g *Graph) { g.Global.WeightDType = "fp6" }},
		{"no layer kinds", func(g *Graph) { g.LayerKinds = nil }},
		{"duplicate layer-kind id", func(g *Graph) {
			g.LayerKinds = append(g.LayerKinds, g.LayerKinds[0])
		}},
		{"pattern names unknown kind", func(g *Graph) {
			g.Stack.Pattern = []string{"nope"}
		}},
		{"zero repeat", func(g *Graph) { g.Stack.Repeat = 0 }},
		{"unknown op", func(g *Graph) { g.LayerKinds[0].Nodes[0].Op = "Magic" }},
		{"edge out of range", func(g *Graph) {
			g.LayerKinds[0].Edges = append(g.LayerKinds[0].Edges, [2]int{0, 99})
		}},
		{"self edge", func(g *Graph) {
			g.LayerKinds[0].Edges = append(g.LayerKinds[0].Edges, [2]int{2, 2})
		}},
		{"cyclic graph", func(g *Graph) {
			g.LayerKinds[0].Edges = append(g.LayerKinds[0].Edges, [2]int{6, 0})
		}},
		{"collective without condition", func(g *Graph) {
			g.LayerKinds[0].Nodes[4].Emit = EmitAlways
		}},
		{"gemm without n", func(g *Graph) { g.LayerKinds[0].Nodes[1].N = 0 }},
		{"gemm carrying head count", func(g *Graph) {
			g.LayerKinds[0].Nodes[1].NumQHeads = 8
		}},
		{"grouped gemm top_k exceeds experts", func(g *Graph) {
			g.LayerKinds[0].Nodes[5].TopK = 500
		}},
		{"grouped gemm without experts", func(g *Graph) {
			g.LayerKinds[0].Nodes[5].Experts = 0
		}},
		{"attention n_kv exceeds n_q", func(g *Graph) {
			g.LayerKinds[0].Nodes[2].NumKVHeads = 99
		}},
		{"unknown attention kind", func(g *Graph) {
			g.LayerKinds[0].Nodes[2].AttentionKind = "linear"
		}},
		{"mla with more than one kv head", func(g *Graph) {
			g.LayerKinds[0].Nodes[2].AttentionKind = AttentionMLA
			g.LayerKinds[0].Nodes[2].NumKVHeads = 8
		}},
		{"swa without a window", func(g *Graph) {
			g.LayerKinds[0].Nodes[2].AttentionKind = AttentionSWA
		}},
		{"collective carrying a shape", func(g *Graph) {
			g.LayerKinds[0].Nodes[6].N = 1024
		}},
		{"speculator with zero drafts", func(g *Graph) {
			g.Speculator = &Speculator{Method: "glm4_moe_mtp", NumSpec: 0,
				Stack: Stack{Pattern: []string{"attn_moe"}, Repeat: 1}}
		}},
		{"speculator without a stack", func(g *Graph) {
			g.Speculator = &Speculator{Method: "glm4_moe_mtp", NumSpec: 1}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := validGraph()
			tc.mutate(g)
			if p := g.Validate(); p.OK() {
				t.Fatalf("expected rejection, got none")
			}
		})
	}
}

// TestRecurrentNodeRequirements covers the hybrid path separately, since its
// required parameters differ by recurrent kind.
func TestRecurrentNodeRequirements(t *testing.T) {
	mamba := Node{Op: OpRecurrentUpdate, RecurrentKind: RecurrentMamba2,
		NumHeads: 128, StateSize: 128, NumGroups: 8, ConvKernel: 4,
		IntermediateSize: 8192, StateDType: DTypeFP32}
	g := validGraph()
	g.LayerKinds[0].Nodes[2] = mamba
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a well-formed mamba2 node was rejected:\n%s", p.Error())
	}
	// Mamba2 needs its convolution parameters; KDA does not.
	for _, drop := range []func(*Node){
		func(n *Node) { n.NumGroups = 0 },
		func(n *Node) { n.ConvKernel = 0 },
		func(n *Node) { n.IntermediateSize = 0 },
		func(n *Node) { n.StateSize = 0 },
	} {
		g := validGraph()
		n := mamba
		drop(&n)
		g.LayerKinds[0].Nodes[2] = n
		if p := g.Validate(); p.OK() {
			t.Errorf("expected rejection of an incomplete mamba2 node")
		}
	}
	kda := Node{Op: OpRecurrentUpdate, RecurrentKind: RecurrentKDA,
		NumHeads: 64, StateSize: 128}
	g2 := validGraph()
	g2.LayerKinds[0].Nodes[2] = kda
	if p := g2.Validate(); !p.OK() {
		t.Fatalf("a KDA node without convolution parameters was rejected:\n%s", p.Error())
	}
}

// CompressRatio is the sparse-MLA latent-read compression factor DeepSeek-V4-Pro's
// csa4_moe/csa128_moe layers carry. It is valid only on a sparse_mla attention node
// and must be positive, the same way the other MLA-specific parameters are gated to
// their kind. These check it is governed rather than merely accepted.
func TestCompressRatio(t *testing.T) {
	// A sparse_mla node that already satisfies its kind's other requirements, so each
	// case below isolates the compress_ratio rule rather than tripping another.
	sparse := func() Node {
		return Node{Op: OpAttention, AttentionKind: AttentionSparseMLA,
			NumQHeads: 128, NumKVHeads: 1, HeadDim: 512, IndexTopK: 1024}
	}
	with := func(n Node, r int) Node { n.CompressRatio = r; return n }
	for _, tc := range []struct {
		name string
		node Node
		ok   bool
	}{
		{"positive on a sparse_mla node is valid", with(sparse(), 4), true},
		{"absent is valid when index_topk carries the sparsity", sparse(), true},
		{"a negative value on a sparse_mla node is rejected", with(sparse(), -1), false},
		{"an explicit zero is treated as absent (omitempty), so index_topk still carries it",
			with(sparse(), 0), true},
		// A sparse_mla layer must do at least one of top-k selection or latent
		// compression; DeepSeek-V4-Pro ships one layer kind of each.
		{"compress_ratio alone satisfies a sparse_mla node (csa128_moe)",
			Node{Op: OpAttention, AttentionKind: AttentionSparseMLA,
				NumQHeads: 128, NumKVHeads: 1, HeadDim: 512, CompressRatio: 128}, true},
		{"index_topk alone satisfies a sparse_mla node (csa4_moe)",
			Node{Op: OpAttention, AttentionKind: AttentionSparseMLA,
				NumQHeads: 128, NumKVHeads: 1, HeadDim: 512, IndexTopK: 1024}, true},
		{"a sparse_mla node with neither is rejected",
			Node{Op: OpAttention, AttentionKind: AttentionSparseMLA,
				NumQHeads: 128, NumKVHeads: 1, HeadDim: 512}, false},
		// A positive compress_ratio must not mask a negative index_topk: a negative
		// count is a mistake, not a compress-only layer.
		{"a negative index_topk is rejected even with a valid compress_ratio",
			Node{Op: OpAttention, AttentionKind: AttentionSparseMLA,
				NumQHeads: 128, NumKVHeads: 1, HeadDim: 512, IndexTopK: -5, CompressRatio: 4}, false},
		{"on a non-sparse_mla attention node is rejected",
			Node{Op: OpAttention, AttentionKind: AttentionGQA,
				NumQHeads: 48, NumKVHeads: 8, HeadDim: 64, CompressRatio: 4}, false},
		{"a set (nonzero) value on a node whose op does not price it is rejected",
			Node{Op: OpGEMM, N: 4096, K: 3072, CompressRatio: 4}, false},
		// omitempty makes a zero indistinguishable from an absent field, so an explicit
		// compress_ratio: 0 is an unset value and does not make a GEMM invalid -- the same
		// convention index_topk and every other optional shape field follows.
		{"an unset (zero) value on a node whose op does not price it is accepted",
			Node{Op: OpGEMM, N: 4096, K: 3072, CompressRatio: 0}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := validGraph()
			g.LayerKinds[0].Nodes[2] = tc.node
			p := g.Validate()
			if got := p.OK(); got != tc.ok {
				t.Fatalf("OK() = %v, want %v: %v", got, tc.ok, p.Error())
			}
		})
	}
}

// A node that is both the wrong op for compress_ratio and carries a non-positive
// value must report both faults rather than masking one, matching the validator's
// report-every-problem contract.
func TestCompressRatioAccumulatesProblems(t *testing.T) {
	g := validGraph()
	g.LayerKinds[0].Nodes[2] = Node{Op: OpGEMM, N: 4096, K: 3072, CompressRatio: -1}
	var positive, wrongKind bool
	for _, e := range g.Validate().Errors() {
		if e.Path != "layer_kinds[0].nodes[2].compress_ratio" {
			continue
		}
		switch {
		case strings.Contains(e.Message, "must be positive"):
			positive = true
		case strings.Contains(e.Message, "only priced by"):
			wrongKind = true
		}
	}
	if !positive || !wrongKind {
		t.Fatalf("want both the positivity and the wrong-op problems, got positive=%v wrongKind=%v:\n%s",
			positive, wrongKind, g.Validate().Error())
	}
}

// index_topk, like compress_ratio, selects over the latent cache, so it is valid only on
// a sparse_mla attention node and rejected on any other op or kind.
func TestIndexTopKGating(t *testing.T) {
	for _, tc := range []struct {
		name string
		node Node
		ok   bool
	}{
		{"index_topk on a sparse_mla node is valid",
			Node{Op: OpAttention, AttentionKind: AttentionSparseMLA,
				NumQHeads: 128, NumKVHeads: 1, HeadDim: 512, IndexTopK: 1024}, true},
		{"index_topk on a non-sparse_mla attention node is rejected",
			Node{Op: OpAttention, AttentionKind: AttentionGQA,
				NumQHeads: 48, NumKVHeads: 8, HeadDim: 64, IndexTopK: 1024}, false},
		{"index_topk on a node whose op does not price it is rejected",
			Node{Op: OpGEMM, N: 4096, K: 3072, IndexTopK: 1024}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := validGraph()
			g.LayerKinds[0].Nodes[2] = tc.node
			p := g.Validate()
			if got := p.OK(); got != tc.ok {
				t.Fatalf("OK() = %v, want %v: %v", got, tc.ok, p.Error())
			}
		})
	}
}

// A layer kind the speculator's own stack uses is intentional, not a leftover, so it must
// not draw the "declared but never used" warning; a kind used by neither stack still does.
func TestSpeculatorLayerKindCountsAsUsed(t *testing.T) {
	warned := func(g *Graph, id string) bool {
		for _, it := range g.Validate().All() {
			if strings.Contains(it.Message, `layer kind "`+id+`" is declared but never used`) {
				return true
			}
		}
		return false
	}
	// Start from the valid single-kind graph, add a kind only the speculator references.
	base := func() *Graph {
		g := validGraph()
		draft := g.LayerKinds[0]
		draft.ID = "draft"
		g.LayerKinds = append(g.LayerKinds, draft)
		g.Speculator = &Speculator{Method: "glm4_moe_mtp", NumSpec: 1,
			Stack: Stack{Pattern: []string{"draft"}, Repeat: 1}}
		return g
	}
	if warned(base(), "draft") {
		t.Errorf("a layer kind used only by the speculator was flagged as unused")
	}
	// A kind used by neither the main stack nor the speculator is still flagged.
	g := base()
	orphan := g.LayerKinds[0]
	orphan.ID = "orphan"
	g.LayerKinds = append(g.LayerKinds, orphan)
	if !warned(g, "orphan") {
		t.Errorf("a genuinely unused layer kind was not flagged")
	}
}

func TestLatentKVAndDTypeBytes(t *testing.T) {
	for _, k := range []AttentionKind{AttentionMLA, AttentionSparseMLA} {
		if !k.LatentKV() {
			t.Errorf("%s should report a latent KV cache", k)
		}
	}
	for _, k := range []AttentionKind{AttentionGQA, AttentionSWA} {
		if k.LatentKV() {
			t.Errorf("%s should not report a latent KV cache", k)
		}
	}
	for d, want := range map[DType]float64{
		DTypeFP32: 4, DTypeBF16: 2, DTypeFP8: 1, DTypeNVFP4: 0.5,
	} {
		if got := d.Bytes(); got != want {
			t.Errorf("%s.Bytes() = %v, want %v", d, got, want)
		}
	}
}

// A per-node weight dtype exists for mixed-precision MoE: DeepSeek-V4-Pro stores its routed
// experts at fp4 beside fp8 everywhere else. These check the field is governed rather than
// merely accepted -- an unrecognized width that fell back to the global one would reinstate
// the bug the field was added to fix.
func TestPerNodeWeightDType(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*Graph)
		ok   bool
	}{
		{"absent is valid: the global dtype governs", func(g *Graph) {}, true},
		{"a recognized dtype on a GroupedGEMM is valid", func(g *Graph) {
			for i := range g.LayerKinds[0].Nodes {
				if g.LayerKinds[0].Nodes[i].Op == OpGroupedGEMM {
					g.LayerKinds[0].Nodes[i].WeightDType = DTypeNVFP4
				}
			}
		}, true},
		{"an unrecognized dtype is rejected", func(g *Graph) {
			for i := range g.LayerKinds[0].Nodes {
				if g.LayerKinds[0].Nodes[i].Op == OpGroupedGEMM {
					g.LayerKinds[0].Nodes[i].WeightDType = "fp6"
				}
			}
		}, false},
		{"a dtype on a node holding no parameters is rejected", func(g *Graph) {
			for i := range g.LayerKinds[0].Nodes {
				if g.LayerKinds[0].Nodes[i].Op == OpAttention {
					g.LayerKinds[0].Nodes[i].WeightDType = DTypeFP8
				}
			}
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := validGraph()
			tc.set(g)
			p := g.Validate()
			if got := p.OK(); got != tc.ok {
				t.Fatalf("OK() = %v, want %v: %v", got, tc.ok, p.Error())
			}
		})
	}
}

// A Muse-shaped MoE states one reduction per mode: the per-route rows under tensor
// parallelism, the combined output when expert parallelism keeps every token on every rank.
// Both conditions are recognized, and the boundary bytes of an fp32 residual stream pass.
func TestModeConditionedMoEReductionsValidate(t *testing.T) {
	g := validGraph()
	g.Global.PipelineBoundaryBytesPerToken = 4 * g.Global.HiddenSize
	kind := &g.LayerKinds[0]
	kind.Nodes = append(kind.Nodes,
		Node{Op: OpAllReduce, Role: "moe_route_rows", Emit: EmitTensorParallelWithoutExpertParallel,
			BytesPerToken: 8 * 4096 * 2},
		Node{Op: OpAllReduce, Role: "moe_combine", Emit: EmitExpertParallelWithoutDispatch,
			BytesPerToken: 4096 * 2})
	kind.Edges = append(kind.Edges, [2]int{6, 7}, [2]int{7, 8})
	if p := g.Validate(); !p.OK() {
		t.Fatalf("expected a valid graph, got: %v", p.All())
	}
	for _, c := range []EmitCondition{EmitTensorParallelWithoutExpertParallel, EmitExpertParallelWithoutDispatch} {
		if !c.Valid() {
			t.Errorf("%q is not recognized", c)
		}
	}
}

// A speculator's auxiliary hidden-state layers are capture points in the target's stack: not
// negative, and stated as the engine holds them, sorted without repeats.
func TestSpeculatorAuxHiddenStateLayers(t *testing.T) {
	with := func(ids ...int) *Graph {
		g := validGraph()
		g.Speculator = &Speculator{Method: "eagle3", NumSpec: 3,
			Stack: Stack{Pattern: []string{"attn_moe"}, Repeat: 1}, AuxHiddenStateLayers: ids}
		return g
	}
	if p := with(2, 36, 69).Validate(); !p.OK() {
		t.Fatalf("valid aux layers rejected:\n%s", p.Error())
	}
	if p := with().Validate(); !p.OK() {
		t.Fatalf("a speculator with no aux layers rejected:\n%s", p.Error())
	}
	for name, ids := range map[string][]int{
		"negative": {-1, 3}, "repeated": {3, 3}, "descending": {9, 3},
	} {
		p := with(ids...).Validate()
		if p.OK() || !strings.Contains(p.Error(), "speculator.aux_hidden_state_layers") {
			t.Errorf("%s ids %v: want a speculator.aux_hidden_state_layers problem, got %v", name, ids, p)
		}
	}
	wide := with(2, 9)
	wide.Speculator.AuxHiddenStateBytesPerToken = 4 * 3072
	if p := wide.Validate(); !p.OK() {
		t.Errorf("a stated capture width rejected:\n%s", p.Error())
	}
	for name, g := range map[string]*Graph{"negative": with(2), "without layers": with()} {
		g.Speculator.AuxHiddenStateBytesPerToken = 6144
		if name == "negative" {
			g.Speculator.AuxHiddenStateBytesPerToken = -1
		}
		p := g.Validate()
		if p.OK() || !strings.Contains(p.Error(), "speculator.aux_hidden_state_bytes_per_token") {
			t.Errorf("capture width %s: want a speculator.aux_hidden_state_bytes_per_token problem, got %v", name, p)
		}
	}
}

// ForCapability's copy owns its aux layer list, as it owns every other slice of the graph.
func TestForCapabilityCopiesAuxHiddenStateLayers(t *testing.T) {
	g := validGraph()
	g.Speculator = &Speculator{Method: "eagle3", NumSpec: 3,
		Stack: Stack{Pattern: []string{"attn_moe"}, Repeat: 1}, AuxHiddenStateLayers: []int{2, 9}}
	out, err := g.ForCapability(9)
	if err != nil {
		t.Fatal(err)
	}
	out.Speculator.AuxHiddenStateLayers[0] = 5
	if g.Speculator.AuxHiddenStateLayers[0] != 2 {
		t.Error("the resolved graph shares the source graph's aux layer list")
	}
}

// kv_heads_not_replicated belongs on a non-latent attention node only.
func TestKVHeadsNotReplicatedPlacement(t *testing.T) {
	g := validGraph()
	g.LayerKinds[0].Nodes[2].KVHeadsNotReplicated = true
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a GQA node stating it was rejected:\n%s", p.Error())
	}
	for name, mutate := range map[string]func(*Graph){
		"on a GEMM": func(g *Graph) { g.LayerKinds[0].Nodes[1].KVHeadsNotReplicated = true },
		"on MLA": func(g *Graph) {
			n := &g.LayerKinds[0].Nodes[2]
			n.AttentionKind, n.NumKVHeads, n.KVHeadsNotReplicated = AttentionMLA, 1, true
		},
	} {
		g := validGraph()
		mutate(g)
		if p := g.Validate(); p.OK() || !strings.Contains(p.Error(), "kv_heads_not_replicated") {
			t.Errorf("%s: want a kv_heads_not_replicated problem, got %v", name, p)
		}
	}
}

// sharded_at_decode belongs on a replicated GEMM, with positive bounds.
func TestShardedAtDecodePlacement(t *testing.T) {
	ok := &ShardedAtDecode{CapabilityFamily: 10, MaxTokens: 256, MinTokensAcrossNodes: 32}
	g := validGraph()
	g.LayerKinds[0].Nodes[1].TP, g.LayerKinds[0].Nodes[1].ShardedAtDecode = TPReplicated, ok
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a replicated GEMM stating it was rejected:\n%s", p.Error())
	}
	for name, mutate := range map[string]func(*Node){
		"column GEMM":   func(n *Node) { n.TP = TPColumn },
		"no family":     func(n *Node) { n.ShardedAtDecode.CapabilityFamily = 0 },
		"no max":        func(n *Node) { n.ShardedAtDecode.MaxTokens = 0 },
		"min above max": func(n *Node) { n.ShardedAtDecode.MinTokensAcrossNodes = 300 },
		"negative min":  func(n *Node) { n.ShardedAtDecode.MinTokensAcrossNodes = -1 },
	} {
		g := validGraph()
		n := &g.LayerKinds[0].Nodes[1]
		d := *ok
		n.TP, n.ShardedAtDecode = TPReplicated, &d
		mutate(n)
		if p := g.Validate(); p.OK() || !strings.Contains(p.Error(), "sharded_at_decode") {
			t.Errorf("%s: want a sharded_at_decode problem, got %v", name, p)
		}
	}
	g = validGraph()
	g.LayerKinds[0].Nodes[2].ShardedAtDecode = ok
	if p := g.Validate(); p.OK() {
		t.Error("an Attention node stating sharded_at_decode was accepted")
	}
}
