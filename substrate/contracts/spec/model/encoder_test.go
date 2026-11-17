package model

import (
	"strings"
	"testing"
)

// kimiTower is Kimi-K2.5's MoonViT as vLLM v0.33.0 builds it (kimi_k25_vit.py): 27 layers
// of 1152 width, 16 heads of 72, a 4304-wide plain MLP, a 2x2 merge and a 7168-wide projector.
func kimiTower() Tower {
	return Tower{
		Modality:         TowerImage,
		Input:            TowerInput{PatchSize: 14, Channels: 3},
		MergeRatio:       4,
		OutputHiddenSize: 7168,
		Parallel:         EncoderTensor,
		WeightDType:      DTypeBF16,
		Embedding: []EncoderNode{
			{Op: EncoderGEMM, Role: "patch_embed", N: 1152, K: 588, Bias: true, TP: TPReplicated},
			{Op: EncoderElementwise, Role: "pos_embed", BytesPerRow: 4 * 1152 * 2, Params: 64 * 64 * 1152},
		},
		LayerKinds: []EncoderLayerKind{{ID: "block", Nodes: []EncoderNode{
			{Op: EncoderElementwise, Role: "norms", BytesPerRow: 4 * 1152 * 2, Params: 2 * 2 * 1152},
			{Op: EncoderGEMM, Role: "qkv", N: 3456, K: 1152, Bias: true, TP: TPColumn},
			{Op: EncoderAttention, NumHeads: 16, HeadDim: 72, TP: TPSharded},
			{Op: EncoderGEMM, Role: "o", N: 1152, K: 1152, Bias: true, TP: TPRow},
			{Op: EncoderGEMM, Role: "fc1", N: 4304, K: 1152, Bias: true, TP: TPColumn},
			{Op: EncoderGEMM, Role: "fc2", N: 1152, K: 4304, Bias: true, TP: TPRow},
		}}},
		Stack: Stack{Pattern: []string{"block"}, Repeat: 27},
		Projector: []EncoderNode{
			{Op: EncoderElementwise, Role: "final_norm", BytesPerRow: 2 * 1152 * 2, Params: 2 * 1152},
			{Op: EncoderGEMM, Role: "merge_fc1", N: 4608, K: 4608, Bias: true, Rows: RowsOutputTokens},
			{Op: EncoderGEMM, Role: "merge_fc2", N: 7168, K: 4608, Bias: true, Rows: RowsOutputTokens},
		},
	}
}

func multimodalGraph(t Tower) *Graph {
	return &Graph{
		Kind: "ModelGraph", Name: "vl", Modality: ModalityMultimodal,
		DerivedFrom: Derivation{Format: "hf_config_json", Path: "config.json", SHA256: repeat64('a'), DeriverVersion: 1},
		Global:      GlobalShape{HiddenSize: 7168, VocabSize: 163840, WeightDType: DTypeBF16},
		LayerKinds: []LayerKind{{ID: "block", Nodes: []Node{
			{Op: OpAttention, HoldsEngineKV: Bool(true), AttentionKind: AttentionGQA, NumQHeads: 64, NumKVHeads: 8, HeadDim: 128},
		}}},
		Stack:   Stack{Pattern: []string{"block"}, Repeat: 4},
		Encoder: &Encoder{Towers: []Tower{t}},
	}
}

func TestEncoderGraphValidates(t *testing.T) {
	g := multimodalGraph(kimiTower())
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a well-formed multimodal graph was rejected:\n%s", p.Error())
	}
	if !g.Modality.PricesEncoder() || !g.PricesEncoder() {
		t.Error("a multimodal graph with a tower does not say it prices an encoder")
	}
}

// Each malformed encoder has a failing example (SUB-2): the mutation, and a word the
// problem must contain, so a rejection for another reason does not pass.
func TestEncoderValidatorRejectsEachMalformation(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(g *Graph)
		want   string
	}{
		{"no stack", func(g *Graph) { g.Encoder.Towers[0].Stack = Stack{} }, "no layers"},
		{"merge ratio zero", func(g *Graph) { g.Encoder.Towers[0].MergeRatio = 0 }, "merge_ratio"},
		{"output width differs from the decoder", func(g *Graph) { g.Encoder.Towers[0].OutputHiddenSize = 4096 }, "output_hidden_size"},
		{"unknown parallel mode", func(g *Graph) { g.Encoder.Towers[0].Parallel = "data" }, "parallel"},
		{"unknown dtype", func(g *Graph) { g.Encoder.Towers[0].WeightDType = "bf17" }, "weight_dtype"},
		{"stack names an undefined kind", func(g *Graph) { g.Encoder.Towers[0].Stack.Pattern = []string{"nope"} }, "nope"},
		{"unused layer kind", func(g *Graph) {
			g.Encoder.Towers[0].LayerKinds = append(g.Encoder.Towers[0].LayerKinds,
				EncoderLayerKind{ID: "extra", Nodes: []EncoderNode{{Op: EncoderElementwise, BytesPerRow: 1}}})
		}, "extra"},
		{"gemm without a width", func(g *Graph) { g.Encoder.Towers[0].LayerKinds[0].Nodes[1].K = 0 }, "k"},
		{"head count on a gemm", func(g *Graph) { g.Encoder.Towers[0].LayerKinds[0].Nodes[1].NumHeads = 16 }, "not priced"},
		{"attention without heads", func(g *Graph) { g.Encoder.Towers[0].LayerKinds[0].Nodes[2].NumHeads = 0 }, "n_heads"},
		{"unknown op", func(g *Graph) { g.Encoder.Towers[0].LayerKinds[0].Nodes[0].Op = "conv" }, "op"},
		{"tile not a patch multiple", func(g *Graph) { g.Encoder.Towers[0].Input.TilePixels = 337 }, "tile_pixels"},
		{"audio tower priced", func(g *Graph) { g.Encoder.Towers[0].Modality = TowerAudio }, "cannot be priced"},
		{"omitted tower without an issue", func(g *Graph) {
			g.Modality = ModalityTextDecoderOfMultimodal
			g.Encoder.Omitted = []OmittedTower{{Modality: TowerAudio, Reason: "depthwise convolution"}}
		}, "issue"},
		{"omitted tower with a malformed issue", func(g *Graph) {
			g.Modality = ModalityTextDecoderOfMultimodal
			g.Encoder.Omitted = []OmittedTower{{Modality: TowerAudio, Reason: "x", Issue: "see the tracker"}}
		}, "issue"},
		{"omitted tower under the multimodal modality", func(g *Graph) {
			g.Encoder.Omitted = []OmittedTower{{Modality: TowerAudio, Reason: "x", Issue: "inference-sim/blis-substrate#275"}}
		}, "omits a tower"},
		{"no omitted tower under the text-decoder modality", func(g *Graph) { g.Modality = ModalityTextDecoderOfMultimodal }, "must state"},
		{"multimodal with no encoder", func(g *Graph) { g.Encoder = nil }, "encoder section"},
	}
	for _, c := range cases {
		g := multimodalGraph(kimiTower())
		c.mutate(g)
		p := g.Validate()
		if p.OK() {
			t.Errorf("%s: accepted", c.name)
			continue
		}
		if !strings.Contains(p.Error(), c.want) {
			t.Errorf("%s: rejected, but not for %q:\n%s", c.name, c.want, p.Error())
		}
	}
}

func TestEncoderThatPricesNoTowerNeedsOmittedOnes(t *testing.T) {
	g := multimodalGraph(kimiTower())
	g.Modality = ModalityTextDecoderOfMultimodal
	g.Encoder = &Encoder{}
	if p := g.Validate(); p.OK() {
		t.Fatal("an encoder with neither a tower nor an omitted tower was accepted")
	}
	g.Encoder = &Encoder{Omitted: []OmittedTower{{Modality: TowerImage, Reason: "stored dtype is not stated",
		Issue: "inference-sim/blis-substrate#1"}}}
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a graph that omits its only tower with an issue was rejected:\n%s", p.Error())
	}
	if g.PricesEncoder() {
		t.Error("a graph with no priced tower claims to price an encoder")
	}
}

func TestEncoderOmittedAudioWithIssueValidates(t *testing.T) {
	g := multimodalGraph(kimiTower())
	g.Modality = ModalityTextDecoderOfMultimodal
	g.Encoder.Omitted = []OmittedTower{{Modality: TowerAudio, Reason: "depthwise convolution has no primitive",
		Issue: "inference-sim/blis-substrate#275"}}
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a graph that omits an audio tower with an issue was rejected:\n%s", p.Error())
	}
}

// The evaluation functions are what the engine's encode stage calls: rows the tower runs
// on, tokens the image adds to the prompt, and the embedding bytes of an E-to-P handoff.
func TestTowerEvaluatesRowsAndTokens(t *testing.T) {
	k := kimiTower()
	// A 448x672 still is 32x48 patches; a 2x2 merge leaves 16x24 = 384 tokens.
	if got := k.Patches(448, 672, 1); got != 32*48 {
		t.Errorf("patches = %d, want %d", got, 32*48)
	}
	if got := k.OutputTokens(448, 672, 1); got != 384 {
		t.Errorf("output tokens = %d, want 384", got)
	}
	// Sizes round up to whole patches.
	if got := k.Patches(449, 673, 1); got != 33*49 {
		t.Errorf("patches at 449x673 = %d, want %d", got, 33*49)
	}
	if got := k.OutputBytes(384); got != 384*7168*2 {
		t.Errorf("output bytes = %d, want %d", got, 384*7168*2)
	}

	// Qwen: a temporal patch of 2 holds two frames; a still is one temporal patch.
	q := k
	q.Input = TowerInput{PatchSize: 16, TemporalPatchSize: 2, Channels: 3}
	if q.Patches(320, 320, 1) != q.Patches(320, 320, 2) {
		t.Error("a still and a two-frame clip should both be one temporal patch")
	}
	if q.Patches(320, 320, 4) != 2*q.Patches(320, 320, 2) {
		t.Error("four frames should be two temporal patches")
	}

	// Gemma 4: every image is padded to 2520 rows, a 3x3 pool, at most 280 tokens.
	g := k
	g.Input = TowerInput{PatchSize: 16, Channels: 3, FixedRows: 2520}
	g.MergeRatio, g.MaxOutputTokens = 9, 280
	if got := g.Patches(48, 48, 1); got != 2520 {
		t.Errorf("a small image should still cost 2520 rows, got %d", got)
	}
	if got := g.OutputTokens(48, 48, 1); got != 1 {
		t.Errorf("a 3x3-patch image should yield one token, got %d", got)
	}
	if got := g.OutputTokens(4000, 4000, 1); got != 280 {
		t.Errorf("a large image should be capped at 280 tokens, got %d", got)
	}

	// Llama 4: 336-pixel tiles of 24x24 patches and a class token, a 2x2 pixel shuffle, and
	// one more tile of the whole image when there are several.
	l := k
	l.Input = TowerInput{PatchSize: 14, Channels: 3, TilePixels: 336, ClassTokens: 1}
	l.MergeRatio = 4
	if r := l.Rows(336, 336, 1); r.Patches != 577 || r.Segments != 1 {
		t.Errorf("one tile = %+v, want 577 rows in 1 sequence", r)
	}
	if got := l.OutputTokens(336, 336, 1); got != 144 {
		t.Errorf("one tile yields %d tokens, want 144", got)
	}
	if r := l.Rows(672, 672, 1); r.Patches != 5*577 || r.Segments != 5 {
		t.Errorf("2x2 tiles = %+v, want 4 tiles and a global one", r)
	}
	if got := l.OutputTokens(672, 672, 1); got != 5*144 {
		t.Errorf("2x2 tiles yield %d tokens, want %d", got, 5*144)
	}
}

func TestTowerWeightBytesMatchVLLMParameterCount(t *testing.T) {
	k := kimiTower()
	// Kimi-K2.5's tower and projector are 471,149,104 parameters (vLLM v0.33.0
	// kimi_k25_vit.py): the tower's norms and projector LayerNorm are counted as stored.
	if got := k.Params(); got < 471_000_000 || got > 471_300_000 {
		t.Errorf("params = %d, want about 471.1M", got)
	}
	if got := k.WeightBytes(); got != float64(k.Params())*2 {
		t.Errorf("bf16 weight bytes = %v, want 2 per parameter", got)
	}
	k.Embedding[0].WeightDType = DTypeFP8
	if k.WeightBytes() >= float64(k.Params())*2 {
		t.Error("a node stored in fp8 should lower the tower's bytes")
	}
}
