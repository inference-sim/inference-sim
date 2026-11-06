package model

import "testing"

// These tests state the derivation rules a graph's author must follow, using the
// cases where a vendor configuration does not say what a reader would expect. They
// are about the SCHEMA's contract rather than about any one deriver: the point is
// that the schema's fields are all derivable, and that the derivation is not always
// a field lookup.

// TestDepthIsNotAlwaysAField records that a shipped hybrid states no layer count.
// Its depth is the length of a per-layer type vector, so a stack expressed from that
// vector is the only faithful representation — and a deriver that read a single
// expected key would emit a zero-layer model that still validated against every
// other rule.
func TestDepthIsNotAlwaysAField(t *testing.T) {
	// A 108-entry vector with no repeating unit: 48 state-space, 48 MoE, 12
	// attention layers, matching the counts a published configuration declares.
	vector := make([]string, 0, 108)
	for i := 0; i < 48; i++ {
		vector = append(vector, "mamba", "moe")
	}
	for i := 0; i < 12; i++ {
		vector = append(vector, "attention")
	}
	s := Stack{Prologue: vector}
	if got := s.Layers(); got != 108 {
		t.Errorf("a literal stack's depth = %d, want 108", got)
	}
	if got := len(s.Expand()); got != 108 {
		t.Errorf("expansion length = %d, want 108", got)
	}
	// The same schema must still express the sibling model, which DOES state a count
	// and whose vector length agrees with it.
	sibling := Stack{Prologue: vector[:52]}
	if got := sibling.Layers(); got != 52 {
		t.Errorf("sibling depth = %d, want 52", got)
	}
}

// TestHeadDimIsDerived records that head dimension is hidden size over head count.
// A node states it explicitly because the schema prices shapes rather than parsing
// configurations, but a deriver computes it: several configurations omit the field.
func TestHeadDimIsDerived(t *testing.T) {
	cases := []struct{ hidden, heads, want int }{
		{3072, 48, 64},  // a uniform MoE stack
		{8192, 64, 128}, // a dense stack
		{4096, 32, 128}, // a routed MoE
	}
	for _, c := range cases {
		if got := c.hidden / c.heads; got != c.want {
			t.Errorf("hidden %d over %d heads = %d, want %d",
				c.hidden, c.heads, got, c.want)
		}
	}
	// A graph asserting a head dimension inconsistent with its own global hidden
	// size is not rejected here: a model may project to a head dimension unrelated
	// to hidden size, and latent attention routinely does. The schema records what
	// the kernel must price, and the deriver is responsible for getting it right.
	g := &Graph{
		Kind: "ModelGraph", Name: "latent",
		DerivedFrom: Derivation{Format: "hf_config_json", Path: "config.json",
			SHA256: repeat64('e'), DeriverVersion: 1},
		Global: GlobalShape{HiddenSize: 7168, VocabSize: 1000, WeightDType: DTypeFP8},
		LayerKinds: []LayerKind{{ID: "mla", Nodes: []Node{
			// 576 is the latent width, not 7168/96.
			{Op: OpAttention, AttentionKind: AttentionMLA, NumQHeads: 96,
				NumKVHeads: 1, HeadDim: 576, KVLoRARank: 512, QKRopeHeadDim: 64},
		}}},
		Stack: Stack{Pattern: []string{"mla"}, Repeat: 1},
	}
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a latent head dimension unrelated to hidden size was rejected:\n%s",
			p.Error())
	}
}

// TestGraphIsIndependentOfVendorVocabulary is the property the schema exists for: two
// graphs describing the same computation are identical whatever the source dialect
// called its fields.
func TestGraphIsIndependentOfVendorVocabulary(t *testing.T) {
	build := func(format string) *Graph {
		return &Graph{
			Kind: "ModelGraph", Name: "same-model",
			DerivedFrom: Derivation{Format: format, Path: "config.json",
				SHA256: repeat64('f'), DeriverVersion: 1},
			Global: GlobalShape{HiddenSize: 4096, VocabSize: 32000,
				WeightDType: DTypeBF16},
			LayerKinds: []LayerKind{{ID: "block", Nodes: []Node{
				{Op: OpAttention, AttentionKind: AttentionGQA, NumQHeads: 32,
					NumKVHeads: 8, HeadDim: 128},
			}}},
			Stack: Stack{Pattern: []string{"block"}, Repeat: 32},
		}
	}
	a, b := build("hf_config_json"), build("some_other_dialect")
	if a.Stack.Layers() != b.Stack.Layers() {
		t.Error("the same computation derived from two dialects differs in depth")
	}
	for _, g := range []*Graph{a, b} {
		if p := g.Validate(); !p.OK() {
			t.Fatalf("%s rejected:\n%s", g.DerivedFrom.Format, p.Error())
		}
	}
	// Format is metadata for the reader, so the schema does not constrain it: a
	// deriver for a new dialect needs no schema change.
	if a.DerivedFrom.Format == b.DerivedFrom.Format {
		t.Error("the test did not actually vary the dialect")
	}
}

func repeat64(c byte) string {
	b := make([]byte, 64)
	for i := range b {
		b[i] = c
	}
	return string(b)
}

// TestModalityMakesAnOmissionExplicit covers the multimodal case. Three catalog
// models nest their text shapes alongside a vision or audio tower; a graph derived
// from one prices the decoder alone, and the schema records that rather than leaving
// a reader to discover it from a prediction that is low.
func TestModalityMakesAnOmissionExplicit(t *testing.T) {
	g := &Graph{
		Kind: "ModelGraph", Name: "multimodal-decoder",
		Modality: ModalityTextDecoderOfMultimodal,
		DerivedFrom: Derivation{Format: "hf_config_json", Path: "config.json",
			SHA256: repeat64('a'), DeriverVersion: 1},
		Global: GlobalShape{HiddenSize: 6144, VocabSize: 151552,
			WeightDType: DTypeBF16},
		LayerKinds: []LayerKind{{ID: "block", Nodes: []Node{
			{Op: OpAttention, AttentionKind: AttentionGQA, NumQHeads: 64,
				NumKVHeads: 8, HeadDim: 128},
		}}},
		Stack: Stack{Pattern: []string{"block"}, Repeat: 66},
	}
	if p := g.Validate(); !p.OK() {
		t.Fatalf("a declared text-decoder graph was rejected:\n%s", p.Error())
	}
	// No modality means text-only, which is the ordinary case.
	g.Modality = ""
	if p := g.Validate(); !p.OK() {
		t.Fatalf("an absent modality was rejected:\n%s", p.Error())
	}
	g.Modality = "text_and_vision"
	if p := g.Validate(); p.OK() {
		t.Error("an unrecognized modality was accepted")
	}
	// No modality claims to price an encoder, because no primitive computes one.
	for _, m := range []Modality{"", ModalityTextOnly,
		ModalityTextDecoderOfMultimodal} {
		if m.PricesEncoder() {
			t.Errorf("%q claims to price an encoder, which no primitive computes", m)
		}
	}
}
