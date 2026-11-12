package contracts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// This suite asserts that the model schema expresses every architecture family in
// blis-catalog, and that no schema capability a family needs has silently lost its last
// exerciser. It reads the vendored catalog snapshot under testdata/ (committed by #23)
// rather than hand-transcribed Go literals.
//
// The literals were only ever a stand-in for a catalog the test could not read, and they
// had already drifted from it: an earlier draft of this file made qwen2.5 sliding-window
// where the committed graph is full GQA, gave inkling a speculator the committed graph
// does not carry, and wrote nemotron's 108 layers out as one literal vector where the
// committed graph expresses them as a periodic pattern. Reading the committed graphs
// removes that whole class of drift. A family is covered the moment its fixture lands,
// and a capability a family needs cannot be dropped from the schema or re-derived away
// without a vendored graph failing to load, validate, or exercise it here.
//
// The whole snapshot is also loaded and validated by TestLoadCatalogModels in
// load_test.go; what this suite adds is the per-family capability each graph must exhibit
// and the across-the-catalog guard that every schema primitive, kind and dtype is used.

// loadCatalogGraphs loads every models/<name>/graph.yaml in the vendored snapshot, keyed
// by directory name. A load error fails loudly rather than skipping: a graph that cannot
// be read is a broken fixture, and a skip is indistinguishable from a pass. It reuses the
// catalogFixtures snapshot root that load_test.go pins.
func loadCatalogGraphs(t *testing.T) map[string]*model.Graph {
	t.Helper()
	modelsDir := filepath.Join(catalogFixtures, "models")
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		t.Fatalf("reading %s: %v (the vendored catalog snapshot is missing)", modelsDir, err)
	}
	graphs := map[string]*model.Graph{}
	for _, e := range entries {
		// A model entry is a directory; skip hidden entries so a stray editor/VCS dir is
		// never mistaken for a model.
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		path := filepath.Join(modelsDir, e.Name(), "graph.yaml")
		g, err := LoadModelGraph(path)
		if err != nil {
			t.Fatalf("LoadModelGraph(%s): %v", path, err)
		}
		graphs[e.Name()] = g
	}
	if len(graphs) == 0 {
		t.Fatalf("no model graphs under %s: the vendored catalog snapshot is missing", modelsDir)
	}
	return graphs
}

// anyNode reports whether any node in the graph's layer kinds or head satisfies f. The
// speculator's draft layers are declared in LayerKinds like any other kind, so walking
// LayerKinds reaches a kind only the draft stack uses (DeepSeek-V4-Pro's mtp_moe).
func anyNode(g *model.Graph, f func(model.Node) bool) bool {
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if f(n) {
				return true
			}
		}
	}
	for _, n := range g.Head {
		if f(n) {
			return true
		}
	}
	return false
}

func hasOp(g *model.Graph, op model.Op) bool {
	return anyNode(g, func(n model.Node) bool { return n.Op == op })
}

func hasAttention(g *model.Graph, k model.AttentionKind) bool {
	return anyNode(g, func(n model.Node) bool {
		return n.Op == model.OpAttention && n.AttentionKind == k
	})
}

func hasRecurrent(g *model.Graph, k model.RecurrentKind) bool {
	return anyNode(g, func(n model.Node) bool { return n.RecurrentKind == k })
}

func hasSharedExperts(g *model.Graph) bool {
	return anyNode(g, func(n model.Node) bool { return n.SharedExperts > 0 })
}

func hasLatentMoE(g *model.Graph) bool {
	return anyNode(g, func(n model.Node) bool { return n.LatentSize > 0 })
}

func hasCompressRatio(g *model.Graph) bool {
	return anyNode(g, func(n model.Node) bool { return n.CompressRatio > 0 })
}

func hasIndexTopK(g *model.Graph) bool {
	return anyNode(g, func(n model.Node) bool { return n.IndexTopK > 0 })
}

// hasNodeWeightDType reports a per-node weight-dtype override, used for a mixed-precision
// checkpoint whose experts differ from the rest (DeepSeek-V4-Pro's and gpt-oss's mxfp4
// experts, Nemotron-NVFP4's nvfp4 experts beside fp8).
func hasNodeWeightDType(g *model.Graph, d model.DType) bool {
	return anyNode(g, func(n model.Node) bool { return n.WeightDType == d })
}

func hasPrologue(g *model.Graph) bool { return len(g.Stack.Prologue) > 0 }

func isMultimodal(g *model.Graph) bool {
	return g.Modality == model.ModalityTextDecoderOfMultimodal
}

// TestCatalogArchitectureFamiliesAreExpressible walks every architecture family the
// catalog holds, each represented by one committed graph. A case names the schema
// capability the family exercises, pins the family's layer depth, and asserts the
// vendored graph both validates and still shows that capability — so a case cannot pass
// while the capability it claims has gone missing from the catalog or the schema. The
// `exercises` predicate asserts the family's DEFINING shape, which is why a re-derivation
// that dropped it (a sparse-MLA layer re-derived without its compression, say) fails here
// rather than silently.
func TestCatalogArchitectureFamiliesAreExpressible(t *testing.T) {
	graphs := loadCatalogGraphs(t)
	cases := []struct {
		model      string
		capability string
		layers     int
		exercises  func(*model.Graph) bool
	}{
		{"llama-3.1-70b-instruct", "dense GQA attention with a dense MLP", 80,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionGQA) && len(g.LayerKinds) == 1 &&
					!hasOp(g, model.OpGroupedGEMM)
			}},
		{"mixtral-8x7b-v0.1", "routed MoE with no shared expert", 32,
			func(g *model.Graph) bool {
				return hasOp(g, model.OpGroupedGEMM) && !hasSharedExperts(g)
			}},
		{"deepseek-v2-lite", "latent (MLA) attention with shared experts", 27,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionMLA) && hasSharedExperts(g)
			}},
		{"deepseek-v3", "latent (MLA) MoE with a dense prologue and an MTP speculator", 61,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionMLA) && hasPrologue(g) && g.Speculator != nil
			}},
		{"deepseek-v4-pro", "compress-only and top-k-plus-compress sparse-MLA, mxfp4 experts", 61,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionSparseMLA) && hasCompressRatio(g) &&
					hasIndexTopK(g) && hasNodeWeightDType(g, model.DTypeMXFP4)
			}},
		{"glm-5", "sparse-MLA with an index top-k, a dense prologue and an MTP speculator", 78,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionSparseMLA) && hasIndexTopK(g) &&
					hasPrologue(g) && g.Speculator != nil
			}},
		{"gpt-oss-120b", "alternating sliding-window and full attention, mxfp4 experts", 36,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionSWA) && hasAttention(g, model.AttentionGQA) &&
					hasNodeWeightDType(g, model.DTypeMXFP4)
			}},
		// minimax-m2.5 and m2.7 are the same family at two revisions; both are listed so
		// each vendored graph is loaded, validated and shown to carry the capability.
		{"minimax-m2.5", "full-attention fp8 MoE", 62,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionGQA) && hasOp(g, model.OpGroupedGEMM) &&
					g.Global.WeightDType == model.DTypeFP8
			}},
		{"minimax-m2.7", "full-attention fp8 MoE", 62,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionGQA) && hasOp(g, model.OpGroupedGEMM) &&
					g.Global.WeightDType == model.DTypeFP8
			}},
		{"minimax-m3", "sliding-window attention MoE with a dense prologue, MTP, multimodal", 60,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionSWA) && hasPrologue(g) &&
					g.Speculator != nil && isMultimodal(g)
			}},
		{"kimi-k2.5", "int4 (W4A16) weights, latent-attention MoE", 61,
			func(g *model.Graph) bool {
				return g.Global.WeightDType == model.DTypeINT4 && hasAttention(g, model.AttentionMLA)
			}},
		{"kimi-k3", "linear (KDA) plus latent attention, mxfp4 experts", 93,
			func(g *model.Graph) bool {
				return hasRecurrent(g, model.RecurrentKDA) && hasAttention(g, model.AttentionMLA) &&
					hasNodeWeightDType(g, model.DTypeMXFP4)
			}},
		{"qwen3.5-397b-a17b", "gated-deltanet (GDN) and full attention, MTP, multimodal", 60,
			func(g *model.Graph) bool {
				return hasRecurrent(g, model.RecurrentGDN) && hasAttention(g, model.AttentionGQA) &&
					g.Speculator != nil && isMultimodal(g)
			}},
		{"nemotron-3-ultra-550b-a55b-bf16", "Mamba2, attention and latent-MoE hybrid with an MTP speculator", 108,
			func(g *model.Graph) bool {
				return hasRecurrent(g, model.RecurrentMamba2) && hasAttention(g, model.AttentionGQA) &&
					hasLatentMoE(g) && g.Speculator != nil
			}},
		{"nemotron-3-ultra-550b-a55b-nvfp4", "the same hybrid with per-node nvfp4 expert weights", 108,
			func(g *model.Graph) bool {
				return hasRecurrent(g, model.RecurrentMamba2) && hasNodeWeightDType(g, model.DTypeNVFP4)
			}},
		{"inkling", "sliding-window interleaved with full attention, shared experts, multimodal", 66,
			func(g *model.Graph) bool {
				return hasAttention(g, model.AttentionSWA) && hasAttention(g, model.AttentionGQA) &&
					hasSharedExperts(g) && isMultimodal(g)
			}},
	}
	for _, c := range cases {
		t.Run(c.model, func(t *testing.T) {
			g, ok := graphs[c.model]
			if !ok {
				t.Fatalf("no vendored graph for %q; the snapshot does not hold the family this case covers (%s)",
					c.model, c.capability)
			}
			if p := g.Validate(); !p.OK() {
				t.Fatalf("%s is not expressible:\n%s", c.model, p.Error())
			}
			if got := g.Stack.Layers(); got != c.layers {
				t.Errorf("%s: stack expands to %d layers, want %d", c.model, got, c.layers)
			}
			if !c.exercises(g) {
				t.Errorf("%s no longer exercises %q; the vendored graph does not show the capability this case names",
					c.model, c.capability)
			}
		})
	}
}

// TestEveryPrimitiveAndKindIsExercised guards against a schema capability passing with no
// catalog family that uses it. It reads the WHOLE snapshot and asserts that every member
// of the required lists appears somewhere. That is not a tautology over the data it reads:
// the required lists are the schema capabilities the catalog is expected to exercise, so a
// member losing its last exerciser — a model dropped, or a graph re-derived without it —
// fails here, which validation alone would not catch because a graph that never uses a
// capability still validates. A new schema capability the catalog adopts belongs on one of
// these lists so its coverage is pinned the same way.
//
// The sequence-parallel collectives (AllGather, ReduceScatter) are deliberately absent:
// they are a deployment-time choice no model graph carries, so no catalog family exercises
// them and the list does not require them.
func TestEveryPrimitiveAndKindIsExercised(t *testing.T) {
	graphs := loadCatalogGraphs(t)

	ops := map[model.Op]bool{}
	attnKinds := map[model.AttentionKind]bool{}
	recurrentKinds := map[model.RecurrentKind]bool{}
	weightDTypes := map[model.DType]bool{}
	stateDTypes := map[model.DType]bool{}
	var sawShared, sawLatentMoE, sawCompressRatio, sawIndexTopK bool
	var sawSpeculator, sawMultiKind, sawMultimodal bool

	for _, g := range graphs {
		weightDTypes[g.Global.WeightDType] = true
		if g.Speculator != nil {
			sawSpeculator = true
		}
		if len(g.LayerKinds) > 1 {
			sawMultiKind = true
		}
		if isMultimodal(g) {
			sawMultimodal = true
		}
		record := func(n model.Node) {
			ops[n.Op] = true
			if n.Op == model.OpAttention && n.AttentionKind != "" {
				attnKinds[n.AttentionKind] = true
			}
			if n.RecurrentKind != "" {
				recurrentKinds[n.RecurrentKind] = true
			}
			if n.WeightDType != "" {
				weightDTypes[n.WeightDType] = true
			}
			if n.StateDType != "" {
				stateDTypes[n.StateDType] = true
			}
			if n.SharedExperts > 0 {
				sawShared = true
			}
			if n.LatentSize > 0 {
				sawLatentMoE = true
			}
			if n.CompressRatio > 0 {
				sawCompressRatio = true
			}
			if n.IndexTopK > 0 {
				sawIndexTopK = true
			}
		}
		for _, lk := range g.LayerKinds {
			for _, n := range lk.Nodes {
				record(n)
			}
		}
		for _, n := range g.Head {
			record(n)
		}
	}

	for _, op := range []model.Op{model.OpGEMM, model.OpGroupedGEMM, model.OpAttention,
		model.OpRecurrentUpdate, model.OpElementwise, model.OpAllReduce, model.OpAll2All} {
		if !ops[op] {
			t.Errorf("no catalog family exercises op %s", op)
		}
	}
	for _, k := range []model.AttentionKind{model.AttentionGQA, model.AttentionMLA,
		model.AttentionSparseMLA, model.AttentionSWA} {
		if !attnKinds[k] {
			t.Errorf("no catalog family exercises attention kind %s", k)
		}
	}
	// All three recurrent kinds are now in the catalog: Mamba2 (Nemotron), KDA (Kimi-K3)
	// and GDN (Qwen3.5-397B).
	for _, k := range []model.RecurrentKind{model.RecurrentMamba2, model.RecurrentKDA,
		model.RecurrentGDN} {
		if !recurrentKinds[k] {
			t.Errorf("no catalog family exercises recurrent kind %s", k)
		}
	}
	// Six weight dtypes have a home in the catalog: bf16, fp16 (Llama-2), fp8, nvfp4
	// (Nemotron-NVFP4 experts), mxfp4 (gpt-oss/Kimi-K3/DeepSeek-V4-Pro) and int4
	// (Kimi-K2.5). int8 has no catalog model yet, so it is not required here.
	for _, d := range []model.DType{model.DTypeBF16, model.DTypeFP16, model.DTypeFP8,
		model.DTypeNVFP4, model.DTypeMXFP4, model.DTypeINT4} {
		if !weightDTypes[d] {
			t.Errorf("no catalog family exercises weight dtype %s", d)
		}
	}
	// The recurrent state is stored wider than the weights; every recurrent model keeps
	// it in fp32.
	for _, d := range []model.DType{model.DTypeFP32} {
		if !stateDTypes[d] {
			t.Errorf("no catalog family exercises recurrent state dtype %s", d)
		}
	}
	if !sawShared {
		t.Error("no family exercises shared experts, which several catalog models use")
	}
	if !sawLatentMoE {
		t.Error("no family exercises latent MoE, which narrows the expert input below hidden size")
	}
	if !sawCompressRatio {
		t.Error("no family exercises compress_ratio, the compressed sparse-MLA latent read (DeepSeek-V4-Pro)")
	}
	if !sawIndexTopK {
		t.Error("no family exercises index_topk, the sparse-MLA top-k cache select")
	}
	if !sawSpeculator {
		t.Error("no family exercises a speculator stack")
	}
	if !sawMultiKind {
		t.Error("no family exercises a multi-kind stack, which every hybrid needs")
	}
	if !sawMultimodal {
		t.Error("no family declares itself the text decoder of a multimodal model, though several catalog models are")
	}
}

// TestNemotronStackCompositionMatchesTheConfiguration expands the vendored Nemotron stack
// and checks its layer mix. The hybrid's depth is a per-index layer vector, so a graph can
// carry the right total with the wrong mix (a state-space layer re-derived as an attention
// one). Validation does not catch that, and the layer-count assertion in the family table
// above would not either, so the composition is pinned against the configuration's counts.
func TestNemotronStackCompositionMatchesTheConfiguration(t *testing.T) {
	const name = "nemotron-3-ultra-550b-a55b-bf16"
	g, ok := loadCatalogGraphs(t)[name]
	if !ok {
		t.Fatalf("the vendored snapshot has no %s graph", name)
	}
	counts := map[string]int{}
	for _, id := range g.Stack.Expand() {
		counts[id]++
	}
	// The configuration declares 48 state-space, 48 MoE and 12 attention layers in 108.
	for id, want := range map[string]int{"mamba": 48, "moe": 48, "attention": 12} {
		if counts[id] != want {
			t.Errorf("%q layers = %d, the configuration declares %d", id, counts[id], want)
		}
	}
	if total := g.Stack.Layers(); total != 108 {
		t.Errorf("stack expands to %d layers, want 108", total)
	}
}

// TestStackExpansionShapes covers the three stack forms the schema supports, since each
// exists for a model the catalog holds. It is a unit test of the Stack primitive and does
// not read the snapshot.
func TestStackExpansionShapes(t *testing.T) {
	uniform := model.Stack{Pattern: []string{"a"}, Repeat: 4}
	if got := uniform.Expand(); len(got) != 4 || got[0] != "a" {
		t.Errorf("uniform expansion = %v", got)
	}
	withPrologue := model.Stack{Prologue: []string{"d", "d"},
		Pattern: []string{"s"}, Repeat: 3}
	want := []string{"d", "d", "s", "s", "s"}
	got := withPrologue.Expand()
	if len(got) != len(want) {
		t.Fatalf("prologue expansion = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %q, want %q", i, got[i], want[i])
		}
	}
	literal := model.Stack{Prologue: []string{"a", "b", "c"}}
	if got := literal.Layers(); got != 3 {
		t.Errorf("literal stack layers = %d, want 3", got)
	}
	if got := literal.Expand(); len(got) != 3 {
		t.Errorf("literal expansion = %v", got)
	}
}
