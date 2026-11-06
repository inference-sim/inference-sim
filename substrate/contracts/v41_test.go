package contracts

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
)

// The DeepSeek-V4.1-Flash example is the case issues #47-#54 were filed for, so the
// figures those issues state are checked against it here: if the schema could not express
// them, or a helper computed them differently, the example would not reproduce them.

func v41(t *testing.T) *model.Graph {
	t.Helper()
	g, err := LoadModelGraph("../docs/contracts/examples/model-graph-deepseek-v4.1.yaml")
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// TestV41KVPerContextToken is #47's acceptance case: 1,790 B per context token, against the
// 40 x 584 = 23,360 B a consumer infers when every attention layer is taken to own a
// full-length cache of its read width.
func TestV41KVPerContextToken(t *testing.T) {
	g := v41(t)
	kinds := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		kinds[lk.ID] = lk
	}
	var perToken float64
	var windows, rings, ringBytes int
	tally := func(s model.Stack) {
		for _, id := range s.Expand() {
			for _, n := range kinds[id].Nodes {
				for _, c := range n.Caches {
					perToken += c.BytesPerContextToken("fp8_ds_mla")
					if c.Owned && c.Kind == model.CacheWindow {
						windows++
					}
					if c.Owned && c.Kind == model.CacheState {
						rings++
						ringBytes += c.Entries * c.BytesPerEntry
					}
				}
			}
		}
	}
	tally(g.Stack)
	if g.Stack.Layers() != 40 {
		t.Fatalf("stack expands to %d layers, want 40", g.Stack.Layers())
	}
	// Compressed ratio 2 on 2, 8, 14: 3 x 584 / 2 = 876. Ratio 1 on 20: 584. Indexer keys:
	// 3 x 132 / 2 + 132 = 330.
	if perToken != 1790 {
		t.Errorf("owned bytes per context token = %v, want 1790", perToken)
	}
	// On SM100 under nvfp4_ds_mla the compressed records shrink to 288 B; the indexer keys
	// keep 132: 3 x 288 / 2 + 288 + 330.
	b, err := g.ForCapability(10)
	if err != nil {
		t.Fatal(err)
	}
	var nv float64
	for _, id := range b.Stack.Expand() {
		for _, lk := range b.LayerKinds {
			if lk.ID != id {
				continue
			}
			for _, n := range lk.Nodes {
				for _, c := range n.Caches {
					nv += c.BytesPerContextToken("nvfp4_ds_mla")
				}
			}
		}
	}
	if nv != 1050 {
		t.Errorf("SM100 nvfp4_ds_mla bytes per context token = %v, want 1050", nv)
	}
	tally(g.Speculator.Stack)
	if perToken != 1790 {
		t.Errorf("the draft layers own only window caches, yet they add %v B per context token", perToken-1790)
	}
	// Every layer, the three draft layers included, owns a 128-token window ring.
	if windows != 43 {
		t.Errorf("owned window caches = %d, want 43", windows)
	}
	// The compressor ring: 8 FP32 rows of 4,096 B on each of the three ratio-2 sources.
	if rings != 3 || ringBytes != 3*8*4096 {
		t.Errorf("state rings = %d of %d B in all, want 3 of %d", rings, ringBytes, 3*8*4096)
	}
}

// TestV41WindowRingTerm is the constant term beside #47's 1,790 B: what the 43 window
// rings hold per request, by vLLM v0.31.0's admission cap
// (max_admission_blocks_per_request): cdiv(min(window - 1 + extra retained + in flight,
// max_model_len), block) + 1 pages, each page's bytes rounded up to the alignment. The
// formula is the engine's and so the consumer's to apply; the test shows the graph
// carries every input it needs. One decode token in flight and no extra retention.
func TestV41WindowRingTerm(t *testing.T) {
	g := v41(t)
	kinds := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		kinds[lk.ID] = lk
	}
	// A decode step holds 1 + num_spec_tokens tokens of each request in flight, and
	// DSpark, an EAGLE-family drafter, retains none beyond the window.
	inFlight := 1 + g.Speculator.NumSpec
	const engineBlock, maxModelLen = 64, 1 << 20
	var total int64
	for _, s := range []model.Stack{g.Stack, g.Speculator.Stack} {
		for _, id := range s.Expand() {
			for _, n := range kinds[id].Nodes {
				for _, c := range n.Caches {
					if !c.Owned || c.Kind != model.CacheWindow {
						continue
					}
					pages := windowPagesPerRequest(c, engineBlock, inFlight, maxModelLen)
					page := c.BlockSize * c.BytesPerEntry
					if a := c.PageAlignment; a > 0 {
						page = (page + a - 1) / a * a
					}
					total += int64(pages * page)
				}
			}
		}
	}
	// 43 rings x 6 pages (cdiv(127 + 6, 32) + 1) x 19,008 B (32 x 584 = 18,688, padded
	// to a multiple of 576).
	if want := int64(43 * 6 * 19008); total != want {
		t.Errorf("window rings hold %d B per request, want %d", total, want)
	}
}

// windowPagesPerRequest is the admission cap the comment above states, applied as a consumer
// applies it: cdiv(min(window - 1 + inFlight, maxModelLen), block) + 1, with no extra retention.
func windowPagesPerRequest(c model.Cache, engineBlock, inFlight, maxModelLen int) int {
	block := c.BlockSize
	if block < 1 {
		block = engineBlock
	}
	tokens := min(c.Window-1+inFlight, maxModelLen)
	return (tokens+block-1)/block + 1
}

// TestV41WeightBytes is #54's case: MXFP4 experts at 0.53125 B per parameter, FP8 weights
// with their resident 1 x 32 scales, and a BF16 node that does not inherit them.
func TestV41WeightBytes(t *testing.T) {
	g := v41(t)
	byRole := map[string]model.Node{}
	g.EachNode(func(_ string, n model.Node) {
		if _, ok := byRole[n.Role]; !ok {
			byRole[n.Role] = n
		}
	})
	for role, want := range map[string]float64{
		"experts":                    0.5 + 1.0/32,
		"fused_wqa_wkv":              1 + 1.0/32,
		"compressor_fused_wkv_wgate": 2, // BF16: no inherited FP8 scale
		"wo_a":                       2 + 1.0/32,
		"lm_head":                    2,
	} {
		if got := g.WeightBytesPerParam(byRole[role]); got != want {
			t.Errorf("%s: %v B per parameter, want %v", role, got, want)
		}
	}
	// The routed experts' parameters, from the graph: every V4.1 target layer runs the same
	// routed MoE, which the abridged example states once, so its shape times the stack's
	// depth. Each expert is a SwiGLU MLP of three n x k matrices (gate, up and down).
	e := byRole["experts"]
	params := float64(g.Stack.Layers()) * float64(e.Experts) * 3 * float64(e.N) * float64(e.K)
	if got := params * g.WeightBytesPerParam(byRole["experts"]) / 1e9; math.Abs(got-288.78) > 0.01 {
		t.Errorf("routed experts = %.2f GB, want 288.78", got)
	}

	// On SM100 the MXFP8 BMM runs natively, so wo_a is resident at FP8 with its scales.
	b, err := g.ForCapability(10)
	if err != nil {
		t.Fatal(err)
	}
	var woa model.Node
	b.EachNode(func(_ string, n model.Node) {
		if n.Role == "wo_a" {
			woa = n
		}
	})
	if got := b.WeightBytesPerParam(woa); got != 1+1.0/32 {
		t.Errorf("SM100 wo_a: %v B per parameter, want %v", got, 1+1.0/32)
	}
}

// TestV41ArchOverride is #53's case: the same graph serves a 584 B record on SM90 and a
// 528 B one on SM100, where nvfp4_ds_mla also shrinks the compressed cache to 288 B.
func TestV41ArchOverride(t *testing.T) {
	g := v41(t)
	record := func(g *model.Graph, name, dtype string) (int, int) {
		for _, lk := range g.LayerKinds {
			for _, n := range lk.Nodes {
				for _, c := range n.Caches {
					if c.Name == name && c.Owned {
						return c.EntryBytes(dtype), c.PageAlignment
					}
				}
			}
		}
		t.Fatalf("no owned cache %q", name)
		return 0, 0
	}
	h, err := g.ForCapability(9)
	if err != nil {
		t.Fatal(err)
	}
	b, err := g.ForCapability(10)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		g          *model.Graph
		arch       string
		name       string
		dtype      string
		bytes, pad int
	}{
		{h, "sm90", "swa", "fp8_ds_mla", 584, 576},
		{h, "sm90", "compressed", "fp8_ds_mla", 584, 576},
		{b, "sm100", "swa", "fp8_ds_mla", 528, 512},
		{b, "sm100", "swa", "nvfp4_ds_mla", 528, 512},
		{b, "sm100", "compressed", "fp8_ds_mla", 528, 512},
		{b, "sm100", "compressed", "nvfp4_ds_mla", 288, 512},
		{b, "sm100", "index", "fp8_ds_mla", 132, 512},
	} {
		if bytes, pad := record(c.g, c.name, c.dtype); bytes != c.bytes || pad != c.pad {
			t.Errorf("%s %s under %s: %d B aligned to %d, want %d aligned to %d",
				c.arch, c.name, c.dtype, bytes, pad, c.bytes, c.pad)
		}
	}
	if len(b.ArchOverrides) != 0 {
		t.Error("a resolved graph should carry no overrides")
	}
	// Resolving must not touch the graph it was resolved from.
	if bytes, _ := record(g, "swa", "fp8_ds_mla"); bytes != 584 {
		t.Errorf("ForCapability modified its receiver: base swa is now %d B", bytes)
	}
	if p := b.Validate(); !p.OK() {
		t.Errorf("the SM100 graph does not validate:\n%s", p.Error())
	}
}

// TestV41BundleValidates: the V4.1 graph and a deployment of it validate together, field by
// field; the engine block is the release's rules', which the caller supplies.
func TestV41BundleValidates(t *testing.T) {
	g := v41(t)
	s := &scenario.Scenario{
		Kind: "Scenario", Name: "v41", Model: "deepseek-v4.1-flash",
		Coefficients: []string{"c"}, EngineVersion: "1.0.0",
		Cluster: scenario.Cluster{Hardware: "h200", Nodes: 1, GPUsPerNode: 8},
	}
	d := &deployment.Deployment{Kind: "Deployment", Name: "v41", Pools: []deployment.Pool{{
		Role: deployment.RoleColocated, Nodes: 1,
		Parallel: deployment.Parallelism{TP: 8, PP: 1, DP: 1},
		Engine:   engineBlock("{backend: fast, draft_tokens: 5}"),
	}}}
	rep := Validate(Bundle{Scenario: s, Deployment: d, Model: g, Rules: testRulesFor})
	requireSilent(t, "v41", rep)
	if rep.RulesApplied != "1.0.0" {
		t.Errorf("RulesApplied = %q, want 1.0.0", rep.RulesApplied)
	}
}

// A cache must say whether its layer owns it: an omitted owned would read as a reader and
// silently drop the cache's residency, so decoding refuses it, and an unknown key too.
func TestCacheOwnedIsRequired(t *testing.T) {
	dir := t.TempDir()
	src, err := os.ReadFile("../docs/contracts/examples/model-graph-deepseek-v4.1.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(string) string{
		"owned omitted": func(s string) string {
			return strings.Replace(s, "{name: compressor_ring, kind: state, owned: true,", "{name: compressor_ring, kind: state,", 1)
		},
		"unknown key": func(s string) string {
			return strings.Replace(s, "{name: compressor_ring, kind: state, owned: true,", "{name: compressor_ring, kind: state, owned: true, ownd: true,", 1)
		},
	} {
		changed := mutate(string(src))
		if changed == string(src) {
			t.Fatalf("%s: the mutation did not apply", name)
		}
		path := filepath.Join(dir, "graph.yaml")
		if err := os.WriteFile(path, []byte(changed), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadModelGraph(path); err == nil {
			t.Errorf("%s: the graph loaded", name)
		}
	}
}
