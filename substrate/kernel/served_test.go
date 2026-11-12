package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// mixedPrecisionServed is the layout the engine model states for nvidia/MiniMax-M3-NVFP4: MXFP8
// linears (one scale byte per 32 weights), NVFP4 routed experts, an FP32 router, a BF16 head and
// a BF16 embedding.
var mixedPrecisionServed = enginesettings.ServedWeights{Linear: "fp8", LinearScaleElements: 32,
	RoutedExperts: "nvfp4", Router: "fp32", Head: "bf16", Embedding: "bf16"}

// The engine model states the layout it serves and the kernel proves the graph stores exactly
// it. The real MiniMax-M3 NVFP4 graph matches; each one-edit variant of it, and each other
// catalog layout, does not, and is refused with the class named. The deletions and the wrong
// dtypes are the counterexamples a predicate that checks only what happens to be present
// would accept.
func TestTheServedLayoutIsProvedAgainstTheGraph(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	served := func(in Inputs, sw *enginesettings.ServedWeights) Inputs {
		return withEngine(in, func(e *enginesettings.Settings) { e.ServedWeights = sw })
	}
	base := quantCase(t, d, "minimax-m3-nvfp4", "b200")
	if _, err := New(served(base, &mixedPrecisionServed)); err != nil {
		t.Fatalf("the checkpoint's own layout was refused: %v", err)
	}
	// Nil is the graph as stored, as before.
	if _, err := New(base); err != nil {
		t.Fatal(err)
	}
	edit := func(f func(g *model.Graph)) Inputs {
		in := base
		g, err := in.Model.ForCapability(10)
		if err != nil {
			t.Fatal(err)
		}
		c := *g
		c.LayerKinds = append([]model.LayerKind(nil), g.LayerKinds...)
		for i := range c.LayerKinds {
			c.LayerKinds[i].Nodes = append([]model.Node(nil), g.LayerKinds[i].Nodes...)
		}
		c.Head = append([]model.Node(nil), g.Head...)
		c.Embedding = append([]model.Node(nil), g.Embedding...)
		f(&c)
		in.Model = &c
		return served(in, &mixedPrecisionServed)
	}
	dropNode := func(g *model.Graph, match func(n model.Node) bool) {
		for i := range g.LayerKinds {
			kept := g.LayerKinds[i].Nodes[:0]
			for _, n := range g.LayerKinds[i].Nodes {
				if !match(n) {
					kept = append(kept, n)
				}
			}
			g.LayerKinds[i].Nodes = kept
			g.LayerKinds[i].Edges = nil
		}
	}
	setNode := func(g *model.Graph, match func(n model.Node) bool, dt model.DType) {
		for i := range g.LayerKinds {
			for j := range g.LayerKinds[i].Nodes {
				if match(g.LayerKinds[i].Nodes[j]) {
					g.LayerKinds[i].Nodes[j].WeightDType = dt
				}
			}
		}
	}
	isRouter := func(n model.Node) bool { return n.Op == model.OpGEMM && n.N == 128 && n.Role == "router" }
	isRouted := func(n model.Node) bool { return n.Op == model.OpGroupedGEMM && !n.IsSharedExpert() }
	isAttn := func(n model.Node) bool { return n.Op == model.OpGEMM && n.Role == "qkv_proj" }
	for name, c := range map[string]struct {
		in   Inputs
		want string
	}{
		"router removed": {edit(func(g *model.Graph) { dropNode(g, isRouter) }), "router"},
		"router at bf16": {edit(func(g *model.Graph) { setNode(g, isRouter, model.DTypeBF16) }), "router"},
		"head removed": {edit(func(g *model.Graph) {
			kept := g.Head[:0]
			for _, n := range g.Head {
				if n.Role != "lm_head" {
					kept = append(kept, n)
				}
			}
			g.Head = kept
		}), "head"},
		"head at fp8": {edit(func(g *model.Graph) {
			for i := range g.Head {
				if g.Head[i].Role == "lm_head" {
					g.Head[i].WeightDType = model.DTypeFP8
				}
			}
		}), "head"},
		"embedding removed": {edit(func(g *model.Graph) { g.Embedding = nil }), "embedding"},
		"embedding at fp8": {edit(func(g *model.Graph) {
			for i := range g.Embedding {
				if g.Embedding[i].Op == model.OpGather {
					g.Embedding[i].RowBytes /= 2
				}
			}
		}), "embedding"},
		"routed experts at fp8": {edit(func(g *model.Graph) { setNode(g, isRouted, model.DTypeFP8) }), "routed_experts"},
		"attention at bf16":     {edit(func(g *model.Graph) { setNode(g, isAttn, model.DTypeBF16) }), "linear"},
		"no microscaling scale": {edit(func(g *model.Graph) { g.Global.WeightScale = nil }), "linear scale"},
		"shared expert in the NVFP4 node": {edit(func(g *model.Graph) {
			for i := range g.LayerKinds {
				for j := range g.LayerKinds[i].Nodes {
					if isRouted(g.LayerKinds[i].Nodes[j]) {
						g.LayerKinds[i].Nodes[j].SharedExperts = 1
					}
				}
			}
		}), "linear"},
		"the MXFP8 checkpoint": {served(quantCase(t, d, "minimax-m3-mxfp8", "b200"), &mixedPrecisionServed), "routed_experts"},
		"the bf16 checkpoint":  {served(quantCase(t, d, "minimax-m3", "b200"), &mixedPrecisionServed), "linear"},
		"gpt-oss":              {served(quantCase(t, d, "gpt-oss-120b", "b200"), &mixedPrecisionServed), "linear"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := New(c.in)
			if err == nil {
				t.Fatal("a graph that is not the served layout was priced")
			}
			for _, w := range []string{"served_weights_mismatch", c.want} {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("the refusal does not name %q: %v", w, err)
				}
			}
		})
	}
}

// The graph stores the layout whose quantized checkpoint it describes at a lower weight
// footprint, and prices differently from the MXFP8 checkpoint's.
func TestTheMixedPrecisionGraphPricesNVFP4ExpertsUnderFP8Linears(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"b200", "b300"} {
		mx := mustKernel(t, quantCase(t, d, "minimax-m3-mxfp8", chip))
		nv := mustKernel(t, quantCase(t, d, "minimax-m3-nvfp4", chip))
		if mx.FixedBytes().Weights <= nv.FixedBytes().Weights {
			t.Errorf("%s: NVFP4 experts hold %d weight bytes a rank, MXFP8 %d; the experts are most of the checkpoint",
				chip, nv.FixedBytes().Weights, mx.FixedBytes().Weights)
		}
		if pricesOf(mx) == pricesOf(nv) {
			t.Errorf("%s: the NVFP4 graph prices as the MXFP8 one", chip)
		}
	}
}

// The kernel recognises no engine quantization spelling beyond the ones servedDType prices: a
// ModelOpt or microscaling spelling is the engine model's to resolve, and reaches the kernel as
// the layout it serves (ServedWeights) with Quantization empty.
func TestAnEngineQuantizationSpellingTheKernelDoesNotPriceIsRefused(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := quantCase(t, d, "minimax-m3-nvfp4", "b200")
	for _, spelling := range []string{"modelopt_mixed", "mxfp8", "modelopt_mxfp8", "gpt_oss_mxfp4"} {
		in := withEngine(base, func(e *enginesettings.Settings) { e.Quantization = spelling })
		_, err := New(in)
		if err == nil || !strings.Contains(err.Error(), "not one this kernel prices") {
			t.Errorf("quantization %q: %v", spelling, err)
		}
	}
}
