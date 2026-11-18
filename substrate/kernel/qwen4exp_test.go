package latencykernel

import (
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// The catalog's Qwen3.6 hybrid has Qwen3.8-Flash-Next's stack shape (3 GDN layers to 1
// attention layer, 2 KV heads of 256). These tests make its attention layers sparse GQA and
// add a short convolution, and read what the kernel charges and records.

func qwen36Inputs(t *testing.T, tp string) Inputs {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range perturbationScenarios(t, d, "qwen3.6-35b-a3b") {
		if strings.HasPrefix(c.label, "qwen3.6-35b-a3b/h200/"+tp+"/") {
			return c.in
		}
	}
	t.Fatalf("no qwen3.6 scenario on h200 at %s", tp)
	return Inputs{}
}

// withGraph returns in priced on g, where edit rewrites the nodes of the attention layer kind.
func withGraph(t *testing.T, in Inputs, edit func([]model.Node) []model.Node) *Kernel {
	t.Helper()
	g := *in.Model
	g.LayerKinds = append([]model.LayerKind(nil), g.LayerKinds...)
	for i, lk := range g.LayerKinds {
		if lk.ID == "attn_moe" {
			g.LayerKinds[i].Nodes = edit(append([]model.Node(nil), lk.Nodes...))
		}
	}
	in.Model = &g
	return mustKernel(t, in)
}

func sparseGQA(budget int) func([]model.Node) []model.Node {
	return func(nodes []model.Node) []model.Node {
		var out []model.Node
		for _, n := range nodes {
			if n.Op == model.OpAttention {
				out = append(out, model.Node{Op: model.OpAttention, Role: "indexer_scores", HoldsEngineKV: model.Bool(false),
					AttentionKind: model.AttentionSparseGQA, NumQHeads: 4, NumKVHeads: 1, HeadDim: 128, CompressRatio: 4,
					Caches: []model.Cache{{Name: "index", Kind: model.CacheIndex, Owned: true, BytesPerEntry: 256, TokensPerEntry: 4}}})
				n.AttentionKind, n.IndexTopK = model.AttentionSparseGQA, budget
			}
			out = append(out, n)
		}
		return out
	}
}

func withShortConv(kernelWidth, dilation int) func([]model.Node) []model.Node {
	return func(nodes []model.Node) []model.Node {
		return append(nodes, model.Node{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentShortConv,
			Role: "ple_conv", ConvKernel: kernelWidth, ConvDilation: dilation, IntermediateSize: 8192})
	}
}

func decodeAt(k *Kernel, ctx int) float64 {
	return k.StepTime(kernel.Batch{DecodeThreshold: 1,
		Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: ctx, PromptLen: ctx + 1}}}).NoOverlap.Seconds()
}

func substituted(k *Kernel, name string) bool {
	for _, o := range k.Provenance() {
		if o.Name == name && o.Method == "assumed" && strings.Contains(o.Scope, "substituted") {
			return true
		}
	}
	return false
}

// Past its budget a sparse layer's decode cost stops growing with context; a dense layer's
// keeps growing.
func TestSparseGQADecodeStopsGrowingPastTheBudget(t *testing.T) {
	in := qwen36Inputs(t, "tp2")
	dense, sparse := mustKernel(t, in), withGraph(t, in, sparseGQA(512))
	growth := func(k *Kernel) float64 { return decodeAt(k, 8000) - decodeAt(k, 1024) }
	if g, d := growth(sparse), growth(dense); !(d > 0 && g < d/3) {
		t.Fatalf("decode growth from 1024 to 8000 tokens: sparse %.3g s, dense %.3g s; the sparse read is bounded at 512 selected tokens", g, d)
	}
}

// The QSA kernel is not measured, so a price that rests on it says it is a substitution, and
// a dense model's does not.
func TestSparseGQAPriceRecordsItsSubstitution(t *testing.T) {
	in := qwen36Inputs(t, "tp2")
	if substituted(mustKernel(t, in), "attention_decode_rate_sparse_gqa") {
		t.Error("a dense model's price recorded a sparse-GQA substitution")
	}
	if !substituted(withGraph(t, in, sparseGQA(2048)), "attention_decode_rate_sparse_gqa") {
		t.Error("a sparse-GQA price did not record that its rate is a substitution")
	}
}

// A measured rate for the kernel replaces the stand-in and ends the substitution.
func TestAMeasuredSparseGQARateReplacesTheStandIn(t *testing.T) {
	in := qwen36Inputs(t, "tp2")
	stand := withGraph(t, in, sparseGQA(512))
	// Register the sliding-window rate entry under the new kind's name, fifty times faster.
	fast := in
	fast.Coefficients = slices.Clone(in.Coefficients)
	found := false
	for i, s := range fast.Coefficients {
		c := *s
		c.Coefficients = slices.Clone(s.Coefficients)
		for _, e := range s.Coefficients {
			if e.Name == "attention_decode_rate_swa" {
				e.Name, e.Value = "attention_decode_rate_sparse_gqa", e.Value*50
				c.Coefficients = append(c.Coefficients, e)
				found = true
			}
		}
		fast.Coefficients[i] = &c
	}
	if !found {
		t.Fatal("the registry has no attention_decode_rate_swa entry to copy: a skip here would pass unseen (X16)")
	}
	measured := withGraph(t, fast, sparseGQA(512))
	if substituted(measured, "attention_decode_rate_sparse_gqa") {
		t.Error("the stand-in was recorded although the registry carries a rate")
	}
	if a, b := decodeAt(stand, 8000), decodeAt(measured, 8000); a == b {
		t.Errorf("a registered sparse-GQA rate did not change the price (%.6g s)", a)
	}
}

// A short convolution's window of inputs is per-sequence state the pool must hold: it grows
// with the dilated width, and it is replicated across tensor-parallel ranks, not divided.
func TestShortConvStateIsPerSequenceAndReplicated(t *testing.T) {
	base := func(tp string) int64 { return mustKernel(t, qwen36Inputs(t, tp)).SequenceFixedBytes() }
	extra := func(tp string, width, dilation int) int64 {
		in := qwen36Inputs(t, tp)
		return withGraph(t, in, withShortConv(width, dilation)).SequenceFixedBytes() - base(tp)
	}
	e1, e3 := extra("tp2", 4, 1), extra("tp2", 4, 3)
	if e1 <= 0 {
		t.Fatalf("a short convolution added %d bytes of per-sequence state", e1)
	}
	if e3 != 3*e1 {
		t.Errorf("state at dilation 3 is %d B, want 3 x %d B (the window is (kernel-1) x dilation inputs)", e3, e1)
	}
	if w := extra("tp2", 2, 1); 3*w != e1 {
		t.Errorf("state at kernel 2 is %d B, kernel 4 is %d B; want 1:3", w, e1)
	}
	if t4 := extra("tp4", 4, 3); t4 != e3 {
		t.Errorf("state at tp4 is %d B, at tp2 %d B; the engine replicates it across ranks", t4, e3)
	}
}

func TestShortConvPriceRecordsItsSubstitution(t *testing.T) {
	in := qwen36Inputs(t, "tp2")
	if !substituted(withGraph(t, in, withShortConv(4, 3)), "recurrent_decode_rate_short_conv") {
		t.Error("a short_conv layer's price did not record that its step is not charged")
	}
}
