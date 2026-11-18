package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/load"
)

// withMLAPrefill is in with its model's mla attention nodes' prefill widths set (or, with zeros, cleared:
// the graph as it was before the widths existed) and, with expandable false, its kv_b_proj nodes widened by
// one column so no GEMM matches the latent expansion's shape and the cached tokens' re-expansion is not
// priced.
func withMLAPrefill(in Inputs, qk, v int, expandable bool) Inputs {
	g := *in.Model
	g.LayerKinds = nil
	for _, k := range in.Model.LayerKinds {
		nodes := append([]model.Node(nil), k.Nodes...)
		for i := range nodes {
			n := &nodes[i]
			if n.Op == model.OpAttention && n.AttentionKind == model.AttentionMLA {
				n.PrefillQKHeadDim, n.PrefillVHeadDim = qk, v
			}
			if !expandable && n.Op == model.OpGEMM && n.Role == "kv_b_proj" {
				n.N++
			}
		}
		k.Nodes = nodes
		g.LayerKinds = append(g.LayerKinds, k)
	}
	in.Model = &g
	return in
}

func mlaStep(t *testing.T, in Inputs, scheduled, computed int) float64 {
	t.Helper()
	e := mustKernel(t, in).StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
		{Scheduled: scheduled, Computed: computed, PromptLen: scheduled + computed}}})
	return e.NoOverlap.Seconds()
}

// vLLM prefills a dense latent layer in the multi-head form (q and k 192 wide, v 128 for DeepSeek-V3 and
// Kimi-K2.5) and runs the absorbed form, at the 576-wide cache, for decode only. A prefill priced at the
// cache width charges 3.6 times the attention work, which the table rows of AISimulate's MLA module show
// (Kimi-K2.5 on GB200, 8192 tokens: the attention core alone was 423 ms of an 800 ms step). With the
// widths stated the same step is cheaper, and the saving is the quadratic attention term, so it grows
// faster than the prompt.
func TestMLAPrefillAttentionIsChargedAtThePrefillWidths(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	stated := quantCase(t, d, "kimi-k2.5", "h200")
	var qk, v int
	for _, k := range stated.Model.LayerKinds {
		for _, n := range k.Nodes {
			if n.Op == model.OpAttention && n.AttentionKind == model.AttentionMLA {
				qk, v = n.PrefillQKHeadDim, n.PrefillVHeadDim
			}
		}
	}
	if qk != 192 || v != 128 {
		t.Fatalf("the catalog's kimi-k2.5 states prefill widths %d and %d, want 192 and 128", qk, v)
	}
	cleared := withMLAPrefill(stated, 0, 0, true)
	saving := func(tokens int) float64 {
		return mlaStep(t, cleared, tokens, 0) - mlaStep(t, stated, tokens, 0)
	}
	s2k, s8k := saving(2048), saving(8192)
	if !(s2k > 0) {
		t.Errorf("a 2048-token prefill costs %v s more at the cache width than at the prefill widths, want a saving", s2k)
	}
	if !(s8k > 8*s2k) {
		t.Errorf("the saving is %v s at 8192 tokens and %v s at 2048: attention is quadratic, so 4 times the prompt "+
			"should save over 8 times as much", s8k, s2k)
	}
	if full, part := mlaStep(t, cleared, 8192, 0), mlaStep(t, stated, 8192, 0); !(part < 0.8*full) {
		t.Errorf("8192-token prefill: %v s at the prefill widths against %v s at the cache width; the attention core "+
			"is over half the step at the cache width, so the saving should pass a fifth of it", part, full)
	}
}

// Decode keeps the cache width: a step that schedules one token per request is the same with or without
// the prefill widths.
func TestMLADecodeIsUnchangedByThePrefillWidths(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	stated := quantCase(t, d, "kimi-k2.5", "h200")
	cleared := withMLAPrefill(stated, 0, 0, true)
	decode := func(in Inputs) float64 {
		e := mustKernel(t, in).StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
			{Scheduled: 1, Computed: 8191, PromptLen: 8192}, {Scheduled: 1, Computed: 4095, PromptLen: 4096}}})
		return e.NoOverlap.Seconds()
	}
	if a, b := decode(stated), decode(cleared); a != b {
		t.Errorf("decode step %v s with the prefill widths, %v s without: decode runs the absorbed form and is unchanged", a, b)
	}
}

// A request resuming on a cached prefix expands the prefix's latents through kv_b_proj every step (the
// absorbed form never does): the step pays for it per cached token, and nothing when there is no prefix.
func TestMLAPrefillExpandsTheCachedLatents(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	stated := quantCase(t, d, "kimi-k2.5", "h200")
	unpriced := withMLAPrefill(stated, 192, 128, false)
	extra := func(prefix int) float64 {
		return mlaStep(t, stated, 512, prefix) - mlaStep(t, unpriced, 512, prefix)
	}
	// The widened projection is priced at its own (one column wider) shape, so no-prefix agrees to a microsecond, not to the bit.
	if e := extra(0); e > 1e-6 || e < -1e-6 {
		t.Errorf("with no cached prefix the expansion costs %v s, want 0", e)
	}
	e8, e16, e32 := extra(8192), extra(16384), extra(32768)
	if !(e8 > 0) {
		t.Fatalf("512 tokens over 8192 cached pay %v s to expand the cached latents, want a positive charge", e8)
	}
	if !(e16 > 1.5*e8 && e32 > 1.5*e16) {
		t.Errorf("the expansion is %v, %v and %v s at 8192, 16384 and 32768 cached tokens: it should grow with the "+
			"cached tokens (about twice per doubling)", e8, e16, e32)
	}
}

// The graph states the widths of every dense latent layer whose config it was derived from, and not a
// sparse one's.
func TestCatalogStatesPrefillWidthsOnMLAOnly(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	sawMLA, sawSparse := false, false
	for _, name := range d.Models() {
		g, err := d.Model(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, k := range g.LayerKinds {
			for _, n := range k.Nodes {
				if n.Op != model.OpAttention {
					continue
				}
				has := n.PrefillQKHeadDim != 0 || n.PrefillVHeadDim != 0
				switch n.AttentionKind {
				case model.AttentionMLA:
					sawMLA = true
					if !has {
						t.Errorf("%s: an mla node states no prefill widths; its prefill would be priced at the cache width", name)
					}
				case model.AttentionSparseMLA:
					sawSparse = true
					fallthrough
				default:
					if has {
						t.Errorf("%s: a %s node states prefill widths", name, n.AttentionKind)
					}
				}
			}
		}
	}
	if !sawMLA || !sawSparse {
		t.Errorf("non-vacuity: the catalog has mla %v and sparse_mla %v attention", sawMLA, sawSparse)
	}
}

// SGLang's prefill of a dense latent layer is the multi-head form or the absorbed one by backend, prefix and
// graph capture (attention_backend_handler.py at v0.5.23); the kernel does not model that choice, so a scenario
// of another engine keeps the cache width and its prices are those of a graph with no prefill widths.
func TestMLAPrefillWidthsAreVLLMsOnly(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	stated := quantCase(t, d, "kimi-k2.5", "h200")
	other := stated
	sc := *stated.Scenario
	sc.Engine = &scenario.Engine{Kind: "sglang", Version: sc.EngineRelease()}
	other.Scenario = &sc
	cleared := withMLAPrefill(other, 0, 0, true)
	for _, c := range []struct{ scheduled, computed int }{{8192, 0}, {512, 16384}} {
		if a, b := mlaStep(t, other, c.scheduled, c.computed), mlaStep(t, cleared, c.scheduled, c.computed); a != b {
			t.Errorf("sglang scenario, %d scheduled over %d cached: %v s with the widths, %v s without; an engine other than vLLM keeps the cache width",
				c.scheduled, c.computed, a, b)
		}
	}
}
