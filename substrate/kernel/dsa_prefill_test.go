package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/load"
)

// sparsePrefillFamily is the four names of the sparse-MLA measured prefill form.
var sparsePrefillFamily = []string{
	"attention_prefill_floor_sparse_mla", "attention_prefill_per_request_sparse_mla",
	"attention_prefill_per_token_sparse_mla", "attention_prefill_rate_sparse_mla",
}

// withSparsePrefillFamily is in with a measured sparse-MLA prefill form scoped to model on
// h200, replacing any entry of the same names, at the given rate (bytes per microsecond).
func withSparsePrefillFamily(in Inputs, model string, rate float64) Inputs {
	values := map[string]float64{
		sparsePrefillFamily[0]: 20, sparsePrefillFamily[1]: 1, sparsePrefillFamily[2]: 0.01,
		sparsePrefillFamily[3]: rate,
	}
	units := map[string]string{
		sparsePrefillFamily[0]: "us_per_transfer", sparsePrefillFamily[1]: "us_per_request",
		sparsePrefillFamily[2]: "us_per_token", sparsePrefillFamily[3]: "bytes_per_us",
	}
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			if _, ours := values[e.Name]; !ours {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		if c.Name == "cost-model-attention" {
			for _, name := range sparsePrefillFamily {
				c.Coefficients = append(c.Coefficients, coefficient.Entry{
					Name: name, Value: values[name], Units: vocab.Unit(units[name]), Method: "measured",
					Scope: coefficient.Scope{Hardware: []string{"h200"}, Model: []string{model}},
				})
			}
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

// A DSA layer (GLM-5.3: an index top-k over an uncompressed stream, its latent kept in the
// engine's KV cache, no cache stated in the graph) is priced by the measured sparse-MLA
// prefill form when the registry carries one for it, and that form counts the keys the top-k
// selects. Before, the form was skipped for such a layer, because the graph states no cache
// record, and the selection was empty, because the layer states no compression ratio; so the
// form's rate moved nothing and a long cached prefix read nothing.
func TestDSAPrefillUsesTheMeasuredSparseForm(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := quantCase(t, d, "glm-5.3", "h200")
	price := func(in Inputs, prefix int) float64 {
		e := mustKernel(t, in).StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
			{Scheduled: 512, Computed: prefix, PromptLen: prefix + 512}}})
		return e.NoOverlap.Seconds()
	}
	slow := withSparsePrefillFamily(base, "glm-5.3", 1e5)
	fast := withSparsePrefillFamily(base, "glm-5.3", 2e5)

	// The form is used: halving its cost per byte lowers the price of a prefill over a cached
	// prefix.
	if s, f := price(slow, 16384), price(fast, 16384); !(f < s) {
		t.Errorf("512 tokens over 16,384 cached: price at rate 2e5 is %v s, at 1e5 %v s; "+
			"the measured form's rate should move it", f, s)
	}
	// The selection is the top-k of the context, not nothing: a prefix past the top-k (2,048)
	// costs more than no prefix, and the extra is the form's work, so it halves when the rate
	// doubles.
	extraSlow := price(slow, 16384) - price(slow, 0)
	extraFast := price(fast, 16384) - price(fast, 0)
	if !(extraSlow > 0) {
		t.Fatalf("a 16,384-token cached prefix adds %v s to a 512-token prefill, want more than 0",
			extraSlow)
	}
	if extraFast >= extraSlow*0.75 {
		t.Errorf("the prefix's extra cost is %v s at rate 2e5 and %v s at 1e5: the work over "+
			"the selected keys should scale with the form's rate", extraFast, extraSlow)
	}
	// Past the top-k every query reads the same 2,048 keys, so a longer prefix adds no form
	// work: the step at 65,536 cached costs what it does at 16,384, within what the rest of the
	// layer charges for context.
	if far, near := price(slow, 65536)-price(fast, 65536), price(slow, 16384)-price(fast, 16384); far > near*1.01 {
		t.Errorf("the form's share grows from %v s at 16,384 cached to %v s at 65,536; past the "+
			"top-k it should not", near, far)
	}
}
