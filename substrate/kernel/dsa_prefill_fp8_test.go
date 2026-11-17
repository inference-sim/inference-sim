package latencykernel

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/load"
)

// withFP8PrefillScale is in with attention_prefill_work_scale_sparse_mla_kv_fp8 scoped to model
// on h200, replacing any entry of that name; a scale of 0 only removes them.
func withFP8PrefillScale(in Inputs, model string, scale float64) Inputs {
	const name = "attention_prefill_work_scale_sparse_mla_kv_fp8"
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			if e.Name != name {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		if c.Name == "cost-model-attention" && scale > 0 {
			c.Coefficients = append(c.Coefficients, coefficient.Entry{
				Name: name, Value: scale, Units: vocab.Unit("dimensionless"), Method: "measured",
				Scope: coefficient.Scope{Hardware: []string{"h200"}, Model: []string{model}},
			})
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

// withCacheCell is in with its plan's cache stored as record, elem bytes an element and a packed
// latent cell of cell bytes (0: the per-element width).
func withCacheCell(in Inputs, record string, elem, cell float64) Inputs {
	l := *in.Lowered
	l.Plan.Cache.Record, l.Plan.Cache.ElementBytes, l.Plan.Cache.LatentCellBytes = record, elem, cell
	in.Lowered = &l
	return in
}

// A GLM-5.3 prefill over an fp8_ds_mla cache is charged the measured fp8 path's cost per key
// slot, the BF16 form's times the registry's scale, not the packed cell's share of the BF16
// cost. vLLM runs that prefill through FlashMLA's FP8 decode kernel at fewer than 32 heads a
// rank, and the measured fp8 path is dearer than the BF16 one, not 656/1152 of it.
//
// The form's work term is read off the step price as the difference between two rates of the
// form (W/1e5 - W/2e5 = W/2e5), so everything else the step charges cancels.
func TestDSAPrefillAtAnFP8CacheIsChargedTheMeasuredScale(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := quantCase(t, d, "glm-5.3", "h200")
	work := func(in Inputs) float64 {
		price := func(rate float64) float64 {
			e := mustKernel(t, withSparsePrefillFamily(in, "glm-5.3", rate)).StepTime(kernel.Batch{
				DecodeThreshold: 1, Reqs: []kernel.ReqShape{
					{Scheduled: 2048, Computed: 16384, PromptLen: 16384 + 2048}}})
			return e.NoOverlap.Seconds()
		}
		return price(1e5) - price(2e5)
	}
	bf16 := withFP8PrefillScale(withCacheCell(base, "bfloat16", 2, 0), "glm-5.3", 0)
	fp8 := withFP8PrefillScale(withCacheCell(base, "fp8_ds_mla", 1, 656), "glm-5.3", 0)
	wBF16 := work(bf16)
	if !(wBF16 > 0) {
		t.Fatalf("the form's work at a BF16 cache is %v s, want more than 0", wBF16)
	}
	const scale = 1.85
	if got := work(withFP8PrefillScale(fp8, "glm-5.3", scale)) / wBF16; math.Abs(got/scale-1) > 1e-6 {
		t.Errorf("fp8_ds_mla over BF16 work with the measured scale %v: %v; want the scale", scale, got)
	}
	// Without the entry the form is used as fitted, over the cell it counts.
	if got, want := work(fp8)/wBF16, 656.0/1152; math.Abs(got/want-1) > 1e-6 {
		t.Errorf("fp8_ds_mla over BF16 work with no scale entry: %v, want the cell's share %v", got, want)
	}
	// The scale is the fp8 cell's: a BF16 cache does not read it.
	if got := work(withFP8PrefillScale(bf16, "glm-5.3", scale)) / wBF16; math.Abs(got-1) > 1e-9 {
		t.Errorf("a BF16 cache's work moves by %v with the fp8 scale entry present; want 1", got)
	}
}

// packedSparseVariant is in at a packed fp8_ds_mla latent cache, for a model with a sparse-MLA
// layer that states no cache records of its own, which is where the fp8 prefill scale is read
// (packedFP8Prefill); false where in has no such layer or does not build that way. The
// perturbation probe adds it beside each case it prices, since its first buildable setting for
// a DSA model is an unpacked cache.
func packedSparseVariant(in Inputs) (Inputs, bool) {
	k, err := New(in)
	if err != nil {
		return in, false
	}
	dsa := false
	for _, l := range k.plan.Layers {
		if l.AttnKind == model.AttentionSparseMLA && l.AttnQHeads > 0 && len(l.AttnCaches) == 0 {
			dsa = true
		}
	}
	if !dsa || in.Lowered.Plan.Cache.Record == "fp8_ds_mla" {
		return in, false
	}
	out := withCacheCell(in, "fp8_ds_mla", 1, 656)
	// The model's sparse-MLA prefill entries are keyed to the build of the kernel they were moved
	// to (D5); the pool reports running it, so the form, and with it the scale, is read.
	var run []plan.KernelImpl
	for _, set := range in.Coefficients {
		for _, e := range set.Coefficients {
			if e.Name == "attention_prefill_floor_sparse_mla" && e.Method == vocab.MethodTransferred {
				for _, i := range e.KernelImpls {
					run = append(run, plan.KernelImpl{Library: i.Library, Variant: i.Variant, CodeHash: i.CodeHash})
				}
			}
		}
	}
	if len(run) > 0 {
		out = withPrefill(out, append(append([]plan.KernelImpl(nil), in.Implementations[plan.PrimAttentionPrefill]...), run...)...)
	}
	if _, err := New(out); err != nil {
		return in, false
	}
	return out, true
}
