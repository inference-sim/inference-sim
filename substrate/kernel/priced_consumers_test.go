package latencykernel

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/load"
)

// reportedOf prices the inputs and returns the Result's selected coefficients by name.
func reportedOf(t *testing.T, in Inputs) map[string]result.SelectedCoefficient {
	t.Helper()
	k := mustKernel(t, in)
	dd, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	k.RecordSelected(dd)
	out := map[string]result.SelectedCoefficient{}
	for _, c := range dd.Report().Result().Coefficients {
		out[c.Name] = c
	}
	return out
}

// Priced names the kernel that actually consumed an entry, after the kernel's fallback. The
// non-latent prefill is priced from the trtllm-gen lane when its whole family (floor and work
// scale) is carried exact and from the part-wide pair otherwise, and the report says which entry
// priced it on which kernel: the lane's, with a bf16 cache; the part-wide pair, with the lane
// incomplete or the cache float16, and then not the lane, which was not consumed.
func TestPricedNamesTheKernelThatConsumedTheEntryAfterFallback(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	raw := laneCase(t, d, "b200")
	implB := asKey(laneBF16)
	wide := map[string]plan.KernelImpl{"attention_prefill_floor": laneBF16, "attention_prefill_work_scale": laneBF16}
	keyed := keyMeasured(keyLaneEntries(raw, map[string]plan.KernelImpl{"bf16": laneBF16, "fp8": laneFP8}), wide)
	at := func(in Inputs, dtype plan.Dtype) Inputs {
		in.Operations = prefillOps(t, in, bf16Cache, dtype, laneBF16)
		return in
	}
	priced := func(r map[string]result.SelectedCoefficient, name string) []coefficient.KernelImpl {
		c, ok := r[name]
		if !ok {
			return nil
		}
		return c.Priced
	}
	laneFloor, laneScale := trtllmGenPrefillFloor+"bf16", trtllmGenPrefillScale+"bf16"
	one := []coefficient.KernelImpl{implB}

	// A whole lane at a bf16 cache prices the prefill: the lane's entries are reported on the
	// kernel, and the part-wide pair, which no operation reads, is not.
	r := reportedOf(t, at(keyed, "bf16"))
	if !reflect.DeepEqual(priced(r, laneFloor), one) || !reflect.DeepEqual(priced(r, laneScale), one) {
		t.Errorf("a whole bf16 lane: priced %v and %v, want the kernel that runs the prefill", priced(r, laneFloor), priced(r, laneScale))
	}
	if p := priced(r, "attention_prefill_floor"); len(p) != 0 {
		t.Errorf("the part-wide floor is reported priced on %v although the lane priced the prefill", p)
	}

	// A float16 cache is not priced from the bf16-measured lane: the lane is not reported, and the
	// part-wide pair that consumed the prefill is, on the kernel that runs it.
	r = reportedOf(t, at(keyed, "fp16"))
	for _, n := range []string{laneFloor, laneScale} {
		if p := priced(r, n); len(p) != 0 {
			t.Errorf("%s is reported priced on %v for a float16 cache", n, p)
		}
	}
	for _, n := range []string{"attention_prefill_floor", "attention_prefill_work_scale"} {
		if got := priced(r, n); !reflect.DeepEqual(got, one) {
			t.Errorf("%s priced the float16 prefill on %v, want %v", n, got, one)
		}
	}

	// A lane without its work scale is incomplete: the kernel falls back to the part-wide pair,
	// which the report names, and the lane's floor, carried but unused, is not reported.
	partial := keyed
	partial.Coefficients = dropEntries(keyed.Coefficients, laneScale)
	r = reportedOf(t, at(partial, "bf16"))
	if p := priced(r, laneFloor); len(p) != 0 {
		t.Errorf("the lane's floor is reported priced on %v although its family is incomplete", p)
	}
	for _, n := range []string{"attention_prefill_floor", "attention_prefill_work_scale"} {
		if got := priced(r, n); !reflect.DeepEqual(got, one) {
			t.Errorf("%s with an incomplete lane: priced on %v, want the fallback's kernel %v", n, got, one)
		}
	}

	// A malformed element type is refused at construction, so no entry is reported as priced by
	// a lane chosen from a label that is no dtype.
	if _, err := New(at(keyed, "bf116")); err == nil {
		t.Error("a cache of dtype bf116 was priced")
	}
}
