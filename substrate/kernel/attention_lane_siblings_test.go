package latencykernel

import (
	"slices"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/load"
)

// A lane coefficient can have one entry per build: one keyed to the build a run measured and one
// moved to the engine's pin (D5, VAL-8). The lookup refuses the entry of the build that does not
// run and admits the one of the build that does, and the lane prices from the admitted one: the
// refusal of a sibling build says nothing against an exact match. Without that, keying the
// registry by build would switch the lane off on every part that carries two builds.
func TestALaneEntryOfAnotherBuildBesideTheRunningOneDoesNotHideIt(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	raw := laneCase(t, d, "b200")
	pooled := pricesOf(mustKernel(t, raw))
	one := keyedLane(raw, laneBF16)
	if pricesOf(mustKernel(t, one)) == pooled {
		t.Fatal("the reported bf16 build runs and the lane is not read")
	}
	other := coefficient.KernelImpl{Library: "flashinfer", Variant: "fixture-trtllm-prefill-bf16", CodeHash: "fedcba987654"}
	two := withSiblingBuild(one, other)
	if got, want := pricesOf(mustKernel(t, two)), pricesOf(mustKernel(t, one)); got != want {
		t.Errorf("an entry keyed to another build beside the running one moves the prices: %v, want %v", got, want)
	}
}

// withSiblingBuild adds, beside each lane entry keyed to a build, an entry of the same value keyed
// to other, as a registry keyed by build carries one entry per build.
func withSiblingBuild(in Inputs, other coefficient.KernelImpl) Inputs {
	names := append(laneNames("bf16"), laneNames("fp8")...)
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for _, e := range s.Coefficients {
			if slices.Contains(names, e.Name) && len(e.KernelImpls) > 0 {
				sib := e
				sib.KernelImpls = []coefficient.KernelImpl{other}
				c.Coefficients = append(c.Coefficients, sib)
			}
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}
