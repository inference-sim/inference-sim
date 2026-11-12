package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/load"
)

// On SM90 a dense MXFP8 linear runs vLLM's Marlin W8A16 kernel. The registry carries its ramp as a
// `transferred` pair moved from the same part's measured Marlin grouped GEMM, beside the `assumed`
// BF16-envelope pair that is its named upper bound (design VAL-4: the most trusted rank is used).
// These tests price MiniMax-M3 MXFP8 on the registry as committed and on the same registry with
// the transferred entries removed, which is the bound alone.

func isDenseMarlinMXFP8(e coefficient.Entry) bool {
	return e.Name == "gemm_eps_max_w8a16_mxfp8" || e.Name == "gemm_m_half_w8a16_mxfp8"
}

// withoutTransferredDenseRamp drops the transferred entries of the dense MXFP8 ramp.
func withoutTransferredDenseRamp(in Inputs) Inputs {
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			if isDenseMarlinMXFP8(e) && e.Method == "transferred" {
				continue
			}
			c.Coefficients = append(c.Coefficients, e)
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

func prefillSeconds(k *Kernel, tokens int) float64 {
	return k.StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: tokens, PromptLen: tokens}}}).Overlap.Seconds()
}

func decodeSeconds(k *Kernel, n int) float64 {
	reqs := make([]kernel.ReqShape, n)
	for i := range reqs {
		reqs[i] = kernel.ReqShape{Scheduled: 1, Computed: 1023, PromptLen: 1024}
	}
	return k.StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: reqs}).Overlap.Seconds()
}

func TestHopperDenseMXFP8PricesOnTheTransferredRampNotTheBF16Bound(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"h100", "h200"} {
		in := quantCase(t, d, "minimax-m3-mxfp8", chip)
		got, bound := mustKernel(t, in), mustKernel(t, withoutTransferredDenseRamp(in))
		var last float64
		for _, n := range []int{512, 2048, 8192} {
			g, b := prefillSeconds(got, n), prefillSeconds(bound, n)
			t.Logf("%s prefill %d tokens: ramp %.4g s, bound %.4g s, ratio %.3f", chip, n, g, b, g/b)
			if g <= b {
				t.Errorf("%s: a %d-token prefill costs %.4g s on the transferred ramp and %.4g s at the BF16 bound; the ramp reaches a fraction of the peak, so it must cost more", chip, n, g, b)
			}
			if r := g / b; r < last {
				t.Errorf("%s: the gap to the bound falls from %.3f to %.3f as the prefill grows; the GEMM share and its deficit both grow", chip, last, r)
			} else {
				last = r
			}
		}
		// A decode step is bound mostly by the FP8 weight read, which the ramp does not touch. The
		// ramp's knee is lower than the bound's (15 tokens against 170 on h100), so at a handful of
		// rows it is slightly FASTER; it must not move a decode step by more than a tenth.
		g, b := decodeSeconds(got, 4), decodeSeconds(bound, 4)
		t.Logf("%s decode 4 requests: ramp %.4g s, bound %.4g s", chip, g, b)
		if g < b*0.90 || g > b*1.10 {
			t.Errorf("%s: a 4-request decode step costs %.4g s on the ramp and %.4g s at the bound; the weight read dominates, so the two must agree within a tenth", chip, g, b)
		}
	}
}

func TestBlackwellDenseMXFP8IsPricedExactlyAsBefore(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"b200", "b300"} {
		in := quantCase(t, d, "minimax-m3-mxfp8", chip)
		got, before := mustKernel(t, in), mustKernel(t, withoutTransferredDenseRamp(in))
		for _, n := range []int{512, 8192} {
			if g, b := prefillSeconds(got, n), prefillSeconds(before, n); g != b {
				t.Errorf("%s: a %d-token prefill costs %v s with the Hopper entries in the registry and %v s without them", chip, n, g, b)
			}
		}
	}
}

func TestTheProvenanceNamesTheDenseRampAsTransferredAndUnkeyed(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"h100", "h200"} {
		k := mustKernel(t, quantCase(t, d, "minimax-m3-mxfp8", chip))
		seen := map[string]kernel.CoefficientOrigin{}
		for _, o := range k.Provenance() {
			if o.Name == "gemm_eps_max_w8a16_mxfp8" || o.Name == "gemm_m_half_w8a16_mxfp8" {
				seen[o.Name] = o
			}
		}
		for name, o := range seen {
			if o.Method != "transferred" {
				t.Errorf("%s %s is reported as %q, want transferred", chip, name, o.Method)
			}
			if o.KernelMatch != string(coefficient.KernelUnkeyed) {
				t.Errorf("%s %s kernel match is %q; no implementation is reported for a GEMM, so it is unkeyed and must not read as exact", chip, name, o.KernelMatch)
			}
		}
		if len(seen) != 2 {
			t.Fatalf("%s: provenance lists %d of the dense ramp's two coefficients", chip, len(seen))
		}
	}
}
