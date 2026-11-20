// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// vLLM v0.33.0's linear_backend=flashinfer_cutlass on SM10x runs FlashInferCutlassNvFp4LinearKernel
// for the NVFP4 linears, which no allowed table measures; the registry's NVFP4 ramp measures the
// CuteDSL kernel auto selects (blis-vllm#80). The field list prices that case as linear_backend auto,
// an execution mode substituted, never as an exact stand-in; every other value and part stays refused.
func TestFlashInferCutlassLinearsOnSM10xArePricedAsAnExecutionModeSubstituted(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"linear_backend", "kernel_config.linear_backend"} {
		var f *fields.Field
		for i := range l.Fields {
			if l.Fields[i].Name == name {
				f = &l.Fields[i]
			}
		}
		if f == nil {
			t.Fatalf("%s is not in the field list", name)
		}
		for _, c := range []struct {
			value, cc string
			covered   bool
		}{
			{"flashinfer_cutlass", "10.0", true}, {"flashinfer_cutlass", "10.3", true},
			{"flashinfer_cutlass", "9.0", false}, {"flashinfer_cutlass", "12.0", false},
			{"marlin", "10.0", false}, {"cutlass", "10.0", false},
		} {
			facts := fields.Facts{Value: c.value, ComputeCapability: c.cc}
			a, ok := fields.Approximate(f.Estimation, facts)
			if ok != c.covered {
				t.Errorf("%s=%s on %s: approximated %v, want %v", name, c.value, c.cc, ok, c.covered)
			}
			if ok && a.Mode["kernel_config.linear_backend"] != "auto" {
				t.Errorf("%s=%s on %s: priced as %v, want linear_backend auto", name, c.value, c.cc, a.Mode)
			}
			if fields.Exact(f.Estimation, facts) {
				t.Errorf("%s=%s on %s: called an exact stand-in", name, c.value, c.cc)
			}
		}
	}
}
