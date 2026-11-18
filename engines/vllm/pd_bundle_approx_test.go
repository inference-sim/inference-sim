// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// The Kimi and DeepSeek P/D recipes state backends no allowed table measures. The field list prices each
// stated case as an execution mode substituted, named on the point and never exact; every other value, and
// a part below compute capability 10, stays refused.
func TestThePDBundleSettingsAreExecutionModesSubstituted(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	find := func(name string) *fields.Field {
		for i := range l.Fields {
			if l.Fields[i].Name == name {
				return &l.Fields[i]
			}
		}
		t.Fatalf("%s is not in the field list", name)
		return nil
	}
	for _, c := range []struct {
		field, value, cc string
		covered          bool
	}{
		{"all2all_backend", "flashinfer_nvlink_one_sided", "10.0", true},
		{"all2all_backend", "flashinfer_nvlink_one_sided", "9.0", false},
		{"all2all_backend", "deepep_low_latency", "10.0", false},
		{"VLLM_USE_NCCL_SYMM_MEM", "1", "10.0", true},
		{"VLLM_USE_NCCL_SYMM_MEM", "2", "10.0", false},
		{"compilation_config.custom_ops", "+quant_fp8,+rms_norm,+rotary_embedding", "10.0", true},
		{"compilation_config.custom_ops", "+something_else", "10.0", false},
		{"attention_config.mla_prefill_backend", "TRTLLM_RAGGED", "10.3", true},
		{"attention_config.mla_prefill_backend", "FLASHINFER", "10.0", false},
		{"moe_backend", "marlin", "10.0", true},
		{"moe_backend", "marlin", "9.0", true},
		{"moe_backend", "triton", "10.0", false},
		{"compilation_config.mode", "0", "10.0", true},
		{"compilation_config.mode", "3", "10.0", false},
	} {
		f := find(c.field)
		facts := fields.Facts{Value: c.value, ComputeCapability: c.cc}
		if _, ok := fields.Approximate(f.Estimation, facts); ok != c.covered {
			t.Errorf("%s=%s on %s: approximated %v, want %v", c.field, c.value, c.cc, ok, c.covered)
		}
		if fields.Exact(f.Estimation, facts) {
			t.Errorf("%s=%s on %s: called an exact stand-in", c.field, c.value, c.cc)
		}
	}
}
