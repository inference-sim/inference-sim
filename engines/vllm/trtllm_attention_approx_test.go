// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// A stated use_trtllm_attention=true on SM10x forces FlashInfer's trtllm-gen kernels. The engine
// reports no attention implementation, so the registry's pooled lanes price the step; the field list
// prices that case as the auto-detected choice, an execution mode substituted, never as an exact
// stand-in. false, other parts and unstated values stay refused (blis-vllm#80).
func TestAStatedTRTLLMAttentionOnSM10xIsPricedAsAnExecutionModeSubstituted(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	var f *fields.Field
	for i := range l.Fields {
		if l.Fields[i].Name == "attention_config.use_trtllm_attention" {
			f = &l.Fields[i]
		}
	}
	if f == nil {
		t.Fatal("attention_config.use_trtllm_attention is not in the field list")
	}
	for _, c := range []struct {
		value, cc string
		covered   bool
	}{{"true", "10.0", true}, {"true", "10.3", true}, {"true", "9.0", false}, {"true", "12.0", false}, {"false", "10.0", false}} {
		facts := fields.Facts{Value: c.value, ComputeCapability: c.cc}
		_, ok := fields.Approximate(f.Estimation, facts)
		if ok != c.covered {
			t.Errorf("use_trtllm_attention=%s on %s: approximated %v, want %v", c.value, c.cc, ok, c.covered)
		}
		if fields.Exact(f.Estimation, facts) {
			t.Errorf("use_trtllm_attention=%s on %s: called an exact stand-in", c.value, c.cc)
		}
	}
}
