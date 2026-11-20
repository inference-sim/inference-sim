// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// enforce_eager makes vLLM run every step with no CUDA graph, which the lowering prices (a stated
// cudagraph_mode NONE), and also turns torch.compile off, which the lowering's launch counts do not
// follow (blis-vllm#59). The field list prices a stated true with no graph and the compiled model's fusions, an execution
// mode substituted, never as an exact stand-in; false and any other value stay unapproximated.
func TestEnforceEagerIsPricedAsAnExecutionModeSubstituted(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	var f *fields.Field
	for i := range l.Fields {
		if l.Fields[i].Name == "enforce_eager" {
			f = &l.Fields[i]
		}
	}
	if f == nil {
		t.Fatal("enforce_eager is not in the field list")
	}
	for _, c := range []struct {
		value   string
		covered bool
	}{{"true", true}, {"True", true}, {"false", false}, {"", false}} {
		facts := fields.Facts{Value: c.value, ComputeCapability: "10.0"}
		a, ok := fields.Approximate(f.Estimation, facts)
		if ok != c.covered {
			t.Errorf("enforce_eager=%q: approximated %v, want %v", c.value, ok, c.covered)
		}
		if ok && a.Mode["enforce_eager"] != true {
			t.Errorf("enforce_eager=%q: priced as %v, want enforce_eager true (no graph)", c.value, a.Mode)
		}
		if fields.Exact(f.Estimation, facts) {
			t.Errorf("enforce_eager=%q: called an exact stand-in", c.value)
		}
	}
}
