// Original to blis-vllm.

package dplb

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// The modes are vLLM's three, and an unstated one is internal. They are blis-schemas' own
// spellings, so a scenario's parallel.data_parallel_load_balancing means the same mode here.
func TestModesAreVLLMsThree(t *testing.T) {
	for _, c := range []struct {
		mode      string
		valid     bool
		eachRank  bool
		schemaHas deployment.DPLoadBalancing
	}{
		{"", true, false, ""},
		{Internal, true, false, deployment.DPLBInternal},
		{Hybrid, true, false, deployment.DPLBHybrid},
		{External, true, true, deployment.DPLBExternal},
		{"round-robin", false, false, ""},
		{"External", false, false, ""},
	} {
		if got := ValidMode(c.mode); got != c.valid {
			t.Errorf("ValidMode(%q) = %v, want %v", c.mode, got, c.valid)
		}
		if got := EachRankIsAnEndpoint(c.mode); got != c.eachRank {
			t.Errorf("EachRankIsAnEndpoint(%q) = %v, want %v", c.mode, got, c.eachRank)
		}
		if c.schemaHas != "" && string(c.schemaHas) != c.mode {
			t.Errorf("mode %q is spelt %q in blis-schemas", c.mode, c.schemaHas)
		}
		if c.valid != deployment.DPLoadBalancing(c.mode).Valid() {
			t.Errorf("mode %q: valid here %v, in blis-schemas %v", c.mode, c.valid, deployment.DPLoadBalancing(c.mode).Valid())
		}
	}
}
