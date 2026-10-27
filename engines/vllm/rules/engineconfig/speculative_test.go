// Moved from inference-sim/blis-schemas/spec/deployment/speculative_offload_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package engineconfig

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The speculative knobs a launch states beside method and K: an external drafter by catalog
// name (blis-schemas#97), the drafter's KV dtype, and the sampling knobs. Acceptance is
// supplied (inference-sim#2021 B3), so the sampling knobs are recorded, not acted on; their
// closed vocabularies are still checked, since a misspelled one is a launch vLLM refuses.
func TestSpeculativeKnobs(t *testing.T) {
	spec := func(s Speculative) validated {
		s.Method, s.NumSpecTokens = "eagle3", 3
		return engineOnly(Engine{Speculative: &s})
	}
	for _, c := range []struct {
		name string
		s    Speculative
		bad  string
	}{
		{"none stated", Speculative{}, ""},
		{"an external drafter", Speculative{DraftModel: "minimax-m3-eagle3-gqa"}, ""},
		{"a blank drafter name", Speculative{DraftModel: "  "}, "speculative.draft_model"},
		{"drafter kv dtype", Speculative{KVCacheDType: "fp8"}, ""},
		{"probabilistic drafting", Speculative{DraftSampleMethod: "probabilistic"}, ""},
		{"an unknown draft sample method", Speculative{DraftSampleMethod: "beam"}, "speculative.draft_sample_method"},
		{"block rejection sampling", Speculative{RejectionSampleMethod: "block"}, ""},
		{"an unknown rejection sample method", Speculative{RejectionSampleMethod: "strict"}, "speculative.rejection_sample_method"},
		{"adaptive verification", Speculative{EnableAdaptiveVerification: ptrBool(true)}, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := spec(c.s).Validate()
			if c.bad == "" {
				if !p.OK() {
					t.Fatalf("refused:\n%s", p.Error())
				}
				return
			}
			if p.OK() || !strings.Contains(p.Error(), ".engine."+c.bad) {
				t.Fatalf("want a finding on engine.%s, got:\n%s", c.bad, p.Error())
			}
		})
	}
	// An absent draft_model is today's document: the target graph's own speculator.
	var s Speculative
	if err := yaml.Unmarshal([]byte("method: dspark\nnum_spec_tokens: 5\n"), &s); err != nil || s.DraftModel != "" {
		t.Fatalf("a document without draft_model loaded as %+v (%v)", s, err)
	}
}
