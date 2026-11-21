// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// Three environment variables name the directory a Hugging Face model cache lives in. The serving process
// reads that directory only to find model files before the first request (HF_HUB_CACHE: the Rust front
// end's file resolution and the page-cache eviction after a DeepSeek-V4.1 weight load), or not at all
// (HUGGINGFACE_HUB_CACHE and TRANSFORMERS_CACHE are named nowhere in vLLM's serving process). A step, an
// admission and a KV budget are the same wherever the cache is. Each is proven inert, so a harness records
// it and states nothing to the engine, rather than refusing the point for a field the release does not
// list (HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE) or does not model (HF_HUB_CACHE).
func TestTheModelCacheLocationVariablesAreProvenInert(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"} {
		var f *fields.Field
		for i := range l.Fields {
			if l.Fields[i].Name == name && l.Fields[i].Source == fields.SourceEnv {
				f = &l.Fields[i]
			}
		}
		if f == nil {
			t.Errorf("%s: not in the list, so the harness refuses it as a field the release does not have", name)
			continue
		}
		if f.Status != fields.StatusProvenInert {
			t.Errorf("%s: status %s, want proven_inert", name, f.Status)
		}
	}
}
