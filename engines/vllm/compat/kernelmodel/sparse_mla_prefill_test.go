// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A pool whose plan states the sparse-MLA kernel for the prefill prices every prefill step from the family keyed
// to it, though vLLM runs a prefill of at most the top-k tokens as dense multi-head attention. The pool says so,
// with the top-k; a model that does not run FlashInfer's SM90 sparse-MLA backend (GLM-5 selects FlashAttention
// sparse first) states no kernel and lists nothing.
func TestASparseMLAPrefillPricedFromTheFamilyNamesTheDenseMHARoute(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", "glm5-h200-tp8.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "model: glm-5\n") {
		t.Fatal("the scenario no longer names glm-5, which this test swaps for glm-5.3-flash")
	}
	open := func(t *testing.T, name string) *Model {
		t.Helper()
		r := repos()
		r.Scenarios = t.TempDir()
		body := strings.Replace(string(raw), "model: glm-5\n", "model: "+name+"\n", 1)
		if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		m, err := Open("v.yaml", r)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	listed := func(m *Model) (out []string) {
		for _, s := range m.Provenance().Substitutions {
			if s.Setting == "sparse_mla_prefill_path" {
				out = append(out, s.Reason)
			}
		}
		return
	}
	got := listed(open(t, "glm-5.3-flash"))
	if len(got) != 1 || !strings.Contains(got[0], "at most 2048 tokens") || !strings.Contains(got[0], "dense multi-head") {
		t.Errorf("GLM-5.3-Flash on an H200: substitutions %q, want one naming the dense MHA route at 2048 tokens", got)
	}
	if got := listed(open(t, "glm-5")); len(got) != 0 {
		t.Errorf("GLM-5 states no sparse-MLA kernel and lists %q", got)
	}
}
