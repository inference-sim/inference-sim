// Original to blis-vllm.

package sched

import "testing"

// n-gram speculation drafts from the prompt on the CPU and is not an EAGLE-family method
// (SpeculativeConfig.use_eagle is false for it, vllm/config/speculative.py:1944-1948@5df63a72), so
// the scheduler keeps the last cached block that EAGLE drops, reserves no KV lookahead and no
// prefill lookahead, and adds no draft slots (vllm/config/vllm.py:625-665@5df63a72;
// speculative.py:1895-1926). The tier-2 scenarios spec-ngram and spec-ngram-prefix-hit show the
// same against vLLM's own Scheduler; this checks the numbers they rest on, beside the EAGLE
// family's, so a method table that sorted ngram with the EAGLE methods fails here.
func TestNgramResolvesNoLookaheadAndKeepsTheLastBlock(t *testing.T) {
	got := ResolveSpec(SpecOptions{Method: "ngram", NumTokens: 3})
	if want := (Spec{NumTokens: 3}); got != want {
		t.Errorf("ngram K=3: %+v, want %+v", got, want)
	}
	for _, m := range []string{"ngram_gpu", "suffix"} {
		if g := ResolveSpec(SpecOptions{Method: m, NumTokens: 3}); g.EagleBlockDrop || g.Lookahead != 0 || g.PrefillLookahead != 0 {
			t.Errorf("%s resolves like an EAGLE-family method: %+v", m, g)
		}
	}
	e := ResolveSpec(SpecOptions{Method: "eagle3", NumTokens: 3})
	if !e.EagleBlockDrop || e.Lookahead != 3 || e.PrefillLookahead != 1 {
		t.Errorf("eagle3 K=3: %+v; the pair above would pass for a table with no EAGLE rule", e)
	}
}
