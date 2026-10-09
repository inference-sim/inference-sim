package kernelmodel

import (
	"testing"

	"pgregory.net/rapid"

	"github.com/inference-sim/inference-sim/sim"
)

// Laws of the adapter's step price over generated batches, on deployments spanning the
// attention families the kernel distinguishes: dense GQA, sliding-window alternation, DSA
// sparse MLA, and a routed expert model with data parallelism.
var propertyScenarios = []string{
	"llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml",
	"gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
	"glm-5-h200-fp8-sglang-tp8.yaml",
	"minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml",
}

// request draws one scheduled request in either regime: a prefill chunk part-way through its
// prompt, or a decode past the end of it.
func request(t *rapid.T) *sim.Request {
	prompt := rapid.IntRange(1, 8192).Draw(t, "prompt")
	r := &sim.Request{InputTokens: make([]sim.TokenID, prompt)}
	if rapid.Bool().Draw(t, "decode") {
		r.ProgressIndex = int64(prompt + rapid.IntRange(0, 4096).Draw(t, "generated"))
		r.NumNewTokens = 1
		return r
	}
	r.ProgressIndex = int64(rapid.IntRange(0, prompt-1).Draw(t, "computed"))
	r.NumNewTokens = rapid.IntRange(1, prompt-int(r.ProgressIndex)).Draw(t, "chunk")
	return r
}

// Three laws of a step price over any batch:
//
//   - it is at least one tick (INV-3: a zero step stalls the simulation clock), whatever the
//     batch, including an empty one;
//   - scheduling one more request never makes the step cheaper;
//   - the order requests appear in does not change it, because a forward pass prices the set
//     it runs, and BLIS's batch slice order is a scheduling artefact.
//
// These are laws of the composition, not of the field mapping: a mistranslated field can
// satisfy all three, so the mapping is pinned separately (TestPrefillAndDecodeArePricedByDifferentLaws,
// TestScheduledTokensComeFromNumNewTokensNotThePrompt).
func TestStepPriceIsPositiveMonotoneAndOrderFree(t *testing.T) {
	for _, scenario := range propertyScenarios {
		m := open(t, scenario)
		t.Run(scenario, func(t *testing.T) {
			rapid.Check(t, func(rt *rapid.T) {
				batch := rapid.SliceOfN(rapid.Custom(request), 0, 64).Draw(rt, "batch")
				before := m.StepTime(batch)
				if before < 1 {
					rt.Fatalf("a %d-request batch priced at %d ticks; the floor is 1", len(batch), before)
				}
				extra := request(rt)
				if after := m.StepTime(append(batch[:len(batch):len(batch)], extra)); after < before {
					rt.Fatalf("adding a request cut the step from %d to %d ticks", before, after)
				}
				shuffled := rapid.Permutation(batch).Draw(rt, "order")
				if again := m.StepTime(shuffled); again != before {
					rt.Fatalf("reordering the batch moved the step from %d to %d ticks", before, again)
				}
			})
		})
	}
}
