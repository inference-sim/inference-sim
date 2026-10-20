package kernelmodel

import (
	"reflect"
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

// Laws of the adapter's step price over any batch:
//
//   - it is at least one tick (INV-3: a zero step stalls the simulation clock), whatever the
//     batch, including an empty one;
//   - it is the kernel's own price for the batch the adapter hands it, in whole ticks: the
//     adapter adds nothing and loses nothing;
//   - scheduling one more request hands the kernel that request appended to the same batch,
//     with one more scheduled token at least, and the step it prices is never cheaper;
//   - the order requests appear in does not change it, because a forward pass prices the set
//     it runs, and BLIS's batch slice order is a scheduling artefact.
//
// The last is the kernel's law as much as the adapter's: blis-latency-kernel v0.1.0 broke it
// at the CUDA-graph capture ceiling (blis-latency-kernel#42), fixed in v0.2.0.
//
// These are laws of the composition, not of the field mapping: a mistranslated field can
// satisfy all of them, so the mapping is pinned separately (TestPrefillAndDecodeArePricedByDifferentLaws,
// TestScheduledTokensComeFromNumNewTokensNotThePrompt).
func TestStepPriceIsPositiveFaithfulGrowingAndOrderFree(t *testing.T) {
	for _, scenario := range propertyScenarios {
		m := open(t, scenario)
		t.Run(scenario, func(t *testing.T) {
			rapid.Check(t, func(rt *rapid.T) {
				batch := rapid.SliceOfN(rapid.Custom(request), 0, 64).Draw(rt, "batch")
				before := m.StepTime(batch)
				if before < 1 {
					rt.Fatalf("a %d-request batch priced at %d ticks; the floor is 1", len(batch), before)
				}
				kb := m.batchOf(batch)
				if want := max(1, m.k.StepTime(kb).NoOverlap.Microseconds()); before != want {
					rt.Fatalf("the adapter priced %d ticks; the kernel prices the batch at %d", before, want)
				}
				more := append(batch[:len(batch):len(batch)], request(rt))
				grown := m.batchOf(more)
				if len(grown.Reqs) != len(kb.Reqs)+1 || !reflect.DeepEqual(grown.Reqs[:len(kb.Reqs)], kb.Reqs) ||
					grown.Tokens() < kb.Tokens()+1 {
					rt.Fatalf("one more request did not append exactly one scheduled request to the kernel's batch")
				}
				if after := m.StepTime(more); after < before {
					rt.Fatalf("one more request made the step cheaper: %d ticks, down from %d", after, before)
				}
				shuffled := rapid.Permutation(batch).Draw(rt, "order")
				if again := m.StepTime(shuffled); again != before {
					rt.Fatalf("reordering the batch moved the step from %d to %d ticks", before, again)
				}
			})
		})
	}
}

// The monotone law at the point blis-latency-kernel#42 broke it, pinned rather than left to the
// generator: on GLM-5 (DSA sparse MLA) on H200, a mixed step of exactly 512 scheduled tokens --
// the capture ceiling -- and the same step plus one decode, just past it. v0.1.0 priced the
// second cheaper; the step past the ceiling must cost at least as much.
func TestOneMoreRequestPastTheCaptureCeilingIsNotCheaper(t *testing.T) {
	m := open(t, "glm-5-h200-fp8-sglang-tp8.yaml")
	decode := func() *sim.Request {
		return &sim.Request{InputTokens: make([]sim.TokenID, 1), ProgressIndex: 1, NumNewTokens: 1}
	}
	var batch []*sim.Request
	for i := 0; i < 9; i++ {
		batch = append(batch, decode(), &sim.Request{InputTokens: make([]sim.TokenID, 1), NumNewTokens: 1})
	}
	batch = append(batch, &sim.Request{InputTokens: make([]sim.TokenID, 494), NumNewTokens: 494})
	past := append(batch[:len(batch):len(batch)], decode())
	if at, over := m.batchOf(batch).Tokens(), m.batchOf(past).Tokens(); at != 512 || over != 513 {
		t.Fatalf("fixture lost its point: %d and %d scheduled tokens, not 512 and 513 either side of the ceiling", at, over)
	}
	if at, over := m.StepTime(batch), m.StepTime(past); over < at {
		t.Fatalf("one more decode past the 512-token capture ceiling made the step cheaper: %d ticks, down from %d", over, at)
	}
}
