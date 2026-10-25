// Moved from inference-sim/sim/kernelmodel/speculative_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"pgregory.net/rapid"

	sim "github.com/inference-sim/blis-vllm/compat/engine"
	_ "github.com/inference-sim/blis-vllm/compat/kv"
)

// openSpeculative opens a committed scenario with its pool given a draft configuration of k
// tokens, through the same New the kernel's own OpenPool ends in.
func openSpeculative(t *testing.T, scenario string, k int) *Model {
	t.Helper()
	in, err := latencykernel.OpenInputs(scenario, repos(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if k > 0 {
		in.Deployment.Pools[0].Engine.Speculative = &deployment.Speculative{Method: "mtp", NumSpecTokens: k}
	}
	kn, err := latencykernel.New(in)
	if err != nil {
		t.Fatalf("New with %d draft tokens: %v", k, err)
	}
	m, err := newModel(kn, identityOf(in))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// Under speculation a decode costs its verify width -- 1 + the draft length -- whatever it later
// advanced by: the engine runs every draft position (the blis-schemas contract states
// Scheduled that way), while BLIS's NumNewTokens is the advance (1 + accepted).
//
// Relations, none of which reads the price off the code under test:
//
//   - the price of a decode does not depend on how many drafts were accepted;
//   - a speculative decode never costs less than a plain one, and a longer draft never less
//     than a shorter one;
//   - a request that has just finished its prompt (ProgressIndex == InputLen) is a decode;
//   - a prefill chunk costs the same with and without a draft configuration.
func TestASpeculativeDecodeIsPricedAtItsVerifyWidth(t *testing.T) {
	const scenario = "glm-5-h200-fp8-sglang-tp8.yaml"
	decodeAt := func(progress int64, advance int) []*sim.Request {
		return []*sim.Request{{InputTokens: make([]sim.TokenID, 2048), ProgressIndex: progress, NumNewTokens: advance}}
	}
	prefill := []*sim.Request{{InputTokens: make([]sim.TokenID, 2048), NumNewTokens: 512}}
	plain := openSpeculative(t, scenario, 0)
	prev := plain.StepTime(decodeAt(2100, 1))
	for _, k := range []int{1, 2, 3} {
		m := openSpeculative(t, scenario, k)
		price := m.StepTime(decodeAt(2100, 1))
		for advance := 2; advance <= k+1; advance++ {
			if got := m.StepTime(decodeAt(2100, advance)); got != price {
				t.Errorf("k=%d: a decode advancing %d priced %d, advancing 1 priced %d; acceptance must not move the price",
					k, advance, got, price)
			}
		}
		if price < prev {
			t.Errorf("k=%d: a decode priced %d, below the shorter draft's %d", k, price, prev)
		}
		prev = price
		// The first decode step after the prompt is a decode too: it costs exactly what a decode
		// over the same computed context costs one token later in its life -- a request whose
		// prompt ended a token earlier -- and so strictly more than the plain engine's decode
		// whenever the draft widens the step at all.
		atBoundary := m.StepTime(decodeAt(2048, 1))
		laterDecode := m.StepTime([]*sim.Request{{InputTokens: make([]sim.TokenID, 2047), ProgressIndex: 2048, NumNewTokens: 1}})
		if atBoundary != laterDecode {
			t.Errorf("k=%d: the first decode after the prompt priced %d, a decode over the same context %d; "+
				"it was not priced at its verify width", k, atBoundary, laterDecode)
		}
		if plainBoundary := plain.StepTime(decodeAt(2048, 1)); laterDecode > plain.StepTime(
			[]*sim.Request{{InputTokens: make([]sim.TokenID, 2047), ProgressIndex: 2048, NumNewTokens: 1}}) &&
			atBoundary <= plainBoundary {
			t.Errorf("k=%d: the first decode after the prompt priced %d, no more than a plain decode's %d", k,
				atBoundary, plainBoundary)
		}
		if got, want := m.StepTime(prefill), plain.StepTime(prefill); got != want {
			t.Errorf("k=%d: a prefill chunk priced %d with a draft configuration, %d without", k, got, want)
		}
	}
	if last := openSpeculative(t, scenario, 3).StepTime(decodeAt(2100, 1)); last <= plain.StepTime(decodeAt(2100, 1)) {
		t.Errorf("a 3-token draft's decode priced %d, no more than a plain decode's %d; the verify width did not reach the kernel",
			last, plain.StepTime(decodeAt(2100, 1)))
	}
}

// dspark is the committed DeepSeek-V4.1-Flash scenario that drafts five tokens with DSpark.
const dspark = "deepseek-v4.1-flash-h100-tp8-dspark5.yaml"

// openDrafting opens a scenario from testdata/scenarios with its pool's speculative block
// replaced by sp (nil: no speculation), through the path OpenPool takes.
func openDrafting(t *testing.T, scenario string, sp *deployment.Speculative) (*Model, error) {
	t.Helper()
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	in, err := latencykernel.OpenInputs(scenario, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	in.Deployment.Pools[0].Engine.Speculative = sp
	return fromInputs(in)
}

// Which draft methods exist is the engine release's fact, read from the scenario's rules
// pack: DSpark opens at vLLM 0.31.0, a name the release does not list is refused naming the
// release even on a graph whose speculator would not otherwise stop it, and a listed method
// the model cannot run (MTP on DeepSeek-V4.1) is still refused, by the kernel.
func TestTheReleaseRulesPackDecidesTheSpeculativeMethod(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	if _, err := OpenPool(dspark, r, 0); err != nil {
		t.Errorf("DSpark at vLLM 0.31.0: %v", err)
	}
	for _, tt := range []struct {
		name, scenario, method, want string
	}{
		{"a method no release lists", "llama-3.1-70b-instruct-h200-tp4-4node.yaml", "bogus",
			`"bogus" is not accepted by engine 0.29.0`},
		{"a BLIS shorthand, not a vLLM method", "llama-3.1-70b-instruct-h200-tp4-4node.yaml", "draft",
			`"draft" is not accepted by engine 0.29.0`},
		{"MTP on DeepSeek-V4.1", dspark, "mtp", "mtp"},
	} {
		_, err := openDrafting(t, tt.scenario, &deployment.Speculative{Method: tt.method, NumSpecTokens: 3})
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: got %v, want a refusal containing %q", tt.name, err, tt.want)
		}
	}
	if _, err := openDrafting(t, "llama-3.1-70b-instruct-h200-tp4-4node.yaml",
		&deployment.Speculative{Method: "ngram", NumSpecTokens: 3}); err != nil {
		t.Errorf("ngram, which 0.29.0 lists, on a dense model: %v", err)
	}
}

// INV-K1: BLIS adds no draft cost of its own. On a speculating pool the step price is the
// kernel's StepTime of the verify-width batch -- every decode scheduling 1 + K, every
// prefill chunk its own tokens, under a decode threshold covering the verify width -- built
// here from the requests rather than by the adapter, floored at one tick. And the draft cost
// is already in that price: the same batch costs strictly less on the same deployment
// without a drafter whenever it holds a decode, so a BLIS-side draft term would count the
// drafts twice.
func TestNoDraftCostIsAddedOnTheBLISSide(t *testing.T) {
	const k = 5
	spec, err := openDrafting(t, dspark, &deployment.Speculative{Method: "dspark", NumSpecTokens: k})
	if err != nil {
		t.Fatal(err)
	}
	plain, err := openDrafting(t, dspark, nil)
	if err != nil {
		t.Fatal(err)
	}
	rapid.Check(t, func(rt *rapid.T) {
		batch := rapid.SliceOfN(rapid.Custom(request), 1, 32).Draw(rt, "batch")
		for _, r := range batch {
			if r.ProgressIndex >= r.InputLen() {
				r.NumNewTokens = rapid.IntRange(1, 1+k).Draw(rt, "advance")
			}
		}
		verify := kernel.Batch{DecodeThreshold: max(DecodeThreshold, 1+k), SMBudget: spec.smBudget}
		decodes := 0
		for _, r := range batch {
			s := kernel.ReqShape{Scheduled: r.NumNewTokens, Computed: int(r.ProgressIndex), PromptLen: int(r.InputLen())}
			if r.ProgressIndex >= r.InputLen() {
				s.Scheduled = 1 + k
				decodes++
			}
			verify.Reqs = append(verify.Reqs, s)
		}
		want := max(1, spec.Kernel().StepTime(verify).NoOverlap.Microseconds())
		if got := spec.StepTime(batch); got != want {
			rt.Fatalf("BLIS priced %d ticks; the kernel prices the verify-width batch at %d", got, want)
		}
		if without := plain.Kernel().StepTime(verify).NoOverlap; decodes > 0 &&
			without >= spec.Kernel().StepTime(verify).NoOverlap {
			rt.Fatalf("with %d decodes the batch costs %v without a drafter and %v with one; the "+
				"kernel's price does not carry the draft passes", decodes, without,
				spec.Kernel().StepTime(verify).NoOverlap)
		}
	})
}

// vLLM treats a verify of up to 1 + K tokens as a decode, so the batch the kernel is handed
// classifies one as a decode however long the draft: at K = 8 a threshold of 8 would price a
// 9-token verify as a prefill chunk (#1932).
func TestTheDecodeThresholdCoversTheVerifyWidth(t *testing.T) {
	for _, k := range []int{0, 5, 8, 12} {
		var sp *deployment.Speculative
		if k > 0 {
			sp = &deployment.Speculative{Method: "dspark", NumSpecTokens: k}
		}
		m, err := openDrafting(t, dspark, sp)
		if err != nil {
			t.Fatal(err)
		}
		decode := []*sim.Request{{InputTokens: make([]sim.TokenID, 100), ProgressIndex: 150, NumNewTokens: 1}}
		b := m.batchOf(decode)
		if b.Reqs[0].Scheduled > b.DecodeThreshold || b.DecodeThreshold < DecodeThreshold {
			t.Errorf("k=%d: a decode scheduling %d against a threshold of %d is not classified as a decode",
				k, b.Reqs[0].Scheduled, b.DecodeThreshold)
		}
	}
}
