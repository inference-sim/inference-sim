package latencykernel

import (
	"github.com/inference-sim/blis-substrate/contracts/conformance/kerneltest"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

func res(name string, m vocab.Method, value float64, ci *coefficient.Interval, fit ...coefficient.Bound) resolve.Resolved {
	return resolve.Resolved{Set: "s", Entry: coefficient.Entry{Name: name, Method: m, Value: value, CI95: ci, FitRange: fit}}
}

// VAL-3: a price records the least trusted effective rank among the coefficients it used,
// names the method and entry that set it, and says why when the rank is not the method's own.
func TestTrustIsTheLeastTrustedEffectiveRank(t *testing.T) {
	tokens := func(lo, hi float64) coefficient.Bound {
		return coefficient.Bound{Dim: string(vocab.FitTokens), Low: lo, High: hi}
	}
	wide := &coefficient.Interval{Low: 0, High: 100}
	in := func(tok float64) map[string]float64 {
		return map[string]float64{"tokens": tok, "requests": 1, "context_tokens": 10}
	}
	cases := []struct {
		name     string
		selected []resolve.Resolved
		in       map[string]float64
		rank     int
		method   vocab.Method
		flag     string
		entry    string
	}{
		{"all measured, inside every range",
			[]resolve.Resolved{res("a", vocab.MethodMeasured, 10, nil, tokens(1, 100)), res("b", vocab.MethodMeasured, 10, nil)},
			in(50), 1, vocab.MethodMeasured, "", "a"},
		{"one assumed entry sets the rank",
			[]resolve.Resolved{res("a", vocab.MethodMeasured, 10, nil), res("z", vocab.MethodAssumed, 10, nil)},
			in(50), 7, vocab.MethodAssumed, "", "z"},
		{"a measured entry read outside its range counts as transferred",
			[]resolve.Resolved{res("a", vocab.MethodMeasured, 10, nil, tokens(1, 100))},
			in(500), 3, vocab.MethodMeasured, "extrapolated", "a"},
		{"a measured entry with an unbounded-width interval counts as assumed",
			[]resolve.Resolved{res("a", vocab.MethodMeasured, 10, wide)},
			in(50), 7, vocab.MethodMeasured, "unidentified", "a"},
		{"literature is below transferred, so extrapolation does not lower it",
			[]resolve.Resolved{res("a", vocab.MethodLiterature, 10, nil, tokens(1, 100))},
			in(500), 4, vocab.MethodLiterature, "", "a"},
		{"not_charged has no rank and is ignored",
			[]resolve.Resolved{res("a", vocab.MethodNotCharged, 0, nil), res("b", vocab.MethodCalibrated, 10, nil)},
			in(50), 2, vocab.MethodCalibrated, "", "b"},
		{"a tie goes to the name that sorts first, whatever the lookup order",
			[]resolve.Resolved{res("y", vocab.MethodAssumed, 10, nil), res("x", vocab.MethodAssumed, 10, nil)},
			in(50), 7, vocab.MethodAssumed, "", "x"},
	}
	for _, c := range cases {
		for _, order := range [][]resolve.Resolved{c.selected, reversed(c.selected)} {
			got := trustOf(buildTrust(order), c.in)
			if got.Rank != c.rank || got.Method != string(c.method) || got.Flag != c.flag || got.Entry != c.entry {
				t.Errorf("%s: got %+v, want rank %d method %s flag %q entry %s", c.name, got, c.rank, c.method, c.flag, c.entry)
			}
		}
	}
	if got := trustOf(nil, in(1)); got.Rank != 0 {
		t.Errorf("a price that used no ranked coefficient reports the zero value, got %+v", got)
	}
}

func reversed(in []resolve.Resolved) []resolve.Resolved {
	out := make([]resolve.Resolved, len(in))
	for i, r := range in {
		out[len(in)-1-i] = r
	}
	return out
}

// VAL-3 end to end: a step priced by a kernel built from the committed registry reports the
// least trusted effective rank of what it used. Muse Spark on a GB200 pool reads measured
// GEMM, attention and collective entries and assumed host and memory ones, so the least
// trusted is assumed; the entry named is the first by name among the assumed ones read.
func TestAStepOnTheCommittedRegistryReportsItsTrust(t *testing.T) {
	est, k := priceOnGB200(t, 8, fewTokens, false)
	if est.Trust.Rank != 7 || est.Trust.Method != string(vocab.MethodAssumed) || est.Trust.Entry == "" {
		t.Fatalf("Trust = %+v, want the assumed rank 7 with the entry named", est.Trust)
	}
	// The named entry is one the kernel actually selected, and it is assumed.
	found := false
	for _, r := range k.selected {
		if r.Entry.Name == est.Trust.Entry && r.Entry.Method == vocab.MethodAssumed {
			found = true
		}
	}
	if !found {
		t.Errorf("Trust names %q, which is not an assumed entry the kernel selected", est.Trust.Entry)
	}
	// An empty step prices only the host's cost and still reports a trust.
	if empty := k.StepTime(kernel.Batch{}); empty.Trust.Rank == 0 {
		t.Errorf("an empty step reports no trust: %+v", empty.Trust)
	}
	// The same step priced twice reports the same trust.
	if again := k.StepTime(kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 4095, PromptLen: 4096}}}); again.Trust.Rank != 7 {
		t.Errorf("a repeated step changed its trust: %+v", again.Trust)
	}
}

// The real kernel meets the conformance suite's trust and stage laws (design VAL-3): over
// decode-only, prefill-only and mixed batches, and the empty batch.
func TestTheKernelMeetsTheTrustAndStageLaws(t *testing.T) {
	_, k := priceOnGB200(t, 8, fewTokens, false)
	decode := kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096}
	prefill := kernel.ReqShape{Scheduled: 512, Computed: 0, PromptLen: 512}
	batches := []kernel.Batch{
		{},
		{DecodeThreshold: 1, Reqs: []kernel.ReqShape{decode}},
		{DecodeThreshold: 1, Reqs: []kernel.ReqShape{prefill}},
		{DecodeThreshold: 1, Reqs: []kernel.ReqShape{decode, prefill, decode}},
	}
	kerneltest.TestTrust(t, k, batches)
	kerneltest.TestStages(t, k, batches)
}
