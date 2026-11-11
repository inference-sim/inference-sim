package latencykernel

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/load"
)

// The first pass of an EAGLE3 drafter runs its layers and its token lookup over every token the
// target scheduled that step, on the target's batch shape (vllm/v1/worker/gpu/spec_decode/
// target_dependent_ar/speculator.py, _prefill :432-478 and its call :232-295 at v0.32.0); only
// the later K - 1 passes run one row per request. priceDraft used to charge every pass at one
// row per request, so a step with a drafter that schedules more than one token a request was
// priced without that work and nothing in the price said so.

// passesOf is the same inputs with the engine's statement of how the draft passes run, as the
// pool's plan carries it (plan.DraftPolicy).
func passesOf(in Inputs, rows plan.FirstPassRows, reqs plan.PassRequests) Inputs {
	in.DraftFirstPassRows, in.DraftPassRequests = rows, reqs
	return in
}

// eagle3Kernel is the kernel for a MiniMax-M3 target speculating with a named EAGLE3 head at
// the given draft length, the engine model stating that its first pass processes every
// scheduled token and that the head and later passes run for every request (vLLM v0.32.0).
func eagle3Kernel(t *testing.T, d *load.Data, target, drafter, chip string, n int) *Kernel {
	t.Helper()
	k := mustKernel(t, withNamedDrafter(t, d, target, drafter, chip, "eagle3", n))
	if !k.draftFirstPass || !k.draftAllRequests {
		t.Fatalf("%s with %s: first pass over the scheduled tokens %v, all requests %v", target, drafter, k.draftFirstPass, k.draftAllRequests)
	}
	return k
}

// withoutFirstPass is the same kernel priced as before: every autoregressive pass at one row
// per request.
func withoutFirstPass(k *Kernel) *Kernel {
	c := *k
	c.draftFirstPass = false
	return &c
}

// stepOf prices a step whose requests are verify rows of up to four tokens (a draft length of
// three), which the engine's decode threshold counts as decodes.
func stepOf(k *Kernel, reqs ...kernel.ReqShape) float64 {
	return k.StepTime(kernel.Batch{DecodeThreshold: 4, Reqs: reqs}).Overlap.Seconds()
}

// A step is priced higher by the draft layers' cost over the scheduled tokens than without it,
// for a decode step that verifies K draft tokens a request and for a prefill chunk; the
// increase is the draft layers and lookup priced on the step's own shape less the one-row
// share the later passes would otherwise have counted, and it grows with the tokens scheduled.
func TestAnEagle3FirstPassIsChargedOverTheScheduledTokens(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"h200", "b200"} {
		for _, drafter := range []string{"minimax-m3-eagle3", "minimax-m3-eagle3-gqa"} {
			t.Run(chip+"/"+drafter, func(t *testing.T) {
				k := eagle3Kernel(t, d, "minimax-m3-mxfp8", drafter, chip, 3)
				old := withoutFirstPass(k)
				verify := func(reqs int) []kernel.ReqShape {
					out := make([]kernel.ReqShape, reqs)
					for i := range out {
						out[i] = kernel.ReqShape{Scheduled: 4, Computed: 4092, PromptLen: 4096}
					}
					return out
				}
				chunk := func(n int) []kernel.ReqShape {
					return []kernel.ReqShape{{Scheduled: n, PromptLen: n}}
				}
				var last float64
				for _, c := range []struct {
					name string
					reqs []kernel.ReqShape
				}{
					{"decode 1 request, 4 tokens", verify(1)},
					{"decode 16 requests, 4 tokens", verify(16)},
					{"decode 64 requests, 4 tokens", verify(64)},
				} {
					with, without := stepOf(k, c.reqs...), stepOf(old, c.reqs...)
					if with <= without {
						t.Errorf("%s: %.6g s with the first pass, %.6g s without; it must cost more", c.name, with, without)
					}
					if delta := with - without; delta < last {
						t.Errorf("%s: the first pass's cost %.3g s fell with more requests (%.3g s)", c.name, delta, last)
					} else {
						last = delta
					}
				}
				last = 0
				for _, n := range []int{512, 2048, 8192} {
					with, without := stepOf(k, chunk(n)...), stepOf(old, chunk(n)...)
					delta := with - without
					if delta <= 0 || delta < last {
						t.Errorf("prefill of %d tokens: the first pass costs %.3g s after %.3g s", n, delta, last)
					}
					last = delta
				}
			})
		}
	}
}

// With one token a request the first pass has as many rows as the single-row pass it replaces,
// so the two ways of pricing agree: the charge is for the extra rows only, never a pass on top.
func TestTheFirstPassAddsNothingWhereTheRowsAreOnePerRequest(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	k := eagle3Kernel(t, d, "minimax-m3-mxfp8", "minimax-m3-eagle3", "b200", 3)
	old := withoutFirstPass(k)
	one := []kernel.ReqShape{{Scheduled: 1, Computed: 4095, PromptLen: 4096}}
	with, without := stepOf(k, one...), stepOf(old, one...)
	// They differ by the one context token the single-row pass reads beyond the step's own
	// shape (priceDraft prices a draft pass at the context plus its new token): nanoseconds in
	// nine milliseconds.
	if math.Abs(with-without) > 1e-5*without {
		t.Errorf("one row a request: %.9g s with the first pass, %.9g s without; they must agree", with, without)
	}
}

// The increase is the draft layers and the token lookup priced over the step's shape, less the
// same priced over one row a request: computed here from the kernel's own pricing of each shape.
func TestTheFirstPassIsTheDraftLayersPricedOverTheScheduledRows(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	k := eagle3Kernel(t, d, "minimax-m3-mxfp8", "minimax-m3-eagle3", "b200", 3)
	old := withoutFirstPass(k)
	reqs := make([]kernel.ReqShape, 8)
	for i := range reqs {
		reqs[i] = kernel.ReqShape{Scheduled: 4, Computed: 4092, PromptLen: 4096}
	}
	b := kernel.Batch{DecodeThreshold: 4, Reqs: reqs}
	body := func(s stepShape) float64 {
		var per [numResources]float64
		pl := k.plan.Draft
		t := k.priceLayers(pl.Layers, nil, k.draftLayerKV, &s, &per)
		return t + k.priceLayers([]price.PlannedLayer{pl.Embedding}, nil, []layerKV{{}}, &s, &per)
	}
	scheduled := k.shapeOf(b)
	rows := make([]kernel.ReqShape, len(reqs))
	for i := range rows {
		rows[i] = kernel.ReqShape{Scheduled: 1, Computed: 4096, PromptLen: 4096}
	}
	oneRow := k.shapeOf(kernel.Batch{Reqs: rows, DecodeThreshold: 1})
	want := body(scheduled) - body(oneRow)
	got := stepOf(k, reqs...) - stepOf(old, reqs...)
	if want <= 0 {
		t.Fatalf("the draft layers over %d rows cost no more than over %d", len(reqs)*4, len(reqs))
	}
	// The step adds the host and launch terms, which the first pass does not change, so the
	// difference of two steps is the device work alone.
	if math.Abs(got-want) > 1e-6*want+1e-9 {
		t.Errorf("the first pass adds %.9g s; the draft layers over the scheduled rows less one row a request is %.9g s", got, want)
	}
}

// The first pass is the engine model's statement, not the cost model's knowledge of a method:
// the target's own MTP, which vLLM runs through the same class, is charged the same way when the
// engine says so, and a multi-token step is priced higher than at one row a request. The same
// words under another method name price identically.
func TestAnyAutoregressiveDrafterIsChargedByTheStatementNotTheMethod(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	in := quantCase(t, d, "minimax-m3-mxfp8", "b200")
	mtp := func(rows plan.FirstPassRows, reqs plan.PassRequests) *Kernel {
		return mustKernel(t, passesOf(withEngine(in, func(e *enginesettings.Settings) {
			e.Speculative = &enginesettings.Speculative{Method: "mtp", NumSpecTokens: 3}
		}), rows, reqs))
	}
	verify := make([]kernel.ReqShape, 16)
	for i := range verify {
		verify[i] = kernel.ReqShape{Scheduled: 4, Computed: 4092, PromptLen: 4096}
	}
	chunk := []kernel.ReqShape{{Scheduled: 8192, PromptLen: 8192}}
	scheduled := mtp(plan.FirstPassScheduled, plan.PassRequestsAll)
	oneRow := mtp(plan.FirstPassPerRequest, plan.PassRequestsAll)
	for name, reqs := range map[string][]kernel.ReqShape{"verify rows": verify, "a prefill chunk": chunk} {
		if with, without := stepOf(scheduled, reqs...), stepOf(oneRow, reqs...); with <= without {
			t.Errorf("MTP, %s: %.9g s with the first pass over the scheduled tokens, %.9g s at one row a request; it must cost more", name, with, without)
		}
	}
	// Stated per_request, nothing is charged beyond the single-row passes.
	if k := oneRow; k.draftFirstPass {
		t.Error("per_request charged a first pass over the scheduled tokens")
	}
	// The kernel reads no method: a different name for the drafter, the same statement, the same price.
	other := mustKernel(t, passesOf(withEngine(in, func(e *enginesettings.Settings) {
		e.Speculative = &enginesettings.Speculative{Method: "mtp", NumSpecTokens: 3}
	}), plan.FirstPassScheduled, plan.PassRequestsAll))
	if pricesOf(other) != pricesOf(scheduled) {
		t.Error("two kernels with the same statement price differently")
	}
}

// vLLM's propose runs the head and the later passes for every request, a request still in the
// middle of its prompt included (_prefill samples for all num_reqs, _multi_step_decode runs over
// all of them). Where the engine model says so they are charged for those requests: a batch of
// unfinished chunks is no longer free of them, and in a mixed batch the chunks add to the decodes'.
func TestTheHeadAndLaterPassesAreChargedForRequestsStillInTheirPrompt(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	in := withNamedDrafter(t, d, "minimax-m3-mxfp8", "minimax-m3-eagle3", "b200", "eagle3", 3)
	all := mustKernel(t, passesOf(in, plan.FirstPassScheduled, plan.PassRequestsAll))
	sampling := mustKernel(t, passesOf(in, plan.FirstPassScheduled, plan.PassRequestsSampling))
	unfinished := func(n int) kernel.ReqShape { return kernel.ReqShape{Scheduled: n, Computed: 0, PromptLen: 4 * n} }
	decode := kernel.ReqShape{Scheduled: 4, Computed: 4092, PromptLen: 4096}
	finishing := kernel.ReqShape{Scheduled: 2048, Computed: 2048, PromptLen: 4096}
	for name, c := range map[string]struct {
		reqs   []kernel.ReqShape
		differ bool
	}{
		"only unfinished chunks":           {[]kernel.ReqShape{unfinished(2048), unfinished(1024)}, true},
		"unfinished chunks with decodes":   {[]kernel.ReqShape{unfinished(2048), decode, decode}, true},
		"a chunk that finishes its prompt": {[]kernel.ReqShape{finishing}, false},
		"decodes only":                     {[]kernel.ReqShape{decode, decode, decode}, false},
	} {
		t.Run(name, func(t *testing.T) {
			a, s := stepOf(all, c.reqs...), stepOf(sampling, c.reqs...)
			switch {
			case c.differ && a <= s:
				t.Errorf("%.9g s for all requests, %.9g s for the sampling ones; the unfinished requests' head and later passes are missing", a, s)
			case !c.differ && a != s:
				t.Errorf("%.9g s for all requests, %.9g s for the sampling ones; every request here samples", a, s)
			}
		})
	}
	// An all-unfinished batch under "sampling" charges the first pass only: no head, no later pass.
	only := []kernel.ReqShape{unfinished(2048), unfinished(1024)}
	if stepOf(all, only...) <= stepOf(sampling, only...)+1e-9 {
		t.Error("an all-unfinished batch pays nothing for the head and the later passes")
	}
}
