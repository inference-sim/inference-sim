package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// A step whose drafter proposes no drafts (kernel.Batch.SkipDrafts, vLLM v0.33.0's skip_drafts)
// is priced without the passes that propose. The EAGLE3 drafter here runs its head and later
// passes for every request (PassRequestsAll), so a step whose only request is a non-final
// prefill chunk pays for them unless the step skips its drafts. Skipped, it pays what a drafter
// that drafts for no request pays: the first pass over the scheduled tokens, which keeps the
// drafter's KV in step (target_dependent_ar/speculator.py:281-300@4ba787ec), and nothing more.
func TestAStepThatSkipsItsDraftsIsPricedWithoutTheProposingPasses(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	in := withNamedDrafter(t, d, "minimax-m3-mxfp8", "minimax-m3-eagle3", "b200", 3)
	all := mustKernel(t, passesOf(in, plan.FirstPassScheduled, plan.PassRequestsAll))
	sampling := mustKernel(t, passesOf(in, plan.FirstPassScheduled, plan.PassRequestsSampling))
	price := func(k *Kernel, skip bool, reqs ...kernel.ReqShape) float64 {
		return k.StepTime(kernel.Batch{DecodeThreshold: 4, SkipDrafts: skip, Reqs: reqs}).Overlap.Seconds()
	}
	chunk := kernel.ReqShape{Scheduled: 2048, PromptLen: 8192}
	decode := kernel.ReqShape{Scheduled: 4, Computed: 4092, PromptLen: 4096}

	drafting, skipped := price(all, false, chunk), price(all, true, chunk)
	if skipped >= drafting {
		t.Errorf("a non-final chunk: %.9g s skipping its drafts, %.9g s proposing them; skipping must cost less", skipped, drafting)
	}
	if want := price(sampling, false, chunk); skipped != want {
		t.Errorf("a non-final chunk skipping its drafts: %.9g s, want %.9g s, the first pass alone", skipped, want)
	}
	// The first pass is still charged: without it the skipped step is the target's alone.
	if bare := price(withoutFirstPass(all), true, chunk); skipped <= bare {
		t.Errorf("a non-final chunk skipping its drafts: %.9g s, and %.9g s without the drafter's first pass; "+
			"the drafter's KV work must still be charged", skipped, bare)
	}
	// Decodes that skip (each reaches max_tokens this step) drop their head and later passes too.
	if s, p := price(all, true, decode, decode), price(all, false, decode, decode); s >= p {
		t.Errorf("two last decodes: %.9g s skipping, %.9g s proposing; skipping must cost less", s, p)
	}
	// A deployment that does not speculate prices the flag as nothing.
	plain := *all
	plain.speculating = false
	if s, p := price(&plain, true, chunk), price(&plain, false, chunk); s != p {
		t.Errorf("no drafter: %.9g s skipping, %.9g s not; the flag must change nothing", s, p)
	}
}
