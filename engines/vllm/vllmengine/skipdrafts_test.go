// Original to blis-vllm.

package vllmengine

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"

	vllm "github.com/inference-sim/blis-vllm/core/vllm"
)

// A batch that states vLLM v0.33.0's skip_drafts is priced without the drafter's proposing
// passes, alone and across a data-parallel group. d034 is deepseek-v4.1-flash on H200 with DSpark
// K=2 on the V2 runner. The batch is four decodes, each verifying its two drafts and sampling its
// last output token, so no step verifies the drafts proposed after it (Scheduler._drafts_unused,
// vllm/v1/core/sched/scheduler.py:502-544@4ba787ec). Skipping costs less, and still more than an
// empty step, because the target's forward pass and DSpark's context K/V still run
// (spec_decode/dflash/speculator.py:546-553@4ba787ec).
func TestASkippingBatchIsPricedWithoutTheProposingPasses(t *testing.T) {
	k := pricerOf(openPool(t, "d034-032.yaml"))
	var decodes engine.Batch
	for _, id := range []string{"a", "b", "c", "d"} {
		decodes.Reqs = append(decodes.Reqs, engine.Scheduled{ID: id, Tokens: 3, Computed: 4096, PromptLen: 2048, Drafts: 2})
	}
	skip := decodes
	skip.SkipDrafts = true
	drafting, skipped := k.StepTime(decodes), k.StepTime(skip)
	if skipped >= drafting {
		t.Errorf("four last decodes: %d skipping its drafts, %d proposing them; skipping must cost less", skipped, drafting)
	}
	if empty := k.StepTime(engine.Batch{}); skipped <= empty {
		t.Errorf("four last decodes skipping their drafts: %d, an empty step %d; the step still runs", skipped, empty)
	}
	dp, ok := k.(vllm.DPPricer)
	if !ok {
		t.Fatal("the kernel pricer prices no data-parallel group")
	}
	if s, p := dp.StepTimesAcrossDP([]engine.Batch{skip, skip}), dp.StepTimesAcrossDP([]engine.Batch{decodes, decodes}); s[0] >= p[0] || s[1] >= p[1] {
		t.Errorf("two ranks of last decodes: %v skipping, %v proposing; each rank must cost less", s, p)
	}
}
