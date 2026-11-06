package latencykernel

import (
	"math"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
)

// Two properties no other test pinned (blis-latency-kernel#66, findings 6): a mutant of either
// survived the suite. TestGuardedPropertiesKillTheirMutants (mutation_test.go) breaks each and
// requires the test named here to fail.

// A token-sharded mHC node runs on ceil(tokens / tp) rows under the model's own sequence
// parallelism (sp_shard pads the count to a multiple of tp), and on every token without it.
// V4.1's mHC form is priced at floor + rows x hidden x 8 B / rate, so a form that charged
// all tokens under sequence parallelism would over-charge the SM time tp-fold. The case is
// TP4 with sequence parallelism on (expert parallelism with data parallelism above 1) and
// not TP8 with it off, which is the only one the vLLM-side suite builds. A node that is not
// token-sharded runs on every token either way.
func TestAnMHCNodeRunsOnItsSequenceParallelShard(t *testing.T) {
	const tp, hidden, tokens = 4, 5120, 1001 // 1001 is not a multiple of tp: the shard pads up
	k := &Kernel{tp: tp, hiddenSize: hidden,
		mhcPre:  floorRate{floor: 9 * time.Microsecond, rate: 1.8e12},
		mhcPost: floorRate{floor: 3 * time.Microsecond, rate: 1.85e12}}
	s := &stepShape{tokensF: tokens, spRows: math.Ceil(float64(tokens) / tp)}
	if s.spRows != 251 {
		t.Fatalf("the shard is %v rows; the golden assumes ceil(1001/4) = 251", s.spRows)
	}
	sharded := price.PlannedLayer{MHC: []price.PlannedMHC{
		{Pre: true, Post: true, Bytes: 100, TokenSharded: true}}}
	whole := price.PlannedLayer{MHC: []price.PlannedMHC{
		{Pre: true, Post: true, Bytes: 100, Replicated: true}}}
	form := func(rows float64) float64 {
		w := rows * hidden * mhcBytesPerHidden
		return (9e-6 + w/1.8e12) + (3e-6 + w/1.85e12)
	}
	for _, c := range []struct {
		name string
		l    *price.PlannedLayer
		sp   bool
		rows float64
	}{
		{"token-sharded, sequence parallelism on", &sharded, true, 251},
		{"token-sharded, sequence parallelism off", &sharded, false, tokens},
		{"not token-sharded, sequence parallelism on", &whole, true, tokens},
	} {
		got, _ := k.mhcSeconds(c.l, s, c.sp)
		if want := form(c.rows); math.Abs(got-want) > 1e-12 {
			t.Errorf("%s: %.9f s of SM, want the form over %v rows, %.9f s", c.name, got, c.rows, want)
		}
	}
}

// A parallel-block drafter drafts every request in the batch, prefill rows included
// (dflash/speculator.py:363, num_query_tokens = num_reqs x num_query_per_req), so the width
// compared with its capture ceiling counts every request that has a token, not only the decode
// rows. Here K = 3 and the ceiling is 6 tokens, two requests: one decode row and one prefill
// chunk are two requests and graph; adding a second prefill chunk makes three, past the
// ceiling, and the drafter runs eagerly. The cases at the boundary on both sides: a batch of
// decode rows alone at the same request counts prices the same.
func TestTheDrafterCeilingCountsPrefillRows(t *testing.T) {
	const kSpec = 3
	k := &Kernel{
		engine:      enginesettings.Settings{Speculative: &enginesettings.Speculative{Method: "dspark", NumSpecTokens: kSpec}},
		plan:        &price.Plan{Draft: &price.DraftPlan{ContextKernels: 2, Drafting: model.DraftingParallelBlock}},
		graphDecode: graphModeFull, draftCaptureTokens: 2 * kSpec,
		launchEager: time.Microsecond, replayPerStep: 5 * time.Microsecond,
	}
	decode := kernel.ReqShape{Scheduled: kSpec + 1, Computed: 100, PromptLen: 100}
	prefill := kernel.ReqShape{Scheduled: 64, Computed: 0, PromptLen: 4096}
	const kernels = 40
	eager := time.Duration(kernels) * time.Microsecond
	graphed := func(b kernel.Batch) time.Duration {
		// A replay, plus the context kernels eagerly while the step schedules more than one
		// verify per request.
		return k.draftHost(b, kernels)
	}
	batch := func(rs ...kernel.ReqShape) kernel.Batch {
		return kernel.Batch{DecodeThreshold: kSpec + 1, Reqs: rs}
	}
	two := batch(decode, prefill)
	if got := graphed(two); got == eager {
		t.Errorf("decode + prefill: %v, eager; two requests draft 2K = %d tokens, within the "+
			"ceiling %d, so they replay", got, 2*kSpec, k.draftCaptureTokens)
	}
	three := batch(decode, prefill, prefill)
	if got := graphed(three); got != eager {
		t.Errorf("decode + 2 prefill rows: %v, want eager %v; three requests draft 3K = %d "+
			"tokens, past the ceiling %d, and a drafter that counted decode rows alone would "+
			"graph", got, eager, 3*kSpec, k.draftCaptureTokens)
	}
	if got := graphed(batch(decode, decode, decode)); got != eager {
		t.Errorf("three decode rows: %v, want eager %v", got, eager)
	}
}
