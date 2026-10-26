package vllm

import (
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

// The factory builds an MTP drafter once the engine block states the drafter checkpoint's
// num_nextn_predict_layers (mtp_nextn_predict_layers), which sets the prefill lookahead vLLM
// v0.32.0 reserves: num_speculative_tokens for multi-module MTP (more than one layer and K above
// one), else one (VllmConfig.num_prefill_lookahead_tokens, vllm/config/vllm.py:644-664;
// use_multi_module_mtp, vllm/config/speculative.py:1982-1988@5df63a72). Without it the
// lookahead is not knowable, and the block is refused naming the field.
func TestFactoryBuildsMTPWithItsStatedLookahead(t *testing.T) {
	for _, c := range []struct {
		name, extra   string
		wantLookahead int
	}{
		{"one nextn layer", "speculative: {method: mtp, num_spec_tokens: 2}\nmtp_nextn_predict_layers: 1\n", 1},
		{"three nextn layers", "speculative: {method: mtp, num_spec_tokens: 2}\nmtp_nextn_predict_layers: 3\n", 2},
		{"three nextn layers, K 1", "speculative: {method: mtp, num_spec_tokens: 1}\nmtp_nextn_predict_layers: 3\n", 1},
	} {
		st, err := resolve(mustConfig(t, c.extra))
		if err != nil {
			t.Fatalf("%s: refused: %v", c.name, err)
		}
		if got := st.sched.Spec.PrefillLookahead; got != c.wantLookahead {
			t.Errorf("%s: prefill lookahead %d, want %d", c.name, got, c.wantLookahead)
		}
	}
	_, err := build(t, "speculative: {method: mtp, num_spec_tokens: 2}\n")()
	if err == nil || !strings.Contains(err.Error(), "mtp_nextn_predict_layers") {
		t.Errorf("MTP without its layer count: err %v, want a refusal naming mtp_nextn_predict_layers", err)
	}
}

// With async scheduling, vLLM's default, an MTP or DSpark drafter runs through the batch queue
// (max_concurrent_batches 2) and the AsyncScheduler's placeholder drafts, which tier 3 diff-tests
// against vLLM's own EngineCore (core/engine/testdata/golden/t3-async-mtp*.json, t3-async-dspark.json).
func TestFactoryBuildsAsyncDrafters(t *testing.T) {
	for _, extra := range []string{
		"async_scheduling: true\nspeculative: {method: mtp, num_spec_tokens: 2}\nmtp_nextn_predict_layers: 1\n",
		"async_scheduling: true\nuse_v2_model_runner: true\nspeculative: {method: dspark, num_spec_tokens: 2}\n",
	} {
		e, err := build(t, extra)()
		if err != nil {
			t.Fatalf("%q: refused: %v", extra, err)
		}
		if got := e.InFlightLimit(); got != 2 {
			t.Errorf("%q: InFlightLimit %d, want 2 (async scheduling's batch queue)", extra, got)
		}
		// Every request finishes, every contract law holds, and decodes verify drafts.
		tr := enginetest.TestEngine(t, build(t, extra), arrivals(req("a", 40, 12), req("b", 33, 9), req("c", 20, 7)),
			enginetest.Options{Accepted: func(_ engine.Batch, s engine.Scheduled) int { return s.Drafts / 2 }})
		drafts := 0
		for _, st := range tr.Steps {
			for _, s := range st.Batch.Reqs {
				drafts += s.Drafts
			}
		}
		if drafts == 0 || len(tr.Finished) != 3 {
			t.Errorf("%q: %d drafts verified and %d of 3 requests finished", extra, drafts, len(tr.Finished))
		}
	}
}

func mustConfig(t *testing.T, extra string) Config {
	t.Helper()
	c, err := decodeConfig(block(t, extra))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// Under async scheduling EngineCore.post_step proposes no drafts (vllm/v1/engine/core.py:689-696
// @5df63a72): when a batch completes, a decoding request's next drafts stay the AsyncScheduler's
// -1 placeholders, set when the next batch was scheduled, which the worker fills itself.
func TestAsyncCompleteKeepsThePlaceholderDrafts(t *testing.T) {
	e, err := build(t, "async_scheduling: true\nspeculative: {method: mtp, num_spec_tokens: 2}\nmtp_nextn_predict_layers: 1\n")()
	if err != nil {
		t.Fatal(err)
	}
	r := req("a", 20, 8)
	if err := e.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	first, err := e.Schedule(0)
	if err != nil {
		t.Fatal(err)
	}
	second, err := e.Schedule(0)
	if err != nil || len(second.Reqs) != 1 || second.Reqs[0].Drafts != 2 {
		t.Fatalf("second batch %+v (%v), want a's decode verifying 2 placeholders", second, err)
	}
	if _, err := e.Complete(first.ID, engine.Outcome{}, 1); err != nil {
		t.Fatal(err)
	}
	if got := e.(*Engine).s.Request("a").SpecTokens; !slices.Equal(got, []int64{-1, -1}) {
		t.Errorf("after the prefill completed, a's next drafts are %v, want the placeholders [-1 -1]", got)
	}
}
