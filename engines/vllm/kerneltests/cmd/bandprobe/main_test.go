// Moved from blis-latency-kernel/cmd/bandprobe/main_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package main

import (
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/blis-vllm/kerneltests/internal/harness"
)

// TestDecodeShapeMatchesTheHarness pins this command's batch construction against
// harness.DecodeBatch, which cmd/worked-table and every scoring path already use.
//
// It exists because two construction bugs here produced a step time 1.08x off the
// harness's for the same stated shape, and neither was visible in the output: Computed
// was set to the context rather than context-1, so a decode read one token too many,
// and DecodeThreshold was left at zero, which selects the prefill attention form for a
// single-token request. A probe that disagrees with the harness is measuring a
// different model, and a blend fitted on it would be fitted on the difference.
func TestDecodeShapeMatchesTheHarness(t *testing.T) {
	for _, c := range []struct{ batch, ctx int }{
		{1, 1024}, {32, 1024}, {256, 1024}, {4, 8192},
	} {
		want := harness.DecodeBatch(c.batch, c.ctx)
		got := decodeBatch(c.batch, c.ctx)
		if got.DecodeThreshold != want.DecodeThreshold {
			t.Fatalf("batch %d ctx %d: DecodeThreshold %d, harness uses %d",
				c.batch, c.ctx, got.DecodeThreshold, want.DecodeThreshold)
		}
		if len(got.Reqs) != len(want.Reqs) {
			t.Fatalf("batch %d: %d requests, want %d",
				c.batch, len(got.Reqs), len(want.Reqs))
		}
		for i := range want.Reqs {
			if got.Reqs[i] != want.Reqs[i] {
				t.Fatalf("batch %d ctx %d req %d: %+v, harness builds %+v",
					c.batch, c.ctx, i, got.Reqs[i], want.Reqs[i])
			}
		}
	}
}

// TestPrefillShapeSchedulesTheChunk pins the mixed shape: a chunked prefill schedules
// its chunk, carries the already-computed prefix, and declares the whole prompt.
func TestPrefillShapeSchedulesTheChunk(t *testing.T) {
	b := prefillBatch(2, 512, 1024)
	if len(b.Reqs) != 2 {
		t.Fatalf("got %d requests, want 2", len(b.Reqs))
	}
	// Pinned here as well as on the decode path: DecodeThreshold selects the attention
	// form per request, so a mixed step that omits it prices its decode-width requests
	// with the prefill kernel.
	if b.DecodeThreshold != harness.DecodeThreshold {
		t.Fatalf("DecodeThreshold %d, harness uses %d",
			b.DecodeThreshold, harness.DecodeThreshold)
	}
	want := kernel.ReqShape{Scheduled: 512, Computed: 1024, PromptLen: 1536}
	for i, r := range b.Reqs {
		if r != want {
			t.Fatalf("req %d: %+v, want %+v", i, r, want)
		}
	}
}
