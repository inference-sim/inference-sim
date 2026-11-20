// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// sample is one series' value from an engine's exposition, by its full name and labels.
func sample(m engine.Metrics, name string, labels map[string]string) (float64, bool) {
	for _, f := range m.Families {
		for _, s := range f.Samples {
			if s.Name != name {
				continue
			}
			ok := true
			for k, v := range labels {
				ok = ok && s.Labels[k] == v
			}
			if ok {
				return s.Value, true
			}
		}
	}
	return 0, false
}

// run drives an engine to completion and returns its exposition at the end and the drafts its
// trace scheduled and had accepted.
func runSpec(t *testing.T, extra string, accept int) (engine.Metrics, int, int, int, []int) {
	t.Helper()
	var e engine.Engine
	newEngine := func() (engine.Engine, error) {
		var err error
		e, err = build(t, extra)()
		return e, err
	}
	tr := enginetest.TestEngine(t, newEngine, arrivals(req("a", 40, 12), req("b", 70, 9), req("c", 20, 7)),
		enginetest.Options{Accepted: func(engine.Batch, engine.Scheduled) int { return accept }})
	var drafts, accepted, verifies int
	perPos := make([]int, 8)
	for _, st := range tr.Steps {
		got := st.Result.Emitted
		for _, r := range st.Batch.Reqs {
			if r.Drafts == 0 {
				continue
			}
			verifies++
			drafts += r.Drafts
			// The tokens the request emitted beyond the one the step always samples are its accepted
			// drafts; a stop mid-verify discards the rest, so the trace bounds them from above.
			a := min(max(got[r.ID]-1, 0), r.Drafts)
			accepted += a
			for i := 0; i < a; i++ {
				perPos[i]++
			}
		}
	}
	return e.Metrics(tr.End), verifies, drafts, accepted, perPos
}

// A pool that speculates exposes vllm:spec_decode_num_drafts, _num_draft_tokens,
// _num_accepted_tokens and _num_accepted_tokens_per_pos, and their values are the sums of what its
// batches scheduled and accepted (SpecDecodingStats.observe_draft); one that does not exposes none.
// core/metrics' own tests compare the counters with vLLM's logger; this shows the engine feeds them.
func TestTheEngineExposesTheSpecDecodeCounters(t *testing.T) {
	const k = 3
	for _, c := range []struct {
		name, extra string
		accept      int
	}{
		{"ngram, two accepted", "speculative_config: {method: ngram, num_speculative_tokens: 3}\n", 2},
		{"eagle3, none accepted", "speculative_config: {method: eagle3, num_speculative_tokens: 3}\n", 0},
		{"eagle3, all accepted", "speculative_config: {method: eagle3, num_speculative_tokens: 3}\n", 3},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, verifies, drafts, accepted, perPos := runSpec(t, c.extra, c.accept)
			if verifies == 0 {
				t.Fatal("the run verified no drafts; the test proves nothing")
			}
			for name, want := range map[string]int{
				"vllm:spec_decode_num_drafts_total":          verifies,
				"vllm:spec_decode_num_draft_tokens_total":    drafts,
				"vllm:spec_decode_num_accepted_tokens_total": accepted,
			} {
				if got, ok := sample(m, name, nil); !ok || int(got) != want {
					t.Errorf("%s = %v (present %v), want %d", name, got, ok, want)
				}
			}
			for pos := 0; pos < k; pos++ {
				lbl := map[string]string{"position": string(rune('0' + pos))}
				if got, ok := sample(m, "vllm:spec_decode_num_accepted_tokens_per_pos_total", lbl); !ok || int(got) != perPos[pos] {
					t.Errorf("position %d = %v (present %v), want %d", pos, got, ok, perPos[pos])
				}
			}
			if _, ok := sample(m, "vllm:spec_decode_num_accepted_tokens_per_pos_total", map[string]string{"position": "3"}); ok {
				t.Errorf("a counter exists for position %d of a pool that drafts %d tokens", k, k)
			}
		})
	}
	t.Run("without speculation", func(t *testing.T) {
		m, _, _, _, _ := runSpec(t, "", 0)
		for _, f := range m.Families {
			if strings.HasPrefix(f.Name, "vllm:spec_decode") {
				t.Errorf("a pool that does not speculate exposes %s", f.Name)
			}
		}
	})
}

// A data-parallel group exposes the counters too, summed over its ranks.
func TestADataParallelGroupExposesTheSpecDecodeCounters(t *testing.T) {
	g, err := build(t, "data_parallel_size: 2\nspeculative_config: {method: ngram, num_speculative_tokens: 3}\n")()
	if err != nil {
		t.Fatal(err)
	}
	m := g.Metrics(0)
	if _, ok := sample(m, "vllm:spec_decode_num_drafts_total", nil); !ok {
		t.Error("a speculating data-parallel group exposes no spec_decode counters")
	}
}
