package vllm

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"gopkg.in/yaml.v3"
)

// skipRecorder prices every batch at one microsecond and records the SkipDrafts of each batch it
// was asked to price, as a pricer of one rank's batch sees it.
type skipRecorder struct{ seen *[]bool }

func (p skipRecorder) StepTime(b engine.Batch) engine.Duration {
	*p.seen = append(*p.seen, b.SkipDrafts)
	return 1
}

// skipEngine is an EAGLE3 engine on the V2 model runner with a budget of 8 tokens a step, over dp
// data-parallel ranks, whose pricer records what it is asked to price.
func skipEngine(t *testing.T, dp int, seen *[]bool) engine.Engine {
	t.Helper()
	src := "block_size: 4\nmax_num_seqs: 4\nmax_num_batched_tokens: 8\nmax_model_len: 256\nnum_gpu_blocks: 80\n" +
		"use_v2_model_runner: true\nasync_scheduling: false\npipeline_parallel_size: 1\n" +
		"speculative_config: {method: eagle3, num_speculative_tokens: 2}\n"
	if dp > 1 {
		src += "data_parallel_size: 2\n"
	}
	var node yaml.Node
	if err := yaml.Unmarshal([]byte(src), &node); err != nil {
		t.Fatal(err)
	}
	e, err := Factory{Pricer: func(engine.Spec) (Pricer, error) { return skipRecorder{seen}, nil }}.New(
		engine.Spec{Kind: Kind, UpstreamVersion: Version, Config: *node.Content[0]})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// step schedules one batch at now, prices it, completes it, and returns it.
func step(t *testing.T, e engine.Engine, now engine.Time) engine.Batch {
	t.Helper()
	b, err := e.Schedule(now)
	if err != nil {
		t.Fatal(err)
	}
	if b.Empty() {
		t.Fatalf("at %d: an empty batch", now)
	}
	d := e.StepTime(b)
	if _, err := e.Complete(b.ID, engine.Outcome{}, now+d); err != nil {
		t.Fatal(err)
	}
	return b
}

// A batch states vLLM v0.33.0's skip_drafts, and the pricer is asked to price it so: a 20-token
// prompt at a budget of 8 runs two non-final chunks, whose drafts no step verifies, and then its
// final chunk, which samples and drafts (Scheduler._drafts_unused,
// vllm/v1/core/sched/scheduler.py:502-544@4ba787ec).
func TestABatchStatesSkipDraftsAndIsPricedSo(t *testing.T) {
	var seen []bool
	e := skipEngine(t, 1, &seen)
	if err := e.Add(req("a", 20, 4), 0); err != nil {
		t.Fatal(err)
	}
	var got []bool
	for now := engine.Time(0); len(got) < 3; now += 10 {
		got = append(got, step(t, e, now).SkipDrafts)
	}
	if want := []bool{true, true, false}; !equalBools(got, want) || !equalBools(seen, want) {
		t.Errorf("batches state skip_drafts %v and were priced with %v, want %v", got, seen, want)
	}
}

// Under data parallelism the drafters propose nothing only when every rank allows it
// (dp_utils.py:75-88@4ba787ec). Two ranks each running a non-final chunk skip together, and every
// rank's batch is priced so. A rank with nothing to run runs a dummy batch, whose empty scheduler
// output does not allow it, so a lone non-final chunk on the other rank still drafts.
func TestTheDPGroupSkipsDraftsOnlyWhenEveryRankAllowsIt(t *testing.T) {
	t.Run("both ranks run non-final chunks", func(t *testing.T) {
		var seen []bool
		e := skipEngine(t, 2, &seen)
		for _, id := range []string{"a", "b"} {
			if err := e.Add(req(id, 20, 4), 0); err != nil {
				t.Fatal(err)
			}
		}
		if b := step(t, e, 0); !b.SkipDrafts || !equalBools(seen, []bool{true, true}) {
			t.Errorf("batch skip_drafts %t, ranks priced with %v; want true on both", b.SkipDrafts, seen)
		}
	})
	t.Run("one rank idle", func(t *testing.T) {
		var seen []bool
		e := skipEngine(t, 2, &seen)
		if err := e.Add(req("a", 20, 4), 0); err != nil {
			t.Fatal(err)
		}
		if b := step(t, e, 0); b.SkipDrafts {
			t.Errorf("batch skip_drafts %t with the other rank idle; its dummy batch keeps the group drafting", b.SkipDrafts)
		}
		for _, s := range seen {
			if s {
				t.Errorf("a rank priced with skip_drafts %v; want none", seen)
			}
		}
	})
}

func equalBools(a, b []bool) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
