// Original to blis-vllm.

package vllm

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
)

// drain schedules and completes until no batch runs, at time steps of one microsecond.
func drain(t *testing.T, e engine.Engine, now engine.Time) engine.Time {
	t.Helper()
	for range 1000 {
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			if w, ok := e.NextWake(now); ok {
				now = max(w, now+1)
				continue
			}
			return now
		}
		now += e.StepTime(b)
		if _, err := e.Complete(b.ID, engine.Outcome{}, now); err != nil {
			t.Fatal(err)
		}
	}
	t.Fatal("the engine never went idle")
	return now
}

// A request's entry is freed once its final step is consumed: an engine that served many
// requests holds none of them after, and the scheduler holds none either.
func TestAFinishedRequestIsFreed(t *testing.T) {
	e, err := build(t, "")()
	if err != nil {
		t.Fatal(err)
	}
	eng := e.(*Engine)
	for _, id := range []string{"a", "b", "c"} {
		r := req(id, 40, 3)
		if err := e.Add(r, 0); err != nil {
			t.Fatal(err)
		}
	}
	drain(t, e, 0)
	if len(eng.reqs) != 0 || len(eng.aborted) != 0 {
		t.Errorf("after every request finished: %d entries, %d aborted, want none", len(eng.reqs), len(eng.aborted))
	}
	// Freed, the ID can be added again, as vLLM accepts a request ID once its request is gone.
	if err := e.Add(req("a", 40, 3), 0); err != nil {
		t.Errorf("re-adding a finished request's ID: %v", err)
	}
}

// An aborted request is freed: at once when no batch in flight names it, and once that batch
// completes when one does. A request still running keeps its entry, so adding its ID twice is
// still refused.
func TestAnAbortedRequestIsFreed(t *testing.T) {
	e, err := build(t, "")()
	if err != nil {
		t.Fatal(err)
	}
	eng := e.(*Engine)
	if err := e.Add(req("waiting", 40, 20), 0); err != nil {
		t.Fatal(err)
	}
	if err := e.Finish("waiting", engine.FinishAbort, 0); err != nil {
		t.Fatal(err)
	}
	if _, ok := eng.reqs["waiting"]; ok {
		t.Error("a request aborted before it ran is still held")
	}
	if err := e.Add(req("a", 40, 20), 0); err != nil {
		t.Fatal(err)
	}
	if err := e.Add(req("a", 40, 20), 0); err == nil {
		t.Error("adding a running request's ID again was accepted")
	}
	b, err := e.Schedule(0)
	if err != nil || b.Empty() {
		t.Fatalf("schedule: %v %v", b, err)
	}
	if err := e.Finish("a", engine.FinishAbort, 5); err != nil {
		t.Fatal(err)
	}
	if _, ok := eng.reqs["a"]; !ok {
		t.Error("a request aborted with its batch in flight was dropped before that batch completed")
	}
	if _, err := e.Complete(b.ID, engine.Outcome{}, 100); err != nil {
		t.Fatal(err)
	}
	if len(eng.reqs) != 0 || len(eng.aborted) != 0 {
		t.Errorf("after the batch completed: %d entries, %d aborted, want none", len(eng.reqs), len(eng.aborted))
	}
}

// A data-parallel group frees a request's rank entry and its rank engine's entry, on completion
// and on abort.
func TestADataParallelRequestIsFreed(t *testing.T) {
	src := "block_size: 4\nmax_num_seqs: 4\nmax_num_batched_tokens: 32\nmax_model_len: 256\nnum_gpu_blocks: 80\n" +
		"use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\ndata_parallel_size: 2\n"
	e, err := build(t, src)()
	if err != nil {
		t.Fatal(err)
	}
	g := e.(*DPGroup)
	held := func() (n int) {
		n = len(g.rankOf) + len(g.abortedOn)
		for _, r := range g.ranks {
			n += len(r.e.reqs) + len(r.e.aborted)
		}
		return n
	}
	for _, id := range []string{"a", "b", "c", "d"} {
		if err := e.Add(req(id, 12, 3), 0); err != nil {
			t.Fatal(err)
		}
	}
	now := drain(t, e, 0)
	if n := held(); n != 0 {
		t.Errorf("after every request finished: %d entries held, want none", n)
	}
	// Abort one waiting and one in flight.
	for _, id := range []string{"x", "y"} {
		if err := e.Add(req(id, 12, 20), now); err != nil {
			t.Fatal(err)
		}
	}
	var b engine.Batch
	for range 100 {
		if b, err = e.Schedule(now); err != nil {
			t.Fatal(err)
		}
		if !b.Empty() {
			break
		}
		w, ok := e.NextWake(now)
		if !ok {
			t.Fatal("the group neither scheduled nor asked to be woken")
		}
		now = max(w, now+1)
	}
	if b.Empty() {
		t.Fatal("the group never scheduled the new requests")
	}
	for _, id := range []string{"x", "y"} {
		if err := e.Finish(id, engine.FinishAbort, now); err != nil {
			t.Fatal(err)
		}
	}
	now += e.StepTime(b)
	if _, err := e.Complete(b.ID, engine.Outcome{}, now); err != nil {
		t.Fatal(err)
	}
	drain(t, e, now)
	if n := held(); n != 0 {
		t.Errorf("after the aborts: %d entries held, want none", n)
	}
}
