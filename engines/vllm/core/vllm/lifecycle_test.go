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

// Under async scheduling a request can finish while a later batch in flight still names it. Its
// ID is refused until that batch completes, and completing the stale batch does not touch a new
// request that then reuses the ID (blis-vllm#38).
func TestAFinishedIDIsNotReusedWhileAQueuedBatchNamesIt(t *testing.T) {
	e, err := build(t, "async_scheduling: true\n")()
	if err != nil {
		t.Fatal(err)
	}
	eng := e.(*Engine)
	if err := e.Add(req("a", 8, 1), 0); err != nil {
		t.Fatal(err)
	}
	b1, err := e.Schedule(0)
	if err != nil || b1.Empty() {
		t.Fatalf("first batch: %v %v", b1, err)
	}
	b2, err := e.Schedule(1)
	if err != nil || b2.Empty() {
		t.Fatalf("second batch, queued behind the first: %v %v", b2, err)
	}
	res, err := e.Complete(b1.ID, engine.Outcome{}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Finished) != 1 || res.Finished[0].ID != "a" {
		t.Fatalf("a should finish with its first batch: %+v", res.Finished)
	}
	if err := e.Add(req("a", 8, 4), 11); err == nil {
		t.Fatal("reusing a finished request's ID while a queued batch names it was accepted")
	}
	if _, err := e.Complete(b2.ID, engine.Outcome{}, 20); err != nil {
		t.Fatal(err)
	}
	if len(eng.reqs) != 0 || len(eng.aborted) != 0 {
		t.Fatalf("after the stale batch: %d entries, %d deferred, want none", len(eng.reqs), len(eng.aborted))
	}
	// Freed now, the ID is a new request, untouched by the batches before it.
	if err := e.Add(req("a", 8, 4), 21); err != nil {
		t.Fatalf("reusing the freed ID: %v", err)
	}
	if st := eng.reqs["a"]; st == nil || st.emitted != 0 || st.sampled != 0 || st.finished {
		t.Fatalf("the reused ID's request starts with state %+v", st)
	}
	tr := 21
	got := 0
	for range 100 {
		b, err := e.Schedule(engine.Time(tr))
		if err != nil {
			t.Fatal(err)
		}
		if b.Empty() {
			if w, ok := e.NextWake(engine.Time(tr)); ok {
				tr = int(max(w, engine.Time(tr+1)))
				continue
			}
			break
		}
		tr += 10
		r, err := e.Complete(b.ID, engine.Outcome{}, engine.Time(tr))
		if err != nil {
			t.Fatal(err)
		}
		got += r.Emitted["a"]
		for _, f := range r.Finished {
			if f.ID == "a" && f.OutputTokens != 4 {
				t.Errorf("the reused ID's request finished with %d tokens, want 4", f.OutputTokens)
			}
		}
	}
	if got != 4 {
		t.Errorf("the reused ID's request emitted %d tokens, want 4", got)
	}
}

// A data-parallel group keeps a finished request's rank entry, and refuses its ID, while an
// iteration in flight names it.
func TestADataParallelFinishedIDIsNotReusedWhileAQueuedIterationNamesIt(t *testing.T) {
	src := "block_size: 4\nmax_num_seqs: 4\nmax_num_batched_tokens: 32\nmax_model_len: 256\nnum_gpu_blocks: 80\n" +
		"use_v2_model_runner: false\nasync_scheduling: true\npipeline_parallel_size: 1\ndata_parallel_size: 2\n"
	e, err := build(t, src)()
	if err != nil {
		t.Fatal(err)
	}
	g := e.(*DPGroup)
	if err := e.Add(req("a", 8, 1), 0); err != nil {
		t.Fatal(err)
	}
	var its []engine.Batch
	now := engine.Time(0)
	for len(its) < 2 {
		b, err := e.Schedule(now)
		if err != nil {
			if err == engine.ErrNoCapacity {
				break
			}
			t.Fatal(err)
		}
		if !b.Empty() {
			its = append(its, b)
		} else if w, ok := e.NextWake(now); ok {
			now = max(w, now+1)
			continue
		} else {
			break
		}
		now++
	}
	if len(its) < 2 {
		t.Skipf("the group queued %d iteration(s) naming the request; this case needs two", len(its))
	}
	res, err := e.Complete(its[0].ID, engine.Outcome{}, now+10)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Finished) != 1 || res.Finished[0].ID != "a" {
		t.Fatalf("a should finish with its first iteration: %+v", res.Finished)
	}
	if _, ok := g.rankOf["a"]; !ok {
		t.Fatal("a finished request's rank entry went while an iteration in flight names it")
	}
	if err := e.Add(req("a", 8, 4), now+11); err == nil {
		t.Fatal("reusing a finished request's ID while a queued iteration names it was accepted")
	}
	if _, err := e.Complete(its[1].ID, engine.Outcome{}, now+20); err != nil {
		t.Fatal(err)
	}
	if len(g.rankOf) != 0 || len(g.abortedOn) != 0 {
		t.Errorf("after the stale iteration: %d rank entries, %d deferred, want none", len(g.rankOf), len(g.abortedOn))
	}
	if err := e.Add(req("a", 8, 4), now+21); err != nil {
		t.Errorf("reusing the freed ID: %v", err)
	}
}
