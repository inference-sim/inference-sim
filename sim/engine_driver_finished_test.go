package sim

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

// failsUnrun is a FIFO that never runs request "a": its first schedule after "a" arrives ends it
// in Batch.Finished as an error, as blis-vllm's v0.32.0 engine ends a request whose KV load
// failed under kv_load_failure_policy "fail" (update_from_output of a step that ran nothing,
// vllm/v1/core/sched/scheduler.py:2306-2325@5df63a72).
type failsUnrun struct {
	*enginetest.FIFO
	held, done bool
}

func (f *failsUnrun) Add(r engine.Request, now engine.Time) error {
	if r.ID == "a" {
		f.held = true
		return nil
	}
	return f.FIFO.Add(r, now)
}

func (f *failsUnrun) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := f.FIFO.Schedule(now)
	if err == nil && f.held && !f.done {
		f.done = true
		b.Finished = append(b.Finished, engine.Finished{ID: "a", Reason: engine.FinishError})
	}
	return b, err
}

// A request the engine ends in Batch.Finished, never having run it (blis-schemas 45f1d4e), leaves
// the driver's books: it is dropped as unservable, as a request the engine refuses is, its done
// callback fires once, its deadline does not count it again as a timeout, and INV-1's five terms
// still sum to the requests injected. That holds whether the decision that ends it runs another
// request or nothing.
func TestEngineDriver_RequestFinishedInABatchLeavesTheBooks(t *testing.T) {
	for _, alone := range []bool{false, true} {
		fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 2, TokenBudget: 64, StepBaseUs: 10})
		if err != nil {
			t.Fatal(err)
		}
		d := NewEngineDriver(&failsUnrun{FIFO: fifo}, EngineDriverConfig{Horizon: 1 << 40})
		done := map[string]int{}
		d.OnRequestDone = func(r *Request, _ int64) []*Request { done[r.ID]++; return nil }
		reqs := []*Request{{ID: "a", InputTokens: tokens(4), OutputTokens: tokens(2), Deadline: 5000}}
		if !alone {
			reqs = append(reqs, &Request{ID: "b", InputTokens: tokens(4), OutputTokens: tokens(2)})
		}
		for _, r := range reqs {
			d.InjectArrival(r)
		}
		d.Run()
		m := d.Metrics
		if m.DroppedUnservable != 1 || m.TimedOutRequests != 0 || done["a"] != 1 {
			t.Errorf("alone %v: a dropped %d, timed out %d, done %d times; want 1, 0, 1",
				alone, m.DroppedUnservable, m.TimedOutRequests, done["a"])
		}
		if _, ok := m.Requests["a"]; ok {
			t.Errorf("alone %v: a keeps a request row", alone)
		}
		if got := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable + m.TimedOutRequests; got != len(reqs) {
			t.Errorf("alone %v: INV-1 terms sum to %d, want %d", alone, got, len(reqs))
		}
		if !alone && (m.CompletedRequests != 1 || done["b"] != 1) {
			t.Errorf("b completed %d, done %d times; want 1 and 1", m.CompletedRequests, done["b"])
		}
	}
}
