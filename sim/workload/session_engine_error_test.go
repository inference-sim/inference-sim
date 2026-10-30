package workload

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"

	"github.com/inference-sim/inference-sim/sim"
)

// failsAfterAdmit is a FIFO that admits request "s0" and runs the first chunk of its prefill,
// then ends it in Batch.Finished as an error before any token is out, as blis-vllm's v0.32.0 engine ends a running
// request whose KV load failed under kv_load_failure_policy "fail"
// (vllm/v1/core/sched/scheduler.py:2306-2325@5df63a72). By then the driver has marked it running.
type failsAfterAdmit struct {
	*enginetest.FIFO
	steps int
}

func (f *failsAfterAdmit) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := f.FIFO.Schedule(now)
	if err != nil {
		return b, err
	}
	f.steps++
	if f.steps == 2 {
		if ferr := f.Finish("s0", engine.FinishAbort, now); ferr != nil {
			return b, ferr
		}
		b = engine.Batch{Finished: []engine.Finished{{ID: "s0", Reason: engine.FinishError}}}
	}
	return b, nil
}

// A session round the engine fails with no output cancels its session (BC-17), end to end: the
// driver drops it as unservable and hands it to the session manager in the dropped state the
// manager recognizes, so no next round is generated. The round was admitted and running when
// the engine failed it (its 10-token prompt prefills 4 tokens a step, so it has run and has no
// output), which is the case the dropped state must not read as a completion.
func TestSession_EngineFailedRoundCancelsTheSession(t *testing.T) {
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 4, StepBaseUs: 10})
	if err != nil {
		t.Fatal(err)
	}
	sm := NewSessionManager([]SessionBlueprint{makeTestBlueprint("sess", 5, 1000, "", 1_000_000)})
	d := sim.NewEngineDriver(&failsAfterAdmit{FIFO: fifo}, sim.EngineDriverConfig{Horizon: 1 << 40})
	var states []sim.RequestState
	var followUps int
	d.OnRequestDone = func(r *sim.Request, tick int64) []*sim.Request {
		states = append(states, r.State)
		next := sm.OnComplete(r, tick)
		followUps += len(next)
		return next
	}
	d.InjectArrival(&sim.Request{ID: "s0", SessionID: "sess", RoundIndex: 0, State: sim.StateQueued,
		InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5)})
	d.Run()

	if d.Metrics.DroppedUnservable != 1 || d.Metrics.CompletedRequests != 0 {
		t.Fatalf("dropped %d, completed %d; want the round dropped, not completed",
			d.Metrics.DroppedUnservable, d.Metrics.CompletedRequests)
	}
	if len(states) != 1 || states[0] != sim.StateQueued {
		t.Errorf("the done callback saw states %v; want one call in the dropped state %q", states, sim.StateQueued)
	}
	if followUps != 0 {
		t.Errorf("the session generated %d follow-up rounds after an engine-failed round; want 0", followUps)
	}
}
