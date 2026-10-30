package sim

import (
	"testing"

	"github.com/inference-sim/blis-schemas/engine"
	"github.com/inference-sim/blis-schemas/engine/enginetest"
)

// tiered is a FIFO whose first batch that runs requests starts one CPU-tier store and two loads,
// each priced at 100 us, and records when each is reported done.
type tiered struct {
	*enginetest.FIFO
	started bool
	next    engine.TransferID
	done    map[engine.TransferID]int64
}

func (e *tiered) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	if err != nil || b.Empty() || e.started {
		return b, err
	}
	e.started = true
	for _, k := range []engine.TransferKind{engine.TransferStore, engine.TransferLoad, engine.TransferLoad} {
		e.next++
		b.TransfersStarted = append(b.TransfersStarted, engine.Transfer{ID: 1000 + e.next, Kind: k, Tier: "cpu", Blocks: 1})
	}
	return b, nil
}

func (e *tiered) TransferTime(engine.Transfer) (engine.Duration, bool) { return 100, true }

func (e *tiered) TransferDone(id engine.TransferID, ok bool, now engine.Time) error {
	if id > 1000 {
		e.done[id] = now
		return nil
	}
	return e.FIFO.TransferDone(id, ok, now)
}

// vLLM's CPU offload worker runs stores and loads on two handlers, one per direction, each
// chaining its transfers (a transfer's stream waits on the one before it), and every transfer's
// stream first waits on the compute stream, so it starts after the forward pass of the step whose
// metadata carries it (SingleDirectionOffloadingHandler.transfer_async,
// vllm/v1/kv_offload/cpu/gpu_worker.py:665-670, 834-852@5df63a72). A batch of 50 us at 0 starting a
// store and two loads: the store runs 50-150, beside the loads, which run 50-150 and 150-250.
func TestEngineDriver_TierTransfersRunOneStreamPerDirectionAfterTheirStep(t *testing.T) {
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 1, StepBaseUs: 50})
	if err != nil {
		t.Fatal(err)
	}
	eng := &tiered{FIFO: fifo, done: map[engine.TransferID]int64{}}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40})
	d.InjectArrival(&Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1)})
	d.Run()
	want := map[engine.TransferID]int64{1001: 150, 1002: 150, 1003: 250}
	for id, at := range want {
		if eng.done[id] != at {
			t.Errorf("transfers done at %v, want %v", eng.done, want)
			break
		}
	}
}
