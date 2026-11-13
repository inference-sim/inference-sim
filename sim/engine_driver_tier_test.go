package sim

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/units"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// tierDevice is the catalog device the fakes' tier stands for, and tierPlan the pool plan they
// state (engine.Planned): one node, one tier.
const tierDevice = "cpu_dram"

func tierPlan() plan.PoolPlan {
	return plan.PoolPlan{Hardware: "fake-chip", Layout: plan.Layout{Tensor: 1, RanksPerNode: 1, Nodes: 1},
		Tiers: []plan.KVTier{{Device: tierDevice}}}
}

// tierTransfer is a copy of one block of 16 tokens as an engine states it: its own report (the
// tier's name in the engine's configuration) and the Move the cost model prices.
func tierTransfer(id engine.TransferID, req string, k engine.TransferKind) engine.Transfer {
	mk := plan.TransferToTier
	if k == engine.TransferLoad {
		mk = plan.TransferFromTier
	}
	return engine.Transfer{ID: id, Request: req, Kind: k, Tier: "cpu", Tokens: 16, Blocks: 1,
		Move: plan.Transfer{Tokens: units.Tokens(16), Kind: mk, Tier: tierDevice}}
}

// tierEnv prices every tier copy at ticks microseconds alone, whatever its size: a demand of a
// floor and no payload.
func tierEnv(ticks int64) testutil.PullEnv {
	return testutil.PullEnv{BlockTokens: 16, Tier: func(plan.Transfer) kernel.TransferDemand {
		return kernel.TransferDemand{Floor: time.Duration(ticks) * time.Microsecond}
	}}
}

// tiered is a FIFO whose first batch that runs requests starts one CPU-tier store and two loads,
// each priced at 100 us by the Environment, and records when each is reported done.
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
		b.TransfersStarted = append(b.TransfersStarted, tierTransfer(1000+e.next, "a", k))
	}
	return b, nil
}

// Plan implements engine.Planned.
func (e *tiered) Plan() plan.PoolPlan { return tierPlan() }

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
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, Environment: tierEnv(100)})
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

// emptyTiered is a FIFO whose first schedule that runs nothing, after one that ran a batch,
// starts one CPU-tier store priced at 100 us.
type emptyTiered struct {
	*enginetest.FIFO
	ran, started bool
	done         map[engine.TransferID]int64
}

func (e *emptyTiered) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	if err != nil {
		return b, err
	}
	if !b.Empty() {
		e.ran = true
		return b, nil
	}
	if e.ran && !e.started {
		e.started = true
		b.TransfersStarted = []engine.Transfer{tierTransfer(1001, "a", engine.TransferStore)}
	}
	return b, nil
}

// Plan implements engine.Planned.
func (e *emptyTiered) Plan() plan.PoolPlan { return tierPlan() }

func (e *emptyTiered) TransferDone(id engine.TransferID, ok bool, now engine.Time) error {
	if id > 1000 {
		e.done[id] = now
		return nil
	}
	return e.FIFO.TransferDone(id, ok, now)
}

// A copy a schedule that ran no forward pass starts still waits on the compute stream, which
// holds the batch already in flight (SingleDirectionOffloadingHandler.transfer_async's wait on
// the current stream, vllm/v1/kv_offload/cpu/gpu_worker.py:665-670@5df63a72): with a 50 us batch
// in flight from 0, a 100 us store an empty schedule at 0 starts runs 50-150, not 0-100.
func TestEngineDriver_TierTransferOfAnEmptyScheduleWaitsForTheBatchInFlight(t *testing.T) {
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 2, StepBaseUs: 50})
	if err != nil {
		t.Fatal(err)
	}
	eng := &emptyTiered{FIFO: fifo, done: map[engine.TransferID]int64{}}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, Environment: tierEnv(100)})
	d.InjectArrival(&Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1)})
	d.Run()
	if !eng.started || eng.done[1001] != 150 {
		t.Fatalf("store started %v, done at %d; want done at 150", eng.started, eng.done[1001])
	}
}
