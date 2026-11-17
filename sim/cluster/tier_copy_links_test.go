package cluster

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/units"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// storingEngine is a FIFO whose first batch that runs requests starts one store to the CPU tier, and
// records when the simulator reports it done.
type storingEngine struct {
	*enginetest.FIFO
	id      engine.TransferID
	started bool
	done    map[engine.TransferID]int64
}

func (e *storingEngine) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	if err != nil || b.Empty() || e.started {
		return b, err
	}
	e.started = true
	b.TransfersStarted = []engine.Transfer{{ID: e.id, Request: b.Reqs[0].ID, Kind: engine.TransferStore, Tier: "cpu", Tokens: 16,
		Blocks: 1, Move: plan.Transfer{Tokens: units.Tokens(16), Kind: plan.TransferToTier, Tier: "cpu_dram"}}}
	return b, nil
}

func (e *storingEngine) Plan() plan.PoolPlan {
	return plan.PoolPlan{Hardware: "fake-chip", Layout: plan.Layout{Tensor: 1, RanksPerNode: 1, Nodes: 1},
		Tiers: []plan.KVTier{{Device: "cpu_dram"}}}
}

func (e *storingEngine) TransferDone(id engine.TransferID, ok bool, now engine.Time) error {
	if id == e.id {
		e.done[id] = now
		return nil
	}
	return e.FIFO.TransferDone(id, ok, now)
}

// tierCopyDone runs a cluster of two instances that each start one tier copy of a megabyte over the
// same link, at 1 GB/s (1000 us alone), for the requests given, and returns when each copy was done.
func tierCopyDone(t *testing.T, contention bool, requests int) []int64 {
	t.Helper()
	done := map[engine.TransferID]int64{}
	cfg := newTestDeploymentConfig(2)
	cfg.TierCopyContention = contention
	cfg.NewEngine = func(index int, _ sim.InstanceConfig) (*sim.EngineDriver, error) {
		fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 64, InFlight: 1, StepBaseUs: 10})
		if err != nil {
			return nil, err
		}
		eng := &storingEngine{FIFO: fifo, id: engine.TransferID(1001 + index), done: done}
		environment := testutil.PullEnv{BlockTokens: 16, Tier: func(plan.Transfer) kernel.TransferDemand {
			return kernel.TransferDemand{Bytes: 1 << 20, Links: []kernel.LinkUse{{Link: "tier/cpu_dram/store/node0", BytesPerSecond: 1 << 30}}}
		}}
		return sim.NewEngineDriver(eng, sim.EngineDriverConfig{Horizon: 1 << 40, Environment: environment}), nil
	}
	var reqs []*sim.Request
	for i := 0; i < requests; i++ {
		reqs = append(reqs, &sim.Request{ID: string(rune('a' + i)), ArrivalTime: 0, InputTokens: make([]sim.TokenID, 16),
			OutputTokens: make([]sim.TokenID, 1), State: sim.StateQueued})
	}
	cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	var out []int64
	for _, at := range done {
		out = append(out, at)
	}
	return out
}

// Offload-tier copies share the links they cross with each other (#2108): two instances each store a
// megabyte over one link at the same time and each takes longer than the same copy alone, while with
// the arbiter off, each takes its demand alone.
func TestTierCopiesOnOneLinkShareItsCapacity(t *testing.T) {
	alone := tierCopyDone(t, true, 1)
	if len(alone) != 1 {
		t.Fatalf("%d copies done with one request, want 1", len(alone))
	}
	shared := tierCopyDone(t, true, 2)
	if len(shared) != 2 {
		t.Fatalf("%d copies done with two requests, want 2", len(shared))
	}
	unshared := tierCopyDone(t, false, 2)
	for _, at := range shared {
		if at <= alone[0]+500 {
			t.Errorf("a copy sharing its link was done at %d, want well past %d, when the same copy alone was", at, alone[0])
		}
	}
	for _, at := range unshared {
		// The arbiter rounds a copy's time up to a whole tick, as it does a handoff's, and the copy
		// without it truncates (sim.TierTicks): a lone copy differs by at most that tick.
		if at < alone[0]-1 || at > alone[0]+1 {
			t.Errorf("without the arbiter a copy was done at %d, want %d (within a tick), as when alone", at, alone[0])
		}
	}
}
