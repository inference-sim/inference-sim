package sim

import (
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/units"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// puller is a FIFO that holds a remote prefill and, at its next schedule, starts the pull of 40
// tokens it states as a Move from the peer the request named to this engine's endpoint, as an
// engine that disaggregates does (engine.Request.Peer, engine.Transfer.Move).
type puller struct {
	*enginetest.FIFO
	peer    map[string]plan.Endpoint
	pending []string
	self    plan.Endpoint
	tokens  int
}

func (e *puller) Add(r engine.Request, now engine.Time) error {
	if r.RemotePrefill {
		if err := r.Validate(); err != nil {
			return err
		}
		e.peer[r.ID] = *r.Peer
		e.pending = append(e.pending, r.ID)
		return nil
	}
	return e.FIFO.Add(r, now)
}

func (e *puller) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	for i, id := range e.pending {
		b.TransfersStarted = append(b.TransfersStarted, engine.Transfer{ID: engine.TransferID(2000 + i), Request: id,
			Kind: engine.TransferLoad, Tokens: e.tokens, Blocks: (e.tokens + 15) / 16,
			Move: plan.Transfer{Tokens: units.Tokens(e.tokens), From: e.peer[id], To: e.self, Kind: plan.TransferPeer}})
	}
	e.pending = nil
	return b, err
}

func (e *puller) Plan() plan.PoolPlan { return tierPlan() }

func newPuller(t *testing.T, tokens int) *puller {
	t.Helper()
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 1, StepBaseUs: 50})
	if err != nil {
		t.Fatal(err)
	}
	return &puller{FIFO: fifo, peer: map[string]plan.Endpoint{}, self: plan.Endpoint{Node: 1, Rack: 0}, tokens: tokens}
}

// The host tells a decode engine where it pulls from, and the pull it starts reaches the host
// priced by the Environment from the engine's own Move: the peer the host stated is the Move's
// From, the pull's size is rounded to whole blocks by the cost model, and the driver adds nothing.
func TestEngineDriver_APullIsPricedFromTheMoveTheEngineStates(t *testing.T) {
	eng := newPuller(t, 40)
	type ask struct {
		tokens   int64
		from, to plan.Endpoint
	}
	var asked []ask
	pe := testutil.PullEnv{BlockTokens: 16, Pull: func(tokens int64, from, to plan.Endpoint) kernel.TransferDemand {
		asked = append(asked, ask{tokens, from, to})
		return kernel.TransferDemand{Bytes: tokens * 1000, Floor: 7 * time.Microsecond,
			Links: []kernel.LinkUse{{Link: "l", BytesPerSecond: 1e9}}}
	}}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, Environment: pe})
	var got kernel.TransferDemand
	var pulled, at int64
	d.OnPullPriced = func(_ *Request, tokens int64, demand kernel.TransferDemand, now int64) {
		got, pulled, at = demand, tokens, now
	}
	peer := plan.Endpoint{Node: 3, Rack: 2}
	d.AddRemote(&Request{ID: "r", InputTokens: tokens(40), OutputTokens: tokens(2), State: StateWaitingForRemoteKVs}, peer, 5)

	if eng.peer["r"] != peer {
		t.Errorf("the engine was told its peer is %+v, want %+v", eng.peer["r"], peer)
	}
	want := ask{48, peer, plan.Endpoint{Node: 1}}
	if len(asked) != 1 || asked[0] != want {
		t.Errorf("the Environment was asked %+v, want exactly %+v", asked, want)
	}
	if got.Bytes != 48_000 || pulled != 40 || at != 5 {
		t.Errorf("the host received demand %+v for %d tokens at %d; want 48000 bytes for 40 tokens at 5", got, pulled, at)
	}
}

// noPlan is an engine that starts a tier copy and states no plan.
type noPlan struct{ engine.Engine }

func (e noPlan) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.Engine.Schedule(now)
	if err == nil && !b.Empty() {
		b.TransfersStarted = []engine.Transfer{tierTransfer(1001, "a", engine.TransferStore)}
	}
	return b, err
}

func runOneRequest(d *EngineDriver) {
	d.InjectArrival(&Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1)})
	d.Run()
}

func panicOf(f func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg, _ = r.(string)
		}
	}()
	f()
	return ""
}

// A transfer the cost model cannot price is a failure of the run, never a transfer priced at
// zero: an engine that starts one and states no plan, a host with no Environment, and a Move
// the plan cannot make (a tier it does not list) each stop the run and say which.
func TestEngineDriver_ATransferThatCannotBePricedStopsTheRun(t *testing.T) {
	newFIFO := func() *enginetest.FIFO {
		f, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 1, StepBaseUs: 50})
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	cases := []struct {
		name string
		d    *EngineDriver
		want string
	}{
		{"not planned", NewEngineDriver(noPlan{newFIFO()}, EngineDriverConfig{Horizon: 1 << 40, Environment: tierEnv(100)}),
			"states no plan"},
		{"no environment", NewEngineDriver(&tiered{FIFO: newFIFO(), done: map[engine.TransferID]int64{}},
			EngineDriverConfig{Horizon: 1 << 40}), "no Environment"},
		{"unlisted tier", NewEngineDriver(&unlistedTier{tiered{FIFO: newFIFO(), done: map[engine.TransferID]int64{}}},
			EngineDriverConfig{Horizon: 1 << 40, Environment: tierEnv(100)}), "not a device of this pool's plan"},
	}
	for _, c := range cases {
		if msg := panicOf(func() { runOneRequest(c.d) }); !strings.Contains(msg, c.want) {
			t.Errorf("%s: the run stopped with %q, want a message containing %q", c.name, msg, c.want)
		}
	}
}

// unlistedTier is tiered with a plan that lists no tier.
type unlistedTier struct{ tiered }

func (e *unlistedTier) Plan() plan.PoolPlan {
	p := tierPlan()
	p.Tiers = nil
	return p
}

// An offload copy takes its demand alone in whole microseconds, truncated, and at least one; a
// P/D pull is rounded up by the host (cluster.AloneTicks). The two rules are the ones the retired
// pricers had, so that moving the price to the Environment moves none.
func TestEngineDriver_ATierCopyIsTruncatedToWholeMicroseconds(t *testing.T) {
	for _, c := range []struct {
		floor time.Duration
		want  int64
	}{
		{1999 * time.Nanosecond, 1}, {2 * time.Microsecond, 2}, {100*time.Microsecond + 900*time.Nanosecond, 100},
		{0, 1}, {400 * time.Nanosecond, 1},
	} {
		if got := TierTicks(kernel.TransferDemand{Floor: c.floor}); got != c.want {
			t.Errorf("a tier copy with floor %v takes %d ticks, want %d", c.floor, got, c.want)
		}
	}
}
