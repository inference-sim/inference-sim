package sim

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/units"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// peering is a FIFO that is engine.Peered: its first batch that runs a request queues one message
// to "b:2" and starts one WRITE of 16 tokens to a peer; it records the messages delivered to it
// (and answers the first with a message of its own) and when its WRITE was reported done.
type peering struct {
	*enginetest.FIFO
	addr      string
	started   bool
	outbox    []engine.PeerMessage
	seq       uint64
	got       []engine.PeerMessage
	gotAt     []engine.Time
	writeDone map[engine.TransferID]int64
	placed    map[string]plan.Endpoint
}

func (e *peering) queue(to string) {
	e.seq++
	e.outbox = append(e.outbox, engine.PeerMessage{From: e.addr, To: to, Seq: e.seq, Bytes: 8})
}

func (e *peering) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	if err != nil || b.Empty() || e.started {
		return b, err
	}
	e.started = true
	e.queue("b:2")
	b.TransfersStarted = append(b.TransfersStarted, engine.Transfer{ID: 2001, Request: "a", Kind: engine.TransferStore,
		Tokens: 16, Blocks: 1,
		Move: plan.Transfer{Tokens: units.Tokens(16), Kind: plan.TransferPeer, From: plan.Endpoint{}, To: plan.Endpoint{}}})
	return b, nil
}

func (e *peering) Plan() plan.PoolPlan { return tierPlan() }
func (e *peering) PeerAddr() string    { return e.addr }
func (e *peering) Outbox() []engine.PeerMessage {
	o := e.outbox
	e.outbox = nil
	return o
}

func (e *peering) Deliver(m engine.PeerMessage, now engine.Time) error {
	e.got, e.gotAt = append(e.got, m), append(e.gotAt, now)
	if len(e.got) == 1 {
		e.queue(m.From)
	}
	return nil
}

func (e *peering) PlacePeer(addr string, at plan.Endpoint) error {
	e.placed[addr] = at
	return nil
}

func (e *peering) TransferDone(id engine.TransferID, ok bool, now engine.Time) error {
	if id == 2001 {
		e.writeDone[id] = int64(now)
		return nil
	}
	return e.FIFO.TransferDone(id, ok, now)
}

type sendRecord struct {
	m   engine.PeerMessage
	now int64
}

type recordingRouter struct{ sent []sendRecord }

func (r *recordingRouter) SendPeer(m engine.PeerMessage, now int64) {
	r.sent = append(r.sent, sendRecord{m, now})
}

func newPeering(t *testing.T, addr string) *peering {
	t.Helper()
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 1, StepBaseUs: 50})
	if err != nil {
		t.Fatal(err)
	}
	return &peering{FIFO: fifo, addr: addr, writeDone: map[engine.TransferID]int64{}, placed: map[string]plan.Endpoint{}}
}

// writeBytes is what the peering engine states for its WRITE (PeerWriteBytes).
const writeBytes = 4096

func (e *peering) PeerWriteBytes(t engine.Transfer) (int64, bool) { return writeBytes, t.ID == 2001 }

// writeEnv is a PullEnv that prices a peer WRITE (PeerWritePricer) as its bytes at a fixed floor, and
// records the bytes it was asked to price.
type writeEnv struct {
	testutil.PullEnv
	ticks int64
	asked *[]int64
}

func (w writeEnv) PeerWriteDemand(_ plan.PoolPlan, _ engine.Transfer, bytes int64) (kernel.TransferDemand, error) {
	*w.asked = append(*w.asked, bytes)
	return kernel.TransferDemand{Floor: time.Duration(w.ticks) * time.Microsecond}, nil
}

func peerEnv(ticks int64) writeEnv {
	return writeEnv{PullEnv: testutil.FixedPull(16, func(int64, plan.Endpoint, plan.Endpoint) int64 { return ticks }),
		ticks: ticks, asked: new([]int64)}
}

// The driver hands the router each message an engine queued, at the instant of the call that
// queued it, and does not hand it twice.
func TestEngineDriver_PeerMessagesReachTheRouterOnceAtTheInstantTheEngineQueuedThem(t *testing.T) {
	eng := newPeering(t, "a:1")
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, Environment: peerEnv(100)})
	r := &recordingRouter{}
	d.SetPeerRouter(r)
	d.InjectArrival(&Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1)})
	d.Run()
	if len(r.sent) != 1 || r.sent[0].m.To != "b:2" || r.sent[0].now != 0 {
		t.Fatalf("router saw %+v, want one message to b:2 at tick 0", r.sent)
	}
}

// A message delivered to the engine reaches it with the instant, and what it queues in answer
// goes to the router at that instant.
func TestEngineDriver_ADeliveredPeerMessageIsAnsweredAtTheDeliveryInstant(t *testing.T) {
	eng := newPeering(t, "a:1")
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, Environment: peerEnv(100)})
	r := &recordingRouter{}
	d.SetPeerRouter(r)
	d.DeliverPeer(engine.PeerMessage{From: "b:2", To: "a:1", Seq: 1, Bytes: 8}, 700)
	if len(eng.got) != 1 || eng.gotAt[0] != 700 {
		t.Fatalf("engine got %v at %v, want one message at 700", eng.got, eng.gotAt)
	}
	if len(r.sent) != 1 || r.sent[0].m.To != "b:2" || r.sent[0].now != 700 {
		t.Fatalf("router saw %+v, want the answer to b:2 at 700", r.sent)
	}
}

// Without a host that shares links, a peer WRITE takes its demand alone and the engine hears it done
// when that has passed; with one (OnPeerWritePriced) the engine hears it done only when the host says.
func TestEngineDriver_APeerWriteLandsAfterItsDemandAloneOrWhenTheHostSays(t *testing.T) {
	alone := newPeering(t, "a:1")
	d := NewEngineDriver(alone, EngineDriverConfig{Horizon: 1 << 40, Environment: peerEnv(100)})
	d.InjectArrival(&Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1)})
	d.Run()
	if got := alone.writeDone[2001]; got != 100 {
		t.Errorf("an unshared WRITE started at 0 was done at %d, want 100", got)
	}

	shared := newPeering(t, "a:1")
	d = NewEngineDriver(shared, EngineDriverConfig{Horizon: 1 << 40, Environment: peerEnv(100)})
	var started int64 = -1
	d.OnPeerWritePriced = func(id engine.TransferID, _ kernel.TransferDemand, at int64) {
		started = at
		d.schedule(&peerLandEvent{at: at + 300, id: id})
	}
	d.InjectArrival(&Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1)})
	d.Run()
	if started != 0 || shared.writeDone[2001] != 300 {
		t.Errorf("a host-shared WRITE started at %d and was done at %d, want 0 and 300", started, shared.writeDone[2001])
	}
}

type peerLandEvent struct {
	at int64
	id engine.TransferID
}

func (e *peerLandEvent) Timestamp() int64    { return e.at }
func (e *peerLandEvent) Priority() int       { return PriorityQueued }
func (e *peerLandEvent) run(d *EngineDriver) { d.PeerWriteLanded(e.id, e.at) }

// An engine that is not Peered has no address and neither delivers nor places peers.
func TestEngineDriver_AnEngineThatIsNotPeeredHasNoPeerAddress(t *testing.T) {
	fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 8, InFlight: 1, StepBaseUs: 50})
	if err != nil {
		t.Fatal(err)
	}
	d := NewEngineDriver(fifo, EngineDriverConfig{Horizon: 1 << 40})
	if d.PeerAddr() != "" {
		t.Errorf("PeerAddr = %q, want none", d.PeerAddr())
	}
	if err := d.PlacePeer("x:1", plan.Endpoint{}); err != nil {
		t.Errorf("PlacePeer on an engine that is not PeerPlaced: %v", err)
	}
}

// A WRITE is priced from the bytes the engine states, through the Environment's PeerWritePricer,
// and not as a pull.
func TestEngineDriver_APeerWriteIsPricedFromTheBytesTheEngineStates(t *testing.T) {
	eng := newPeering(t, "a:1")
	env := peerEnv(100)
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, Environment: env})
	d.InjectArrival(&Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1)})
	d.Run()
	if len(*env.asked) != 1 || (*env.asked)[0] != writeBytes {
		t.Errorf("the Environment was asked to price WRITEs of %v bytes, want one of %d", *env.asked, writeBytes)
	}
}

type noBytes struct{ *peering }

func (noBytes) PeerWriteBytes(engine.Transfer) (int64, bool) { return 0, false }

// A WRITE the engine states no bytes for, an engine without PeerWriteBytes, and an Environment
// that cannot price a WRITE each stop the run naming what is missing; none is priced at zero.
func TestEngineDriver_AWriteWithoutBytesOrAPricerStopsTheRun(t *testing.T) {
	run := func(eng engine.Engine, e env.Environment) (msg string) {
		defer func() { msg = strings.TrimSpace(strings.Join([]string{msg, toString(recover())}, " ")) }()
		d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1 << 40, Environment: e})
		d.InjectArrival(&Request{ID: "a", InputTokens: tokens(8), OutputTokens: tokens(1)})
		d.Run()
		return ""
	}
	if got := run(noBytes{newPeering(t, "a:1")}, peerEnv(100)); !strings.Contains(got, "states no bytes") {
		t.Errorf("an engine that states no bytes: %q", got)
	}
	if got := run(newPeering(t, "a:1"), peerEnv(100).PullEnv); !strings.Contains(got, "PeerWriteDemand") {
		t.Errorf("an Environment that cannot price a WRITE: %q", got)
	}
}

func toString(v any) string {
	if v == nil {
		return ""
	}
	return fmt.Sprint(v)
}
