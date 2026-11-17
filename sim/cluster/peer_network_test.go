package cluster

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	"github.com/inference-sim/inference-sim/sim"
)

// talker is a FIFO that is engine.Peered: its first batch that runs a request queues one message
// to the next engine's address and one to an address nobody has, each carrying the tick it was sent
// at, and it records the ticks messages reach it and the places it was told peers sit.
type talker struct {
	*enginetest.FIFO
	addr, next string
	started    bool
	seq        uint64
	outbox     []engine.PeerMessage
	arrivals   *[]arrival
	placed     map[string]plan.Endpoint
}

type arrival struct {
	to, from string
	sentAt   int64
	at       engine.Time
}

func (e *talker) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	if err != nil || b.Empty() || e.started {
		return b, err
	}
	e.started = true
	for _, to := range []string{e.next, "ghost:9"} {
		e.seq++
		e.outbox = append(e.outbox, engine.PeerMessage{From: e.addr, To: to, Seq: e.seq, Bytes: 8,
			Payload: []byte(strconv.FormatInt(int64(now), 10))})
	}
	return b, nil
}

func (e *talker) PeerAddr() string { return e.addr }
func (e *talker) Outbox() []engine.PeerMessage {
	o := e.outbox
	e.outbox = nil
	return o
}
func (e *talker) Deliver(m engine.PeerMessage, now engine.Time) error {
	sent, _ := strconv.ParseInt(string(m.Payload), 10, 64)
	*e.arrivals = append(*e.arrivals, arrival{to: e.addr, from: m.From, sentAt: sent, at: now})
	return nil
}
func (e *talker) PlacePeer(addr string, at plan.Endpoint) error {
	e.placed[addr] = at
	return nil
}

func runTalkers(t *testing.T, latency int64) (*ClusterSimulator, []arrival, []*talker) {
	t.Helper()
	var arrivals []arrival
	var engines []*talker
	cfg := newTestDeploymentConfig(2)
	cfg.PeerMessageLatency = latency
	cfg.PeerEndpoint = func(id InstanceID) plan.Endpoint { return plan.Endpoint{Node: len(string(id)) % 3} }
	cfg.NewEngine = func(index int, _ sim.InstanceConfig) (*sim.EngineDriver, error) {
		fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 64, InFlight: 1, StepBaseUs: 10})
		if err != nil {
			return nil, err
		}
		eng := &talker{FIFO: fifo, addr: fmt.Sprintf("host%d:7000", index), next: fmt.Sprintf("host%d:7000", 1-index),
			arrivals: &arrivals, placed: map[string]plan.Endpoint{}}
		engines = append(engines, eng)
		return sim.NewEngineDriver(eng, sim.EngineDriverConfig{Horizon: 1 << 40}), nil
	}
	var reqs []*sim.Request
	for i := 0; i < 4; i++ {
		reqs = append(reqs, &sim.Request{ID: string(rune('a' + i)), ArrivalTime: int64(i), InputTokens: make([]sim.TokenID, 16),
			OutputTokens: make([]sim.TokenID, 1), State: sim.StateQueued})
	}
	cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	return cs, arrivals, engines
}

// A peer message arrives after the stated latency, the message to an address nobody has is dropped
// and counted without being delivered, and every message sent is either delivered or dropped.
func TestPeerMessagesArriveAfterTheStatedLatencyAndUnknownAddressesAreDropped(t *testing.T) {
	cs, arrivals, engines := runTalkers(t, 250)
	if len(arrivals) == 0 {
		t.Fatal("no peer message was delivered")
	}
	for _, a := range arrivals {
		if int64(a.at) != a.sentAt+250 {
			t.Errorf("message %s -> %s sent at %d arrived at %d, want %d", a.from, a.to, a.sentAt, a.at, a.sentAt+250)
		}
	}
	ps := cs.PeerStats()
	sent, delivered, dropped := ps.Sent, ps.Delivered, ps.Dropped
	started := 0
	for _, e := range engines {
		if e.started {
			started++
		}
	}
	if sent != 2*started || delivered != len(arrivals) || dropped != started || sent != delivered+dropped {
		t.Errorf("sent %d, delivered %d, dropped %d for %d talking engines and %d arrivals", sent, delivered, dropped, started, len(arrivals))
	}
}

// Every engine is told where every other engine sits, before it runs, in the pool numbering the
// cluster's PeerEndpoint gives.
func TestPeersArePlacedBothWaysBeforeTheRun(t *testing.T) {
	_, _, engines := runTalkers(t, 0)
	if len(engines) != 2 {
		t.Fatalf("%d engines built, want 2", len(engines))
	}
	for i, e := range engines {
		other := engines[1-i].addr
		if _, ok := e.placed[other]; !ok {
			t.Errorf("engine %s was not told where %s sits (placed %v)", e.addr, other, e.placed)
		}
		if _, ok := e.placed[e.addr]; ok {
			t.Errorf("engine %s was told where it sits itself", e.addr)
		}
	}
}
