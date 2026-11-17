package enginetest

import (
	"errors"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// echo is a FIFO engine that answers each peer message with one message back to its sender.
// Its fault, if set, breaks one law of engine.Peered.
type echo struct {
	*FIFO
	addr   string
	seq    uint64
	out    []engine.PeerMessage
	fault  string
	random *int
}

func (e *echo) PeerAddr() string { return e.addr }

// placed is an echo engine that takes peer placements; its fault, if set, breaks the law.
type placed struct {
	*echo
	peers map[string]plan.Endpoint
}

func (p *placed) PlacePeer(addr string, at plan.Endpoint) error {
	if (at.Node < 0 || at.Rack < 0) && p.fault != "places-negative" {
		return errors.New("placed: negative endpoint")
	}
	p.peers[addr] = at
	return nil
}

func (e *echo) Outbox() []engine.PeerMessage {
	out := e.out
	if e.fault != "keeps" {
		e.out = nil
	}
	return out
}

func (e *echo) Deliver(m engine.PeerMessage, _ engine.Time) error {
	if m.To != e.addr && e.fault != "accepts-any" {
		return errors.New("echo: not my address")
	}
	if m.To != e.addr {
		return nil
	}
	e.seq++
	reply := engine.PeerMessage{From: e.addr, To: m.From, Seq: e.seq, Bytes: m.Bytes, Payload: m.Payload}
	switch e.fault {
	case "skips-seq":
		reply.Seq++
	case "wrong-from":
		reply.From = "elsewhere:1"
	case "nondeterministic":
		*e.random++
		reply.Bytes += int64(*e.random)
	}
	e.out = append(e.out, reply)
	return nil
}

func newEcho(fault string) func() (engine.Engine, error) {
	n := 0
	return func() (engine.Engine, error) {
		f, err := NewFIFO(fifoCfg())
		if err != nil {
			return nil, err
		}
		return &echo{FIFO: f, addr: "a:1", fault: fault, random: &n}, nil
	}
}

var peerScript = []PeerStep{
	{At: 10, Msg: engine.PeerMessage{From: "b:2", To: "a:1", Seq: 1, Bytes: 32, Payload: []byte("connect")}},
	{At: 20, Msg: engine.PeerMessage{From: "b:2", To: "a:1", Seq: 2, Bytes: 48, Payload: []byte("fetch")}},
}

func TestAPeerEngineThatKeepsTheLawsPasses(t *testing.T) {
	if err := CheckPeered(newEcho(""), peerScript); err != nil {
		t.Fatal(err)
	}
}

// Each broken law is caught, naming what broke.
func TestAPeerEngineThatBreaksALawIsCaught(t *testing.T) {
	for fault, want := range map[string]string{
		"keeps":            "drained again",
		"skips-seq":        "Seq 2, want 1",
		"wrong-from":       `from "elsewhere:1"`,
		"accepts-any":      "accepted a message addressed to",
		"nondeterministic": "different messages",
	} {
		t.Run(fault, func(t *testing.T) {
			err := CheckPeered(newEcho(fault), peerScript)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("CheckPeered = %v, want an error naming %q", err, want)
			}
		})
	}
}

func TestAnEngineThatIsNotPeeredIsCaught(t *testing.T) {
	err := CheckPeered(newFIFO(fifoCfg()), peerScript)
	if err == nil || !strings.Contains(err.Error(), "not an engine.Peered") {
		t.Fatalf("CheckPeered on a FIFO = %v, want it refused as not Peered", err)
	}
}

func newPlaced(fault string) func() (engine.Engine, error) {
	inner := newEcho(fault)
	return func() (engine.Engine, error) {
		e, err := inner()
		if err != nil {
			return nil, err
		}
		return &placed{echo: e.(*echo), peers: map[string]plan.Endpoint{}}, nil
	}
}

func TestAPlacedPeerEngineThatKeepsTheLawsPasses(t *testing.T) {
	if err := CheckPeered(newPlaced(""), peerScript); err != nil {
		t.Fatal(err)
	}
}

func TestAPlacedPeerEngineThatAcceptsANegativeEndpointIsCaught(t *testing.T) {
	err := CheckPeered(newPlaced("places-negative"), peerScript)
	if err == nil || !strings.Contains(err.Error(), "negative node") {
		t.Fatalf("CheckPeered = %v, want an error naming the negative node", err)
	}
}
