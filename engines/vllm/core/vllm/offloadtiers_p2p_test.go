package vllm

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	kvtier "github.com/inference-sim/blis-vllm/core/connector/tiering"
)

// p2pTier is TieringOffloadingSpec beside the base block's GPU blocks: a CPU tier of 200 one-block
// chunks of 4 KiB and a p2p tier whose side channel binds the given port.
func p2pTier(port int, extra string) string {
	return fmt.Sprintf("kv_offload: {connector: OffloadingConnector, spec: TieringOffloadingSpec, cpu_bytes: 819200, "+
		"bytes_per_block: 4096, secondary_tiers: [{type: p2p, port: %d%s}]}\n", port, extra)
}

// The endpoints the two engines are placed at: different nodes of the base block's two-node pool.
var (
	prefillerAt = plan.Endpoint{Node: 0, Rack: 0}
	decoderAt   = plan.Endpoint{Node: 1, Rack: 0}
)

const (
	prefillerAddr = "localhost:5710"
	decoderAddr   = "localhost:5711"
)

// p2pEngine is an engine with a p2p tier on the port, placed at at.
func p2pEngine(t *testing.T, port int, at plan.Endpoint, extra ...string) *Engine {
	t.Helper()
	e := mustEngine(t, strings.Join(extra, "")+p2pTier(port, ""))
	if err := e.PlaceAt(at); err != nil {
		t.Fatal(err)
	}
	return e
}

// kvParams is a request's kv_transfer_params tag.
func kvParams(r engine.Request, json string) engine.Request {
	r.Tags = map[string]string{TagKVTransferParams: json}
	return r
}

func fromPrefiller(id string) string {
	return `{"remote_prefiller":{"remote_host":"localhost","remote_port":5710,"kv_request_id":"` + id + `"}}`
}

func fromSource(id string) string {
	return `{"remote_kv_source":{"remote_host":"localhost","remote_port":5710,"kv_request_id":"` + id + `"}}`
}

// p2pNode is one engine of a p2pWorld and what the world knows of its run.
type p2pNode struct {
	addr  string
	e     *Engine
	batch *engine.Batch
	end   engine.Time
	// xfers are the transfers the engine started, due at the time the world ends them.
	xfers []p2pDue
	// first is the tokens of each request's first scheduled step; computed the tokens it ran in all.
	first, computed map[string]int
	finished        map[string]engine.Time
}

type p2pDue struct {
	at engine.Time
	id engine.TransferID
}

type p2pFlight struct {
	at engine.Time
	m  engine.PeerMessage
}

type p2pAdd struct {
	at   engine.Time
	node int
	r    engine.Request
}

// startedTransfer is a transfer an engine reported, and which.
type startedTransfer struct {
	addr string
	tr   engine.Transfer
	// bytes is what the engine states the transfer moves when it starts (PeerWriteBytes); ok false
	// for a transfer that is not a WRITE to a peer.
	bytes int64
	ok    bool
}

// p2pWorld is a hand-run simulator for engines with a p2p tier: it delivers each engine's outbox
// to its peer after a latency, runs the transfers the engines report for a time, steps an engine
// at its wakes, and reports nothing the contract does not let a simulator know.
type p2pWorld struct {
	t       *testing.T
	nodes   []*p2pNode
	latency engine.Duration
	xfer    engine.Duration
	now     engine.Time
	flight  []p2pFlight
	adds    []p2pAdd
	// drop reports a message the network loses.
	drop    func(m engine.PeerMessage, at engine.Time) bool
	dropped int
	// failXfer reports a transfer the world ends failed.
	failXfer func(tr engine.Transfer) bool
	started  []startedTransfer
	// each, if set, runs once every instant the world steps.
	each func()
	// sent are the messages the engines sent, with the instant each was sent.
	sent []engine.PeerMessage
}

func newP2PWorld(t *testing.T, engines ...*Engine) *p2pWorld {
	t.Helper()
	w := &p2pWorld{t: t, latency: 50, xfer: 200}
	for _, e := range engines {
		w.nodes = append(w.nodes, &p2pNode{addr: e.PeerAddr(), e: e, first: map[string]int{}, computed: map[string]int{},
			finished: map[string]engine.Time{}})
	}
	return w
}

// placeAll tells every engine where each of the others runs, as the simulator does before the
// first Schedule; ats are the engines' endpoints in order.
func (w *p2pWorld) placeAll(ats ...plan.Endpoint) {
	w.t.Helper()
	for i, n := range w.nodes {
		for j, m := range w.nodes {
			if i != j {
				if err := n.e.PlacePeer(m.addr, ats[j]); err != nil {
					w.t.Fatal(err)
				}
			}
		}
	}
}

func (w *p2pWorld) add(at engine.Time, node int, r engine.Request) {
	w.adds = append(w.adds, p2pAdd{at, node, r})
}

func (w *p2pWorld) node(addr string) *p2pNode {
	for _, n := range w.nodes {
		if n.addr == addr {
			return n
		}
	}
	return nil
}

func (w *p2pWorld) drain(n *p2pNode) {
	for _, m := range n.e.Outbox() {
		if err := m.Validate(); err != nil {
			w.t.Fatal(err)
		}
		w.sent = append(w.sent, m)
		if w.drop != nil && w.drop(m, w.now) {
			w.dropped++
			continue
		}
		w.flight = append(w.flight, p2pFlight{w.now + w.latency, m})
	}
}

// run steps the world until nothing is due, or until the instant limit.
func (w *p2pWorld) run(limit engine.Time) {
	w.t.Helper()
	for iter := 0; iter < 2_000_000; iter++ {
		now := w.now
		for i := 0; i < len(w.adds); i++ {
			if a := w.adds[i]; a.at <= now {
				if err := w.nodes[a.node].e.Add(a.r, now); err != nil {
					w.t.Fatal(err)
				}
				w.drain(w.nodes[a.node])
				w.adds = append(w.adds[:i], w.adds[i+1:]...)
				i--
			}
		}
		var due []p2pFlight
		keep := w.flight[:0]
		for _, f := range w.flight {
			if f.at <= now {
				due = append(due, f)
			} else {
				keep = append(keep, f)
			}
		}
		w.flight = keep
		sort.SliceStable(due, func(i, j int) bool {
			if due[i].at != due[j].at {
				return due[i].at < due[j].at
			}
			if due[i].m.From != due[j].m.From {
				return due[i].m.From < due[j].m.From
			}
			return due[i].m.Seq < due[j].m.Seq
		})
		for _, f := range due {
			n := w.node(f.m.To)
			if n == nil {
				w.dropped++
				continue
			}
			if err := n.e.Deliver(f.m, now); err != nil {
				w.t.Fatal(err)
			}
			w.drain(n)
		}
		for _, n := range w.nodes {
			var rest []p2pDue
			for _, x := range n.xfers {
				if x.at > now {
					rest = append(rest, x)
					continue
				}
				ok := true
				if w.failXfer != nil {
					for _, s := range w.started {
						if s.addr == n.addr && s.tr.ID == x.id && w.failXfer(s.tr) {
							ok = false
						}
					}
				}
				if err := n.e.TransferDone(x.id, ok, now); err != nil {
					w.t.Fatal(err)
				}
				w.drain(n)
			}
			n.xfers = rest
		}
		for _, n := range w.nodes {
			if n.batch != nil && n.end <= now {
				res, err := n.e.Complete(n.batch.ID, engine.Outcome{}, now)
				if err != nil {
					w.t.Fatal(err)
				}
				for _, f := range res.Finished {
					n.finished[f.ID] = now
				}
				n.batch = nil
				w.drain(n)
			}
		}
		for _, n := range w.nodes {
			if n.batch != nil {
				continue
			}
			b, err := n.e.Schedule(now)
			if err != nil {
				w.t.Fatal(err)
			}
			for _, tr := range b.TransfersStarted {
				if err := tr.Validate(); err != nil {
					w.t.Fatal(err)
				}
				if err := n.e.Plan().CheckTransfer(tr.Move); err != nil {
					w.t.Fatal(err)
				}
				bytes, ok := n.e.PeerWriteBytes(tr)
				w.started = append(w.started, startedTransfer{n.addr, tr, bytes, ok})
				n.xfers = append(n.xfers, p2pDue{now + w.xfer, tr.ID})
			}
			for _, f := range b.Finished {
				n.finished[f.ID] = now
			}
			for _, s := range b.Reqs {
				if _, seen := n.first[s.ID]; !seen {
					n.first[s.ID] = s.Tokens
				}
				n.computed[s.ID] += s.Tokens
			}
			if !b.Empty() {
				bb := b
				n.batch, n.end = &bb, now+n.e.StepTime(b)
			}
			w.drain(n)
		}
		if w.each != nil {
			w.each()
		}
		next, any := engine.Time(0), false
		consider := func(at engine.Time) {
			if !any || at < next {
				next, any = at, true
			}
		}
		for _, a := range w.adds {
			consider(a.at)
		}
		for _, f := range w.flight {
			consider(f.at)
		}
		for _, n := range w.nodes {
			if n.batch != nil {
				consider(n.end)
			}
			for _, x := range n.xfers {
				consider(x.at)
			}
			if at, ok := n.e.NextWake(now); ok {
				consider(at)
			}
		}
		if !any || next > limit {
			return
		}
		w.now = max(next, now+1)
	}
	w.t.Fatal("the world made no end")
}

// writes are the WRITEs to peers the engines reported.
func (w *p2pWorld) writes() []startedTransfer {
	var out []startedTransfer
	for _, s := range w.started {
		if s.tr.Move.Kind == plan.TransferPeer {
			out = append(out, s)
		}
	}
	return out
}

// A P/D decoder loads its prompt from the prefiller: the prefiller's request (a remote decoder)
// parks its stores, the decoder's fetch binds them, and each chunk crosses as a WRITE the
// simulator is told of as a store from the prefiller's endpoint to the decoder's. The decoder
// computes only the tail past the prompt.
func TestAP2PDecoderLoadsItsPromptFromThePrefiller(t *testing.T) {
	p, d := p2pEngine(t, 5710, prefillerAt), p2pEngine(t, 5711, decoderAt)
	w := newP2PWorld(t, p, d)
	w.placeAll(prefillerAt, decoderAt)
	pr := kvParams(req("p1", 96, 1), `{"remote_decoder":{"kv_request_id":"kv1"}}`)
	dr := kvParams(req("d1", 100, 2), fromPrefiller("kv1"))
	w.add(0, 0, pr)
	w.add(0, 1, dr)
	w.run(10_000_000)

	if _, ok := w.nodes[1].finished["d1"]; !ok {
		t.Fatal("the decoder's request did not finish")
	}
	if got := w.nodes[1].first["d1"]; got != 4 {
		t.Errorf("the decoder's first step computed %d tokens of d1, want the 4 past the prompt", got)
	}
	moved := 0
	for _, s := range w.writes() {
		tr := s.tr
		if s.addr != prefillerAddr || tr.Kind != engine.TransferStore || tr.Tier != "" || tr.Request != "kv1" {
			t.Errorf("WRITE %+v from %s, want a store for kv1 from the prefiller, naming no tier", tr, s.addr)
		}
		if tr.Move.From != prefillerAt || tr.Move.To != decoderAt {
			t.Errorf("WRITE moves %+v to %+v, want the prefiller's endpoint to the decoder's", tr.Move.From, tr.Move.To)
		}
		if tr.Blocks != tr.Tokens/16 {
			t.Errorf("WRITE of %d blocks and %d tokens", tr.Blocks, tr.Tokens)
		}
		moved += tr.Tokens
	}
	if moved != 96 {
		t.Errorf("the WRITEs moved %d tokens, want the prompt's 96", moved)
	}
	if _, priced := p.TransferTime(w.writes()[0].tr); priced {
		t.Error("the engine priced a WRITE to a peer; the simulator prices it from its Move")
	}
	// Each WRITE moves its chunks of the CPU tier: one 4 KiB block a chunk, one worker.
	bytes := int64(0)
	for _, s := range w.writes() {
		if !s.ok {
			t.Fatalf("the engine stated no bytes for its WRITE %+v when it started", s.tr)
		}
		bytes += s.bytes
	}
	if bytes != 96/16*4096 {
		t.Errorf("the WRITEs move %d bytes, want the prompt's 6 chunks of 4096", bytes)
	}
	if _, ok := p.PeerWriteBytes(engine.Transfer{ID: 1 << 40}); ok {
		t.Error("the engine states bytes for a transfer that is not a WRITE to a peer")
	}
}

// A symmetric consumer probes the producer, which answers from its own CPU tier: the prompt a
// prior request left there crosses as WRITEs, and the consumer computes only the tail.
func TestAP2PConsumerFetchesAPromptFromTheProducersCPUTier(t *testing.T) {
	p, d := p2pEngine(t, 5710, prefillerAt), p2pEngine(t, 5711, decoderAt)
	w := newP2PWorld(t, p, d)
	w.placeAll(prefillerAt, decoderAt)
	w.add(0, 0, req("a1", 96, 2))
	w.add(500_000, 1, kvParams(req("b1", 100, 2), fromSource("kvs")))
	w.run(10_000_000)

	if _, ok := w.nodes[1].finished["b1"]; !ok {
		t.Fatal("the consumer's request did not finish")
	}
	if got := w.nodes[1].first["b1"]; got != 4 {
		t.Errorf("the consumer's first step computed %d tokens of b1, want the 4 past the producer's prompt", got)
	}
	moved := 0
	for _, s := range w.writes() {
		if s.addr != prefillerAddr || s.tr.Request != "kvs" || s.tr.Move.To != decoderAt {
			t.Errorf("WRITE %+v from %s, want one for kvs from the producer to the consumer", s.tr, s.addr)
		}
		moved += s.tr.Tokens
	}
	if moved != 96 {
		t.Errorf("the producer wrote %d tokens, want a1's 96", moved)
	}
}

// A WRITE to a peer the simulator never placed is refused: no WRITE is reported, the decoder's
// load fails, the request's lookups MISS from then on, and the decoder computes its whole prompt.
func TestAWriteToAPeerNeverPlacedIsRefused(t *testing.T) {
	p, d := p2pEngine(t, 5710, prefillerAt), p2pEngine(t, 5711, decoderAt)
	w := newP2PWorld(t, p, d)
	// Only the decoder knows where the prefiller is.
	if err := d.PlacePeer(prefillerAddr, prefillerAt); err != nil {
		t.Fatal(err)
	}
	w.add(0, 0, kvParams(req("p1", 96, 1), `{"remote_decoder":{"kv_request_id":"kv1"}}`))
	dr := kvParams(req("d1", 100, 2), fromPrefiller("kv1"))
	w.add(0, 1, dr)
	// The tier's lookups for the request HIT while its session is up and MISS once its load failed.
	tier := d.off.peer().tier
	ctx := &kvtier.ReqContext{ReqID: "d1", Params: map[string]any{"remote_prefiller": map[string]any{
		"remote_host": "localhost", "remote_port": 5710.0, "kv_request_id": "kv1"}}}
	var hit, miss bool
	w.each = func() {
		_, done := w.nodes[1].finished["d1"]
		switch tier.Lookup(kvtier.MakeKey([]byte{1}, 0), ctx) {
		case kvtier.Hit:
			hit = hit || !done
		case kvtier.Miss:
			miss = miss || !done && hit
		}
	}
	w.run(10_000_000)

	if ws := w.writes(); len(ws) != 0 {
		t.Fatalf("the prefiller reported WRITEs %+v to a peer it was never told the place of", ws)
	}
	if _, ok := w.nodes[1].finished["d1"]; !ok {
		t.Fatal("the decoder's request did not finish")
	}
	if got := w.nodes[1].computed["d1"]; got != 100+1 {
		t.Errorf("the decoder ran %d tokens of d1, want its whole 100-token prompt and one decode step", got)
	}
	if !hit || !miss {
		t.Errorf("lookups for the request were HIT before its load failed: %v, and MISS after it: %v; want both", hit, miss)
	}
}

// PlacePeer takes a node and rack, refuses a negative one, and keeps the last for an address: a
// WRITE reported after a second placement ends at the second endpoint.
func TestPlacePeerRefusesANegativeEndpointAndTheLastPlacementWins(t *testing.T) {
	e := p2pEngine(t, 5710, prefillerAt)
	for _, at := range []plan.Endpoint{{Node: -1}, {Rack: -1}} {
		if err := e.PlacePeer(decoderAddr, at); err == nil {
			t.Errorf("PlacePeer at %+v: no error", at)
		}
	}
	if err := e.PlacePeer("", decoderAt); err == nil {
		t.Error("PlacePeer with no address: no error")
	}
	moved := plan.Endpoint{Node: 1, Rack: 3}
	p, d := p2pEngine(t, 5710, prefillerAt), p2pEngine(t, 5711, decoderAt)
	w := newP2PWorld(t, p, d)
	w.placeAll(prefillerAt, decoderAt)
	if err := p.PlacePeer(decoderAddr, moved); err != nil {
		t.Fatal(err)
	}
	w.add(0, 0, kvParams(req("p1", 96, 1), `{"remote_decoder":{"kv_request_id":"kv1"}}`))
	w.add(0, 1, kvParams(req("d1", 100, 2), fromPrefiller("kv1")))
	w.run(10_000_000)
	ws := w.writes()
	if len(ws) == 0 {
		t.Fatal("no WRITE")
	}
	for _, s := range ws {
		if s.tr.Move.To != moved {
			t.Errorf("WRITE ends at %+v, want the last placement %+v", s.tr.Move.To, moved)
		}
	}
}

// A peer that stops answering: the prefiller never hears the decoder (the network loses every
// message to it). The decoder's session never completes its handshake, the load fails when the
// handshake timeout of the engine's environment (2 s) passes on the engine's own clock, and the
// request recomputes its prompt.
func TestADecoderRecomputesWhenItsPrefillerStopsAnswering(t *testing.T) {
	p := p2pEngine(t, 5710, prefillerAt, "env: {VLLM_P2P_HANDSHAKE_TIMEOUT_S: '2'}\n")
	d := p2pEngine(t, 5711, decoderAt, "env: {VLLM_P2P_HANDSHAKE_TIMEOUT_S: '2'}\n")
	w := newP2PWorld(t, p, d)
	w.placeAll(prefillerAt, decoderAt)
	w.drop = func(m engine.PeerMessage, _ engine.Time) bool { return m.To == prefillerAddr }
	w.add(0, 1, kvParams(req("d1", 100, 2), fromPrefiller("kv1")))
	w.run(60_000_000)

	at, ok := w.nodes[1].finished["d1"]
	if !ok {
		t.Fatal("the decoder's request did not finish")
	}
	if at < 2_000_000 {
		t.Errorf("d1 finished at %d us, before the 2 s handshake timeout could have failed its load", at)
	}
	if at > 3_000_000 {
		t.Errorf("d1 finished at %d us, long after the 2 s handshake timeout", at)
	}
	if got := w.nodes[1].computed["d1"]; got != 100+1 {
		t.Errorf("the decoder ran %d tokens of d1, want its whole prompt and one decode step", got)
	}
	if w.dropped == 0 {
		t.Error("the network dropped nothing")
	}
}

// engine.Peered's laws hold for the engine: its messages are valid, from its address, numbered
// 1, 2, ... in send order, the same for the same inputs, and a message to another address is
// refused. The engine under test has a request waiting for a prefiller, whose connect is its
// first message; the script delivers a peer's connect.
func TestTheEngineKeepsThePeeredLaws(t *testing.T) {
	newEngine := func() (engine.Engine, error) {
		e := mustEngine(t, p2pTier(5711, ""))
		if err := e.Add(kvParams(req("d1", 100, 2), fromPrefiller("kv1")), 0); err != nil {
			return nil, err
		}
		// The connector meets the request at its first schedule (on_new_request).
		if _, err := e.Schedule(0); err != nil {
			return nil, err
		}
		return e, nil
	}
	connect := encodeP2PMsg(kvtier.P2PMsg{Type: "connect", PeerID: prefillerAddr, AgentMetadata: []byte("agent"), BlockLen: 4096,
		NumBlocks: 200})
	script := []enginetest.PeerStep{{At: 10, Msg: engine.PeerMessage{From: prefillerAddr, To: decoderAddr, Seq: 1,
		Bytes: int64(len(connect)), Payload: connect}}}
	if err := enginetest.CheckPeered(newEngine, script); err != nil {
		t.Fatal(err)
	}
	e, _ := newEngine()
	out := e.(*Engine).Outbox()
	if len(out) != 1 {
		t.Fatalf("the engine sent %d messages at its request's first schedule, want the connect to its prefiller", len(out))
	}
	if m, err := decodeP2PMsg(out[0].Payload); err != nil || m.Type != "connect" || out[0].To != prefillerAddr {
		t.Errorf("first message %+v (%v), want a connect to %s", out[0], err, prefillerAddr)
	}
}

// An engine with no p2p tier is a Peered with no address, and a message it is handed is an error;
// a payload that is not an encoded control message is an error for one with a tier.
func TestPeerMessagesAreRefusedWhereTheEngineCannotTakeThem(t *testing.T) {
	plain := mustEngine(t, fsTier)
	if plain.PeerAddr() != "" || plain.Outbox() != nil {
		t.Error("an engine with no p2p tier has an address or an outbox")
	}
	m := engine.PeerMessage{From: decoderAddr, To: prefillerAddr, Seq: 1, Bytes: 1, Payload: []byte("{}")}
	if err := plain.Deliver(m, 0); err == nil {
		t.Error("an engine with no p2p tier took a peer message")
	}
	e := p2pEngine(t, 5710, prefillerAt)
	for name, payload := range map[string][]byte{"not json": []byte("connect"), "unknown type": []byte(`{"type":"hello"}`),
		"unknown field": []byte(`{"type":"connect","x":1}`), "bad key": []byte(`{"type":"lookup","keys":["zz"]}`)} {
		m.Payload = payload
		if err := e.Deliver(m, 0); err == nil {
			t.Errorf("%s: Deliver took the payload", name)
		}
	}
	// A message from the engine itself is not deliverable.
	m.From, m.Payload = prefillerAddr, []byte(`{"type":"connect"}`)
	if err := e.Deliver(m, 0); err == nil {
		t.Error("Deliver took a message from the engine's own address")
	}
}

// The engine asks for poll wakes only while the tier has work: an idle engine has none (or a
// simulator would tick forever), a request waiting for a prefiller has one within the tier-poll
// interval, and tier_poll_interval_s of 0 runs no poll thread, so none is asked for.
func TestTheEngineAsksForPollWakesOnlyWhileTheTierHasWork(t *testing.T) {
	e := p2pEngine(t, 5711, decoderAt)
	if at, ok := e.NextWake(0); ok {
		t.Fatalf("an idle engine wakes at %d", at)
	}
	if err := e.Add(kvParams(req("d1", 100, 2), fromPrefiller("kv1")), 100); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Schedule(100); err != nil {
		t.Fatal(err)
	}
	at, ok := e.NextWake(100)
	if !ok || at <= 100 || at > 100+1000 {
		t.Fatalf("a request waiting on a peer wakes at %d (%v), want within the 1 ms poll interval", at, ok)
	}

	none := mustEngine(t, strings.Replace(p2pTier(5711, ""), "bytes_per_block", "tier_poll_interval_s: 0, bytes_per_block", 1))
	if err := none.Add(kvParams(req("d1", 100, 2), fromPrefiller("kv1")), 100); err != nil {
		t.Fatal(err)
	}
	if _, err := none.Schedule(100); err != nil {
		t.Fatal(err)
	}
	if at, ok := none.NextWake(100); ok && at != 100+idlePollUs {
		t.Errorf("with no poll thread the engine wakes at %d for the tier, want only the engine loop's idle poll", at)
	}
}

// A p2p tier the engine cannot run is refused at build, naming its key.
func TestAP2PTierItCannotRunIsRefused(t *testing.T) {
	tier := func(keys string) string {
		return "kv_offload: {connector: OffloadingConnector, spec: TieringOffloadingSpec, cpu_bytes: 819200, " +
			"bytes_per_block: 4096, secondary_tiers: [{type: p2p" + keys + "}]}\n"
	}
	for name, c := range map[string]struct{ extra, want string }{
		"unknown key":       {tier(", root_dir: /kv"), `"root_dir"`},
		"locality":          {tier(", locality: local"), `"locality"`},
		"non-UCX backend":   {tier(", backends: [UCX, GDS]"), "GDS"},
		"backpressure":      {tier(", backpressure: {backpressure_cls: EMABackpressureDetector}"), "issues/13"},
		"port zero":         {tier(", port: 0"), "port"},
		"port too large":    {tier(", port: 70000"), "port"},
		"port not a number": {tier(", port: http"), "port"},
		"zero timeout":      {tier(", unbound_store_timeout_s: 0"), "unbound_store_timeout_s"},
		"negative timeout":  {tier(", unbound_store_timeout_s: -3"), "unbound_store_timeout_s"},
		"text timeout":      {tier(", unbound_store_timeout_s: soon"), "unbound_store_timeout_s"},
		"no threads":        {tier(", num_threads: 0"), "num_threads"},
		"empty host":        {tier(", host: ''"), "host"},
		"two p2p tiers": {"kv_offload: {connector: OffloadingConnector, spec: TieringOffloadingSpec, cpu_bytes: 819200, " +
			"bytes_per_block: 4096, secondary_tiers: [{type: p2p}, {type: p2p, port: 5711}]}\n", "one p2p tier"},
		"top-level backpressure": {"kv_offload: {connector: OffloadingConnector, spec: TieringOffloadingSpec, cpu_bytes: 819200, " +
			"bytes_per_block: 4096, backpressure: {backpressure_cls: EMABackpressureDetector}, secondary_tiers: [{type: p2p}]}\n", "issues/13"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := build(t, c.extra)()
			if err == nil {
				t.Fatal("built, want refused")
			}
			if !strings.Contains(err.Error(), "(p2p)") || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refusal %v, want one naming (p2p) and %s", err, c.want)
			}
		})
	}
}

// The tier's side channel is vLLM's: the host and port default from the environment's
// VLLM_P2P_SIDE_CHANNEL_* (localhost:5710 unstated) and the tier's own keys win, and the engine
// that states a limit it does not read is refused.
func TestTheSideChannelAddressIsTheEnvironmentsOrTheTiers(t *testing.T) {
	for name, c := range map[string]struct{ extra, tier, want string }{
		"default":    {"", "{type: p2p}", "localhost:5710"},
		"tier keys":  {"", "{type: p2p, host: 10.0.0.7, port: 6001}", "10.0.0.7:6001"},
		"env":        {"env: {VLLM_P2P_SIDE_CHANNEL_HOST: gpu-3, VLLM_P2P_SIDE_CHANNEL_PORT: '7000'}\n", "{type: p2p}", "gpu-3:7000"},
		"tier wins":  {"env: {VLLM_P2P_SIDE_CHANNEL_PORT: '7000'}\n", "{type: p2p, port: 7100}", "localhost:7100"},
		"env + only": {"env: {VLLM_P2P_SIDE_CHANNEL_HOST: gpu-3}\n", "{type: p2p, port: 7100}", "gpu-3:7100"},
	} {
		t.Run(name, func(t *testing.T) {
			e := mustEngine(t, c.extra+"kv_offload: {connector: OffloadingConnector, spec: TieringOffloadingSpec, cpu_bytes: 819200, "+
				"bytes_per_block: 4096, secondary_tiers: ["+c.tier+"]}\n")
			if got := e.PeerAddr(); got != c.want {
				t.Fatalf("PeerAddr %q, want %q", got, c.want)
			}
		})
	}
	_, err := build(t, "env: {VLLM_P2P_MAX_PEERS: lots}\n"+p2pTier(5710, ""))()
	if err == nil || !strings.Contains(err.Error(), "VLLM_P2P_MAX_PEERS") {
		t.Errorf("a non-numeric VLLM_P2P_MAX_PEERS: %v, want a refusal naming it", err)
	}
}

// The tier-poll thread starts each round one tier_poll_interval_s after the last: its loop ends a
// round with `self._tier_poll_stop.wait(self._tier_poll_interval_s)` (vllm/v1/kv_offload/tiering/
// manager.py:395@4ba787ec), and Event.wait returns once the timeout has elapsed, so with a round that
// takes no virtual time the next one is due at exactly last + interval (default 1 ms, manager.py:70).
// The engine runs a round at the first schedule at or after that instant, and asks to be woken at it.
//
// The gate shows only in that wake. A p2p tier is also polled on every step (get_finished_jobs does one
// _poll_once per step, p2p/manager.py:636-645), and a Schedule makes that sweep whether or not a thread
// round is due, so the messages and results a call produces are the same either way. What the round
// decides is when the next one is due: one interval after this one, or, if it was skipped, at the next
// instant the engine is asked. The oracle drives rounds explicitly (tier_poll_interval_s 0) and the
// thread's wait has wall-clock jitter, so neither pins the gate; the model takes the earliest instant
// the wait permits, and this test holds it there to the microsecond from below: a round one microsecond
// early, or an interval one microsecond long, fails it.
//
// One mutant of the gate cannot be caught: a round skipped at its due instant (clock <= nextPoll) is
// re-armed one interval later by arm(), and the same schedule's step sweep does the round's work, so no
// message, result or wake differs. That equivalence is the reason the gate needs no further test.
func TestTheTierPollRoundIsDueExactlyOneIntervalAfterTheLast(t *testing.T) {
	e := p2pEngine(t, 5711, decoderAt) // the poll interval is vLLM's default, 1 ms
	if err := e.Add(kvParams(req("d1", 100, 2), fromPrefiller("kv1")), 0); err != nil {
		t.Fatal(err)
	}
	wake := func(now engine.Time) engine.Time {
		t.Helper()
		at, ok := e.NextWake(now)
		if !ok {
			t.Fatalf("a request waiting on a peer asks for no wake at %d", now)
		}
		return at
	}
	step := func(at engine.Time) {
		t.Helper()
		if _, err := e.Schedule(at); err != nil {
			t.Fatal(err)
		}
	}
	step(0) // the first round is due at once
	if got := wake(0); got != 1000 {
		t.Fatalf("after a round at 0 the next is due at %d, want 1000 (one interval later)", got)
	}
	step(999) // 1 us early: no round, so the next stays due at 1000
	if got := wake(999); got != 1000 {
		t.Fatalf("a schedule at 999 ran a round that was due at 1000; the next is now due at %d", got)
	}
	step(1000) // due: a round runs, and the next is one interval later
	if got := wake(1000); got != 2000 {
		t.Fatalf("a schedule at 1000 skipped the round due then; the next wake is %d, want 2000", got)
	}
}

// Two instances built from one engine block state one side-channel address, so a host gives each its
// own (SetPeerAddr), as a deployment sets VLLM_P2P_SIDE_CHANNEL_HOST per node. The address is the one
// the engine's messages come from and the one a peer's message must name; it cannot change once the
// tier has a session, and an address that is not host:port is refused.
func TestAHostGivesEachInstanceItsOwnPeerAddress(t *testing.T) {
	a, b := p2pEngine(t, 5711, decoderAt), p2pEngine(t, 5711, prefillerAt)
	if a.PeerAddr() != b.PeerAddr() {
		t.Fatalf("two engines of one block state %q and %q; the test needs them alike", a.PeerAddr(), b.PeerAddr())
	}
	if err := a.SetPeerAddr("blis-node-1:5711"); err != nil {
		t.Fatal(err)
	}
	if err := b.SetPeerAddr("blis-node-0:5711"); err != nil {
		t.Fatal(err)
	}
	if a.PeerAddr() == b.PeerAddr() || a.PeerAddr() != "blis-node-1:5711" {
		t.Fatalf("after SetPeerAddr the engines are at %q and %q", a.PeerAddr(), b.PeerAddr())
	}
	// a's request names a prefiller; a's connect to it comes from a's new address.
	if err := a.Add(kvParams(req("d1", 100, 2), fromPrefiller("kv1")), 0); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Schedule(0); err != nil {
		t.Fatal(err)
	}
	out := a.Outbox()
	if len(out) != 1 || out[0].From != "blis-node-1:5711" {
		t.Fatalf("the engine's connect is %+v, want one from blis-node-1:5711", out)
	}
	// The connect names the identity the peer dials back: the tier's own, which must be the new one.
	if m, err := decodeP2PMsg(out[0].Payload); err != nil || m.PeerID != "blis-node-1:5711" {
		t.Fatalf("the connect names the tier as %q (%v), want blis-node-1:5711", m.PeerID, err)
	}
	connect := encodeP2PMsg(kvtier.P2PMsg{Type: "connect", PeerID: prefillerAddr, AgentMetadata: []byte("agent"),
		BlockLen: 4096, NumBlocks: 200})
	msg := engine.PeerMessage{From: prefillerAddr, To: "localhost:5711", Seq: 1, Bytes: int64(len(connect)), Payload: connect}
	if err := a.Deliver(msg, 10); err == nil {
		t.Error("a message to the block's address was taken after the engine was given its own")
	}
	msg.To = "blis-node-1:5711"
	if err := a.Deliver(msg, 10); err != nil {
		t.Errorf("a message to the engine's own address was refused: %v", err)
	}
	if err := a.SetPeerAddr("blis-node-1:5799"); err == nil {
		t.Error("the address changed after the tier had a session")
	}
	for _, bad := range []string{"blis-node-0", ":5711", "blis-node-0:0", "blis-node-0:70000", "blis-node-0:x"} {
		if err := b.SetPeerAddr(bad); err == nil {
			t.Errorf("SetPeerAddr(%q) was accepted", bad)
		}
	}
	if err := mustEngine(t, fsTier).SetPeerAddr("blis-node-0:5711"); err == nil {
		t.Error("an engine with no p2p tier took a peer address")
	}
}
