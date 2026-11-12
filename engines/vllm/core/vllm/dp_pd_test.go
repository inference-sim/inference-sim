package vllm

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// A vLLM data-parallel engine runs one EngineCore per rank, each with its own scheduler, KV cache and NIXL
// connector (vllm/v1/engine/core.py:1417-1426, nixl/base_scheduler.py:76-80@5df63a72). A decode rank pulls the
// whole KV of one request from the prefill rank that computed it. These tests drive a data-parallel group as a
// P/D pool through the engine contract and assert what the host sees.

const dpAsync = "async_scheduling: true\n"

// dpPool is the pool of a group of dp ranks, one rank per GPU, on nodes of perNode GPUs.
func dpPool(dp, perNode, nodes int) string {
	return fmt.Sprintf("data_parallel_size: %d\nplacement: {hardware: h100, fabric: ib-400g, tensor: 1, data: %d, nodes: %d, ranks_per_node: %d, node_racks: [%s]}\n",
		dp, dp, nodes, perNode, strings.TrimSuffix(strings.Repeat("0, ", nodes), ", "))
}

func pullOf(id string, prompt, output, node int) engine.Request {
	r := req(id, prompt, output)
	r.RemotePrefill, r.Peer = true, &plan.Endpoint{Node: node}
	return r
}

// A decode group passes the contract's laws: with ranks on different nodes, every pull is a transfer of the
// request's whole prompt from the peer the request named, to the node of the rank the balancer chose; no two
// ranks state the same transfer ID; a pull that fails ends only its request.
func TestADataParallelDecodeGroupPassesTheContract(t *testing.T) {
	for _, c := range []struct {
		name  string
		extra string
		dp    int
	}{
		{"dp2 rank per node", consumer + dpPool(2, 1, 2), 2},
		{"dp4 rank per node", consumer + dpPool(4, 1, 4), 4},
		{"dp4 one node", consumer + dpPool(4, 8, 1), 4},
		{"dp2 batch queue", consumer + dpPool(2, 1, 2) + dpAsync, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			var arr []enginetest.Arrival
			for i := 0; i < 3*c.dp; i++ {
				arr = append(arr, enginetest.Arrival{At: engine.Time(i * 7), Req: pullOf(fmt.Sprintf("r%02d", i), 40+i, 6, 0)})
			}
			// One request's pull fails: under the default policy that request ends in error and nothing else does.
			failing := "r01"
			ids := map[engine.TransferID]string{}
			loads := map[string]int{}
			nodes := map[int]bool{}
			var started []engine.Transfer
			tr := enginetest.TestEngine(t, build(t, c.extra), arr, enginetest.Options{
				TransferTime:  func(x engine.Transfer) engine.Duration { return engine.Duration(300 + 90*(x.ID%5)) },
				TransferFails: func(x engine.Transfer) bool { return x.Request == failing },
				Boundary: func(_ plan.PoolPlan, x engine.Transfer) error {
					if other, dup := ids[x.ID]; dup {
						if other != x.Request {
							return fmt.Errorf("transfer ID %d stated for %s and for %s", x.ID, other, x.Request)
						}
						return nil // the harness checks a transfer again as it ends
					}
					ids[x.ID] = x.Request
					loads[x.Request]++
					nodes[x.Move.To.Node] = true
					started = append(started, x)
					return nil
				},
			})
			for _, x := range started {
				if x.Kind != engine.TransferLoad || x.Move.Kind != plan.TransferPeer {
					t.Errorf("transfer %d of %s is %s/%s, want a load from a peer", x.ID, x.Request, x.Kind, x.Move.Kind)
				}
			}
			for _, a := range arr {
				if loads[a.Req.ID] != 1 {
					t.Errorf("%s started %d pulls, want 1", a.Req.ID, loads[a.Req.ID])
				}
				f, ok := tr.Finished[a.Req.ID]
				switch {
				case !ok:
					t.Errorf("%s never finished", a.Req.ID)
				case a.Req.ID == failing && f.Reason != engine.FinishError:
					t.Errorf("%s: failed pull finished %q, want error", a.Req.ID, f.Reason)
				case a.Req.ID != failing && (f.Reason == engine.FinishError || f.OutputTokens != 6):
					t.Errorf("%s finished %+v, want its 6 tokens", a.Req.ID, f)
				}
			}
			if c.dp > 1 && strings.Contains(c.extra, "ranks_per_node: 1") && len(nodes) < 2 {
				t.Errorf("every pull landed on node set %v: the balancer should spread requests over ranks on different nodes", nodes)
			}
		})
	}
}

// host drives engines on one clock as the simulator does: it completes each batch after its step time, reports a
// transfer done after a delay, and wakes an engine at NextWake.
type host struct {
	t      *testing.T
	engs   []*hostEngine
	now    engine.Time
	onDone func(h *hostEngine, x engine.Transfer)
}

type hostEngine struct {
	e        engine.Engine
	delay    func(engine.Transfer) engine.Duration
	inflight []flight
	due      []dueTransfer
	started  []engine.Transfer
	handoffs []engine.Handoff
	finished map[string]engine.Finished
	emitted  map[string]int
	onResult func(engine.Result)
}

type flight struct {
	b  engine.Batch
	at engine.Time
}

type dueTransfer struct {
	at engine.Time
	x  engine.Transfer
}

func newHostEngine(t *testing.T, extra string, delay engine.Duration) *hostEngine {
	e, err := build(t, extra)()
	if err != nil {
		t.Fatal(err)
	}
	return &hostEngine{e: e, delay: func(engine.Transfer) engine.Duration { return delay }, finished: map[string]engine.Finished{},
		emitted: map[string]int{}}
}

func (h *host) tick(he *hostEngine) {
	t := h.t
	t.Helper()
	sort.SliceStable(he.due, func(i, j int) bool { return he.due[i].at < he.due[j].at })
	for len(he.due) > 0 && he.due[0].at <= h.now {
		d := he.due[0]
		he.due = he.due[1:]
		if err := he.e.TransferDone(d.x.ID, true, h.now); err != nil {
			t.Fatalf("TransferDone(%d) of %s: %v", d.x.ID, d.x.Request, err)
		}
		if h.onDone != nil {
			h.onDone(he, d.x)
		}
	}
	for len(he.inflight) > 0 && he.inflight[0].at <= h.now {
		f := he.inflight[0]
		he.inflight = he.inflight[1:]
		res, err := he.e.Complete(f.b.ID, engine.Outcome{}, h.now)
		if err != nil {
			t.Fatal(err)
		}
		for id, n := range res.Emitted {
			he.emitted[id] += n
		}
		for _, f := range res.Finished {
			he.finished[f.ID] = f
		}
		he.handoffs = append(he.handoffs, res.Handoffs...)
		if he.onResult != nil {
			he.onResult(res)
		}
	}
	for {
		b, err := he.e.Schedule(h.now)
		if errors.Is(err, engine.ErrNoCapacity) {
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range b.Finished {
			he.finished[f.ID] = f
		}
		for _, x := range b.TransfersStarted {
			he.started = append(he.started, x)
			he.due = append(he.due, dueTransfer{h.now + he.delay(x), x})
		}
		if b.Empty() {
			return
		}
		he.inflight = append(he.inflight, flight{b, h.now + max(he.e.StepTime(b), 1)})
		if len(he.inflight) >= he.e.InFlightLimit() {
			return
		}
	}
}

// run advances the clock until every engine is idle with nothing due, or fails after a bound.
func (h *host) run(until func() bool) {
	h.t.Helper()
	for step := 0; step < 100000; step++ {
		for _, he := range h.engs {
			h.tick(he)
		}
		if until() {
			return
		}
		next := engine.Time(-1)
		consider := func(at engine.Time) {
			if at > h.now && (next < 0 || at < next) {
				next = at
			}
		}
		for _, he := range h.engs {
			for _, d := range he.due {
				consider(d.at)
			}
			for _, f := range he.inflight {
				consider(f.at)
			}
			if at, ok := he.e.NextWake(h.now); ok {
				consider(at)
			}
		}
		if next < 0 {
			return
		}
		h.now = next
	}
	h.t.Fatal("no progress")
}

// A prefill group keeps each finished request's KV on the rank that computed it. Its handoffs carry transfer IDs
// and block IDs that no two ranks share, and ending one hold frees that rank's blocks and no other's.
func TestADataParallelPrefillGroupHoldsEachRequestOnItsRank(t *testing.T) {
	pe := newHostEngine(t, producer+dpPool(2, 8, 1), 0)
	h := &host{t: t, engs: []*hostEngine{pe}}
	g := pe.e.(*DPGroup)
	for i := 0; i < 4; i++ {
		// Distinct prompts: a prefix cache shares blocks between requests of one rank, which would
		// appear in two handoffs legitimately.
		r := req(strings.Repeat("p", i+1), 80, 6)
		r.RemoteDecode, r.MaxOutputTokens = true, 1
		if err := g.Add(r, 0); err != nil {
			t.Fatal(err)
		}
	}
	h.run(func() bool { return len(pe.handoffs) == 4 })
	if len(pe.handoffs) != 4 {
		t.Fatalf("%d handoffs, want 4", len(pe.handoffs))
	}
	ids := map[engine.TransferID]bool{}
	blocks := map[int64]string{}
	for _, ho := range pe.handoffs {
		if ids[ho.Transfer] {
			t.Errorf("transfer ID %d is two handoffs", ho.Transfer)
		}
		ids[ho.Transfer] = true
		for _, b := range ho.Blocks {
			if other, dup := blocks[b]; dup {
				t.Errorf("block %d is held for %s and for %s: ranks must not report the same block ID", b, other, ho.ID)
			}
			blocks[b] = ho.ID
		}
	}
	// Ending one hold frees that request's blocks and no others': the group's used blocks fall by exactly
	// the blocks that handoff named, at the next step.
	ho := pe.handoffs[0]
	before := g.Snapshot().KVBlocksUsed
	if err := g.TransferDone(ho.Transfer, true, h.now); err != nil {
		t.Fatal(err)
	}
	h.run(func() bool { return g.Snapshot().KVBlocksUsed < before })
	if freed := before - g.Snapshot().KVBlocksUsed; int(freed) != len(ho.Blocks) {
		t.Errorf("ending the hold of %s freed %d blocks, want the %d it held", ho.ID, freed, len(ho.Blocks))
	}
	if err := g.TransferDone(ho.Transfer, true, h.now); !errors.Is(err, engine.ErrUnknownTransfer) {
		t.Errorf("TransferDone of an ended hold = %v, want ErrUnknownTransfer", err)
	}
	for _, bad := range []engine.TransferID{0, 1 << 40} {
		if err := g.TransferDone(bad, true, h.now); !errors.Is(err, engine.ErrUnknownTransfer) {
			t.Errorf("TransferDone(%d) = %v, want ErrUnknownTransfer", bad, err)
		}
	}
}

// A prefill group and a decode group, wired as the simulator wires them: the decode leg of a request reaches the
// decode group when the prefill leg finishes, naming the prefill's node, and the prefill's hold ends when the pull
// lands. Every request completes; each pull is stated from the named node to the node of the decode rank that
// serves it; a rank whose pull is still in flight does not hold up the other rank's decode.
func TestADataParallelPrefillAndDecodeGroupsHandOff(t *testing.T) {
	for _, async := range []string{"", dpAsync} {
		t.Run(fmt.Sprintf("batch queue %t", async != ""), func(t *testing.T) {
			pe := newHostEngine(t, producer+dpPool(2, 8, 2)+async, 0)
			de := newHostEngine(t, consumer+dpPool(2, 1, 2)+async, 0)
			de.delay = func(x engine.Transfer) engine.Duration { return engine.Duration(200 + 700*(x.ID%3)) }
			if err := engine.Place(pe.e, plan.Endpoint{Node: 1}, 2); err != nil {
				t.Fatal(err)
			}
			prefillAt := plan.Endpoint{Node: 1}
			h := &host{t: t, engs: []*hostEngine{pe, de}}
			legs := map[string]engine.Request{}
			released := map[engine.TransferID]bool{}
			for i := 0; i < 4; i++ {
				id := fmt.Sprintf("d%d", i)
				legs[id] = req(id, 64+8*i, 5)
				r := legs[id]
				r.RemoteDecode, r.MaxOutputTokens = true, 1
				if err := pe.e.Add(r, 0); err != nil {
					t.Fatal(err)
				}
			}
			pe.onResult = func(res engine.Result) {
				for _, ho := range res.Handoffs {
					r := legs[ho.ID]
					r.RemotePrefill, r.Peer = true, &prefillAt
					if err := de.e.Add(r, h.now); err != nil {
						t.Fatal(err)
					}
				}
			}
			holds := map[string]engine.TransferID{}
			pe.onResult = wrap(pe.onResult, func(res engine.Result) {
				for _, ho := range res.Handoffs {
					holds[ho.ID] = ho.Transfer
				}
			})
			h.onDone = func(he *hostEngine, x engine.Transfer) {
				if he != de {
					return
				}
				id := holds[x.Request]
				if err := pe.e.TransferDone(id, true, h.now); err != nil {
					t.Fatalf("ending the prefill's hold for %s: %v", x.Request, err)
				}
				released[id] = true
			}
			h.run(func() bool { return len(de.finished) == 4 })
			for id := range legs {
				f, ok := de.finished[id]
				if !ok || f.OutputTokens != 5 || f.Reason == engine.FinishError {
					t.Errorf("%s finished %+v (%t), want its 5 tokens", id, f, ok)
				}
			}
			if len(released) != 4 {
				t.Errorf("%d prefill holds ended, want 4", len(released))
			}
			toNode := map[string]int{}
			for _, x := range de.started {
				if x.Move.From != prefillAt {
					t.Errorf("pull of %s reads from %+v, want the prefill's %+v", x.Request, x.Move.From, prefillAt)
				}
				if int(x.Move.Tokens) != len(legs[x.Request].Prompt) {
					t.Errorf("pull of %s moves %d tokens, want the prompt's %d", x.Request, x.Move.Tokens, len(legs[x.Request].Prompt))
				}
				toNode[x.Request] = x.Move.To.Node
			}
			if len(unique(toNode)) < 2 {
				t.Errorf("pulls ended at nodes %v: ranks on different nodes must state their own", toNode)
			}
		})
	}
}

func wrap(a, b func(engine.Result)) func(engine.Result) {
	return func(r engine.Result) {
		if a != nil {
			a(r)
		}
		b(r)
	}
}

func unique(m map[string]int) map[int]bool {
	out := map[int]bool{}
	for _, v := range m {
		out[v] = true
	}
	return out
}

// While a decode rank waits for its pull, the group keeps stepping: the other rank decodes its own request to the
// end before the slow pull lands, and the waiting request then runs and finishes.
func TestADecodeRankWaitingForItsPullDoesNotStopTheOtherRank(t *testing.T) {
	de := newHostEngine(t, consumer+dpPool(2, 1, 2), 0)
	de.delay = func(x engine.Transfer) engine.Duration {
		if x.Request == "slow" {
			return 20000
		}
		return 100
	}
	h := &host{t: t, engs: []*hostEngine{de}}
	g := de.e.(*DPGroup)
	for _, id := range []string{"slow", "fast"} {
		if err := g.Add(pullOf(id, 48, 8, 0), 0); err != nil {
			t.Fatal(err)
		}
	}
	var fastDone engine.Time
	de.onResult = func(res engine.Result) {
		for _, f := range res.Finished {
			if f.ID == "fast" {
				fastDone = h.now
			}
		}
	}
	h.run(func() bool { return len(de.finished) == 2 })
	if len(de.finished) != 2 || de.finished["slow"].OutputTokens != 8 || de.finished["fast"].OutputTokens != 8 {
		t.Fatalf("finished %+v, want both requests with 8 tokens", de.finished)
	}
	if fastDone == 0 || fastDone >= 20000 {
		t.Errorf("the other rank's request finished at %d: it should not wait for the slow pull (lands at 20000)", fastDone)
	}
}

// A prefill group whose ranks span nodes is refused: a handoff names no endpoint, so every pull would be priced from
// the group's first node. A group on one node builds, and so does a decode group across nodes, whose ranks each state
// their own node as the end of their pulls.
func TestAPrefillGroupAcrossNodesIsRefusedAndADecodeGroupIsNot(t *testing.T) {
	for _, c := range []struct {
		name, extra string
		refused     bool
	}{
		{"kv_producer across nodes", producer + dpPool(2, 1, 2), true},
		{"kv_both across nodes", "kv_transfer: {connector: NixlConnector, kv_role: kv_both}\n" + dpPool(2, 1, 2), true},
		{"kv_producer on one node", producer + dpPool(8, 8, 2), false},
		{"kv_consumer across nodes", consumer + dpPool(2, 1, 2), false},
	} {
		_, err := build(t, c.extra)()
		if c.refused != (err != nil) {
			t.Errorf("%s: err = %v, want refused %t", c.name, err, c.refused)
		}
		if c.refused && err != nil && !strings.Contains(err.Error(), "substrate#48") {
			t.Errorf("%s: %q does not name the contract limit", c.name, err)
		}
	}
}

func TestDataParallelOffloadStaysRefusedByName(t *testing.T) {
	if _, err := resolve(mustConfig(t, cpuTier+"data_parallel_size: 2\n")); err == nil || !strings.Contains(err.Error(), "blis-vllm/issues/47") {
		t.Errorf("kv_offload at dp 2: %v, want a refusal naming blis-vllm/issues/47", err)
	}
}

// A pool of several nodes holds several groups, one a pod; the host gives each its first node. A second group is
// placed on the second node and its pulls end there; a group that would run past the pool's last node is refused.
func TestSeveralDataParallelGroupsShareAPool(t *testing.T) {
	pool := dpPool(8, 8, 2)
	first := newHostEngine(t, consumer+pool, 10)
	second := newHostEngine(t, consumer+pool, 10)
	if err := engine.Place(first.e, plan.Endpoint{Node: 0}, 2); err != nil {
		t.Fatal(err)
	}
	if err := engine.Place(second.e, plan.Endpoint{Node: 1, Rack: 1}, 2); err != nil {
		t.Fatalf("the second group at node 1 of a two-node pool: %v", err)
	}
	h := &host{t: t, engs: []*hostEngine{first, second}}
	for _, he := range h.engs {
		for i := 0; i < 4; i++ {
			if err := he.e.Add(pullOf(fmt.Sprintf("x%d", i), 32, 3, 0), 0); err != nil {
				t.Fatal(err)
			}
		}
	}
	h.run(func() bool { return len(first.finished) == 4 && len(second.finished) == 4 })
	for name, c := range map[string]struct {
		he   *hostEngine
		want plan.Endpoint
	}{"first": {first, plan.Endpoint{Node: 0}}, "second": {second, plan.Endpoint{Node: 1, Rack: 1}}} {
		if len(c.he.started) != 4 {
			t.Fatalf("%s group started %d pulls, want 4", name, len(c.he.started))
		}
		for _, x := range c.he.started {
			if x.Move.To != c.want {
				t.Errorf("%s group: pull of %s ends at %+v, want %+v", name, x.Request, x.Move.To, c.want)
			}
		}
	}
	third := newHostEngine(t, consumer+pool, 10)
	if err := engine.Place(third.e, plan.Endpoint{Node: 2}, 3); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Errorf("a group past the pool's last node: %v, want ErrPlacementRejected", err)
	}
	// A group of 16 ranks fills two nodes and does not fit at the second.
	wide := newHostEngine(t, consumer+dpPool(16, 8, 2), 10)
	if err := engine.Place(wide.e, plan.Endpoint{Node: 1}, 2); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Errorf("a two-node group at node 1 of two: %v, want ErrPlacementRejected", err)
	}
	if err := engine.Place(wide.e, plan.Endpoint{Node: 0}, 2); err != nil {
		t.Errorf("a two-node group at node 0: %v", err)
	}
}

// A pull that fails under the default policy ends its request in error even when the request's rank has nothing
// else to run: the rank's step runs no tokens, and the finish rides the schedule that applied the failure.
func TestAFailedPullOnAnOtherwiseIdleRankEndsItsRequest(t *testing.T) {
	for _, extra := range []string{consumer + dpPool(2, 8, 1), consumer + dpPool(2, 8, 1) + dpAsync} {
		tr := enginetest.TestEngine(t, build(t, extra), []enginetest.Arrival{{At: 0, Req: pullOf("only", 40, 4, 0)}},
			enginetest.Options{TransferFails: func(engine.Transfer) bool { return true }})
		if f, ok := tr.Finished["only"]; !ok || f.Reason != engine.FinishError {
			t.Errorf("a failed pull finished %+v (%t), want an error finish", f, ok)
		}
	}
}

// The team's model shape: DeepSeek V4.1's KV cache groups (a bounded-replay sliding-window MLA group, an MLA group
// of its own block size, a ring) in a TP1 x DP group, as a decode pod and as a prefill pod, pass the engine contract
// beside the connector. Each pull reports the blocks its rank's groups hold.
func TestADataParallelPodWithDeepSeekV41LayersPassesTheContract(t *testing.T) {
	const shape = "block_size: 256\nnum_gpu_blocks: 400\nmax_model_len: 1024\nmax_num_batched_tokens: 256\n" + dsv41Layers
	for _, c := range []struct {
		name, extra string
		mark        func(*engine.Request)
	}{
		{"decode pod", consumer + shape + dpPool(4, 8, 1), func(r *engine.Request) { r.RemotePrefill, r.Peer = true, peer }},
		{"prefill pod", producer + shape + dpPool(4, 8, 1), func(r *engine.Request) { r.RemoteDecode, r.MaxOutputTokens = true, 1 }},
		{"decode pod with a batch queue", consumer + shape + dpPool(2, 8, 1) + dpAsync,
			func(r *engine.Request) { r.RemotePrefill, r.Peer = true, peer }},
	} {
		t.Run(c.name, func(t *testing.T) {
			var reqs []engine.Request
			for i := 0; i < 8; i++ {
				r := req(strings.Repeat("v", i+1), 90+80*(i%5), 4)
				c.mark(&r)
				reqs = append(reqs, r)
			}
			var pulls int
			enginetest.TestEngine(t, build(t, c.extra), arrivals(reqs...), enginetest.Options{
				Boundary: func(p plan.PoolPlan, x engine.Transfer) error {
					pulls++
					if x.Blocks <= 0 {
						return fmt.Errorf("a pull of %s reports %d blocks", x.Request, x.Blocks)
					}
					return boundary(p, x)
				},
				TransferTime: func(engine.Transfer) engine.Duration { return 250 },
			})
			if c.name != "prefill pod" && pulls == 0 {
				t.Errorf("no pull started")
			}
		})
	}
}
