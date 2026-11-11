// Original to blis-vllm.

package vllm

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/conformance/envtest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"
)

// boundary is the simulator's side of the transfer boundary: every transfer an engine starts goes through
// env.DemandFor against a reference Environment, as a simulator does for each entry of TransfersStarted.
var boundary = envtest.Boundary(&envtest.Reference{BytesPerSecond: 1e9, BytesPerToken: 1000})

// An engine that can start a transfer is Planned: its plan names the pool's chip, network and layout, is the
// same on every call and shares no memory with the caller, lists cpu_dram when kv_offload is configured
// (per rank: the CPU bytes over the workers, which each hold their share), and lists no tier for a P/D pool.
func TestThePlanStatesThePlacementAndTheCPUTier(t *testing.T) {
	var _ engine.Planned = (*Engine)(nil)
	e := mustEngine(t, "kv_offload: {connector: OffloadingConnector, cpu_bytes: 3276800, bytes_per_block: 4096, workers: 4}\n")
	p := e.Plan()
	if err := p.Validate(); err != nil {
		t.Fatalf("the plan does not validate: %v", err)
	}
	if p.Hardware != "h100" || p.Fabric != "ib-400g" || p.Layout.Nodes != 2 || p.Layout.RanksPerNode != 8 {
		t.Errorf("plan %+v, want the placement's chip, network and layout", p)
	}
	want := []plan.KVTier{{Device: "cpu_dram", Bytes: units.Rank(units.Bytes(3276800 / 4))}}
	if !reflect.DeepEqual(p.Tiers, want) {
		t.Errorf("tiers %+v, want %+v", p.Tiers, want)
	}
	p.Tiers[0].Device = "mutated"
	if got := e.Plan(); !reflect.DeepEqual(got, e.Plan()) || got.Tiers[0].Device != "cpu_dram" {
		t.Errorf("the plan changed with the caller's copy or between calls: %+v", got)
	}
	if got := mustEngine(t, consumer).Plan().Tiers; len(got) != 0 {
		t.Errorf("a P/D pool lists tiers %+v; it has no KV tier below GPU memory", got)
	}
	if got := mustEngine(t, "").Plan(); !reflect.DeepEqual(got, plan.PoolPlan{}) {
		t.Errorf("a pool that states no placement and starts no transfer has plan %+v, want the empty plan", got)
	}
}

// An offload transfer states its move as the cost model prices it: the catalog device the tier name stands
// for (cpu_dram), a load from it or a store to it, the tokens it reports, and the node that holds the tier at
// both ends. Mutation check: taking the node from the placement's first node instead of its node field, or
// leaving Move out, fails it.
func TestAnOffloadTransferStatesItsMove(t *testing.T) {
	extra := cpuTier + tightGPU + "placement: {hardware: h100, fabric: ib-400g, nodes: 2, ranks_per_node: 8, node: 1, node_racks: [0, 3]}\n"
	var arr []enginetest.Arrival
	at := engine.Time(0)
	for i, id := range []string{"a", "b", "a2", "c", "b2", "d", "a3", "e"} {
		base, n := 100000*(i+1), 100+4*i
		switch id[0] {
		case 'a':
			base, n = 1000, 96
		case 'b':
			base, n = 2000, 64
		}
		r := req(id, 0, 3)
		for k := 0; k < n; k++ {
			r.Prompt = append(r.Prompt, int32(base+k))
		}
		r.Prompt = append(r.Prompt, int32(i))
		arr = append(arr, enginetest.Arrival{At: at, Req: r})
		at += 3000
	}
	seen := map[engine.TransferKind]int{}
	enginetest.TestEngine(t, build(t, extra), arr, enginetest.Options{
		Boundary: func(p plan.PoolPlan, tr engine.Transfer) error {
			seen[tr.Kind]++
			at := plan.Endpoint{Node: 1, Rack: 3}
			wantKind := plan.TransferToTier
			if tr.Kind == engine.TransferLoad {
				wantKind = plan.TransferFromTier
			}
			if tr.Tier != "cpu" || tr.Move.Tier != "cpu_dram" || tr.Move.Kind != wantKind || tr.Move.From != at || tr.Move.To != at ||
				tr.Move.Tokens != units.Tokens(tr.Tokens) {
				t.Errorf("transfer %+v: want a %s of cpu_dram on node 1 rack 3 for its %d tokens", tr, wantKind, tr.Tokens)
			}
			if err := p.CheckTransfer(tr.Move); err != nil {
				t.Errorf("the plan refuses the move: %v", err)
			}
			return boundary(p, tr)
		},
	})
	if seen[engine.TransferLoad] == 0 || seen[engine.TransferStore] == 0 {
		t.Errorf("transfers %v: the CPU tier was not exercised both ways", seen)
	}
}

// A pull states its move as a transfer from the peer the request named to this engine's node, using no tier.
// Mutation check: reading the peer from the engine's own node fails it.
func TestAPullStatesItsMoveFromThePeer(t *testing.T) {
	d := mustEngine(t, consumer+"placement: {hardware: h100, fabric: ib-400g, nodes: 4, ranks_per_node: 8, node: 2, node_racks: [0, 0, 1, 1]}\n")
	r := req("a", 80, 6)
	r.RemotePrefill, r.Peer = true, &plan.Endpoint{Node: 0, Rack: 0}
	if err := d.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	_, started, _ := runOne(t, d, 500)
	if len(started) != 1 {
		t.Fatalf("started %+v, want one pull", started)
	}
	tr := started[0]
	want := plan.Transfer{Tokens: units.Tokens(tr.Tokens), From: plan.Endpoint{Node: 0, Rack: 0}, To: plan.Endpoint{Node: 2, Rack: 1},
		Kind: plan.TransferPeer}
	if tr.Tier != "" || !reflect.DeepEqual(tr.Move, want) {
		t.Errorf("pull %+v, want Move %+v and no tier", tr, want)
	}
	if err := tr.Validate(); err != nil {
		t.Error(err)
	}
	if err := d.Plan().CheckTransfer(tr.Move); err != nil {
		t.Error(err)
	}
}

// A request the contract says cannot exist is refused by Add before the engine sees it: a remote prefill
// with no peer to pull from, and a peer on a request that pulls nothing.
func TestAddRefusesARemotePrefillWithNoPeerAndAPeerWithNoPull(t *testing.T) {
	d := mustEngine(t, consumer)
	r := req("a", 80, 6)
	r.RemotePrefill = true
	if err := d.Add(r, 0); err == nil || !strings.Contains(err.Error(), "no peer") {
		t.Errorf("a remote prefill with no peer: %v, want a refusal naming the peer", err)
	}
	r = req("b", 80, 6)
	r.Peer = peer
	if err := d.Add(r, 0); err == nil || !strings.Contains(err.Error(), "peer") {
		t.Errorf("a peer on a request that pulls nothing: %v, want a refusal", err)
	}
}

// A pool that can start a transfer must say where it sits: the transfers are priced against that placement.
func TestAPoolThatStartsTransfersMustStateItsPlacement(t *testing.T) {
	for name, extra := range map[string]string{"kv_transfer": producer, "kv_offload": cpuTier} {
		_, err := build(t, extra+"placement: null\n")()
		if err == nil || !strings.Contains(err.Error(), "placement") {
			t.Errorf("%s with no placement: %v, want a refusal naming the placement", name, err)
		}
	}
	if _, err := build(t, producer+"placement: {hardware: h100, nodes: 2, node: 2}\n")(); err == nil || !strings.Contains(err.Error(), "node 2") {
		t.Errorf("an engine on a node the pool lacks: %v, want a refusal", err)
	}
}

// The ranks of a data-parallel group sit on the nodes their workers fill, in order: with one rank to a node,
// rank 1 is on node 1, and the group's plan is the pool's.
func TestTheRanksOfAGroupSitOnTheirOwnNodes(t *testing.T) {
	g, err := build(t, "data_parallel_size: 2\nplacement: {hardware: h100, fabric: ib-400g, data: 2, nodes: 2, ranks_per_node: 1, node_racks: [0, 1]}\n")()
	if err != nil {
		t.Fatal(err)
	}
	dp := g.(*DPGroup)
	if dp.ranks[0].e.at != (plan.Endpoint{Node: 0, Rack: 0}) || dp.ranks[1].e.at != (plan.Endpoint{Node: 1, Rack: 1}) {
		t.Errorf("rank endpoints %+v and %+v, want node 0 rack 0 and node 1 rack 1", dp.ranks[0].e.at, dp.ranks[1].e.at)
	}
	if p := dp.Plan(); p.Layout.Data != 2 || p.Hardware != "h100" {
		t.Errorf("the group's plan %+v, want the pool's layout and chip", p)
	}
}
