// Original to blis-vllm.

package contract

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/core/placement"
)

func placedEngine(t *testing.T) *Engine {
	t.Helper()
	cfg := cengine.SimConfig{
		Horizon:             1 << 40,
		KVCacheConfig:       cengine.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
		BatchConfig:         cengine.NewBatchConfig(16, 2048, 0),
		LatencyModel:        fixedLatency{},
		ModelHardwareConfig: cengine.ModelHardwareConfig{GPU: "h100", TP: 2, DP: 1},
	}
	e, err := New("instance_0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// The compatibility engine is Planned: its plan is the one node of the configuration's GPU at its tensor
// width, which SetPlacement replaces, and it is the same value on every call.
func TestThePlanIsThePlacement(t *testing.T) {
	var _ engine.Planned = (*Engine)(nil)
	e := placedEngine(t)
	p := e.Plan()
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if p.Hardware != "h100" || p.Layout.Tensor != 2 || p.Layout.Nodes != 1 || len(p.Tiers) != 0 {
		t.Errorf("plan %+v, want one node of h100 at tensor 2 and no tier", p)
	}
	if err := e.SetPlacement(placement.Placement{Hardware: "b200", Fabric: "ib", Nodes: 2, RanksPerNode: 8, Node: 1, NodeRacks: []int{0, 4}}); err != nil {
		t.Fatal(err)
	}
	if got := e.Plan(); got.Hardware != "b200" || got.Fabric != "ib" || got.Layout.Nodes != 2 {
		t.Errorf("plan after SetPlacement %+v", got)
	}
	if err := e.SetPlacement(placement.Placement{Hardware: "b200", Nodes: 2, Node: 2}); err == nil {
		t.Error("a placement on a node the pool lacks was accepted")
	}
}

// A pull states its move from the peer the request named to the engine's own node, using no tier, and the
// engine's plan accepts it. Mutation check: reading the peer from the engine's node fails it.
func TestAPullStatesItsMoveFromThePeer(t *testing.T) {
	e := placedEngine(t)
	if err := e.SetPlacement(placement.Placement{Hardware: "h100", Fabric: "ib", Nodes: 2, RanksPerNode: 8, Node: 1, NodeRacks: []int{0, 4}}); err != nil {
		t.Fatal(err)
	}
	r := req("a", 80, 4)
	r.RemotePrefill, r.Peer = true, &plan.Endpoint{Node: 0, Rack: 0}
	if err := e.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	var started []engine.Transfer
	for now := engine.Time(0); len(started) == 0 && now < 100000; now += 1000 {
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		started = b.TransfersStarted
	}
	if len(started) != 1 {
		t.Fatalf("started %+v, want one pull", started)
	}
	tr := started[0]
	want := plan.Transfer{Tokens: 80, From: plan.Endpoint{Node: 0, Rack: 0}, To: plan.Endpoint{Node: 1, Rack: 4}, Kind: plan.TransferPeer}
	if tr.Tier != "" || tr.Move != want {
		t.Errorf("pull %+v, want Move %+v and no tier", tr, want)
	}
	if err := tr.Validate(); err != nil {
		t.Error(err)
	}
	if err := e.Plan().CheckTransfer(tr.Move); err != nil {
		t.Error(err)
	}
}

// Add refuses a remote prefill with no peer and a peer on a request that pulls nothing, for an engine and for
// a group's Add and AddToRank.
func TestAddRefusesWhatTheContractSaysCannotExist(t *testing.T) {
	e := placedEngine(t)
	noPeer := req("a", 80, 4)
	noPeer.RemotePrefill = true
	if err := e.Add(noPeer, 0); err == nil || !strings.Contains(err.Error(), "peer") {
		t.Errorf("a remote prefill with no peer: %v", err)
	}
	stray := req("b", 80, 4)
	stray.Peer = &plan.Endpoint{}
	if err := e.Add(stray, 0); err == nil || !strings.Contains(err.Error(), "peer") {
		t.Errorf("a peer on a request that pulls nothing: %v", err)
	}
	g, err := NewDPGroup([]string{"r0", "r1"}, cengine.SimConfig{
		Horizon: 1 << 40, KVCacheConfig: cengine.NewKVCacheConfig(64, 16, 0, 0, 0, 0), BatchConfig: cengine.NewBatchConfig(16, 2048, 0),
		LatencyModel: fixedLatency{}, ModelHardwareConfig: cengine.ModelHardwareConfig{GPU: "h100", TP: 1, DP: 2}})
	if err != nil {
		t.Skipf("a data-parallel group needs a configuration this fixture lacks: %v", err)
	}
	if err := g.Add(noPeer, 0); err == nil || !strings.Contains(err.Error(), "peer") {
		t.Errorf("the group's Add: %v", err)
	}
	if err := g.AddToRank(1, noPeer, 0); err == nil || !strings.Contains(err.Error(), "peer") {
		t.Errorf("the group's AddToRank: %v", err)
	}
	if p := g.Plan(); p.Layout.Data != 2 {
		t.Errorf("the group's plan %+v, want data width 2", p)
	}
}

// Rank k of a lockstep group is an engine of its own, and a Planned one: its plan is the group's, so a P/D pool that is
// a lockstep group has the plan its pulls are priced against. The group states its ranks' nodes from the pool's
// placement, and the host that builds one view for each instance states one instance's node.
func TestARankViewIsPlannedAndPlaced(t *testing.T) {
	g, err := NewDPGroup([]string{"r0", "r1"}, cengine.SimConfig{
		Horizon: 1 << 40, KVCacheConfig: cengine.NewKVCacheConfig(64, 16, 0, 0, 0, 0), BatchConfig: cengine.NewBatchConfig(16, 2048, 0),
		LatencyModel: fixedLatency{}, ModelHardwareConfig: cengine.ModelHardwareConfig{GPU: "h100", TP: 1, DP: 2}})
	if err != nil {
		t.Fatal(err)
	}
	var v engine.Planned = g.View(1)
	if got, want := v.Plan(), g.Plan(); !reflect.DeepEqual(got, want) || got.Validate() != nil {
		t.Errorf("a rank's plan %+v, want the group's %+v and valid", got, want)
	}
	pool := placement.Placement{Hardware: "h100", Fabric: "ib", Data: 2, Nodes: 2, RanksPerNode: 1, NodeRacks: []int{0, 3}}
	if err := g.SetPlacement(pool); err != nil {
		t.Fatal(err)
	}
	if a, b := g.ranks[0].pl.Endpoint(), g.ranks[1].pl.Endpoint(); a != (plan.Endpoint{Node: 0, Rack: 0}) || b != (plan.Endpoint{Node: 1, Rack: 3}) {
		t.Errorf("ranks at %+v and %+v, want node 0 rack 0 and node 1 rack 3", a, b)
	}
	one := pool
	one.Node = 0
	if err := g.View(1).SetPlacement(one); err != nil {
		t.Fatal(err)
	}
	if g.ranks[1].pl.Endpoint() != (plan.Endpoint{Node: 0, Rack: 0}) || g.ranks[0].pl.Endpoint() != (plan.Endpoint{Node: 0, Rack: 0}) {
		t.Errorf("a view's SetPlacement moved %+v / %+v; it moves only its own rank", g.ranks[0].pl.Endpoint(), g.ranks[1].pl.Endpoint())
	}
	bad := pool
	bad.Node = 2
	if err := g.View(0).SetPlacement(bad); err == nil {
		t.Error("a node the pool lacks was accepted")
	}
}

// The compatibility engine is placeable through the contract's one call: placed at a node and rack of its
// pool, a pull ends there, a node the pool lacks is refused with the contract's rejection, and a rank view
// moves its own rank only.
func TestTheCompatibilityEngineIsPlacedByTheHost(t *testing.T) {
	e := placedEngine(t)
	if err := e.SetPlacement(placement.Placement{Hardware: "h100", Fabric: "ib", Nodes: 4, RanksPerNode: 8}); err != nil {
		t.Fatal(err)
	}
	if err := engine.Place(e, plan.Endpoint{Node: 3, Rack: 2}, 4); err != nil {
		t.Fatal(err)
	}
	r := req("a", 80, 4)
	r.RemotePrefill, r.Peer = true, &plan.Endpoint{Node: 0, Rack: 0}
	if err := e.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	var started []engine.Transfer
	for now := engine.Time(0); len(started) == 0 && now < 100000; now += 1000 {
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		started = b.TransfersStarted
	}
	if len(started) != 1 || started[0].Move.To != (plan.Endpoint{Node: 3, Rack: 2}) {
		t.Fatalf("started %+v, want one pull ending at node 3 rack 2", started)
	}
	if err := engine.Place(e, plan.Endpoint{Node: 4}, 4); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Errorf("a node the pool lacks: %v, want ErrPlacementRejected", err)
	}

	g, err := NewDPGroup([]string{"r0", "r1"}, cengine.SimConfig{
		Horizon: 1 << 40, KVCacheConfig: cengine.NewKVCacheConfig(64, 16, 0, 0, 0, 0), BatchConfig: cengine.NewBatchConfig(16, 2048, 0),
		LatencyModel: fixedLatency{}, ModelHardwareConfig: cengine.ModelHardwareConfig{GPU: "h100", TP: 1, DP: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if err := g.SetPlacement(placement.Placement{Hardware: "h100", Fabric: "ib", Data: 2, Nodes: 2, RanksPerNode: 1, NodeRacks: []int{0, 3}}); err != nil {
		t.Fatal(err)
	}
	if err := engine.Place(g.View(1), plan.Endpoint{Node: 0, Rack: 5}, 2); err != nil {
		t.Fatal(err)
	}
	if a, b := g.ranks[0].pl.Endpoint(), g.ranks[1].pl.Endpoint(); a != (plan.Endpoint{Node: 0, Rack: 0}) || b != (plan.Endpoint{Node: 0, Rack: 5}) {
		t.Errorf("ranks at %+v and %+v, want rank 1 moved to node 0 rack 5 and rank 0 where it was", a, b)
	}
	if err := engine.Place(g, plan.Endpoint{Node: 1}, 2); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Errorf("a group whose ranks fill the nodes placed at node 1: %v, want ErrPlacementRejected", err)
	}
}
