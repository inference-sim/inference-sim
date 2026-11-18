package plan

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/units"
)

func twoNodePeer() Transfer {
	a, b := Endpoint{Node: 0}, Endpoint{Node: 1}
	return Transfer{Tokens: 512, Kind: TransferPeer, From: a, To: a,
		Nodes: &NodeMoves{[]NodeMove{{From: a, To: a}, {From: b, To: b}}}}
}

// A move that every node of a pool makes names each node's endpoints; the single From and To are
// the first entry, and a report that disagrees with it is refused.
func TestNodeMovesAgreeWithFromAndTo(t *testing.T) {
	good := twoNodePeer()
	if err := good.Validate(); err != nil {
		t.Fatalf("a two-node pull: %v", err)
	}
	bad := good
	bad.From = Endpoint{Node: 1}
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "first node move") {
		t.Errorf("From that is not the first node move was not refused by name: %v", err)
	}
	neg := good
	neg.Nodes = &NodeMoves{[]NodeMove{good.NodeList()[0], {From: Endpoint{Node: -1}, To: Endpoint{Node: 1}}}}
	if err := neg.Validate(); err == nil {
		t.Error("a negative node endpoint in an entry was accepted")
	}
	off := Transfer{Tokens: 8, Kind: TransferToTier, Tier: "host", From: Endpoint{}, To: Endpoint{},
		Nodes: &NodeMoves{[]NodeMove{{}, {From: Endpoint{Node: 1}, To: Endpoint{Node: 0}}}}}
	if err := off.Validate(); err == nil || !strings.Contains(err.Error(), "offload move") {
		t.Errorf("an offload entry that leaves one node and lands on another was accepted: %v", err)
	}
}

// An offload move that every node makes is checked against the pool's nodes entry by entry.
func TestEveryNodeMoveOfAnOffloadIsInThePool(t *testing.T) {
	p := PoolPlan{Layout: Layout{Nodes: 2}, Tiers: []KVTier{{Device: "host"}}}
	e0, e1, e2 := Endpoint{Node: 0}, Endpoint{Node: 1}, Endpoint{Node: 2}
	ok := Transfer{Tokens: 8, Kind: TransferToTier, Tier: "host", From: e0, To: e0,
		Nodes: &NodeMoves{[]NodeMove{{From: e0, To: e0}, {From: e1, To: e1}}}}
	if err := p.CheckTransfer(ok); err != nil {
		t.Fatalf("both nodes of a two-node pool: %v", err)
	}
	ok.Nodes = &NodeMoves{append(append([]NodeMove(nil), ok.NodeList()...), NodeMove{From: e2, To: e2})}
	if err := p.CheckTransfer(ok); err == nil {
		t.Error("a node move on a third node of a two-node pool was accepted")
	}
}

// The tokens each group moves are bounded by the move's Tokens, named once, and must be groups
// of the plan; the bytes round each group up to its pages.
func TestGroupMovesAreCheckedAndSized(t *testing.T) {
	p := PoolPlan{KV: []KVGroup{
		{BytesPerPage: 1000, TokensPerPage: 16},            // full attention
		{BytesPerPage: 400, TokensPerPage: 16, Window: 64}, // sliding window
	}}
	m := Transfer{Tokens: 100, Kind: TransferPeer, Groups: &GroupMoves{[]GroupMove{{Group: 0, Tokens: 100}, {Group: 1, Tokens: 64}}}}
	if err := p.CheckTransfer(m); err != nil {
		t.Fatalf("a two-group move: %v", err)
	}
	got, ok := p.GroupMoveBytes(m)
	// 100 tokens = 7 pages of 1000; 64 tokens = 4 pages of 400.
	if !ok || got != units.Bytes(7*1000+4*400) {
		t.Errorf("GroupMoveBytes = %v, %v; want %d", got, ok, 7*1000+4*400)
	}
	if _, ok := p.GroupMoveBytes(Transfer{Tokens: 100, Kind: TransferPeer}); ok {
		t.Error("a move that states no groups reported bytes of its own")
	}
	for name, g := range map[string][]GroupMove{
		"a group above the move's tokens": {{Group: 0, Tokens: 101}},
		"a group named twice":             {{Group: 0, Tokens: 1}, {Group: 0, Tokens: 2}},
		"a negative group":                {{Group: -1, Tokens: 1}},
	} {
		if err := (Transfer{Tokens: 100, Kind: TransferPeer, Groups: &GroupMoves{g}}).Validate(); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
	if err := p.CheckTransfer(Transfer{Tokens: 100, Kind: TransferPeer, Groups: &GroupMoves{[]GroupMove{{Group: 2, Tokens: 1}}}}); err == nil {
		t.Error("a group the plan does not have was accepted")
	}
}

// Two transfers are the same move when their fields and the contents of their lists agree. == on
// a Transfer compares the list pointers, so Equal is how a consumer compares. A nil list and an
// empty one state the same.
func TestEqualComparesTheListsContents(t *testing.T) {
	a, b := twoNodePeer(), twoNodePeer()
	if a == b {
		t.Fatal("two separately built lists share a pointer; the test cannot tell == from Equal")
	}
	if !a.Equal(b) {
		t.Error("two transfers with the same node moves are not Equal")
	}
	b.Nodes = &NodeMoves{[]NodeMove{b.NodeList()[0], {From: Endpoint{Node: 1}, To: Endpoint{Node: 1, Rack: 1}}}}
	if a.Equal(b) {
		t.Error("transfers with different node moves are Equal")
	}
	plain := Transfer{Tokens: 8, Kind: TransferPeer}
	empty := plain
	empty.Nodes, empty.Groups = &NodeMoves{}, &GroupMoves{}
	if !plain.Equal(empty) {
		t.Error("a nil list and an empty one differ")
	}
	g1 := Transfer{Tokens: 8, Kind: TransferPeer, Groups: &GroupMoves{[]GroupMove{{Group: 0, Tokens: 8}}}}
	g2 := Transfer{Tokens: 8, Kind: TransferPeer, Groups: &GroupMoves{[]GroupMove{{Group: 0, Tokens: 4}}}}
	if g1.Equal(g2) || !g1.Equal(g1) {
		t.Error("Equal does not tell group lists apart")
	}
	if plain != (Transfer{Tokens: 8, Kind: TransferPeer}) {
		t.Error("a Transfer that states no lists is no longer comparable with ==")
	}
}
