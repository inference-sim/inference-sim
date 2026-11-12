package engine

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

func offloadLoad() Transfer {
	return Transfer{ID: 7, Request: "r1", Kind: TransferLoad, Tokens: 64, Blocks: 4, Tier: "cpu",
		Move: plan.Transfer{Tokens: 64, Kind: plan.TransferFromTier, Tier: "cpu_dram"}}
}

func offloadStore() Transfer {
	t := offloadLoad()
	t.Kind, t.Move.Kind = TransferStore, plan.TransferToTier
	return t
}

func peerPull() Transfer {
	return Transfer{ID: 8, Request: "r2", Kind: TransferLoad, Tokens: 32,
		Move: plan.Transfer{Tokens: 32, Kind: plan.TransferPeer, From: plan.Endpoint{Node: 3, Rack: 1}}}
}

// An engine reports a move once, as one Transfer, and states the catalog tier and the endpoints
// in it (design §6.2). A report that agrees with its Move is accepted whatever the direction.
func TestATransferThatAgreesWithItsMoveIsAccepted(t *testing.T) {
	for name, tr := range map[string]Transfer{"load from a tier": offloadLoad(), "store to a tier": offloadStore(), "pull from a peer": peerPull()} {
		if err := tr.Validate(); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	// A store to a peer is a handoff push; it carries no tier either.
	push := peerPull()
	push.Kind = TransferStore
	if err := push.Validate(); err != nil {
		t.Errorf("push to a peer: %v", err)
	}
}

// The two unconnected contracts the review found: a Transfer whose Move was never set, or was set
// to something else, must not pass. Each case breaks one agreement.
func TestATransferThatDisagreesWithItsMoveIsRefused(t *testing.T) {
	for name, c := range map[string]struct {
		mutate func(*Transfer)
		want   string
	}{
		"no Move at all":          {func(t *Transfer) { t.Move = plan.Transfer{} }, "kind"},
		"Move sized differently":  {func(t *Transfer) { t.Move.Tokens = 65 }, "reports 64 tokens and Move 65"},
		"load that goes to tier":  {func(t *Transfer) { t.Move.Kind = plan.TransferToTier }, "a load whose Move goes to a tier"},
		"store from a tier":       {func(t *Transfer) { t.Kind, t.Move.Kind = TransferStore, plan.TransferFromTier }, "a store whose Move comes from a tier"},
		"tier move with no name":  {func(t *Transfer) { t.Tier = "" }, "tier"},
		"peer move naming a tier": {func(t *Transfer) { t.Move = peerPull().Move; t.Tokens = 32; t.Move.Tokens = 32 }, "tier"},
		"unknown direction":       {func(t *Transfer) { t.Kind = "sideways" }, `kind "sideways"`},
		"no request":              {func(t *Transfer) { t.Request = "" }, "no request"},
		"no tokens":               {func(t *Transfer) { t.Tokens, t.Move.Tokens = 0, 0 }, "0 tokens"},
		"negative blocks":         {func(t *Transfer) { t.Blocks = -1 }, "-1 blocks"},
		"Move with no device":     {func(t *Transfer) { t.Move.Tier = "" }, "names no tier"},
		"negative endpoint":       {func(t *Transfer) { t.Move.To = plan.Endpoint{Node: -1} }, "endpoint"},
	} {
		tr := offloadLoad()
		c.mutate(&tr)
		err := tr.Validate()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Validate = %v, want a refusal mentioning %q", name, err, c.want)
		}
	}
}

// A decode engine learns where to pull from only from its request, so a remote prefill without a
// peer cannot be priced and is refused where it enters, and a peer on any other request is a
// host that does not know what it is asking for.
func TestAPullNamesItsPeerOnTheRequest(t *testing.T) {
	peer := plan.Endpoint{Node: 2, Rack: 0}
	if err := (Request{ID: "d", RemotePrefill: true, Peer: &peer}).Validate(); err != nil {
		t.Errorf("remote prefill with a peer: %v", err)
	}
	for name, r := range map[string]Request{
		"remote prefill with no peer": {ID: "d", RemotePrefill: true},
		"a peer on a plain request":   {ID: "p", Peer: &peer},
		"a negative peer node":        {ID: "d", RemotePrefill: true, Peer: &plan.Endpoint{Node: -1}},
	} {
		if err := r.Validate(); err == nil {
			t.Errorf("%s: Validate accepted it", name)
		}
	}
	if err := (Request{ID: "x", RemoteDecode: true}).Validate(); err != nil {
		t.Errorf("a remote decode needs no peer: %v", err)
	}
}

// A remote prefill can find all of its KV already here and pull nothing; a load or store to a tier
// of nothing is a bug in the engine.
func TestAPullOfNothingIsAcceptedAndATierMoveOfNothingIsNot(t *testing.T) {
	empty := peerPull()
	empty.Tokens, empty.Move.Tokens = 0, 0
	if err := empty.Validate(); err != nil {
		t.Errorf("a pull of no tokens: %v", err)
	}
	for name, tr := range map[string]Transfer{"load": offloadLoad(), "store": offloadStore()} {
		tr.Tokens, tr.Move.Tokens, tr.Blocks = 0, 0, 0
		if err := tr.Validate(); err == nil || !strings.Contains(err.Error(), "0 tokens") {
			t.Errorf("tier %s of no tokens: Validate = %v, want a refusal", name, err)
		}
	}
	neg := peerPull()
	neg.Tokens, neg.Move.Tokens = -1, -1
	if err := neg.Validate(); err == nil {
		t.Error("a pull of -1 tokens was accepted")
	}
}
