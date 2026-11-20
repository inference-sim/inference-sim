package plan

import (
	"strings"
	"testing"
)

func poolWithTier() PoolPlan {
	return PoolPlan{Hardware: "h100", Layout: Layout{Tensor: 2, RanksPerNode: 8, Nodes: 1},
		Tiers: []KVTier{{Device: "cpu_dram"}}}
}

// A move to or from a tier is made by this pool only if the tier is one the plan lists, which is
// how an engine's own tier name reaches a catalog device: it resolves the name into Tiers and
// states the device in the move.
func TestAPoolMakesOnlyMovesToItsOwnTiers(t *testing.T) {
	p := poolWithTier()
	for name, tr := range map[string]Transfer{
		"load from the listed tier": {Tokens: 8, Kind: TransferFromTier, Tier: "cpu_dram"},
		"store to the listed tier":  {Tokens: 8, Kind: TransferToTier, Tier: "cpu_dram"},
		"pull from a peer":          {Tokens: 8, Kind: TransferPeer, From: Endpoint{Node: 5}},
	} {
		if err := p.CheckTransfer(tr); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	for name, c := range map[string]struct {
		tr   Transfer
		want string
	}{
		"a tier the plan lacks":      {Transfer{Tokens: 8, Kind: TransferFromTier, Tier: "nvme_gen4"}, "not a device of this pool's plan"},
		"the engine's own tier name": {Transfer{Tokens: 8, Kind: TransferFromTier, Tier: "cpu"}, "not a device"},
		"a node the pool lacks":      {Transfer{Tokens: 8, Kind: TransferToTier, Tier: "cpu_dram", To: Endpoint{Node: 1}}, "node 1 of a pool with 1"},
		"a move that is invalid":     {Transfer{Kind: TransferToTier, Tier: "cpu_dram"}, "0 tokens"},
	} {
		if err := p.CheckTransfer(c.tr); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: CheckTransfer = %v, want %q", name, err, c.want)
		}
	}
}

// A pull between pools may move nothing; a move to or from a tier may not, and no move has a
// negative size.
func TestOnlyAPullBetweenPoolsMayMoveNothing(t *testing.T) {
	p := poolWithTier()
	if err := p.CheckTransfer(Transfer{Kind: TransferPeer, From: Endpoint{Node: 2}}); err != nil {
		t.Errorf("a pull of no tokens: %v", err)
	}
	for name, tr := range map[string]Transfer{
		"a load of nothing":  {Kind: TransferFromTier, Tier: "cpu_dram"},
		"a store of nothing": {Kind: TransferToTier, Tier: "cpu_dram"},
		"a negative pull":    {Tokens: -1, Kind: TransferPeer},
	} {
		if err := p.CheckTransfer(tr); err == nil || !strings.Contains(err.Error(), "tokens") {
			t.Errorf("%s: CheckTransfer = %v, want a refusal", name, err)
		}
	}
}
