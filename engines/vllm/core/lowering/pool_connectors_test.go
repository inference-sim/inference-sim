package lowering

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// The connectors a deployment configures are those of every block it states, a pool's own included:
// a MultiConnector on one pool leaves the NIXL child unknowable, and a NixlConnector on a pool pins the
// interleave. (A mixed deployment, one pool stating a connector and another none, is refused by the
// substrate's validation, which requires every prefill and decode pool to resolve one, so a pool whose
// connector is empty cannot reach here.)
func TestConnectorNamesUnionEveryPoolsBlocks(t *testing.T) {
	pool := func(role deployment.Role, pd string, off *deployment.Offload) deployment.Pool {
		p := deployment.Pool{Role: role, Offload: off}
		if pd != "" {
			p.PDTransfer = &deployment.PDTransfer{Connector: pd}
		}
		return p
	}
	cpu := &deployment.Offload{Connector: "OffloadingConnector", Tiers: []deployment.Tier{{Device: "cpu_dram", Bytes: 1}}}
	d := &deployment.Deployment{Pools: []deployment.Pool{
		pool(deployment.RolePrefill, "MultiConnector", cpu),
		pool(deployment.RoleDecode, "NixlConnector", nil),
	}}
	names := connectorNames(d)
	for _, want := range []string{"MultiConnector", "NixlConnector", "OffloadingConnector"} {
		if !names[want] {
			t.Errorf("connectors %v: want %s (a pool's own block)", names, want)
		}
	}
	if _, err := nixlPinsInterleave(d); err == nil {
		t.Error("a MultiConnector on the prefill pool left the interleave pin answerable")
	}
	d.Pools[0].PDTransfer.Connector = "NixlConnector"
	if pinned, err := nixlPinsInterleave(d); err != nil || !pinned {
		t.Errorf("NIXL on both pools: pinned %v, %v; want pinned", pinned, err)
	}
	d.Pools[0].PDTransfer, d.Pools[1].PDTransfer = nil, nil
	if pinned, err := nixlPinsInterleave(d); err != nil || pinned {
		t.Errorf("no P/D connector: pinned %v, %v; want unpinned", pinned, err)
	}
}
