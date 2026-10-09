package kernelmodel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"pgregory.net/rapid"
)

// Laws of placing a disaggregated deployment's ranks, on the committed 3P1D fixture:
//
//   - every rank a pool holds sits on one of that pool's nodes;
//   - the two pools' nodes are disjoint, as the deployment's pools are;
//   - ranks are packed: two ranks on one node never need more GPUs than the node has.
//
// And of pricing a handoff between them: moving more KV never costs less.
func TestPlacementAndHandoffLaws(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	pools := map[deployment.Role]*Model{}
	for _, role := range []deployment.Role{deployment.RolePrefill, deployment.RoleDecode} {
		m, err := OpenRole("glm-5-h200-3p1d-ib.yaml", r, role)
		if err != nil {
			t.Fatal(err)
		}
		pools[role] = m
	}
	nodesOf := func(m *Model) map[int]int { // node -> ranks placed on it
		n := map[int]int{}
		for rank := 0; rank < m.RankCapacity(); rank++ {
			pl := m.PlacementOf(rank)
			if pl.Node < m.id.firstNode || pl.Node >= m.id.firstNode+m.id.poolNodes || pl.Pool != m.Role() {
				t.Errorf("%s rank %d placed at %+v, outside the pool's nodes [%d, %d)", m.Role(), rank, pl,
					m.id.firstNode, m.id.firstNode+m.id.poolNodes)
			}
			n[pl.Node]++
		}
		for node, ranks := range n {
			if ranks*m.id.rankGPUs > m.id.gpusPerNode {
				t.Errorf("%s: node %d holds %d ranks of %d GPUs on a %d-GPU node", m.Role(), node, ranks,
					m.id.rankGPUs, m.id.gpusPerNode)
			}
		}
		return n
	}
	prefillNodes, decodeNodes := nodesOf(pools[deployment.RolePrefill]), nodesOf(pools[deployment.RoleDecode])
	for node := range prefillNodes {
		if decodeNodes[node] > 0 {
			t.Errorf("node %d holds both prefill and decode ranks", node)
		}
	}

	pf, dc := pools[deployment.RolePrefill], pools[deployment.RoleDecode]
	rapid.Check(t, func(rt *rapid.T) {
		from := pf.PlacementOf(rapid.IntRange(0, pf.RankCapacity()-1).Draw(rt, "from"))
		to := dc.PlacementOf(rapid.IntRange(0, dc.RankCapacity()-1).Draw(rt, "to"))
		a := rapid.Int64Range(1, 1<<20).Draw(rt, "tokens")
		b := rapid.Int64Range(a, 1<<20).Draw(rt, "more")
		if pa, pb := pf.PDTransferTicks(a, from, to), pf.PDTransferTicks(b, from, to); pb < pa {
			rt.Fatalf("moving %d tokens cost %d ticks, less than %d tokens' %d", b, pb, a, pa)
		}
	})
}
