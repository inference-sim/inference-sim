package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
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

// pdFixtureWithFabric opens the 3P1D fixture's prefill and decode pools with cluster.fabric
// set to fabric (a catalog network), written to a fresh scenario directory.
func pdFixtureWithFabric(t *testing.T, fabric string) (prefill, decode *Model) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("../../testdata/scenarios", "glm-5-h200-3p1d-ib.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(string(raw), "  fabric: ib-400g\n", "  fabric: "+fabric+"\n", 1)
	if fabric != "ib-400g" && body == string(raw) {
		t.Fatal("fixture names no ib-400g fabric to replace")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	r := repos()
	r.Scenarios = dir
	pf, err := OpenRole("variant.yaml", r, deployment.RolePrefill)
	if err != nil {
		t.Fatal(err)
	}
	dc, err := OpenRole("variant.yaml", r, deployment.RoleDecode)
	if err != nil {
		t.Fatal(err)
	}
	return pf, dc
}

// PDTransferTicks is the kernel's PDTransferTime in whole ticks rounded up: at least one tick,
// enough to cover the duration, and one fewer would not -- a handoff is not done until its last
// byte lands, and a zero-tick one would let decode start in the instant prefill ends.
func TestPDTransferTicks_RoundsTheKernelsTimeUpToWholeTicks(t *testing.T) {
	pf, dc := pdFixtureWithFabric(t, "ib-400g")
	rapid.Check(t, func(rt *rapid.T) {
		from := pf.PlacementOf(rapid.IntRange(0, pf.RankCapacity()-1).Draw(rt, "from"))
		to := dc.PlacementOf(rapid.IntRange(0, dc.RankCapacity()-1).Draw(rt, "to"))
		tokens := rapid.Int64Range(1, 1<<20).Draw(rt, "tokens")
		ticks := pf.PDTransferTicks(tokens, from, to)
		ns := pf.Kernel().PDTransferTime(int(tokens), from, to).Nanoseconds()
		if ticks < 1 || ticks*1000 < ns || (ticks > 1 && (ticks-1)*1000 >= ns) {
			rt.Fatalf("%d tokens: %d ns charged as %d ticks; want the least whole number of ticks covering it, at least 1",
				tokens, ns, ticks)
		}
	})
}

// A slower fabric never makes a handoff cheaper: across the catalog's fabrics, ordered by
// nominal bandwidth, every prefill-to-decode handoff costs at least as much on the slower one,
// and strictly more for some -- the pools sit on different nodes, so every handoff crosses it.
func TestPDHandoff_ASlowerFabricIsNeverCheaper(t *testing.T) {
	// Fastest first: ib-400g 50 GB/s, roce-200g 25 GB/s, ethernet-100gbe 12.5 GB/s.
	fabrics := []string{"ib-400g", "roce-200g", "ethernet-100gbe"}
	type pools struct{ pf, dc *Model }
	opened := make([]pools, len(fabrics))
	for i, f := range fabrics {
		pf, dc := pdFixtureWithFabric(t, f)
		opened[i] = pools{pf, dc}
	}
	strictly := false
	rapid.Check(t, func(rt *rapid.T) {
		i := rapid.IntRange(0, opened[0].pf.RankCapacity()-1).Draw(rt, "from")
		j := rapid.IntRange(0, opened[0].dc.RankCapacity()-1).Draw(rt, "to")
		tokens := rapid.Int64Range(1, 1<<20).Draw(rt, "tokens")
		prev := int64(-1)
		for k, p := range opened {
			got := p.pf.PDTransferTicks(tokens, p.pf.PlacementOf(i), p.dc.PlacementOf(j))
			if got < prev {
				rt.Fatalf("%d tokens rank %d -> %d: %s prices %d ticks, below faster %s's %d",
					tokens, i, j, fabrics[k], got, fabrics[k-1], prev)
			}
			if k > 0 && got > prev {
				strictly = true
			}
			prev = got
		}
	})
	if !strictly {
		t.Error("no handoff cost more on a slower fabric; the fabric does not reach the handoff price")
	}
}
