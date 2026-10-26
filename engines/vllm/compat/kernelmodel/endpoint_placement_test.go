// Moved from inference-sim/sim/kernelmodel/endpoint_placement_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// A rank may cross a node boundary even when it uses fewer GPUs than one node
// holds. PlacementOf names its first node for pricing; one Pod must not claim
// that node as the rank's sole host.
func TestEndpointPlacementDetectsCrossingRank(t *testing.T) {
	m := &Model{id: identity{gpusPerNode: 4, rankGPUs: 3}}
	first, nodes, declaredRack := m.EndpointPlacement(0)
	if first.Node != 0 || nodes != 1 || declaredRack {
		t.Errorf("first rank: placement=%+v nodes=%d declaredRack=%v", first, nodes, declaredRack)
	}
	second, nodes, declaredRack := m.EndpointPlacement(1)
	if second.Node != 0 || nodes != 2 || declaredRack {
		t.Errorf("crossing rank: placement=%+v nodes=%d declaredRack=%v", second, nodes, declaredRack)
	}
}

// A stated dp_local is refused where vLLM's local layout (dp_local ranks to a node, each from
// device i x world) puts a replica's ranks on other nodes than PlacementOf's packing, and
// accepted where the two agree, as for the shipped DEP8 scenario (dp 8, dp_local 8, one GPU a
// rank, 8-GPU nodes). Unstated is the packing.
func TestCheckDPLocal(t *testing.T) {
	for _, c := range []struct {
		name        string
		pl          deployment.Parallelism
		gpusPerNode int
		ok          bool
	}{
		{"unstated", deployment.Parallelism{TP: 1, PP: 1, DP: 8}, 8, true},
		{"DEP8: every rank local, one node", deployment.Parallelism{TP: 1, PP: 1, DP: 8, DPLocal: 8}, 8, true},
		{"the packed density", deployment.Parallelism{TP: 2, PP: 1, DP: 8, DPLocal: 4}, 8, true},
		{"fewer ranks a node than the packing", deployment.Parallelism{TP: 2, PP: 1, DP: 8, DPLocal: 2}, 8, false},
		{"one rank a node of eight GPUs", deployment.Parallelism{TP: 1, PP: 1, DP: 4, DPLocal: 1}, 8, false},
		{"pcp widens the rank", deployment.Parallelism{TP: 2, PP: 1, PCP: 2, DP: 4, DPLocal: 4}, 8, false},
		{"pp widens the rank", deployment.Parallelism{TP: 2, PP: 2, DP: 4, DPLocal: 2}, 8, true},
		{"a replica on fewer ranks than a node holds", deployment.Parallelism{TP: 1, PP: 1, DP: 2, DPLocal: 2}, 8, true},
		{"a rank of whole nodes, one replica a node group", deployment.Parallelism{TP: 16, PP: 1, DP: 2, DPLocal: 1}, 8, true},
		{"a rank of whole nodes, two replicas sharing nodes", deployment.Parallelism{TP: 16, PP: 1, DP: 2, DPLocal: 2}, 8, false},
		{"no node size", deployment.Parallelism{TP: 1, PP: 1, DP: 8, DPLocal: 2}, 0, true},
		// Ranks that do not tile a node: the packing puts rank 1 across nodes 0 and 1 (tp 3 on
		// 4-GPU nodes), so it is on no single node whichever end decides.
		{"a rank across a node boundary, two to a node", deployment.Parallelism{TP: 3, PP: 1, DP: 2, DPLocal: 2}, 4, false},
		{"a rank across a node boundary, one to a node", deployment.Parallelism{TP: 3, PP: 1, DP: 2, DPLocal: 1}, 4, false},
	} {
		err := checkDPLocal(c.pl, c.gpusPerNode)
		switch {
		case c.ok && err != nil:
			t.Errorf("%s: refused: %v", c.name, err)
		case !c.ok && (err == nil || !strings.Contains(err.Error(), "parallel.dp_local")):
			t.Errorf("%s: %v, want a refusal naming parallel.dp_local", c.name, err)
		}
	}
}
