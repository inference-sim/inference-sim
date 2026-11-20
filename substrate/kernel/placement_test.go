package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// vllmGroups transcribes how vLLM v0.31.0 builds one parallel group, rank by rank, for a
// world of dp x pcp x tp ranks (pp is 1 throughout this repository's layouts).
//
// It is the ORACLE for groupPerNode: rather than restating the stride rule the kernel
// encodes, it rebuilds the groups the way initialize_model_parallel does
// (vllm/distributed/parallel_state.py at v0.31.0) and lets the test count where their
// members land.
//
//	all_ranks = arange(world).reshape(-1, dp, pp, pcp, tp)            :2054-2060
//	TP   all_ranks.view(-1, tp)                                       :2065
//	DCP  all_ranks.transpose(-1, -2).reshape(-1, dcp)  when dcp > 1   :2139-2144
//	PCP  all_ranks.transpose(3, 4).reshape(-1, pcp)                   :2155-2160
//	EP   all_ranks.transpose(1, 2).reshape(-1, dp*pcp*tp)             :2212-2220
func vllmGroups(axis price.GroupAxis, dp, pcp, tp, dcp int) [][]int {
	// rank(d, p, t) = ((d*pcp)+p)*tp + t, with pp = 1.
	rank := func(d, p, t int) int { return (d*pcp+p)*tp + t }
	var groups [][]int
	switch axis {
	case price.GroupTP:
		for d := 0; d < dp; d++ {
			for p := 0; p < pcp; p++ {
				var g []int
				for t := 0; t < tp; t++ {
					g = append(g, rank(d, p, t))
				}
				groups = append(groups, g)
			}
		}
	case price.GroupPCP:
		for d := 0; d < dp; d++ {
			for t := 0; t < tp; t++ {
				var g []int
				for p := 0; p < pcp; p++ {
					g = append(g, rank(d, p, t))
				}
				groups = append(groups, g)
			}
		}
	case price.GroupDCP:
		// Transposing the last two axes orders each data-parallel block tp-major, pcp-minor;
		// consecutive runs of dcp in that order are the groups.
		for d := 0; d < dp; d++ {
			var order []int
			for t := 0; t < tp; t++ {
				for p := 0; p < pcp; p++ {
					order = append(order, rank(d, p, t))
				}
			}
			for i := 0; i+dcp <= len(order); i += dcp {
				groups = append(groups, append([]int(nil), order[i:i+dcp]...))
			}
		}
	case price.GroupExpert:
		var g []int
		for r := 0; r < dp*pcp*tp; r++ {
			g = append(g, r)
		}
		groups = append(groups, g)
	}
	return groups
}

// The kernel's placement must agree with the engine's for every group of every admissible
// layout: whether the group crosses a node, and how many of its members share one.
//
// Checked over the grid a deployment can state -- 4- and 8-GPU nodes, tp 1 to 8, pcp 1 to
// 4, dp 1 and 2, and every dcp the engine admits for that pcp (parallel.py:563-578) -- with
// the oracle above. A group whose members are spread unevenly has no single per-node
// count, so the comparison is against the BUSIEST node, which is the count a uniform
// placement would give; every layout in the grid places its groups evenly.
func TestCollectivePlacementMatchesTheEnginesRankLayout(t *testing.T) {
	for _, gpn := range []int{4, 8} {
		for _, tp := range []int{1, 2, 4, 8} {
			for _, pcp := range []int{1, 2, 4} {
				for _, dp := range []int{1, 2} {
					var dcps []int
					if pcp == 1 {
						for d := 1; d <= tp; d++ {
							if tp%d == 0 {
								dcps = append(dcps, d)
							}
						}
					} else {
						dcps = []int{1, pcp, tp * pcp}
					}
					for _, dcp := range dcps {
						k := &Kernel{layout: resolve.Layout{
							TP: tp, PCP: pcp, DCP: dcp, DP: dp,
							ExpertWidth: tp * pcp * dp, GPUsPerNode: gpn,
						}}
						for _, axis := range []price.GroupAxis{
							price.GroupTP, price.GroupPCP, price.GroupDCP, price.GroupExpert,
						} {
							width, _ := k.groupSize(axis)
							if width <= 1 {
								continue
							}
							for _, g := range vllmGroups(axis, dp, pcp, tp, width) {
								nodes := map[int]int{}
								for _, r := range g {
									nodes[r/gpn]++
								}
								busiest := 0
								for _, n := range nodes {
									busiest = max(busiest, n)
								}
								crosses := len(nodes) > 1
								key := collKey{Group: axis}
								if got := k.crossesNodes(key); got != crosses {
									t.Errorf("gpn=%d tp=%d pcp=%d dcp=%d dp=%d axis %d: "+
										"group %v crosses a node: %v, the kernel says %v",
										gpn, tp, pcp, dcp, dp, axis, g, crosses, got)
								}
								if got := min(k.groupPerNode(axis), width); got != busiest {
									t.Errorf("gpn=%d tp=%d pcp=%d dcp=%d dp=%d axis %d: "+
										"group %v puts %d member(s) on its busiest node, "+
										"the kernel %d", gpn, tp, pcp, dcp, dp, axis, g,
										busiest, got)
								}
							}
						}
					}
				}
			}
		}
	}
}
