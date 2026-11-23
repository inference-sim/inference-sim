package resolve

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/scenario"
)

// scenarioWith builds the pair a resolver needs: the immutable problem and the
// deployment applied to it. They are two documents in the schema, so a helper that
// returned only a scenario would leave every caller to construct the other half.
func scenarioWith(nodes, gpusPerNode int,
	pools ...deployment.Pool) (*scenario.Scenario, *deployment.Deployment) {
	return &scenario.Scenario{
		Kind: "Scenario", Name: "t", Model: "m",
		Coefficients: []string{"c"}, Engine: &scenario.Engine{Kind: "vllm", Version: "0.29.0"},
		Cluster: scenario.Cluster{
			Hardware: "h", Nodes: nodes, GPUsPerNode: gpusPerNode,
		},
	}, &deployment.Deployment{
		Kind: "Deployment", Name: "t", Pools: pools,
	}
}

// packagingOf is the packaging a chip that leaves both fields open resolves the test
// scenario's cluster to: the cluster's own values. The resolver itself is tested in
// contracts/spec/scenario.
func packagingOf(s *scenario.Scenario) scenario.Packaging {
	return scenario.Packaging{GPUsPerNode: s.Cluster.GPUsPerNode, GPUsPerRack: s.Cluster.GPUsPerRack}
}

// pool is a colocated pool with the parallel block pl. Its engine block is unstated: resolve
// reads no engine setting.
func pool(pl deployment.Parallelism) deployment.Pool {
	return deployment.Pool{Role: deployment.RoleColocated, Nodes: 1, Parallel: pl}
}

func TestNodesSpannedFollowsTheWidestGroup(t *testing.T) {
	cases := []struct {
		what        string
		gpusPerNode int
		pl          deployment.Parallelism
		rack        int
		rackDomain  bool
		want        int
	}{
		{"tp 8 on 8-GPU nodes stays on one node", 8,
			deployment.Parallelism{TP: 8, PP: 1, DP: 1}, 0, false, 1},
		{"tp 16 on 8-GPU nodes spans two", 8,
			deployment.Parallelism{TP: 16, PP: 1, DP: 1}, 0, false, 2},
		{"expert width is the widest group", 8,
			deployment.Parallelism{TP: 1, PP: 1, DP: 16, EnableExpertParallel: true}, 0, false, 2},
		{"multi-node NVLink makes a rack one domain", 4,
			deployment.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}, 72, true, 1},
		{"a rack that is not one domain still spans nodes", 4,
			deployment.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}, 72, false, 18},
		// A two-rank prefill-context group at tp=8 has a member on each of two nodes: its
		// ranks are tp apart. Expert parallelism is off, so no wider group hides it.
		{"a prefill-context group spans its tp x pcp block", 8,
			deployment.Parallelism{TP: 8, PP: 1, DP: 1, PCP: 2}, 0, false, 2},
		{"a prefill-context group inside one node stays on it", 8,
			deployment.Parallelism{TP: 2, PP: 1, DP: 1, PCP: 4}, 0, false, 1},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			s, d := scenarioWith(1, c.gpusPerNode, pool(c.pl))
			s.Cluster.GPUsPerRack = c.rack
			l, err := ResolveGeometry(packagingOf(s), d.Pools[0], Fabric{RackIsOneDomain: c.rackDomain})
			if err != nil {
				t.Fatalf("ResolveGeometry: %v", err)
			}
			if l.NodesSpanned != c.want {
				t.Errorf("NodesSpanned = %d, want %d", l.NodesSpanned, c.want)
			}
		})
	}
}

func TestFabricRatio(t *testing.T) {
	cases := []struct {
		intra, inter, want float64
	}{
		{450, 50, 9},  // NVLink 4 against InfiniBand NDR
		{450, 25, 18}, // the same chip behind RoCE
		{900, 900, 1}, // multi-node NVLink: no step at the boundary
		{450, 0, 1},   // no fabric stated: single node, nothing crosses
		{0, 50, 1},    // no chip bandwidth: not a division by an unknown
	}
	for _, c := range cases {
		f := Fabric{IntraNodeBwGBps: c.intra, InterNodeBwGBps: c.inter}
		if got := f.Ratio(); got != c.want {
			t.Errorf("Ratio(%v, %v) = %v, want %v", c.intra, c.inter, got, c.want)
		}
	}
}
