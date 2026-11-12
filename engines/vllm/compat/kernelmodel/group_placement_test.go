package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A host that runs a data-parallel group as one instance (core/vllm's DPGroup) counts groups, not ranks: a pool
// of 16 GPUs at dp 8 holds two groups, the second starting on the node after the first group's, and the engine is
// placed there. The rank accessors keep counting ranks, for a host that runs one instance for each.
func TestAPoolHoldsGroupsOfDataParallelRanks(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "vllmengine", "testdata", "muse-gb200-nvl-dep8-dspark.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	// Four nodes of four GPUs: two groups of eight.
	body := strings.ReplaceAll(string(raw), "nodes: 2", "nodes: 4")
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open("v.yaml", r)
	if err != nil {
		t.Fatal(err)
	}
	if g := m.GroupGPUs(); g != 8 {
		t.Fatalf("a group is %d GPUs, want dp 8 x 1", g)
	}
	if got := m.GroupCapacity(); got != 2 {
		t.Errorf("the pool holds %d groups, want 2", got)
	}
	if got := m.RankCapacity(); got != 16 {
		t.Errorf("the pool holds %d ranks, want 16", got)
	}
	for g, node := range []int{0, 2} {
		if got := m.GroupPlacementOf(g).Node; got != node {
			t.Errorf("group %d is on node %d, want %d", g, got, node)
		}
		if got := m.EngineGroupPlacementOf(g); got.Node != node || got.Nodes != 4 || got.Data != 8 {
			t.Errorf("group %d placement %+v, want node %d of 4, data 8", g, got, node)
		}
	}
	// Per rank the old accessors are unchanged: rank 1 shares group 0's node.
	if got := m.PlacementOf(1).Node; got != 0 {
		t.Errorf("rank 1 is on node %d, want 0", got)
	}
}

// A group is never started part-way into a node it does not fill. Groups of 6 GPUs on 8-GPU nodes take one node
// each (6 of its 8 GPUs) and are placed on nodes 0, 1, 2 with no overlap; the formula "group x GPUs / GPUs per
// node" would put group 1 on node 0 again (GPU 6) and run it into node 1, over group 2's GPUs. A group wider than a
// node must be whole nodes: 12 GPUs on 8-GPU nodes is refused, and the pool holds none. (The layout functions are
// tested directly because the registry measures collectives at 2, 4, 8 and 16 ranks only, so a 6-GPU group cannot be
// opened as a scenario.)
func TestGroupsOfAnAwkwardSizeAreRefusedOrPackedWithoutOverlap(t *testing.T) {
	if _, _, err := groupLayoutOf(6, 8); err != nil {
		t.Fatalf("6-GPU groups on 8-GPU nodes: %v", err)
	}
	if got := groupCapacityOf(3, 6, 8); got != 3 {
		t.Errorf("three 8-GPU nodes hold %d groups of 6, want 3 (one a node)", got)
	}
	for i, node := range []int{0, 1, 2} {
		if got := groupNodeOf(i, 6, 8); got != node {
			t.Errorf("group %d of 6 is on node %d, want %d", i, got, node)
		}
	}
	// Groups that divide the node pack several to a node.
	if got, node := groupCapacityOf(2, 4, 8), groupNodeOf(3, 4, 8); got != 4 || node != 1 {
		t.Errorf("4-GPU groups on two 8-GPU nodes: capacity %d, group 3 on node %d; want 4 and 1", got, node)
	}
	_, _, err := groupLayoutOf(12, 8)
	if err == nil {
		t.Errorf("12-GPU groups on 8-GPU nodes: %v, want a refusal", err)
	}
	if got := groupCapacityOf(4, 12, 8); got != 0 {
		t.Errorf("a refused layout holds %d groups, want 0", got)
	}
	// Whole nodes: 16 GPUs on 8-GPU nodes take two each.
	if got := groupCapacityOf(5, 16, 8); got != 2 {
		t.Errorf("five 8-GPU nodes hold %d groups of 16, want 2", got)
	}
	for i, node := range []int{0, 2} {
		if got := groupNodeOf(i, 16, 8); got != node {
			t.Errorf("group %d of 16 is on node %d, want %d", i, got, node)
		}
	}
}
