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
