package build

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// A host that runs a data-parallel group as one instance counts groups: a pool of 16 GPUs at dp 8 holds two, so
// three instances are refused, though the pool holds sixteen ranks and RequireCapacity (one instance for each
// rank) accepts them.
func TestRequireGroupCapacityCountsGroupsNotRanks(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "vllmengine", "testdata", "muse-gb200-nvl-dep8-dspark.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	repos := vendoredRepos(t)
	repos.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(repos.Scenarios, "v.yaml"), []byte(strings.ReplaceAll(string(raw), "nodes: 2", "nodes: 4")), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := kernelmodel.Open("v.yaml", repos)
	if err != nil {
		t.Fatal(err)
	}
	two, three := Topology{NumInstances: 2}, Topology{NumInstances: 3}
	if err := RequireGroupCapacity("v.yaml", m, nil, two); err != nil {
		t.Errorf("two groups in a pool of two: %v", err)
	}
	err = RequireGroupCapacity("v.yaml", m, nil, three)
	if err == nil || !strings.Contains(err.Error(), "holds 2 data-parallel group(s)") {
		t.Errorf("three groups in a pool of two: %v, want a refusal naming the capacity", err)
	}
	if err := RequireCapacity("v.yaml", m, nil, three); err != nil {
		t.Errorf("by rank the pool holds sixteen, so three instances fit: %v", err)
	}
}

// The endpoint topology for a host that runs a group as one instance: each instance is a group, placed at the
// group's first node, with the nodes it spans (a group of 8 GPUs on 4-GPU nodes spans two, so it has no single
// hostname); a third instance in a pool of two groups is refused. The rank-based topology would give sixteen
// single-node endpoints.
func TestEndpointGroupPlacementsAreWholeGroups(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "vllmengine", "testdata", "muse-gb200-nvl-dep8-dspark.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	repos := vendoredRepos(t)
	repos.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(repos.Scenarios, "v.yaml"), []byte(strings.ReplaceAll(string(raw), "nodes: 2", "nodes: 4")), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := kernelmodel.Open("v.yaml", repos)
	if err != nil {
		t.Fatal(err)
	}
	got, err := EndpointGroupPlacements(m, nil, 2, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got["instance_0"].NodeCount != 2 || got["instance_1"].NodeCount != 2 || got["instance_0"].Hostname != "" {
		t.Errorf("placements %+v, want two groups of two nodes each and no single hostname", got)
	}
	if _, err := EndpointGroupPlacements(m, nil, 3, 0, 0); err == nil || !strings.Contains(err.Error(), "2-group capacity") {
		t.Errorf("three instances in a pool of two groups: %v, want a capacity refusal", err)
	}
	rank, err := EndpointPlacements(m, nil, 3, 0, 0)
	if err != nil || len(rank) != 3 {
		t.Errorf("the rank-based topology accepts three: %v", err)
	}
}
