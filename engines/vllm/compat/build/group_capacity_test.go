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
