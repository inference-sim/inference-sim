package cmd

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root
)

// startingKVBlocks is the per-replica KV block count a run logs at info level, after
// DP-as-placement has decided the replicas: the value every instance is constructed with.
var startingKVBlocks = regexp.MustCompile(`Starting simulation with (\d+) KV blocks`)

// dpScenarioSettings is the kernel's own answer for the dp=2 MoE scenario -- the oracle the
// run's sizing is checked against, read independently of the run.
func dpScenarioSettings(t *testing.T) kernelmodel.Settings {
	t.Helper()
	scenarios, catalog, registry := kernelRepos(t)
	m, err := kernelmodel.Open(dpScenario, kernelmodel.Repos{Scenarios: scenarios, Catalog: catalog, Registry: registry})
	if err != nil {
		t.Fatal(err)
	}
	st, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	// Non-vacuity: the law below distinguishes per-rank from aggregate only if they differ.
	if st.DataParallel < 2 || st.AggregateKVBlocks == st.KVBlocks {
		t.Fatalf("fixture lost its point: dp %d, per-rank %d, aggregate %d blocks",
			st.DataParallel, st.KVBlocks, st.AggregateKVBlocks)
	}
	return st
}

// loggedReplicaKVBlocks reads the per-replica KV block count from a run's info log.
func loggedReplicaKVBlocks(t *testing.T, stderr string) int64 {
	t.Helper()
	m := startingKVBlocks.FindStringSubmatch(stderr)
	if m == nil {
		t.Fatalf("the run logged no per-replica KV block count:\n%s", lastLines(stderr, 5))
	}
	n, err := strconv.ParseInt(m[1], 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// The dp^2 double-count law (BC-2), end to end on the kernel path. With DP-as-placement each
// of a logical engine's dp replicas is one data-parallel rank, so:
//
//   - every replica is sized with the kernel's PER-RANK budget, never the aggregate;
//   - a logical engine's dp replicas together hold exactly the kernel's aggregate -- not dp
//     times it, which is what sizing each replica from the aggregate would give;
//   - the run spawns logical instances x dp replicas, and conserves its requests.
func TestRunCmd_MoEDPPlacement_EachReplicaIsSizedPerRank(t *testing.T) {
	st := dpScenarioSettings(t)
	const logical = 2
	out, stderr, err := runKernelCLI(t, dpRunArgs(logical, "--log", "info")...)
	if err != nil {
		t.Fatalf("dp=%d scenario run: %v\n%s", st.DataParallel, err, lastLines(stderr, 3))
	}
	if ids := instanceIDs(t, out); len(ids) != logical*st.DataParallel {
		t.Fatalf("%d logical instances x dp %d must run %d replicas, got %d", logical, st.DataParallel,
			logical*st.DataParallel, len(ids))
	}
	perReplica := loggedReplicaKVBlocks(t, stderr)
	if perReplica != st.KVBlocks {
		t.Errorf("each replica holds %d KV blocks; the kernel's per-rank budget is %d (its aggregate is %d)",
			perReplica, st.KVBlocks, st.AggregateKVBlocks)
	}
	if got := perReplica * int64(st.DataParallel); got != st.AggregateKVBlocks {
		t.Errorf("one logical engine's %d replicas hold %d blocks together; the kernel's aggregate is %d "+
			"(a dp^2 double-count would give %d)", st.DataParallel, got, st.AggregateKVBlocks,
			st.AggregateKVBlocks*int64(st.DataParallel))
	}
	clusterConservationHolds(t, out, dpFixtureNumRequests)
}

// Node pools with a dp>1 MoE scenario (BC-6): the logical x dp replicas are placed from the
// pool's inventory -- together reserving logical x dp x tp GPUs -- each is sized per rank like
// any other replica, and every request is conserved, so every reservation succeeded.
func TestRunCmd_MoEDPPlacement_NodePools_NxM(t *testing.T) {
	st := dpScenarioSettings(t)
	const logical = 2 // x dp 2 = 4 replicas x tp 2 = 8 GPUs: exactly the pool's one node
	bundle := filepath.Join(t.TempDir(), "nodepools.yaml")
	if err := os.WriteFile(bundle, []byte("node_pools:\n  - name: pool-b200\n    gpu_type: b200\n"+
		"    gpus_per_node: 8\n    gpu_memory_gib: 178.35107422\n    initial_nodes: 1\n    min_nodes: 1\n"+
		"    max_nodes: 1\n    cost_per_hour: 40.0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, stderr, err := runKernelCLI(t, dpRunArgs(logical, "--policy-config", bundle, "--log", "info")...)
	if err != nil {
		t.Fatalf("node pools + dp=%d: %v\n%s", st.DataParallel, err, lastLines(stderr, 3))
	}
	if ids := instanceIDs(t, out); len(ids) != logical*st.DataParallel {
		t.Errorf("%d logical instances x dp %d must place %d replicas from the pool, got %d: %v",
			logical, st.DataParallel, logical*st.DataParallel, len(ids), ids)
	}
	if perReplica := loggedReplicaKVBlocks(t, stderr); perReplica != st.KVBlocks {
		t.Errorf("a pool-placed replica holds %d KV blocks; the kernel's per-rank budget is %d", perReplica, st.KVBlocks)
	}
	clusterConservationHolds(t, out, dpFixtureNumRequests)
}
