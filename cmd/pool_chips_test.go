package cmd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const fixtureCluster = "  nodes: 1\n  # gpus_per_node: 8 assumed: the corpus states none; every chip in it is an 8-GPU HGX board.\n  gpus_per_node: 8\n"

// twoChipEdits turns the colocated fixture into a two-node scenario whose cluster states a
// pool of H200 and a pool of H100 (SIM-17): two tp4 groups fit on each chip's 8-GPU node.
func twoChipEdits(hardware string) []string {
	return []string{
		fixtureCluster, "  nodes: 2\n  fabric: ib-400g\n  gpus_per_node: 8\n  pools:\n    - name: fast\n      hardware: " + hardware + "\n    - name: older\n      hardware: h100\n",
		"    nodes: 1\n    parallel", "    nodes: 2\n    parallel",
	}
}

// writePoolsBundle writes a --policy-config bundle of one node pool per (name, gpuType,
// memory, price) entry, each one 8-GPU node.
func writePoolsBundle(t *testing.T, pools ...[4]string) string {
	t.Helper()
	body := "node_pools:\n"
	for _, p := range pools {
		body += "  - name: " + p[0] + "\n    gpu_type: " + p[1] + "\n    gpus_per_node: 8\n    gpu_memory_gib: " + p[2] +
			"\n    initial_nodes: 1\n    min_nodes: 1\n    max_nodes: 1\n    cost_per_hour: " + p[3] + "\n"
	}
	path := filepath.Join(t.TempDir(), "bundle.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

type chipRun struct {
	kvTokens float64
	itlMs    map[string]float64
}

func runChips(t *testing.T, dir string, extra ...string) chipRun {
	t.Helper()
	out, stderr, err := runKernelCLI(t, append([]string{"run", "--scenarios", dir, "--scenario", "variant.yaml",
		"--num-requests", "40", "--rate", "1"}, extra...)...)
	if err != nil {
		t.Fatalf("run failed: %v\n%s", err, lastLines(stderr, 3))
	}
	var r struct {
		Reports struct {
			Engine struct {
				Cluster struct {
					KV float64 `json:"kv_capacity_tokens_per_replica"`
				} `json:"cluster"`
				Instances map[string]struct {
					ITL float64 `json:"itl_mean_ms"`
				} `json:"instances"`
			} `json:"engine"`
		} `json:"reports"`
	}
	if err := json.Unmarshal([]byte(out), &r); err != nil {
		t.Fatal(err)
	}
	c := chipRun{kvTokens: r.Reports.Engine.Cluster.KV, itlMs: map[string]float64{}}
	for id, in := range r.Reports.Engine.Instances {
		c.itlMs[id] = in.ITL
	}
	return c
}

// TestRunCmd_PoolsOfTwoChipsEachRunOnTheirOwn is SIM-17's check (#1925): an H200 pool and an
// H100 pool under one workload. The instances placed in the H200 pool step as an H200 run does,
// those in the H100 pool as an H100 run does, and the smallest KV budget in the cluster is the
// H100's. Before the change every instance was priced and sized for the scenario's own chip,
// and a pool of another chip was refused.
func TestRunCmd_PoolsOfTwoChipsEachRunOnTheirOwn(t *testing.T) {
	mixed := runChips(t, writeColocatedVariant(t, twoChipEdits("h200")...), "--num-instances", "4",
		"--policy-config", writePoolsBundle(t, [4]string{"fast", "h200", "139.80072021", "32"},
			[4]string{"older", "h100", "79.1895752", "20"}))
	h200 := runChips(t, writeColocatedVariant(t), "--num-instances", "2")
	h100 := runChips(t, writeColocatedVariant(t, "hardware: h200", "hardware: h100"), "--num-instances", "2")

	if h100.kvTokens >= h200.kvTokens {
		t.Fatalf("fixture: an H100 run holds %v KV tokens, an H200 run %v; the chips must differ", h100.kvTokens, h200.kvTokens)
	}
	if mixed.kvTokens != h100.kvTokens {
		t.Errorf("smallest KV budget = %v tokens, want the H100 pool's %v", mixed.kvTokens, h100.kvTokens)
	}
	mean := func(c chipRun) float64 {
		var s float64
		for _, v := range c.itlMs {
			s += v
		}
		return s / float64(len(c.itlMs))
	}
	for id, want := range map[string]float64{"instance_0": mean(h200), "instance_1": mean(h200),
		"instance_2": mean(h100), "instance_3": mean(h100)} {
		if got := mixed.itlMs[id]; math.Abs(got-want) > 0.05*want {
			t.Errorf("%s steps at ITL %.2f ms, want within 5%% of its pool's chip run, %.2f ms", id, got, want)
		}
	}
}

// A pool whose chip has no catalog entry is refused with the chip's name, as is a bundle pool
// the scenario does not list or one of another chip than the scenario states for it.
func TestRunCmd_PoolChipRefusals(t *testing.T) {
	bundle := func(pools ...[4]string) string { return writePoolsBundle(t, pools...) }
	for _, tt := range []struct {
		name, want string
		edits      []string
		pools      [][4]string
	}{
		{"a chip with no catalog entry", "z999",
			[]string{fixtureCluster, "  nodes: 2\n  fabric: ib-400g\n  gpus_per_node: 8\n  pools:\n    - name: odd\n      hardware: z999\n",
				"    nodes: 1\n    parallel", "    nodes: 2\n    parallel"},
			[][4]string{{"odd", "z999", "80", "1"}}},
		{"a bundle pool the scenario does not list", "is not a pool of scenario",
			twoChipEdits("h200"), [][4]string{{"ghost", "h200", "139.80072021", "1"}}},
		{"a bundle chip other than the scenario's", "for that pool",
			twoChipEdits("h200"), [][4]string{{"older", "h200", "139.80072021", "1"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := writeColocatedVariant(t, tt.edits...)
			_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
				"--num-requests", "4", "--rate", "2", "--policy-config", bundle(tt.pools...))
			if err == nil || !strings.Contains(stderr, tt.want) {
				t.Errorf("want a refusal containing %q, got err=%v\n%s", tt.want, err, lastLines(stderr, 3))
			}
		})
	}
}
