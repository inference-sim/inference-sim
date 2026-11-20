package cmd

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
)

// P/D pools at different data-parallel widths (inference-sim/inference-sim#1952). The pools are separate engines with
// their own parallel layout, so the prefill pool may run wide in dp while the decode pool runs tensor-parallel, and
// the reverse. Each pool is counted in its own data-parallel groups, and every request is handed off.
const (
	pdTP8DP1 = "      tp: 8\n      pp: 1\n      dp: 1\n      enable_expert_parallel: false\n"
	pdTP4DP2 = "      tp: 4\n      pp: 1\n      dp: 2\n      enable_expert_parallel: true\n"
)

func TestRunCmd_KernelBackend_DisaggregatedPoolsAtDifferentDP(t *testing.T) {
	for _, c := range []struct {
		name    string
		edits   []string
		topo    []string
		wantDec int
	}{
		// 3 prefill instances of tp8 dp1 in front of one decode group of tp4 dp2 (8 GPUs).
		{"decode pool wider in dp", []string{"  - role: decode\n    nodes: 1\n    parallel:\n" + pdTP8DP1,
			"  - role: decode\n    nodes: 1\n    parallel:\n" + pdTP4DP2},
			[]string{"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1"}, 1},
		// The reverse: the prefill pool's three nodes hold three tp4 dp2 groups.
		{"prefill pool wider in dp", []string{"  - role: prefill\n    nodes: 3\n    parallel:\n" + pdTP8DP1,
			"  - role: prefill\n    nodes: 3\n    parallel:\n" + pdTP4DP2},
			[]string{"--num-instances", "4", "--prefill-instances", "3", "--decode-instances", "1"}, 1},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := writeScenarioVariant(t, c.edits...)
			prefix := filepath.Join(t.TempDir(), "trace")
			common := append([]string{"--scenarios", dir, "--scenario", "variant.yaml", "--seed", "3",
				"--horizon", "600000000"}, c.topo...)
			run, stderr, err := runKernelCLI(t, append([]string{"run", "--num-requests", "12", "--rate", "4",
				"--trace-output", prefix}, common...)...)
			if err != nil || !completedIs(run, 12) {
				t.Fatalf("the run did not complete its 12 requests: %v\n%s", err, lastLines(stderr, 3))
			}
			// Every request crossed the pools: one transfer each, and the handoff took time (it is priced from the
			// prefill pool's layout, so a pool of another width does not make it free).
			var r result.Result
			if err := result.Unmarshal([]byte(run), &r); err != nil || r.Breakdowns == nil {
				t.Fatalf("no result breakdowns: %v", err)
			}
			if n := r.Breakdowns.PD["kv_transfers"]; n != 12 {
				t.Errorf("%v KV transfers for 12 requests, want one each", n)
			}
			if r.Breakdowns.PD["kv_transfer_mean_ms"] <= 0 {
				t.Errorf("KV transfer mean %v ms: the handoff was priced as free", r.Breakdowns.PD["kv_transfer_mean_ms"])
			}
			// Run and replay agree on the pools and their instance counts (INV-13).
			rep, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml",
				"--trace-data", prefix + ".csv"}, common...)...)
			if err != nil {
				t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
			}
			if inv13View(t, rep) != inv13View(t, run) {
				t.Errorf("the replay differs from the run\n--- run\n%s\n--- replay\n%s", run, rep)
			}
		})
	}
}
