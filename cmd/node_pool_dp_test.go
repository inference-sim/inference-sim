package cmd

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// An instance of a data-parallel deployment is the whole group its engine runs, so a node pool
// reserves, spans and bills it for tp x pp x dp GPUs, as it does a tensor-parallel instance of the
// same size (inference-sim#2260). The issue's repro: two instances of 8 GPUs each on a pool that
// starts with one 8-GPU node.

// dpLayoutScenarios is the data-parallel MoE fixture at the given layout on a two-node cluster, in
// a temporary scenarios directory, with the scenario's name.
func dpLayoutScenarios(t *testing.T, tp, dp int) (dir, name string) {
	t.Helper()
	body := readFile(t, filepath.Join(testScenariosDir(t), dpScenario))
	for _, e := range [][2]string{
		{"\n  nodes: 1\n", "\n  nodes: 2\n  fabric: ib-400g\n"},
		{"\n    nodes: 1\n", "\n    nodes: 2\n"},
		{"      tp: 2\n", "      tp: " + strconv.Itoa(tp) + "\n"},
		{"      dp: 2\n", "      dp: " + strconv.Itoa(dp) + "\n"},
	} {
		if strings.Count(body, e[0]) != 1 {
			t.Fatalf("the fixture no longer states %q exactly once", e[0])
		}
		body = strings.Replace(body, e[0], e[1], 1)
	}
	dir = t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, dpScenario), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir, dpScenario
}

// Two instances of 8 GPUs each do not both fit one 8-GPU node, however the 8 are split between tp
// and dp: the second waits for capacity, as a tp 8 instance's does. Two instances of 4 GPUs
// (tp 2 x dp 2) do both fit.
func TestRunCmd_NodePool_DPInstanceReservesItsWholeGroup(t *testing.T) {
	const deferred = "1 of 2 instance(s) not placed at startup"
	for _, c := range []struct {
		name         string
		tp, dp       int
		wantDeferred bool
	}{
		{"tp8 dp1 takes the node", 8, 1, true},
		{"tp4 dp2 takes the node as tp8 does", 4, 2, true},
		{"tp2 dp2 shares the node", 2, 2, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir, name := dpLayoutScenarios(t, c.tp, c.dp)
			bundle := writePoolsBundle(t, [4]string{"pool-b200", "b200", "178.35107422", "30"})
			out, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name, "--num-instances", "2",
				"--rate", "10", "--num-requests", "20", "--seed", "42", "--policy-config", bundle)
			if err != nil {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 5))
			}
			if got := strings.Contains(stderr, deferred); got != c.wantDeferred {
				t.Errorf("a second instance deferred = %t, want %t\n%s", got, c.wantDeferred, lastLines(stderr, 3))
			}
			clusterConservationHolds(t, out, 20)
		})
	}
}
