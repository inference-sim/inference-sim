package cmd

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// The engine builds one instance for a whole data-parallel group, so the host places and counts
// instances by group, not by rank. The fixture's pools are TP4 x DP2 on four-GPU trays: a group
// is eight GPUs, two trays.

// The second pod of a two-pod pool is placed on the trays after the first pod's, in each pool.
// Counted by rank (four GPUs) it would sit on the next tray, inside the first pod's second tray.
func TestInstanceEndpoint_PodsOfAPoolLandOnDistinctNodes(t *testing.T) {
	dir, _ := dpPoolScenarios(t, 4, 4)
	pools, _ := openDPPools(t, dir, 2, 2)
	at := pools.instanceEndpoint()
	for _, c := range []struct {
		id   cluster.InstanceID
		node int
	}{
		{"instance_0", 0}, {"instance_1", 2}, // prefill pods: trays 0-1 and 2-3
		{"instance_2", 0}, {"instance_3", 2}, // decode pods, counted from the decode pool's first tray
	} {
		if got := at(c.id).Node; got != c.node {
			t.Errorf("%s is placed on node %d of its pool, want %d (a pod fills two four-GPU trays)", c.id, got, c.node)
		}
	}
}

// A pool of four trays holds two groups. Asking for three prefill instances is refused naming the
// groups; counted by rank the pool would hold four and accept it.
func TestCapacity_AGroupThatDoesNotFitIsRefused(t *testing.T) {
	dir, name := dpPoolScenarios(t, 4, 4)
	run := func(prefill, decode string) (string, error) {
		_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name,
			"--num-instances", "4", "--prefill-instances", prefill, "--decode-instances", decode,
			"--rate", "5", "--num-requests", "4", "--seed", "3")
		return stderr, err
	}
	stderr, err := run("3", "1")
	if err == nil {
		t.Fatal("three prefill groups were accepted in a pool that holds two")
	}
	if !strings.Contains(stderr, "holds 2 data-parallel group(s)") {
		t.Errorf("the refusal does not name the pool's group capacity:\n%s", lastLines(stderr, 3))
	}
	if stderr, err = run("2", "2"); err != nil {
		t.Errorf("the two groups the pools hold were refused: %v\n%s", err, lastLines(stderr, 4))
	}
}
