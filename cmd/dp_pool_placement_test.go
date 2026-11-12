package cmd

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// The engine builds one instance for a whole data-parallel group, so the host places and counts
// instances by group, not by rank. The fixture's prefill pool is TP2 x DP2 on four-GPU trays (a
// group is one tray) and its decode pool TP4 x DP2 (a group is eight GPUs, two trays).

// The second pod of a pool is placed after the first pod's trays, in each pool. Counted by rank it
// would sit inside the first pod: a decode pod is two trays wide, so its second pod is on tray 2
// and not tray 1.
func TestInstanceEndpoint_PodsOfAPoolLandOnDistinctNodes(t *testing.T) {
	dir, _ := dpPoolScenarios(t, 4, 4)
	pools, _ := openDPPools(t, dir, 2, 2)
	at := pools.instanceEndpoint()
	for _, c := range []struct {
		id   cluster.InstanceID
		node int
	}{
		{"instance_0", 0}, {"instance_1", 1}, // prefill pods: one tray each
		{"instance_2", 0}, {"instance_3", 2}, // decode pods: two trays each, from the decode pool's first tray
	} {
		if got := at(c.id).Node; got != c.node {
			t.Errorf("%s is placed on node %d of its pool, want %d", c.id, got, c.node)
		}
	}
}

// The router is told the same places the engines and the transfer price use: the node of each pod's
// first tray, and for a pod that spans two trays no single host. Counted by rank, the second decode
// pod would be given tray 1 while its engine sits on tray 2.
func TestScenarioEndpointTopology_IsThePlacementTheEnginesUse(t *testing.T) {
	dir, _ := dpPoolScenarios(t, 4, 4)
	pools, _ := openDPPools(t, dir, 2, 2)
	got, err := scenarioEndpointTopology(kernelOpened, pools, 4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	at := pools.instanceEndpoint()
	for _, id := range []cluster.InstanceID{"instance_0", "instance_1"} {
		m := pools.prefill
		want := fmt.Sprintf("blis-node-%d", poolOrigin(m).Node+at(id).Node)
		if got[id].Hostname != want || got[id].NodeCount != 1 {
			t.Errorf("%s: the router is given %+v, the engine is on %s", id, got[id], want)
		}
	}
	for _, id := range []cluster.InstanceID{"instance_2", "instance_3"} {
		if got[id].NodeCount != 2 || got[id].Hostname != "" {
			t.Errorf("%s spans two trays, and the router is given %+v", id, got[id])
		}
	}
}

// A pool of four trays holds four one-tray prefill groups and two two-tray decode groups. A fifth
// prefill or a third decode instance is refused naming the groups; counted by rank the pools would
// hold more and accept them.
func TestCapacity_AGroupThatDoesNotFitIsRefused(t *testing.T) {
	dir, name := dpPoolScenarios(t, 4, 4)
	total := func(p, d string) string { a, _ := strconv.Atoi(p); b, _ := strconv.Atoi(d); return strconv.Itoa(a + b) }
	run := func(prefill, decode string) (string, error) {
		_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name,
			"--num-instances", total(prefill, decode), "--prefill-instances", prefill, "--decode-instances", decode,
			"--rate", "5", "--num-requests", "4", "--seed", "3")
		return stderr, err
	}
	for _, c := range []struct{ prefill, decode, want string }{
		{"5", "1", "holds 4 data-parallel group(s)"},
		{"1", "3", "holds 2 data-parallel group(s)"},
	} {
		stderr, err := run(c.prefill, c.decode)
		if err == nil {
			t.Errorf("%s prefill and %s decode instances were accepted", c.prefill, c.decode)
		} else if !strings.Contains(stderr, c.want) {
			t.Errorf("%s+%s: the refusal does not name %q:\n%s", c.prefill, c.decode, c.want, lastLines(stderr, 3))
		}
	}
	if stderr, err := run("4", "2"); err != nil {
		t.Errorf("the groups the pools hold were refused: %v\n%s", err, lastLines(stderr, 4))
	}
}

// The Result shows where the pods sit: with transfer contention on it reports a peak for each link
// a pull used, named by node. Two prefill pods on distinct trays send over the egress links of
// nodes 0 and 1; counted by rank the second pod would share node 0's link and node 1 would not appear.
func TestResult_PullsLeaveFromEachPrefillPodsOwnNode(t *testing.T) {
	dir, name := dpPoolScenarios(t, 4, 4)
	stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name,
		"--num-instances", "4", "--prefill-instances", "2", "--decode-instances", "2",
		"--rate", "5", "--num-requests", "8", "--seed", "3", "--pd-transfer-contention")
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	pd, _ := resultDoc(t, stdout)["breakdowns"].(map[string]any)["pd"].(map[string]any)
	var egress []string
	for k := range pd {
		if n, ok := strings.CutPrefix(k, "link_peak_concurrent_transfers_"); ok && strings.Contains(n, "egress-node-") {
			egress = append(egress, n[strings.LastIndex(n, "-")+1:])
		}
	}
	slices.Sort(egress)
	if !slices.Equal(egress, []string{"0", "1"}) {
		t.Errorf("pulls left from the egress links of nodes %v, want the two prefill pods' nodes 0 and 1", egress)
	}
}
