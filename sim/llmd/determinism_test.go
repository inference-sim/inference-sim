package llmd

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// determinismChildEnv selects the child half of TestDeterminism_AcrossProcesses.
const determinismChildEnv = "LLMD_DETERMINISM_CHILD"

// routingDigest runs a flow-controlled P/D-free cluster and a P/D cluster and digests every
// request's placement, dispatch time and latency.
func routingDigest(t *testing.T) string {
	h := sha256.New()
	for _, cfg := range []string{"", softCeilingConfig, boundedQueueConfig} {
		reqs := testRequests(150, 9, 300, 32)
		dep := testDeployment(3)
		dep.BatchConfig = sim.NewBatchConfig(8, 2048, 0)
		cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
		flags := append([]string{}, DefaultEPPFlags...)
		if cfg != "" {
			flags = append(flags, "--allow-experimental-plugins", "--config-text="+cfg)
		}
		cs.SetGateway(New(Options{Model: "test-model", EPPFlags: flags}))
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		m := cs.AggregatedMetrics()
		for _, r := range reqs {
			fmt.Fprintf(h, "%s %s %d %v %v\n", r.ID, r.AssignedInstance, r.GatewayDispatchTime, m.RequestTTFTs[r.ID], m.RequestE2Es[r.ID])
		}
		fmt.Fprintf(h, "rej %d %d %d\n", cs.RejectedRequests(), cs.GatewayQueueRejected(), cs.GatewayExpired())
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func TestDeterminism_ChildProcess(t *testing.T) {
	if os.Getenv(determinismChildEnv) == "" {
		t.Skip("run by TestDeterminism_AcrossProcesses")
	}
	fmt.Println("DIGEST " + routingDigest(t))
}

// TestDeterminism_AcrossProcesses runs the same simulations in separate processes at
// GOMAXPROCS 1 and 8 (INV-6). Go randomizes map iteration per process and the Go scheduler
// varies with GOMAXPROCS, so a router decision that depended on either would change the digest.
func TestDeterminism_AcrossProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns child test processes")
	}
	var digests []string
	for _, procs := range []string{"1", "8", "8"} {
		cmd := exec.Command(os.Args[0], "-test.run=^TestDeterminism_ChildProcess$", "-test.count=1")
		cmd.Env = append(os.Environ(), determinismChildEnv+"=1", "GOMAXPROCS="+procs)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("child (GOMAXPROCS=%s): %v\n%s", procs, err, out)
		}
		var digest string
		for _, line := range strings.Split(string(out), "\n") {
			if strings.HasPrefix(line, "DIGEST ") {
				digest = strings.TrimPrefix(line, "DIGEST ")
			}
		}
		if digest == "" {
			t.Fatalf("child (GOMAXPROCS=%s) printed no digest:\n%s", procs, out)
		}
		digests = append(digests, digest)
	}
	for i := 1; i < len(digests); i++ {
		if digests[i] != digests[0] {
			t.Fatalf("routing differs across processes: %v", digests)
		}
	}
}

// TestDeterminism_SameProcessRepeat catches router state shared between simulations in one
// process (for example a package-level random source or counter).
func TestDeterminism_SameProcessRepeat(t *testing.T) {
	if a, b := routingDigest(t), routingDigest(t); a != b {
		t.Fatalf("two identical simulations in one process differ: %s vs %s", a, b)
	}
}
