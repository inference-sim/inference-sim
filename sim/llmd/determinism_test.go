package llmd

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// determinismChildEnv selects the child half of TestDeterminism_AcrossProcesses.
const determinismChildEnv = "LLMD_DETERMINISM_CHILD"

// routingDigest runs the default router, two flow-control configurations, tied multi-tenant
// arrivals in two priority bands, and a P/D cluster, and digests every request's placement,
// dispatch time and latency.
func routingDigest(t *testing.T) string {
	h := sha256.New()
	for _, cfg := range []string{"", softCeilingConfig, boundedQueueConfig} {
		reqs := testRequests(150, 9, 300, 32)
		dep := testDeployment(3)
		editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(8, 2048, 0) })
		cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
		flags := DefaultEPPFlags()
		if cfg != "" {
			flags = append(flags, "--allow-experimental-plugins", "--config-text="+cfg)
		}
		cs.SetGateway(New(Options{Model: "test-model", EPPFlags: flags}))
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		m := cs.AggregatedMetrics()
		for _, r := range reqs {
			_, _ = fmt.Fprintf(h, "%s %s %d %v %v\n", r.ID, r.AssignedInstance, r.GatewayDispatchTime, m.RequestTTFTs[r.ID], m.RequestE2Es[r.ID])
		}
		_, _ = fmt.Fprintf(h, "rej %d %d %d\n", cs.RejectedRequests(), cs.GatewayQueueRejected(), cs.GatewayExpired())
	}
	digestTiedTenants(t, h)
	digestPD(t, h)
	digestPrecise(t, h)
	return fmt.Sprintf("%x", h.Sum(nil))
}

// digestTiedTenants: four tenants whose requests arrive at the same microsecond, in two priority
// bands (InferenceObjectives), under global-strict fairness, which breaks ties between flows by
// the registry's queue order: the case where a flow registry on the wall clock and sync.Map order
// diverged across processes.
func digestTiedTenants(t *testing.T, h io.Writer) {
	reqs := testRequests(600, 4, 0, 16) // 150 ms of arrivals, across several scrapes
	for i, r := range reqs {
		r.ArrivalTime = int64(i/8) * 2000 // eight requests per microsecond instant, every 2 ms
		r.TenantID = fmt.Sprintf("tenant-%d", i%4)
		r.SLOClass = map[bool]string{true: "critical", false: "sheddable"}[i%3 == 0]
	}
	dep := testDeployment(2)
	editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(4, 2048, 0) })
	cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
	cs.SetGateway(New(Options{
		Model:     "test-model",
		EPPFlags:  []string{"--pool-name=vllm", "--config-text=" + globalStrictConfig},
		Manifests: []byte(poolManifests),
		PodLabels: func(cluster.InstanceID, cluster.PoolRole) map[string]string { return map[string]string{"app": "vllm"} },
	}))
	if err := cs.Run(); err != nil {
		t.Fatalf("tied tenants: Run: %v", err)
	}
	held := map[string]int{}
	for _, r := range reqs {
		if r.GatewayDispatchTime > r.ArrivalTime {
			held[r.SLOClass]++
		}
		_, _ = fmt.Fprintf(h, "tt %s %s %d\n", r.ID, r.AssignedInstance, r.GatewayDispatchTime)
	}
	if held["critical"] == 0 || held["sheddable"] == 0 {
		t.Fatalf("tied tenants: held per band %v; both bands must queue for fairness to break ties in each", held)
	}
}

// digestPD digests a disaggregated run: prefill and decode placement and transfer timing.
func digestPD(t *testing.T, h io.Writer) {
	reqs := testRequests(60, 3, 1000, 0)
	dep := testDeployment(4)
	dep.PrefillInstances, dep.DecodeInstances = 2, 2
	dep.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 + tokens/50 }
	cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
	cs.SetGateway(New(Options{Model: "test-model", PodLabels: roleLabels,
		EPPFlags: append(DefaultEPPFlags(), "--config-text="+pdConfig)}))
	if err := cs.Run(); err != nil {
		t.Fatalf("P/D: Run: %v", err)
	}
	m := cs.AggregatedMetrics()
	for _, p := range cs.ParentRequests() {
		_, _ = fmt.Fprintf(h, "pd %s %s %s %v\n", p.ID, p.PrefillInstanceID, p.DecodeInstanceID, m.RequestTTFTs[p.OriginalRequest.ID])
	}
	if len(cs.ParentRequests()) == 0 {
		t.Fatal("P/D: nothing disaggregated")
	}
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
	for _, procs := range []string{"1", "8", "8", "1", "8"} {
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

// digestPrecise digests a run routed by the precise prefix cache, fed by the engines' KV events.
func digestPrecise(t *testing.T, h io.Writer) {
	dep := testDeployment(3)
	editEngine(&dep, func(c *cengine.SimConfig) { c.KVCacheConfig = cengine.NewKVCacheConfig(150, 16, 0, 0, 0, 0) })
	run := runPrecise(t, dep, prefixGroups(5, 8, 256, 10000), preciseConfig("    speculativeIndexing: true\n"), Options{KVEventDelayUs: 2000})
	m := run.cs.AggregatedMetrics()
	for _, r := range run.reqs {
		_, _ = fmt.Fprintf(h, "pp %s %s %d %v\n", r.ID, r.AssignedInstance, r.GatewayDispatchTime, m.RequestTTFTs[r.ID])
	}
}
