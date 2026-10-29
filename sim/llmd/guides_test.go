package llmd

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// TestLLMDGuides_RunOrAreRefusedWithAReason: the llm-d guides' EndpointPickerConfigs
// (testdata/llm-d-guides, extracted verbatim from llm-d/llm-d by #1946, with their source and
// sha256 in each file) are what real deployments start the router with. Each runs through the
// embedded router, on a colocated or a 2P2D deployment as its profile handler requires, with
// every request accounted for (INV-1) -- or the router refuses it at startup, naming why. The
// outcome of every guide is pinned, so a router bump that changes one is seen.
func TestLLMDGuides_RunOrAreRefusedWithAReason(t *testing.T) {
	want := map[string]string{ // guide -> "" (runs) or a substring of its startup refusal
		"flow-control.yaml": "",
		// prefix-cache-affinity-filter reads the external latency predictor's sidecar, which a
		// simulation has no stand-in for (sim/llmd/support.go).
		"optimized-baseline.yaml": `"prefix-cache-affinity-filter" (type prefix-cache-affinity-filter): it consumes the external latency predictor`,
		"pd-disaggregation.yaml":  `"prefix-cache-affinity-filter" (type prefix-cache-affinity-filter): it consumes the external latency predictor`,
		// Past the block-size check (run on 64-token engines below), it is refused for its
		// prefix-cache-affinity-filter too.
		"precise-prefix-cache-routing.yaml": `"prefix-cache-affinity-filter" (type prefix-cache-affinity-filter): it consumes the external latency predictor`,
		"tiered-prefix-cache-cpu.yaml":      "",
		// Its revision split draws from process-wide randomness, which would break INV-6.
		"wide-ep.yaml": `"disaggregatedset-rollout-screener" (type disaggregatedset-rollout-screener): its revision split draws from process-wide randomness`,
	}
	files, err := filepath.Glob(filepath.Join("testdata", "llm-d-guides", "*.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(files)
	if len(files) != len(want) {
		t.Fatalf("%d guide fixtures, %d pinned outcomes", len(files), len(want))
	}
	for _, path := range files {
		name := filepath.Base(path)
		t.Run(name, func(t *testing.T) {
			body, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			config := string(body)
			pd := strings.Contains(config, "disagg-profile-handler")
			dep := testDeployment(2)
			if strings.Contains(config, "blockSizeTokens: 64") {
				// The guide's engines use 64-token KV blocks; the router refuses a producer that
				// cuts block keys at another size than the engines'.
				editEngine(&dep, func(c *cengine.SimConfig) { c.KVCacheConfig = cengine.NewKVCacheConfig(10000, 64, 0, 0, 0, 0) })
			}
			labels := func(cluster.InstanceID, cluster.PoolRole) map[string]string { return nil }
			if pd {
				dep = testDeployment(4)
				dep.PrefillInstances, dep.DecodeInstances = 2, 2
				dep.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 + tokens/50 }
				labels = roleLabels
			}
			reqs := testRequests(40, 9, 2000, 64)
			for _, r := range reqs {
				r.Model = "test-model"
			}
			cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
			gw := New(Options{Model: "test-model", PodLabels: labels,
				EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)})
			cs.SetGateway(gw)
			runErr := cs.Run()
			refusal, ok := want[name]
			if !ok {
				t.Fatalf("no pinned outcome for %s", name)
			}
			if refusal != "" {
				if runErr == nil || !strings.Contains(runErr.Error(), refusal) {
					t.Fatalf("want a startup refusal naming %q, got %v", refusal, runErr)
				}
				return
			}
			if runErr != nil {
				t.Fatalf("the guide does not run: %v", runErr)
			}
			res := runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}
			assertConservation(t, res, len(reqs))
			if res.metrics.CompletedRequests == 0 {
				t.Fatal("no request completed")
			}
			if pd && len(cs.ParentRequests()) == 0 {
				t.Error("a disaggregating guide disaggregated no request")
			}
		})
	}
}
