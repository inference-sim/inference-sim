package cmd

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// A colocated run with an fs tier below its CPU tier completes on 1, 2 and 4 instances: every
// prompt the CPU tier stores cascades into the fs tier, so copies of different sizes share the
// tier's link, which must keep one rate (the link arbiter refuses a link stated at two). Before,
// the second copy's demand restated the rate it was priced at, and the run panicked
// ("LinkArbiter.Start: link tier/nvme_gen4/to/node/0 stated at ..., already in use at ...").
func TestKVTierScenario_AnFSTierRuns(t *testing.T) {
	const name = "llama-3.1-70b-instruct-h200-tp4-4node.yaml"
	dir := writeOffloadScenario(t, pdScenarios, name, "", "offload:\n  connector: OffloadingConnector\n"+
		"  spec: TieringOffloadingSpec\n  tiers:\n    - {tier: cpu_dram, bytes: 549755813888}\n"+
		"    - {tier: nvme_gen4, bytes: 4398046511104}\n")
	for _, n := range []string{"1", "2", "4"} {
		t.Run(n+" instances", func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "result.json")
			_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name, "--num-instances", n,
				"--rate", "8", "--num-requests", "40", "--prefix-tokens", "512", "--seed", "1", "--output", out)
			if err != nil {
				t.Fatalf("the fs-tier run: %v\n%s", err, lastLines(stderr, 8))
			}
			var doc any
			if err := json.Unmarshal([]byte(readFile(t, out)), &doc); err != nil {
				t.Fatal(err)
			}
			links := map[string]bool{}
			collectKeys(doc, "link_peak_concurrent_transfers_", links)
			stored := false
			for k := range links {
				stored = stored || strings.Contains(k, "nvme_gen4")
			}
			if !stored {
				t.Errorf("no copy used the fs tier's link; links used: %v", links)
			}
		})
	}
}

// collectKeys lists the names anywhere in a Result document that start with prefix.
func collectKeys(v any, prefix string, out map[string]bool) {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			if strings.HasPrefix(k, prefix) {
				out[k] = true
			}
			collectKeys(e, prefix, out)
		}
	case []any:
		for _, e := range x {
			collectKeys(e, prefix, out)
		}
	}
}
