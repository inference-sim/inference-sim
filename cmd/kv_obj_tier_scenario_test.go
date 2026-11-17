package cmd

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// A colocated run with an object-store tier below its CPU tier completes on 1, 2 and 4 instances and
// reports the tier's traffic: every prompt the CPU tier stores cascades into the bucket, which the
// run shares among the engines of the model (one bucket for each model and bucket name). Before the
// run gave its engines a shared bucket, the scenario was refused ("BLIS simulates the fs tier only").
func TestKVTierScenario_AnObjectStoreTierRuns(t *testing.T) {
	const name = "llama-3.1-70b-instruct-h200-tp4-4node.yaml"
	dir := writeOffloadScenario(t, pdScenarios, name, "", "offload:\n  connector: OffloadingConnector\n"+
		"  spec: TieringOffloadingSpec\n  tiers:\n    - {tier: cpu_dram, bytes: 549755813888}\n"+
		"    - {tier: s3, bytes: 4398046511104}\n")
	for _, n := range []string{"1", "2", "4"} {
		t.Run(n+" instances", func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "result.json")
			_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", name, "--num-instances", n,
				"--rate", "8", "--num-requests", "40", "--prefix-tokens", "512", "--seed", "1", "--output", out)
			if err != nil {
				t.Fatalf("the object-store tier run: %v\n%s", err, lastLines(stderr, 8))
			}
			var doc any
			if err := json.Unmarshal([]byte(readFile(t, out)), &doc); err != nil {
				t.Fatal(err)
			}
			links := map[string]bool{}
			collectLinkKeys(doc, "link_peak_concurrent_transfers_", links)
			used := false
			for k := range links {
				used = used || strings.Contains(k, "s3")
			}
			if !used {
				t.Errorf("no copy used the object store's link; links used: %v", links)
			}
		})
	}
}

// collectLinkKeys lists the names anywhere in a Result document that start with prefix.
func collectLinkKeys(v any, prefix string, out map[string]bool) {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			if strings.HasPrefix(k, prefix) {
				out[k] = true
			}
			collectLinkKeys(e, prefix, out)
		}
	case []any:
		for _, e := range x {
			collectLinkKeys(e, prefix, out)
		}
	}
}
